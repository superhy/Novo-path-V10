'''
@author: Yang Hu
'''

import csv

import cv2
from sklearn import metrics
from torch import optim, nn
import torch
from torch.nn.modules.loss import BCEWithLogitsLoss, BCELoss, MSELoss, L1Loss, \
    NLLLoss, CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

import numpy as np
from support.env import ENV
from support.tools import Time


'''
this module contains the general used function for DL
'''
''' ------ loss function ------ '''
def bce_logits_loss():
    return BCEWithLogitsLoss(reduction='mean').cuda()

def bce_loss():
    return BCELoss(reduction='mean').cuda()

def mse_loss():
    return MSELoss().cuda()

class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()
 
    def forward(self, logits, targets):
        num = targets.size(0)
        smooth = 1
#         probs = torch.sigmoid(logits)
        probs = logits
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)
 
        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.bce_loss = BCEWithLogitsLoss(reduction='mean')

    def forward(self, inputs, targets, smooth=1):
        #comment out if your model contains a sigmoid or equivalent activation layer
#         inputs = torch.sigmoid(inputs)       
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = self.bce_loss(inputs, targets)
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

def dice_bce_loss():
    return DiceBCELoss().cuda()

''' <up> segmentation loss, <down> classification loss '''

def dice_loss():
    return SoftDiceLoss().cuda()

def l1_loss():
    return L1Loss().cuda()

def nll_loss():
    return NLLLoss().cuda()

def cel_loss():
    return CrossEntropyLoss().cuda()

def weighted_cel_loss(weight=0.5):
    w = torch.Tensor([1 - weight, weight])
    loss = CrossEntropyLoss(w).cuda()
    return loss


''' ------ optimizer ------ '''
def optimizer_rmsprop_basic(net, lr=1e-4):
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    return optimizer

def optimizer_adam_basic(net, lr=1e-4, wd=1e-4):
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    return optimizer


''' ------ transform ------ '''
def get_transform(resize=512):
    '''
    data transform with only image normalization
    '''
    normalize = transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.1, 0.1, 0.1]
    )
    transform_augs = transforms.Compose([
        transforms.Resize(size=(resize, resize)),
        transforms.ToTensor(),
        normalize
        ])
    return transform_augs

def get_data_arg_transform():
    '''
    data transform with slight data augumentation
    '''
    normalize = transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.1, 0.1, 0.1]
    )
#     _resize = (256, 256) if ENV.TRANSFORMS_RESIZE < 300 else (ENV.TRANSFORMS_RESIZE - 20, TRANSFORMS_RESIZE - 20)
    transform_augs = transforms.Compose([
        transforms.RandomResizedCrop(size=ENV.TRANSFORMS_RESIZE, scale=(0.8, 1.0)),
#         transforms.CenterCrop(size=_resize),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
#         transforms.RandomRotation(degrees=15),
        transforms.Resize(size=(ENV.TRANSFORMS_RESIZE, ENV.TRANSFORMS_RESIZE)),
        transforms.ToTensor(),
        normalize
        ])
    return transform_augs

''' ------ data loader ------ '''
def get_data_loader(dataset, seg_batch_size, SEG_NUM_WORKERs=4, sf=False, p_mem=True):
    data_loader = DataLoader(dataset, seg_batch_size=seg_batch_size,
                             SEG_NUM_WORKERs=SEG_NUM_WORKERs, shuffle=sf, pin_memory=p_mem)
    return data_loader


'''
----------------- functions of ViT encoder pre-training -----------------
'''
def dino_epoch(learner, train_loader, optimizer, epoch_info: tuple=(-2, -2)):
    """
    self-supervised pre-training epoch with Dino
    
    Args:
        learner:
        train_loader:
        optimizer:
        epoch: the idx of running epoch (default: None (unknown))
    """
    learner.train()
    epoch_loss_sum, batch_count, time = 0.0, 0, Time()
    
    for X in train_loader:
        X = X.cuda()
        # feed forward
        batch_loss = learner(X)
        # BP
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        learner.update_moving_average()
        # loss count
        epoch_loss_sum += batch_loss.cpu().item()
        batch_count += 1
    
    epoch_log = 'epoch [%d/%d], batch_loss-> %.4f, time: %s sec' % (epoch_info[0] + 1, epoch_info[1],
                                                                    epoch_loss_sum / batch_count,
                                                                    str(time.elapsed())[:-5])
    return epoch_log

def mae_epoch(learner, train_loader, optimizer, epoch_info: tuple=(-2, -2)):
    """
    self-supervised pre-training epoch with MAE
    
    Args:
        learner:
        train_loader:
        optimizer:
        epoch: the idx of running epoch (default: None (unknown))
    """
    learner.train()
    epoch_loss_sum, batch_count, time = 0.0, 0, Time()
    
    for X in train_loader:
        X = X.cuda()
        # feed forward
        batch_loss = learner(X)
        # BP
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        # loss count
        epoch_loss_sum += batch_loss.cpu().item()
        batch_count += 1
    
    epoch_log = 'epoch [%d/%d], batch_loss-> %.4f, time: %s sec' % (epoch_info[0] + 1, epoch_info[1],
                                                                    epoch_loss_sum / batch_count,
                                                                    str(time.elapsed())[:-5])
    return epoch_log


''' --------------- functions of classification training --------------- '''  
def train_enc_epoch(net, train_loader, loss, optimizer, epoch_info: tuple=(-2, -2)):
    """
    trainer for tile-level encoder 
    
    Args:
        net:
        train_loader:
        loss:
        optimizer:
        epoch: the idx of running epoch (default: None (unknown))
    """
    net.train()
    epoch_loss_sum, epoch_acc_sum, batch_count, time = 0.0, 0.0, 0, Time()
    
    for X, y in train_loader:
        X = X.cuda()
        y = y.cuda()
        # feed forward
        y_pred = net(X)
        batch_loss = loss(y_pred, y)
        # BP
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        # loss count
        epoch_loss_sum += batch_loss.cpu().item()
        epoch_acc_sum += (y_pred.argmax(dim=1) == y).sum().cpu().item()
        batch_count += 1
    
#     train_log = 'batch_loss-> %.6f, train acc-> %.4f, time: %s sec' % (epoch_loss_sum / batch_count, epoch_acc_sum / len(train_loader.dataset), str(time.elapsed()))
    train_log = 'epoch [%d/%d], batch_loss-> %.4f, train acc (on tiles)-> %.4f, time: %s sec' % (epoch_info[0] + 1, epoch_info[1],
                                                                                                 epoch_loss_sum / batch_count,
                                                                                                 epoch_acc_sum / len(train_loader.dataset),
                                                                                                 str(time.elapsed())[:-5])
    return train_log

def train_agt_epoch(net, train_loader, loss, optimizer, epoch_info: tuple=(-2, -2)):
    """
    trainer for slide-level(WSI) feature aggregation and/or classification
    
    Args:
        net: diff with other training function, net need to input <mat_X, bag_dim>
        data_loader:
        loss:
        optimizer:
        epoch: the idx of running epoch (default: None (unknown))
    """
    net.train()
    epoch_loss_sum, epoch_acc_sum, batch_count, time = 0.0, 0.0, 0, Time()
    
    for mat_X, bag_dim, y in train_loader:
        mat_X = mat_X.cuda()
        bag_dim = bag_dim.cuda()
        y = y.cuda()
        # feed forward
        y_pred, _, _ = net(mat_X, bag_dim)
        batch_loss = loss(y_pred, y)
        # BP
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        # loss count
        epoch_loss_sum += batch_loss.cpu().item()
        epoch_acc_sum += (y_pred.argmax(dim=1) == y).sum().cpu().item()
        batch_count += 1
    
#     train_log = 'batch_loss-> %.6f, train acc-> %.4f, time: %s sec' % (epoch_loss_sum / batch_count, epoch_acc_sum / len(train_loader.dataset), str(time.elapsed()))
    train_log = 'epoch [%d/%d], batch_loss-> %.4f, train acc-> %.4f, time: %s sec' % (epoch_info[0] + 1, epoch_info[1],
                                                                                      epoch_loss_sum / batch_count,
                                                                                      epoch_acc_sum / len(train_loader.dataset),
                                                                                      str(time.elapsed())[:-5])
    return train_log


''' -------------- function of classification evaluation -------------- '''
    
def regular_evaluation(y_scores, y_label):
    '''
    '''
    y_pred = np.array([1 if score > 0.5 else 0 for score in y_scores.tolist()])
    acc = metrics.balanced_accuracy_score(y_label, y_pred)
    fpr, tpr, threshold = metrics.roc_curve(y_label, y_scores)
    auc = metrics.auc(fpr, tpr)
        
    return acc, fpr, tpr, auc

def store_evaluation_roc(csv_path, roc_set):
    '''
    store the evaluation results as ROC as csv file
    '''
    acc, fpr, tpr, auc = roc_set
    with open(csv_path, 'w', newline='') as record_file:
        csv_writer = csv.writer(record_file)
        csv_writer.writerow(['acc', 'auc', 'fpr', 'tpr'])
        for i in range(len(fpr)):
            csv_writer.writerow([acc, auc, fpr[i], tpr[i]])
    print('write roc record: {}'.format(csv_path))
            
def load_evaluation_roc(csv_path):
    '''
    load the evaluation ROC from csv file
    '''
    with open(csv_path, 'r', newline='') as roc_file:
        print('load record from: {}'.format(csv_path))
        csv_reader = csv.reader(roc_file)
        acc, auc, fpr, tpr = 0.0, 0.0, [], []
        line = 0
        for record_line in csv_reader:
            if line == 0:
                line += 1
                continue
            if line == 1:
                acc = float(record_line[0])
                auc = float(record_line[1])
            fpr.append(float(record_line[2]))
            tpr.append(float(record_line[3]))
            line += 1
    return acc, auc, fpr, tpr

''' ----------- for segmentation training/testing ------------ '''
def train_seg_epoch(train_loader, net, loss, optimizer, epoch_info=(-2, -2)):
    """
    """
    # set the model in training mode
    net.train()
    train_loss_sum, batch_count, time = 0.0, 0, Time()
    
    for X, y in train_loader:
        X = X.cuda()
        y = y.cuda()
        y_pred = net(X)
        
        batch_loss = loss(y_pred, y)
        
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        
        train_loss_sum += batch_loss.cpu().item()
#         train_acc_sum += (y_pred.argmax(dim=1) == y).sum().cpu().item()
        batch_count += 1
            
#         test_acc = float('inf')
            
    print('epoch %d/%d, LR %f, batch_loss %.4f, time %s'
          % (epoch_info[0] + 1, epoch_info[1], 
             optimizer.state_dict()['param_groups'][0]['lr'], 
             train_loss_sum / batch_count, str(time.elapsed())[:-5] ))
    
def test_seg_epoch(test_loader, net, loss, prediction=False):
    """
    """
    net.eval()
    test_loss_sum, batch_count, time = 0.0, 0, Time()
    
    with torch.no_grad():
        for X, y, pred_paths in test_loader:
            X = X.cuda()
            y = y.cuda()
            y_pred = net(X)
            
            batch_loss = loss(y_pred, y)
            test_loss_sum += batch_loss.cpu().item()
            batch_count += 1
            
            if prediction:
                for i, pred_path in enumerate(pred_paths):
                    res_pred = np.array(y_pred.data.cpu()[i])[0]
                    res_pred[res_pred >= 0.5] = 255
                    res_pred[res_pred < 0.5] = 0
                    cv2.imwrite(pred_path, res_pred)
    
    print('test_batch_loss %.4f, time %s, with output prediction [%s]'
          % (test_loss_sum / batch_count, str(time.elapsed())[:-5], str(prediction) ))
    
    return test_loss_sum / batch_count


def show_heatmap_on_image(img: np.ndarray,
                          heatmap: np.ndarray):
    '''
    '''
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        img = np.float32(img) / 255

    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)
    
    

if __name__ == '__main__':
    pass