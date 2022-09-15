'''
@author: Yang Hu
'''
'''
this module contains the general used function for DL
'''
''' ------ loss function ------ '''

import cv2
from torch import optim, nn
import torch
from torch.nn.modules.loss import BCEWithLogitsLoss, BCELoss, MSELoss
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

import numpy as np
from support.env import ENV
from support.tools import Time


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

def dice_loss():
    return SoftDiceLoss().cuda()

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

''' ------ data loader ------ '''
def get_data_loader(dataset, batch_size, num_workers=4, sf=False, p_mem=True):
    data_loader = DataLoader(dataset, batch_size=batch_size,
                             num_workers=num_workers, shuffle=sf, pin_memory=p_mem)
    return data_loader


''' ----------- for training/testing ------------ '''
def train_epoch(train_loader, net, loss, optimizer, epoch_info=(-2, -2)):
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
    
def test_epoch(test_loader, net, loss, prediction=False):
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