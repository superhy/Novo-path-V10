'''
Created on 11 Jul 2023

@author: yang hu
'''

import os
import warnings

from sklearn import metrics
import torch
from torch.nn.functional import softmax

from models import functions
from models.datasets import load_richtileslist_fromfile, TryK_MIL_Dataset
from models.networks import reload_net, store_net, BasicResNet18
import numpy as np
from support.files import parse_caseid_from_slideid
from support.metadata import query_task_label_dict_fromcsv
from support.tools import Time


def filter_topKtiles_4eachslide(prediction_scores, tileidx_slideid_dict, label_dict, K_0=1, K_1=1):
    """
    Args:
        prediction_scores: numpy array of trying prediction results, with [EMT, Differentiated] classification
        tileidx_slideid_dict: {tile_idx (Int): slide_idx (String)}
        K_0: hyper-parameter of extract top K_0 tiles with highest EMT score.
    """
    
    ''' tileidx <-> prediction_scores: (0, 1, 2, ...) <-> (float, float, float, ...) '''
    tileidx_array = np.array(list(range(len(tileidx_slideid_dict))))
    order = np.lexsort((tileidx_array, prediction_scores))
    # from max to min
    tileidx_sort_array = np.flipud(tileidx_array[order])
    
    filter_slide_tileidx_dict = {}
    for tileidx in tileidx_sort_array:
        # query get slide_id then check&creat slide_id key in filter dict
        slide_id = tileidx_slideid_dict[tileidx]
        case_id = parse_caseid_from_slideid(slide_id)
        if slide_id not in filter_slide_tileidx_dict.keys():
            filter_slide_tileidx_dict[slide_id] = []
            
        K = K_0 if label_dict[case_id] == 0 else K_1
        if len(filter_slide_tileidx_dict[slide_id]) >= K:
            continue
        else:
            filter_slide_tileidx_dict[slide_id].append(tileidx)
    
    return filter_slide_tileidx_dict


class TryK_MIL():
    '''
    MIL with Try top k,
    ref from paper: https://www.nature.com/articles/s41591-019-0508-1 (Thomas J. Fuchs's paper)
    '''
    def __init__(self, ENV_task, net, net_filename=None, test_mode=False):
        
        if net_filename is None and test_mode is True:
            warnings.warn('no trained model for testing, please check!')
            return
        
        self.ENV_task = ENV_task
        ''' prepare some parames '''
        _env_task_name = self.ENV_task.TASK_NAME
        _env_loss_package = self.ENV_task.LOSS_PACKAGE
        
        self.model_store_dir = self.ENV_task.MODEL_FOLDER
        pt_prefix = ''
        if net_filename is not None and net_filename.find('PT-') != -1:
            pt_prefix = 'pt_'
        elif net_filename is not None and net_filename.find('ROI-CLS') != -1:
            pt_prefix = 'roi_'
        elif net_filename is not None and net_filename.find('ROI-STK') != -1:
            pt_prefix = 'stk_'
            
        self.alg_name = '{}TK_MIL{}_{}'.format(pt_prefix, self.ENV_task.FOLD_SUFFIX, _env_task_name)
        
        print('![Initial Stage] test mode: {}'.format(test_mode))
        print('Initializing the training/testing datasets...')
        
        self.num_epoch = self.ENV_task.NUM_TK_EPOCH
        self.batch_size_ontiles = self.ENV_task.MINI_BATCH_TILE
        # self.batch_size_onslides = self.ENV_task.MINI_BATCH_SLIDEMAT
        self.tile_loader_num_workers = self.ENV_task.TILE_DATALOADER_WORKER
        self.slidemat_loader_num_workers = self.ENV_task.SLIDEMAT_DATALOADER_WORKER
        self.try_K_0 = ENV_task.TRY_K_0
        self.try_K_1 = ENV_task.TRY_K_1
        
        self.net = net
        if net_filename is not None:
            net_filepath = os.path.join(self.model_store_dir, net_filename)
            print('load network model from: {}'.format(net_filepath))
            self.net, _ = reload_net(self.net, net_filepath)
        self.net = self.net.cuda()
        
        # make tiles data
        self.train_tiles_list, self.train_tileidx_slideid_dict, _ = load_richtileslist_fromfile(ENV_task, 
                                                                                                for_train=True)
        if self.ENV_task.TEST_PART_PROP <= 0.0:
            self.test_tiles_list, self.test_tileidx_slideid_dict = [], {}
        else:
            self.test_tiles_list, self.test_tileidx_slideid_dict, _ = load_richtileslist_fromfile(ENV_task, 
                                                                                                  for_train=False)

        # make label
        self.label_dict = query_task_label_dict_fromcsv(self.ENV_task)
        
        if _env_loss_package[0] == 'wce':
            self.criterion = functions.weighted_cel_loss(_env_loss_package[1])
        else:
            self.criterion = functions.cel_loss()
        self.optimizer = functions.optimizer_adam_basic(self.net, lr=ENV_task.LR_TILE)
        
        self.transform = functions.get_transform()
        self.train_set = TryK_MIL_Dataset(tiles_list=self.train_tiles_list,
                                          label_dict=self.label_dict,
                                          transform=self.transform)
        self.test_set = TryK_MIL_Dataset(tiles_list=self.test_tiles_list,
                                         label_dict=self.label_dict,
                                         transform=self.transform, try_mode=True)
    
    def optimize(self, not_eval=False):
        train_loader = functions.get_data_loader(self.train_set, self.batch_size_ontiles, 
                                                 num_workers=self.tile_loader_num_workers, sf=False)
        test_loader = functions.get_data_loader(self.test_set, self.batch_size_ontiles, 
                                                num_workers=self.tile_loader_num_workers, sf=False)
        
        checkpoint_auc = 0.0
        for epoch in range(self.num_epoch):
            print('In training...', end='')
            
            self.train_set.switch_mode(True)
    #         try_scores = np.load('test_emt_score.npz')['arr_0'] # this is only for debug
            try_scores, _ = self.try_predict(train_loader, epoch_info=(epoch, self.num_epoch))
            topK_slide_tileidx_dict = filter_topKtiles_4eachslide(try_scores, 
                                                                  self.train_tileidx_slideid_dict, 
                                                                  label_dict=self.label_dict,
                                                                  K_0=self.try_K_0, K_1=self.try_K_1)
            
            ''' ! Switch to Training Mode ''' 
            self.train_set.refresh_filter_traindata(topK_slide_tileidx_dict)
            self.train_set.manual_shuffle_traindata()
            self.train_set.switch_mode(False)
            
            ''' begin to train the EMT MIL network '''
            train_log = functions.train_enc_epoch(self.net, train_loader, self.criterion,
                                                  self.optimizer, epoch_info=(epoch, self.num_epoch))
            print(train_log)
            
            # evaluation
            if ((epoch + 1) % 5 == 0 or epoch >= self.num_epoch - 1):
                if not_eval is False:
                    print('>>> Just record.')
                    self.record(epoch, None)
                else:
                    print('>>> In testing...', end='')
                    test_time = Time()
                    test_cls_scores, test_loss = self.try_predict(test_loader, epoch_info=(epoch, self.num_epoch))
                    max_slide_tileidx_dict = filter_topKtiles_4eachslide(self.ENV_task, test_cls_scores, self.test_tileidx_slideid_dict,
                                                                         label_dict=self.label_dict, 
                                                                         K_0=1, K_1=1)  # as default, K=1
                    test_acc, _, _, test_auc = self.tk_evaluation(max_slide_tileidx_dict,
                                                                  test_cls_scores, self.label_dict)
                    checkpoint_auc = test_auc
                    if epoch >= self.num_epoch - 1:
                        self.record(epoch, checkpoint_auc)
                    
                    print('test_loss-> %.4f, test acc-> %.4f, test auc-> %.4f, time: %s sec' % (test_loss, test_acc, test_auc, 
                                                                                                str(test_time.elapsed())))
                
    def record(self, epoch, checkpoint_auc):
        '''
        store the trained models
        '''
        alg_store_name = self.alg_name + '_[{}]'.format(epoch + 1)
        init_obj_dict = {'epoch': epoch + 1,
                         'auc': checkpoint_auc}
        store_filepath = store_net(self.model_store_dir, self.net, 
                                   alg_store_name, self.optimizer, init_obj_dict)
        print('store the milestone point<{}>, '.format(store_filepath), end='')    
    
    def try_predict(self, data_loader, epoch_info: tuple=(-2, -2)):
        """
        conduct feed-forward process to get the Top-K scored samples
            with tile image input
        """
        self.net.eval()
        # initial the score for all instance
        scores = torch.FloatTensor(len(data_loader.dataset)).zero_()
        
        # force to set the shuffle in try prediction stage with False
    #     data_loader.shuffle = False # WARNINGl: this is not work!
        with torch.no_grad():
            print('Perform a trying prediction in Epoch: [{}/{}], '.format(epoch_info[0] + 1, epoch_info[1]), end='')
            
            test_loss_sum, batch_count, time = 0.0, 0, Time()
            for i, try_x in enumerate(data_loader):
                try_X = try_x[0].cuda()
                y = try_x[1].cuda()
                y_pred = self.net(try_X)
                # record loss
                test_loss_sum += self.criterion(y_pred, y).cpu().item()
                # the positive dim of output
                output_pos = softmax(y_pred, dim=1)
                scores[i * self.batch_size_ontiles: i * self.batch_size_ontiles + try_X.size(0)] = output_pos.detach()[:, 1].clone()
                
                batch_count += 1
    #             print('get batch result: {}, at minibatch: {}'.format(scores[i * batch_size: i * batch_size + try_x.size(0)], i))
            print('with time: {} sec'.format(str(time.elapsed())[:-5]), end='; ')
            
        return scores.cpu().numpy(), test_loss_sum / batch_count
    
    def tk_evaluation(self, max_slide_tileidx_dict, cls_scores, label_dict):
        
        y_pred = [1 if cls_scores[max_slide_tileidx_dict[slide_id][0]] > 0.5 else 0 for slide_id in max_slide_tileidx_dict.keys()]
        score_pred = [cls_scores[max_slide_tileidx_dict[slide_id][0]] for slide_id in max_slide_tileidx_dict.keys()]
        y = [label_dict[parse_caseid_from_slideid(slide_id)] for slide_id in max_slide_tileidx_dict.keys()]
        
        y_pred, y = np.asarray(y_pred), np.asarray(y)
        acc = metrics.accuracy_score(y, y_pred)
        fpr, tpr, threshold = metrics.roc_curve(y, score_pred)
        auc = metrics.auc(fpr, tpr)
            
        return acc, fpr, tpr, auc
    
    
''' ---------------- running functions can be directly called ---------------- '''   
    
def _run_train_tkmil_resnet18(ENV_task, prep_model_name=None):
    net = BasicResNet18(output_dim=2)
        
    method = TryK_MIL(ENV_task, net, net_filename=prep_model_name)
    method.optimize(not_eval=True)
    
if __name__ == '__main__':
    pass



