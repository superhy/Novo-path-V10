'''
@author: Yang Hu
'''

import warnings

import torch
from torch.nn.functional import softmax

from models import functions
from models.datasets import SlideMatrix_Dataset
from models.functions_attpool import check_load_slide_matrix_files
from models.networks import store_net, BasicResNet18
from models_bl import functions_bl
from models_bl.networks_clam import CLAM_SB
import numpy as np
from support.env import devices
from support.metadata import query_task_label_dict_fromcsv
from support.tools import Time


def _run_train_CLAM_MIL_classification(ENV_task, num_epoch=80, num_least_epoch=10,
                                       tile_loader_num_workers=4, slidemat_loader_num_workers=2,
                                       last_eval_epochs=5, test_epoch=None, record_points=[79]):
    """
    -- comparison method --
    
    CLAM MIL in paper: 
        Data Efficient and Weakly Supervised Computational Pathology on Whole Slide Images. 
        Nature Biomedical Engineering
    (can be called directly)
    """
    
    _env_task_name = ENV_task.TASK_NAME
    _env_model_store_dir = ENV_task.MODEL_FOLDER
    _env_loss_package = ENV_task.LOSS_PACKAGE
    batch_size_ontiles = ENV_task.MINI_BATCH_TILE
    batch_size_onslides = ENV_task.MINI_BATCH_SLIDEMAT
    overall_stop_loss = ENV_task.OVERALL_STOP_LOSS
    
    ''' a nick name '''
    alg_name = 'CLAM-MIL_{}_{}'.format(ENV_task.FOLD_SUFFIX, _env_task_name)
    
    ''' check the training (testing) slides matrices on disk '''
    print('Initializing the training/testing slide matrices...', end=', ')
    init_time = Time()
    
    resnet = BasicResNet18(output_dim=2)
    resnet.cuda()
    
    train_slidemat_file_sets = check_load_slide_matrix_files(ENV_task, batch_size_ontiles, tile_loader_num_workers,
                                                             encoder_net=resnet.backbone,
                                                             force_refresh=True, for_train=True, 
                                                             print_info=False)
    test_slidemat_file_sets = check_load_slide_matrix_files(ENV_task, batch_size_ontiles, tile_loader_num_workers, 
                                                            encoder_net=resnet.backbone, force_refresh=True,
                                                            for_train=False if not ENV_task.DEBUG_MODE else True,
                                                            print_info=False)
    print('train slides: %d, test slides: %d, time: %s' % (len(train_slidemat_file_sets), len(test_slidemat_file_sets), str(init_time.elapsed())))
    embedding_dim = np.load(train_slidemat_file_sets[0][2]).shape[-1]
    
    clam_net = CLAM_SB(embedding_dim=embedding_dim, dropout=True, subtyping=True)
    clam_net = clam_net.cuda()
    print('On-standby: {} algorithm'.format(alg_name), end=', ')
    print('Network: {}, train on -> {}'.format(clam_net.name, devices))
    
    label_dict = query_task_label_dict_fromcsv(ENV_task)
        
    if _env_loss_package[0] == 'ce':
        criterion = functions.weighted_cel_loss(_env_loss_package[1][0])
    else:
        criterion = functions.cel_loss()
    optimizer = functions.optimizer_adam_basic(clam_net, lr=1e-4)
        
#     cudnn.benchmark = True # not use at the changing input X here
    train_slidemat_set = SlideMatrix_Dataset(train_slidemat_file_sets, label_dict)
    train_slidemat_loader = functions.get_data_loader(train_slidemat_set, batch_size_onslides,
                                                      num_workers=slidemat_loader_num_workers, sf=True)
    test_slidemat_set = SlideMatrix_Dataset(test_slidemat_file_sets, label_dict)
    test_slidemat_loader = functions.get_data_loader(test_slidemat_set, batch_size_onslides,
                                                     num_workers=slidemat_loader_num_workers, sf=False)
    
    checkpoint_auc = 0.0
    epoch = 0
    
    overall_epoch_stop = False
    queue_auc = []
    while epoch < num_epoch and overall_epoch_stop == False:
        print('In training...', end='')
        train_log = functions_bl.train_clam_epoch(clam_net, train_slidemat_loader, criterion,
                                                  optimizer, epoch_info=(epoch, num_epoch))
        print(train_log)
        
        attpool_current_loss = float(train_log[train_log.find('loss->') + 6: train_log.find(', clustering_loss')])
        if epoch >= num_least_epoch - 1 and attpool_current_loss < overall_stop_loss:
            overall_epoch_stop = True
            
        # evaluation
        if not test_epoch == None and epoch + 1 >= test_epoch:
            print('>>> In testing...', end='')
            test_log, test_loss, y_pred_scores, y_labels = test_clam_epoch(clam_net, test_slidemat_loader, criterion)
            test_acc, _, _, test_auc = functions.regular_evaluation(y_pred_scores, y_labels)
            
            queue_auc.append(test_auc)
            if len(queue_auc) > last_eval_epochs:
                queue_auc.remove(queue_auc[0])
            if epoch in record_points or overall_epoch_stop == True:
                checkpoint_auc = test_auc
                alg_store_name = alg_name + '_[{}]'.format(epoch + 1)
                init_obj_dict = {'epoch': epoch + 1,
                                 'auc': checkpoint_auc}
                store_filepath = store_net(False, _env_model_store_dir, clam_net, alg_store_name, optimizer, init_obj_dict)
                print('store the milestone point<{}>, '.format(store_filepath), end='')
                
            print('>>> on clam -> test acc: %.4f, test auc: %.4f' % (test_acc, test_auc))
        epoch += 1
        
    ''' calculate the final performance '''
    print(queue_auc)
    final_avg_auc = np.average(queue_auc)
    print('### final evaluation on attpool -> average test auc: %.4f' % (final_avg_auc))
    

def test_clam_epoch(clam_net, test_loader, loss):
    """
    test & evaluation function for CLAM MIL algorithm 
    """
    clam_net.eval()
    epoch_loss_sum, epoch_acc_sum, batch_count, time = 0.0, 0.0, 0, Time()
    
    y_pred_scores, y_labels = [], []
    with torch.no_grad():
        for mat_X, bag_dim, y in test_loader:
            mat_X = mat_X.cuda()
            bag_dim = bag_dim.cuda()
            y = y.cuda()
            # feed forward
            y_pred, _, _, _, _ = clam_net(mat_X, bag_dim)
            batch_loss = loss(y_pred, y)
            # loss count
            epoch_loss_sum += batch_loss.cpu().item()
            y_pred = softmax(y_pred, dim=-1)
            epoch_acc_sum += (y_pred.argmax(dim=1) == y).sum().cpu().item()
            batch_count += 1
            
            y_pred_scores.extend(y_pred.detach().cpu().numpy()[:, -1].tolist())
            y_labels.extend(y.cpu().numpy().tolist())
    
    test_log = 'test_loss-> %.4f, time: %s sec' % (epoch_loss_sum / batch_count,
                                                   str(time.elapsed())[:-5])
    
    return test_log, epoch_loss_sum / batch_count, np.array(y_pred_scores), np.array(y_labels)


if __name__ == '__main__':
    pass








