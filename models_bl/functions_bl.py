'''
@author: Yang Hu
'''
from support.tools import Time


def train_clam_epoch(clam_net, train_loader, loss, optimizer, bag_weight=0.7, epoch_info: tuple=(-2, -2)):
    """
    Args:
        clam_net: apply clam code from https://github.com/mahmoodlab/CLAM, just receive (X, label) as input
        data_loader:
        loss:
        optimizer:
        bag_weight: to adjust 2 sub-losses
        epoch: the idx of running epoch (default: None (unknown))
    """
    
    clam_net.train()
    epoch_t_loss_sum, epoch_acc_sum, batch_count, time = 0.0, 0.0, 0, Time()
    epoch_inst_loss_sum = 0.0
    
    
    for batch_i, (mat_X, bag_dim, y) in enumerate(train_loader):
        mat_X = mat_X.cuda()
        bag_dim = bag_dim.cuda()
        y = y.cuda()
        # main loss
        y_pred, _, _, att, instance_dict = clam_net(mat_X, bag_dim, label=y, instance_eval=True)
        batch_loss = loss(y_pred, y)
        # instance loss in clam
        instance_loss = instance_dict['instance_loss']
        total_loss = bag_weight * batch_loss + (1 - bag_weight) * instance_loss
        # BP
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        # loss count
        epoch_t_loss_sum += batch_loss.cpu().item()
        epoch_acc_sum += (y_pred.argmax(dim=1) == y).sum().cpu().item()
        epoch_inst_loss_sum += instance_loss.cpu().item()
        batch_count += 1
        
    train_log = 'epoch [%d/%d], batch_loss-> %.4f, clustering_loss-> %.4f, train acc-> %.4f, time: %s sec' % (epoch_info[0] + 1, epoch_info[1],
                                                                                      epoch_t_loss_sum / batch_count,
                                                                                      epoch_inst_loss_sum / batch_count,
                                                                                      epoch_acc_sum / len(train_loader.dataset),
                                                                                      str(time.elapsed())[:-5])
    return train_log


if __name__ == '__main__':
    pass