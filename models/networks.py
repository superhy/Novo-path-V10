'''
@author: Yang Hu

network models, self created or loaded from [torchvision, vit-pytorch, timm, hugging_face, etc.]
the list:
    https://github.com/pytorch/vision
    https://github.com/lucidrains/vit-pytorch
'''
'''
------------------ some basic functions for networks -------------------
'''

import os
import warnings

from torch import nn
import torch
from torch.autograd.function import Function
from torchvision import models
from torchvision.models.resnet import ResNet18_Weights
from vit_pytorch.dino import Dino, get_module_device
from vit_pytorch.es_vit import EsViTTrainer
from vit_pytorch.extractor import Extractor
from vit_pytorch.mae import MAE
from vit_pytorch.recorder import Recorder
from vit_pytorch.vit import ViT

from support.tools import Time
import torch.nn.functional as F


def store_net(store_dir, trained_net, 
              algorithm_name, optimizer, init_obj_dict={}, with_time=True):
    """
    store the trained models
    
    Args:
        ENV_task: a object which packaged all parames in specific ENV_TASK
        trained_net:
        algorithm_name:
        optimizer:
        init_obj_dict:
    """
    
    if not os.path.exists(store_dir):
        os.makedirs(store_dir)
    
    str_time = Time().date if with_time else ''
    store_filename = 'checkpoint_' + trained_net.name + '-' + algorithm_name + str_time + '.pth'
    init_obj_dict.update({'state_dict': trained_net.state_dict(),
                          'optimizer': optimizer.state_dict()})
    
    store_filepath = os.path.join(store_dir, store_filename)
    torch.save(init_obj_dict, store_filepath)
    
    return store_filepath

def reload_net(model_net, model_filepath):
    """
    reload network models only for testing
    
    Args:
        model_net: an empty network need to reload
        model_filepath:
        
    Return: only the 'state_dict' of models
    """
    checkpoint = torch.load(model_filepath)
    model_net.load_state_dict(checkpoint['state_dict'], False)
    print('load model from: {}'.format(model_filepath))
    
    return model_net, checkpoint

def check_reuse_net(target_net, load_net, model_filepath):
    """
    reuse the parameters in load_net to target_net
    automatically check if the shape of each layer is consistent with target_net,
    skip the layers which not with the same shape of parameters
    
    Return: only the 'state_dict' of target_net
    """
    checkpoint = torch.load(model_filepath)
    load_net.load_state_dict(checkpoint['state_dict'], False)
    
    for name1, param1 in load_net.state_dict().items():
        if name1 not in target_net.state_dict():
            continue
        param2 = target_net.state_dict()[name1]
        if param1.shape == param2.shape:
            target_net.state_dict()[name1].copy_(param1)
            print('>> copy layer:', name1)
            
    print('reuse model from: {}'.format(model_filepath))
    
    return target_net, checkpoint

''' ------------------ gradient reversed networks (encoder) ------------------ '''
   
class ReverseGrad_Layer(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None
    
class BasicResNet18(nn.Module):
    
    def __init__(self, output_dim, imagenet_pretrained=True):
        super(BasicResNet18, self).__init__()
        """
        Args: 
            output_dim: number of classes
            imagenet_pretrained: use the weight with pre-trained on ImageNet
        """
        
        self.name = 'ResNet18'
        
        self.backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if imagenet_pretrained else None)
        self.fc_id = nn.Identity()
        self.backbone.fc = self.fc_id
        
        self.fc = nn.Linear(in_features=512, out_features=output_dim, bias=True)
    
    def forward(self, X):
        x = self.backbone(X)
        output = self.fc(x)  
        return output
    
    
class ReverseResNet18(nn.Module):
    
    def __init__(self, output_dim, imagenet_pretrained=True):
        super(ReverseResNet18, self).__init__()
        """
        Args: 
            output_dim: number of classes
            imagenet_pretrained: use the weight with pre-trained on ImageNet
        """
        
        self.name = 'ReResNet18'
        
        self.backbone = models.resnet18(pretrained=imagenet_pretrained)
        self.fc_id = nn.Identity()
        self.backbone.fc = self.fc_id
        
        self.fc = nn.Linear(in_features=512, out_features=output_dim, bias=True)
        
    def forward_ahd(self, X_pos):
        x_pos = self.backbone(X_pos)

        output_pos = self.fc(x_pos)
        return output_pos
    
    def forward_rev(self, X_neg, alpha):
        x_neg = self.backbone(X_neg)
        x_reversed = ReverseGrad_Layer.apply(x_neg, alpha)
        
        output_neg = self.fc(x_reversed)
        return output_neg
    
    def forward(self, X, ahead=True, alpha=1e-4):
        '''
        ahead (ahd) means with normal gradient BP
        otherwise with reversed gradient BP
        '''
        if ahead == True:
            output = self.forward_ahd(X)
        else:
            output = self.forward_rev(X, alpha)
          
        return output
    
''' ------------------- Transformer (encoder) --------------------- '''
class ViT_base(nn.Module):
    
    def __init__(self, image_size, patch_size, output_dim,
                 depth, heads):
        '''
        apply the version of ViT for small datasets
        https://arxiv.org/abs/2112.13492
        '''
        super(ViT_base, self).__init__()
        
        self.name = 'ViT_base'
        
        self.image_size = image_size
        self.dim = int(self.image_size) # 256 for size: 256, 512 for size: 512
        self.mlp_dim = int(self.dim / 2) # 128 for dim: 256, 256 for dim: 512
        
        self.backbone = ViT(
            image_size = self.image_size,
            patch_size = patch_size,
            num_classes = output_dim,
            dim = self.dim,
            depth = depth,
            heads = heads,
            mlp_dim = self.mlp_dim,
            dropout = 0.1,
            emb_dropout = 0.1
            )
        self.with_wrapper = False
        ''' 
        cut the output head of the original network and extract the backbone for accessing images' encoding
        re-put another fc layer for classification output
        '''
        self.backbone.mlp_head = nn.Identity()
        self.fc = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(in_features=self.dim, out_features=output_dim, bias=True)
        )
        
    def get_dino_learner(self):
        ''' 
        setup the Dino self-supervision learner for ViT
        https://openaccess.thecvf.com/content/ICCV2021/html/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.html
        '''
        learner = Dino(
            self.backbone,
            image_size = self.image_size,
            hidden_layer = 'to_latent',        # hidden layer name or index, from which to extract the embedding
            projection_hidden_size = 256,      # projector network hidden dimension
            projection_layers = 4,             # number of layers in projection network
            num_classes_K = 50,                # output logits dimensions (referenced as K in paper)
            student_temp = 0.9,                # student temperature
            teacher_temp = 0.04,               # teacher temperature, needs to be annealed from 0.04 to 0.07 over 30 epochs
            local_upper_crop_scale = 0.4,      # upper bound for local crop - 0.4 was recommended in the paper 
            global_lower_crop_scale = 0.5,     # lower bound for global crop - 0.5 was recommended in the paper
            moving_average_decay = 0.9,        # moving average of encoder - paper showed anywhere from 0.9 to 0.999 was ok
            center_moving_average_decay = 0.9, # moving average of teacher centers - paper showed anywhere from 0.9 to 0.999 was ok
            )
        
        # in Dino's code, has already copy the learner to same device with backbone
        return learner
    
    def get_mae_learner(self):
        ''' 
        setup the MAE self-supervision learner for ViT
        https://arxiv.org/abs/2111.06377
        '''
        learner = MAE(
            encoder = self.backbone,
            masking_ratio = 0.75,   # the paper recommended 75% masked patches
            decoder_dim = 512,      # paper showed good results with just 512
            decoder_depth = 6       # anywhere from 1 to 8
            )
        
        device = get_module_device(self.backbone)
        learner.to(device)
        return learner
        
    def deploy_recorder(self):
        self.backbone = Recorder(self.backbone)
        self.with_wrapper = True
    
    def deploy_extractor (self):
        self.backbone = Extractor(self.backbone)
        self.with_wrapper = True
    
    def discard_wrapper(self):
        self.backbone.eject()
        self.with_wrapper = False
        
    def forward(self, X):
        e = self.backbone(X)
        # in case with pytorch wrapper and produce 2 outcomes
        x = e[0] if self.with_wrapper else e
        output = self.fc(x)  
        return output
    
class ViT_D4_H6(ViT_base):
    
    def __init__(self, image_size, patch_size, output_dim):
        ''' 
        D4(4 in name): depth
        H6(6 in name): heads
        Dino: Dino self-supervised learner
        '''
        depth, heads = 4, 6
        super(ViT_D4_H6, self).__init__(image_size, patch_size, output_dim,
                                        depth, heads)
        self.name = 'ViT-4-6'
    
class ViT_D6_H8(ViT_base):
    
    def __init__(self, image_size, patch_size, output_dim):
        ''' 
        D6(6 in name): depth
        H8(8 in name): heads
        Dino: Dino self-supervised learner
        '''
        depth, heads = 6, 8
        super(ViT_D6_H8, self).__init__(image_size, patch_size, output_dim,
                                        depth, heads)
        self.name = 'ViT-6-8'
    
    
class ViT_D9_H12(ViT_base):
    
    def __init__(self, image_size, patch_size, output_dim):
        ''' 
        D9(9 in name): depth
        H12(12 in name): heads
        Dino: Dino self-supervised learner
        '''
        depth, heads = 9, 12
        super(ViT_D9_H12, self).__init__(image_size, patch_size, output_dim, 
                                         depth, heads)
        self.name = 'ViT-9-12'
        
''' ------------------- Reversed Transformer (encoder) --------------------- '''
class ReverseViT_D6_H8(ViT_base):
    
    def __init__(self, image_size, patch_size, output_dim):
        ''' 
        D6(6 in name): depth
        H8(8 in name): heads
        Dino: Dino self-supervised learner
        '''
        depth, heads = 6, 8
        super(ReverseViT_D6_H8, self).__init__(image_size, patch_size, output_dim,
                                               depth, heads)
        self.name = 'ReViT-6-8'
    
    def forward_ahd(self, X_pos):
        e_pos = self.backbone(X_pos)
        # in case with pytorch wrapper and produce 2 outcomes
        x_pos = e_pos[0] if self.with_wrapper else e_pos
        output_pos = self.fc(x_pos)  
        return output_pos
    
    def forward_rev(self, X_neg, alpha):
        # in case with pytorch wrapper
        if self.with_wrapper:
            self.discard_wrapper()
        x = self.backbone(X_neg)
        x_reversed = ReverseGrad_Layer.apply(x, alpha)
        output = self.fc(x_reversed)
        return output
    
    def forward(self, X, ahead=True, alpha=0.1):
        '''
        ahead (ahd) means with normal gradient BP
        otherwise with reversed gradient BP
        '''
        if ahead == True:
            output = self.forward_ahd(X)
        else:
            output = self.forward_rev(X, alpha)
          
        return output
        
class ReverseViT_D9_H12(ViT_base):
    
    def __init__(self, image_size, patch_size, output_dim):
        ''' 
        D6(6 in name): depth
        H8(8 in name): heads
        Dino: Dino self-supervised learner
        '''
        depth, heads = 9, 12
        super(ReverseViT_D9_H12, self).__init__(image_size, patch_size, output_dim,
                                                depth, heads)
        self.name = 'ReViT-9-12'
    
    def forward_ahd(self, X_pos):
        e_pos = self.backbone(X_pos)
        # in case with pytorch wrapper and produce 2 outcomes
        x_pos = e_pos[0] if self.with_wrapper else e_pos
        output_pos = self.fc(x_pos)  
        return output_pos
    
    def forward_rev(self, X_neg, alpha):
        # in case with pytorch wrapper
        if self.with_wrapper:
            self.discard_wrapper()
        x = self.backbone(X_neg)
        x_reversed = ReverseGrad_Layer.apply(x, alpha)
        output = self.fc(x_reversed)
        return output
    
    def forward(self, X, ahead=True, alpha=0.1):
        '''
        ahead (ahd) means with normal gradient BP
        otherwise with reversed gradient BP
        '''
        if ahead == True:
            output = self.forward_ahd(X)
        else:
            output = self.forward_rev(X, alpha)
          
        return output


''' --- some networks for patch-region context modeling and encoding combination --- '''
class ViT_Region_4_6(ViT_base):
    
    def __init__(self, image_size, patch_size, channels, pseudo_dim=2):
        '''
        ViT for Region context modeling
        
        Args:
            channels: when pre-training, use 3 means rgb image,
                when encoding, use 256 (or other) means embedding of patch
        '''
        depth, heads=4, 6
        super(ViT_Region_4_6, self).__init__(image_size, patch_size, pseudo_dim,
                                             depth=depth, heads=heads)
        self.image_size = image_size
        self.dim = 256
        self.mlp_dim = 128
        self.backbone = ViT(
            image_size = self.image_size,
            patch_size = patch_size,
            num_classes = pseudo_dim,
            dim = self.dim,
            depth = depth,
            heads = heads,
            mlp_dim = self.mlp_dim,
            channels=channels,
            dropout = 0.1,
            emb_dropout = 0.1
            )
        ''' 
        cut the output head of the original network and extract the backbone for accessing images' encoding
        re-put another fc layer for classification output
        '''
        self.backbone.mlp_head = nn.Identity()
        self.fc = nn.Identity() # the real encode dim is self.dim = 256
        
        self.name = 'ViT-Reg-4-6-img{}'.format(str(self.image_size))
        
class CombLayers(nn.Module):
    
    def __init__(self, in_dim1, in_dim2, out_dim):
        '''
        fully-connected layers for combining 2 part of features
        
        now the application purpose is: 
            feature-1 (256-dim encoded with ViT_? for single tile)
            feature-2 (256-dim encoded with ViT_Region_? for regional context for a specific tile)
            combine [feature-1, feature-2] (512-dim) -> transform to -> (256-dim)
            
        In the above case:
            in_dim (1, 2) = 256, 256 (in_dim * 2 = 512)
            out_dim = 256
        '''
        super(CombLayers, self).__init__()
        
        self.fc_comb = nn.Sequential(
            nn.LayerNorm(in_dim1 + in_dim2),
            nn.Linear(in_features=in_dim1 + in_dim2, out_features=out_dim, bias=True)
        )
        nn.init.normal_(self.fc_comb[-1].weight) # mean=0, std=1
        nn.init.zeros_(self.fc_comb[-1].bias)        
        
    def reuse_weights(self, inh_weights):
        '''
        inh_weights: should be [model.fc1.weight, model.fc1.bias]
        
        ! now, there is no suitable reuse network layers.
        '''
        shapes_match = (
            self.fc1.weight.shape == inh_weights[0].shape and
            self.fc1.bias.shape == inh_weights[1].shape
        )
        if not shapes_match:
            warnings.warn("the shape of weights is not matching, please check!")
            return
        
        self.fc_comb[-1].weight = nn.Parameter(inh_weights[0].clone())
        self.fc_comb[-1].bias = nn.Parameter(inh_weights[1].clone())
        
    def forward(self, e1, e2):
        comb_e = torch.cat((e1, e2), dim=-1)
        output = self.fc_comb(comb_e)  
        return output
    
''' --- some test encoders with Transformer --- '''
class ViT_D3_H4_T(ViT_base):
    
    def __init__(self, image_size, patch_size, output_dim):
        ''' tiny (T) ViT encoder for test on PC '''
        super(ViT_D3_H4_T, self).__init__(image_size, patch_size, output_dim, depth=3, heads=4)       
        self.image_size = image_size
        self.dim = 64 # 256 for size: 256, 512 for size: 512
        self.mlp_dim = int(self.dim / 2) # 128 for dim: 256, 256 for dim: 512
        self.backbone = ViT(
            image_size = self.image_size,
            patch_size = patch_size,
            num_classes = output_dim,
            dim = self.dim,
            depth = 3,
            heads = 4,
            mlp_dim = self.mlp_dim,
            dropout = 0.1,
            emb_dropout = 0.1
            )
        ''' 
        cut the output head of the original network and extract the backbone for accessing images' encoding
        re-put another fc layer for classification output
        '''
        self.backbone.mlp_head = nn.Identity()
        self.fc = nn.Linear(in_features=self.dim, out_features=output_dim, bias=True)
        
        self.name = 'ViT-Tiny'
        
    def get_dino_learner(self):
        ''' 
        setup the Dino self-supervision learner for ViT
        https://openaccess.thecvf.com/content/ICCV2021/html/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.html
        '''
        learner = Dino(
            self.backbone,
            image_size = self.image_size,
            hidden_layer = 'to_latent',        # hidden layer name or index, from which to extract the embedding
            projection_hidden_size = 256,      # projector network hidden dimension
            projection_layers = 2,             # number of layers in projection network
            num_classes_K = 20,                # output logits dimensions (referenced as K in paper)
            student_temp = 0.9,                # student temperature
            teacher_temp = 0.04,               # teacher temperature, needs to be annealed from 0.04 to 0.07 over 30 epochs
            local_upper_crop_scale = 0.4,      # upper bound for local crop - 0.4 was recommended in the paper 
            global_lower_crop_scale = 0.5,     # lower bound for global crop - 0.5 was recommended in the paper
            moving_average_decay = 0.9,        # moving average of encoder - paper showed anywhere from 0.9 to 0.999 was ok
            center_moving_average_decay = 0.9, # moving average of teacher centers - paper showed anywhere from 0.9 to 0.999 was ok
            )
        
        # in Dino's code, has already copy the learner to same device with backbone
        return learner
    
    def get_mae_learner(self):
        ''' 
        setup the MAE self-supervision learner for ViT
        https://arxiv.org/abs/2111.06377
        '''
        learner = MAE(
            encoder = self.backbone,
            masking_ratio = 0.75,   # the paper recommended 75% masked patches
            decoder_dim = 128,      # paper showed good results with just 512
            decoder_depth = 6       # anywhere from 1 to 8
            )
        
        device = get_module_device(self.backbone)
        learner.to(device)
        return learner
    

''' 
---------------- attention based feature aggregation networks -----------------
'''
   
class AttentionPool(nn.Module):
    
    def __init__(self, embedding_dim, output_dim):
        super(AttentionPool, self).__init__()
        
        self.name = 'AttPool'
        
        self.embedding_dim = embedding_dim
        self.att_layer_width = [256, 128]
        self.output_layer_width = 128
        self.att_dim = 1
        
        self.encoder = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=self.embedding_dim, out_features=self.att_layer_width[0]),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=self.att_layer_width[0], out_features=self.att_layer_width[1]),
            nn.Tanh()
            )
        
        self.bn = nn.BatchNorm1d(self.att_layer_width[1])
        self.attention = nn.Linear(in_features=self.att_layer_width[1],
                                   out_features=self.att_dim, bias=True)
        
#         self.softmax = nn.Softmax(dim=-1)
        self.fc_out = nn.Linear(in_features=self.output_layer_width,
                                out_features=output_dim, bias=True)
        
    def forward(self, X_e, bag_lens):
        """
        Args:
            X_e: Embedding of the tiles in slide X as the input
            bag_lens: 
        """
        X_e = self.encoder(X_e)
        X_e = self.bn(X_e.transpose(-2, -1)).transpose(-2, -1)
        att = self.attention(X_e)
        att = att.transpose(-2, -1)
        ''' record the attention value (before softmax) '''
#         att_r = torch.squeeze(att, dim=1)
        att = F.softmax(att, dim=-1)
#         att = torch.sigmoid(att)
        mask = (torch.arange(att.shape[-1], device=att.device).expand(att.shape) < bag_lens.unsqueeze(1).unsqueeze(1)).byte()
        att = att * mask
        
        att_H = att.matmul(X_e)
        output = self.fc_out(att_H).squeeze(1)
        
        ''' record the attention value (after softmax) '''
        att_r = torch.squeeze(att, dim=1)
#         output_label = F.softmax(output, dim=1).argmax(dim=1)

        return output, att_r, att_H

    
class GatedAttentionPool(nn.Module):
    
    def __init__(self, embedding_dim, output_dim):
        super(GatedAttentionPool, self).__init__()
        
        self.name = 'GatedAttPool'
        
        self.embedding_dim = embedding_dim
        self.att_layer_width = [256, 128]
        self.output_layer_width = 128
        self.att_dim = 1
        
        self.encoder = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=self.embedding_dim, out_features=self.att_layer_width[0]),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=self.att_layer_width[0], out_features=self.att_layer_width[1]),
            )
        
        self.attention_U = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=self.att_layer_width[1], out_features=self.att_layer_width[1]),
            nn.Tanh()
            )
        self.attention_V = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=self.att_layer_width[1], out_features=self.att_layer_width[1]),
            nn.Sigmoid()
            )
        
        self.bn = nn.BatchNorm1d(self.att_layer_width[1])
        self.attention = nn.Linear(in_features=self.att_layer_width[1],
                                   out_features=self.att_dim, bias=True)
        
        self.fc_out = nn.Linear(in_features=self.output_layer_width,
                                out_features=output_dim, bias=True)
        
    def forward(self, X_e, bag_lens):
        """
        Args:
            X_e: Embedding of the tiles in slide X as the input
            bag_lens: 
        """
        X_e = self.encoder(X_e)
        X_e = self.bn(X_e.transpose(-2, -1)).transpose(-2, -1)
        att_U = self.attention_U(X_e)
        att_V = self.attention_V(X_e)
        att = self.attention(att_V * att_U)
        att = att.transpose(-2, -1)
        ''' record the attention value (before softmax) '''
#         att_r = torch.squeeze(att, dim=1)
        att = F.softmax(att, dim=-1)
        
        mask = (torch.arange(att.shape[-1], device=att.device).expand(att.shape) < bag_lens.unsqueeze(1).unsqueeze(1)).byte()
        att = att * mask
        
        att_H = att.matmul(X_e)
        output = self.fc_out(att_H).squeeze(1)
        
        ''' record the attention value (after softmax) '''
        att_r = torch.squeeze(att, dim=1)
#         output_label = F.softmax(output, dim=1).argmax(dim=1)

        return output, att_r, att_H
    

if __name__ == '__main__':
    pass