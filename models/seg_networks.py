'''
@author: Yang Hu
'''

import os

from torch import nn
import torch

from support.tools import Time
import torch.nn.functional as F


def store_net(store_dir, trained_net, 
              algorithm_name, optimizer, init_obj_dict={}):
    """
    store the trained models
    
    Args:
        trained_net:
        algorithm_name:
        optimizer:
        init_obj_dict:
    """
    
    if not os.path.exists(store_dir):
        os.makedirs(store_dir)
    
    store_filename = 'checkpoint_' + trained_net.name + '-' + algorithm_name + Time().date + '.pth'
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
    return model_net, checkpoint


''' some module for unet '''
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)
    
class DownSampling(nn.Module):
    """Downscaling with maxpool then double conv"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels=in_channels, out_channels=out_channels)
        )
        
    def forward(self, x):
        return self.maxpool_conv(x)
    
class UpSampling(nn.Module):
    """Upscaling then double conv"""
    
    def __init__(self, in_channels, out_channels, bi_linear=True):
        super().__init__()
        
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bi_linear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=2, stride=2)
 
        self.conv = DoubleConv(in_channels=in_channels, out_channels=out_channels)
        
    def forward(self, x1, x2):
        
        x1 = self.up(x1)
        
        # calculate the difference of source tensor and target tensor
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
        
        # padding on source tensor
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            nn.Sigmoid()
#             nn.Identity()
            )
        
        
    def forward(self, x):
        return self.conv(x)
    
    
''' unet '''
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, x_width=1, bi_linear=False):
        super(UNet, self).__init__()
        
        self.name = 'UNet'
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bi_linear = bi_linear
        
        self.inc = DoubleConv(in_channels=self.n_channels, out_channels=4 * x_width)
        self.down_1 = DownSampling(in_channels=4 * x_width, out_channels=8 * x_width)
        self.down_2 = DownSampling(in_channels=8 * x_width, out_channels=16 * x_width)
        self.down_3 = DownSampling(in_channels=16 * x_width, out_channels=32 * x_width)
        self.down_4 = DownSampling(in_channels=32 * x_width, out_channels=64 * x_width)
        
        self.up_1 = UpSampling(in_channels=64 * x_width, out_channels=32 * x_width, bi_linear=self.bi_linear)
        self.up_2 = UpSampling(in_channels=32 * x_width, out_channels=16 * x_width, bi_linear=self.bi_linear)
        self.up_3 = UpSampling(in_channels=16 * x_width, out_channels=8 * x_width, bi_linear=self.bi_linear)
        self.up_4 = UpSampling(in_channels=8 * x_width, out_channels=4 * x_width, bi_linear=self.bi_linear)
        self.outc = OutConv(in_channels=4 * x_width, out_channels=self.n_classes)
        
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down_1(x1)
        x3 = self.down_2(x2)
        x4 = self.down_3(x3)
        x5 = self.down_4(x4)
        
        x = self.up_1(x5, x4)
        x = self.up_2(x, x3)
        x = self.up_3(x, x2)
        x = self.up_4(x, x1)
        logits = self.outc(x)
        return logits
    


if __name__ == '__main__':
    pass