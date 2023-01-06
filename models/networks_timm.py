'''
@author: Yang Hu

some network models loaded from TIMM:
https://github.com/rwightman/pytorch-image-models

'''


from torch import nn
from torchvision.models.resnet import ResNet18_Weights

from torchvision import models


'''
models from repositories: torchvision or timm, usually with the pre-trained weights on ImageNet
1. CNN (encoder)
'''

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

if __name__ == '__main__':
    pass