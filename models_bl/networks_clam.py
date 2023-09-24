'''
@author: Yang Hu
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np



"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 256, D = 128, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        
#         A = torch.squeeze(A, dim=-1)
        return A, x

"""
args:
    embedding_dim: dim of input matrices
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
    instance_loss_fn: loss function to supervise instance-level training
    subtyping: whether it's a subtyping problem
"""
class CLAM_SB(nn.Module):
    def __init__(self, embedding_dim, dropout = False, k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False):
        super(CLAM_SB, self).__init__()
        
        self.name = 'CLAM'
        
#         self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        self.size = [embedding_dim, 256, 128]
        fc = [nn.Linear(self.size[0], self.size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
            
        attention_net = Attn_Net_Gated(L = self.size[1], D = self.size[2], dropout = dropout, n_classes = 1)

        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(self.size[1], n_classes)
        instance_classifiers = [nn.Linear(self.size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
        
        self.p_targets = torch.full((self.k_sample, ), 1).cuda()
        self.n_targets = torch.full((self.k_sample, ), 0).cuda()

#         initialize_weights(self)

#     def relocate(self):
#         device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.attention_net = self.attention_net.to(device)
#         self.classifiers = self.classifiers.to(device)
#         self.instance_classifiers = self.instance_classifiers.to(device)
    
#     @staticmethod
#     def create_positive_targets(length):
#         return torch.full((length, 2), 1)
#     @staticmethod
#     def create_negative_targets(length):
#         return torch.full((length, 2), 0)
    
    #instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier): 
#         device=h.device
        h = h.view(-1, self.size[1])
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids.squeeze())
        top_n_ids = torch.topk(-A, self.k_sample)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids.squeeze())

        all_targets = torch.cat([self.p_targets, self.n_targets], dim=0)
        
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
#         logits = logits.squeeze()
#         print(logits.shape, all_targets.shape)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss
    
    #instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
#         device=h.device
        h = h.view(-1, self.size[1])
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids.squeeze())
        p_targets = self.p_targets
        
        logits = classifier(top_p)
#         logits = logits.squeeze()
#         print(logits.shape, p_targets.shape)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss

    def forward(self, h, bag_lens=None, label=None, instance_eval=False, return_features=False, attention_only=False):
        A, h = self.attention_net(h)  # NxK        
        A = torch.transpose(A, -2, -1)  # KxN
#         A_raw = A
        A = F.softmax(A, dim=-1)  # softmax over N
        
        if bag_lens is not None:
            mask = (torch.arange(A.shape[-1], device=A.device).expand(A.shape) < bag_lens.unsqueeze(1).unsqueeze(1)).byte()
            A = A * mask
        if attention_only:
            return torch.squeeze(A, dim=1)

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes) #binarize label
            if type(label) == int: label = [label] 
            for i in range(len(self.instance_classifiers) if len(label) > 1 else 1):
                inst_label = inst_labels[i][1].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss = self.inst_eval(A, h, classifier)
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss = self.inst_eval_out(A, h, classifier)
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)
        
        M = A.matmul(h)
        logits = self.classifiers(M)
        logits = logits.squeeze(1)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, torch.squeeze(A, dim=1), results_dict


if __name__ == '__main__':
    L = torch.full((8, ), 1)
    L = torch.cat([L.unsqueeze(dim=0)] * 2, dim=0)
    print(L)