import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import convnext_large, ConvNeXt_Large_Weights
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import timm
from PIL import Image
import glob
import csv
import random
import numpy as np
import os
import pandas as pd


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ModelConfig:
    classes = 3
    att_dim = 1536
    in_size = 1536
    # classes = 3
    dropout = 0.5
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)


# class LKA(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
#         self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
#         self.conv1 = nn.Conv2d(dim, dim, 1)


#     def forward(self, x):
#         u = x.clone()        
#         attn = self.conv0(x)
#         attn = self.conv_spatial(attn)
#         attn = self.conv1(attn)

#         return u * attn


# class Attention(nn.Module):
#     def __init__(self, d_model):
#         super().__init__()

#         self.proj_1 = nn.Conv2d(d_model, d_model, 1)
#         self.activation = nn.GELU()
#         self.spatial_gating_unit = LKA(d_model)
#         self.proj_2 = nn.Conv2d(d_model, d_model, 1)

#     def forward(self, x):
#         shorcut = x.clone()
#         x = self.proj_1(x)
#         x = self.activation(x)
#         x = self.spatial_gating_unit(x)
#         x = self.proj_2(x)
#         x = x + shorcut
#         return x


class InstanceClassifier(nn.Module):
    def __init__(self,config):
        super(InstanceClassifier, self).__init__()
        # self.FE = FeatureExtractor
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(config.in_size, config.classes)
        # nn.init.kaiming_normal_(self.classifier.weight)
    def forward(self,features):
        # features = self.FE(x)

        # h = self.avgpool(features)
       
        # feats = h.view(h.size(0), -1)
    
        C = self.classifier(features)

        return features, C


class AttDual(nn.Module):
    
    def __init__(self,config):
        super(AttDual,self).__init__()
        self.config = config
        # self.vis_att = Attention(config.att_dim)   #visual att_dim features channel size 1536
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.i_classifier = InstanceClassifier(config)

        self.key = nn.Sequential(nn.Linear(config.in_size, config.in_size),
        nn.Dropout(config.dropout),
        nn.LayerNorm(config.in_size),
        nn.GELU())  #in_size=1536 after average pooling of features

        self.query = nn.Sequential(nn.Linear(config.in_size, config.in_size),
        nn.Dropout(config.dropout),
        nn.LayerNorm(config.in_size),
        nn.GELU())

        self.value = nn.Sequential(nn.Linear(config.in_size, config.in_size),
        nn.Dropout(config.dropout),
        nn.LayerNorm(config.in_size),
        nn.GELU())

        self.head = nn.Conv1d(config.classes, config.classes, kernel_size=config.in_size)


    def forward(self,features,c):
        # features = features.view(features.size(0), 1536,7,7)
        # features = self.vis_att(features) #apply visual spactial attention to features
        # print(f"features shape after visual spatial attention: {features.shape}")

      #classifier output is features after pooling/reshape to (N,K) and instance classes

        K = self.key(features)
        # print(f"Key after applying self.key: {K.shape}")
        V = self.value(K)  #N * K , unsorted
        Q = self.query(K)# The QK("query-key")circuits controls which features/tokens the head prefers to attend to.
        # print(f"Query after applying self.query to Key and no view(B,-1): {Q.shape}")

        # handle multiple classes without for loop
        _, m_indices = torch.sort(c, 0, descending=True) # sort class scores along the instance dimension, m_indices in shape N x C
        # print(f"m_indices shape: {m_indices.shape}")
        m_feats = torch.index_select(K, dim=0, index=m_indices[0, :]) # select critical instances, m_feats in shape C x K 
        # print(f"m_feats shape: {m_feats.shape}")
        q_max = self.query(m_feats) # compute queries of critical instances, q_max in shape C x Q
        # print(f"q_max shape: {q_max.shape}")
        A = torch.mm(Q, q_max.transpose(0, 1)) # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        # print(f"A score shape: {A.shape}")
        A = F.softmax( A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device = self.config.device)), 0) # normalize attention scores, A in shape N x C, 
        # print(f"A softmax normalized score shape: {A.shape}")
        B = torch.mm(A.transpose(0, 1), V) # compute bag representation, B in shape C x V
        # print(f"results after score multiply with Value shape: {B.shape}")
                
        B = B.view(1, B.shape[0], B.shape[1]) # 1 x C x V
        # print(f"B unsqueezed shape: {B.shape}")
        C = self.head(B) # 1 x C x 1
        # print(f"Class shape after 1 d convolution: {C.shape}")
        C = C.view(1, -1)
        # print(f"Class shape after reshape: {C.shape}")
        return C, A, B 


# class MultiheadAttention(nn.Module):



class DSMILNet(nn.Module):
    def __init__(self, i_classifier, b_classifier):
        super(DSMILNet, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier
        
        self.apply(self.init_weights)
        
    def init_weights(self, m):
       
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
            
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        feats, classes = self.i_classifier(x)
        prediction_bag, A, B = self.b_classifier(feats, classes)
        
        return classes, prediction_bag, A, B

## model
class AttentionBase(nn.Module):
    def __init__(self,config):
        super(AttentionBase, self).__init__()
        
        self.repr_length = config.in_size # size of representation per tile
        self.D = 128                               # N = batch size
        self.att = 1

        self.attention = nn.Sequential(             # N * repr_length
            nn.Linear(self.repr_length, self.D),    # N * D
            nn.Tanh(),                              # N * D
            nn.Linear(self.D, self.att)             # N * att
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.repr_length*self.att, config.out_size)
        )

        self.apply(self.init_weights)
        
    def init_weights(self, m):
       
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
            
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.float()                              # N * repr_lenght
        # print('x: ', x.shape)

        A = self.attention(x)                      # N * att

        # print(f'attention shape: {A.shape}')
        A = torch.transpose(A, 1,0)                # att * N
        A = nn.functional.softmax(A, dim=1)        # softmax over N
        #print('A: ', A)#.shape)
        M = torch.mm(A, x)             # att * repr_length
        # print('value shape: ', M.shape)
        
        out_prob = self.classifier(M)

        # print(f'out put: {out_prob} \n  output shape: {out_prob.shape}')
        # out_hat = torch.ge(out_prob, 0.5).float()
        
        return out_prob, A
    
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()
        
        return error, Y_hat
    
    def calculate_objective(self, Y_prob, Y):
        Y = Y.float()
        print(f"Y: {Y}")
        # Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        print(f"Y_prob: {Y_prob}")
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob)) # negative log bernoulli
        
        return neg_log_likelihood[0]


