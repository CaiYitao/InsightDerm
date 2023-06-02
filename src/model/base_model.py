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




import math
def scaled_dot_product(q, k, v):
    d_k = q.size()[-1]
    # print(f"dk {d_k}")
    # print(f"k with shape after transpose  {k.transpose(-2, -1).shape}")
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    # print(f"attn_logits{attn_logits} with shape [num_head, seq_length, num_classes]  {attn_logits.shape}")
    attn_logits = attn_logits / math.sqrt(d_k)
    attention = F.softmax(attn_logits, dim=-1)
    # print(f"attention {attention} with shape [seq_length, num_classes] {attention.shape}")
    values = torch.matmul(attention.transpose(-2,-1), v)
    # print(f"values with shape [num_head, num_classes, head_dim] {values.shape}")
    return values, attention

#Old version of DUAL STREAM TOPK Critical Features MultiHEAD ATTENTION, NEW VERSION IS IN MODEL_1.PY
class TopkCFMultiHeadAttention(nn.Module):

    def __init__(self, config):
        super(TopkCFMultiHeadAttention,self).__init__()
        self.embed_dim = config.in_size * config.num_heads
        self.topk = config.topk
        assert self.embed_dim % config.num_heads == 0, "Embedding dimension must be 0 modulo number of heads."
        self.num_heads = config.num_heads
        self.head_dim = self.embed_dim // config.num_heads
        self.num_classes = config.classes
        self.classification = config.topkhead_classification
        self.critical_features_from = config.critical_features_from

        if config.embed_module =="Dense":
            self.embedding = nn.Linear(config.in_size, self.embed_dim)
            self.key_o = nn.Sequential(nn.Linear(self.topk * config.in_size, self.embed_dim),
                                        nn.LayerNorm(self.embed_dim),
                                        )

            self.key_e = nn.Sequential(nn.Linear(self.topk * self.embed_dim, self.embed_dim),
                                        nn.LayerNorm(self.embed_dim),
                                        )

            self.query = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim),
                                        nn.LayerNorm(self.embed_dim),
                                        )
            self.value = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim),
                                        nn.LayerNorm(self.embed_dim),
                                        )
    
        elif config.embed_module =="Sparse":
            self.embedding = nn.Sequential(
                nn.Linear(config.in_size, self.embed_dim),
                nn.Dropout(config.dropout),
                nn.GELU(),
                nn.LayerNorm(self.embed_dim),
            )
            self.key_o = nn.Sequential(
                nn.Linear(self.topk * config.in_size, self.embed_dim),
                nn.Dropout(config.dropout),
                nn.GELU(),
                nn.LayerNorm(self.embed_dim),                
            )
            self.key_e = nn.Sequential(
                nn.Linear(self.topk * self.embed_dim, self.embed_dim),
                nn.Dropout(config.dropout),
                nn.GELU(),
                nn.LayerNorm(self.embed_dim),                
            )
            self.query = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.Dropout(config.dropout),
                nn.GELU(),
                nn.LayerNorm(self.embed_dim),                
            )
            self.value = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.Dropout(config.dropout),
                nn.GELU(),
                nn.LayerNorm(self.embed_dim),
            )

        self.projection_out = nn.Sequential(nn.Linear(self.embed_dim, config.in_size),nn.Dropout(config.dropout))
        # self.projection = config.projection
        self.norm = nn.LayerNorm(config.in_size)
        self.norm_c = nn.LayerNorm( self.head_dim * self.num_heads)
        # self.norm_k = nn.LayerNorm( self.head_dim * self.num_heads)
        # self.norm_q = nn.LayerNorm( self.head_dim * self.num_heads)
        # self.norm_v = nn.LayerNorm( self.head_dim * self.num_heads)

        self.instance_head = nn.Linear(config.in_size, config.classes)
        self.head = nn.Conv1d(config.classes, config.classes, kernel_size=self.num_heads * config.in_size)

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
        # print(f"x {x} should be with shape [seq_length, in_size]: {x.shape}")
        seq_length, _ = x.shape
        # print(f"x  {x[0]} ")
        c = self.instance_head(x).squeeze()
        # print(f"c {c} with shape {c.shape} should be with shape [seq_length, num_classes]: {c.shape}")
        _,topk_idx = torch.topk(c, self.topk, dim=0)
        # print(f"topk_idx {topk_idx} should be with shape [num_heads,num_classes]: {topk_idx.shape}")

        embed_x = self.embedding(x)
        # print(f"embed_x {embed_x} should be with shape[seq_length,embed_dim]: {embed_x.shape}")
        # print(f"embed_features after permutation should be with shape[num_heads, seq_length, head_dim]: {embed_features.shape}")

        # if self.critical_features_from == "embedding":
        #     embed_features = embed_x.reshape(seq_length, self.num_heads, self.head_dim)
        #     # print(f"embed_features {embed_features} after reshape should be with shape[seq_length, num_heads, head_dim]: {embed_features.shape}")
        #     embed_features = embed_features.permute(1, 0, 2) # [num_heads, seq_length, head_dim]
        #     critical_features = torch.stack([torch.index_select(embed_features[i], 0, topk_idx[i]) for i in range(self.topk)])
        #     # print(f"critical_features from embedding {critical_features} should be with shape[num_topk, num_classes, head_dim]: {critical_features.shape}")
        # else:
        if self.critical_features_from == "index_add":
            critical_features = torch.zeros((seq_length,self.topk, self.head_dim)).to(x.device)
            for i in range(self.topk):
                critical_features[:,i,:] = critical_features[:,i,:].index_add(0, topk_idx[i], torch.index_select(x, 0, topk_idx[i]))
                # print(f"topk index {topk_idx[i]}  selected x {torch.index_select(x, 0, topk_idx[i])} ")
                # print(f"critical_features[:,{i},:] {critical_features[:,i,:][topk_idx[i]]} should be with shape[seq_length, head_dim]: {critical_features[:,i,:].shape}")
            critical_features = critical_features.reshape(seq_length, self.topk * self.head_dim)
            # print(f"critical_features {critical_features[topk_idx.flatten()]} ")
            k = self.norm_k(self.key_o(critical_features)).reshape(seq_length,self.num_heads,self.head_dim) # [head_dim, num_heads, num_classes]

    
        elif self.critical_features_from == "embedding":
                critical_features = torch.stack([torch.index_select(embed_x, 0, topk_idx[i]) for i in range(self.topk)])

                critical_features = critical_features.permute(1,0,2).reshape(self.num_classes, self.topk * self.embed_dim)
                k = self.norm_k(self.key_e(critical_features)).reshape(self.num_classes,self.num_heads,self.head_dim) # [head_dim, num_heads, num_classes]

        elif self.critical_features_from == "original":
                critical_features = torch.stack([torch.index_select(x, 0, topk_idx[i]) for i in range(self.topk)])
                # print(f"critical_features from input {critical_features} should be with shape[num_topk, num_classes, head_dim]: {critical_features.shape}")
                
                critical_features = critical_features.permute(1,0,2).reshape(self.num_classes, self.topk * self.head_dim)
                
                # k = self.norm_k(self.key_o(critical_features)).reshape(self.num_classes,self.num_heads,self.head_dim) # [head_dim, num_heads, num_classes]
                k = self.norm_c(self.key_o(critical_features)).reshape(self.num_classes,self.num_heads,self.head_dim) # [head_dim, num_heads, num_classes]
            

        # q = self.norm_q(self.query(embed_x)).reshape(seq_length, self.num_heads, self.head_dim)
        q = self.norm_c(self.query(embed_x)).reshape(seq_length, self.num_heads, self.head_dim)
        # v = self.norm_v(self.value(embed_x)).reshape(seq_length, self.num_heads, self.head_dim)
        v = self.norm_c(self.value(embed_x)).reshape(seq_length, self.num_heads, self.head_dim)
        # print(f"k should be with shape[num_classes, num_heads, head_dim]: {k.shape}")
        #permute q k, v to [num_heads, seq_length, head_dim]
        q = q.permute(1, 0, 2)
        # print(f"q should be with shape[num_heads, seq_length, head_dim]: {q.shape}")
        k = k.permute(1, 0, 2)
        # print(f"k should be with shape[num_heads, num_classes or seq_length, head_dim]: {k.shape}")
        v = v.permute(1, 0, 2)
        # print(f"v {v}should be with shape[num_heads, seq_length, head_dim]: {v.shape}")
        values,attention = scaled_dot_product(q, k, v)
        values = values.permute(1, 0, 2) # [num_classes or seq_length, num_heads, head_dim]
        # print(f"values {v} should be with shape[num_classes or seq_length, num_heads, head_dim]: {values.shape}")
        if self.classification:
            v = values.reshape(self.num_classes, self.head_dim * self.num_heads)
            v = self.norm_c(v)
            o = self.head(v)   
        else:
            if self.critical_features_from == "index_add":
                values = values.reshape(seq_length,self.head_dim * self.num_heads)
            # print(f"values {values} should be with shape[seq_length, embed_dim]: {values.shape}")
            else:
                values = values.reshape(self.num_classes, self.head_dim * self.num_heads)
            o = self.projection_out(values)
            # print(f"o with projection {o} should be with shape[seq_length or num_classes, in_size]: {o.shape}")
            o = self.norm(o)
  
                # print(f"o with torch.sum  {o} should be with shape[seq_length, in_size]: {o.shape}")
              
        return o,c,attention,topk_idx



class CriticalFeaturesBlock(nn.Module):
    def __init__(self, config):
        super(CriticalFeaturesBlock, self).__init__()
        self.topk_attention = TopkCFMultiHeadAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.in_size, config.in_size),
            nn.Dropout(config.dropout),
            nn.LayerNorm(config.in_size),
            nn.GELU()
        )
        self.mlp_c = nn.Sequential(
            nn.Linear(config.classes, config.classes),
            nn.Dropout(config.dropout),
            nn.LayerNorm(config.classes),
            nn.GELU()
        )
        self.instance_head = nn.Linear(config.in_size, config.classes)
        self.add_mlp = config.add_mlp 
        self.norm = nn.LayerNorm(config.in_size)
        self.norm_mlp = nn.LayerNorm(config.in_size)
        self.norm_c = nn.LayerNorm(config.classes)
        self.norm_mlp_c = nn.LayerNorm(config.classes)
        self.dropout = nn.Dropout(config.dropout)
        self.num_heads = config.num_heads
        self.norm_o = nn.LayerNorm(config.in_size)
        self.topk = config.topk
        self.critical_features_from = config.critical_features_from


    def forward(self, x):
        
        o,c,attention,topk_idx = self.topk_attention(x)
        # print(f"the output 'o' of TopkCFMultiHeadAttention: {o} o should be with shape[num_classes, in_size ]: {o.shape}")

        # o = self.norm_o(o)

        # xo = torch.zeros_like(x)
        # for idxs in topk_idx:
        #     for i,j in enumerate(idxs):
        #         xo[j] = xo[j] + o[i]
        if self.critical_features_from == "index_add":
            x = x + o
        else:
            for i in range(self.topk):
                x = x.index_add(0, topk_idx[i], o.float())
        # print(f"x in the block :{x} after index_add should be with shape[seq_length, in_size]: {x.shape}")
        x = self.norm(x)
        
        if self.add_mlp:
            x = self.norm_mlp(x + self.dropout(self.mlp(x)))
        else:
            x = x 
        return x,c,attention


class CF_Transformer(nn.Module):
    def __init__(self, config):
        super(CF_Transformer, self).__init__()
        self.config = config
        self.blocks = nn.ModuleList([CriticalFeaturesBlock(config) for _ in range(config.num_layers)])
        self.head = nn.Linear(config.in_size, config.classes)
        # self.head = nn.Conv1d(1, config.classes, kernel_size=config.in_size)       
        # self.head = nn.Conv1d(config.classes, config.classes, kernel_size=config.in_size)
        self.apply(self.init_weights)
        self.classification = config.classification

        
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
        seq_length = x.shape[0]
        attention = []
        for block in self.blocks:
            x,c,attention_block = block(x)
            attention.append(attention_block)
        
        # print(f"x in transformer {x} should be with shape[seq_length,in_size]: {x.shape}")
        output = self.head(x).squeeze()
        # print(f"output should be with shape[seq_length, num_classes]: {output.shape}")
        attention = torch.stack(attention).squeeze()
        if self.classification == "mean":
            output = torch.mean(output,dim=0)
        elif self.classification == "avgpool1d":
            output = F.adaptive_avg_pool1d(output.T, 1).squeeze()
        elif self.classification == "LPPool1d":
            output = F.lp_pool1d(output.T, 2, seq_length).squeeze()
        elif self.classification == "maxpool1d":
            output = F.adaptive_max_pool1d(output.T, 1).squeeze()
        return output,c , attention




class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention,self).__init__()
        self.embed_dim = config.in_size * config.topk_heads
        assert self.embed_dim % config.topk_heads == 0, "Embedding dimension must be 0 modulo number of heads."
        self.num_heads = config.topk_heads
        self.head_dim = self.embed_dim // config.topk_heads
        self.num_classes = config.classes
        # self.mask = config.mask
        self.classification = config.topkhead_classification
        self.critical_features_from = config.critical_features_from

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        if config.embed_module =="Dense":

            self.key = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim),
                                        nn.LayerNorm(self.embed_dim),
                                        )

            self.query = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim),
                                        nn.LayerNorm(self.embed_dim),
                                        )
            self.value = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim),
                                        nn.LayerNorm(self.embed_dim),
                                        )

        elif config.embed_module =="Sparse":
            self.key = nn.Sequential(
                nn.Linear(config.in_size, self.embed_dim),
                nn.Dropout(config.dropout),
                nn.LayerNorm(self.embed_dim),
                nn.GELU()
            )
            self.query = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.Dropout(config.dropout),
                nn.LayerNorm(self.embed_dim),
                nn.GELU()
            )
            self.value = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.Dropout(config.dropout),
                nn.LayerNorm(self.embed_dim),
                nn.GELU()
            )

        self.projection_out = nn.Sequential(nn.Linear(self.embed_dim, config.in_size),nn.Dropout(config.dropout))
        self.projection = config.projection
        self.norm = nn.LayerNorm(config.in_size)
        self.norm_c = nn.LayerNorm(self.num_heads * self.head_dim)

        self.instance_head = nn.Linear(config.in_size, config.classes)
        self.head = nn.Conv1d(config.classes, config.classes, kernel_size=config.in_size)
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

    def forward(self,x):
        seq_length = x.shape[0]
        c = self.instance_head(x).squeeze()
        k = self.key(x).reshape(seq_length, self.num_heads, self.head_dim)
        q= self.query(x).reshape(seq_length, self.num_heads, self.head_dim)
        v = self.value(x).reshape(seq_length, self.num_heads, self.head_dim)
        q = q.permute(1, 0, 2)
        # print(f"q should be with shape[num_heads, seq_length, head_dim]: {q.shape}")
        k = k.permute(1, 0, 2)
        # print(f"k should be with shape[num_heads, seq_length, head_dim]: {k.shape}")
        v = v.permute(1, 0, 2)
        # print(f"v {v}should be with shape[num_heads, seq_length, head_dim]: {v.shape}")
        values,attention = scaled_dot_product(q, k, v)
        values = values.permute(1, 0, 2) # [seq_length, num_heads, head_dim]

        values = values.reshape(seq_length,self.head_dim * self.num_heads)
        print(f"values {values} should be with shape[num_classes, embed_dim]: {values.shape}")
        o = self.projection_out(values)
        print(f"o with projection {o} should be with shape[num_classes, in_size]: {o.shape}")

        return o, c, attention



class AttentionBlock(nn.Module):
    def __init__(self, config):
        super(AttentionBlock, self).__init__()
        self.multihead_attention = MultiHeadAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.in_size, config.in_size),
            nn.Dropout(config.dropout),
            nn.LayerNorm(config.in_size),
            nn.GELU()
        )
        self.mlp_c = nn.Sequential(
            nn.Linear(config.classes, config.classes),
            nn.Dropout(config.dropout),
            nn.LayerNorm(config.classes),
            nn.GELU()
        )
        self.instance_head = nn.Linear(config.in_size, config.classes)
        self.add_mlp = config.add_mlp 
        self.norm = nn.LayerNorm(config.in_size)
        self.norm_mlp = nn.LayerNorm(config.in_size)
        self.norm_c = nn.LayerNorm(config.classes)
        self.norm_mlp_c = nn.LayerNorm(config.classes)
        self.dropout = nn.Dropout(config.dropout)
        self.num_heads = config.topk_heads
        self.norm_o = nn.LayerNorm(config.in_size)


    def forward(self, x):
        
        o,c,attention = self.multihead_attention(x)
        print(f"the output 'o' of TopkCFMultiHeadAttention: {o} o should be with shape[num_classes, in_size ]: {o.shape}")
        x = x + self.dropout(o)
        print(f"x in the block :{x} after index_add should be with shape[seq_length, in_size]: {x.shape}")
        x = self.norm(x)
        if self.add_mlp:
            x = self.norm_mlp(x + self.dropout(self.mlp(x)))
        else:
            x = x 
        return x,c,attention



class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.config = config
        self.classification = config.classification
        self.blocks = nn.ModuleList([AttentionBlock(config) for _ in range(config.num_layers)])
        self.head = nn.Linear(config.in_size, config.classes)
        # self.head = nn.Conv1d(1, config.classes, kernel_size=config.in_size)       
        # self.head = nn.Conv1d(config.classes, config.classes, kernel_size=config.in_size)
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
        seq_length = x.shape[0]
        attention = []
        for block in self.blocks:
            x,c,attention_block = block(x)
            attention.append(attention_block)
        
        print(f"x in transformer {x} should be with shape[seq_length,in_size]: {x.shape}")
        output = self.head(x).squeeze()
        print(f"output should be with shape[seq_length, num_classes]: {output.shape}")
        attention = torch.stack(attention).squeeze()
        if self.classification == "mean":
            output = torch.mean(output,dim=0)
        elif self.classification == "avgpool1d":
            output = F.adaptive_avg_pool1d(output.T, 1).squeeze()
        elif self.classification == "LPPool1d":
            output = F.lp_pool1d(output.T, 2, seq_length).squeeze()
        elif self.classification == "maxpool1d":
            output = F.adaptive_max_pool1d(output.T, 1).squeeze()
        return output,c,attention