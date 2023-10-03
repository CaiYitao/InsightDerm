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
import math






def scaled_dot_product(q, k, v):
    d_k = q.size()[-1]

    attn_logits = torch.matmul(q, k.transpose(-2, -1))

    attn_logits = attn_logits / math.sqrt(d_k)
    attention = F.softmax(attn_logits, dim=-1)

    values = torch.matmul(attention.transpose(-2,-1), v)

    return values, attention

#DUAL STREAM TOPK MultiHEAD ATTENTION
class TopkCFMultiHeadAttention(nn.Module):

    def __init__(self, config):
        super(TopkCFMultiHeadAttention,self).__init__()
        self.embed_dim = config.head_dim * config.num_heads
        self.embed_x_dim = config.embed_x_dim
        self.in_size = config.in_size
        self.topk = config.topk
        assert self.embed_dim % config.num_heads == 0, "Embedding dimension must be 0 modulo number of heads."
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.num_classes = config.classes
        self.classification = config.topkhead_classification
        self.critical_features_from = config.critical_features_from

        if config.embed_module =="Dense":
            self.embedding = nn.Linear(self.in_size, self.embed_x_dim)
            self.key_o = nn.Sequential(nn.Linear(self.topk * self.in_size, self.embed_dim),
                                        nn.LayerNorm(self.embed_dim),
                                        )

            self.key_e = nn.Sequential(nn.Linear(self.topk * self.embed_x_dim, self.embed_dim),
                                        nn.LayerNorm(self.embed_dim),
                                        )

            self.query_o = nn.Sequential(nn.Linear(self.in_size, self.embed_dim),
                                        nn.LayerNorm(self.embed_dim),
                                        )
            self.value_o = nn.Sequential(nn.Linear(self.in_size, self.embed_dim),
                                        nn.LayerNorm(self.embed_dim),
                                        )
    
        elif config.embed_module =="Sparse":
            self.embedding = nn.Sequential(
                nn.Linear(self.in_size, self.embed_x_dim),
                nn.Dropout(config.dropout),
                nn.GELU(),
                nn.LayerNorm(self.embed_x_dim),
            )
            self.key_o = nn.Sequential(
                nn.Linear(self.topk * config.in_size, self.embed_dim),
                nn.Dropout(config.dropout),
                nn.GELU(),
                nn.LayerNorm(self.embed_dim),                
            )
            self.key_e = nn.Sequential(
                nn.Linear(self.topk * self.embed_x_dim, self.embed_dim),
                nn.Dropout(config.dropout),
                nn.GELU(),
                nn.LayerNorm(self.embed_dim),                
            )
            self.query_o = nn.Sequential(
                nn.Linear(config.in_size, self.embed_dim),
                nn.Dropout(config.dropout),
                nn.GELU(),
                nn.LayerNorm(self.embed_dim),                
            )
            self.query_e = nn.Sequential(
                nn.Linear(self.embed_x_dim, self.embed_dim),
                nn.Dropout(config.dropout),
                nn.GELU(),
                nn.LayerNorm(self.embed_dim),                
            )
            self.value_o = nn.Sequential(
                nn.Linear(config.in_size, self.embed_dim),
                nn.Dropout(config.dropout),
                nn.GELU(),
                nn.LayerNorm(self.embed_dim),
            )
            self.value_e = nn.Sequential(
                nn.Linear(self.embed_x_dim, self.embed_dim),
                nn.Dropout(config.dropout),
                nn.GELU(),
                nn.LayerNorm(self.embed_dim),
            )
            

        self.projection_out = nn.Sequential(nn.Linear(self.embed_dim, config.in_size),nn.Dropout(config.dropout))

        self.norm = nn.LayerNorm(config.in_size)
        self.norm_c = nn.LayerNorm( self.head_dim * self.num_heads)

        self.instance_head = nn.Linear(config.in_size, config.classes)

        self.head_proj = config.head_proj

        if self.head_proj == "conv1d":
            self.head = nn.Conv1d(config.classes, config.classes, kernel_size=self.num_heads * self.head_dim)
        elif self.head_proj == "linear_diag":
            self.head = nn.Linear(self.embed_dim, config.classes)
        elif self.head_proj == "conv1d_ckernels":
            self.head = nn.Conv1d(config.classes, config.classes, kernel_size=self.num_heads * self.head_dim, groups=config.classes)

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

        seq_length, _ = x.shape
    
        c = self.instance_head(x).squeeze()

        _,topk_idx = torch.topk(c, self.topk, dim=0)

        if self.critical_features_from == "embedding":
            embed_x = self.embedding(x)
            critical_features = torch.stack([torch.index_select(embed_x, 0, topk_idx[i]) for i in range(self.topk)])
            critical_features = critical_features.permute(1,0,2).reshape(self.num_classes, self.topk * self.embed_x_dim)
            k = self.key_e(critical_features).reshape(self.num_classes,self.num_heads,self.head_dim) # [head_dim, num_heads, num_classes]
            q = self.query_e(embed_x).reshape(seq_length,self.num_heads,self.head_dim) # [head_dim, num_heads, num_classes]
            v = self.value_e(embed_x).reshape(seq_length,self.num_heads,self.head_dim) # [head_dim, num_heads, num_classes]

        elif self.critical_features_from == "original":
            critical_features = torch.stack([torch.index_select(x, 0, topk_idx[i]) for i in range(self.topk)])
       
            critical_features = critical_features.permute(1,0,2).reshape(self.num_classes, self.topk * self.in_size)           

            k = self.key_o(critical_features).reshape(self.num_classes,self.num_heads,self.head_dim) # [head_dim, num_heads, num_classes]

            q = self.query_o(x).reshape(seq_length, self.num_heads, self.head_dim)

            v = self.value_o(x).reshape(seq_length, self.num_heads, self.head_dim)

        elif self.critical_features_from == "index_add":
            critical_features = torch.zeros((seq_length,self.topk, self.in_size)).to(x.device)
            for i in range(self.topk):
                critical_features[:,i,:] = critical_features[:,i,:].index_add(0, topk_idx[i], torch.index_select(x, 0, topk_idx[i]))

            critical_features = critical_features.reshape(seq_length, self.topk * self.in_size)

            k = self.key_o(critical_features).reshape(seq_length,self.num_heads,self.head_dim) # [head_dim, num_heads, num_classes]

        q = q.permute(1, 0, 2)

        k = k.permute(1, 0, 2)

        v = v.permute(1, 0, 2)

        values,attention = scaled_dot_product(q, k, v)
        values = values.permute(1, 0, 2) # [num_classes or seq_length, num_heads, head_dim]

        if self.classification:
            v = values.reshape(self.num_classes, self.head_dim * self.num_heads)
            v = self.norm_c(v)
            o = self.head(v)
            if self.head_proj == "linear_diag":
                o = torch.diagonal(o)

 
        else:
            if self.critical_features_from == "index_add":
                values = values.reshape(seq_length,self.head_dim * self.num_heads)

            else:
                values = values.reshape(self.num_classes, self.head_dim * self.num_heads)
            o = self.projection_out(values)
            o = self.norm(o)
              
        return o,c,attention,topk_idx



class CriticalFeaturesBlock(nn.Module):
    def __init__(self, config):
        super(CriticalFeaturesBlock, self).__init__()
        self.topk_attention = TopkCFMultiHeadAttention(config)
        self.dim_feedforward = config.head_dim
        self.mlp = nn.Sequential(
            nn.Linear(config.in_size, self.dim_feedforward),
            nn.Dropout(config.dropout),
            nn.GELU(),
            nn.Linear(self.dim_feedforward, config.in_size),
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

        if self.critical_features_from == "index_add":
            x = x + o
        else:
            for i in range(self.topk):
                x = x.index_add(0, topk_idx[i], o.float())
    
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

        output = self.head(x).squeeze()

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
    
