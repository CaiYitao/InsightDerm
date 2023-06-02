
import torch, gc
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import convnext_large, ConvNeXt_Large_Weights
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import timm
from PIL import Image
import glob
import csv
import random
import numpy as np
import os
import pandas as pd
import wandb
from sklearn.utils import shuffle, class_weight
from sklearn.metrics import roc_auc_score,roc_curve,auc,balanced_accuracy_score,accuracy_score,cohen_kappa_score,matthews_corrcoef,f1_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from base_model import InstanceClassifier, AttDual, DSMILNet
import datetime, copy
import matplotlib.pyplot as plt
import argparse
from utils import get_feats_for_dataset,build_testloader,get_working_df,get_feats_df,get_bag_feats,compute_roc,multi_label_roc,plot_auc,build_optimizer,build_scheduler



def run_bags(model, train_df, optimizer, criterion, config):
    model.train()
    # train_df = shuffle(train_df).reset_index(drop=True)
    Loss = 0 
    scaler = GradScaler()   
    for i in range(len(train_df)):
        optimizer.zero_grad()
        feats, label = get_bag_feats(train_df.iloc[i])
        
        feats = torch.from_numpy(feats).float().to(config.device)
        
        feats = get_feats_for_dataset(feats, config.dataset)

        label = torch.tensor(label).long().to(config.device)
        # print(f'label: {label}   label size: {label.shape}')
        with torch.cuda.amp.autocast():
            
            if config.topkhead_classification:
                bag_pred, ins_pred, _,_ = model(feats)
            else:
                bag_pred,ins_pred,_ = model(feats)
         
            max_pred,_ = torch.max(ins_pred, dim=0)

            # print(f'bag_pred : {bag_pred}  \n instance pred: {ins_pred}')
    
            # print(f'max_pred: {max_pred}    label: {label}')
            bag_pred = bag_pred.squeeze()
            loss_b = criterion(bag_pred, label)
            # print(f'loss_b: {loss_b}')
            loss_m = criterion(max_pred, label)
            # print(f'loss_m: {loss_m}')
            loss = config.alpha * loss_b + (1-config.alpha) * loss_m
            # print(f"loss during training: {loss}")
   
            # print(f"loss after regulation during training: {loss}")         
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
        scaler.step(optimizer)
        scaler.update()
        if torch.isnan(loss):
            loss = torch.tensor(1e-6).to(config.device)
        else:
            loss = loss
        Loss += loss.item()
        # print('Training (shuffled) bag [%d/%d] bag loss: %.4f' % (i, len(train_df), loss.item()))
        wandb.log({"Step Train Loss": loss.item()})

    return Loss / len(train_df)


def run_weighted_bags(model, train_df, optimizer, criterion, config):
    model.train()
    # train_df = shuffle(train_df).reset_index(drop=True)
    Loss = 0 
    scaler = GradScaler()  
    n = len(train_df)

    for i in range(n//config.batch_size):
        if config.classes == 3:
            w = train_df['labels'].map({0: config.w0, 1: config.w1, 2: config.w2})
        elif config.classes == 2:
            w = train_df['labels'].map({0: config.w0, 1: config.w1})
        
        weighted_spl_df = train_df.sample(n=config.batch_size, weights=w, random_state= config.seed,axis=0)
        weighted_spl_df = shuffle(weighted_spl_df).reset_index(drop=True)
        for j in range(len(weighted_spl_df)):
            optimizer.zero_grad()
            feats, label = get_bag_feats(weighted_spl_df.iloc[j])
            
            feats = torch.from_numpy(feats).float().to(config.device)
            
            feats = get_feats_for_dataset(feats, config.dataset)

            label = torch.tensor(label).long().to(config.device)

            with torch.cuda.amp.autocast():
                if config.topkhead_classification:
                    bag_pred, ins_pred, _,_ = model(feats)
                else:
                    bag_pred,ins_pred,_ = model(feats)
                max_pred,_ = torch.max(ins_pred, dim=0)

                # print(f'bag_pred : {bag_pred}  \n instance pred: {ins_pred}')
        
                # print(f'max_pred: {max_pred}    label: {label}')
                bag_pred = bag_pred.squeeze()
                loss_b = criterion(bag_pred, label)
                # print(f'loss_b: {loss_b}')
                loss_m = criterion(max_pred, label)
                # print(f'loss_m: {loss_m}')
                loss = config.alpha * loss_b + (1-config.alpha) * loss_m

            if torch.isnan(loss):
                loss = torch.tensor(1e-6).to(config.device)
            else:
                loss = loss     

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            scaler.step(optimizer)
            scaler.update()

            Loss += loss.item()
        # print('Training (shuffled) bag [%d/%d] bag loss: %.4f' % (i, len(weighted_spl_df), loss.item()))
            wandb.log({"Step Train Loss": loss.item()})
            # gc.collect()
            # torch.cuda.empty_cache()
    return Loss / len(train_df)

def evaluate_bags(model, valid_df, criterion, config):
    model.eval()
    # valid_df= shuffle(valid_df).reset_index(drop=True)
    Loss = 0
    pred = []
    labels = []
    with torch.no_grad():
        for i in range(len(valid_df)):
            feats, label = get_bag_feats(valid_df.iloc[i])
            labels.extend([label])
            feats = torch.from_numpy(feats).float().to(config.device)
            feats = get_feats_for_dataset(feats, config.dataset)
                
            label = torch.tensor(label).long().to(config.device)
            if config.topkhead_classification:
                bag_pred, ins_pred, _,_ = model(feats)
            else:
                bag_pred,ins_pred,_ = model(feats)
            # print(f'bag_pred : {bag_pred} bag pred softmax:{torch.softmax(bag_pred,dim=1)}  \n instance pred: {ins_pred}')
            max_pred,_ = torch.max(ins_pred, dim=0)
            # print(f'max_pred: {max_pred}  ')
            # print(f'max pred softmax: {torch.softmax(max_pred,dim=1)}')
            bag_pred = bag_pred.squeeze()
            loss_b = criterion(bag_pred, label)
            loss_m = criterion(max_pred, label)
            loss = config.alpha * loss_b + (1-config.alpha) * loss_m
            pred.extend([(config.alpha * torch.softmax(bag_pred,dim=0) + (1-config.alpha) * torch.softmax(max_pred,dim=0)).squeeze().cpu().numpy()])
            # print(f'prediction after softmax : {pred[-1]}')

            if torch.isnan(loss):
                loss = torch.tensor(1e-6).to(config.device)
            else:
                loss = loss
            
            Loss += loss.item()
            # print('Testing bag [%d/%d] bag loss: %.4f' % (i, len(valid_df), loss.item()))
            wandb.log({"Step Test Loss": loss.item()})
   
    pred = np.array(pred)
    labels = np.array(labels)
    # print(f'Prediction : {pred}  size : {len(pred)}')
    # print(f'Labels array : {labels}  size : {labels.shape}')



    if config.classes == 3:
        binary_labels = label_binarize(labels, classes=[0, 1, 2])
        auc_value = multi_label_roc(binary_labels, pred, config)
        plot = wandb.plot.roc_curve(labels, pred,labels=['NORMAL vs. BAS-SCC','BAS vs. NORMAL-SCC','SCC vs. NORMAL-BAS'])
        precision_recall = wandb.plot.pr_curve(labels, pred,labels=['NORMAL vs. BAS-SCC','BAS vs. NORMAL-SCC','SCC vs. NORMAL-BAS'])
    elif config.classes == 2:
        binary_labels = nn.functional.one_hot(torch.tensor(labels),config.classes).numpy()
        auc_value = multi_label_roc(binary_labels, pred, config)
        plot = wandb.plot.roc_curve(labels, pred,labels=['NORMAL','BAS'])
        precision_recall = wandb.plot.pr_curve(labels, pred,labels=['NORMAL','BAS'])
        

    wandb.log({"roc": plot})
    # wandb.sklearn.plot_roc(labels, pred, labels=['NORMAL vs. BAS-SCC','BAS vs. NORMAL-SCC','SCC vs. NORMAL-BAS'])
    wandb.log({"pr": precision_recall})
    # wandb.sklearn.plot_precision_recall(labels, pred, labels=['NORMAL vs. BAS-SCC','BAS vs. NORMAL-SCC','SCC vs. NORMAL-BAS'])
    # pred = np.where(pred >= thresholds_optimal, 1, 0)
    pred_ = np.argmax(pred, axis=1)

    # return Loss / len(valid_df)
    return Loss / len(valid_df), pred_,labels,pred,auc_value




