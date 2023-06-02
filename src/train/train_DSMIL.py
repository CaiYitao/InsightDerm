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

parser = argparse.ArgumentParser(description='DSMIL Training')
parser.add_argument('--seed', default=0, type=int, help='seed')
parser.add_argument('--device', default='cuda:0', type=str, help='device')
parser.add_argument('--dataset', default='CNT', type=str, help='features dataset name')
parser.add_argument('--num_classes', default=3, type=int, help='number of classes')
parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run')
parser.add_argument('--n', default=1, type=int, help='number of training process')
parser.add_argument('--split', default="determined", type=str, help='split method if ,dtermined, use working_file sets as split, if ,random, use random stratified split')
parser.add_argument('--df', default="df_combined", type=str, help='dataframe to use for training and validation, for example: df_combined, df, df_vig,df_van')
parser.add_argument('--weighted_sampling', default=False, type=bool, help='use weighted sampling for training')
parser.add_argument('--batch_size', default=32, type=int, help='weighted sampling batch size for training')
parser.add_argument('--w0', default=33, type=int, help='weighted sampling w0 for class 0 normal')
parser.add_argument('--w1', default=83, type=int, help='weighted sampling w1 for class 1 BAS')
parser.add_argument('--w2', default=65, type=int, help='weighted sampling w2 for class 2 SCC')
parser.add_argument('--in_size', default=1000, type=int, help='input size of the model')



args = parser.parse_args()



def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class Config:
    grad_norm_clip = 1.02580285460309
    alpha = 0.849435264741892
    in_size  = 1000
    classes = args.num_classes
    out_size = args.num_classes
    dropout = 0.3
    lr = 0.0203373691328209
    weight_decay = 0.0001
    # batch_size = 40
    epochs = args.epochs
    device = torch.device(args.device)
    optimizer = "SGD"
    scheduler = "cosine"
    betas = (0.5, 0.999)
    momentum = 0.9
    minlr = 0.000005
    thresh_prob = 0
    dataset= args.dataset
    w0 = 74   #84
    w1 =  16   #64
    w2 =   79  #86

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)


def get_working_df(feats_files,metadata_file):
    """
    Collects the features files from the files in the given path.
    """
    metadata = pd.read_csv(metadata_file)
    metadata = metadata[metadata["diagnosis"]!="unsure"]
    # metadata['label'] = metadata['diagnosis'].apply(lambda x: 0 if x =='normal' else (2 if x == 'scc' or x=='plep' or x== 'mb bowen_bowen_plep' else 1))

    labels = []
    sets = []
    files = []
    for file in feats_files:
          bag_name = os.path.basename(file).split('.')[0]
          if bag_name in metadata["image_nr"].values:
           
      #     print(metadata.loc[metadata["image_nr"] == bag_name, 'label'].values)
            bag_label = metadata.loc[metadata["image_nr"] == bag_name, 'label'].values[0]
            bag_set = metadata.loc[metadata["image_nr"] == bag_name, 'set'].values[0]
            labels.append(bag_label)
            sets.append(bag_set)
            files.append(file)


    df = pd.DataFrame(files, columns=["feats_files"])
  
    df["labels"] = labels
    df["sets"] = sets
    return df


def get_feats_df(path,metadata_file):
    """
    Collects the features files from the files in the given path.
    """
    metadata = pd.read_csv(metadata_file)
    feats_files = sorted(glob.glob(os.path.join(path, "*.csv"), recursive=True))
    labels = []
    for file in feats_files:
          bag_name = os.path.basename(file).split('.')[0]
      

          bag_label = metadata.loc[metadata["image_nr"] == bag_name, 'label'].values[0]
          labels.append(bag_label)
    df = pd.DataFrame(feats_files, columns=["feats_files"])
  
    df["labels"] = labels
    return df

def get_bag_feats(feats_df):
    feats= pd.read_csv(feats_df.iloc[0], header=None)
    feats = feats.to_numpy()
    label = feats_df.iloc[1]
    return feats, label


def get_feats_for_dataset(feats,dataset):
        f_cnt= feats[:,:1000]
        f_vig = feats[:,1000:2000]
        f_van = feats[:,2000:3000]
        if dataset == 'VAN':
            feats = f_van
        elif dataset == 'VIG':
            feats = f_vig
        elif dataset == 'CNT':
            feats = f_cnt
        elif dataset == 'CNT_VIG':
            feats = torch.cat((f_cnt, f_vig), dim=1)
        elif dataset == 'VIG_CNT':
            feats = torch.cat((f_vig, f_cnt), dim=1)
        elif dataset == 'VAN_VIG':
            feats = torch.cat((f_van, f_vig), dim=1)    
        elif dataset == 'CNT_VAN':
            feats = torch.cat((f_cnt, f_van), dim=1)
        elif dataset == 'VIG_VAN':
            feats = torch.cat((f_vig, f_van), dim=1)
        elif dataset == 'VAN_CNT':
            feats = torch.cat((f_van, f_cnt), dim=1)
        
        elif dataset == 'CNT_VIG_VAN':
            feats = torch.cat((f_cnt, f_vig, f_van), dim=1)
        elif dataset == 'VAN_VIG_CNT':
            feats = torch.cat((f_van, f_vig, f_cnt), dim=1)
        elif dataset == 'VIG_CNT_VAN':
            feats = torch.cat((f_vig, f_cnt, f_van), dim=1)
        elif dataset == 'CNT_VAN_VIG':
            feats = torch.cat((f_cnt, f_van, f_vig), dim=1)
        elif dataset == 'VAN_CNT_VIG':
            feats = torch.cat((f_van, f_cnt, f_vig), dim=1)
        elif dataset == 'VIG_VAN_CNT':
            feats = torch.cat((f_vig, f_van, f_cnt), dim=1)
        return feats

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
        with torch.cuda.amp.autocast():
            ins_pred, bag_pred, _,_ = model(feats)
            max_pred,_ = torch.max(ins_pred, dim=0)

            # print(f'bag_pred : {bag_pred}  \n instance pred: {ins_pred}')
     
            # print(f'max_pred: {max_pred}    label: {label}')
            bag_pred = bag_pred.squeeze()
            loss_b = criterion(bag_pred, label)
            # print(f'loss_b: {loss_b}')
            loss_m = criterion(max_pred, label)
            # print(f'loss_m: {loss_m}')
            loss = config.alpha * loss_b + (1-config.alpha) * loss_m
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
        scaler.step(optimizer)
        scaler.update()

        Loss += loss.item()
    # print('Training (shuffled) bag [%d/%d] bag loss: %.4f' % (i, len(train_df), loss.item()))
        wandb.log({"Step Train Loss": loss.item()})
        # gc.collect()
        # torch.cuda.empty_cache()
    return Loss / len(train_df)


def run_weighted_bags(model, train_df, optimizer, criterion, config):
    model.train()
    # train_df = shuffle(train_df).reset_index(drop=True)
    Loss = 0 
    scaler = GradScaler()  
    n = len(train_df)


    for i in range(n//args.batch_size):
        if args.num_classes == 3:
            w = train_df['labels'].map({0: config.w0, 1: config.w1, 2: config.w2})
        elif args.num_classes == 2:
            w = train_df['labels'].map({0: config.w0, 1: config.w1})
        
        weighted_spl_df = train_df.sample(n=args.batch_size, weights=w, random_state= args.seed,axis=0)
        weighted_spl_df = shuffle(weighted_spl_df).reset_index(drop=True)
        for j in range(len(weighted_spl_df)):
            optimizer.zero_grad()
            feats, label = get_bag_feats(weighted_spl_df.iloc[j])
            
            feats = torch.from_numpy(feats).float().to(config.device)
            
            feats = get_feats_for_dataset(feats, config.dataset)

            label = torch.tensor(label).long().to(config.device)
            with torch.cuda.amp.autocast():
                ins_pred, bag_pred, _,_ = model(feats)
                max_pred,_ = torch.max(ins_pred, dim=0)

                # print(f'bag_pred : {bag_pred}  \n instance pred: {ins_pred}')
        
                # print(f'max_pred: {max_pred}    label: {label}')
                bag_pred = bag_pred.squeeze()
                loss_b = criterion(bag_pred, label)
                # print(f'loss_b: {loss_b}')
                loss_m = criterion(max_pred, label)
                # print(f'loss_m: {loss_m}')
                loss = config.alpha * loss_b + (1-config.alpha) * loss_m
                
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
            ins_pred, bag_pred, _,_ = model(feats)
            # print(f'bag_pred : {bag_pred} bag pred softmax:{torch.softmax(bag_pred,dim=1)}  \n instance pred: {ins_pred}')
            max_pred,_ = torch.max(ins_pred, dim=0)
            # print(f'max_pred: {max_pred}  ')
            # print(f'max pred softmax: {torch.softmax(max_pred,dim=1)}')
            bag_pred = bag_pred.squeeze()
            loss_b = criterion(bag_pred, label)
            loss_m = criterion(max_pred, label)
            loss = config.alpha * loss_b + (1-config.alpha) * loss_m
            pred.extend([(config.alpha * torch.softmax(bag_pred,dim=0) + (1-config.alpha) * torch.softmax(max_pred,dim=0)).squeeze().cpu().numpy()])
            # print(f'prediction after sigmoid and alpha combination: {pred[-1]}')
            
            Loss += loss.item()
            # print('Testing bag [%d/%d] bag loss: %.4f' % (i, len(valid_df), loss.item()))
            wandb.log({"Step Test Loss": loss.item()})
   
    pred = np.array(pred)
    labels = np.array(labels)
    # print(f'Prediction : {pred}  size : {len(pred)}')
    # print(f'Labels array : {labels}  size : {labels.shape}')


    # print(f"Threshold: {thresholds}      Threshold optimal: {thresholds_optimal}")
    if args.num_classes == 3:
        binary_labels = label_binarize(labels, classes=[0, 1, 2])
    # auc_value, thresholds, thresholds_optimal = multi_label_roc(binary_labels, pred, config)
        auc_value = multi_label_roc(binary_labels, pred, config)
        plot = wandb.plot.roc_curve(labels, pred,labels=['NORMAL vs. BAS-SCC','BAS vs. NORMAL-SCC','SCC vs. NORMAL-BAS'])
        precision_recall = wandb.plot.pr_curve(labels, pred,labels=['NORMAL vs. BAS-SCC','BAS vs. NORMAL-SCC','SCC vs. NORMAL-BAS'])
    elif args.num_classes == 2:
        binary_labels = nn.functional.one_hot(torch.tensor(labels),args.num_classes).numpy()
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

def compute_roc(label, pred, n_classes):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(label[:, i], pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    return fpr, tpr, roc_auc


def optimal_thresh(fpr, tpr, thresholds, p = 0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]


def multi_label_roc(labels, predictions, config):
    # fprs = []
    # tprs = []
    # thresholds = []
    # thresholds_optimal = []
    aucs = []
    if len(predictions.shape)==1:
        predictions = predictions[:, None]
    for c in range(config.classes):
        label = labels[:, c]
        prediction = predictions[:, c]
        # fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
        # fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold,config)
        c_auc = roc_auc_score(label, prediction)
        aucs.append(c_auc)
        # thresholds.append(threshold)
        # thresholds_optimal.append(threshold_optimal)
    # return aucs, thresholds, thresholds_optimal
    return aucs




def plot_auc(pred,y,config):
    fpr,tpr,roc_auc = dict(),dict(),dict()

    for i in range(config.classes):
    
        fpr[i],tpr[i],_ = roc_curve(y[:,i], pred[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])


    fpr["micro"],tpr["micro"],_ = roc_curve(y.ravel() , pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(config.classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(config.classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= config.classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(figsize=[8.6,6.5])
    plt.plot(
    fpr["micro"],
    tpr["micro"],
    label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
    color="deeppink",
    linestyle=":",
    linewidth=4,
    )

    plt.plot(
    fpr["macro"],
    tpr["macro"],
    label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
    color="navy",
    linestyle=":",
    linewidth=4,
    )

    # colors = ["aqua", "darkorange", "cornflowerblue", "green", "red", "purple", "yellow", "blue", "black", "brown", "pink", "grey"]
    colors = ["aqua", "darkorange", "cornflowerblue"]
    if args.num_classes == 3:
        classesname = ["NORMAL vs BAS&SCC", "BAS vs NORMAL&SCC", "SCC vs NORMAL&BAS"]
    elif args.num_classes == 2:
        classesname = ["NORMAL", "BAS"]

    for i, color,name in zip(range(config.classes), colors, classesname):
        plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        lw=2,
        label="  ROC curve of class {0}     (area = {1:0.2f})".format(name, roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.legend(loc="lower right")
    plt.show()
    save_path = "/system/user/publicwork/yitaocai/Master_Thesis/auc/auc_plot"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    auc_plot = os.path.join(save_path, f"{args.num_classes}C"+"_"+f"seed{args.seed}"+"_"+ f"{args.dataset}"+"_"+f"n{args.n}"+".png")

    plt.savefig(auc_plot)



def build_dataloader(config,mode="train"):
    working_file = "/restricteddata/skincancer_kuk/Scanned_WSI/metadata_workingfile_label.csv"
    path = "/restricteddata/skincancer_kuk/tiles_20x/Features/ConvNext_20X"
    path_vig = "/restricteddata/skincancer_kuk/tiles_20x/Features/VIG_20X"
    path_van = "/restricteddata/skincancer_kuk/tiles_20x/Features/Van_20X"
    path_combined = "/system/user/publicwork/yitaocai/Master_Thesis/Integrated_CVV20X_Dataset"
    feats_files_cnt = sorted(glob.glob(os.path.join(path,"*", "*.csv"), recursive=True))
    feats_files_vig = sorted(glob.glob(os.path.join(path_vig,"*", "*.csv"), recursive=True))
    feats_files_van = sorted(glob.glob(os.path.join(path_van,"*", "*.csv"), recursive=True))
    feats_files_combined = sorted(glob.glob(os.path.join(path_combined, "*.csv"), recursive=True))

    if args.num_classes == 3:
      
        df  =  get_working_df(feats_files_cnt, working_file)
        df = shuffle(df).reset_index(drop=True)
        df_vig = get_working_df(feats_files_vig, working_file)
        # df_vig = shuffle(df_vig).reset_index(drop=True)
        df_van = get_working_df(feats_files_van, working_file)
        # df_van = shuffle(df_van).reset_index(drop=True)
        df_combined = get_working_df(feats_files_combined, working_file)
        # df_combined = shuffle(df_combined[df_combined["labels"]!=2]).reset_index(drop=True)


    elif args.num_classes == 2:
        df_combined = get_working_df(feats_files_combined, working_file)
        df_combined = shuffle(df_combined[df_combined["labels"]!=2]).reset_index(drop=True)
        df  =  get_working_df(feats_files_cnt, working_file)
        df = shuffle(df[df["labels"]!=2]).reset_index(drop=True)
        df_vig = get_working_df(feats_files_vig, working_file)
        df_vig = shuffle(df_vig[df_vig["labels"]!=2]).reset_index(drop=True)
        df_van = get_working_df(feats_files_van, working_file)
        df_van = shuffle(df_van[df_van["labels"]!=2]).reset_index(drop=True)
    
    # ls =df.labels.to_numpy()
    # # print(np.unique(ls, return_counts=True))
    # from collections import Counter
    # cw = Counter(ls)
    if mode == "train":
        if args.split == "determined":
            dft = df[df['sets']=='train']
            dft = shuffle(dft).reset_index(drop=True)
            dft_vig = df_vig[df_vig['sets']=='train']
            dft_vig = shuffle(dft_vig).reset_index(drop=True)
            dft_van = df_van[df_van['sets']=='train']
            dft_van = shuffle(dft_van).reset_index(drop=True)
            dft_combined = df_combined[df_combined['sets']=='train']
            dft_combined = shuffle(dft_combined).reset_index(drop=True)
            dfv = df[df['sets']=='val']
            dfv = shuffle(dfv).reset_index(drop=True)
            dfv_vig = df_vig[df_vig['sets']=='val']
            dfv_vig = shuffle(dfv_vig).reset_index(drop=True)
            dfv_van = df_van[df_van['sets']=='val']
            dfv_van = shuffle(dfv_van).reset_index(drop=True)
            dfv_combined = df_combined[df_combined['sets']=='val']
            dfv_combined = shuffle(dfv_combined).reset_index(drop=True)
        else:
            df = df[df["sets"] != "test"]
            df_vig = df_vig[df_vig["sets"] != "test"]
            df_van = df_van[df_van["sets"] != "test"]
            df_combined = df_combined[df_combined["sets"] != "test"]
            
            # sd = random.randint(0,99999)
            dft, dfv = train_test_split(df, test_size=0.1, random_state=config.sd, stratify=df['labels'])
            dft_vig, dfv_vig = train_test_split(df_vig, test_size=0.1, random_state=config.sd, stratify=df_vig['labels'])
            dft_vig,dfv_vig = shuffle(dft_vig).reset_index(drop=True), shuffle(dfv_vig).reset_index(drop=True)
            dft_van, dfv_van = train_test_split(df_van, test_size=0.1, random_state=config.sd, stratify=df_van['labels'])
            dft_van,dfv_van = shuffle(dft_van).reset_index(drop=True), shuffle(dfv_van).reset_index(drop=True)
            dft_combined, dfv_combined = train_test_split(df_combined, test_size=0.1, random_state=config.sd, stratify=df_combined['labels'])
            dft_combined,dfv_combined = shuffle(dft_combined).reset_index(drop=True), shuffle(dfv_combined).reset_index(drop=True)

        # df_test = df[df['sets']=='test']
        # # df_test = shuffle(df_test).reset_index(drop=True)
        # df_vig_test = df_vig[df_vig['sets']=='test']
        # # df_vig_test = shuffle(df_vig_test).reset_index(drop=True)
        # df_van_test = df_van[df_van['sets']=='test']
        # # df_van_test = shuffle(df_van_test).reset_index(drop=True)
        # df_combined_test = df_combined[df_combined['sets']=='test']
        # df_combined_test = shuffle(df_combined_test).reset_index(drop=True)
        if args.df == 'df_combined':
            train_df = dft_combined
            valid_df = dfv_combined

        elif args.df == 'df':

            train_df = dft
            valid_df = dfv
        
        elif args.df == 'df_vig':
            train_df = dft_vig
            valid_df = dfv_vig
        
        elif args.df == 'df_van':
            train_df = dft_van
            valid_df = dfv_van

        return train_df, valid_df
        
    elif mode == "test":
        if args.df == 'df_combined':
            test_df = df_combined[df_combined['sets']=='test']

        elif args.df == 'df':
            test_df = df[df['sets']=='test']

        elif args.df == 'df_vig':
            test_df = df_vig[df_vig['sets']=='test']

        elif args.df == 'df_van':
            test_df = df_van[df_van['sets']=='test']

        return test_df

def build_optimizer(model,config):
    if config.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(),
                              lr=config.lr, momentum=config.momentum)
    elif config.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=config.betas, weight_decay=config.weight_decay)
    elif config.optimizer == "Adamax":
        optimizer = torch.optim.Adamax(model.parameters(), lr=config.lr, betas=config.betas, weight_decay=config.weight_decay)

    elif config.optimizer == "ASGD":
        optimizer = torch.optim.ASGD(model.parameters(), lr=config.lr,  weight_decay=config.weight_decay)
    return optimizer

def build_scheduler(optimizer, config):
    
    if config.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs, config.minlr)
    elif config.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', verbose=True)
    return scheduler


def train(config=None):

    # with wandb.init(config):
    #     config = wandb.config
        # datalist = ['CNT','VIG','VAN','CNT_VIG','VIG_CNT','CNT_VAN','VIG_VAN','CNT_VIG_VAN','VIG_CNT_VAN','VAN_VIG_CNT','CNT_VAN_VIG']
        
        # config.dataset = datalist[8]
        # print(config.data)
        # config.update(config,allow_val_change=True)
        # config.in_size = 1000 if len(config.dataset) == 3 else (2000 if len(config.dataset) == 7 else 3000)
        # config.update(config,allow_val_change=True)
        train_df, valid_df = build_dataloader(config)

        print(f"the length of train_df is {len(train_df)} \n the length of valid_df is {len(valid_df)}")
        ins_classifier = InstanceClassifier(config).to(config.device)
        bag_classifier = AttDual(config).to(config.device)

        model = DSMILNet(ins_classifier, bag_classifier).to(config.device)
        wandb.watch(model, log_freq=100, log_graph=True)
        criterion = nn.CrossEntropyLoss()

        optimizer = build_optimizer(model, config)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs, 0.000005)
        scheduler = build_scheduler(optimizer, config)

        save_dir = "/system/user/publicwork/yitaocai/Master_Thesis/model"
        d = datetime.date.today().strftime("%m%d%Y")
        best_score = 0
        auc_list = []
       
 
        # b_acc = []
 
        for e in range(config.epochs):
            print("Epoch %d" % e)

            if args.weighted_sampling:
                train_loss = run_weighted_bags(model, train_df, optimizer, criterion, config)
            else:
                train_loss = run_bags(model, train_df, optimizer, criterion, config)


            valid_loss,pred,labels,pred_prob,auc = evaluate_bags(model, valid_df, criterion, config)
            auc_list.append(auc)

            if config.scheduler == "plateau":
                scheduler.step(valid_loss)
            else:
                scheduler.step()

                   
            accuracy=accuracy_score(labels, pred)
            f1_macro=f1_score(labels, pred, average='macro')
            f1_micro=f1_score(labels, pred, average='micro')

            balanced_acc = balanced_accuracy_score(labels, pred)
            
            # sample_weight = class_weight.compute_sample_weight(cw, labels)
            # weighted_balanced_acc = balanced_accuracy_score(labels, pred, sample_weight=sample_weight)
            mcc = matthews_corrcoef(labels, pred)
            cohen_kappa = cohen_kappa_score(labels, pred)
            current_score = (np.mean(auc) + accuracy+balanced_acc+f1_macro+mcc)/5
          

            # wandb.log({'AUC': np.mean(auc)})  
            wandb.log({'Train Loss': train_loss})
            wandb.log({'Val loss': valid_loss})
            wandb.log({'Accuracy': accuracy})
            wandb.log({'F1_macro': f1_macro})
            wandb.log({'F1_micro': f1_micro})
            wandb.log({'Balanced Accuracy': balanced_acc})
            # wandb.log({'Weighted Balanced Accuracy': weighted_balanced_acc})
            wandb.log({'MCC': mcc})
            wandb.log({'Cohen Kappa': cohen_kappa})

            if args.num_classes == 3:
                keys = ["NORMAL vs BAS&SCC","BAS vs NORMAL&SCC","SCC vs NORMAL&BAS"]
                wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                            y_true=labels, preds=pred,
                            class_names=["NORMAL","BAS","SCC"])})
                if  config.epochs - e == 1:
                    wandb.sklearn.plot_confusion_matrix(labels, pred, labels=["NORMAL","BAS","SCC"])
                    binary_labels = nn.functional.one_hot(torch.tensor(labels),args.num_classes).numpy()
                    # print(f"binary labels: {binary_labels}")
                    # print(f"prediction probability: {pred_prob}")
                    plot_auc(pred_prob,binary_labels,config)
            elif args.num_classes == 2:
                keys = ["NORMAL","BAS"]
                wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                            y_true=labels, preds=pred,
                            class_names=["NORMAL","BAS"])})
                if  config.epochs - e == 1:
                    wandb.sklearn.plot_confusion_matrix(labels, pred, labels=["NORMAL","BAS"])
                    binary_labels = nn.functional.one_hot(torch.tensor(labels),args.num_classes).numpy()
                    # print(f"binary labels: {binary_labels}")
                    # print(f"prediction probability: {pred_prob}")
                    plot_auc(pred_prob,binary_labels,config)
            if current_score >= best_score:
                best_score = current_score
                save_path = os.path.join(save_dir, f"{args.num_classes}C_{args.dataset}_n{args.n}_weighted{args.weighted_sampling}_{args.split}_{d}.pt")
                torch.save(model.state_dict(), save_path)
                print("Model saved to %s with current score %.4f" % (save_path, current_score))

        aucs = np.array(auc_list)
        xs = [ i for i in range(len(aucs)) ]
        ys = [aucs[:,c] for c in range(aucs.shape[1])]
        wandb.log({"AUC" : wandb.plot.line_series(  xs=xs, ys=ys, keys=keys,title="AUC",xname="Epochs")})
        return auc_list

def main():

    gc.collect()
    torch.cuda.empty_cache()

    
    config = {"grad_norm_clip": 1.02580285460309, 
            "alpha": 0.849435264741892,
            "in_size": args.in_size,
            "classes": args.num_classes,
            "out_size": args.num_classes,
            "dropout": 0.3,
            "lr": 0.0203373691328209,
            "weight_decay": 0.0001,
            "epochs": args.epochs,
            "device": torch.device(args.device),
            "optimizer": "SGD",
            "scheduler": "cosine",
            "betas": (0.5, 0.999),
            "momentum": 0.9,
            "minlr": 0.000005,
            "thresh_prob": 0,
            "dataset": args.dataset,
            "w0":args.w0, #44,#36,#48, #39, # 84,
            "w1":args.w1,#76,#72,# 20,#17,#16, #64,
            "w2":args.w2,#71,#21 # 96 # 93  #86
            "sd":random.randint(0,99999) #args.seed
            }

    if args.weighted_sampling:
        wandb.init(project=f"DSMIL_{args.dataset}_{args.num_classes}C_weighted", name=f"DSMILATT_seed{args.seed}_{args.split}_{args.dataset}", config=config,allow_val_change=True)
        save_path = os.path.join("/system/user/publicwork/yitaocai/Master_Thesis/auc/auc_file/weighted", f"auc_{args.num_classes}C_{args.split}_{args.seed}_{args.dataset}_n{args.n}_weighted.csv")
    
    else:
        wandb.init(project=f"DSMIL_{args.num_classes}C_R3", name=f"DSMILATT_seed{args.seed}_{args.split}_{args.dataset}",config=config,allow_val_change=True)
        save_path = os.path.join("/system/user/publicwork/yitaocai/Master_Thesis/auc/auc_file", f"auc_{args.num_classes}C_{args.split}_{args.seed}_{args.dataset}_n{args.n}.csv")

    wandb.config.update(args, allow_val_change=True)
    config = wandb.config
    set_seed(args.seed)
    auc_list = train(config)
    auc_df = pd.DataFrame(auc_list)

    auc_df.to_csv(save_path, index=False)

if __name__ == "__main__":
    main()
