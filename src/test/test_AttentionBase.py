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
from base_model import AttentionBase
from utils import get_feats_for_dataset,build_testloader,get_working_df,get_feats_df,get_bag_feats,compute_roc,multi_label_roc,plot_auc,build_optimizer,build_scheduler,build_dataloader


parser = argparse.ArgumentParser(description='AttentionBase Training')
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
parser.add_argument('--model_path', default="model.pt", type=str, help='path to model to load')


args = parser.parse_args()



def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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

            bag_pred,_ = model(feats)
            # print(f'bag_pred : {bag_pred}  \n instance pred: {ins_pred}')
    
            # print(f'max_pred: {max_pred}    label: {label}')
            bag_pred = bag_pred.squeeze()
            loss = criterion(bag_pred, label)
  
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
            bag_pred,_ = model(feats)
            # print(f'bag_pred : {bag_pred} bag pred softmax:{torch.softmax(bag_pred,dim=1)}  \n instance pred: {ins_pred}')

            # print(f'max pred softmax: {torch.softmax(max_pred,dim=1)}')
            bag_pred = bag_pred.squeeze()
            loss = criterion(bag_pred, label)

            pred.extend([torch.softmax(bag_pred,dim=0).squeeze().cpu().numpy()])
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
        # print(f'binary labels: {binary_labels}  size: {binary_labels.shape} \n pred: {pred}  size: {pred.shape}')
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


def test(config,path):

    test_df = build_testloader(config)
    model = AttentionBase(config).to(config.device)
    model.load_state_dict(torch.load(path))
    criterion = nn.CrossEntropyLoss()

    optimizer = build_optimizer(model, config)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs, 0.000005)
    # scheduler = build_scheduler(optimizer, config)
    model.eval()
    Loss = 0
    pred = []
    labels = []
    d = datetime.date.today().strftime("%m%d%Y")
    with torch.no_grad():
        for i in range(len(test_df)):
            feats, label = get_bag_feats(test_df.iloc[i])
            labels.extend([label])
            feats = torch.from_numpy(feats).float().to(config.device)
            feats = get_feats_for_dataset(feats, config.dataset)
            
            label = torch.tensor(label).long().to(config.device)
            bag_pred,_ = model(feats)
        # print(f'bag_pred : {bag_pred} bag pred softmax:{torch.softmax(bag_pred,dim=1)}  \n instance pred: {ins_pred}')

        # print(f'max pred softmax: {torch.softmax(max_pred,dim=1)}')
            bag_pred = bag_pred.squeeze()
            loss = criterion(bag_pred, label)

            pred.extend([torch.softmax(bag_pred,dim=0).squeeze().cpu().numpy()])
            
            Loss += loss.item()
            # print('Testing bag [%d/%d] bag loss: %.4f' % (i, len(valid_df), loss.item()))
            wandb.log({"Step Test Loss": loss.item()})

    pred_prob = np.array(pred)
    pred = np.argmax(pred, axis=1)
    labels = np.array(labels)

    # print(f'Prediction : {pred}  size : {len(pred)}')
    # print(f'Labels array : {labels}  size : {labels.shape}')

    if args.num_classes == 3:
        # binary_labels = label_binarize(labels, classes=[0, 1, 2])
        binary_labels = nn.functional.one_hot(torch.tensor(labels),args.num_classes).numpy()

        # auc_value = multi_label_roc(binary_labels, pred, config)
        plot = wandb.plot.roc_curve(labels, pred_prob,labels=['NORMAL vs. BAS-SCC','BAS vs. NORMAL-SCC','SCC vs. NORMAL-BAS'])
        precision_recall = wandb.plot.pr_curve(labels, pred_prob,labels=['NORMAL vs. BAS-SCC','BAS vs. NORMAL-SCC','SCC vs. NORMAL-BAS'])
        keys = ["NORMAL vs BAS&SCC","BAS vs NORMAL&SCC","SCC vs NORMAL&BAS"]
        
        wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                    y_true=labels, preds=pred,
                    class_names=["NORMAL","BAS","SCC"])})
 
        wandb.sklearn.plot_confusion_matrix(labels, pred, labels=["NORMAL","BAS","SCC"])

        # print(f"binary labels: {binary_labels}")
        # print(f"prediction probability: {pred_prob}")
        plot_confusion_matrix(labels,pred,config)
        plot_auc(pred_prob,binary_labels,config)
    elif args.num_classes == 2:
        binary_labels = nn.functional.one_hot(torch.tensor(labels),args.num_classes).numpy()
        # auc_value = multi_label_roc(binary_labels, pred, config)
        plot = wandb.plot.roc_curve(labels, pred_prob,labels=['NORMAL','BAS'])
        precision_recall = wandb.plot.pr_curve(labels, pred_prob,labels=['NORMAL','BAS'])
        keys = ["NORMAL","BAS"]
        
        wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                    y_true=labels, preds=pred,
                    class_names=["NORMAL","BAS"])})

        wandb.sklearn.plot_confusion_matrix(labels, pred, labels=["NORMAL","BAS"])
        # print(f"binary labels: {binary_labels}")
        # print(f"prediction probability: {pred_prob}")
        plot_confusion_matrix(labels,pred,config)
        plot_auc(pred_prob,binary_labels,config)
        

    wandb.log({"roc": plot})
    # wandb.sklearn.plot_roc(labels, pred, labels=['NORMAL vs. BAS-SCC','BAS vs. NORMAL-SCC','SCC vs. NORMAL-BAS'])
    wandb.log({"pr": precision_recall})
    # wandb.sklearn.plot_precision_recall(labels, pred, labels=['NORMAL vs. BAS-SCC','BAS vs. NORMAL-SCC','SCC vs. NORMAL-BAS'])
    # pred = np.where(pred >= thresholds_optimal, 1, 0)
   

    accuracy=accuracy_score(labels, pred)
    f1_macro=f1_score(labels, pred, average='macro')
    f1_micro=f1_score(labels, pred, average='micro')

    balanced_acc = balanced_accuracy_score(labels, pred)
    mcc = matthews_corrcoef(labels, pred)
    cohen_kappa = cohen_kappa_score(labels, pred)


    wandb.log({'Accuracy': accuracy})
    wandb.log({'F1_macro': f1_macro})
    wandb.log({'F1_micro': f1_micro})
    wandb.log({'Balanced Accuracy': balanced_acc})
    # wandb.log({'Weighted Balanced Accuracy': weighted_balanced_acc})
    wandb.log({'MCC': mcc})
    wandb.log({'Cohen Kappa': cohen_kappa})

def plot_confusion_matrix(label,pred,config):
    import plotly.figure_factory as ff
    from sklearn.metrics import confusion_matrix


    if config.classes == 3:
        z = confusion_matrix(label,pred,labels=[1,0,2])

    # invert z idx values
        z = z[::-1]
        x = ['BAS', 'Normal', 'SCC']
        y =  x[::-1].copy() # invert idx values of x
    elif config.classes == 2:
        z = confusion_matrix(label,pred,labels=[1,0])

    # invert z idx values
        z = z[::-1]
        x = ['BAS', 'Normal']
        y =  x[::-1].copy()

    # change each element of z to type string for annotations
    z_text = [[str(y) for y in x] for x in z]

    # set up figure 
    fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='Plasma')

    # add title
    fig.update_layout(title_text='<i><b>Confusion matrix</b></i>',
                    #xaxis = dict(title='x'),
                    #yaxis = dict(title='x')
                    )

    # add custom xaxis title
    fig.add_annotation(dict(font=dict(color="black",size=26),
                            x=0.5,
                            y=-0.15,
                            showarrow=False,
                            text="Predicted value",
                            xref="paper",
                            yref="paper"))

    # add custom yaxis title
    fig.add_annotation(dict(font=dict(color="black",size=26),
                            x=-0.2,
                            y=0.5,
                            showarrow=False,
                            text="Real value",
                            textangle=-90,
                            xref="paper",
                            yref="paper"))

    # adjust margins to make room for yaxis title
    fig.update_layout(margin=dict(t=50, l=150))

    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].font.size = 22

    # add colorbar
    fig['data'][0]['showscale'] = True
    cfg = {
  'toImageButtonOptions': {
    'format': 'pdf', # one of png, svg, jpeg, webp
    'filename': f'{config.classes}C_confusion_matrix.pdf',
    'height': None,
    'width': None,
    'scale':6 # Multiply title/legend/axis/canvas sizes by this factor
  }
}
    fig.show(config =cfg)
    # fig.write_image(config.save_path + 'confusion_matrix.png')
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)
    
    import plotly.io as pio
    pio.write_image(fig, os.path.join(config.save_path,f'{config.classes}C_confusion_matrix.pdf'),scale=6)
    # fig.write_image(os.path.join(config.save_path,'confusion_matrix.png'))
    # fig.write_image(os.path.join(config.save_path,f'{config.classes}C_confusion_matrix.pdf'),format='pdf',scale=6)
    

def main():

    gc.collect()
    torch.cuda.empty_cache()

    
    config = {"grad_norm_clip": 1.02580285460309, 
            "alpha": 0.849435264741892,
            "in_size": args.in_size,
            "classes": args.num_classes,
            "out_size": args.num_classes,
            "dropout": 0.496501578372041,
            "lr": 0.23020001673290413,
            "weight_decay": 0.023904432570835858,
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
            "sd":random.randint(0,99999), #args.seed
            "AUC_save_path": "/system/user/publicwork/yitaocai/Master_Thesis/auc_test/AttentionBase",
            "save_path": "/system/user/publicwork/yitaocai/Master_Thesis/confusionmatrix/AttentionBase",
            "weighted_sampling": "Weighted_Sampling" if args.weighted_sampling else "_"

            }

    set_seed(args.seed)
    wandb.init(project=f"AttentionBase_{args.num_classes}C_TEST", name=f"AttentionBase_seed{args.seed}_{args.dataset}_{os.path.basename(args.model_path)}",config=config,allow_val_change=True)

    wandb.config.update(args, allow_val_change=True)
    config = wandb.config
    set_seed(args.seed)
    test(config,args.model_path)


if __name__ == "__main__":
    main()