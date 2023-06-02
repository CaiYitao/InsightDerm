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
from utils import get_feats_for_dataset,build_testloader,get_working_df,get_feats_df,get_bag_feats,compute_roc,multi_label_roc,plot_auc,build_optimizer,build_scheduler,build_dataloader
# from base_model import CF_Transformer,TopkCFMultiHeadAttention
from Model_1 import CF_Transformer,TopkCFMultiHeadAttention
from run import run_bags,run_weighted_bags,evaluate_bags



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
parser.add_argument('--num_heads', default=3, type=int, help='number of heads for multihead attention')
parser.add_argument('--num_layers', default=1, type=int, help='number of layers')
parser.add_argument('--classification', default="mean", type=str, help='classification method for topk heads, mean or maxpool1d or LPPool1d or avgpool1d')
parser.add_argument('--topk', default=5, type=int, help='the number of topk')
parser.add_argument('--head_dim', default=512, type=int, help='the dimension of each head')
parser.add_argument('--embed_x_dim', default=525, type=int, help='the dimension of the embedded x')
parser.add_argument('--head_proj', default="linear_diag", type=str, help='the final projection of head output, linear_diag or conv1d')
parser.add_argument('--critical_features_from', default="original", type=str, help='the input of critical features, original or embedding')



args = parser.parse_args()



def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)




def train(config):

        train_df, valid_df = build_dataloader(config)

        print(f"the length of train_df is {len(train_df)} \n the length of valid_df is {len(valid_df)}")
        if config.topkhead_classification:
            model = TopkCFMultiHeadAttention(config).to(config.device)
        else:
            model = CF_Transformer(config).to(config.device)
    

        wandb.watch(model, log_freq=100, log_graph=True)
        criterion = nn.CrossEntropyLoss()

        optimizer = build_optimizer(model, config)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs, 0.000005)
        scheduler = build_scheduler(optimizer, config)


        d = datetime.date.today().strftime("%m%d%Y")
        best_score = 0
        auc_list = []
 
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
                if not os.path.exists(config.model_save_dir):
                    os.makedirs(config.model_save_dir)
                save_path = os.path.join(config.model_save_dir, f"Topk{args.topk}_{args.num_heads}heads_{args.head_proj}_CFfrom{args.critical_features_from}_{args.num_classes}C_{args.dataset}_weighted{args.weighted_sampling}_{args.split}_{d}.pt")
                torch.save(model.state_dict(), save_path)
                print("Model saved to %s with current score %.4f" % (save_path, current_score))

        aucs = np.array(auc_list)
        xs = [ i for i in range(len(aucs)) ]
        ys = [aucs[:,c] for c in range(aucs.shape[1])]
        wandb.log({"AUC" : wandb.plot.line_series(xs=xs, ys=ys, keys=keys,title="AUC",xname="Epochs")})
        return auc_list
def main():

    gc.collect()
    torch.cuda.empty_cache()

    
    config = {"grad_norm_clip": 1.02, # 1.02580285460309, 
            # "alpha": 0.849435264741892,
            "alpha": 0.2663,
            "batch_size": args.batch_size,
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
            "sd":args.seed, #args.seed
            "num_heads":args.num_heads,
            "topk":args.topk,
            "seed":args.seed,
            "critical_features_from":args.critical_features_from,  # "original" or "embedding" or"index_add"
            "embed_module": "Sparse", # "Sparse" or "Dense"
            # "num_layers":args.num_layers,
            # "add_mlp":True,
            "topkhead_classification":True,
            'head_proj': args.head_proj,
            'head_dim': args.head_dim,
            'embed_x_dim':args.embed_x_dim,
            "AUC_save_path": "/system/user/publicwork/yitaocai/Master_Thesis/auc/auc_plot_topk_multihead",
            # "projection":True,
            "classification":"mean", #"mean" ,"avgpool1d" ,"LPPool1d" ,"maxpool1d" 
            "model_save_dir":f"/system/user/publicwork/yitaocai/Master_Thesis/model/topk_multihead_{args.num_classes}C",
  
            }

    if args.weighted_sampling:
        wandb.init(project=f"TopkCFMultiHead_{args.num_classes}C_weighted", name=f"Topk{args.topk}MultiHead_{args.num_heads}heads_seed{args.seed}_{args.split}_{args.dataset}", config=config,allow_val_change=True)
        # save_path = os.path.join("/system/user/publicwork/yitaocai/Master_Thesis/auc/auc_file/weighted", f"auc_{args.num_classes}C_{args.split}_{args.seed}_{args.dataset}_n{args.n}_weighted.csv")
    
    else:
        wandb.init(project=f"TopkCFMultiHead_{args.num_classes}C", name=f"Topk{args.topk}MultiHead_{args.num_heads}heads_seed{args.seed}_{args.split}_{args.head_dim}",config=config,allow_val_change=True)
        # save_path = os.path.join("/system/user/publicwork/yitaocai/Master_Thesis/auc/auc_file", f"auc_{args.num_classes}C_{args.split}_{args.seed}_{args.dataset}_n{args.n}.csv")

    wandb.config.update(args, allow_val_change=True)
    config = wandb.config
    set_seed(args.seed)
    auc_list = train(config)
    # auc_df = pd.DataFrame(auc_list)

    # auc_df.to_csv(save_path, index=False)

if __name__ == "__main__":
    main()