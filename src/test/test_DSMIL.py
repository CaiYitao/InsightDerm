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
import plotly
# import seaborn as sns
import plotly.figure_factory as ff
# from train_DSMIL import evaluate_bags,set_seed,get_working_df, get_feats_df, get_bag_feats, get_feats_for_dataset, multi_label_roc, plot_auc, build_dataloader, build_optimizer, build_scheduler


parser = argparse.ArgumentParser(description='DSMIL Test')
parser.add_argument('--seed', default=0, type=int, help='seed')
parser.add_argument('--device', default='cuda:0', type=str, help='device')
parser.add_argument('--dataset', default='CNT', type=str, help='features dataset name')
parser.add_argument('--num_classes', default=3, type=int, help='number of classes')
parser.add_argument('--df', default="df_combined", type=str, help='dataframe to use for training and validation, for example: df_combined, df, df_vig,df_van')
parser.add_argument('--model_path', default="model.pt", type=str, help='path to model to load')
parser.add_argument('--in_size', default=1000, type=int, help='input size of the model')

args = parser.parse_args()

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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




def plot_auc(pred,y,config,path):
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
    save_path = "/system/user/publicwork/yitaocai/Master_Thesis/auc_test/auc_plot"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    auc_plot = os.path.join(save_path, f"{args.num_classes}C"+"_"+ f"{args.dataset}"+"_"+f"{os.path.basename(path)}"+".pdf")

    plt.savefig(auc_plot)



def build_testloader():
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
        # df = shuffle(df).reset_index(drop=True)
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


def test(config,path):

    test_df = build_testloader()
    ins_classifier = InstanceClassifier(config).to(config.device)
    bag_classifier = AttDual(config).to(config.device)
    model = DSMILNet(ins_classifier, bag_classifier).to(config.device)
    model.load_state_dict(torch.load(path))
    criterion = nn.CrossEntropyLoss()

    model.eval()
    Loss = 0
    pred = []
    labels = []
    with torch.no_grad():
        for i in range(len(test_df)):
            feats, label = get_bag_feats(test_df.iloc[i])
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
            # print(f'prediction after softmax and alpha combination: {pred[-1]}')
            
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
        plot_auc(pred_prob,binary_labels,config,path)
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
        plot_auc(pred_prob,binary_labels,config,path)
        

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
    config = {"grad_norm_clip": 1.02580285460309, 
        "alpha": 0.849435264741892,
        "in_size": args.in_size,
        "classes": args.num_classes,
        "out_size": args.num_classes,
        "dropout": 0.3,
        "lr": 0.0203373691328209,
        "weight_decay": 0.0001,
        # "epochs": args.epochs,
        "device": torch.device(args.device),
        "optimizer": "SGD",
        "scheduler": "cosine",
        "betas": (0.5, 0.999),
        "momentum": 0.9,
        "minlr": 0.000005,
        "thresh_prob": 0,
        "dataset": args.dataset,
        "save_path": "/system/user/publicwork/yitaocai/Master_Thesis/auc_test"
   
        }


    set_seed(args.seed)
    wandb.init(project=f"DSMIL_{args.num_classes}C_TEST", name=f"DSMILATT_seed{args.seed}_{args.dataset}_{os.path.basename(args.model_path)}",config=config,allow_val_change=True)

    wandb.config.update(args, allow_val_change=True)
    config = wandb.config
    test(config,args.model_path)
    # path_list = sorted(glob.glob(os.path.join(args.model_path, '*.pt'),recursive=True))
    # print(f"The number of models: {len(path_list)}")
    # i = 0
    # for path in path_list:
    #     i+=1
    #     wandb.init(project=f"DSMIL_{args.num_classes}C_TEST", name=f"DSMILATT_seed{args.seed}_{args.dataset}_{os.path.basename(path)}",config=config,allow_val_change=True)

    #     wandb.config.update(args, allow_val_change=True)
    #     config = wandb.config
    #     test(config,path)

if __name__ == '__main__':
    main()
