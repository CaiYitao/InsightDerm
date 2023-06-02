import torch, gc
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import convnext_large, ConvNeXt_Large_Weights
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
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
from histolab.tiler import GridTiler
from histolab.slide import Slide
from histolab.masks import TissueMask
from tqdm import tqdm





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

def build_testloader(config):
    working_file = "/restricteddata/skincancer_kuk/Scanned_WSI/metadata_workingfile_label.csv"
    path = "/restricteddata/skincancer_kuk/tiles_20x/Features/ConvNext_20X"
    path_vig = "/restricteddata/skincancer_kuk/tiles_20x/Features/VIG_20X"
    path_van = "/restricteddata/skincancer_kuk/tiles_20x/Features/Van_20X"
    path_combined = "/system/user/publicwork/yitaocai/Master_Thesis/Integrated_CVV20X_Dataset"
    feats_files_cnt = sorted(glob.glob(os.path.join(path,"*", "*.csv"), recursive=True))
    feats_files_vig = sorted(glob.glob(os.path.join(path_vig,"*", "*.csv"), recursive=True))
    feats_files_van = sorted(glob.glob(os.path.join(path_van,"*", "*.csv"), recursive=True))
    feats_files_combined = sorted(glob.glob(os.path.join(path_combined, "*.csv"), recursive=True))

    if config.classes == 3:
      
        df  =  get_working_df(feats_files_cnt, working_file)
        # df = shuffle(df).reset_index(drop=True)
        df_vig = get_working_df(feats_files_vig, working_file)
        # df_vig = shuffle(df_vig).reset_index(drop=True)
        df_van = get_working_df(feats_files_van, working_file)
        # df_van = shuffle(df_van).reset_index(drop=True)
        df_combined = get_working_df(feats_files_combined, working_file)
        # df_combined = shuffle(df_combined[df_combined["labels"]!=2]).reset_index(drop=True)


    elif config.classes == 2:
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

    if config.df == 'df_combined':
        test_df = df_combined[df_combined['sets']=='test']

    elif config.df == 'df':
        test_df = df[df['sets']=='test']

    elif config.df == 'df_vig':
        test_df = df_vig[df_vig['sets']=='test']

    elif config.df == 'df_van':
        test_df = df_van[df_van['sets']=='test']

    return test_df



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
    feats = feats.to_numpy()[1:]
    label = feats_df.iloc[1]
    return feats, label



def get_test_bags(bags_list,df_test,config):
    test_bags = []
    bags= [os.path.basename(bag) for bag in bags_list]
    test_files = [os.path.basename(file).rstrip(".csv") for file in df_test["feats_files"].values]

    test_csv = []
    for i, (bag,label) in enumerate(zip(test_files,df_test["labels"].values)):
        if bag in bags:
            if label == 0:
                test_bags.append(os.path.join(config.bag_dir + "/0_normal/", bag))
            elif label == 1:
                test_bags.append(os.path.join(config.bag_dir + "/1_bas/",  bag))
            elif label == 2:
                test_bags.append(os.path.join(config.bag_dir + "/2_scc/",  bag))
            test_csv.append(df_test["feats_files"].values[i])

    return test_bags,test_csv



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

    aucs = []
    if len(predictions.shape)==1:
        predictions = predictions[:, None]
    for c in range(config.classes):
        label = labels[:, c]
        prediction = predictions[:, c]
        c_auc = roc_auc_score(label, prediction)
        aucs.append(c_auc)

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
    colors = ["purple", "darkorange", "green"]
    if config.classes == 3:
        classesname = ["NORMAL vs BAS&SCC", "BAS vs NORMAL&SCC", "SCC vs NORMAL&BAS"]
    elif config.classes == 2:
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
    d = datetime.date.today().strftime("%m%d%Y")
    # save_path = "/system/user/publicwork/yitaocai/Master_Thesis/auc/auc_plot"
    if not os.path.exists(config.AUC_save_path):
        os.makedirs(config.AUC_save_path)
    # auc_plot = os.path.join(config.AUC_save_path, f"topk{config.topk}_{config.num_heads}h_{config.num_layers}layers_{config.classes}C"+"_"+f"seed{config.seed}"+"_"+ f"{config.dataset}"+f"{d}"+".png")
    auc_plot = os.path.join(config.AUC_save_path, f"{config.classes}C"+"_"+f"seed{config.seed}"+"_"+ f"{config.dataset}"+f"{d}"+".png")

    plt.savefig(auc_plot)


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

    if config.classes == 3:
      
        df  =  get_working_df(feats_files_cnt, working_file)
        df = shuffle(df).reset_index(drop=True)
        df_vig = get_working_df(feats_files_vig, working_file)
        # df_vig = shuffle(df_vig).reset_index(drop=True)
        df_van = get_working_df(feats_files_van, working_file)
        # df_van = shuffle(df_van).reset_index(drop=True)
        df_combined = get_working_df(feats_files_combined, working_file)
        # df_combined = shuffle(df_combined[df_combined["labels"]!=2]).reset_index(drop=True)


    elif config.classes == 2:
        df_combined = get_working_df(feats_files_combined, working_file)
        df_combined = shuffle(df_combined[df_combined["labels"]!=2]).reset_index(drop=True)
        df  =  get_working_df(feats_files_cnt, working_file)
        df = shuffle(df[df["labels"]!=2]).reset_index(drop=True)
        df_vig = get_working_df(feats_files_vig, working_file)
        df_vig = shuffle(df_vig[df_vig["labels"]!=2]).reset_index(drop=True)
        df_van = get_working_df(feats_files_van, working_file)
        df_van = shuffle(df_van[df_van["labels"]!=2]).reset_index(drop=True)
    
    if mode == "train":
        if config.split == "determined":
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
            dft, dfv = train_test_split(df, test_size=0.1, random_state=config.seed, stratify=df['labels'])
            dft_vig, dfv_vig = train_test_split(df_vig, test_size=0.1, random_state=config.seed, stratify=df_vig['labels'])
            dft_vig,dfv_vig = shuffle(dft_vig).reset_index(drop=True), shuffle(dfv_vig).reset_index(drop=True)
            dft_van, dfv_van = train_test_split(df_van, test_size=0.1, random_state=config.seed, stratify=df_van['labels'])
            dft_van,dfv_van = shuffle(dft_van).reset_index(drop=True), shuffle(dfv_van).reset_index(drop=True)
            dft_combined, dfv_combined = train_test_split(df_combined, test_size=0.1, random_state=config.seed, stratify=df_combined['labels'])
            dft_combined,dfv_combined = shuffle(dft_combined).reset_index(drop=True), shuffle(dfv_combined).reset_index(drop=True)

        if config.df == 'df_combined':
            train_df = dft_combined
            valid_df = dfv_combined

        elif config.df == 'df':

            train_df = dft
            valid_df = dfv
        
        elif config.df == 'df_vig':
            train_df = dft_vig
            valid_df = dfv_vig
        
        elif config.df == 'df_van':
            train_df = dft_van
            valid_df = dfv_van

        return train_df, valid_df
        
    elif mode == "test":
        if config.df == 'df_combined':
            test_df = df_combined[df_combined['sets']=='test']

        elif config.df == 'df':
            test_df = df[df['sets']=='test']

        elif config.df == 'df_vig':
            test_df = df_vig[df_vig['sets']=='test']

        elif config.df == 'df_van':
            test_df = df_van[df_van['sets']=='test']

        return test_df
    


def get_pos(dir):
    import re
    if os.path.exists(dir):
        files = sorted(glob.glob(os.path.join(dir, '*.png'), recursive=True), key=lambda f: int(re.split("[_ -]",os.path.basename(f))[1]))
    else:
        raise ValueError("{} does not exist".format(dir))

    image_pos_list = []
    for file in files:
        basename = os.path.basename(file).rstrip(".png")
        image_pos = [re.split("[_ -]", basename)[1]] + re.split("[_ -]", basename)[3:7]

        image_pos = [int(x) for x in image_pos]
        image_pos_list.append(image_pos)
    return image_pos_list



def rename(path,format,df):
    import re
    mrxs_files = sorted(glob.glob(os.path.join(path, '*'+ format), recursive=True))
    data_files = sorted(glob.glob(os.path.join(path,'*/'), recursive=True))

    for mrxs, data in zip(mrxs_files, data_files):

        basename_mrxs = os.path.basename(mrxs)
        basename_data = os.path.basename(os.path.dirname(data))       
        filename_mrxs = "T" + re.split("T", basename_data)[1] + format
        
        try:
            image_name = df.loc[df["filename"] == filename_mrxs]["image_nr"].values[0]
            os.rename(mrxs, os.path.join(path, str(image_name) + format))
            os.rename(data, os.path.join(path, str(image_name)))
        except:
            continue





def get_tiles(slidepath, basepath, labels, serial_no, format = ".mrxs",tile_size = 512 ):


    slides_files = sorted(glob.glob(os.path.join(slidepath, '*'+ format), recursive=True))
    # print(slides_files[0:5])

    slides = [os.path.basename(f).rstrip(format) for f in slides_files]

    PROCESS_NORMAL_PATH = os.path.join(basepath,'0_normal')
    PROCESS_BAS_PATH = os.path.join(basepath,'1_bas')
    PROCESS_SCC_PATH = os.path.join(basepath,'2_scc')
 
    if not os.path.exists(PROCESS_NORMAL_PATH) or not os.path.exists(PROCESS_BAS_PATH) or not os.path.exists(PROCESS_SCC_PATH):
        os.makedirs(PROCESS_NORMAL_PATH)
        os.makedirs(PROCESS_BAS_PATH)
        os.makedirs(PROCESS_SCC_PATH)

    print(f"start processing {len(slides)} slides")
    i=0
    pbar = tqdm(enumerate(slides), total=len(slides))
    for _ , filename in pbar:


        print(f"filename is {filename}")
        print(f"filesname is in serial_no {filename in serial_no}")
        if filename in serial_no:
            i+=1
            # pbar.set_description(f"Start processing slide {i} {filename} ")
            temp_slidepath = os.path.join(slidepath, filename + format)  

            
            idx = serial_no.index(filename)
            if labels[idx] == 0:

                slide = Slide(temp_slidepath, processed_path = PROCESS_NORMAL_PATH)
            elif labels[idx] == 1:
  
                slide = Slide(temp_slidepath, processed_path = PROCESS_BAS_PATH)
            elif labels[idx] == 2:

                slide = Slide(temp_slidepath, processed_path = PROCESS_SCC_PATH)
            

            grid_tiles_extractor = GridTiler(
                tile_size=(tile_size, tile_size),
                level=1,
                check_tissue=True, # default
                tissue_percent=0.95,
                pixel_overlap=0, # default
                prefix = filename +'/', # save tiles in the "grid" subdirectory of slide's processed_path
                suffix=".png" 
                )
            
            grid_tiles_extractor.locate_tiles(
                slide=slide,
                extraction_mask = TissueMask(),
                scale_factor=64,
                alpha=64,
                outline="#046C4C",
            )
            grid_tiles_extractor.extract(slide)
            
           
            pbar.set_description(f"The tiling of the slide {i}  {filename}  is done")
        else:
            continue
    print(f"{i} slides are processed")



def tile_a_slide(slidepath, basepath, labels, serial_no, format = ".mrxs",tile_size = 512 ):

    filename = os.path.basename(slidepath).rstrip(format) 

    PROCESS_NORMAL_PATH = os.path.join(basepath,'0_normal')
    PROCESS_BAS_PATH = os.path.join(basepath,'1_bas')
    PROCESS_SCC_PATH = os.path.join(basepath,'2_scc')
 
    if not os.path.exists(PROCESS_NORMAL_PATH) or not os.path.exists(PROCESS_BAS_PATH) or not os.path.exists(PROCESS_SCC_PATH):
        os.makedirs(PROCESS_NORMAL_PATH)
        os.makedirs(PROCESS_BAS_PATH)
        os.makedirs(PROCESS_SCC_PATH)

    print(f"filename is {filename}")
    print(f"filesname is in serial_no {filename in serial_no}")
    if filename in serial_no:
       
        temp_slidepath = slidepath                                         
        
        idx = serial_no.index(filename)
        if labels[idx] == 0:

            slide = Slide(temp_slidepath, processed_path = PROCESS_NORMAL_PATH)
        elif labels[idx] == 1:

            slide = Slide(temp_slidepath, processed_path = PROCESS_BAS_PATH)
        elif labels[idx] == 2:

            slide = Slide(temp_slidepath, processed_path = PROCESS_SCC_PATH)
        

        grid_tiles_extractor = GridTiler(
            tile_size=(tile_size, tile_size),
            level=1,
            check_tissue=True, # default
            tissue_percent=0.95,
            pixel_overlap=0, # default
            prefix = filename +'/', # save tiles in the "grid" subdirectory of slide's processed_path
            suffix=".png" 
            )
        
        grid_tiles_extractor.locate_tiles(
            slide=slide,
            extraction_mask = TissueMask(),
            scale_factor=64,
            alpha=64,
            outline="#046C4C",
        )
        grid_tiles_extractor.extract(slide)
        
    print(f"the slide {filename} is processed")
        
       