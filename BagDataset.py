

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms.functional as VF
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode

import sys, argparse, os, glob, copy
import pandas as pd
import numpy as np
from PIL import Image
from collections import OrderedDict
# from sklearn.utils import shuffle
from torch.utils.data import Dataset
import re



def get_pos(data_dir):

    os.path.exists(data_dir)
    files = sorted(glob.glob(os.path.join(data_dir, '*.png'), recursive=True), key=lambda f: int(re.split("[_ -]",os.path.basename(f))[1]))
    # print("Found {} position files, the files are {}".format(len(files),files) )

    image_pos_list = []
    for file in files:
        basename = os.path.basename(file).rstrip(".png")
        image_pos = [re.split("[_ -]", basename)[1]] + re.split("[_ -]", basename)[3:7]

        image_pos = [int(x) for x in image_pos]
        image_pos_list.append(image_pos)

    return image_pos_list, files


class BagDataset(Dataset):
    def __init__(self, data_dir: str, label_file, transforms=None):
        '''
        data_dir:      directory to find files
        label_file:    path to label_file
        transforms:    which transformations should be used for the data set
     
        '''
        super(BagDataset).__init__()
        self.data_dir=data_dir
        self.transforms = transforms
        # self.files = sorted(glob.glob(os.path.join(data_dir, '*.png'), recursive=True), key=lambda f: int(re.split("[_ -]",os.path.basename(f))[1]))
        # self.files = sorted(glob.glob(os.path.join(data_dir, '*.png'), recursive=True), key=lambda f: int(re.split("[_ -]",os.path.basename(f))[1]))
        # print("Found {} tiles files, the files are {}".format(len(self.files),self.files) )
        #extract tile position from image name
        self.tile_pos,self.files = get_pos(data_dir)
        #extract bag name from slide name/image folder name
        self.bag_name = os.path.basename(data_dir)

        #read label file
        metadata = pd.read_csv(label_file)
        #transform word label into integer labels and add to the dataframe
        #extract bag label
        self.bag_label = metadata.loc[metadata["image_nr"] == self.bag_name, 'label'].values[0]
 
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filepath = self.files[idx]
        tile_pos = self.tile_pos[idx]
        image = Image.open(filepath)
        image = image.convert('RGB')
                    
        if self.transforms:
  
            image = self.transforms(image)      
        # mean, std = image.mean([0,1]), image.std([0,1])
        # image = transforms.Normalize(mean, std)(image)
        # return dict(tile=image, tile_pos=tile_pos, bag_label=self.bag_label[idx])
        return dict(tile=image, tile_pos=tile_pos, bag_label=self.bag_label)


