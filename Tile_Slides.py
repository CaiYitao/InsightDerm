import torch
from PIL import Image
import glob
import csv
import random
import numpy as np
from histolab.tiler import GridTiler
from histolab.slide import Slide
from histolab.masks import TissueMask
import os
import pandas as pd
from tqdm import tqdm
from utils import tile_a_slide



def main():
    data= pd.read_csv('PATH TO DATA')

    diagnosis = data['diagnosis'].values
    labels = data['label'].values
    serial_no = list(data['image_nr'].values)
    # serial_no = list(data['slidename'].values)
    filenames =[ fn.rstrip('.mrxs') for fn in data['filename'].values]

    basepath = 'PATH TO TILES'

    WSI_PATH = 'PATH TO WSI'
    TILES_PATH = 'TILES PATH'
    all_mrxs_slides = sorted(glob.glob(os.path.join(WSI_PATH, '*'+ '.mrxs'), recursive=True))
    all_svs_slides = sorted(glob.glob(os.path.join(WSI_PATH, '*'+ '.svs'), recursive=True))
    processed_slides = sorted(glob.glob(os.path.join(TILES_PATH, "*",'*/'), recursive=True))

    processed_slides_list = [os.path.basename(os.path.dirname(slide))for slide in processed_slides]
    # print(processed_slides_list)
    mrxs_slide_list = all_mrxs_slides.copy()
    svs_slide_list = all_svs_slides.copy()
    for slide in all_mrxs_slides:
        basename = os.path.basename(slide).rstrip('.mrxs')

        if basename in processed_slides_list:
            mrxs_slide_list.remove(slide)
    for slide in all_svs_slides:
        basename = os.path.basename(slide).rstrip('.svs')

        if basename in processed_slides_list:
            svs_slide_list.remove(slide)
    
    pbar = tqdm(range(len(mrxs_slide_list)),total=len(mrxs_slide_list),desc="Processing slides")
    for i in pbar:
        tile_a_slide(mrxs_slide_list[i], basepath, labels, serial_no, format = ".mrxs",tile_size = 512 )


if __name__ == '__main__':
    main()
