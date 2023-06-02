import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms.functional as VF
from torchvision import transforms
from torchvision.models import convnext_large, ConvNeXt_Large_Weights
import sys, argparse, os, glob
import pandas as pd
import numpy as np
from PIL import Image
from collections import OrderedDict
from skimage import exposure, io, img_as_ubyte, transform
from sklearn.utils import shuffle
import warnings
from base_model import InstanceClassifier, AttDual, DSMILNet
from BagDataset import BagDataset,get_pos
from tqdm import tqdm
from glob import glob
from utils import get_feats_for_dataset,build_testloader,get_working_df,get_feats_df,get_bag_feats,compute_roc,multi_label_roc,plot_auc,build_optimizer,build_scheduler,build_dataloader,get_test_bags
from Model_1 import CF_Transformer,TopkCFMultiHeadAttention
from run import run_bags,run_weighted_bags,evaluate_bags
import cv2
import pandas as pd

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
parser.add_argument('--model_path', default="model.pth", type=str, help='the path of the model')
parser.add_argument('--model', default="", type=str, help='the model to use')

args = parser.parse_args()


def compute_attn_map(bags_list,label_file,df_test, save_path,config):

    # load model for computing attention map
    model = TopkCFMultiHeadAttention(config).to(device=config.device)
    model.load_state_dict(torch.load(config.model_path))
    #load ConvNeXt model for computing features
    ConvNeXt = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)
    ConvNeXt_model = ConvNeXt.eval()
    ConvNeXt_model.to(config.device)

    # tiles image processing
    img_transform = transforms.Compose([transforms.Resize(224),transforms.ToTensor()])

    model.eval()
    # print(f"the length of bags_list is {len(bags_list)}, the length of df_test is {len(df_test)}")
    # get test slides list
    test_bags,test_csv = get_test_bags(bags_list,df_test,config)
    # print(f"the length of test_bags is {len(test_bags)}")

    for b in test_bags:
        if b == "/restricteddata/skincancer_kuk/tiles_20x/Tiles/0_normal/ICNNBCC00044":
            test_bags.remove(b)

    # colors = [np.random.choice(range(256), size=3) for i in range(config.classes)]
    attn_color = np.array([255,69,0]) / 255
    topk_color = (126,255,0)
    # print(f"get colors: {colors}  the shape is {np.array(colors).shape}")
    bar = tqdm(range(len(test_bags)))

    # define the final output size of tiles of the attention map
    scale = 56
    s = 224 // scale

    if config.classes == 3:
        class_name = ["Normal","BAS","SCC"]
    elif config.classes == 2:
        class_name = ["Normal","BAS"]

    for i in bar:
        feats = []
        pos_list = []
        tiles = []

        print(f" data_dir is {test_bags[i]}")
        bag_dataset = BagDataset(data_dir=test_bags[i], label_file=label_file, transforms=img_transform)
        # print(f"the length of bag_dataset is {len(bag_dataset)}")
        dataloader = DataLoader(bag_dataset, batch_size=1, shuffle=False)
        # print(f"the length of dataloader is {len(dataloader)}")

        slide_name = test_bags[i].split(os.sep)[-1]

        with torch.no_grad():
            pbar = tqdm(enumerate(dataloader), total=len(dataloader))
            for iter, batch in pbar:
                tile = batch['tile'].float().to(config.device)
                feat = ConvNeXt_model(tile)
                feats.append(feat.squeeze())
                tile = tile.squeeze().cpu().numpy()
                tile = np.transpose(tile,(2,1,0))
            
                tile = transform.resize(tile, (tile.shape[0]// s, tile.shape[1]//s), anti_aliasing=True)
            
                tiles.append(tile)
                tile_pos = torch.cat(batch['tile_pos']).cpu().numpy()
                if iter == 0:
                    label = batch['bag_label'].cpu().numpy()
                pos_list.append(tile_pos)
    
            # _ , label = get_bag_feats(df_test.iloc[i])
            
            # labels.extend([label])
            pos_arr = np.asarray(pos_list)
            tiles = np.asarray(tiles)

            feats = torch.stack(feats)

            bag_pred, ins_pred, A, idx = model(feats)
            # print(f"Attention: {A} and the shape is {A.shape}")
            topk_idx = idx[:,1:].squeeze().cpu().numpy()
            # print(f"topk_idx: {topk_idx}")
            # print(f'bag_pred : {bag_pred} bag pred softmax:{torch.softmax(bag_pred,dim=1)}  \n instance pred: {ins_pred}')
            max_pred,_ = torch.max(ins_pred, dim=0)
            # print(f'max_pred: {max_pred}  ')
            # print(f'max pred softmax: {torch.softmax(max_pred,dim=1)}')
            bag_pred = bag_pred.squeeze()
            prediction_p = (config.alpha * torch.softmax(bag_pred,dim=0) + (1-config.alpha) * torch.softmax(max_pred,dim=0)).squeeze().cpu().numpy()
            
            prediction = np.argmax(prediction_p)
            print(f"prediction_p: {prediction_p}  prediction: {prediction}")
            img_save_path = os.path.join(save_path, slide_name+'/')

            if not os.path.exists(img_save_path):
                os.makedirs(img_save_path)                         # num_pos_classes = 0
            for c in range(config.classes):
                if c == prediction:
                    print(f"The slide {os.path.basename(test_bags[i])} is predicted as {class_name[c]} with probability {prediction_p[c]}. The actual label is {class_name[label[0]]} ")
                    attention = A[:,:,c].squeeze().cpu().numpy()
                    attn_df = pd.DataFrame(attention)
                    attn_df.to_csv(os.path.join(img_save_path, f'{slide_name}_{prediction_p[c]}_{prediction}_{label.item()}.csv'))
                    # print(f"attention shape: {attention.shape}  attention: {attention}")
                    # num_pos_classes += 1

                    # if prediction == 0:
                    #     print(test_bags[i] + ' is detected as: benign')
                    #     # attentions = torch.sum(A, 2).cpu().numpy()
                    #     print(f"attentions shape when it is normal: {attention.shape}  attentions: {attention}")
                    #     colored_tiles = np.matmul(attention[:,:, None], attn_color[None, :]) * 0
                    #     # print(f"colored tiles shape when it is normal: {colored_tiles.shape}  colored tiles: {colored_tiles}")
                    
                    # else:
                        # print(test_bags[i] + ' is detected as: malignant /BAS')
                        # print(f"attentions shape when it is normal: {attention.shape}  attentions: {attention}")
                    colored_tiles = np.matmul(attention[:,:, None], attn_color[None, :])


                        # print(f"colored tiles shape when it is not normal: {colored_tiles.shape}  colored tiles: {colored_tiles}")

           
            # benign = True
            # num_pos_classes = 0
            # for c in range(args.num_classes):          
            #     if c == prediction:
            #         print(f"The slide {os.path.basename(test_bags[i])} is predicted as {class_name[c]} with probability {prediction_p[c]}. The actual label is {class_name[label[0]]} ")
            #         attentions = A[:,:, c].squeeze().cpu().numpy()
            #         attn_df = pd.DataFrame(attentions)
            #         attn_df.to_csv(os.path.join(img_save_path, f'score_{slide_name}_{prediction}_{label.item()}.csv'))
            #         num_pos_classes += 1
            #         # print(f"attentions shape when it is normal: {attentions.shape}  attentions: {attentions}")
            #         if benign: # first class detected
            #             print(test_bags[i] + ' is detected as: ' + class_name[c])
            #             colored_tiles = np.matmul(attentions[:,:, None], attn_color[None, :])
            #             # print(f"colored tiles shape when it is normal: {colored_tiles.shape}  colored tiles: {colored_tiles}")
            #         else:
            #             print('and ' + class_name[c])          
            #             colored_tiles = colored_tiles + np.matmul(attentions[:,:, None], attn_color[None, :])
            #             # print(f"colored tiles shape when it is not normal: {colored_tiles.shape}  colored tiles: {colored_tiles}")
            #         benign = False # set flag

            # if benign:
            #     print(test_bags[i] + ' is detected as: benign')
            #     attentions = torch.sum(A, 2).cpu().numpy()
            #     colored_tiles = np.matmul(attentions[:,:, None], attn_color[None, :]) * 0

            # colored_tiles = (colored_tiles / num_pos_classes)
            # colored_tiles = exposure.rescale_intensity(colored_tiles, out_range=(0, 1))  # type: ignore

            dx = pos_arr[0][3]-pos_arr[0][1]
            dy = pos_arr[0][4]-pos_arr[0][2]


            pos = np.stack(((pos_arr[:,1]-pos_arr[:,1].min())/dx, (pos_arr[:,2]-pos_arr[:,2].min())/dy), axis=1 ).astype(int)

            for j, cts in enumerate(colored_tiles): # type: ignore
                color_map = np.zeros(((np.amax(pos[:,0], 0)+1).astype(int) * scale, int(np.amax(pos[:,1], 0)+1) * scale, 3))
                tile_map = np.ones(((np.amax(pos[:,0], 0)+1).astype(int) * scale, int(np.amax(pos[:,1], 0)+1) * scale, 3))
                output_map = tile_map.copy()

                for k, p in enumerate(pos):

                    color_map[p[0]*scale:(p[0]+1)*scale, p[1]*scale:(p[1]+1)*scale] = cts[k]
                    tile_map[p[0]*scale:(p[0]+1)*scale, p[1]*scale:(p[1]+1)*scale] = tiles[k]

                    color_map_img = Image.fromarray(img_as_ubyte(color_map[p[0]*scale:(p[0]+1)*scale, p[1]*scale:(p[1]+1)*scale]))
                    tile_map_img = Image.fromarray(img_as_ubyte(tile_map[p[0]*scale:(p[0]+1)*scale, p[1]*scale:(p[1]+1)*scale]))
                    temp_output_map_img = Image.blend(tile_map_img, color_map_img, 0.56)
                    temp_output_map = np.array(temp_output_map_img)

                    if k in topk_idx:
                        a = 16
                        t = np.where(topk_idx == k)[0][0]
                        temp_output_map = cv2.rectangle(temp_output_map, (0,0), (scale,scale), color = topk_color, thickness = int(a * (1 - t * 1/len(topk_idx))))
 
                    else:
                        temp_output_map = temp_output_map
                    
                    output_map[p[0]*scale:(p[0]+1)*scale, p[1]*scale:(p[1]+1)*scale] = exposure.rescale_intensity(temp_output_map, out_range=(0, 1))
                    # output_map[p[0]*scale:(p[0]+1)*scale, p[1]*scale:(p[1]+1)*scale] = temp_output_map
                # output_map = exposure.rescale_intensity(output_map, out_range=(0, 1))
                output_map_img = Image.fromarray(img_as_ubyte(output_map))                
                output_map_img.save(os.path.join(img_save_path, f'head_{j}'+'.png'))



def main():
    import random
    class Config:
        grad_norm_clip = 1.02580285460309
        classes = 2
        alpha = 0.849435264741892
        in_size = 1000
        out_size = 2
        dropout = 0.3
        lr = 0.0203373691328209
        weight_decay = 0.0001
        batch_size = 420
        device = torch.device("cuda:0")
        optimizer = "SGD"
        scheduler = "cosine"
        betas = (0.5, 0.9)
        momentum = 0.9
        minlr = 0.000005
        dataset = "CNT"
        num_heads = 12
        topk = 8
        seed = 1337
        critical_features_from = "original"  # "original" or "embedding" or"index_add"
        embed_module = "Sparse" # "Sparse" or "Dense"
        topkhead_classification = True
        head_proj =  "conv1d_ckernels"
        head_dim =  128 
        embed_x_dim = 256
        projection = True
        # "classification":"mean",
        split = "random"
        sd = random.randint(1, 9999)
        df = "df_combined"
        bag_dir = "/restricteddata/skincancer_kuk/tiles_20x/Tiles"
        model_path = "/system/user/publicwork/yitaocai/Master_Thesis/model/topk_multihead_2C/Topk8_12heads_conv1d_ckernels_CFfromembedding_2C_CNT_weightedFalse_determined_03052023.pt"


        def __init__(self, **kwargs):
            for k,v in kwargs.items():
                setattr(self, k, v)

    config = Config()
    bags_list = sorted(glob(os.path.join(config.bag_dir,"*","*"), recursive=True))
    save_path = "/system/user/publicwork/yitaocai/Master_Thesis/attention_map_TopkMultihead_2C"
    label_file = '/restricteddata/skincancer_kuk/Scanned_WSI/metadata_workingfile_label.csv'
    
    df_test = build_testloader(config)

    # model = DSMILNet(ins_classifier, bag_classifier)
    compute_attn_map(bags_list,label_file,df_test,save_path,config)


if __name__ == "__main__":
    main()