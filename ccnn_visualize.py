import os
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse
import skimage.data
import skimage.io
import skimage.transform
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import argparse
from utils.model import MISLnet as Model
from tensorboardX import SummaryWriter
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import transforms

# Creates writer1 object.
# The log will be saved in 'log'
writer = SummaryWriter('./log')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="conf")
    return parser.parse_args()


def get_picture(pic_name, transform):
    img_load = Image.open(pic_name).convert('L').resize((256, 256)) 
    img_data = transform(img_load)
    return img_data

def make_dirs(path):
    if os.path.exists(path) is False:
        os.makedirs(path)


def get_feature(i):
    pic_dir = './images/rumor/{}.jpg'.format(i)
    transform = transforms.Compose([transforms.ToTensor(),
                                ])
    img = get_picture(pic_dir, transform)
    device = torch.device("cuda")
    # 插入维度
    img = img.unsqueeze(0)
    img = img.to(device)
    args = get_args()
    conf = __import__("config." + args.config, globals(), locals(), ["Conf"]).Conf

    net = Model(conf,writer).to(conf.device)
    checkpoint = torch.load('./model/60-epoch.pkl')
    net.load_state_dict(checkpoint['model_state_dict'], strict=True)
    exact_list = None
    dst = './all_feature1/r_{}_feautures'.format(i)
    therd_size = 256

    logist, output, constrained_image, outputs = net(img)
    outs = outputs
    for k, v in outs.items():
        features = v[0]
        iter_range = features.shape[0]
        for i in range(iter_range):      
            feature = features.data.cpu().numpy()
            feature_img = feature[i,:,:]
            feature_img = np.asarray(feature_img * 255, dtype=np.uint8)
            dst_path = os.path.join(dst, k)
            make_dirs(dst_path)
            feature_img = cv2.applyColorMap(feature_img, cv2.COLOR_BGR2GRAY)
            if feature_img.shape[0] < therd_size:
                tmp_file = os.path.join(dst_path, str(i) + '_' + str(therd_size) + '.png')
                tmp_img = feature_img.copy()
                tmp_img = cv2.resize(tmp_img, (therd_size,therd_size), interpolation =  cv2.INTER_NEAREST)
                cv2.imwrite(tmp_file, tmp_img)
                dst_file = os.path.join(dst_path, str(i) + '.png')
                cv2.imwrite(dst_file, feature_img)
            
if __name__ == '__main__':
    i = [0, 1, 2, 3, 5, 6, 7, 8, 9]
    for j in i:
        get_feature(j)
