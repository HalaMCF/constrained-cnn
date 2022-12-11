import cv2
import numpy as np
import os
from torchvision import transforms
import torch
from utils.model import MISLnet as Model
from tensorboardX import SummaryWriter
import argparse
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
import time
import matplotlib.pyplot as plt

transform = transforms.Compose([transforms.ToTensor(),
                                ])

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

writer = SummaryWriter('./log')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="conf")
    return parser.parse_args()


args = get_args()
conf = __import__("config." + args.config, globals(), locals(), ["Conf"]).Conf
    
net = Model(conf,writer).to(conf.device)
checkpoint = torch.load('./model/60-epoch.pkl')
net.load_state_dict(checkpoint['model_state_dict'], strict=True)
image_path = './images/91.72/truth/false/3.jpg'
transform = transforms.Compose([transforms.ToTensor(),
                                ])
img_load = Image.open(image_path).convert('L').resize((256, 256)) # convert to gray image, resize to (256, 256)
img_data = transform(img_load)
device = torch.device("cuda")
    # 插入维度
img = img_data.unsqueeze(0)
img = img.to(device)

net(img)