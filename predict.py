import argparse
import numpy as np
import os
import torch
import torch.nn.functional as F
from utils.data import Data
from utils.model import MISLnet as Model
from tensorboardX import SummaryWriter
from progressbar import *
from torchvision.utils import make_grid
from torchvision import transforms
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F

transform = transforms.Compose([transforms.ToTensor(),
                                ])

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="conf")
    return parser.parse_args()

writer = SummaryWriter('./log')

args = get_args()
conf = __import__("config." + args.config, globals(), locals(), ["Conf"]).Conf
   
   
data = Data(conf)
data.pred_data()
    
model = Model(conf,writer).to(conf.device)

checkpoint = torch.load('./model/fair3.pkl')
model.load_state_dict(checkpoint['model_state_dict'], strict=True)
#model.eval()
# truth:1 rumor:0

def getfiles(file):
    path_list = []
    filenames = os.listdir(file)
    for filename in filenames:
        a = os.path.join(file, filename)
        path_list.append(a)
    return path_list

file = './images/rumor'
is_correct = 0
for i in getfiles(file):
    img_load = Image.open(i).convert('L').resize((256, 256))
    img_data = transform(img_load)
    device = torch.device("cuda")
    img_data = img_data.unsqueeze(0)
    img_data = img_data.to(device)
    logist, output, constrained_image, outputs = model(img_data)
    output = output.data.max(1)[1]
    if output == 1:
        is_correct += 1
    print(i, output)
print(is_correct)

'''
progress = ProgressBar()
correct = 0
for step, (x, y) in enumerate(progress(data.train_loader)):
    #x, y = x.to(conf.device), y.to(conf.device)
    x, y = x.to(conf.device), y.to(conf.device)
    logist, output, constrained_image, outputs = model(x)
    pred = output.data.max(1)[1]
    correct += pred.eq(y.data.view_as(pred)).cpu().sum().item()
acc = 100.0 * (correct / len(data.train_loader.dataset))
print(acc)
'''