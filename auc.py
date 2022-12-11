import numpy as np
from sklearn.metrics import f1_score
import argparse
import numpy as np
import os
import torch
import torch.nn.functional as F
from utils.data import Data
from utils.model import MISLnet as Model
from tensorboardX import SummaryWriter
from progressbar import *
from torchvision import transforms
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score

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

checkpoint = torch.load('./model/91.72 83.3.pkl')
model.load_state_dict(checkpoint['model_state_dict'], strict=True)
#model.eval()
# truth:1 rumor:0

progress = ProgressBar()
prob_all = []
label_all = []
for step, (x, y) in enumerate(progress(data.train_loader)):
    #x, y = x.to(conf.device), y.to(conf.device)
    x, y = x.to(conf.device), y.to(conf.device)
    logist, output, constrained_image, outputs = model(x)
    prob = output
    prob_all.extend(prob[:,1].cpu().detach().numpy()) 
    label_all.extend(y.cpu().numpy())
   
print("AUC:{:.4f}".format(roc_auc_score(label_all,prob_all)))

