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

checkpoint = torch.load('./model/89.95 84.22.pkl')
model.load_state_dict(checkpoint['model_state_dict'], strict=True)
#model.eval()
# truth:1 rumor:0

progress = ProgressBar()
correct = 0
prob_all = []
label_all = []

for step, (x, y) in enumerate(progress(data.train_loader)):
    #x, y = x.to(conf.device), y.to(conf.device)
    x, y = x.to(conf.device), y.to(conf.device)
    logist, output, constrained_image, outputs = model(x)
    prob = output
    prob = prob.cpu().detach().numpy()
    prob_all.extend(np.argmax(prob,axis=1))
    label_all.extend(y.cpu().numpy())
    pred = output.data.max(1)[1]
    correct += pred.eq(y.data.view_as(pred)).cpu().sum().item()
acc = 100.0 * (correct / len(data.train_loader.dataset))
print(acc)
print("F1-Score:{:.4f}".format(f1_score(label_all,prob_all)))

