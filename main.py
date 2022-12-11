import argparse
import numpy as np
import os
import torch
import time
import torch.nn.functional as F
from utils.data import Data
from utils.model import MISLnet as Model
from tensorboardX import SummaryWriter
from progressbar import *
from torchvision.utils import make_grid

writer = SummaryWriter('./log')

np.random.seed(100)
torch.manual_seed(100)
torch.cuda.manual_seed(100)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="conf")
    return parser.parse_args()

def adjust_learning_rate(learning_rate, learning_rate_decay, optimizer, epoch):
    """Sets the learning rate to the initial LR multiplied by learning_rate_decay(set 0.98, usually) every epoch"""
    learning_rate = learning_rate * (learning_rate_decay ** epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    return learning_rate

def model_training(conf, model, loader_train, loader_val):
    
    learning_rate = conf.learning_rate
    optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)
    
    print("\n-> start to train model!")
    max_step = 0
    for epoch in range(conf.total_epoch):
        progress = ProgressBar()
        lr = adjust_learning_rate(learning_rate=learning_rate,
                                  learning_rate_decay=0.98, optimizer=optimizer, epoch=epoch) 
        loss = 0.0 
        correct = 0
        train_loss = 0.0
        for step, (x, y) in enumerate(progress(loader_train)):
            max_step = max(max_step, step)
            global_step = epoch * max(max_step, step) + step
            x, y = x.to(conf.device), y.to(conf.device)
            #print(x,y)
            logist, output, constrained_image, outputs = model(x)
            loss = F.cross_entropy(output, y)
            train_loss += loss
            # writer.add_image("CONSTRAINED IMAGE",make_grid(constrained_image),global_step)
            # writer.add_image("IMAGE",make_grid(x),global_step)
            writer.add_scalar("TRAIN-SET LOSS", loss, global_step=global_step)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            pred = output.data.max(1)[1]
            correct += pred.eq(y.data.view_as(pred)).cpu().sum().item()
        acc = 100.0 * (correct / len(loader_train.dataset))
        print("-> training epoch={:d} loss={:.3f} acc={:.3f}% learning rate={:.5f}".format(epoch, train_loss*32/len(loader_train.dataset), acc, lr))
        now = time.strftime("%Y-%m-%d %H-%M-%S")
        torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'learning rate': lr
        }, './model/parameter' + str(now) + '.pkl') 
        val_loss, val_acc = model_validating(conf, model, loader_val) 
        writer.add_scalar("VALIDATION-SET LOSS", val_loss, global_step=global_step)
        writer.add_scalar("VALIDATION-SET ACCURACY", val_acc, global_step=global_step)

def model_validating(conf, model, val_loader):
    correct = 0
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(conf.device), y.to(conf.device)
            logist, output, temp, outputs = model(x)
            val_loss += F.cross_entropy(output, y, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()
    val_loss /= len(val_loader.dataset)
    acc = 100. * correct / len(val_loader.dataset)
    print("-> validating loss={} acc={}".format(val_loss, acc))
    return val_loss, acc

def model_testing(conf, model, test_loader):
    if os.listdir('./model'): 
        checkpoint = torch.load('./model/parameter' + "2020-09-25 08-00-24" + '.pkl')
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        print("\n-> finish to load model!")
        model.eval() 
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(conf.device), y.to(conf.device)
            logist, output, temp = model(x)
            test_loss += F.cross_entropy(output, y, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    print("-> testing loss={} acc={}".format(test_loss, acc))
    return test_loss, acc

def main():
    args = get_args()
    conf = __import__("config." + args.config, globals(), locals(), ["Conf"]).Conf
   
    data = Data(conf)
    data.load_data()
    
    model = Model(conf,writer).to(conf.device)
    print(model)
    model_training(conf, model, data.train_loader, data.val_loader)

if __name__ == "__main__":
    main()