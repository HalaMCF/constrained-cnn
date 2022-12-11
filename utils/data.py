import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms
import os
import torch
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
transform = transforms.Compose([transforms.ToTensor(),
                                ])

class ImgDataset(Dataset):
    def __init__(self, root='../dataset', mode='train'):
        self.root_path = os.path.join(root, mode) #
        self.classes = [os.path.join(self.root_path, i)
                        for i in os.listdir(self.root_path)] 

        self.imgs = []
        real = 0
        manipulated = 0
        for index in range(len(self.classes)):
            for j in os.listdir(self.classes[index]):
                if index == 0:
                    manipulated += 1
                else:
                    real += 1
                self.imgs.append(os.path.join(self.classes[index], j)) # path of images
        self.labels = np.zeros(np.int(manipulated), dtype=np.long).tolist() + np.ones(np.int(real), dtype=np.long).tolist()
        self.transforms = transform

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img_label = self.labels[index]
        img_load = Image.open(img_path).convert('L').resize((256, 256)) # convert to gray image, resize to (256, 256)
        if self.transforms:
            img_data = self.transforms(img_load)
        else:
            img_temp = np.asarray(img_load)
            img_data = torch.from_numpy(img_temp)
        return img_data, img_label
    def __len__(self):
        return len(self.imgs)

class Data():
    def __init__(self, conf):
        self.conf = conf
        self.data_path = conf.data_path
        self.train_loader = None
        self.test_loader = None
        self.val_loader = None

    def load_data(self, batch_size=0):
        print("-> load data from: {}".format(self.data_path))
        if not batch_size:
            batch_size = self.conf.batch_size
        total_train_db = ImgDataset(root=self.data_path, mode="train")
        train_size = int(0.8*len(total_train_db))
        val_size = int(len(total_train_db) - train_size)
        train_db, val_db = random_split(total_train_db, [train_size, val_size])
        self.train_loader = DataLoader(train_db, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_db, batch_size=batch_size, shuffle=True)
        return self
    
    def pred_data(self, batch_size=0):
        if not batch_size:
            batch_size = self.conf.batch_size
        total_train_db = ImgDataset(root=self.data_path, mode="train")
        self.train_loader = DataLoader(total_train_db, batch_size=batch_size, shuffle=True)
        return self

if __name__ == '__main__':
    img_dataset = ImgDataset(root='../dataset', mode='train')