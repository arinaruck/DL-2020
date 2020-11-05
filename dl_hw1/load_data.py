import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pandas as pd
import os
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.models as models

from PIL import Image
import numpy as np

class ImgDataset(Dataset):
    
    def __init__(self, root, datapath, lblpath=None, transform=None):
        super(ImgDataset).__init__()
        self.root = os.path.join(root, datapath)
        self.targets = None
        self.transform = None
        if lblpath:
            meta = pd.read_csv(os.path.join(root, lblpath))
            self.files = meta.Id.values
            self.targets = meta.Category.values
        else:
            self.files = os.listdir(self.root)
        if transform is not None:
            self.transform = transform
        
        
    def __getitem__(self, idx):
        filepath = os.path.join(self.root, self.files[idx])
        img = Image.open(filepath)
        if self.transform is not None:
            img = self.transform(img)
        if self.targets is not None:
            target = int(self.targets[idx])
            return img, target
        return img, self.files[idx]
        
  

    def __len__(self):
        return len(self.files)


def make_loader(root, datapath, transform, lblpath=None, mode='eval', bs=512):
	dataset = ImgDataset(root, datapath, lblpath, transform=transform)
	if mode == 'eval':
		dataloader = DataLoader(dataset, shuffle=True, batch_size=bs)
		return dataloader

	len_full = len(dataset)
	len_train, len_val = int(len_full * 0.8), len_full - int(len_full * 0.8)
	train_dataset, val_dataset = random_split(dataset, [len_train, len_val])
	train_loader = DataLoader(train_dataset, shuffle=True, batch_size=bs, num_workers=5, pin_memory=True)
	val_loader = DataLoader(val_dataset, shuffle=True, batch_size=bs, num_workers=5, pin_memory=True)
	return train_loader, val_loader

