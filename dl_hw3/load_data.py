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
import random

from PIL import Image
import numpy as np

class ImgDataset(Dataset):
    
    def __init__(self, root, folder, transform=None):
        super(ImgDataset).__init__()
        self.root = os.path.join(root, folder)
        self.files = os.listdir(self.root)
        
        
    def __getitem__(self, idx):
        filepath = os.path.join(self.root, self.files[idx])
        img = Image.open(filepath)
        img = transforms.ToTensor()(img)
        width = img.shape[2]
        #x, y = img[:, :, width // 2 :], img[:, :, : width // 2] # for facades
        x, y =  img[:, :, : width // 2], img[:, :, width // 2 :] # for handbags
        return x, y


    def __len__(self):
        return len(self.files)


def make_loader(root, modes=['train', 'val', 'test'], bs=512):
    loaders = []
    for mode, batch_size in zip(modes, [bs, 2, 2]):
	      dataset = ImgDataset(root, mode)
	      loaders.append(DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=1, pin_memory=True))
    return loaders
