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

from model import ResNet34Model, EarlyStopping, LabelSmoothLoss, DenseNetModel, MobileNetModel
from load_data import make_loader
from train_n_pred import train, predict

import sys

ROOT = 'simple_image_classification'
LBLPATH = 'labels_trainval.csv'
DATAPATH_TRAIN = 'trainval'
DATAPATH_TEST = 'test'
N_CLASSES = 200 
CONFIG = {'seed': 1992, 'lr': 5e-4, 'epochs': 55, 'batch_size': 160}
MEAN = torch.Tensor([0.5746, 0.5327, 0.4449])
STD = torch.Tensor([0.3058, 0.3135, 0.3006])
MODE = 'train'
LOAD_MODEL = False

composed = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomAffine(degrees=(-15, 15),
        scale=(0.85, 1.15),
        shear=(-10, 10)),
    transforms.ColorJitter(0, 5),
    transforms.RandomPerspective(0.1, p=0.2),
    transforms.RandomHorizontalFlip(),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])


test_dummy = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
])

test_full = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomAffine(degrees=(-15, 15),
        scale=(0.85, 1.15),
        shear=(-10, 10)),
    transforms.ColorJitter(0, 5),
    transforms.RandomPerspective(0.1, p=0.2),
    transforms.RandomHorizontalFlip(),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def main():

    CONFIG['weight_decay'] = 0.001
    print(CONFIG['weight_decay'], CONFIG['lr'])
    set_seed(CONFIG['seed'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = MobileNetModel(N_CLASSES).to(device)
    if LOAD_MODEL:
        model.load_state_dict(torch.load(f'./checkpoint.pt'))
    if MODE == 'eval':
        train_loader = make_loader(ROOT, DATAPATH_TRAIN, composed, LBLPATH, bs=CONFIG['batch_size'])
    else:
        train_loader, val_loader = make_loader(ROOT, DATAPATH_TRAIN, composed, LBLPATH, bs=CONFIG['batch_size'], mode=MODE)
        early_stopping = EarlyStopping(checkpoint=f'./checkpoint_{sys.argv[1]}.pt', patience=10, verbose=True)
        grad_acum = 2
        optimizer = Adam(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=1, factor=0.5)
        train(CONFIG['epochs'], model, optimizer, scheduler, device, early_stopping, train_loader, val_loader, criterion=LabelSmoothLoss(0.2), grad_acum=1)
    test_loader = make_loader(ROOT, DATAPATH_TEST, test_dummy, bs=1)
    res = predict(test_loader, model, device, transforms=test_full, iters=5)
    res.to_csv(f'labels_test.csv', sep=',', index=False)


if __name__ == '__main__':
	main()
