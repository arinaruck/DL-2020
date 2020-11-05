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


#https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, checkpoint, patience=7, verbose=False, delta=0, min_loss=np.inf):
        """
        :param
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None if min_loss == np.inf else  -min_loss
        self.early_stop = False
        self.val_loss_min = min_loss
        self.delta = delta
        self.checkpoint = checkpoint

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Saves model when validation loss decrease.
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.checkpoint)
        self.val_loss_min = val_loss

        
class CNNModel(nn.Module):
    def __init__(self, n_classes, dropout):
        super(CNNModel, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(1600, 1024),
            nn.Dropout(dropout),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.Dropout(dropout),
            nn.ReLU(True),
            nn.Linear(512, n_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.net(x)
        return x


class ResNet18Model(nn.Module):
    def __init__(self, n_classes):
        super(ResNet18Model, self).__init__()
        self.resnet18 = torchvision.models.resnet18()
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.ReLU(True),
            nn.BatchNorm1d(1000),
            nn.Linear(1000, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, n_classes),
        )
        

    def forward(self, x):
        x = self.resnet18(x)
        x = self.head(x)
        return x


class ResNet34Model(nn.Module):
    def __init__(self, n_classes):
        super(ResNet34Model, self).__init__()
        self.resnet34 = torchvision.models.resnet34()
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.ReLU(True),
            nn.BatchNorm1d(1000),
            nn.Linear(1000, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, n_classes),
        )
        

    def forward(self, x):
        x = self.resnet34(x)
        x = self.head(x)
        return x

class DenseNetModel(nn.Module):
    def __init__(self, n_classes):
        super(DenseNetModel, self).__init__()
        self.densenet = torchvision.models.densenet161()
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.ReLU(True),
            nn.BatchNorm1d(1000),
            nn.Linear(1000, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, n_classes),
        )
        

    def forward(self, x):
        x = self.densenet(x)
        x = self.head(x)
        return x

class MobileNetModel(nn.Module):
    def __init__(self, n_classes):
        super(MobileNetModel, self).__init__()
        self.densenet = torchvision.models.mobilenet_v2()
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.ReLU(True),
            nn.BatchNorm1d(1000),
            nn.Linear(1000, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, n_classes),
        )
        

    def forward(self, x):
        x = self.densenet(x)
        x = self.head(x)
        return x


class LabelSmoothLoss(nn.Module):

    def __init__(self, smoothing=0.0):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * \
            self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss

