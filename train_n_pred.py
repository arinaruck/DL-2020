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
from load_data import ImgDataset

N_CLASSES = 200

def train(epochs, model, optimizer, scheduler, device, early_stopping,
	train_loader, valid_loader=None, grad_acum=1, criterion=nn.NLLLoss()):
    for epoch in range(epochs):
        optimizer.zero_grad()
        tr_loss, val_loss = 0, 0
        tr_steps, val_steps = 0, 0
        for X, y in train_loader:
            model.train()
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = criterion(pred, y)
            tr_loss += loss.item()
            loss.backward()
            tr_steps += 1
            if tr_steps % grad_acum == 0:
                optimizer.step()
                optimizer.zero_grad()
        all_preds = torch.tensor([])
        all_labels = torch.tensor([])
        if valid_loader is not None:
            for X, y in valid_loader:
                model.eval()
                with torch.no_grad():
                    X, y = X.to(device), y.to(device)
                    pred = model(X)
                loss = criterion(pred, y)
                pred = torch.argmax(pred, 1)
                val_loss += loss.item()
                val_steps += 1
                all_preds = torch.cat([all_preds, pred.to('cpu')])
                all_labels = torch.cat([all_labels, y.to('cpu')])
            early_stopping(val_loss, model)
            scheduler.step(val_loss)
            accuracy = torch.true_divide((all_labels ==  all_preds).sum(), len(all_labels))
            print(f'accuracy: {accuracy}')
            if early_stopping.early_stop:
                print("Early stopping")
                break


def predict(dataset, model, device, transforms, iters=1):
    model.eval()
    ids = []
    preds = []
    with torch.no_grad():
         for X, file_ids in dataset:
            X_inp = X.squeeze(0)
            pred = torch.zeros((1, N_CLASSES), dtype=torch.float).to(device)
            for i in range(iters):
                X = transforms(X_inp)
                pred = model(X.to(device).unsqueeze(0))
            pred /= iters
            pred = torch.argmax(pred, dim=1)
            ids.extend(list(file_ids))
            preds.extend(pred.to('cpu').tolist())
    res = pd.DataFrame({'Id' : ids, 'Category' : preds})
    res.Category = res.Category.apply(lambda label: "{0:0>4}".format(label))
    return res
