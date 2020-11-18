import torch
import re
import torch.nn as nn
from io import open
import string
import random
from torch import optim
import torch.nn.functional as F
from torchtext import data, datasets
from torchtext.data import Iterator, BucketIterator
import numpy as np
from eval import *

from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(data_iter, model, criterion, optimizer):
    total_loss = 0
    for batch in tqdm(data_iter):
        optimizer.zero_grad()
        src, trg, src_mask, trg_mask = batch.src, batch.trg, batch.src_mask, batch.trg_mask
        out = model.forward(src, trg, src_mask, trg_mask)
        loss = criterion(out, batch.pred)

        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item()

    total_loss /= len(data_iter)
    return total_loss

def valid_epoch(data_iter, model, criterion, optimizer):
    total_loss = 0
    for batch in tqdm(data_iter):
        src, trg, src_mask, trg_mask = batch.src, batch.trg, batch.src_mask, batch.trg_mask
        out = model.forward(src, trg, src_mask, trg_mask)
        loss = criterion(out, batch.pred)
        total_loss += loss.detach().item()
    total_loss /= len(data_iter)
    return total_loss


def train(config, model, criterion, optimizer, train_iter, valid_iter):
    for epoch in range(config['num_epochs']):
        model.train()
        loss = train_epoch(train_iter, model, criterion, optimizer)
        print('train', loss)
        
        model.eval()
        with torch.no_grad():
            loss = valid_epoch(valid_iter, model, criterion, optimizer)
            print('valid', loss)


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
