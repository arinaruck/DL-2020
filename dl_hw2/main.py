import re
import torch.nn as nn
from io import open
import string
import random
import torch
from torch import optim
import torch.nn.functional as F
from torchtext import data, datasets
from torchtext.data import Iterator, BucketIterator
from train import set_seed, train
from model import Transformer, CECriterion, init_weights
from load_data import make_datasets
import numpy as np
from eval import make_predictions

from tqdm import tqdm

TRAIN = 'homework_machine_translation_de-en/train.de-en'
VAL = 'homework_machine_translation_de-en/val.de-en'
TEST = 'homework_machine_translation_de-en/test1.de-en'
d_model = 256
BATCH_SIZE = 48
MAX_LENGTH = 85
lr = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


config = {
            'batch_size': BATCH_SIZE,
            'd_model': d_model,
            'max_len': MAX_LENGTH,
            'n_heads': 4, 
            'n_blocks': 4, 
            'd_ffd': 4 * d_model,
            'emb_dim': d_model,
            'emb_dropout': 0.4,
            'd_k': d_model // 4,
            'd_v': d_model // 4, 
            'dropout' : 0,
            'num_epochs': 10,
            'device': device,
            'lr': lr,
            'optimizer': 'Adam',
            'seed': 1992,
            'checkpoint': 'checkpoint.pt'
        }


def main():
    set_seed(1992)
    train_iter, val_iter, SRC, TRG = make_datasets(config, TRAIN, VAL)

    config['src_vocab'] = len(SRC.vocab)
    config['trg_vocab'] = len(TRG.vocab)

    model = Transformer(config).to(device)
    model.apply(init_weights)
    model.proj.weight = model.trg_embedding.embed.weight

    criterion = CECriterion(TRG.vocab.stoi['<pad>'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    train(config, model, criterion, optimizer, train_iter, val_iter)
    make_predictions(config, VAL, SRC, TRG, val=True)
    make_predictions(config, TEST, SRC, TRG)


main()
