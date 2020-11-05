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
from train import set_seed, train, init_weights
from model import EncoderRNN, AttnDecoderRNN
from load_data import make_datasets
import numpy as np
from eval import make_predictions

from tqdm import tqdm

MAX_LENGTH = 90
TRAIN = 'homework_machine_translation_de-en/train.de-en'
VAL = 'homework_machine_translation_de-en/val.de-en'
TEST = 'homework_machine_translation_de-en/test1.de-en'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    set_seed(1992)
    hidden_size = 128
    learning_rate=3e-4
    num_epochs = 10
    train_iter, val_iter, SRC, TRG = make_datasets(TRAIN, VAL, TEST)
    learning_rate = 1e-3

    encoder = EncoderRNN(len(SRC.vocab), hidden_size).to(device)
    attn_decoder = AttnDecoderRNN(hidden_size, len(TRG.vocab), dropout_p=0.1).to(device)

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(attn_decoder.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss(ignore_index=TRG.vocab.stoi['<pad>'])
    train(num_epochs, train_iter, encoder, attn_decoder, encoder_optimizer, decoder_optimizer, criterion, 
          SRC, TRG, max_length=MAX_LENGTH)
    make_predictions(VAL, encoder, attn_decoder, SRC, TRG, val=True)
    make_predictions(TEST, encoder, attn_decoder, SRC, TRG)


main()
