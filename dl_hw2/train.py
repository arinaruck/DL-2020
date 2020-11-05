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


def train(epochs, train_iter, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, 
          SRC, TRG, max_length=MAX_LENGTH):
    SOS_token, EOS_token, PAD_token = TRG.vocab.stoi['<sos>'], TRG.vocab.stoi['<eos>'], TRG.vocab.stoi['<pad>']
    n_iters = len(train_iter)
    for epoch in tqdm(range(epochs)):
        loss_total = 0
        teacher_forcing_ratio = 1 - epoch // 10
        for i, batch in enumerate(train_iter):
            input_tensor = batch.src.to(device)
            target_tensor = batch.trg.to(device)

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            batch_size = input_tensor.size(0)
            input_length = input_tensor.size(1)
            target_length = target_tensor.size(1)
            encoder_outputs = torch.full((batch_size, max_length, encoder.hidden_size), 
                                        PAD_token, dtype=torch.float32, device=device)
            
            encoder_hidden = encoder.initHidden(batch_size)
          
            loss = 0

            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(
                    input_tensor[:, ei].unsqueeze(0), encoder_hidden)
                encoder_outputs[:, ei] = encoder_output

            decoder_input = torch.full((1, batch_size), SOS_token, dtype=torch.int64, device=device)

            decoder_hidden = encoder_hidden

            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                loss += criterion(decoder_output, target_tensor[:, di])
                if random.random() < teacher_forcing_ratio:
                    decoder_input = target_tensor[:, di].unsqueeze(0)
                else:
                    topv, topi = decoder_output.topk(1)
                    decoder_input = topi.detach().transpose(1, 0)
            
            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()

            loss_total += loss.item()
        print(f'epoch: {epoch}, avg loss: {loss_total / n_iters}')  
    return loss.item() / target_length

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)
