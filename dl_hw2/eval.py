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
from model import Transformer

MAX_LENGTH=90


def from_tokens(tokens, SRC, TRG):
    return ' '.join(TRG.vocab.itos[tok] for tok in tokens) 


def to_tokens(sentence, SRC, TRG):
    tokens = [SRC.vocab.stoi['<sos>']] + [SRC.vocab.stoi[token.lower()] for token in sentence.split()]\
    + [SRC.vocab.stoi['<eos>']]
    source = torch.LongTensor(tokens).unsqueeze(0).to(DEVICE)
    return source


def greedy_decode(model, src, src_mask, max_len, TRG):
    embedded_src = model.src_embedding(src)
    enc = model.encoder(embedded_src, src_mask)
    ys = torch.LongTensor([[TRG.vocab.stoi['<sos>']]]).to(DEVICE)
    for i in range(1, max_len):
        embedded_trg = model.trg_embedding(ys)
        out = model.decoder(embedded_trg, enc, src_mask, subsequent_mask(i).to(DEVICE))
        prob = model.proj(out)
        _, next_word = torch.max(prob, dim = -1)
        next_word = next_word[:, -1].item()
        ys = torch.cat([ys, torch.LongTensor([[next_word]]).to(DEVICE)], dim=1)
        if next_word == TRG.vocab.stoi['<eos>']:
          break
    return ys.squeeze(0)


def beam_search_decode(model, src, src_mask, max_len, beam_width, TGT):
    to_prob = nn.Softmax(dim=-1).to(DEVICE)
    embedded_src = model.src_embedding(src)
    enc = model.encoder(embedded_src, src_mask)
    preds = torch.LongTensor([[TGT.vocab.stoi['<sos>']]])
    probs = torch.Tensor([[1.0]])
    eos_idx, pad_idx = TGT.vocab.stoi['<eos>'], TGT.vocab.stoi['<pad>']
    for i in range(1, max_len):
        curr_preds = torch.LongTensor([])
        curr_probs = torch.LongTensor([])
        for y, y_prob in zip(preds, probs):
            if y[-1].item() in [pad_idx, eos_idx]:
                word = torch.LongTensor([pad_idx])
                prediction = torch.cat([y, word])
                curr_preds = torch.cat([curr_preds, prediction.unsqueeze(0)], dim=0)
                curr_probs = torch.cat([curr_probs, y_prob.unsqueeze(0)], dim=0)
                continue
            embedded_trg = model.trg_embedding(y.to(DEVICE))
            out = model.decoder(embedded_trg, enc, src_mask, subsequent_mask(i).to(DEVICE))
            prob = to_prob(model.proj(out)).detach().cpu()
            word_probs, words = torch.topk(prob, k=beam_width, dim=-1)
            word_prob, word = word_probs[:, -1], words[:, -1]
            prediction = torch.cat([y.repeat(beam_width, 1), word.T], dim=1)
            curr_preds = torch.cat([curr_preds, prediction], dim=0)
            curr_probs = torch.cat([curr_probs, word_prob.T * y_prob], dim=0)
        probs, idx = torch.topk(curr_probs, k=beam_width, dim=0)
        idx = idx.squeeze()
        preds = curr_preds[idx, :]
    return preds[0]


def make_predictions(config, filename, SRC, TRG, val=False):
    model = Transformer(config).to(config['device'])
    model.load_state_dict(torch.load(config[checkpoint])['model_state_dict'])
    model.eval()

    preds = []
    suffix = '_pred' if val else ''
    with open(f'{filename}.de', 'r') as in_file:
        lines = in_file.readlines()
        for line in lines:
            src = to_tokens(line, SRC, TGT)
            src_mask = src != SRC.vocab.stoi["<pad>"]
            out = beam_search_decode(model, src, src_mask, max_len=MAX_LENGTH, TGT=TGT, beam_width=config['beam_width'])
            output_words = []
            for j in range(1, out.shape[0]):
                sym = TGT.vocab.itos[out[j]]
                if sym == "<eos>": break
                if sym == "<pad>": continue
                output_words.append(sym)
            test_preds.append(' '.join(output_words))
    with open(f'{filename}{suffix}.en', 'w') as out_file:
        out_file.write('\n'.join(preds))
        out_file.write('\n')

