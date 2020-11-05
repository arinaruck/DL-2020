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

MAX_LENGTH=90
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def from_tokens(tokens, SRC, TRG):
    return ' '.join(TRG.vocab.itos[tok] for tok in tokens) 


def to_tokens(sentence, SRC, TRG):
    tokens = [SRC.vocab.stoi['<sos>']] + [SRC.vocab.stoi[token.lower()] for token in sentence.split()]\
    + [SRC.vocab.stoi['<eos>']]
    source = torch.LongTensor(tokens).unsqueeze(0).to(device)
    return source


def evaluate(encoder, decoder, input_tensor, SRC, TRG,  max_length=MAX_LENGTH):
    encoder.eval()
    decoder.eval()

    SOS_token, EOS_token, PAD_token = TRG.vocab.stoi['<sos>'], TRG.vocab.stoi['<eos>'], TRG.vocab.stoi['<pad>']

    with torch.no_grad():
        batch_size = input_tensor.size()[0]
        input_length = input_tensor.size()[1]
        encoder_hidden = encoder.initHidden(batch_size)
        encoder_outputs = torch.full((batch_size, max_length, encoder.hidden_size),
                                 PAD_token, dtype=torch.float32, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[:, ei].unsqueeze(0),
                                                     encoder_hidden)
            encoder_outputs[:, ei] = encoder_output

        decoder_input = torch.full((1, batch_size), SOS_token, dtype=torch.int64, device=device)

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(batch_size, max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[:, di] = decoder_attention.data
            topv, topi = decoder_output.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<eos>')
                break
            else:
                decoded_words.append(TRG.vocab.itos[topi.item()])

            decoder_input = topi.detach()
        return decoded_words, decoder_attentions[:, :di + 1]


def make_predictions(filename, encoder, decoder, SRC, TRG, val=False):
    preds = []
    suffix = '_pred' if val else ''
    with open(f'{filename}.de', 'r') as in_file:
        lines = in_file.readlines()
        for line in lines:
            tokens = to_tokens(line, SRC, TRG)
            output_words, _ = evaluate(encoder, decoder, tokens, SRC, TRG)
            preds.append(' '.join(output_words[1:-1]))
    with open(f'{filename}{suffix}.en', 'w') as out_file:
        out_file.write('\n'.join(preds))

