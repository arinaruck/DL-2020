import torch.nn as nn
import torch.functional as F
import torch
import numpy as np
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FeedForward(nn.Module):
    def __init__(self, config):
        super(FeedForward, self).__init__()
        self.ffd = nn.Sequential(
            nn.Linear(config['d_model'], config['d_ffd']),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['d_ffd'], config['d_model'])                  
        )
    
    def forward(self, x):
        x = self.ffd(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        d_model, d_k, d_v = config['d_model'], config['d_k'], config['d_v']
        n_heads = config['n_heads']
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.K = nn.Linear(d_model, d_k * n_heads)
        self.Q = nn.Linear(d_model, d_k * n_heads)
        self.V = nn.Linear(d_model, d_v * n_heads)
        self.proj = nn.Linear(n_heads * d_v, d_model)
        self.dropout = nn.Dropout(config['dropout'])
        self.to_prob = nn.Softmax(dim=-1)
    
    def forward(self, x_k, x_q, x_v, mask):
        b_sz, seq_len, emb_dim = x_k.shape
        key = self.K(x_k).view(b_sz, -1, self.n_heads, self.d_k).transpose(1, 2)
        query = self.Q(x_q).view(b_sz, -1, self.n_heads, self.d_k).transpose(1, 2)
        value = self.V(x_v).view(b_sz, -1, self.n_heads, self.d_k).transpose(1, 2)
        logits = torch.matmul(query, key.transpose(-1, -2)) / np.sqrt(self.d_k)
        mask = mask.unsqueeze(1)
        logits = logits.masked_fill(~mask, -1e15)
        probs = self.to_prob(logits)
        z = torch.matmul(self.dropout(probs), value)
        z = z.transpose(1, 2).contiguous().view(b_sz, -1, self.n_heads * self.d_v)
        z = self.proj(z)
        return z


class EncoderBlock(nn.Module):
    def __init__(self, config):
        super(EncoderBlock, self).__init__()
        self.self_attn = MultiHeadAttention(config)
        self.layer_norm_attn = nn.LayerNorm(config['d_model'])
        self.feed_forward = FeedForward(config)
        self.layer_norm_ffd = nn.LayerNorm(config['d_model'])
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, x, mask):
        z = self.layer_norm_attn(x)
        x = x + self.self_attn(z, z, z, mask)
        z = self.layer_norm_ffd(x)
        x = x + self.dropout(self.feed_forward(z))
        return x


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        n_blocks = config['n_blocks']
        self.blocks = nn.ModuleList([EncoderBlock(config) for _ in range(n_blocks)])
        self.layer_norm = nn.LayerNorm(config['d_model'])
        
    def forward(self, x, src_mask):
        for enc_block in self.blocks:
            x = enc_block(x, src_mask)
        return self.layer_norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super(DecoderBlock, self).__init__()
        self.self_attn = MultiHeadAttention(config)
        self.layer_norm_s_attn = nn.LayerNorm(config['d_model'])
        self.enc_dec_attn = MultiHeadAttention(config)
        self.layer_norm_ed_attn = nn.LayerNorm(config['d_model'])
        self.feed_forward = FeedForward(config)
        self.layer_norm_ffd = nn.LayerNorm(config['d_model'])
        self.dropout = nn.Dropout(config['dropout'])
 
    def forward(self, x, enc, src_mask, trg_mask):
        z = self.layer_norm_s_attn(x)
        x = x + self.self_attn(z, z, z, trg_mask)
        z = self.layer_norm_ed_attn(x)
        x =  x + self.enc_dec_attn(enc, z, enc, src_mask)
        z = self.layer_norm_ffd(x)
        x = x + self.dropout(self.feed_forward(z))
        return x


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        n_blocks = config['n_blocks']
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(n_blocks)])
        self.layer_norm = nn.LayerNorm(config['d_model'])
        
    def forward(self, x, enc, src_mask, tgt_mask):
        for dec_block in self.blocks:
            x = dec_block(x, enc, src_mask, tgt_mask)
        return self.layer_norm(x)


def make_pos_emb(max_len, emb_dim, d_model):
    embedding = torch.zeros((max_len, emb_dim))
    pos = torch.arange(max_len).unsqueeze(1)
    freq =  torch.exp(torch.arange(0, d_model, 2) *
                             -(np.log(10000.0) / d_model)).unsqueeze(0)
    args = pos * freq
    embedding[:, 0::2] = torch.sin(args)
    embedding[:, 1::2] = torch.cos(args)
    return embedding.unsqueeze(0)


class Embedding(nn.Module):
    def __init__(self, config, vocab_size):
        super(Embedding, self).__init__()
        emb_dim = config['emb_dim']
        max_len = config['max_len']
        d_model = config['d_model']
        dropout = config['emb_dropout']
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.pos_embed = make_pos_emb(max_len, emb_dim, d_model).to(config['device'])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        seq_len = x.shape[-1]
        x = self.dropout(self.embed(x) * np.sqrt(self.d_model)
        + self.pos_embed[:, : seq_len,:])
        return x


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.config = config
        emb_dim = config['emb_dim']
        max_len = config['max_len']
        d_model = config['d_model']
        self.src_embedding = Embedding(config, config['src_vocab'])
        self.trg_embedding = Embedding(config, config['trg_vocab'])
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.proj = nn.Linear(d_model, config['trg_vocab'])


    def forward(self, src, trg, src_mask, trg_mask):
        src_emb = self.src_embedding(src)
        trg_emb = self.trg_embedding(trg)
        encoded = self.encoder(src_emb, src_mask)
        decoded = self.decoder(trg_emb, encoded, src_mask, trg_mask)
        return self.proj(decoded)


class EarlyStopping:
    def __init__(self, checkpoint, patience=7, verbose=True, delta=0, min_loss=np.inf):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None if min_loss == np.inf else  -min_loss
        self.early_stop = False
        self.val_loss_min = min_loss
        self.delta = delta
        self.checkpoint = checkpoint

    def __call__(self, val_loss, model, optimizer, scheduler, epoch):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, scheduler, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, scheduler, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer, scheduler, epoch):

        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            #'scheduler_state_dict': scheduler.state_dict(),
            'loss': val_loss
            }, self.checkpoint)

        self.val_loss_min = val_loss


class CECriterion(nn.Module):
    def __init__(self, pad_idx):
        super(CECriterion, self).__init__()
        self.pad_idx = pad_idx
        self.criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=pad_idx)
        
    def forward(self, x, target):
        x = x.contiguous().permute(0,2,1)
        ntokens = (target != self.pad_idx).data.sum()
        return self.criterion(x, target) / ntokens


def init_weights(model):
    if hasattr(model, 'weight') and model.weight.dim() > 1:
        nn.init.xavier_uniform_(model.weight.data)
