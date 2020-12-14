import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
from dataclasses import dataclass


@dataclass
class Config:
    up: List[Tuple[int, int]]
    down: List[Tuple[int, int]]
    discriminator: List[Tuple[int, int]]
    dropout_p: List[float]
    n_layers: int
    discriminator_layers: int
    lmbd: float
    epochs: int
    device: 'torch.device' = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


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

    def __call__(self, val_loss, model, optimizer, scheduler):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, scheduler)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, scheduler)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer, scheduler):
        """
        Saves model when validation loss decrease.
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': val_loss
            }, self.checkpoint)

        self.val_loss_min = val_loss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def gaussian(window_size, sigma):
    gauss_k = torch.exp(- (torch.arange(window_size) - window_size // 2)**2 / (2 * sigma**2))
    return gauss_k / gauss_k.sum()

def create_window(window_size, ch):
    g = gaussian(window_size, 1.5).unsqueeze(1)
    window = torch.matmul(g, g.T).unsqueeze(0).unsqueeze(0)
    window = window.expand(ch, 1, window_size, window_size).contiguous()
    return window

def SSIM(x, y, window_size=11):
    c1 = 1e-4
    c2 = 9e-4

    ch = x.shape[1]
    window = create_window(window_size, ch).to(x.device)
   
    mu_x = F.conv2d(x, window, padding = window_size // 2, groups = ch)
    mu_y = F.conv2d(y, window, padding = window_size // 2, groups = ch)

    mu_x_sq = torch.square(mu_x)
    mu_y_sq = torch.square(mu_y)
    mu_xy = mu_x * mu_y

    var_x = F.conv2d(x * x, window, padding = window_size // 2, groups = ch) - mu_x_sq
    var_y = F.conv2d(y * y, window, padding = window_size // 2, groups = ch) - mu_y_sq
    cov = F.conv2d(x * y, window, padding = window_size // 2, groups = ch) - mu_xy

    num = (2 * mu_xy + c1) * (2 * cov + c2)
    denom = (mu_x_sq + mu_y_sq + c1) * (var_x + var_y + c2)
    ssim = num / denom
    return torch.mean(ssim)

def PSNR(x, y):
    mse = torch.mean(torch.square(x - y))
    return 20 * torch.log10(255.0 / torch.sqrt(mse))

