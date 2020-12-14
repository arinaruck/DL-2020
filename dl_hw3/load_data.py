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
import random

from PIL import Image
import numpy as np

import wandb
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
import random
from tqdm.notebook import tqdm

from PIL import Image
import numpy as np

class ImgDataset(Dataset):
    
    def __init__(self, root, folder, transform=None):
        super(ImgDataset).__init__()
        self.root = os.path.join(root, folder)
        self.files = os.listdir(self.root)
        
    def __getitem__(self, idx):
        filepath = os.path.join(self.root, self.files[idx])
        img = Image.open(filepath)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
        width = img.shape[2]
        #x, y = img[:, :, width // 2 :], img[:, :, : width // 2] # for facades
        x, y = img[:, :, : width // 2], img[:, :, width // 2 :] # for maps and edges
        x, y = self.transform(x, y)
        return x, y


    def __len__(self):
        return len(self.files)

    def transform(self, x, y):
        new_h, new_w = 286, 286
        resize = transforms.Resize(size=(new_h, new_w))
        x, y = resize(x), resize(y)

        i, j, h, w = transforms.RandomCrop.get_params(
            x, output_size=(256, 256))
        x = transforms.functional.crop(x, i, j, h, w)
        y = transforms.functional.crop(y, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            x = transforms.functional.hflip(x)
            y = transforms.functional.hflip(y)

        return x, y


def make_loader(root, modes=['train', 'val', 'test'], bs=512):
    loaders = []
    for mode, batch_size in zip(modes, [bs, 2, 2]):
          dataset = ImgDataset(root, mode)
          loaders.append(DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=1, pin_memory=True))
    return loaders

