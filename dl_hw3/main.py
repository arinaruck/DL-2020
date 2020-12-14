import torch
import subprocess
from utils import seed_torch, EarlyStopping, Config
from load_data import make_loader
from train import train
from model import UNet, Discriminator


config = Config(
    [(512, 512), (1024, 512), (768, 512), (640, 256), (320, 128)],
    [(512, 512), (256, 512), (128, 256), (64, 128), (3, 64)],
    [(6, 64), (64, 128), (128, 256)],
    [0.5, 0.5, 0.5, 0, 0, 0, 0, 0],
    5, 3, 100, 150
)


def main():
	SEED=1992
	seed_torch(SEED)

	generator = UNet(config).to(config.device)
	discriminator = Discriminator(config).to(config.device)
	print(f'total parameters in generator: {count_parameters(generator)}, in discriminator: {count_parameters(discriminator)}')
	models = [generator, discriminator]

	optimizer_G = torch.optim.Adam(generator.parameters(), lr=4e-4, weight_decay=1e-6)
	optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=4e-4, weight_decay=1e-6)
	optimizers = [optimizer_G, optimizer_D]

	train_loader, val_loader = make_loader(root, modes=['train', 'val'], bs=16)
	train(config, models, optimizers, schedulers, train_loader, val_loader)


if __name__ == '__main__':
	main()
