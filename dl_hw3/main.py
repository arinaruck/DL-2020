import torch
import subprocess
from utils import seed_torch, EarlyStopping, Config
from load_data import make_loader
from train import train
from model import UNet


config = Config(up=[(512, 512), (1024, 512), (1024, 512), (1024, 512),  (1024, 512), (768, 512), (640, 256), (320, 128)],
    			down=[(512, 512), (512, 512), (512, 512), (512, 512), (256, 512), (128, 256), (64, 128), (3, 64)],
    			dropout_p=[0.5, 0.5, 0.5, 0, 0, 0, 0, 0],
    			n_layers=8,
    			epochs=3,
    			device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    			)

def main():
	subprocess.run(['bash', 'download.sh'])
	SEED=1992
	seed_torch(SEED)
	model = UNet(config).to(config.device)
	optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-6)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.75)
	early_stopping = EarlyStopping(checkpoint='./checkpoint', patience=5, verbose=True)
	root = 'edges2handbags'
	train_loader, val_loader = make_loader(root, modes=['train', 'val'], bs=32)
	train(config, model, optimizer, scheduler, early_stopping, train_loader, val_loader)


if __name__ == '__main__':
	main()
