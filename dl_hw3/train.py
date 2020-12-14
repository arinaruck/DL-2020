import torch
import torchvision.transforms as transforms
from utils import PSNR, SSIM


def train(config, models, optimizers, train_loader, val_loader=None):
    epochs = config.epochs

    gan_criterion = torch.nn.BCEWithLogitsLoss().to(config.device)
    reg_criterion = torch.nn.L1Loss().to(config.device)
    criterions = [gan_criterion, reg_criterion]

    for epoch in range(epochs):
        gen_train_loss, disc_train_loss = train_epoch(config, train_loader, models, optimizers, criterions)
        gen_val_loss, disc_val_loss = val_epoch(config, val_loader, models, criterions, epoch)

        checkpoint = f'./checkpoint.pt'
        torch.save({
            'generator_state_dict': models[0].state_dict(),
            'discriminator_state_dict': models[1].state_dict(),
            'gen_opt': optimizers[0].state_dict(),
            'disc_opt': optimizers[1].state_dict()
            }, checkpoint)


def train_epoch(config, train_loader, models, optimizers, criterions):
    G, D = models
    G.train()
    D.train()
    gen_tr_loss = 0
    disc_tr_loss = 0
    tr_steps = 0
    device = config.device

    gan_criterion, reg_criterion = criterions
    G_opt, D_opt = optimizers
    for batch in train_loader:
          x, y = batch
          x = x.to(device)
          y = y.to(device, non_blocking=True)
          fake = G(x)
          fake_c = torch.cat((x, fake), 1)
          fake_pred = D(fake_c)
          gen_loss = gan_criterion(fake_pred, torch.ones_like(fake_pred)) + config.lmbd * reg_criterion(fake, y)
          gen_tr_loss += gen_loss.item()
          G_opt.zero_grad()
          gen_loss.backward()
          torch.nn.utils.clip_grad_norm_(G.parameters(), 20.0)
          G_opt.step()

          fake = G(x)
          fake_c, real_c = torch.cat((x, fake.detach()), 1), torch.cat((x, y), 1)
          fake_pred, real_pred = D(fake_c), D(real_c)
          disc_loss = 0.75 * (gan_criterion(fake_pred, torch.zeros_like(fake_pred)) + \
                             gan_criterion(real_pred, torch.ones_like(real_pred)))
          disc_tr_loss += disc_loss.item()
          D_opt.zero_grad()
          disc_loss.backward()
          torch.nn.utils.clip_grad_norm_(D.parameters(), 20.0)
          D_opt.step()
          tr_steps += 1
    return gen_tr_loss / tr_steps, disc_tr_loss / tr_steps
                

@torch.no_grad()         
def val_epoch(config, val_loader, models, criterions, epoch):
    G, D = models
    G.train()
    D.eval()
    gen_val_loss = 0
    disc_val_loss = 0
    val_steps = 0
    device = config.device

    gan_criterion, reg_criterion = criterions

    to_img = transforms.Compose([
        transforms.Normalize(mean=(-1.0, -1.0, -1.0), std=(2.0, 2.0, 2.0)),
        transforms.ToPILImage()
    ])
    for batch in val_loader:
        x, y = batch
        x = x.to(device)
        y = y.to(device, non_blocking=True)
        fake = G(x)
        fake_c = torch.cat((x, fake), 1)
        real_c = torch.cat((x, y), 1)

        fake_pred = D(fake_c)
        gen_loss = gan_criterion(fake_pred, torch.ones_like(fake_pred)) + config.lmbd * reg_criterion(fake, y)
        gen_val_loss += gen_loss.item()

        fake_c, real_c = torch.cat((x, fake.detach()), 1), torch.cat((x, y), 1)
        fake_pred, real_pred = D(fake_c), D(real_c)
        disc_loss = 0.75 * (gan_criterion(fake_pred, torch.zeros_like(fake_pred)) + \
                             gan_criterion(real_pred, torch.ones_like(real_pred)))
        disc_val_loss += disc_loss.item()
        val_steps += 1
    b_sz = x.shape[0]
    return gen_val_loss / val_steps, disc_val_loss / val_steps
