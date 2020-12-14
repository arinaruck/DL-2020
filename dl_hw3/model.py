import torch
from torch import nn
from torch.nn import functional as F 


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, is_top=False,
               kernel_size=4, stride=2):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 
                            kernel_size=kernel_size, stride=stride, padding=1)
        self.batch_norm = nn.BatchNorm2d(out_channels) if not is_top else nn.Identity()
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)


    def forward(self, x):
        in_shape = x.shape
        x = self.batch_norm(self.conv(x))
        x = self.leaky_relu(x)
        return x


class ConvTBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dropout_p=0):
        super(ConvTBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout_p = dropout_p

    def forward(self, x):
        in_shape = x.shape
        x = self.batch_norm(self.conv(x))
        x = F.dropout(x, p=self.dropout_p)
        x = self.relu(x)
        return x


class UNetLayer(nn.Module):

    def __init__(self, config, subnet, i):

        super(UNetLayer, self).__init__()
        is_bottom = i == 0
        self.is_top = i == config.n_layers - 1

        in_up, out_up = config.up[i]
        in_down, out_down = config.down[i]
        down = ConvBlock(*config.down[i], self.is_top)
        up = nn.Sequential(
            ConvTBlock(*config.up[i], config.dropout_p[i]),
            ConvBlock(out_up, out_up, kernel_size=3, stride=1)
        )
        
        
        if is_bottom:
            self.net = nn.Sequential(down, up)
        else:
            self.net = nn.Sequential(down, subnet, up)

    def forward(self, x):
        if self.is_top:
            return self.net(x)
        else:
            return torch.cat([x, self.net(x)], 1)



class UNet(nn.Module):

    def __init__(self, config):
        super(UNet, self).__init__()
        net = UNetLayer(config, None, 0)
        for i in range(1, config.n_layers):
            net = UNetLayer(config, net, i)
        self.net = nn.Sequential(
            net,
            nn.Conv2d(config.up[-1][-1], 3, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):

    def __init__(self, config):
        super(Discriminator, self).__init__()
        n_layers = config.discriminator_layers
        self.layers = nn.ModuleList(
           [ConvBlock(*config.discriminator[i], i == 0) for i in  range(n_layers)]
        )
        chanels_out = config.discriminator[-1][-1]
        self.final = nn.Conv2d(chanels_out, 1, kernel_size=4, padding=1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.final(x)
        return x
      