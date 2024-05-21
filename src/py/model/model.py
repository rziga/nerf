import torch
from torch import nn
from torch.nn import functional as F


class Embedder(nn.Module):
    
    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim

    def forward(self, x):
        rets = [x]
        for i in range(self.emb_dim):
            rets.append(torch.sin(2.**i * x))
            rets.append(torch.cos(2.**i * x))
        return torch.cat(rets, dim=-3)
    
    def out_channels(self, in_channels):
        return in_channels * (2*self.emb_dim + 1)


class ResNetBlock(nn.Module):

    def __init__(self, channels, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.ReLU()

    def forward(self, x):
        skip = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return x + skip


class SuperResBlock(nn.Sequential):

    def __init__(self, in_channels, out_channels, num_convs):
        layers = [
            #nn.Upsample(scale_factor=2),
            #nn.Conv2d(in_channels, out_channels, 1)
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1)
        ]
        for _ in range(num_convs):
            layers.append(ResNetBlock(out_channels, 1))
        super().__init__(*layers)


class MobileR2L(nn.Sequential):

    def __init__(self, in_channels, emb_dim, hidden_channels, num_layers, num_sr_modules):
        super().__init__()
        self.emb = Embedder(emb_dim)
        self.head = nn.Sequential(
            nn.Conv2d(self.emb.out_channels(in_channels), hidden_channels, 1), nn.ReLU()
        )
        self.body = nn.Sequential(*[
            ResNetBlock(hidden_channels, 1)
            for _ in range(num_layers)
        ])
        self.upscale = nn.Sequential(*[
            SuperResBlock(hidden_channels//(2**i), hidden_channels//(2**(i+1)), 2)
            for i in range(num_sr_modules)
        ])
        self.tail = nn.Conv2d(hidden_channels//(2**num_sr_modules), 3, 1)
    
    def forward(self, x):
        x = self.emb(x)
        x = self.head(x)
        x = self.body(x) + x
        x = self.upscale(x)
        return F.sigmoid(self.tail(x))