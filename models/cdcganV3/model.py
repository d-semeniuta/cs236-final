"""
Adapted from CS236 starter code
"""
import numpy as np
import pdb

import torch.nn as nn
import torch.nn.functional as F
import torch

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        # self.ngpu = ngpu
        ngf = opt.n_filters
        self.main = nn.Sequential(
            # input is Z + n_classes, going into a convolution
            nn.ConvTranspose2d(opt.latent_dim+opt.n_classes, ngf * 4, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(ngf * 8),
            # nn.ReLU(True),
            # # state size. (ngf*8) x 4 x 4
            # nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 4 x 4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 8 x 8
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 16 x 16
            nn.ConvTranspose2d(ngf, opt.channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (opt.n_channels) x 32 x 32
        )

    def forward(self, noise, labels):
        input = torch.cat([noise, labels], 1)
        output = self.main(input)
        return output

class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        ndf = opt.n_filters

        self.img_conv = nn.Conv2d(opt.channels, 7 * ndf // 8, 4, 2, 1, bias=False)
        self.label_conv = nn.Conv2d(opt.n_classes, ndf // 8, 4, 2, 1, bias=False)

        self.main = nn.Sequential(
            # # input is (nc) x 32 x 32
            # nn.Conv2d(opt.channels, ndf, 4, 2, 1, bias=False),
            # nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            # nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 8),
            # nn.LeakyReLU(0.2, inplace=True),
            # # state size. (ndf*8) x 2 x 2
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input, labels):
        x = F.leaky_relu(self.img_conv(input), 0.2)
        y = F.leaky_relu(self.label_conv(labels), 0.2)
        input = torch.cat([x, y], 1)
        output = self.main(input)

        return output.view(-1, 1).squeeze(1)
