"""
Adapted from CS236 starter code
"""
import numpy as np
import pdb

import torch.nn as nn
import torch.nn.functional as F
import torch

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class Discriminator(nn.Module):
    # initializers
    def __init__(self, opt):
    # def __init__(self, embed_dim=10, n_filters=128):
        super(Discriminator, self).__init__()
        n_filters = opt.n_filters
        self.conv1_1 = nn.Conv2d(opt.channels, n_filters//2, 4, 2, 1)
        self.conv1_2 = nn.Conv2d(opt.n_classes, n_filters//2, 4, 2, 1)
        self.conv2 = nn.Conv2d(n_filters, n_filters*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(n_filters*2)
        self.conv3 = nn.Conv2d(n_filters*2, n_filters*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(n_filters*4)
        self.conv4 = nn.Conv2d(n_filters*4, 1, 4, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, labels):
        x = F.leaky_relu(self.conv1_1(input), 0.2)
        y = F.leaky_relu(self.conv1_2(labels), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = torch.sigmoid(self.conv4(x))

        return x

class Generator(nn.Module):
    def __init__(self, opt):
    # def __init__(self, z_dim=100, embed_dim=10, n_filters=128):
        super(Generator, self).__init__()
        n_filters = opt.n_filters

        self.deconv1_1 = nn.ConvTranspose2d(opt.latent_dim, n_filters*2, 4, 1, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(n_filters*2)
        self.deconv1_2 = nn.ConvTranspose2d(opt.n_classes, n_filters*2, 4, 1, 0)
        self.deconv1_2_bn = nn.BatchNorm2d(n_filters*2)
        self.deconv2 = nn.ConvTranspose2d(n_filters*4, n_filters*2, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(n_filters*2)
        self.deconv3 = nn.ConvTranspose2d(n_filters*2, n_filters, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(n_filters)
        self.deconv4 = nn.ConvTranspose2d(n_filters, opt.channels, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, labels):
        x = F.relu(self.deconv1_1_bn(self.deconv1_1(input)))
        y = F.relu(self.deconv1_2_bn(self.deconv1_2(labels)))
        x = torch.cat([x, y], 1)
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.tanh(self.deconv4(x))

        return x
