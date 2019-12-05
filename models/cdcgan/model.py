
import numpy as np
import pdb

import torch.nn as nn
import torch.nn.functional as F
import torch

class Reshape(torch.nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape
    def forward(self, x):
        return x.reshape(x.size(0), *self.shape)

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()

        self.n_classes = opt.n_classes
        self.img_shape = (opt.channels, opt.img_size, opt.img_size)

        def lin_block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat))
            layers.append(nn.LeakyReLU(0.05, inplace=True))
            return layers

        conv_size = opt.img_size // 4

        self.model = nn.Sequential(
            *lin_block(opt.latent_dim + opt.n_classes, 256, normalize=False),
            *lin_block(256, 512),
            *lin_block(512, 64 * conv_size * conv_size),
            Reshape(64, conv_size, conv_size),

            nn.PixelShuffle(2),
            nn.Conv2d(64 // 4, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.PixelShuffle(2),
            nn.Conv2d(32 // 4, opt.channels, kernel_size=3, padding=1),
        )

    def forward(self, noise, labels):
        one_hot_labels = torch.eye(self.n_classes, device=labels.device)[labels]
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat([noise, one_hot_labels], -1)
        img = self.model(gen_input)
        return img


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(opt.n_classes, opt.n_classes)
        self.img_shape = (opt.channels, opt.img_size, opt.img_size)

        conv_size = opt.img_size // 4

        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(opt.channels, 32, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(0.1),

            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(0.1),

            Reshape(64 * conv_size * conv_size),
            torch.nn.Linear(64 * conv_size * conv_size, 512),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(512, 256),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(256, opt.n_classes)
        )

        # self.model = nn.Sequential(
        #     nn.Linear(opt.n_classes + int(np.prod(self.img_shape)), 512),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(512, 512),
        #     nn.Dropout(0.4),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(512, 512),
        #     nn.Dropout(0.4),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(512, 1),
        # )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        return self.net(img).gather(1, labels.unsqueeze(1)).squeeze()
        # d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        # validity = self.model(d_in)
        # return validity
