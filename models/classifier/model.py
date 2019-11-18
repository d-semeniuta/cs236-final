"""
Simple ConvNet for classifying MNIST
Adapted from own CS231N assignment

"""

import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch

class ConvBnBlock(nn.Module):
    # submodule
    # [conv-batchnorm-relu]
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size-1)//2)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(self.batchnorm(x))
        return x

# class ConvPoolBlock(nn.Module):
#     # [conv-relu-conv-relu-pool]
#     def __init__(self, in_channels, hidden_channels, out_channels, kernel_size):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size)
#         self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size)

class MNISTModel(nn.Module):
    def __init__(self, opt):
        super().__init__()
        in_channel = opt.channels
        input_size = (opt.img_size, opt.img_size)
        num_classes = opt.n_classes
        num_channels = [32, 64, 64, 64]
        self.conv_blocks = nn.ModuleList([ConvBnBlock(in_channel, num_channels[0], 5)])
        self.conv_blocks.extend([ConvBnBlock(num_channels[i], num_channels[i+1], 3)
                for i in range(len(num_channels)-1)])

        flat_size = num_channels[-1] * (input_size[0]) * (input_size[1])
        hidden_dim = num_classes*5
        self.fc1 = nn.Linear(flat_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        batch_size = x.size()[0]
        res = x
        for i, conv_block in enumerate(self.conv_blocks):
            x = conv_block(x)

        x = x.view((batch_size, -1))
        x = self.fc1(x)
        x = self.fc2(x)

        return x

def get_MNIST_model(opt):
    model = MNISTModel(opt)
    def init_weights(m):
        if type(m) == nn.Conv2d or type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight)
    model.apply(init_weights)
    return model
