import argparse
import os
import numpy as np
from time import time

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch
import torch.nn as nn

from tqdm import tqdm

from model import Generator, Discriminator
from args import get_cgan_args

def get_mnist_dataloader(args):
    # Configure data loader
    os.makedirs("../../data/mnist", exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../../data/mnist",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(args.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )
    return dataloader

def sample_image(generator, *, n_row, batches_done, args):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    FloatTensor = torch.cuda.FloatTensor if args.use_cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if args.use_cuda else torch.LongTensor

    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, args.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)


def train(generator, discriminator, adversarial_loss, train_loader, args):
    # dataloader = get_dataloader(args)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    FloatTensor = torch.cuda.FloatTensor if args.use_cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if args.use_cuda else torch.LongTensor

    with tqdm(total=len(train_loader)) as progress_bar:
        for epoch in range(args.n_epochs):
            for i, (imgs, labels) in enumerate(train_loader):

                batch_size = imgs.shape[0]

                # Adversarial ground truths
                valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
                fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

                # Configure input
                real_imgs = Variable(imgs.type(FloatTensor))
                labels = Variable(labels.type(LongTensor))

                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()

                # Sample noise and labels as generator input
                z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, args.latent_dim))))
                gen_labels = Variable(LongTensor(np.random.randint(0, args.n_classes, batch_size)))

                # Generate a batch of images
                gen_imgs = generator(z, gen_labels)

                # Loss measures generator's ability to fool the discriminator
                validity = discriminator(gen_imgs, gen_labels)
                g_loss = adversarial_loss(validity, valid)

                g_loss.backward()
                optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # Loss for real images
                validity_real = discriminator(real_imgs, labels)
                d_real_loss = adversarial_loss(validity_real, valid)

                # Loss for fake images
                validity_fake = discriminator(gen_imgs.detach(), gen_labels)
                d_fake_loss = adversarial_loss(validity_fake, fake)

                # Total discriminator loss
                d_loss = (d_real_loss + d_fake_loss) / 2

                d_loss.backward()
                optimizer_D.step()

                progress_bar.update(1)
                progress_bar.set_postfix(
                    epoch=epoch,
                    D_Loss=d_loss.item(),
                    G_Loss=g_loss.item()
                )
                batches_done = epoch * len(dataloader) + i
                if batches_done % args.sample_interval == 0:
                    sample_image(generator, n_row=10, batches_done=batches_done, args=args)
            if epoch % args.save_every == 0:
                save_loc = os.path.join(args.checkpoint_dir, '{}.last.pth'.format(args.run_name))
                torch.save({
                    'epoch': epoch,
                    'g_model_state_dict': generator.state_dict(),
                    'g_optim_state_dict': optimizer_G.state_dict(),
                    'd_model_state_dict': discriminator.state_dict(),
                    'd_optim_state_dict': optimizer_D.state_dict(),
                    'args': args
                }, save_loc)
    return generator, discriminator

def mnist_train():
    os.makedirs("images", exist_ok=True) # for outputting samples

    args = get_cgan_args()

    # Loss functions
    adversarial_loss = torch.nn.MSELoss()

    # Initialize generator and discriminator
    generator = Generator(args)
    discriminator = Discriminator(args)

    # cuda = torch.cuda.is_available()
    if args.use_cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    train_loader = get_dataloader(args)
    generator, discriminator = train(generator, discriminator, adversarial_loss, train_loader, args)

def main():
    pass

if __name__ == '__main__':
    main()
