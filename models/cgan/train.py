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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
    parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
    parser.add_argument("--save_every", type=int, default=25, help="interval between saving the model")
    parser.add_argument("--checkpoint_dir", type=str, default='./checkpoints', help="Checkpoint directory")
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--use_cuda", type=bool, default=False, help="Use CUDA if available")
    opt = parser.parse_args()
    if opt.use_cuda:
        opt.use_cuda = torch.cuda.is_available()
    return opt

def get_dataloader(opt):
    # Configure data loader
    os.makedirs("../../data/mnist", exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../../data/mnist",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=opt.batch_size,
        shuffle=True,
    )
    return dataloader

def sample_image(generator, n_row, batches_done, opt):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    FloatTensor = torch.cuda.FloatTensor if opt.use_cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if opt.use_cuda else torch.LongTensor

    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)


def train(generator, discriminator, adversarial_loss, opt):
    dataloader = get_dataloader(opt)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    FloatTensor = torch.cuda.FloatTensor if opt.use_cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if opt.use_cuda else torch.LongTensor

    with tqdm(total=len(dataloader)) as progress_bar:
        for epoch in range(opt.n_epochs):
            for i, (imgs, labels) in enumerate(dataloader):

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
                z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
                gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))

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
                if batches_done % opt.sample_interval == 0:
                    sample_image(generator, n_row=10, batches_done=batches_done, opt=opt)
            if epoch % opt.save_every == 0:
                save_loc = os.path.join(opt.checkpoint_dir, opt.run_name, 'last.pth')
                torch.save({
                    'epoch': epoch,
                    'g_model_state_dict': generator.state_dict(),
                    'g_optim_state_dict': optimizer_G.state_dict(),
                    'd_model_state_dict': discriminator.state_dict(),
                    'd_optim_state_dict': optimizer_D.state_dict()
                }, save_loc)
    return generator, discriminator

def main():
    os.makedirs("images", exist_ok=True) # for outputting samples

    opt = parse_args()

    # Loss functions
    adversarial_loss = torch.nn.MSELoss()

    # Initialize generator and discriminator
    generator = Generator(opt)
    discriminator = Discriminator(opt)

    # cuda = torch.cuda.is_available()
    if opt.use_cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    generator, discriminator = train(generator, discriminator, adversarial_loss, opt)

if __name__ == '__main__':
    main()
