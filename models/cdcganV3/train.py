import argparse
import os, pdb
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

from model import Generator, Discriminator, weight_init
# from models.cdcgan.model import Generator, Discriminator

def get_default_args():
    parser = argparse.ArgumentParser('CDCGAN Training')
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
    parser.add_argument("--n_filters", type=int, default=128, help="number of filters in G convs")
    parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
    parser.add_argument("--save_every", type=int, default=25, help="interval between saving the model")
    parser.add_argument("--checkpoint_dir", type=str, default='./checkpoints', help="Checkpoint directory")
    parser.add_argument("--run_name", required=True, type=str)
    parser.add_argument("--use_cuda", action='store_true', help="Use CUDA if available")
    parser.add_argument("--load_checkpoint", action='store_true', help="Run from checkpoint")
    args = parser.parse_args()
    if args.use_cuda:
        args.use_cuda = torch.cuda.is_available()
        print('using cuda') if args.use_cuda else print('not using cuda')
    return args

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
    device = torch.device('cuda') if args.use_cuda else torch.device('cpu')
    generator.eval()
    with torch.no_grad():
        # Sample noise
        z = torch.randn(n_row**2, args.latent_dim, 1, 1).to(device)
        # Get labels ranging from 0 to n_classes for n rows
        labels = np.array([num for _ in range(n_row) for num in range(n_row)])
        labels = Variable(LongTensor(labels))
        one_hot_labels = torch.eye(args.n_classes, device=labels.device)[labels]
        one_hot_labels = one_hot_labels.unsqueeze(dim=-1).unsqueeze(dim=-1) # [batch_size, n_classes, 1, 1]
        # pdb.set_trace()
        gen_imgs = generator(z, one_hot_labels)
        out_dir = os.path.join(args.checkpoint_dir, 'samples')
        os.makedirs(out_dir, exist_ok=True)
        out_loc = os.path.join(out_dir, '{}.png'.format(batches_done))
        save_image(gen_imgs.data, out_loc, nrow=n_row, normalize=True)


def train(generator, discriminator, adversarial_loss, train_loader, args):
    # dataloader = get_dataloader(args)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    FloatTensor = torch.cuda.FloatTensor if args.use_cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if args.use_cuda else torch.LongTensor

    device = torch.device('cuda') if args.use_cuda else torch.device('cpu')

    with tqdm(total=len(train_loader)*args.n_epochs) as progress_bar:
        for epoch in range(args.n_epochs):
            generator.train()
            discriminator.train()
            for i, (imgs, labels) in enumerate(train_loader):

                batch_size = imgs.shape[0]

                # Adversarial ground truths
                # valid = Variable(FloatTensor(batch_size).fill_(1.0), requires_grad=False)
                valid = torch.ones((batch_size,1), device=device)
                # fake = Variable(FloatTensor(batch_size).fill_(0.0), requires_grad=False)
                fake = torch.zeros((batch_size,1), device=device)

                # Configure input
                # real_imgs = Variable(imgs.type(FloatTensor))
                real_imgs = imgs.to(device)
                # labels = Variable(labels.type(LongTensor))
                labels = labels.to(device)
                one_hot_labels = torch.eye(args.n_classes, device=labels.device)[labels]
                one_hot_labels = one_hot_labels.unsqueeze(dim=-1).unsqueeze(dim=-1) # [batch_size, n_classes, 1, 1]
                one_hot_labels_exp = one_hot_labels.expand(-1, -1, args.img_size, args.img_size)
                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()

                # Sample noise and labels as generator input
                # z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, args.latent_dim))))
                z = torch.randn(batch_size, args.latent_dim, 1, 1).to(device)

                # gen_labels = Variable(FloatTensor(np.random.randint(0, args.n_classes, batch_size)))
                # gen_labels = labels

                # Generate a batch of images
                gen_imgs = generator(z, one_hot_labels)

                # Loss measures generator's ability to fool the discriminator
                validity = discriminator(gen_imgs, one_hot_labels_exp)
                validity = validity.squeeze().unsqueeze(-1)
                # pdb.set_trace()
                g_loss = adversarial_loss(validity, valid)

                g_loss.backward()
                optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # Loss for real images
                validity_real = discriminator(real_imgs, one_hot_labels_exp).squeeze().unsqueeze(-1)
                d_real_loss = adversarial_loss(validity_real, valid)

                # Loss for fake images
                validity_fake = discriminator(gen_imgs.detach(), one_hot_labels_exp).squeeze().unsqueeze(-1)
                d_fake_loss = adversarial_loss(validity_fake, fake)

                # Total discriminator loss
                d_loss = (d_real_loss + d_fake_loss) / 2

                d_loss.backward()
                optimizer_D.step()

                progress_bar.update(1)
                progress_bar.set_postfix(
                    epoch=epoch+1,
                    D_Loss=d_loss.item(),
                    G_Loss=g_loss.item()
                )
                batches_done = epoch * len(train_loader) + i
                if batches_done % args.sample_interval == 0:
                    sample_image(generator, n_row=10, batches_done=batches_done, args=args)
                    generator.train()
            if epoch % args.save_every == 0:
                save_loc = os.path.join(args.checkpoint_dir, 'gen.last.pth')
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

    args = get_default_args()

    # Loss functions
    adversarial_loss = nn.BCELoss()

    # Initialize generator and discriminator
    generator = Generator(args)
    discriminator = Discriminator(args)
    generator.apply(weight_init)
    discriminator.apply(weight_init)

    # cuda = torch.cuda.is_available()
    if args.use_cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    train_loader = get_mnist_dataloader(args)
    generator, discriminator = train(generator, discriminator, adversarial_loss, train_loader, args)

def main():
    mnist_train()
    pass

if __name__ == '__main__':
    main()
