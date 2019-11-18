import os
import argparse
import pdb

from models.cgan.model import Generator
from models.classifier.model import get_MNIST_model
import models.classifier.train
import models.classifier.evaluate

import numpy as np

import torch
from torch.autograd import Variable
import torch.utils.data
from torch.utils.data import DataLoader

from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image

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
    # parser.add_argument("--checkpoint_dir", type=str, default='./checkpoints', help="Checkpoint directory")
    # parser.add_argument("--run_name", required=True)
    parser.add_argument("--use_cuda", type=bool, default=False, help="Use CUDA if available")
    parser.add_argument("--load_checkpoint", type=bool, default=False, help="Run from checkpoint")
    opt = parser.parse_args()
    if opt.use_cuda:
        opt.use_cuda = torch.cuda.is_available()
    return opt

def load_generator(opt):
    checkpoint_file = './models/cgan/checkpoints/test1.last.pth'
    generator = Generator(opt)
    checkpoint = torch.load(checkpoint_file)
    generator.load_state_dict(checkpoint['g_model_state_dict'])
    generator.eval()
    return generator

def generate_images(generator, num_images, opt, out_loc='./data/mnist/generated'):
    print('Beginning to generate images in {}...'.format(out_loc))
    if os.path.exists(out_loc):
        print('Out loc already exists...')
        return
    else:
        os.makedirs(out_loc)

    FloatTensor = torch.cuda.FloatTensor if opt.use_cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if opt.use_cuda else torch.LongTensor

    batch_size = opt.batch_size
    img_per_class = num_images // opt.n_classes
    for i in range(opt.n_classes):
        labels = np.array([i for _ in range(img_per_class)])
        num_generated = 0
        class_out_loc = os.path.join(out_loc, str(i))
        os.makedirs(class_out_loc, exist_ok=True)
        while num_generated < img_per_class:
            # Sample noise
            if num_generated + batch_size >= img_per_class:
                this_batch_size = img_per_class - num_generated
            else:
                this_batch_size = batch_size
            z = Variable(FloatTensor(np.random.normal(0, 1, (this_batch_size, opt.latent_dim))))
            these_labels = Variable(LongTensor(labels[num_generated:num_generated+this_batch_size]))
            gen_imgs = generator(z, these_labels)
            for i in range(gen_imgs.size()[0]):
                out_file = os.path.join(class_out_loc, '{}.png'.format(i+num_generated))
                save_image(gen_imgs[i,:,:,:], out_file)
            num_generated += len(these_labels)
    print('Done generating images!')

def get_val_loader(opt):
    return torch.utils.data.DataLoader(
        datasets.MNIST(
            "./data/mnist",
            train=False,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=opt.batch_size,
        shuffle=True,
    )

def get_train_loader(num_true, num_gen, opt, gen_root='./data/mnist/generated'):
    transform_list = [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    true = datasets.MNIST(
        "./data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(transform_list),
    )
    gen = datasets.ImageFolder(
        gen_root,
        transform=transforms.Compose([transforms.Grayscale(num_output_channels=1)] + transform_list)
    )
    true_sample_indices = np.random.choice(len(true), num_true, replace=False)
    gen_sample_indices = np.random.choice(len(gen), num_gen, replace=False)
    true_sample = torch.utils.data.Subset(true, true_sample_indices)
    gen_sample = torch.utils.data.Subset(gen, gen_sample_indices)

    dataset = torch.utils.data.ConcatDataset([true_sample, gen_sample])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
    return dataloader

def train_test_classifier(train_loader, val_loader, opt):
    classifier = get_MNIST_model(opt)
    classifier, train_acc = models.classifier.train.train_model(classifier, train_loader, opt)
    print('Done training, now evaluating...')
    val_acc = models.classifier.evaluate.evaluate_model(classifier, val_loader, opt)
    return train_acc, val_acc


def main():
    true_data_sizes = [0, 256, 512, 1024, 2048]
    generated_data_sizes = [0, 256, 512, 1024, 2048]
    opt = parse_args()
    generator = load_generator(opt)
    if opt.use_cuda:
        generator.cuda()
    generate_images(generator, max(generated_data_sizes), opt)
    val_loader = get_val_loader(opt)
    for generated_data_size in generated_data_sizes:
        for true_data_size in true_data_sizes:
            if true_data_size == 0 and generated_data_size == 0:
                continue
            print('Running with {} true images and {} generated images'.format(true_data_size, generated_data_size))
            train_loader = get_train_loader(true_data_size, generated_data_size, opt)
            train_acc, val_acc = train_test_classifier(train_loader, val_loader, opt)
            print('Final train accuracy: {:.3f}\tval accuracy: {:.3f}'.format(train_acc, val_acc))


if __name__ == '__main__':
    main()
