import os
import argparse
import pdb

# from models.cdcgan.model import Generator, Discriminator
# from models.cdcgan.train import train as train_gen
# from models.cdcganV2.model import Generator, Discriminator
# from models.cdcganV2.train import train as train_gen
from models.cdcganV3.model import Generator, Discriminator
from models.cdcganV3.train import train as train_gen
from models.classifier.model import get_MNIST_model
import models.classifier.train
import models.classifier.evaluate

import util.data
import util.utils

import numpy as np

import torch
from torch.autograd import Variable
import torch.utils.data
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

from torchvision import datasets
import torchvision.transforms as transforms
import torchvision.models
from torchvision.utils import save_image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=300, help="number of epochs of training for generator")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--n_filters", type=int, default=128, help="number of filters in G convs")
    parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
    parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
    parser.add_argument("--save_every", type=int, default=25, help="interval between saving the model")
    # parser.add_argument("--checkpoint_dir", type=str, default='./checkpoints', help="Checkpoint directory")
    # parser.add_argument("--run_name", required=True)
    parser.add_argument("--no_cuda", action='store_true', help="Use CUDA if available")
    parser.add_argument("--gen_once", action='store_true', help="Generate images from full training set")
    parser.add_argument("--restore_gen", action='store_true', help="Restore generator")
    parser.add_argument("--load_checkpoint", action='store_true', help="Run from checkpoint")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to run on")
    parser.add_argument("--experiment_dir", type=str, required=True, help="Where to save all outputs")
    parser.add_argument("--data_dir", type=str, default='./data', help="Data directory")
    parser.add_argument("--n_epochs_class", type=int, default=50, help="number of epochs of training for classifier")
    parser.add_argument("--save_every_class", type=int, default=10, help="interval between saving the classifier")
    parser.add_argument("--eval_every_class", type=int, default=10, help="interval between evaling the classifier")
    parser.add_argument("--classifier", type=str, default='vgg16_bn', help="torch classifier to use")
    args = parser.parse_args()
    if not args.no_cuda:
        args.use_cuda = torch.cuda.is_available()
        args.device = torch.device('cuda') if args.use_cuda else torch.device('cpu')
    else:
        args.use_cuda = False
    if not os.path.isdir(args.experiment_dir):
        os.makedirs(args.experiment_dir)
    torch.save(args, os.path.join(args.experiment_dir, 'args.pth'))
    return args

def train_test_classifier(classifier, optimizer, train_loader, val_loader, ckpt_dir, writer, args):
    args.checkpoint_dir = ckpt_dir
    classifier, train_acc = models.classifier.train.train_model(classifier, optimizer, train_loader, args, writer, epochs=args.n_epochs_class)
    print('Done training, now evaluating...')
    val_acc = models.classifier.evaluate.evaluate_model(classifier, val_loader, args)
    return train_acc, val_acc

def load_generator(save_dir, args):
    save_loc = os.path.join(save_dir, 'gen.last.pth')
    train_dict = torch.load(save_loc)
    generator = Generator(args)
    generator.load_state_dict(train_dict['g_model_state_dict'])
    generator.to(args.device)
    return generator

def run_generator(train_data, gen_dir, args):
    out_loc = os.path.join(gen_dir, 'gen_imgs')
    if os.path.exists(out_loc):
        print('Images already generated...')
        gen_data = datasets.ImageFolder(
            out_loc,
            transform=transforms.Compose(
                [transforms.Resize(args.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            )
        )
        return gen_data
    if args.restore_gen:
        print('Restoring generator from ckpt...')
        generator = load_generator(gen_dir, args)
    else:
        print('Training generator...')
        train_loader = util.data.datasetToLoader(train_data, args)
        args.checkpoint_dir = gen_dir
        generator, discriminator = Generator(args), Discriminator(args)
        # loss_fn = torch.nn.MSELoss()
        loss_fn = torch.nn.BCELoss()
        if args.use_cuda:
            generator.cuda()
            discriminator.cuda()
            loss_fn.cuda()
        # train generator
        generator, discriminator = train_gen(generator, discriminator, loss_fn, train_loader, args)
        print('Generator trained, now generating images in {}'.format(out_loc))

    # gen images
    num_images = len(train_data)
    util.utils.generateImages(generator, num_images, args, out_loc)
    gen_data = datasets.ImageFolder(
        out_loc,
        transform=transforms.Compose(
            [transforms.Resize(args.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )
    )
    return gen_data

def get_classifier(classifier):
    if classifier == 'vgg19_bn':
        model = torchvision.models.vgg19_bn()
    elif classifier == 'vgg16_bn':
        model = torchvision.models.vgg16_bn()
    else:
        raise ValueError('Unsupported classifier', classifier)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-2)
    return model, optimizer

def gen_imgs_once(args):
    train_data = util.data.getDataset(args.dataset, args, train=True)
    num_train_sample = len(train_data)
    val_data = util.data.getDataset(args.dataset, args, train=False)
    val_loader = util.data.datasetToLoader(val_data, args)

    pcts = [0.2, 0.4, 0.6, 0.8, 1.0]
    log_file = os.path.join(args.experiment_dir, 'res.txt')
    tb_dir = os.path.join(args.experiment_dir, 'tb')
    print('Training generator on {} images...'.format(num_true))
    gen_data = run_generator(train_data, args.experiment_dir, args)

    if os.path.isfile(log_file):
        os.remove(log_file)
    with open(log_file, 'a') as logging:
        for pct_true in pcts:
            num_true = int(pct_true * num_train_sample)
            true_data = util.data.subsampleTrainData(train_data, pct_true)
            true_dir = os.path.join(args.experiment_dir, '{}pct_true'.format(int(pct_true * 100)))
            os.makedirs(true_dir, exist_ok=True)
            for pct_gen in [0.0] + pcts:
                num_gen = int(pct_gen * num_true)
                print('Running with {} true images and {} generated images'.format(num_true, num_gen))
                train_loader = util.data.combineDatasets(true_data, gen_data, num_true, num_gen, args)
                this_classifier, this_optimizer = get_classifier(args.classifier)
                gen_dir = os.path.join(true_dir, '{}pct_gen'.format(int(pct_gen * 100)))
                os.makedirs(gen_dir, exist_ok=True)
                tb_writer_dir = os.path.join(tb_dir, '{}pct_true'.format(int(pct_true * 100)), '{}pct_gen'.format(int(pct_gen * 100)))
                writer = SummaryWriter(tb_writer_dir)
                train_acc, val_acc = train_test_classifier(this_classifier, this_optimizer, train_loader, val_loader, gen_dir, writer, args)
                print('Final train accuracy: {:.3f}\tval accuracy: {:.3f}'.format(train_acc, val_acc))
                print('')
                logging.write('Trained on {} true, {} gen\n'.format(num_true, num_gen))
                logging.write('  Train acc: {:.3f}\tval accuracy: {:.3f}\n\n'.format(train_acc, val_acc))

def gen_imgs_per_train(args):
    train_data = util.data.getDataset(args.dataset, args, train=True)
    num_train_sample = len(train_data)
    val_data = util.data.getDataset(args.dataset, args, train=False)
    val_loader = util.data.datasetToLoader(val_data, args)

    true_pcts = [1.0]
    gen_pcts = [0.0, 0.25, 0.5, 0.75, 1.0]
    log_file = os.path.join(args.experiment_dir, 'res.txt')
    tb_dir = os.path.join(args.experiment_dir, 'tb')
    if os.path.isfile(log_file):
        os.remove(log_file)
    with open(log_file, 'a') as logging:
        for pct_true in true_pcts:
            num_true = int(pct_true * num_train_sample)
            true_data = util.data.subsampleTrainData(train_data, pct_true)
            true_dir = os.path.join(args.experiment_dir, '{}pct_true'.format(int(pct_true * 100)))
            os.makedirs(true_dir, exist_ok=True)
            print('Training generator on {} images...'.format(num_true))
            gen_data = run_generator(true_data, true_dir, args)
            for pct_gen in gen_pcts:
                num_gen = int(pct_gen * num_true)
                print('Running with {} true images and {} generated images'.format(num_true, num_gen))
                train_loader = util.data.combineDatasets(true_data, gen_data, num_true, num_gen, args)
                this_classifier, this_optimizer = get_classifier(args.classifier)
                gen_dir = os.path.join(true_dir, '{}pct_gen'.format(int(pct_gen * 100)))
                os.makedirs(gen_dir, exist_ok=True)
                tb_writer_dir = os.path.join(tb_dir, '{}pct_true'.format(int(pct_true * 100)), '{}pct_gen'.format(int(pct_gen * 100)))
                writer = SummaryWriter(tb_writer_dir)
                train_acc, val_acc = train_test_classifier(this_classifier, this_optimizer, train_loader, val_loader, gen_dir, writer, args)
                print('Final train accuracy: {:.3f}\tval accuracy: {:.3f}'.format(train_acc, val_acc))
                print('')
                logging.write('Trained on {} true, {} gen\n'.format(num_true, num_gen))
                logging.write('  Train acc: {:.3f}\tval accuracy: {:.3f}\n\n'.format(train_acc, val_acc))


def main():
    args = parse_args()

    if args.gen_once:
        gen_imgs_once(args)
    else:
        gen_imgs_per_train(args)

if __name__ == '__main__':
    main()
