import os, math

import numpy as np

from torch.autograd import Variable

import torch.utils.data
from torchvision import datasets
from torchvision.utils import save_image
import torchvision.transforms as transforms

def get_true_mnist():
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../data/mnist",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(32), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=64,
        shuffle=False,
    )
    for i, (imgs, labels) in enumerate(dataloader):
        save_image(imgs.data, "true_mnist.png", nrow=8, normalize=True)
        return

def get_true_cifar10():
    dataloader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            "../data/cifar10",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(32), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=100,
        shuffle=False,
    )
    for i, (imgs, labels) in enumerate(dataloader):
        save_image(imgs.data, "true_cifar10.png", nrow=10, normalize=True)
        return

def get_row_cifar10():
    dataloader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            "../data/cifar10",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(32), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=100,
        shuffle=False,
    )
    found_labels = set()
    imgs_to_save = torch.zeros(10, 3, 32, 32)
    for i, (imgs, labels) in enumerate(dataloader):
        for j, label in enumerate(labels.split(1)):
            label = label.item()
            if label not in found_labels:
                found_labels.add(label)
                imgs_to_save[label,:,:,:] = imgs[j,:,:,:]
            if len(found_labels) == 10:
                break
        else:
            continue
        break

    save_image(imgs_to_save.data, "row_cifar10.png", nrow=10, normalize=True)
    return
get_row_cifar10()


def generateImages(generator, num_images, args, out_loc):
    print('Beginning to generate images in {}...'.format(out_loc))
    if os.path.exists(out_loc):
        print('Out loc already exists...')
        return
    else:
        os.makedirs(out_loc)

    FloatTensor = torch.cuda.FloatTensor if args.use_cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if args.use_cuda else torch.LongTensor
    generator.eval()
    batch_size = args.batch_size
    img_per_class = math.ceil(num_images / args.n_classes)
    with torch.no_grad():
        for i in range(args.n_classes):
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
                # z = Variable(FloatTensor(np.random.normal(0, 1, (this_batch_size, args.latent_dim))))
                z = Variable(FloatTensor(np.random.normal(0, 1, (this_batch_size, args.latent_dim, 1, 1))))
                these_labels = Variable(LongTensor(labels[num_generated:num_generated+this_batch_size]))
                one_hot_labels = torch.eye(args.n_classes, device=these_labels.device)[these_labels]
                one_hot_labels = one_hot_labels.unsqueeze(dim=-1).unsqueeze(dim=-1)
                # gen_imgs = generator(z, these_labels)
                gen_imgs = generator(z, one_hot_labels)
                for i in range(gen_imgs.size()[0]):
                    out_file = os.path.join(class_out_loc, '{}.png'.format(i+num_generated))
                    save_image(gen_imgs[i,:,:,:], out_file)
                num_generated += len(these_labels)
    print('Done generating images!')
