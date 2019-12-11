import os

import numpy as np

from torchvision import datasets
import torchvision.transforms as transforms
import torch.utils.data

def get_mnist_dataset(data_dir, img_size, train):
    return datasets.MNIST(
        os.path.join(data_dir, 'mnist'),
        train=train,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    )

def get_cifar10_dataset(data_dir, img_size, train):
    return datasets.CIFAR10(
        os.path.join(data_dir, 'cifar10'),
        train=train,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    )

def getDataset(dataset_name, args, train=True):
    if dataset_name == 'mnist':
        return get_mnist_dataset(args.data_dir, args.img_size, train)
    elif dataset_name == 'cifar10':
        return get_cifar10_dataset(args.data_dir, args.img_size, train)
    else:
        raise ValueError('Unsupported dataset', dataset_name)

def subsampleTrainData(dataset, pct):
    np.random.seed(44)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(pct * dataset_size))
    np.random.shuffle(indices)
    subsample_indices = indices[:split]
    return torch.utils.data.Subset(dataset, subsample_indices)

def datasetToLoader(dataset, args):
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    return dataloader

def combineDatasets(set1, set2, num1, num2, args):
    set1_indices = np.random.choice(len(set1), num1, replace=False)
    set2_indices = np.random.choice(len(set2), num2, replace=False)
    set1_sample = torch.utils.data.Subset(set1, set1_indices)
    set2_sample = torch.utils.data.Subset(set2, set2_indices)

    dataset = torch.utils.data.ConcatDataset([set1_sample, set2_sample])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader
