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
