import torchvision
import torchvision.transforms as transforms
import torch
from args import *


class AddGaussianNoise(object):
    def __init__(self, std=0.0):
        self.std = std

    def __call__(self, tensor):
        if self.std <= 0:
            return tensor
        noise = torch.randn_like(tensor) * self.std
        return torch.clamp(tensor + noise, 0.0, 1.0)


image_size = args.image_size
train_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((image_size, image_size)),
        torchvision.transforms.ToTensor(),
        AddGaussianNoise(std=args.input_noise_std),
    ]
)
test_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((image_size, image_size)),
        torchvision.transforms.ToTensor(),
    ]
)

trainset = torchvision.datasets.MNIST(
    root=".//data//",
    transform=train_transform,
    train=True,
    download=True,
)
testset = torchvision.datasets.MNIST(
    root=".//data//",
    transform=test_transform,
    train=False,
    download=True,
)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
