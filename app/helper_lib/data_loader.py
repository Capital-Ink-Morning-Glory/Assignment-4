import torch
from torchvision import datasets, transforms


def get_data_loader(data_dir, batch_size, train=True):
    # TODO: create the data loader
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.CIFAR10(root=data_dir, train=train, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    return loader
