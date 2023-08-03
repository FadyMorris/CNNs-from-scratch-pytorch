import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

from torch.utils.data import Subset


def get_data_loaders(
    data_dir, batch_size, dataset="CIFAR10", random_seed=42, valid_size=0.2, shuffle=True, subset_fraction=1, **kwargs
):
    data_loaders = dict()

    # define transforms
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    if dataset == "CIFAR10":
        cifar_dataset = datasets.CIFAR10
    elif dataset == "CIFAR100":
        cifar_dataset = datasets.CIFAR100
    else:
        raise Exception("Dataset unknown!")
        
    # load the dataset
    train_dataset = cifar_dataset(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    
    test_dataset = cifar_dataset(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )
    
    if subset_fraction > 1:
        subset_fraction = 1

    num_train = len(train_dataset)
    num_subset = int(subset_fraction * num_train)
    indices = list(range(num_subset))
    split = int(np.floor(valid_size * num_subset))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    
    data_loaders["train"] = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, **kwargs
    )

    data_loaders["valid"] = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=valid_sampler, **kwargs
    )

    data_loaders["test"] = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs
    )

    return data_loaders