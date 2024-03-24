import torch
import torch.nn as nn

import dataset.cifar.model as cifar
import dataset.cifar10.model as cifar10
import dataset.cifar12.model as cifar12
import dataset.mnist.model as mnist
import dataset.breeds.model as breeds
import dataset.breeds2.model as breeds2
import dataset.inat.model as inat

from typing import *

def init_model(dataset : str, num_classes : List[int], device : torch.device):
    '''
    Load the correct model for each dataset.
    '''
    if dataset == 'MNIST':
        model = nn.DataParallel(mnist.CNN(num_classes)).to(device)
    elif dataset == 'CIFAR':
        model = nn.DataParallel(cifar.ResNet18(num_classes)).to(device)
    elif dataset == 'CIFAR10':
        model = nn.DataParallel(cifar10.ResNet18(num_classes)).to(device) 
    elif dataset == 'CIFAR12':
        model = nn.DataParallel(cifar12.ResNet18(num_classes)).to(device) 
    elif dataset == 'BREEDS':
        model = nn.DataParallel(breeds.ResNet18(num_classes)).to(device)
    elif dataset == 'BREEDS2':
        model = nn.DataParallel(breeds2.ResNet18(num_classes)).to(device)
    elif dataset == 'INAT':
        model = nn.DataParallel(inat.ResNet18(num_classes)).to(device)
    return model