import torch
import torch.nn as nn

import cifar.model as cifar
import cifar10.model as cifar10
import cifar12.model as cifar12
import mnist.model as mnist
import breeds.model as breeds
import breeds2.model as breeds2

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
    return model