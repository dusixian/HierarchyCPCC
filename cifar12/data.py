from typing import *

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10, CIFAR100

from hierarchy.data import Hierarchy, get_k_shot

class CIFAR12(Dataset):
    def __init__(self, train, split, difficulty):
        if difficulty == 'medium':
            if split == 'source':
                self.fine_names = ['maple_tree', 'oak_tree', 'poppy', 'rose']
                self.coarse_names = ['flowers','trees']
                self.fine_map = np.arange(4)
                self.coarse_map = np.array([1,1,0,0])
            elif split == 'target':
                self.fine_names = ['palm_tree', 'pine_tree', 'sunflower','tulip']
                self.coarse_names = ['flowers','trees']
                self.fine_map = np.arange(4)
                self.coarse_map = np.array([1,1,0,0])
            elif split == 'ood':
                self.fine_names = ['bee','beetle','ray','trout']
                self.coarse_names = ['insects','fish']
                self.fine_map = np.arange(4)
                self.coarse_map = np.array([0,0,1,1])
        elif difficulty == 'hard':
            if split == 'source':
                self.fine_names = ['aquarium_fish','dolphin','flatfish', 'whale']
                self.coarse_names = ['aquatic_mammals','fish']
                self.fine_map = np.arange(4)
                self.coarse_map = np.array([1,0,1,0])
            elif split == 'target':
                self.fine_names = ['beaver', 'otter', 'shark','trout']
                self.coarse_names = ['aquatic_mammals','fish']
                self.fine_map = np.arange(4)
                self.coarse_map = np.array([1,1,0,0])
            elif split == 'ood':
                self.fine_names = ['bee','beetle','poppy','rose']
                self.coarse_names = ['insects','flowers']
                self.fine_map = np.arange(4)
                self.coarse_map = np.array([0,0,1,1])
        elif difficulty == 'easy':
            if split == 'source':
                self.fine_names = ['bed','chair','maple_tree','oak_tree']
                self.coarse_names = ['furniture','trees']
                self.fine_map = np.arange(4)
                self.coarse_map = np.array([0,0,1,1])
            elif split == 'target':
                self.fine_names = ['couch','palm_tree', 'pine_tree','table']
                self.coarse_names = ['furniture','trees']
                self.fine_map = np.arange(4)
                self.coarse_map = np.array([0,1,1,0])
            elif split == 'ood':
                self.fine_names = ['bee','beetle','poppy','rose']
                self.coarse_names = ['insects','flowers']
                self.fine_map = np.arange(4)
                self.coarse_map = np.array([0,0,1,1])
        
        self.split = split
        self.img_size = 32
        self.channel = 3
        self.train = train
        if self.train:
            self.transform = transforms.Compose(
                                                    [transforms.RandomCrop(32, padding=4),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.RandomRotation(15),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(
                                                        mean = [0.5071, 0.4867, 0.4408], 
                                                        std = [0.2675, 0.2565, 0.2761],
                                                    )
                                                    ])      
        else:
            self.transform = transforms.Compose(
                                                    [transforms.ToTensor(),
                                                    transforms.Normalize(
                                                        mean = [0.5071, 0.4867, 0.4408], 
                                                        std = [0.2675, 0.2565, 0.2761],
                                                    )
                                                    ])


        cifar100_base = CIFAR100(root = './data',train = self.train)     
        cifar100_base.targets = np.array(cifar100_base.targets)                   
        fine_idx = [cifar100_base.class_to_idx[n] for n in self.fine_names]
        reset_idx_map = {idx:i for i,idx in enumerate(fine_idx)}

        target_idx = np.concatenate([np.argwhere(cifar100_base.targets == i).flatten() for i in fine_idx])
        self.data = cifar100_base.data[target_idx]
        self.targets = cifar100_base.targets[target_idx]
        self.targets = [reset_idx_map[i] for i in self.targets]
        
        self.mid_map = None
        self.coarsest_map = None
        self.mid2coarse = None

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index: int):
        img, target_fine = self.data[index], int(self.targets[index])

        target_coarser = -1 # dummy
        target_mid = -1 # dummy
        target_coarse = int(self.coarse_map[target_fine])
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        return img, target_coarser, target_coarse, target_mid, target_fine

def make_dataloader(num_workers : int, batch_size : int, task : str, difficulty : str = "medium") -> Tuple[DataLoader, DataLoader]:
    '''
    Creat (a subset of) train test dataloader. Train & test has the same number of classes.
    Args:
        num_workers : number of workers of train and test loader.
        batch_size : batch size of train and test loader
        task : if 'split_pretrain', dataset has 60 classes, if 'split_downstream',
        dataset has 40 classes, if 'full', dataset has 100 classes.
    '''
    def make_full_dataset() -> Tuple[Dataset, Dataset]:
        pass

    def make_subpopsplit_dataset(task : str, difficulty : str) -> Tuple[DataLoader, DataLoader]: 
        source_train = CIFAR12(train=True, split="source", difficulty=difficulty)
        source_test = CIFAR12(train=False, split="source", difficulty=difficulty)
        target_train = CIFAR12(train=True, split="target", difficulty=difficulty)
        target_test = CIFAR12(train=False, split="target", difficulty=difficulty)
        if task == 'ss':
            train_dataset, test_dataset = source_train, source_test
        elif task == 'st':
            train_dataset, test_dataset = source_train, target_test
        elif task == 'ts':
            train_dataset, test_dataset = target_train, source_test
        elif task == 'tt':
            train_dataset, test_dataset = target_train, target_test
        return train_dataset, test_dataset

    def make_outlier_dataset() -> Tuple[Dataset, Dataset]:
        train_dataset = CIFAR12(train=True, split="ood", difficulty=difficulty)
        test_dataset = CIFAR12(train=False, split="ood", difficulty=difficulty)
        return train_dataset, test_dataset

    def make_inpopsplit_dataset(train_dataset : Hierarchy, test_dataset : Hierarchy, task : str) -> Tuple[DataLoader, DataLoader]:
        pass

    if task == 'sub_split_pretrain':
        train_dataset, test_dataset = make_subpopsplit_dataset('ss', difficulty)
    elif task == 'sub_split_downstream': # where we use one shot setting for evaluation
        train_dataset, test_dataset = make_subpopsplit_dataset('tt', difficulty)
    elif task == 'sub_split_zero_shot':
        train_dataset, test_dataset = make_subpopsplit_dataset('st', difficulty)
    elif task == 'full':
        train_dataset, test_dataset = init_train_dataset, init_test_dataset
    elif task == 'outlier':
        train_dataset, test_dataset = make_outlier_dataset(difficulty)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader
