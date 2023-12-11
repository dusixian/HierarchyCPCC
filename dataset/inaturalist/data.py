from typing import *

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import INaturalist

from hierarchy.data import Hierarchy, get_k_shot

class INaturalist12(Dataset):
    def __init__(self, train, split):
        if split == 'source':
            self.fine_id = [9979, 9980, 9984, 9985]
            self.fine_names = ['atropurpurea', 'glabella',
                                'cretica', 'macilenta']
            self.coarse_names = ['Pellaea','Pteris']
            self.fine_map = np.arange(4)
            self.coarse_map = np.array([0,0,1,1])
         elif split == 'target':
            self.fine_id = [9981, 9982, 9986, 9987]
            self.fine_names = ['mucronata', 'rotundifolia', 
                                'tremula','vittata']
            self.coarse_names = ['Pellaea','Pteris']
            self.fine_map = np.arange(4)
            self.coarse_map = np.array([0,0,1,1])
        elif split == 'ood':
            self.fine_id = [9976, 9977, 9969, 9970]
            self.fine_names = ['aurea','parryi','hispidulum','jordanii']
            self.coarse_names = ['Myriopteris','Adiantum']
            self.fine_map = np.arange(4)
            self.coarse_map = np.array([0,0,1,1])
        self.split = split
        self.img_size = 224
        self.channel = 3
        self.train = train
        # transform refers to https://github.com/naver-ai/cmo/blob/main/inat18_train.py#L163-L175
        if self.train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.466, 0.471, 0.380], [0.195, 0.194, 0.192])
            ])  
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.466, 0.471, 0.380], [0.195, 0.194, 0.192])
            ])


        if self.train:
            INaturalist_base = INaturalist(root = '/data/common/iNaturalist/train_mini', version = '2021_train_mini', target_type = ['full'], train = self.train) 
        else:
            INaturalist_base = INaturalist(root = '/data/common/iNaturalist/val', version = '2021_valid', target_type = ['full'], train = self.train)
        INaturalist_base.targets = np.array(INaturalist_base.all_categories)                   
        # fine_idx = [INaturalist_base.categories_index['full'][n] for n in self.fine_names]
        reset_idx_map = {idx:i for i,idx in enumerate(self.fine_id)}

        all_image_indices = [idx for idx, (cat_id, _) in enumerate(INaturalist_base.index)]
        selected_image_indices = [idx for idx in all_image_indices if INaturalist_base.index[idx][0] in self.fine_id]
        self.data = [INaturalist_base[idx][0] for idx in selected_image_indices]
        self.targets = [reset_idx_map[INaturalist_base.index[idx][0]] for idx in selected_image_indices]

        # target_idx = np.concatenate([np.argwhere(INaturalist_base.targets == i).flatten() for i in fine_idx])
        # self.data = INaturalist_base.data[target_idx]
        # self.targets = INaturalist_base.targets[target_idx]
        # self.targets = [reset_idx_map[i] for i in self.targets]
        
        self.mid_map = None
        self.coarsest_map = None
        self.mid2coarse = None

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index: int):
        img = self.data[index]
        target_fine = int(self.targets[index])

        target_coarser = -1 # dummy
        target_mid = -1 # dummy
        target_coarse = int(self.coarse_map[target_fine])
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        return img, target_coarser, target_coarse, target_mid, target_fine

def make_dataloader(num_workers : int, batch_size : int, task : str) -> Tuple[DataLoader, DataLoader]:
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

    def make_subpopsplit_dataset(task : str) -> Tuple[DataLoader, DataLoader]: 
        source_train = CIFAR12(train=True, split="source")
        source_test = CIFAR12(train=False, split="source")
        target_train = CIFAR12(train=True, split="target")
        target_test = CIFAR12(train=False, split="target")
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
        train_dataset = CIFAR12(train=True, split="ood")
        test_dataset = CIFAR12(train=False, split="ood")
        return train_dataset, test_dataset

    def make_inpopsplit_dataset(train_dataset : Hierarchy, test_dataset : Hierarchy, task : str) -> Tuple[DataLoader, DataLoader]:
        pass

    if task == 'sub_split_pretrain':
        train_dataset, test_dataset = make_subpopsplit_dataset('ss')
    elif task == 'sub_split_downstream': # where we use one shot setting for evaluation
        train_dataset, test_dataset = make_subpopsplit_dataset('tt')
    elif task == 'sub_split_zero_shot':
        train_dataset, test_dataset = make_subpopsplit_dataset('st')
    elif task == 'full':
        train_dataset, test_dataset = init_train_dataset, init_test_dataset
    elif task == 'outlier':
        train_dataset, test_dataset = make_outlier_dataset()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader
