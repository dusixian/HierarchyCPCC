from typing import *

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import INaturalist

from tqdm import tqdm

# from hierarchy.data import Hierarchy, get_k_shot

import os

class HierarchyINaturalist(INaturalist):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)                
        self.species_label = np.array(self.categories_map)
        self.targets = [self.index[idx][0] for idx in range(len(self.index))] # species
        self.fine_map = np.array(range(len(self.species_label)))
        self.fine_names = np.array([str(x) for x in np.unique(self.fine_map)])
        self.coarse_map = np.array([d['order'] for d in self.species_label])
        self.coarse_names = np.array([str(i) for i in range(len(np.unique(self.coarse_map)))])
        self.img_size = 224
        self.channel = 3

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        cat_id, fname = self.index[index]
        img = Image.open(os.path.join(self.root, self.all_categories[cat_id], fname))

        target = self.targets

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        target_fine = target
        target_coarsest = target_coarse = target_mid = -1
        return img, target_coarsest, target_coarse, target_mid, target_fine

class HierarchyINaturalistSubset(HierarchyINaturalist):
    def __init__(self, indices : List[int], fine_classes : List[int], *args, **kw):
        super(HierarchyINaturalistSubset, self).__init__(*args, **kw)
        self.index = [self.index[i] for i in indices]
        
        old_targets = list(np.array(self.targets)[indices]) # old fine targets, sliced
        fine_classes = np.array(sorted(fine_classes)) # fine class id in HierarchyCIFAR index, range = 0-19
        self.fine_names = [self.fine_names[i] for i in fine_classes] # old fine names, sliced
        self.fine_map = np.array(range(len(fine_classes))) # number of fine classes subset
        self.targets = [list(fine_classes).index(i) for i in old_targets] # reset fine target id from 0

        # reset other hierarchy level index, from 0
        old_coarse_map = np.array([self.coarse_map[i] for i in fine_classes]) # subset of coarse fine map
        coarse_classes = np.unique(old_coarse_map)
        self.coarse_names = [self.coarse_names[cls] for cls in coarse_classes]
        self.coarse_map = np.array([list(coarse_classes).index(i) for i in old_coarse_map]) # argsort

        old_mid_map = None
        mid_classes = None
        self.mid_names = None
        self.mid_map = None

        old_coarsest_map = None
        coarsest_classes = None
        self.coarsest_names = None
        self.coarsest_map = None

    def __len__(self):
        return len(self.index)
    

def make_dataloader(num_workers : int, batch_size : int, task : str) -> Tuple[DataLoader, DataLoader]:
    def make_full_dataset() -> Tuple[Dataset, Dataset]:
        train_dataset = HierarchyINaturalist(root = '/data/common/iNaturalist/inat-mini-val',
                                                  version = '2021_train_mini', target_type = ['full'],                        
                                                transform = transforms.Compose([
                                                transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.466, 0.471, 0.380], [0.195, 0.194, 0.192])
                                            ]))
        # not augment test set except of normalization
        test_dataset = HierarchyINaturalist(
                                                  root = '/data/common/iNaturalist/inat-mini-val',
                                                  version = '2021_valid', target_type = ['full'],                        
                                                transform = transforms.Compose([
                                                transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.466, 0.471, 0.380], [0.195, 0.194, 0.192])
                                            ]))
        return train_dataset, test_dataset

    def make_subpopsplit_dataset(train_dataset, test_dataset, task : str) -> Tuple[DataLoader, DataLoader]: 
        train_all_fine_map = train_dataset.targets
        train_sortidx = np.argsort(train_all_fine_map)
        train_sorted_fine_map = np.array(train_all_fine_map)[train_sortidx]
        # a dictionary that maps coarse id to a list of fine id
        target_fine_dict = {i:[] for i in range(len(train_dataset.coarse_names))}
        idx_train_source = [] # index of image (based on original Pytorch CIFAR dataset) that sends to source
        idx_train_target = []
        f2c = dict(zip(range(len(train_dataset.fine_names)),train_dataset.coarse_map))
        
        # finish c2f, know each coarse -> which set of fine class
        # keep 60% in source, 40% in target
        c2f = {}

        for fine_id, coarse_id in zip(train_dataset.fine_map, train_dataset.coarse_map):
            if coarse_id not in c2f:
                c2f[coarse_id] = [fine_id]
            else:
                c2f[coarse_id].append(fine_id)
        coarse_counts = {coarse_id: len(fine_ids) for coarse_id, fine_ids in c2f.items()}

        # remove coarse and fine id where number of fine classes = 1, 2,
        # since 40-60 split will cause empty source/train coarse id
        small_coarse_set = {key for key, count in coarse_counts.items() if count <= 2}
        
        for idx in tqdm(range(len(train_sortidx)), desc="splitting indices"): # loop thru all argsort fine
            coarse_id = f2c[train_sorted_fine_map[idx]]
            if coarse_id not in small_coarse_set:
                target_fine_dict[coarse_id].append(train_sorted_fine_map[idx])
                if len(set(target_fine_dict[coarse_id])) <= int(0.4 * coarse_counts[coarse_id]): 
                    # 40% to few shot second stage
                    idx_train_target.append(train_sortidx[idx])
                else:
                    # if we have seen the third type of fine class
                    # since sorted, we have checked all images of the first
                    # two types. For the rest of images,
                    # send to source
                    idx_train_source.append(train_sortidx[idx])
        
        for key in target_fine_dict:
            target = target_fine_dict[key] # fine label id for [coarse]
            d = {x: True for x in target}
            target_fine_dict[key] = list(d.keys())[:int(0.4 * coarse_counts[key])] # all UNIQUE fine classes sent to target for [coarse]

        target_fine_cls = [] # all 40% fine classes sent to target
        for key in target_fine_dict:
            target_fine_cls.extend(target_fine_dict[key])

        test_all_fine_map = test_dataset.targets
        idx_test_source = []
        idx_test_target = []
        for idx in range(len(test_all_fine_map)):
            fine_id = test_all_fine_map[idx]
            coarse_id = f2c[fine_id]
            if fine_id in target_fine_dict[coarse_id]:
                idx_test_target.append(idx)
            else:
                idx_test_source.append(idx)

        source_fine_cls = list(set(range(len(train_dataset.fine_names))) - set(target_fine_cls))
        source_train = HierarchyINaturalistSubset(idx_train_source, source_fine_cls, 
                                                  root = '/data/common/iNaturalist/inat-mini-val',
                                                  version = '2021_train_mini', target_type = ['full'],                        
                                                transform = transforms.Compose([
                                                transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.466, 0.471, 0.380], [0.195, 0.194, 0.192])
                                            ]))
        source_test = HierarchyINaturalistSubset(idx_test_source, source_fine_cls, 
                                                  root = '/data/common/iNaturalist/inat-mini-val',
                                                  version = '2021_valid', target_type = ['full'],                        
                                                transform = transforms.Compose([
                                                transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.466, 0.471, 0.380], [0.195, 0.194, 0.192])
                                            ]))
        target_train = HierarchyINaturalistSubset(idx_train_target, target_fine_cls, 
                                                  root = '/data/common/iNaturalist/inat-mini-val',
                                                  version = '2021_train_mini', target_type = ['full'],                        
                                                transform = transforms.Compose([
                                                transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.466, 0.471, 0.380], [0.195, 0.194, 0.192])
                                            ]))
        target_test = HierarchyINaturalistSubset(idx_test_target, target_fine_cls, 
                                                  root = '/data/common/iNaturalist/inat-mini-val',
                                                  version = '2021_valid', target_type = ['full'],                        
                                                transform = transforms.Compose([
                                                transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.466, 0.471, 0.380], [0.195, 0.194, 0.192])
                                            ]))
        if task == 'ss':
            train_dataset, test_dataset = source_train, source_test
        elif task == 'st':
            train_dataset, test_dataset = source_train, target_test
        elif task == 'ts':
            train_dataset, test_dataset = target_train, source_test
        elif task == 'tt':
            train_dataset, test_dataset = target_train, target_test
        return train_dataset, test_dataset

    def make_inpopsplit_dataset(train_dataset, test_dataset, task : str) -> Tuple[DataLoader, DataLoader]:
        pass

    def make_outlier_dataset() -> Tuple[Dataset, Dataset]:
        pass

    init_train_dataset, init_test_dataset = make_full_dataset()
    if task == 'sub_split_pretrain':
        train_dataset, test_dataset = make_subpopsplit_dataset(init_train_dataset, init_test_dataset, 'ss')
    elif task == 'sub_split_downstream': # where we use one shot setting for evaluation
        train_dataset, test_dataset = make_subpopsplit_dataset(init_train_dataset, init_test_dataset, 'tt')
    elif task == 'sub_split_zero_shot':
        train_dataset, test_dataset = make_subpopsplit_dataset(init_train_dataset, init_test_dataset, 'st')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader



# if __name__ == '__main__':
#     num_workers = 1
#     batch_size = 128
#     task = ['sub_split_pretrain','sub_split_downstream','sub_split_zero_shot']
#     for t in task:
#         print(t)
#         train_loader, test_loader = make_dataloader(num_workers, batch_size, t)

# class INaturalist12(Dataset):
#     def __init__(self, train, split):
#         if split == 'source':
#             self.fine_id = [9979, 9980, 9984, 9985]
#             self.fine_names = ['atropurpurea', 'glabella',
#                                 'cretica', 'macilenta']
#             self.coarse_names = ['Pellaea','Pteris']
#             self.fine_map = np.arange(4)
#             self.coarse_map = np.array([0,0,1,1])
#         elif split == 'target':
#             self.fine_id = [9981, 9982, 9986, 9987]
#             self.fine_names = ['mucronata', 'rotundifolia', 
#                                 'tremula','vittata']
#             self.coarse_names = ['Pellaea','Pteris']
#             self.fine_map = np.arange(4)
#             self.coarse_map = np.array([0,0,1,1])
#         elif split == 'ood':
#             self.fine_id = [9976, 9977, 9969, 9970]
#             self.fine_names = ['aurea','parryi','hispidulum','jordanii']
#             self.coarse_names = ['Myriopteris','Adiantum']
#             self.fine_map = np.arange(4)
#             self.coarse_map = np.array([0,0,1,1])
#         self.split = split
#         self.img_size = 224
#         self.channel = 3
#         self.train = train
#         # transform refers to https://github.com/naver-ai/cmo/blob/main/inat18_train.py#L163-L175
#         if self.train:
#             self.transform = transforms.Compose([
#                 transforms.RandomResizedCrop(224),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.466, 0.471, 0.380], [0.195, 0.194, 0.192])
#             ])  
#         else:
#             self.transform = transforms.Compose([
#                 transforms.Resize(256),
#                 transforms.CenterCrop(224),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.466, 0.471, 0.380], [0.195, 0.194, 0.192])
#             ])


#         if self.train:
#             INaturalist_base = INaturalist(root = '/data/common/iNaturalist/train_mini', version = '2021_train_mini', target_type = ['full'], train = self.train) 
#         else:
#             INaturalist_base = INaturalist(root = '/data/common/iNaturalist/val', version = '2021_valid', target_type = ['full'], train = self.train)
#         INaturalist_base.targets = np.array(INaturalist_base.all_categories)                   
#         # fine_idx = [INaturalist_base.categories_index['full'][n] for n in self.fine_names]
#         reset_idx_map = {idx:i for i,idx in enumerate(self.fine_id)}

#         all_image_indices = [idx for idx, (cat_id, _) in enumerate(INaturalist_base.index)]
#         selected_image_indices = [idx for idx in all_image_indices if INaturalist_base.index[idx][0] in self.fine_id]
#         self.data = [INaturalist_base[idx][0] for idx in selected_image_indices]
#         self.targets = [reset_idx_map[INaturalist_base.index[idx][0]] for idx in selected_image_indices]

#         # target_idx = np.concatenate([np.argwhere(INaturalist_base.targets == i).flatten() for i in fine_idx])
#         # self.data = INaturalist_base.data[target_idx]
#         # self.targets = INaturalist_base.targets[target_idx]
#         # self.targets = [reset_idx_map[i] for i in self.targets]
        
#         self.mid_map = None
#         self.coarsest_map = None
#         self.mid2coarse = None

#     def __len__(self):
#         return len(self.targets)

#     def __getitem__(self, index: int):
#         img = self.data[index]
#         target_fine = int(self.targets[index])

#         target_coarser = -1 # dummy
#         target_mid = -1 # dummy
#         target_coarse = int(self.coarse_map[target_fine])
#         img = Image.fromarray(img)

#         if self.transform is not None:
#             img = self.transform(img)
#         return img, target_coarser, target_coarse, target_mid, target_fine

