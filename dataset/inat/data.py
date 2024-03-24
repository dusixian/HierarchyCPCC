from typing import *

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import INaturalist

from tqdm import tqdm

# from hierarchy.data import Hierarchy, get_k_shot

import os
import json

class HierarchyINaturalist(INaturalist):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)             
        self.species_label = np.array(self.categories_map)
        self.targets = [self.index[idx][0] for idx in range(len(self.index))] # index -> species
        self.fine_map = np.array([d['class'] for d in self.species_label]) # species -> fine
        self.fine_names = np.array([str(x) for x in np.unique(self.fine_map)])
        self.targets = [self.fine_map[t] for t in self.targets] # index -> fine

        self.ind2mid_map = np.array([d['phylum'] for d in self.species_label])

        self.ind2coarse = np.array([d['kingdom'] for d in self.species_label])
        # self.coarse_names = np.array([str(i) for i in range(len(np.unique(self.coarse_map)))])

        self.mid_map = np.zeros(len(self.fine_names), dtype=int)
        self.coarse_map = np.zeros(len(self.fine_names), dtype=int)
        for index, fine_id in enumerate(self.fine_map):
            self.mid_map[fine_id] = self.categories_map[index]['phylum']
            self.coarse_map[fine_id] = self.categories_map[index]['kingdom']
        
        self.mid_names = np.array([str(i) for i in range(len(np.unique(self.mid_map)))])
        self.coarse_names = np.array([str(i) for i in range(len(np.unique(self.coarse_map)))])
        self.mid2coarse = None

        self.img_size = 224
        self.channel = 3

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        cat_id, fname = self.index[index]
        img = Image.open(os.path.join(self.root, self.all_categories[cat_id], fname))

        if self.transform is not None:
            img = self.transform(img)

        target_fine = self.targets[index]
        target_mid = self.mid_map[target_fine]
        target_coarse = self.coarse_map[target_fine]
        target_coarsest = -1

        assert self.target_transform is None

        return img, target_coarsest, target_coarse, target_mid, target_fine


class HierarchyINaturalistSubset(HierarchyINaturalist):
    def __init__(self, indices : List[int], fine_classes : List[int], *args, **kw):
        super(HierarchyINaturalistSubset, self).__init__(*args, **kw)
        self.index = [self.index[i] for i in indices]
        
        old_targets = list(np.array(self.targets)[indices]) # old fine targets, sliced. index -> old fine
        fine_classes = np.array(sorted(fine_classes)) # fine class id in HierarchyINAT index
        self.fine_names = [self.fine_names[i] for i in fine_classes] # old fine names, sliced
        # self.fine_map = np.array(range(len(fine_classes))) # number of fine classes subset. new fine
        # print('fine_map', self.fine_map)
        self.targets = [list(fine_classes).index(i) for i in old_targets] # reset fine target id from 0. index -> new fine

        # reset other hierarchy level index, from 0
        old_coarse_map = np.array([self.coarse_map[i] for i in fine_classes]) # subset of coarse fine map. fine->coarse
        coarse_classes = np.unique(old_coarse_map)
        self.coarse_names = [self.coarse_names[cls] for cls in coarse_classes]
        self.coarse_map = np.array([list(coarse_classes).index(i) for i in old_coarse_map]) # argsort

        old_mid_map = np.array([self.mid_map[i] for i in fine_classes]) # subset of mid fine map
        mid_classes = np.unique(old_mid_map)
        self.mid_names = [self.mid_names[cls] for cls in mid_classes]
        self.mid_map = np.array([list(mid_classes).index(i) for i in old_mid_map]) # argsort

        old_coarsest_map = None
        coarsest_classes = None
        self.coarsest_names = None
        self.coarsest_map = None

    def __len__(self):
        return len(self.index)
    

def make_dataloader(num_workers : int, batch_size : int, task : str) -> Tuple[DataLoader, DataLoader]:
    def make_full_dataset() -> Tuple[Dataset, Dataset]:
        train_dataset = HierarchyINaturalist(root = '/data/common/iNaturalist/train_mini',
                                                version = '2021_train_mini', target_type = ['full'],                        
                                                transform = transforms.Compose([
                                                transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.466, 0.471, 0.380], [0.195, 0.194, 0.192])
                                            ]))
        # not augment test set except of normalization
        test_dataset = HierarchyINaturalist(root = '/data/common/iNaturalist/val',
                                                version = '2021_valid', target_type = ['full'],                        
                                                transform = transforms.Compose([
                                                transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.466, 0.471, 0.380], [0.195, 0.194, 0.192])
                                            ]))
        return train_dataset, test_dataset

    def make_subpopsplit_dataset(train_dataset, test_dataset, task : str) -> Tuple[DataLoader, DataLoader]: 
        subset_info_path = "./dataset/inat/subset_info.json"

        if os.path.exists(subset_info_path):
            # Load subset information
            with open(subset_info_path, "r") as f:
                subset_info = json.load(f)
            
            idx_train_source = subset_info["idx_train_source"]
            idx_train_target = subset_info["idx_train_target"]
            source_fine_cls = subset_info["source_fine_cls"]
            target_fine_cls = subset_info["target_fine_cls"]
            idx_test_source = subset_info["idx_test_source"]
            idx_test_target = subset_info["idx_test_target"]

        else:
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

            for fine_id, coarse_id in enumerate(train_dataset.coarse_map):
                if coarse_id not in c2f:
                    c2f[coarse_id] = []
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

            subset_info = {
                'idx_train_source': [int(i) for i in idx_train_source],
                'idx_train_target': [int(i) for i in idx_train_target],
                'source_fine_cls': [int(i) for i in source_fine_cls],
                'target_fine_cls': [int(i) for i in target_fine_cls],
                'idx_test_source': [int(i) for i in idx_test_source],
                'idx_test_target': [int(i) for i in idx_test_target]
            }
            with open(subset_info_path, "w") as f:
                json.dump(subset_info, f)
        
        source_train = HierarchyINaturalistSubset(idx_train_source, source_fine_cls, 
                                                  root = '/data/common/iNaturalist/train_mini',
                                                  version = '2021_train_mini', target_type = ['full'],                        
                                                transform = transforms.Compose([
                                                transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.466, 0.471, 0.380], [0.195, 0.194, 0.192])
                                            ]))                              
        source_test = HierarchyINaturalistSubset(idx_test_source, source_fine_cls, 
                                                  root = '/data/common/iNaturalist/val',
                                                  version = '2021_valid', target_type = ['full'],                        
                                                transform = transforms.Compose([
                                                transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.466, 0.471, 0.380], [0.195, 0.194, 0.192])
                                            ]))
        target_train = HierarchyINaturalistSubset(idx_train_target, target_fine_cls, 
                                                  root = '/data/common/iNaturalist/train_mini',
                                                  version = '2021_train_mini', target_type = ['full'],                        
                                                transform = transforms.Compose([
                                                transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.466, 0.471, 0.380], [0.195, 0.194, 0.192])
                                            ]))
        target_test = HierarchyINaturalistSubset(idx_test_target, target_fine_cls, 
                                                  root = '/data/common/iNaturalist/val',
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



