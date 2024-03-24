import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader
from typing import *
import torch.nn.functional as F

from model import init_model
from data import make_kshot_loader, make_dataloader
from param import init_optim_schedule, load_params
from main import feature_extractor

def select_samples(features, labels, num_samples_per_label):
    selected_features = []
    selected_labels = []
    
    unique_labels = np.unique(labels)
    for label in unique_labels:
        idx = np.where(labels == label)[0]
        selected_idx = np.random.choice(idx, num_samples_per_label, replace=False)
        selected_features.extend(features[selected_idx])
        selected_labels.extend(labels[selected_idx])
    
    return np.array(selected_features), np.array(selected_labels)


def feature_extractor(dataloader : DataLoader, split : str, task : str, dataset_name : str, seed : int):
    dataset = dataloader.dataset
    model = init_model(dataset_name, [len(dataset.fine_names)], device)
    model_dict = model.state_dict()
    ckpt_dict = {k: v for k, v in torch.load(save_dir+f"/{split}{task}_seed{seed}.pth").items() if (k in model_dict) and ('fc' not in k)}
    model_dict.update(ckpt_dict) 
    model.load_state_dict(model_dict)

    features = []
    probs = []
    targets_one = []
    targets_coarse = []
    model.eval()
    with torch.no_grad():
        for item in dataloader:
            data = item[0]
            target_one = item[-1] # add fine target
            data = data.to(device)
            target_one = target_one.to(device)
            feature, output = model(data)
            prob_one = F.softmax(output,dim=1)
            probs.append(prob_one.cpu().detach().numpy())
            features.append(feature.cpu().detach().numpy())
            targets_one.append(target_one.cpu().detach().numpy())
            if len(item) == 5:
                target_coarse = item[2]
                target_coarse = target_coarse.to(device)
                targets_coarse.append(target_coarse.cpu().detach().numpy())
    features = np.concatenate(features,axis=0)
    targets_one = np.concatenate(targets_one,axis=0)
    probs = np.concatenate(probs,axis=0)
    if len(targets_coarse) > 0:
        targets_coarse = np.concatenate(targets_coarse,axis=0)  
    return (features, probs, targets_one, targets_coarse)

def reduce_dimensionality(features, method="pca"):
    if method == "pca":
        pca = PCA(n_components=2)
        return pca.fit_transform(features)
    elif method == "tsne":
        tsne = TSNE(n_components=2, random_state=42)
        return tsne.fit_transform(features)

def plot_features(features_list, labels_list, filename="features_plot.png"):
    markers = ['o', 's', 'x']
    
    for i, (features, labels) in enumerate(zip(features_list, labels_list)):
        plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='jet', marker=markers[i], alpha=0.5, label=f'Dataset {i+1}')
    
    plt.colorbar()
    plt.legend()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_dir = '/home/sixian/my_test/final/hierarchy_results/CIFAR/l2_bs1024'
# train_loader_12_e, test_loader_12_e = make_dataloader(5, 128, f'{task}_split_zero_shot', "CIFAR12", 0, '', "easy")
# train_loader_12_m, test_loader_12_m = make_dataloader(5, 128, f'{task}_split_zero_shot', "CIFAR12", 0, '', "medium")
# train_loader_12_h, test_loader_12_h = make_dataloader(5, 128, f'{task}_split_zero_shot', "CIFAR12", 0, '', "hard")
train_loader_t, _ = make_dataloader(5, 1024, f'sub_split_zero_shot', "CIFAR", 0, '', '')
train_loader_s, _ = make_dataloader(5, 1024, f'sub_split_pretrain', "CIFAR", 0, '', '')

# features, _, fine_labels, coarse_labels = feature_extractor(train_loader, split, task, dataset_name, 0)
features_t2, _, fine_labels_t2, coarse_labels_t2 = feature_extractor(train_loader_t, 'downsub_', 'fine', "CIFAR", 0)
features_s, _, fine_labels_s, coarse_labels_s = feature_extractor(train_loader_s, 'split', 'sub', "CIFAR", 0)
features_t, _, fine_labels_t, coarse_labels_t = feature_extractor(train_loader_t, 'downsub_', 'fine', "CIFAR", 0)
reduced_features_pca_t = reduce_dimensionality(features_t, method="pca")
reduced_features_pca_t2 = reduce_dimensionality(features_t2, method="pca")
reduced_features_pca_s = reduce_dimensionality(features_s, method="pca")

plot_features([reduced_features_pca_s, reduced_features_pca_t, reduced_features_pca_t2], [coarse_labels_s, coarse_labels_t, coarse_labels_t2], filename="./discover/TWD_CIFAR.png")
# 选择样本
# selected_features_t, selected_labels_t = select_samples(reduced_features_pca_t, coarse_labels_t, 20)
# selected_features_s, selected_labels_s = select_samples(reduced_features_pca_s, coarse_labels_s, 5)

# 绘制特征
# plot_features([selected_features_t, selected_features_s], [selected_labels_t, selected_labels_s])


