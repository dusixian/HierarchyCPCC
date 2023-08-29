import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import ot
from itertools import combinations
from treeOT import *

import random
from typing import *
from datetime import datetime


class SK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, M, reg, numItermax):
        m, n = M.shape
        M_np = M.detach().cpu().numpy()
        a = np.full(m, 1 / m)
        b = np.full(n, 1 / n)
        flow = torch.tensor(ot.sinkhorn(a, b, M_np, reg=reg, numItermax=numItermax)).to(M.device)
        # flow = torch.tensor(np.random.rand(m,n)).to(M.device)
        emd = (M * flow).sum()

        ctx.save_for_backward(flow)

        return emd
    
    @staticmethod
    def backward(ctx, grad_output):
        flow, = ctx.saved_tensors
        grad_cost_matrix = flow * grad_output
        
        return grad_cost_matrix, None, None
    
class EMDFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dist_matrix):
        n = dist_matrix.shape[0]
        m = dist_matrix.shape[1]

        # Create the weights for x and y
        wx = np.full((n, 1), 1 / n).astype(np.float32)
        wy = np.full((m, 1), 1 / m).astype(np.float32)

        # Use weights to compute EMD
        emd, _, flow = cv2.EMD(wx, wy, cv2.DIST_USER, dist_matrix.cpu().numpy())

        # Save variables needed for backward in ctx
        ctx.save_for_backward(torch.tensor(flow))
        
        return torch.tensor(emd, dtype=dist_matrix.dtype, device=dist_matrix.device)
    
    @staticmethod
    def backward(ctx, grad_output):
        flow, = ctx.saved_tensors
        grad = flow * grad_output
        return grad
    
class OTEMDFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cost_matrix):
        flow = ot.emd(torch.tensor([]), torch.tensor([]), cost_matrix, numThreads=20)
        emd = torch.sum(cost_matrix * flow)
        ctx.save_for_backward(flow)
        return emd
    
    @staticmethod
    def backward(ctx, grad_output):
        flow, = ctx.saved_tensors
        grad_cost_matrix = flow * grad_output
        return grad_cost_matrix

# class SmoothOT(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, M, semi=True, regul=SquaredL2(gamma=1.0), max_iter=1000):
#         # Convert torch tensor to numpy array
#         M_np = M.detach().cpu().numpy()
        
#         # Compute the weight vectors a and b
#         m, n = M.shape
#         a = np.full(m, 1 / m)
#         b = np.full(n, 1 / n)
        
#         if semi:
#             alpha = solve_semi_dual(a, b, M_np, regul, max_iter=max_iter)
#             T = get_plan_from_semi_dual(alpha, b, M_np, regul)
#         else:
#             alpha, beta = solve_dual(a, b, M_np, regul, max_iter=max_iter)
#             T = get_plan_from_dual(alpha, beta, M_np, regul)

#         # Save T for backward pass
#         T_tensor = torch.from_numpy(T).to(M.device)
#         ctx.save_for_backward(T_tensor)

#         return torch.sum(T_tensor * M) + torch.tensor(regul.Omega(T)).to(M.device)

#     @staticmethod
#     def backward(ctx, grad_output):
#         # Retrieve T
#         T, = ctx.saved_tensors
#         return T * grad_output, None, None, None

def smooth(M, reg):
    M_np = M.detach().cpu().numpy()
        
    # Compute the weight vectors a and b
    m, n = M.shape
    a = np.full(m, 1 / m)
    b = np.full(n, 1 / n)

    return torch.sum(torch.tensor(ot.smooth.smooth_ot_dual(a, b, M_np, reg) * M_np))

def sinkhorn(M, reg, numItermax):
    m, n = M.shape
    M_np = M.detach().cpu().numpy()
    a = np.full(m, 1 / m)
    b = np.full(n, 1 / n)

    return torch.tensor(ot.sinkhorn2(a, b, M_np, reg=reg, numItermax=numItermax))

def compute_tree_ot_distance(dataset, pair, representations_np, target_fine_np):
    current_time0 = datetime.now()
    index_A = pair[0].cpu().numpy()
    index_B = pair[1].cpu().numpy()
    samples_A = representations_np[index_A]
    samples_B = representations_np[index_B]

    # Create probability distributions
    combined_samples_size = samples_A.shape[0] + samples_B.shape[0]
    a = np.ones(combined_samples_size) / samples_A.shape[0]
    b = np.ones(combined_samples_size) / samples_B.shape[0]

    # Adjust 'a' and 'b'
    a[samples_A.shape[0]:] = 0
    b[:samples_B.shape[0]] = 0

    # Use treeOT to compute distance
    tree_ot = treeOT(dataset, samples_A, target_fine_np[index_A][0], samples_B, target_fine_np[index_B][0], lam=0.001, n_slice=1, is_sparse=True)
    build_time, getB_time, learn_time, wB_time = tree_ot.build_time, tree_ot.getB_time, tree_ot.learn_time, tree_ot.wB_time
    current_time1 = datetime.now()
    distance = tree_ot.pairwiseTWD(a, b)
    current_time2 = datetime.now()
    twd_time = (current_time2 - current_time1).total_seconds() 
    # print("TWD: ", distance)
    twd_total_time = (current_time2 - current_time0).total_seconds() 
    
    return distance, np.array([build_time, getB_time, learn_time, wB_time, twd_time, twd_total_time])

def build_tree_and_compute_path(representations, target_fine, fine2mid, fine2coarse):
    unique_fine_classes = torch.unique(target_fine)
    class2reps = {}  # To hold averaged representations for each class
    
    # Create the root node (average over all samples)
    root = torch.mean(representations, dim=0)
    
    # Initialize tree with root
    tree = {'root': {'data': root, 'children': {}}}
    
    for fine_class in unique_fine_classes:
        indices = (target_fine == fine_class)
        avg_rep = torch.mean(representations[indices], dim=0)
        
        mid_class = fine2mid[fine_class.item()]
        coarse_class = fine2coarse[fine_class.item()]
        
        # Update mid-class and coarse-class averaged representations
        if coarse_class not in tree['root']['children']:
            tree['root']['children'][coarse_class] = {'data': None, 'children': {}}
        if mid_class not in tree['root']['children'][coarse_class]['children']:
            tree['root']['children'][coarse_class]['children'][mid_class] = {'data': None, 'children': {}}
        
        tree['root']['children'][coarse_class]['children'][mid_class]['children'][fine_class.item()] = {'data': avg_rep, 'children': {}}
    
    # Update the mid and coarse class node data by averaging children
    for coarse_class, coarse_node in tree['root']['children'].items():
        coarse_data = []
        for mid_class, mid_node in coarse_node['children'].items():
            mid_data = []
            for fine_class, fine_node in mid_node['children'].items():
                mid_data.append(fine_node['data'])
            mid_data = torch.stack(mid_data).mean(dim=0)
            mid_node['data'] = mid_data
            coarse_data.append(mid_data)
        coarse_data = torch.stack(coarse_data).mean(dim=0)
        coarse_node['data'] = coarse_data
    
    # Calculate pairwise distance along the tree
    pairwise_dists = []
    for i, j in combinations(unique_fine_classes, 2):
        mid_i = fine2mid[i.item()]
        mid_j = fine2mid[j.item()]
        coarse_i = fine2coarse[i.item()]
        coarse_j = fine2coarse[j.item()]
        
        path_i = [tree['root']['data'], tree['root']['children'][coarse_i]['data'], tree['root']['children'][coarse_i]['children'][mid_i]['data']]
        path_j = [tree['root']['data'], tree['root']['children'][coarse_j]['data'], tree['root']['children'][coarse_j]['children'][mid_j]['data']]
        
        dists = [torch.norm(a - b, p=2) for a, b in zip(path_i, path_j)]
        pairwise_dists.append(sum(dists))
    
    return torch.tensor(pairwise_dists).to(representations.device)


class CPCCLoss(nn.Module):
    '''
    CPCC as a mini-batch regularizer.
    '''
    def __init__(self, dataset, is_emd, train_on_mid, reg, numItermax, layers : List[str] = ['coarse'], distance_type : str = 'l2'):
        # make sure unique classes in layers[0] 
        super(CPCCLoss, self).__init__()
        
        sizes = []
        for name in layers:
            if name == 'mid':
                sizes.append(len(dataset.mid_names))
            elif name == 'coarse':
                sizes.append(len(dataset.coarse_names))
            elif name == 'coarsest':
                sizes.append(len(dataset.coarsest_names))
        assert (sizes == sorted(sizes)[::-1]), 'Please pass in layers ordered by descending granularity.'       
        
        self.layers = layers
        self.fine2coarse = dataset.coarse_map
        self.fine2mid = dataset.mid_map
        self.fine2coarsest = dataset.coarsest_map
        self.distance_type = distance_type
        self.mid2coarse = dataset.mid2coarse
        self.is_emd = is_emd
        self.train_on_mid = train_on_mid
        self.reg = reg
        self.numItermax = numItermax
        self.dataset = dataset

        # TODO: map = [(weight, class_id)], current setting weight == 1 everywhere
        # four levels always at the same height

    def forward(self, representations, target_fine):

        # assume we only consider two level, fine and coarse
        # where fine and coarse always of the same height
        all_fine = torch.unique(target_fine)
        
        if self.is_emd > 0:
            pairwise_dist = []
            all_pairwise = torch.cdist(representations, representations)
            target_indices = [torch.where(target_fine == fine)[0] for fine in all_fine]
            combidx = [(target_indices[i], target_indices[j]) for (i,j) in combinations(range(len(all_fine)),2)]
            dist_matrices = [all_pairwise.index_select(0,pair[0]).index_select(1,pair[1]) for pair in combidx]

            if self.is_emd == 1: # original EMD
                pairwise_dist = torch.stack([OTEMDFunction.apply(M) for M in dist_matrices])
            elif self.is_emd == 2: # sinkhorn
                # pairwise_dist = torch.stack([sinkhorn(M, self.reg, self.numItermax) for M in dist_matrices])
                pairwise_dist = torch.stack([SK.apply(M, self.reg, self.numItermax) for M in dist_matrices])
            elif self.is_emd == 3: # SmoothOT
                pairwise_dist = torch.stack([smooth(M, self.reg) for M in dist_matrices])
            elif self.is_emd == 5:
                pairwise_dist = build_tree_and_compute_path(representations, target_fine, self.fine2mid, self.fine2coarse)
            else:
                representations_np = representations.detach().cpu().numpy()
                # print('dist_matrices: ', dist_matrices[0])
                target_fine_np = target_fine.detach().cpu().numpy()
                # pairwise_dist = torch.stack([torch.tensor(compute_tree_ot_distance(self.dataset, pair, representations_np, target_fine_np)).to(representations.device) for pair in combidx])
                total_time = np.array([0.0,0.0,0.0,0.0,0.0, 0.0])  # 初始化总时间为0
                distances = []
                for pair in combidx:
                    distance, time = compute_tree_ot_distance(self.dataset, pair, representations_np, target_fine_np)
                    distances.append(torch.tensor(distance).to(representations.device))
                    total_time += time  # 累加每次调用的时间

                # 使用torch.stack只对距离进行堆栈
                self.time = total_time
                pairwise_dist = torch.stack(distances)
                # pairwise_dist = torch.zeros((len(combidx)))
                # print("pairwise_dist: ", pairwise_dist)
                # print("pairwise_dist2: ", torch.stack([OTEMDFunction.apply(M) for M in dist_matrices]))
                # assert(0==1)
                
        else: # use Euclidean distance
            # get the center of all fine classes
            target_fine_list = [torch.mean(torch.index_select(representations, 0, (target_fine == t).nonzero().flatten()),0) for t in all_fine]
            sorted_sums = torch.stack(target_fine_list, 0)

            if self.distance_type == 'l2':
                pairwise_dist = F.pdist(sorted_sums, p=2.0) # get pairwise distance
            elif self.distance_type == 'l1':
                pairwise_dist = F.pdist(sorted_sums, p=1.0)
            elif self.distance_type == 'poincare':
                # Project into the poincare ball with norm <= 1 - epsilon
                # https://www.tensorflow.org/addons/api_docs/python/tfa/layers/PoincareNormalize
                epsilon = 1e-5 
                all_norms = torch.norm(sorted_sums, dim=1, p=2).unsqueeze(-1)
                normalized_sorted_sums = sorted_sums * (1 - epsilon) / all_norms
                all_normalized_norms = torch.norm(normalized_sorted_sums, dim=1, p=2) 
                # F.pdist underflow, might be due to sqrt on very small values, 
                # causing nan in gradients
                # |u-v|^2
                condensed_idx = torch.triu_indices(len(all_fine), len(all_fine), offset=1, device = sorted_sums.device)
                numerator_square = torch.sum((normalized_sorted_sums[None, :] - normalized_sorted_sums[:, None])**2, -1)
                numerator = numerator_square[condensed_idx[0],condensed_idx[1]]
                # (1 - |u|^2) * (1 - |v|^2)
                denominator_square = ((1 - all_normalized_norms**2).reshape(-1,1)) @ (1 - all_normalized_norms**2).reshape(1,-1)
                denominator = denominator_square[condensed_idx[0],condensed_idx[1]]
                pairwise_dist = torch.acosh(1 + 2 * (numerator/denominator))

        all_fine = all_fine.tolist() # all unique fine classes in this batch
        
        if len(self.layers) == 1:
            if self.train_on_mid:
                tree_pairwise_dist = self.two_level_dT(all_fine, self.mid2coarse, pairwise_dist.device)
            else:
                if self.layers[0] == 'coarsest':
                    fine2layer = self.fine2coarsest
                elif self.layers[0] == 'mid':
                    fine2layer = self.fine2mid
                elif self.layers[0] == 'coarse':
                    fine2layer = self.fine2coarse
                tree_pairwise_dist = self.two_level_dT(all_fine, fine2layer, pairwise_dist.device)
        elif len(self.layers) == 2:
            if self.layers[0] == 'mid' and self.layers[1] == 'coarse':
                fine2layers = [self.fine2mid, self.fine2coarse]
            elif self.layers[0] == 'mid' and self.layers[1] == 'coarsest':
                fine2layers = [self.fine2mid, self.fine2coarsest]
            elif self.layers[0] == 'coarse' and self.layers[1] == 'coarsest':
                fine2layers = [self.fine2coarse, self.fine2coarsest]
            tree_pairwise_dist = self.three_level_dT(all_fine, fine2layers, pairwise_dist.device)
        else:
            raise ValueError('Not Implemented')
        
        res = 1 - torch.corrcoef(torch.stack([pairwise_dist, tree_pairwise_dist], 0))[0,1] # maximize cpcc
        # "1" doesn't do anything to the gradient, just for better interpreting CPCCLoss as pearson r.
        if torch.isnan(res): # see nan zero div (cause: batch size small then same value for all tree dist in a batch)
            return torch.tensor(1,device=pairwise_dist.device)
        else:
            return res

    def two_level_dT(self, all_fine : list, fine2layer : np.ndarray, device : torch.device):
        '''
            Args:
                all_fine : all unique fine classes in the batch
                fine2layer : fine to X map
        '''
        # assume unweighted tree
        # when coarse class the same, shortest distance == 2
        # otherwise, shortest distance == 4
        # TODO: weighted tree, arbitrary level on the tree
        tree_pairwise_dist = torch.tensor([2 if fine2layer[all_fine[i]] == fine2layer[all_fine[j]] 
                                           else 4 for (i,j) in combinations(range(len(all_fine)),2)], 
                                           device=device)
        return tree_pairwise_dist
    
    def three_level_dT(self, all_fine, fine2layers, device):
        # tree height = 3
        tree_pairwise_dist = []
        mid_map = fine2layers[0]
        coarse_map = fine2layers[1]
        for i in range(len(all_fine)):
            for j in range(i+1, len(all_fine)):
                if mid_map[all_fine[i]] == mid_map[all_fine[j]]: # same L2
                    tree_pairwise_dist.append(2)
                elif coarse_map[all_fine[i]] == coarse_map[all_fine[j]]: # same L1 but not same L2
                    tree_pairwise_dist.append(4)
                else:
                    tree_pairwise_dist.append(6) # same L0 but not same L1
        tree_pairwise_dist = torch.tensor(tree_pairwise_dist, device=device)
        return tree_pairwise_dist
    


class QuadrupletLoss(nn.Module):
    
    def __init__(self, dataset, m1=0.25, m2=0.15):
        super(QuadrupletLoss, self).__init__()
        assert (m1 > m2) and (m2 > 0)
        self.m1 = m1
        self.m2 = m2
        self.fine2coarse = dataset.coarse_map

    def l2(self, x1, x2): # squared euclidean distance, x1, x2 same shape 1d vector
        return (x1 - x2).pow(2).sum()
    
    def pairwise(self, representation):
        return torch.cdist(representation, representation)**2

    def forward(self, representation, target_fine) -> torch.Tensor:
        in_coarse = 0
        out_coarse = 0
        memo = dict() # store valid quad combination for each anchor class
        valid_quads = 0
        pairwise_distM = self.pairwise(representation)
        
        for idx, t in enumerate(target_fine): # for each anchor, random sample quad
            if t not in memo:
                t = t.item()
                r_coarse_cls = self.fine2coarse[t]
                same_coarse_idx = (self.fine2coarse == r_coarse_cls).nonzero()[0]
                p_minus_maps = torch.zeros((len(target_fine),), device = target_fine.device)
                negative_maps = torch.ones((len(target_fine),), device = target_fine.device)
                for a in same_coarse_idx:
                    if a != t:
                        p_minus_maps += (target_fine == a) # union
                    negative_maps *= (target_fine != a) # intersection
                all_r = (target_fine == t)
                all_r[idx] = False # don't want to use itself as p+
                try_all_p_plus = all_r.nonzero()
                try_all_p_minus = p_minus_maps.nonzero()
                try_all_negative = negative_maps.nonzero()
                if (len(try_all_p_plus) == 0) or (len(try_all_p_minus) == 0) or (len(try_all_negative) == 0): 
                    memo[t] = None # cannot find a valid quad for class r in this batch
                else:
                    # list of indices in all fine_targets
                    all_p_plus = try_all_p_plus[:,0]
                    all_p_minus = try_all_p_minus[:,0] 
                    all_negative = try_all_negative[:,0]
                    memo[t] = (all_p_plus, all_p_minus, all_negative)
            if memo[t] is None:
                continue
            else:
                valid_quads += 1
                r = idx # r : index of anchor
                p_pluss, p_minuss, ns = memo[t] # valid indices

                p_plus = p_pluss[range(len(p_pluss))[0]] # (randomly) select positive sample
                distance_positive = self.l2(representation[r], representation[p_plus])

                # hard mining for faster convergence
                p_minus_losses = torch.relu(distance_positive - pairwise_distM[p_minuss,r] + self.m1 - self.m2)
                p_minus = p_minuss[torch.argmax(p_minus_losses)]
                distance_pos_coarse = self.l2(representation[r], representation[p_minus])

                n_losses = torch.relu(distance_pos_coarse - pairwise_distM[ns,r] + self.m2)
                n = ns[torch.argmax(n_losses)]
                distance_neg_coarse = self.l2(representation[r], representation[n])
                
                in_coarse_loss = torch.relu(distance_positive - distance_pos_coarse + self.m1 - self.m2)
                out_coarse_loss = torch.relu(distance_pos_coarse - distance_neg_coarse + self.m2)
                
                if in_coarse_loss == 0 or out_coarse_loss == 0:
                    valid_quads -= 1 # didn't select a hard quad, remove it
                else:
                    in_coarse += in_coarse_loss
                    out_coarse += out_coarse_loss
        if valid_quads == 0:
            return 0
        else:
            return (in_coarse + out_coarse) / (2 * valid_quads)
        


class GroupLasso(nn.Module): 
    def __init__(self, dataset, lamb = 0.01):
        super(GroupLasso, self).__init__()
        self.fine2coarse = dataset.coarse_map # only assume 2 layers
        # to match with CPCC, gamma = correlation strength
        self.root_gamma = 2 * lamb
        self.coarse_gamma = 4 * lamb
        self.fine_gamma = 6 * lamb
        assert (self.root_gamma < self.coarse_gamma) and (self.coarse_gamma < self.fine_gamma)
        groups = {self.root_gamma : [np.array(range(len(dataset.coarse_map)))],
                  self.coarse_gamma : [],
                  self.fine_gamma : []} 
        for fine in range(len(dataset.coarse_map)):
            groups[self.coarse_gamma].append((dataset.coarse_map == fine).nonzero()[0])
        for fine in range(len(dataset.coarse_map)):
            groups[self.fine_gamma].append(np.array(fine))
        self.groups = groups
        
    def forward(self, fc_weights, fc_bias):
        # base_weights = all_weights[:-2] # ignore everything before representation layer
        # base_l2 = self.add_base_l2()
        l2_regularization = torch.tensor(0, device = fc_bias.device).float()
        for depth in self.groups:
            for group in self.groups[depth]:
                l2_regularization += depth * torch.norm(fc_weights[group,:])**2
                l2_regularization += depth * torch.norm(fc_bias[group])**2
        return l2_regularization

    def add_base_l2(self, weights : List):
        l2_regularization = torch.tensor(0)
        for param in weights:
            l2_regularization += torch.norm(param, 2)**2
        return l2_regularization