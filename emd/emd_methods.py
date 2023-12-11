import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ot
import time
from typing import *

class SK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, M, reg, numItermax):
        m, n = M.shape
        a = np.full(m, 1 / m)
        b = np.full(n, 1 / n)
        M_np = M.detach().cpu().numpy()
        flow = torch.tensor(ot.sinkhorn(a, b, M_np, reg=reg, numItermax=numItermax)).to(M.device)
        emd = (M * flow).sum()

        ctx.save_for_backward(flow)

        return emd
    
    @staticmethod
    def backward(ctx, grad_output):
        flow, = ctx.saved_tensors
        grad_cost_matrix = flow * grad_output
        
        return grad_cost_matrix, None, None
    
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

def simple(P, x, y):
    comparison = (x.reshape(-1, 1) > y.reshape(1, -1)).astype(int)
    comparison_eq = 1 - (x.reshape(-1, 1) == y.reshape(1, -1)).astype(int)
    comparison[comparison == 0] = -1
    comparison *= comparison_eq
    return np.sum(P * comparison, axis=1)

class SlicedWasserstein_np(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, Y, n_projections, p=2):
        # Convert tensors to NumPy arrays for calculations
        X_np = X.detach().cpu().numpy()
        Y_np = Y.detach().cpu().numpy()
        
        d = X_np.shape[1]
        projections = ot.sliced.get_random_projections(d, n_projections, 0)
        
        X_projections = X_np.dot(projections)
        Y_projections = Y_np.dot(projections)
        
        sum_emd = 0
        flow_matrices = []
        
        for X_p, Y_p in zip(X_projections.T, Y_projections.T):
            emd_value, flow_matrix = ot.lp.emd2_1d(X_p, Y_p, log=True, metric='euclidean')
            sum_emd += emd_value
            flow_matrices.append(flow_matrix['G'])
        
        sum_emd /= n_projections
        ctx.save_for_backward(X, Y, torch.tensor(flow_matrices), torch.tensor(projections), torch.tensor(sum_emd), torch.tensor(p))
        sum_emd **= (1.0 / p)
        
        return (torch.tensor(sum_emd, dtype=torch.float32)).to(X.device) # Fixed return value

    @staticmethod
    def backward(ctx, grad_output):
        X, Y, flow_matrices, projections, sum_emd, p = ctx.saved_tensors
        device = X.device
        X = X.cpu().numpy()
        Y = Y.cpu().numpy()
        flow_matrices = flow_matrices.cpu().numpy()
        projections = projections.cpu().numpy().T
        sum_emd = sum_emd.item()
        p = p.item()
        
        grad_X = np.zeros_like(X)
        grad_Y = np.zeros_like(Y)
        
        for i in range(flow_matrices.shape[0]):
            flow_matrix = flow_matrices[i]
            X_p = X.dot(projections[i])
            Y_p = Y.dot(projections[i])
            df_dX = simple(flow_matrix, X_p, Y_p)
            df_dY = simple(flow_matrix.T, Y_p, X_p)
            
            grad_X += df_dX.reshape(-1, 1).dot(projections[i].reshape(1, -1))
            grad_Y += df_dY.reshape(-1, 1).dot(projections[i].reshape(1, -1))
        
        grad_X /= flow_matrices.shape[0]
        grad_Y /= flow_matrices.shape[0]
        
        # apply chain rule for sum_emd ** (1.0 / p)
        chain_coeff = (1.0 / p) * (sum_emd ** ((1.0 / p) - 1))
        grad_X *= chain_coeff * grad_output.item()
        grad_Y *= chain_coeff * grad_output.item()

        return torch.tensor(grad_X, dtype=torch.float32).to(device), torch.tensor(grad_Y, dtype=torch.float32).to(device), None, None
    

def compute_flow_symmetric(a_sample, b_sample):
    start = time.time()
    if isinstance(a_sample, np.ndarray):
        a_sample = torch.as_tensor(a_sample)
        b_sample = torch.as_tensor(b_sample)
    device = a_sample.device
    n, m = a_sample.shape[0], b_sample.shape[0]
    a = torch.full((n,), 1 / n, device=device)
    b = torch.full((m,), 1 / m, device=device)
    
    half_n = n // 2
    compressed_flow = torch.zeros(2*(n+m), device=device)
    index_1 = torch.zeros(2*(n+m), dtype=torch.long)
    index_2 = torch.zeros(2*(n+m), dtype=torch.long)
    
    i, j, k = 0, 0, 0
    while i < half_n and j < m:
        min_val = min(a[i].item(), b[j].item())
        
        compressed_flow[k] = min_val
        index_1[k] = i
        index_2[k] = j
        k += 1
        
        compressed_flow[k] = min_val
        index_1[k] = n - i - 1
        index_2[k] = m - j - 1
        k += 1
        
        a[i] -= min_val
        b[j] -= min_val
        
        if a[i] == 0:
            i += 1
        if b[j] == 0:
            j += 1

    if n % 2 == 1:
        i = half_n
        while j < m:
            min_val = min(a[i].item(), b[j].item())
            
            compressed_flow[k] = min_val
            index_1[k] = i
            index_2[k] = j
            k += 1
            
            a[i] -= min_val
            b[j] -= min_val
            
            if a[i] == 0:
                break
            if b[j] == 0:
                j += 1

    index_1 = index_1[:k]
    index_2 = index_2[:k]
    distances = torch.norm(a_sample[index_1] - b_sample[index_2], dim=1)
    return torch.sum(distances * compressed_flow[:k])