import numpy as np
from treelib import Tree
import copy
from tqdm import tqdm
import random
import spams
from scipy import sparse
from scipy.sparse import csr_matrix
import networkx as nx
import joblib
import torch
import torch.optim as optim
from torch import nn

class treeOT():
    def __init__(self, X, method='cluster', lam=0.0001,nmax=100000, k=5, d=6, n_slice=1, debug_mode=False,is_sparse=False):
        """
         Parameter
         ----------
         X :
             a set of supports
         method :
             'cluster' (clustering tree) or 'quad' (quadtree)
         k : int
             a number of child nodes
         d : int
             depth of a tree
         n_slice : int
             the number of sampled trees
         lam: float
             the regularization parameter of Lasso
         nmax: int
             the number of training samples for Lasso
         """

        self.n_slice = n_slice

        for i in tqdm(range(n_slice)):

            if method=='quad': #Quadtree
                torch.manual_seed(i) # np.random.seed(i)

                tree = self.build_quadtree(X, random_shift=True, width=None, origin=None)
                print("build done")
                #self.D1, self.D2 = self.gen_matrix(tree, X)
            else: #Clustering tree
                random.seed(i)
                tree = self.build_clustertree(X, k, d, debug_mode=debug_mode)
                print("build done")
                #self.D1, self.D2 = self.gen_matrix(tree, X)

            Bsp = self.get_B_matrix(tree,X)

            wv = self.calc_weight(X,Bsp,lam=lam,nmax=nmax)

            if i == 0:
                wB = Bsp.multiply(wv)
            else:
                wB = sparse.vstack([wB,Bsp.multiply(wv)])


        self.wB = wB


    def distance(self, A, B):
        return torch.norm(A - B)
    

    def incremental_farthest_search(self, points, remaining_set, k, debug_mode=False):
        n_points = len(remaining_set)
        remaining_set = copy.deepcopy(remaining_set)

        if not debug_mode:
            torch.manual_seed(0)
            solution_set = [remaining_set[torch.randint(0, n_points, (1,)).item()]]
        else:
            solution_set = [remaining_set[0]]
        remaining_set.remove(solution_set[0])

        for i in range(k - 1):
            distance_list = []

            for idx in remaining_set:
                in_distance_list = [self.distance(points[idx], points[sol_idx]) for sol_idx in solution_set]
                distance_list.append(min(in_distance_list))

            sol_idx = remaining_set[torch.argmax(torch.tensor(distance_list)).item()]
            remaining_set.remove(sol_idx)
            solution_set.append(sol_idx)

        return solution_set

    def grouping(self, points, remaining_set, solution_set):
        torch.manual_seed(0)
        n_points = len(points)
        remaining_set = copy.deepcopy(remaining_set)

        group = [[] for _ in range(len(solution_set))]

        for idx in remaining_set:
            distance_list = [self.distance(points[idx], points[sol_idx]) for sol_idx in solution_set]
            group_idx = torch.argmin(torch.tensor(distance_list))
            group[group_idx].append(idx)

        return group

    def clustering(self, points, remaining_set, k, debug_mode=False):
        solution_set = self.incremental_farthest_search(points, remaining_set, k, debug_mode=debug_mode)
        return self.grouping(points, remaining_set, solution_set)

    def _build_clustertree(self, X, remaining_set, k, d, debug_mode=False):
        tree = Tree()
        tree.create_node(data=None)

        if len(remaining_set) <= k or d == 1:
            for idx in remaining_set:
                tree.create_node(parent=tree.root, data=idx)
            return tree

        groups = self.clustering(X, remaining_set, k, debug_mode=debug_mode)
        for group in groups:
            if len(group) == 1:
                tree.create_node(parent=tree.root, data=group[0])
            else:
                subtree = self._build_clustertree(X, group, k, d - 1, debug_mode=debug_mode)
                tree.paste(tree.root, subtree)
        return tree

    def build_clustertree(self, X, k, d, debug_mode=False):
        """
        k : the number of child nodes
        d : the depth of the tree
        """
        remaining_set = [i for i in range(len(X))]
        return self._build_clustertree(X, remaining_set, k, d, debug_mode=debug_mode)

    def get_B_matrix(self, tree, X):
        n_node = len(tree.all_nodes())
        n_leaf = X.shape[0]
        n_in = n_node - n_leaf

        B = torch.zeros(n_node, n_leaf, dtype=torch.float32)

        in_node = [node.identifier for node in tree.all_nodes() if node.data == None]
        in_node_index = [ii for ii in range(n_in)]
        leaf_node = [node.identifier for node in tree.all_nodes() if node.data != None]
        leaf_node_index = [node.data for node in tree.all_nodes() if node.data != None]

        path_leaves = tree.paths_to_leaves()

        for path in path_leaves:
            leaf_index = leaf_node_index[leaf_node.index(path[-1])]
            B[leaf_index, leaf_index] = 1.0

            for node in path[:-1]:
                in_index = in_node_index[in_node.index(node)] + n_leaf
                B[in_index, leaf_index] = 1.0

        return B


    def calc_weight(self, X, B, lam=0.001, seed=0, nmax=100000):
        n_leaf, d = X.shape
        torch.manual_seed(seed)

        dz = B.shape[0]
        print('X.shape:', X.shape)
        print('B.shape:', B.shape)

        ind1 = torch.randint(0, n_leaf, (nmax,))
        ind2 = torch.randint(0, n_leaf, (nmax,))

        c_all = torch.zeros((nmax, 1), dtype=torch.float32)
        Z_all = torch.zeros((dz, nmax), dtype=torch.float32)

        for ii in range(nmax):
            # print(ii, Z_all.shape)
            c_all[ii] = torch.norm(X[ind1[ii], :] - X[ind2[ii], :], p=2)
            B_ind1 = B[:, ind1[ii]]
            B_ind2 = B[:, ind2[ii]]
            # print('B_ind1.shape: ', B_ind1.shape)
            # print('B_ind2.shape: ', B_ind2.shape)
            Z_all[:, ii] = B_ind1 + B_ind2 - 2 * (B_ind1 * B_ind2)

        n_sample = nmax
        print('here')
        c = c_all[:n_sample, 0].reshape((n_sample, 1)).float()
        print('here1')
        Z = Z_all[:, :n_sample].transpose(0,1).float()
        print('done')

        # Define your optimization problem
        W0 = torch.zeros((Z.shape[1], c.shape[1]), dtype=torch.float32, requires_grad=True)
        print('111')

        def fista(loss_fn, grad_fn, x_init, lr, num_iters, tol=1e-3):
            x = x_init.clone()
            y = x_init.clone()
            t = 1.0

            for i in range(num_iters):
                # Compute gradient
                grad = grad_fn(y)

                # Gradient step
                x_next = y - lr * grad

                # Ensure non-negativity using ReLU
                x_next = torch.nn.functional.relu(x_next)

                # Momentum update
                t_next = (1.0 + torch.sqrt(1.0 + 4.0 * t * t)) / 2.0
                y = x_next + ((t - 1.0) / t_next) * (x_next - x)

                # Convergence check (simplified)
                if torch.norm(x_next - x) < tol:
                    break

                x, t = x_next, t_next

            return x

        # Define the Lasso loss and its gradient
        def loss_fn(W):
            return 0.5 * torch.norm(c - Z @ W, p='fro')**2 + lam * torch.norm(W, p=1)

        def grad_fn(W):
            return Z.t() @ (Z @ W - c) + lam * torch.sign(W)

        # Use FISTA for optimization
        W= fista(loss_fn, grad_fn, W0, lr=0.01, num_iters=2000)

        return W




    def pairwiseTWD(self,a,b):
        # Compute the Tree Wasserstein

        TWD = abs(self.wB.dot(a - b)).sum(0) / self.n_slice

        return TWD