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


class HierarchicalTreeBuilder:
    def __init__(self, dataset, layers):
        self.layers = layers
        self.fine2mid = dataset.mid_map
        self.fine2coarse = dataset.coarse_map

    def build_tree(self, samples_A, label_A, samples_B, label_B):
        tree = Tree()
        tree.create_node("Root", "root")

        # Create nodes for coarse categories
        coarse_A_id = f"coarse_{self.fine2coarse[label_A]}"
        coarse_B_id = f"coarse_{self.fine2coarse[label_B]}"
        tree.create_node(f"Coarse_{self.fine2coarse[label_A]}", coarse_A_id, parent="root")
        if self.fine2coarse[label_A] != self.fine2coarse[label_B]:
            tree.create_node(f"Coarse_{self.fine2coarse[label_B]}", coarse_B_id, parent="root")

        # Create nodes for mid layer
        mid_A_id = f"mid_{self.fine2mid[label_A]}"
        mid_B_id = f"mid_{self.fine2mid[label_B]}"
        tree.create_node(f"Mid_{self.fine2mid[label_A]}", mid_A_id, parent=coarse_A_id)
        if self.fine2mid[label_A] != self.fine2mid[label_B]:
            tree.create_node(f"Mid_{self.fine2mid[label_B]}", mid_B_id, parent=coarse_B_id)

        # Create nodes for fine samples
        for idx, sample in enumerate(samples_A):
            sample_id = f"sample_A_{idx}"
            tree.create_node(f"Sample_{sample}", sample_id, parent=mid_A_id, data=idx)

        for idx, sample in enumerate(samples_B, start=len(samples_A)):
            sample_id = f"sample_B_{idx}"
            tree.create_node(f"Sample_{sample}", sample_id, parent=mid_B_id, data=idx)
        return tree

# class HierarchicalTreeBuilder:
#     def __init__(self, dataset, layers):
#         self.layers = layers
#         self.fine2mid = dataset.mid_map
#         self.fine2coarse = dataset.coarse_map

#     def build_tree(self, samples, labels):
#         tree = Tree()
#         tree.create_node("Root", "root")

#         created_coarse_nodes = set()
#         created_mid_nodes = set()
#         lens = 0

#         for label, sample_group in zip(labels, samples):
#             # Create nodes for coarse categories
#             coarse_id = f"coarse_{self.fine2coarse[label]}"
#             if coarse_id not in created_coarse_nodes:
#                 tree.create_node(f"Coarse_{self.fine2coarse[label]}", coarse_id, parent="root")
#                 created_coarse_nodes.add(coarse_id)

#             # Create nodes for mid layer
#             mid_id = f"mid_{self.fine2mid[label]}"
#             if mid_id not in created_mid_nodes:
#                 tree.create_node(f"Mid_{self.fine2mid[label]}", mid_id, parent=coarse_id)
#                 created_mid_nodes.add(mid_id)

#             # Create nodes for fine samples
#             for idx, sample in enumerate(sample_group):
#                 sample_id = f"sample_{label}_{idx}"
#                 tree.create_node(f"Sample_{sample}", sample_id, parent=mid_id, data=idx)
#             lens = lens + len(sample_group)
        
#         print("len(created_coarse_nodes): ", len(created_coarse_nodes))
#         print("len(created_mid_nodes): ", len(created_mid_nodes))
#         print("sample len: ", lens)

#         return tree



class treeOT():
    # def __init__(self, dataset, samples, labels, lam=0.0001,nmax=100000, k=5, d=6, n_slice=1, debug_mode=False,is_sparse=False):
    def __init__(self, dataset, samples_A, label_A, samples_B, label_B, lam=0.0001,nmax=100000, k=5, d=6, n_slice=1, debug_mode=False,is_sparse=False):
        """
         Parameter
         ----------
         X :
             a set of supports
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

        # X = np.vstack(samples)
        X = np.vstack([samples_A, samples_B])

        # for i in tqdm(range(n_slice)):
        for i in range(n_slice):

            builder = HierarchicalTreeBuilder(dataset, layers=3)
            # tree = builder.build_tree(samples, labels)
            tree = builder.build_tree(samples_A, label_A, samples_B, label_B)

            Bsp = self.get_B_matrix(tree,X)
            # self.D1, self.D2 = self.gen_matrix(tree, X)

            if is_sparse:
                # wv = self.calc_weight_sparse(X, Bsp, samples, lam=lam, nmax=nmax)
                wv = self.calc_weight_sparse(X, Bsp, len(samples_A), len(samples_B), lam=lam, nmax=nmax)
            else:
                wv = self.calc_weight(X,Bsp,lam=lam,nmax=nmax)

            if i == 0:
                wB = Bsp.multiply(wv)
            else:
                wB = sparse.vstack([wB,Bsp.multiply(wv)])


        self.wB = wB


    def distance(self, A, B):
        return np.linalg.norm(A - B)

    def gen_matrix(self, tree, X):
        n_node = len(tree.all_nodes())
        n_leaf = X.shape[0]
        n_in = n_node - n_leaf
        D1 = np.zeros((n_in, n_in))
        D2 = np.zeros((n_in, n_leaf))

        in_node = [node.identifier for node in tree.all_nodes() if node.data == None]

        for node in tree.all_nodes():
            # check node is leaf or not
            if node.data is not None:
                parent_idx = in_node.index(tree.parent(node.identifier).identifier)
                D2[parent_idx, node.data] = 1.0
            elif node.identifier == tree.root:
                continue
            else:
                parent_idx = in_node.index(tree.parent(node.identifier).identifier)
                node_idx = in_node.index(node.identifier)
                D1[parent_idx, node_idx] = 1.0
        return D1, D2


    def get_B_matrix(self, tree, X):
        n_node = len(tree.all_nodes())
        n_leaf = X.shape[0]
        n_in   = n_node - n_leaf

        #B = np.zeros((n_node,n_leaf))

        in_node   = [node.identifier for node in tree.all_nodes() if node.data == None]
        in_node_index = [ii for ii in range(n_in)]
        leaf_node = [node.identifier for node in tree.all_nodes() if node.data != None]
        leaf_node_index = [node.data for node in tree.all_nodes() if node.data != None]
        #leaf_node_index = [node.data for node in tree.all_nodes() if node.data != None]
        path_leaves = tree.paths_to_leaves()

        n_edge = 0
        for path in path_leaves:
            n_edge += len(path)
        col_ind = np.zeros(n_edge)
        row_ind = np.zeros(n_edge)
        cnt = 0
        for path in path_leaves:
            # check node is leaf or not
            leaf_index = leaf_node_index[leaf_node.index(path[-1])]
            #B[leaf_index,leaf_index] = 1.0
            col_ind[cnt] = leaf_index
            row_ind[cnt] = leaf_index
            cnt += 1
            for node in path[:-1]:
                in_index = in_node_index[in_node.index(node)] + n_leaf
                #B[in_index,leaf_index] = 1.0
                col_ind[cnt] = leaf_index
                row_ind[cnt] = in_index
                cnt+=1

        B = sparse.csc_matrix((np.ones(n_edge), (row_ind, col_ind)), shape=(n_node, n_leaf), dtype='float32')
        # print("B.shape: ", B.shape)
        return B


    def calc_weight(self, X, B, lam=0.001, seed=0, nmax=100000):

        n_leaf, d = X.shape
        random.seed(seed)

        # Create B matrix
        n_in = self.D2.shape[0]
        #B1 = np.linalg.solve(np.eye(n_in) - self.D1, self.D2)
        #B = np.concatenate((B1, np.eye(n_leaf)))

        dz = B.shape[0]

        np.random.seed(seed)
        ind1 = np.random.randint(0, n_leaf, nmax)
        ind2 = np.random.randint(0, n_leaf, nmax)

        c_all = np.zeros((nmax, 1))
        Z_all = np.zeros((dz, nmax))

        for ii in range(nmax):
            c_all[ii] = np.linalg.norm(X[ind1[ii], :] - X[ind2[ii], :], ord=2)
            Z_all[:, ii] = B[:, ind1[ii]] + B[:, ind2[ii]] - 2 * (B[:, ind1[ii]] * B[:, ind2[ii]])

        n_sample = nmax
        c = np.asfortranarray(c_all[:n_sample, 0].reshape((n_sample, 1)), dtype='float32')
        Z = np.asfortranarray(Z_all[:, :n_sample].transpose(), dtype='float32')
        Zsp = sparse.csc_matrix(Z)

        # Solving nonnegative Lasso
        param = {'numThreads': -1, 'verbose': False,
                 'lambda1': lam, 'it0': 10, 'max_it': 2000, 'tol': 1e-3, 'intercept': False,
                 'pos': True}

        param['loss'] = 'square'
        param['regul'] = 'l1'

        W0 = np.zeros((Z.shape[1], c.shape[1]), dtype='float32', order="F")

        (W, optim_info) = spams.fistaFlat(c, Zsp, W0, True, **param)

        return W

    def calc_weight_in(self,X, Bsp, ind1, ind2):
        n = len(ind1)
        c_tmp = np.zeros(n)
        for ii in range(n):
            c_tmp[ii] = np.linalg.norm(X[ind1[ii], :] - X[ind2[ii], :], ord=2)
            tmp = Bsp[:, ind1[ii]] + Bsp[:, ind2[ii]] - 2 * (Bsp[:, ind1[ii]].multiply(Bsp[:, ind2[ii]]))

            if ii == 0:
                row_ind = tmp.indices
                col_ind = np.ones(len(row_ind)) * ii
                data = tmp[row_ind].toarray().flatten()
            else:
                row_ind_tmp = tmp.indices
                col_ind_tmp = np.ones(len(row_ind_tmp)) * ii
                row_ind = np.concatenate((row_ind, row_ind_tmp))
                col_ind = np.concatenate((col_ind, col_ind_tmp))
                data = np.concatenate((data, tmp[row_ind_tmp].toarray().flatten()))

        return c_tmp, col_ind, row_ind, data, len(data)

    def calc_weight_sparse(self,X, Bsp, len1, len2, lam=0.001, seed=0, nmax=100000, b=100):
        # print('in calc')
        n_leaf, d = X.shape
        random.seed(seed)

        c_all = np.zeros((nmax, 1))
        dz = Bsp.shape[0]

        np.random.seed(seed)
        # ind1 = np.random.randint(0, n_leaf, nmax)
        # ind2 = np.random.randint(0, n_leaf, nmax)

        ind1 = []
        ind2 = []
        for index_in_i in range(len1):
            for index_in_j in range(len2):
                ind1.append(index_in_i)
                ind2.append(len1 + index_in_j)
        
        nmax = len(ind1)
        # print("nmax: ", nmax)
        if b > nmax:
            b = nmax


        # Multi proces
        result = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(self.calc_weight_in)(X, Bsp, ind1[b * i:(i + 1) * b], ind2[b * i:(i + 1) * b]) for i in
            range(int(nmax / b)))

        # print("result: ", result)

        n_ele = 0
        for ii in range(int(nmax / b)):
            n_ele += result[ii][4]

        col_ind = np.zeros(n_ele)
        row_ind = np.zeros(n_ele)
        data = np.zeros(n_ele)
        st = 0
        ed = result[0][4]
        for ii in range(int(nmax / b) - 1):
            c_all[ii * b:(ii + 1) * b, 0] = result[ii][0]
            col_ind[st:ed] = result[ii][1] + (ii) * b
            row_ind[st:ed] = result[ii][2]
            data[st:ed] = result[ii][3]
            st += result[ii][4]
            ed += result[ii + 1][4]

        ii = int(nmax / b) - 1
        c_all[ii * b:(ii + 1) * b, 0] = result[ii][0]
        col_ind[st:ed] = result[ii][1] + (ii) * b
        row_ind[st:ed] = result[ii][2]
        data[st:ed] = result[ii][3]

        n_sample = nmax
        c = np.asfortranarray(c_all[:n_sample, 0].reshape((n_sample, 1)), dtype='float32')

        Zsp = sparse.csc_matrix((data, (col_ind, row_ind)), shape=(nmax, dz), dtype='float32')

        # Solving nonnegative Lasso
        param = {'numThreads': -1, 'verbose': False,
                 'lambda1': lam, 'it0': 10, 'max_it': 1000, 'tol': 1e-2, 'intercept': False,
                 'pos': True}

        param['loss'] = 'square'
        param['regul'] = 'l1'

        W0 = np.zeros((Zsp.shape[1], c.shape[1]), dtype='float32', order="F")

        (W, optim_info) = spams.fistaFlat(c, Zsp, W0, True, **param)
        # print("W: ", W)

        return W



    def pairwiseTWD(self,a,b):
        # Compute the Tree Wasserstein

        TWD = abs(self.wB.dot(a - b)).sum(0) / self.n_slice

        return TWD