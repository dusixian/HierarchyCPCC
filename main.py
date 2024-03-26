import wandb

import torch
from torch import Tensor 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import re
import ot
from itertools import combinations
from tqdm import tqdm


from sklearn.metrics import roc_auc_score, silhouette_score, average_precision_score
from sklearn.metrics import precision_score, recall_score, label_ranking_average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist
from scipy.stats.mstats import pearsonr
from scipy.spatial.distance import cdist

# Official implementation from: https://github.com/fiveai/making-better-mistakes
from better_mistakes.model.losses import HierarchicalCrossEntropyLoss
from better_mistakes.trees import get_weighting
from better_mistakes.model.labels import make_all_soft_labels, make_batch_soft_labels

from datetime import datetime
import argparse
import os
import json
from typing import *

import matplotlib.pyplot as plt
import seaborn as sns

from model import init_model
from data import make_kshot_loader, make_dataloader
from loss import CPCCLoss, QuadrupletLoss, GroupLasso
from emd.emd_methods import SlicedWasserstein_np, sinkhorn, compute_flow_symmetric
from param import init_optim_schedule, load_params
from utils import get_layer_prob_from_fine, seed_everything


def pretrain_objective(train_loader : DataLoader, test_loader : DataLoader, device : torch.device, 
               save_dir : str, seed : int, split : str, CPCC : bool, exp_name : str, epochs : int,
               task : str, dataset_name : str, breeds_setting : str, hyper) -> None:
    '''
    Pretrain session wrapper. Use extra epochs for curriculum learning.
    Args:
        train_loader : dataset train loader.
        test_loader : dataset test loader
        device : cpu/gpu
        save_dir : directory to save model checkpoint
        seed : random state
        split : split/full
        CPCC : use CPCC as a regularizer or not
        exp_name : experiment name
        epochs : number of training iterations for all experiments except for curriculum
        dataset_name : MNIST/CIFAR/BREEDS
    '''
    def one_stage_pretrain(train_loader : DataLoader, test_loader : DataLoader, 
                     device : torch.device, save_dir : str, seed : int, split : str, 
                     CPCC : bool, exp_name : str, epochs : int, task : str='') -> nn.Module:
        '''
        Main train/test loop for all experiments except for curriculum learning.
        Args:
            train_loader : dataset train loader.
            test_loader : dataset test loader
            device : cpu/gpu
            save_dir : directory to save model checkpoint
            seed : random state
            split : split/full
            CPCC : use CPCC as a regularizer or not
            exp_name : experimental setting, can't be curriculum
            epochs : number of training iterations
            task : in/sub
        '''
        # ================= SETUP STARTS =======================
        assert 'curriculum' not in exp_name, 'Invalid experiment'
        if task == 'in':
            dataset = train_loader.dataset.dataset
        else:
            dataset = train_loader.dataset
        if train_on_mid:
            classes = dataset.mid_names
            num_classes = [len(dataset.mid_names)]
        elif coarse_ce:
            classes = dataset.coarse_names
            num_classes = [len(dataset.coarse_names)]
        else:
            classes = dataset.fine_names
        
            if 'MTL' in exp_name:
                num_classes = [len(dataset.coarse_names), len(dataset.fine_names)]
            else:
                num_classes = [len(dataset.fine_names)]
        
        if train_on_mid:
            coarse_targets_map = dataset.mid2coarse
        else:
            coarse_targets_map = dataset.coarse_map
        mid_targets_map = dataset.mid_map
        num_train_batches = len(train_loader.dataset) // train_loader.batch_size + 1
        last_train_batch_size = len(train_loader.dataset)  % train_loader.batch_size
        num_test_batches = len(test_loader.dataset) // test_loader.batch_size + 1
        last_test_batch_size = len(test_loader.dataset)  % test_loader.batch_size

        if CPCC:
            exp_name = exp_name + 'CPCC'

        if group:
            exp_name = exp_name + 'Group'

        split = split + task

        optim_config, scheduler_config = hyper['optimizer'], hyper['scheduler']
        init_config = {"dataset":dataset_name,
                    "exp_name":exp_name,
                    "_batch_size":train_loader.batch_size,
                    "epochs":epochs,
                    "_optimizer":'SGD',
                    "_scheduler":'StepLR',
                    "seed":seed, # will be replaced until we reach the max seeds
                    "save_dir":save_dir,
                    "_num_workers":train_loader.num_workers,
                    "split":split,
                    "lamb":lamb,
                    "breeds_setting":breeds_setting,
                    "train_on_mid": train_on_mid,
                    "is_emd": is_emd,
                    "reg": reg,
                    "numItermax": numItermax,
                    "n_projections": n_projections
                    }
        if CPCC:
            init_config['cpcc_metric'] = cpcc_metric
        
        config = {**init_config, **optim_config, **scheduler_config}        

        # torch.autograd.set_detect_anomaly(True)
        if 'HXE' in exp_name:
            hierarchy = dataset.build_nltk_tree()
            alpha = 0.4
            config['alpha'] = alpha
            weights = get_weighting(hierarchy, "exponential", value=alpha, normalize=False)
            criterion_ce = HierarchicalCrossEntropyLoss(hierarchy, classes, weights).to(device)
        elif 'soft' in exp_name:
            beta = 10
            config['beta'] = beta
            distances = dataset.make_distances()
            all_soft_labels = make_all_soft_labels(distances, list(range(num_classes[0])), beta)
            criterion_ce = nn.KLDivLoss(reduction='batchmean').to(device) # "entropy flavor", soft version
        elif 'quad' in exp_name:
            m1 = 0.25
            m2 = 0.15
            lambda_s = 0.8
            config['m1'] = m1
            config['m2'] = m2
            config['lambda_s'] = lambda_s
            criterion_quad = QuadrupletLoss(train_loader.dataset, m1, m2).to(device)
            criterion_ce = nn.CrossEntropyLoss().to(device)
        else:
            criterion_ce = nn.CrossEntropyLoss().to(device)
        criterion_cpcc = CPCCLoss(dataset, is_emd, train_on_mid, reg, numItermax, n_projections, cpcc_layers, cpcc_metric).to(device) 
        criterion_group = GroupLasso(dataset).to(device)
        
        with open(save_dir+'/config.json', 'w') as fp:
            json.dump(config, fp, sort_keys=True, indent=4)
        
        out_dir = save_dir+f"/{split}_seed{seed}.pth"

        if os.path.exists(out_dir):
            print("Skipped.")
            return
            # msg = input("File Exists. Override the model? (y/n)")
            # if msg.lower() == 'n':
            #     print("Skipped.")
            #     return
            # elif msg.lower() == 'y':
            #     print("Retrain model.")
        
        wandb.init(project=f"{dataset_name}_onestage_pretrain", 
                entity="structured_task",
                name=datetime.now().strftime("%m%d%Y%H%M%S"),
                config=config,
                settings=wandb.Settings(code_dir="."))

        model = init_model(dataset_name, num_classes, device)
        optimizer, scheduler = init_optim_schedule(model, hyper)
        
        # ================= SETUP ENDS =======================

        def get_different_loss(exp_name : str, model : nn.Module, data : Tensor, 
                        criterion_ce : nn.Module, target_one : Tensor, target_coarse : Tensor, 
                        coarse_targets_map : list, idx : int, num_batches : int, 
                        last_batch_size : int, num_classes : int, 
                        all_soft_labels=None, lambda_s : float = None, criterion_quad : nn.Module = None) -> Tuple[Tensor, Tensor, Tensor]:
            '''
                Helper to calculate non CPCC loss, also return representation and model raw output
            '''
            # if coarse_ce: 
            #     representation, output_fine = model(data)
            #     prob_fine = F.softmax(output_fine,dim=1)
            #     prob_coarse = get_layer_prob_from_fine(prob_fine, coarse_targets_map)
            #     loss_ce_coarse = F.nll_loss(torch.log(prob_coarse), target_coarse)
            #     return representation, output_fine, loss_ce_coarse
            # else: 
            representation, output_one = model(data)
            loss_ce = criterion_ce(output_one, target_one)
            return representation, output_one, loss_ce

            # if 'MTL' in exp_name:
            #     representation, output_coarse, output_fine = model(data)
            #     loss_ce_coarse = criterion_ce(output_fine, target_fine)
            #     loss_ce_fine = criterion_ce(output_coarse, target_coarse)
            #     loss_ce = loss_ce_coarse + loss_ce_fine
            # else:
            #     representation, output_fine = model(data)
            #     if 'sumloss' in exp_name:
            #         loss_ce_fine = criterion_ce(output_fine, target_fine)
            #         prob_fine = F.softmax(output_fine,dim=1)
            #         prob_coarse = get_layer_prob_from_fine(prob_fine, coarse_targets_map)
            #         loss_ce_coarse = F.nll_loss(torch.log(prob_coarse), target_coarse)
            #         loss_ce = loss_ce_fine + loss_ce_coarse
            #     elif 'soft' in exp_name:
            #         if idx == num_batches - 1:
            #             target_distribution = make_batch_soft_labels(all_soft_labels, target_fine, num_classes, last_batch_size, device)
            #         else:
            #             target_distribution = make_batch_soft_labels(all_soft_labels, target_fine, num_classes, batch_size, device)
            #         prob_fine = F.softmax(output_fine,dim=1)
            #         loss_ce = criterion_ce(prob_fine.log(), target_distribution)
            #     elif 'quad' in exp_name:
            #         loss_quad = criterion_quad(F.normalize(representation, dim=-1), target_fine)
            #         loss_cee = criterion_ce(output_fine, target_fine)
            #         loss_ce = (1 - lambda_s) * loss_quad + lambda_s * loss_cee
            #     else:
            #         loss_ce = criterion_ce(output_fine, target_fine)
            # return representation, output_fine, loss_ce

        # Check if a checkpoint exists at the start
        checkpoint_filepath = ''
        start_epoch = 0
        max_epoch_num = -1
        regex = re.compile(r'\d+')

        for file in os.listdir(checkpoint_dir):
            if file.endswith(f"{seed}.pth"):
                epoch_num = int(regex.findall(file)[0])
                if epoch_num > max_epoch_num:
                    max_epoch_num = epoch_num
                    checkpoint_filepath = os.path.join(checkpoint_dir, file)

        if is_down:
            checkpoint = torch.load(checkpoint_filepath)
            model.load_state_dict(checkpoint['model_state_dict'])
            current_epoch = checkpoint['epoch']
            torch.save(model.state_dict(), out_dir)
            wandb.finish()
            print(f"Loaded checkpoint from epoch {current_epoch}, model saved!")
            return


        if max_epoch_num != -1:
            checkpoint = torch.load(checkpoint_filepath)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

        # epochs_no_improve = 0 
        # min_val_loss = np.Inf
        # early_stop_patience = 15
        epochs_durations = []
        
        for epoch in range(start_epoch, epochs):
            t_start = datetime.now() # record the time for each epoch
            model.train()
            # train_fine_accs = []
            train_one_accs = []
            train_coarse_accs = []
            train_losses_ce = []
            train_losses_cpcc = []
            train_losses_group = []
            total_func_time = np.array([0.0,0.0,0.0,0.0,0.0, 0.0])
            last_time_checkpoint = datetime.now()
            
            for idx, (data, _, target_coarse, target_mid, target_fine) in enumerate(train_loader):
                data = data.to(device)
                # target_fine = target_fine.to(device)
                if train_on_mid:
                    target_one = target_mid.to(device)
                elif coarse_ce:
                    target_one = target_coarse.to(device)
                else:
                    target_one = target_fine.to(device)
                target_coarse = target_coarse.to(device)
                optimizer.zero_grad()
                
                if 'soft' in exp_name:
                    representation, output_fine, loss_ce = get_different_loss(exp_name, model, data, criterion_ce, target_fine, target_coarse, 
                                            coarse_targets_map, idx, num_train_batches, 
                                            last_train_batch_size, num_classes[-1], 
                                            all_soft_labels=all_soft_labels)
                elif 'quad' in exp_name:
                    representation, output_fine, loss_ce = get_different_loss(exp_name, model, data, criterion_ce, target_fine, target_coarse, 
                                            coarse_targets_map, idx, num_train_batches, 
                                            last_train_batch_size, num_classes[-1], 
                                            lambda_s=lambda_s, criterion_quad=criterion_quad)
                else:
                    # representation, output_fine, loss_ce = get_different_loss(exp_name, model, data, criterion_ce, target_fine, target_coarse, 
                    #                         coarse_targets_map, idx, num_train_batches, 
                    #                         last_train_batch_size, num_classes[-1])
                    representation, output_one, loss_ce = get_different_loss(exp_name, model, data, criterion_ce, target_one, target_coarse, 
                                            coarse_targets_map, idx, num_train_batches, 
                                            last_train_batch_size, num_classes[-1])
                
                if CPCC:
                    loss_cpcc = lamb * criterion_cpcc(representation, target_one)
                    # if is_emd == 4:
                    #     elapsed_time_cpcc = criterion_cpcc.time
                    loss = loss_ce + loss_cpcc
                    train_losses_cpcc.append(loss_cpcc)
                elif group:
                    loss_group = criterion_group(model.module.fc.weight, model.module.fc.bias)
                    loss = loss_ce + loss_group
                    train_losses_group.append(loss_group)
                else:
                    loss = loss_ce
                train_losses_ce.append(loss_ce)
                loss.backward()
                optimizer.step()
            
                prob_one = F.softmax(output_one,dim=1)
                pred_one = prob_one.argmax(dim=1)
                acc_one = pred_one.eq(target_one).flatten().tolist()
                train_one_accs.extend(acc_one)
                # if is_emd == 4:
                #     total_func_time += elapsed_time_cpcc

                if not coarse_ce:
                    prob_coarse = get_layer_prob_from_fine(prob_one, coarse_targets_map)
                    pred_coarse = prob_coarse.argmax(dim=1)
                    acc_coarse = pred_coarse.eq(target_coarse).flatten().tolist()
                    train_coarse_accs.extend(acc_coarse)

                if idx % 100 == 1:
                    if not CPCC:
                        loss_cpcc = -1
                    if train_on_mid:
                        print(f"Train Loss: {loss}, Acc_mid: {sum(train_one_accs)/len(train_one_accs)}, Acc_coarse: {sum(train_coarse_accs)/len(train_coarse_accs)}, loss_cpcc: {loss_cpcc}")
                    elif coarse_ce:
                        print(f"Train Loss: {loss}, Acc_coarse: {sum(train_one_accs)/len(train_one_accs)}, loss_cpcc: {loss_cpcc}")
                    else:
                        print(f"Train Loss: {loss}, Acc_fine: {sum(train_one_accs)/len(train_one_accs)}, Acc_coarse: {sum(train_coarse_accs)/len(train_coarse_accs)}, loss_cpcc: {loss_cpcc}")
                    if is_emd == 4:
                        current_time_checkpoint = datetime.now()
                        # 计算从上一个时间戳到现在的总时间
                        total_time = (current_time_checkpoint - last_time_checkpoint).total_seconds()

                        print(f"Total time since last checkpoint: {total_time} seconds")
                        
                        # 重置上一次的时间戳和函数时间
                        last_time_checkpoint = current_time_checkpoint
                    if is_emd == 4:
                        print('total_func_time: ', total_func_time)
                        total_func_time = np.array([0.0,0.0,0.0,0.0,0.0, 0.0])
            
            scheduler.step()

            # Save the model every certain number of epochs
            if epoch % 10 == 0:
                checkpoint_filepath = os.path.join(checkpoint_dir, f"checkpoint_{epoch}_{seed}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }, checkpoint_filepath)
            
            model.eval() 
            # test_fine_accs = []
            test_one_accs = []
            test_coarse_accs = []
            test_losses_ce = []
            test_losses_cpcc = []
            test_losses_group = []
            
            with torch.no_grad():
                for idx, (data, _, target_coarse, target_mid, target_fine) in enumerate(test_loader):
                    data = data.to(device)
                    target_coarse = target_coarse.to(device)
                    # target_fine = target_fine.to(device)
                    if train_on_mid:
                        target_one = target_mid.to(device)
                    elif coarse_ce:
                        target_one = target_coarse.to(device)
                    else:
                        target_one = target_fine.to(device)
                    
                    if 'soft' in exp_name:
                        representation, output_fine, loss_ce = get_different_loss(exp_name, model, data, criterion_ce, target_fine, target_coarse, 
                                                coarse_targets_map, idx, num_test_batches, 
                                                last_test_batch_size, num_classes[-1], 
                                                all_soft_labels=all_soft_labels)
                    elif 'quad' in exp_name:
                        representation, output_fine, loss_ce = get_different_loss(exp_name, model, data, criterion_ce, target_fine, target_coarse, 
                                                coarse_targets_map, idx, num_test_batches, 
                                                last_test_batch_size, num_classes[-1], 
                                                lambda_s=lambda_s, criterion_quad=criterion_quad)
                    else:
                        representation, output_one, loss_ce = get_different_loss(exp_name, model, data, criterion_ce, target_one, target_coarse, 
                                                coarse_targets_map, idx, num_test_batches, 
                                                last_test_batch_size, num_classes[-1])
                    
                    if CPCC:
                        loss_cpcc = lamb * criterion_cpcc(representation, target_one)
                        loss = loss_ce + loss_cpcc
                        test_losses_cpcc.append(loss_cpcc)
                    elif group:
                        loss_group = criterion_group(model.module.fc.weight, model.module.fc.bias)
                        loss = loss_ce + loss_group
                        test_losses_group.append(loss_group)
                    else:
                        loss = loss_ce
                    test_losses_ce.append(loss_ce)

                    prob_one = F.softmax(output_one,dim=1)
                    pred_one = prob_one.argmax(dim=1)
                    acc_one = pred_one.eq(target_one).flatten().tolist()
                    test_one_accs.extend(acc_one)

                    if not coarse_ce:
                        prob_coarse = get_layer_prob_from_fine(prob_one, coarse_targets_map)
                        pred_coarse = prob_coarse.argmax(dim=1)
                        acc_coarse = pred_coarse.eq(target_coarse).flatten().tolist()
                        test_coarse_accs.extend(acc_coarse)
            
            t_end = datetime.now()
            t_delta = (t_end-t_start).total_seconds()
            if coarse_ce:
                print(f"Val loss_ce: {sum(test_losses_ce)/len(test_losses_ce)}, Acc_coarse: {sum(test_one_accs)/len(test_one_accs)}")
            else:
                print(f"Val loss_ce: {sum(test_losses_ce)/len(test_losses_ce)}, Acc_fine: {sum(test_one_accs)/len(test_one_accs)}, Acc_coarse: {sum(train_coarse_accs)/len(train_coarse_accs)}")
            print(f"Epoch {epoch} takes {t_delta} sec.")
            epochs_durations.append(t_delta)

            log_dict = {"train_one_acc":sum(train_one_accs)/len(train_one_accs),
                        "train_losses_ce":sum(train_losses_ce)/len(train_losses_ce),
                        "val_one_acc":sum(test_one_accs)/len(test_one_accs),
                        "val_losses_ce":sum(test_losses_ce)/len(test_losses_ce),
                    }
            if CPCC: # batch-CPCC
                log_dict["train_losses_cpcc"] = sum(train_losses_cpcc)/len(train_losses_cpcc)
                log_dict["val_losses_cpcc"] = sum(test_losses_cpcc)/len(test_losses_cpcc)
            if group:
                log_dict["train_losses_group"] = sum(train_losses_group)/len(train_losses_group)
                log_dict["val_losses_group"] = sum(test_losses_group)/len(test_losses_group)
            
            wandb.log(log_dict)
            val_loss = sum(test_losses_ce)/len(test_losses_ce)
        

        # checkpoint = torch.load(save_dir+'/best_model.pth')
        # model.load_state_dict(checkpoint['model_state_dict'])
        torch.save(model.state_dict(), out_dir)
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        wandb.finish()

        avg_epoch_duration = sum(epochs_durations) / len(epochs_durations)
        print(f"Average epoch duration: {avg_epoch_duration} sec.")
        # print(f"Best val loss: {checkpoint['loss']}, at epoch {checkpoint['epoch']}")

        # Log to file
        with open(save_dir+f'/training_log_{seed}.txt', 'w') as f:
            f.write(f"Average epoch duration: {avg_epoch_duration} sec.\n")
            # f.write(f"Best val loss: {checkpoint['loss']}, at epoch {checkpoint['epoch']}\n")
            # f.write(f"epochs_durations: {epochs_durations} \n")

        return model

    def curriculum_pretrain(train_loader : DataLoader, test_loader : DataLoader, 
                        device : torch.device, save_dir : str, seed : int, 
                        split : str, CPCC : bool, exp_name : str, epochs : int, 
                        task : str) -> nn.Module:
        '''
        Main train/test loop for curriculum learning.
        Args:
            train_loader : dataset train loader.
            test_loader : dataset test loader
            device : cpu/gpu
            save_dir : directory to save model checkpoint
            seed : random state
            split : split/full
            CPCC : use CPCC as a regularizer or not
            exp_name : curriculum or curriculumCPCC
            epochs : number of training iterations, note there are extra epochs for curriculum
            compared to other experiments
        '''
        assert 'curriculum' in exp_name, 'Invalid experiment'
        if task == 'in':
            dataset = train_loader.dataset.dataset
        else:
            dataset = train_loader.dataset
        num_fine_classes = [len(dataset.fine_names)]
        num_coarse_classes = [len(dataset.coarse_names)]
        coarse_targets_map = dataset.coarse_map

        split = split + task

        if CPCC:
            exp_name = exp_name + 'CPCC'


        init_config = {"dataset":dataset_name,
                    "exp_name":exp_name,
                    "_batch_size":train_loader.batch_size,
                    "epochs":epochs,
                    "_optimizer":'SGD',
                    "_scheduler":'StepLR',
                    "seed":seed,
                    "save_dir":save_dir,
                    "_num_workers":train_loader.num_workers,
                    "split":split,
                    "lamb":lamb,
                    "breeds_setting":breeds_setting,
                    }
        if CPCC:
            init_config['cpcc_metric'] = cpcc_metric
        
        optim_config, scheduler_config = hyper['optimizer'], hyper['scheduler']
        config = {**init_config, **optim_config, **scheduler_config}  
        
        criterion_ce = nn.CrossEntropyLoss().to(device)
        criterion_cpcc = CPCCLoss(dataset, is_emd, train_on_mid, cpcc_layers, cpcc_metric).to(device)

        with open(save_dir+'/config.json', 'w') as fp:
            json.dump(config, fp, sort_keys=True, indent=4)
        
        out_dir = save_dir+f"/{split}_seed{seed}.pth" # fine model

        if os.path.exists(out_dir):
            print("Skipped.")
            return
            # msg = input("File Exists. Override the model? (y/n)")
            # if msg.lower() == 'n':
            #     print("Skipped.")
            #     return
            # elif msg.lower() == 'y':
            #     print("Retrain model.")
        
        # Step 1: train 20% epochs for coarse class only
        model = init_model(dataset_name, num_coarse_classes, device)
        optimizer, scheduler = init_optim_schedule(model, hyper)

        for epoch in range(int(epochs*0.2)):
            t_start = datetime.now()
            # ============== Stage 1 Train =================
            model.train()
            train_coarse_accs = []
            train_coarse_losses = []
            for idx, (data, _, target_coarse, _, _)  in enumerate(train_loader):
                data = data.to(device)
                target_coarse = target_coarse.to(device)
                optimizer.zero_grad()
                _, output_coarse = model(data) # we only add CPCC for the second stage training on fine level, no need for representation
                loss_ce = criterion_ce(output_coarse, target_coarse)
                loss_ce.backward()
                optimizer.step()
                prob_coarse = F.softmax(output_coarse,dim=1)
                pred_coarse = prob_coarse.argmax(dim=1, keepdim=False)
                acc_coarse = pred_coarse.eq(target_coarse).flatten().tolist()
                train_coarse_accs.extend(acc_coarse)
                train_coarse_losses.append(loss_ce)
                if idx % 100 == 1:
                    print(f"Train Loss: {loss_ce}, Acc_coarse: {sum(train_coarse_accs)/len(train_coarse_accs)}")
            scheduler.step()
            # ============== Stage 2 Test =================
            model.eval() 
            test_coarse_losses = []
            test_coarse_accs = []
            with torch.no_grad():
                for (data, _, target_coarse, _, _) in test_loader:
                    data = data.to(device)
                    target_coarse = target_coarse.to(device)
                    _, output_coarse = model(data)
                    loss_ce = criterion_ce(output_coarse, target_coarse)
                    test_coarse_losses.append(loss_ce)
                    prob_coarse = F.softmax(output_coarse,dim=1)
                    pred_coarse = prob_coarse.argmax(dim=1, keepdim=False)
                    acc_coarse = pred_coarse.eq(target_coarse).flatten().tolist()
                    test_coarse_accs.extend(acc_coarse)
            t_end = datetime.now()
            t_delta = (t_end-t_start).total_seconds()
            print(f"Val loss: {sum(test_coarse_losses)/len(test_coarse_losses)}, Acc_coarse: {sum(test_coarse_accs)/len(test_coarse_accs)}")
            print(f"Epoch {epoch} takes {t_delta} sec.")
        torch.save(model.state_dict(), save_dir+f"/{split}_coarse_seed{seed}.pth")


        wandb.init(project=f"{dataset_name}_onestage_pretrain", # reset, log to one stage
                entity="structured_task",
                name=datetime.now().strftime("%m%d%Y%H%M%S"),
                config=config,
                settings=wandb.Settings(code_dir="."))
        
        # Step 2: train 80% epochs for fine class only
        model = init_model(dataset_name, num_fine_classes, device)
        model_dict = model.state_dict()
        coarse_dict = {k: v for k, v in torch.load(save_dir+f"/{split}_coarse_seed{seed}.pth").items() if (k in model_dict) and ('fc' not in k)}
        model_dict.update(coarse_dict) 
        model.load_state_dict(model_dict)
        # reset optimizer and scheduler
        optimizer, scheduler = init_optim_schedule(model, hyper)
        for epoch in range(epochs - int(epochs*0.2)):
            t_start = datetime.now()
            # ============== Stage 2 Train =================
            model.train()
            train_fine_accs = []
            train_coarse_accs = []
            train_losses_ce = []
            train_losses_cpcc = []

            for idx, (data, _, target_coarse, _, target_fine) in enumerate(train_loader):
                data = data.to(device)
                target_fine = target_fine.to(device)
                target_coarse = target_coarse.to(device)

                optimizer.zero_grad()
                representation, output_fine = model(data)
                loss_ce = criterion_ce(output_fine, target_fine)

                if CPCC:
                    loss_cpcc = lamb * criterion_cpcc(representation, target_fine)
                    loss = loss_ce + loss_cpcc
                    train_losses_cpcc.append(loss_cpcc)
                else:
                    loss = loss_ce
                train_losses_ce.append(loss_ce)

                loss.backward()
                optimizer.step()

                prob_fine = F.softmax(output_fine,dim=1)
                pred_fine = prob_fine.argmax(dim=1)
                acc_fine = pred_fine.eq(target_fine).flatten().tolist()
                train_fine_accs.extend(acc_fine)

                prob_coarse = get_layer_prob_from_fine(prob_fine, coarse_targets_map)
                pred_coarse = prob_coarse.argmax(dim=1)
                acc_coarse = pred_coarse.eq(target_coarse).flatten().tolist()
                train_coarse_accs.extend(acc_coarse)

                if idx % 100 == 1:
                    print(f"Train Loss: {loss}, Acc_fine: {sum(train_fine_accs)/len(train_fine_accs)}")

            scheduler.step()

            # ============== Stage 2 Test =================
            model.eval() 
            test_fine_accs = []
            test_coarse_accs = []
            test_losses_ce = []
            test_losses_cpcc = []

            with torch.no_grad():
                for idx, (data, _, target_coarse, _, target_fine) in enumerate(test_loader):
                    data = data.to(device)
                    target_coarse = target_coarse.to(device)
                    target_fine = target_fine.to(device)

                    representation, output_fine = model(data)
                    loss_ce = criterion_ce(output_fine, target_fine)

                    if CPCC:
                        loss_cpcc = lamb * criterion_cpcc(representation, target_fine)
                        loss = loss_ce + loss_cpcc
                        test_losses_cpcc.append(loss_cpcc)
                    else:
                        loss = loss_ce
                    test_losses_ce.append(loss_ce)

                    prob_fine = F.softmax(output_fine,dim=1)
                    pred_fine = prob_fine.argmax(dim=1)
                    acc_fine = pred_fine.eq(target_fine).flatten().tolist()
                    test_fine_accs.extend(acc_fine)

                    prob_coarse = get_layer_prob_from_fine(prob_fine, coarse_targets_map)
                    pred_coarse = prob_coarse.argmax(dim=1)
                    acc_coarse = pred_coarse.eq(target_coarse).flatten().tolist()
                    test_coarse_accs.extend(acc_coarse)

            t_end = datetime.now()
            t_delta = (t_end-t_start).total_seconds()
            print(f"Val loss_ce: {sum(test_losses_ce)/len(test_losses_ce)}, Acc_fine: {sum(test_fine_accs)/len(test_fine_accs)}")
            print(f"Epoch {epoch} takes {t_delta} sec.")

            log_dict = {"train_fine_acc":sum(train_fine_accs)/len(train_fine_accs),
                        "train_losses_ce":sum(train_losses_ce)/len(train_losses_ce),
                        "val_fine_acc":sum(test_fine_accs)/len(test_fine_accs),
                        "val_losses_ce":sum(test_losses_ce)/len(test_losses_ce),
                    }
            
            if CPCC:
                log_dict["train_losses_cpcc"] = sum(train_losses_cpcc)/len(train_losses_cpcc)
                log_dict["val_losses_cpcc"] = sum(test_losses_cpcc)/len(test_losses_cpcc)
            
            wandb.log(log_dict)

        torch.save(model.state_dict(), out_dir)
        wandb.finish()
        return model

    if 'curriculum' in exp_name:
        if dataset_name == 'CIFAR':
            extra_epochs = int(epochs * 1.25) # default : 250
        elif dataset_name == 'MNIST':
            extra_epochs = int(epochs * 1.2)
        elif dataset_name == 'BREEDS':
            if breeds_setting == 'living17' or breeds_setting == 'nonliving26':
                extra_epochs = int(epochs * 1.2)
            elif breeds_setting == 'entity13' or breeds_setting == 'entity30':
                extra_epochs = int(epochs * 1.25)
        curriculum_pretrain(train_loader, test_loader, device, save_dir, seed, split, CPCC, exp_name, extra_epochs, task)
    else:
        one_stage_pretrain(train_loader, test_loader, device, save_dir, seed, split, CPCC, exp_name, epochs, task)
    return

def downstream_transfer(save_dir : str, seed : int, device : torch.device, 
                        batch_size : int, level : str, CPCC : bool,
                        exp_name : str, num_workers : int, task : str, 
                        dataset_name : str, case : int, breeds_setting : str,
                        hyper, epochs) -> nn.Module:
    '''
        Transfer to target sets on new level.
    '''
    out_dir = save_dir+f"/down{task}_{level}_seed{seed}.pth"

    if os.path.exists(out_dir):
        print("Skipped.")
        return

    train_loader, test_loader = make_kshot_loader(num_workers, batch_size, 1, level, 
                                                task, seed, dataset_name, case, breeds_setting, difficulty) # we use one shot on train set, tt dataloader
    dataset = train_loader.dataset.dataset # loader contains Subset
    if level == 'fine':
        num_classes = len(dataset.fine_names)
    elif level == 'mid':
        num_classes = len(dataset.mid_names)
    elif level == 'coarse':
        num_classes = len(dataset.coarse_names)
    elif level == 'coarsest':
        num_classes = len(dataset.coarsest_names)
    train_size = len(train_loader.dataset)
    test_size = len(test_loader.dataset)
    criterion = nn.CrossEntropyLoss().to(device) # no CPCC in downstream task
    
    model = init_model(dataset_name, [num_classes], device)
    
    model_dict = model.state_dict()
    # load pretrained seed 0, call this function 
    trained_dict = {k: v for k, v in torch.load(save_dir+f"/split{task}_seed0.pth").items() if (k in model_dict) and ("fc" not in k)}
    model_dict.update(trained_dict) 
    model.load_state_dict(model_dict)

    for param in model.parameters(): # Freeze Encoder, fit last linear layer
        param.requires_grad = False
    model.module.fc = nn.Linear(model.module.out_features, num_classes).to(device)

    if CPCC:
        exp_name = exp_name + 'CPCC'
    
    init_config = {"_batch_size":train_loader.batch_size,
                    "epochs":epochs,
                    "_optimizer":'SGD',
                    "_scheduler":'StepLR',
                    "seed":seed,
                    "save_dir":save_dir,
                    "exp_name":exp_name,
                    "_num_workers":train_loader.num_workers,
                    "new_level":level,
                    "task":task,
                    "breeds_setting":breeds_setting,
                    }
    if CPCC:
            init_config['cpcc_metric'] = cpcc_metric
    
    optim_config, scheduler_config = hyper['optimizer'], hyper['scheduler']
    config = {**init_config, **optim_config, **scheduler_config}  

    wandb.init(project=f"{dataset_name}-subpop", # {dataset_name}-subpop/new-level-tuning
               entity="structured_task",
               name=datetime.now().strftime("%m%d%Y%H%M%S"),
               config=config,
               settings=wandb.Settings(code_dir=".")
    )
    
    optimizer, scheduler = init_optim_schedule(model, hyper)

    for epoch in range(epochs):
        t_start = datetime.now()
        model.eval() 
        test_top1 = 0
        test_top2 = 0
        test_losses = []
        
        with torch.no_grad():
            for (data, target_coarsest, target_coarse, target_mid, target_fine) in test_loader:
                data = data.to(device)
                target_coarse = target_coarse.to(device)
                target_fine = target_fine.to(device)
                target_mid = target_mid.to(device)
                target_coarsest = target_coarsest.to(device)
                
                if level == 'coarsest':
                    target = target_coarsest
                elif level == 'mid':
                    target = target_mid
                elif level == 'coarse':
                    target = target_coarse
                elif level == 'fine':
                    target = target_fine

                _, output = model(data)
                loss = criterion(output, target)
                test_losses.append(loss)

                prob = F.softmax(output,dim=1)

                # top 1
                pred1 = prob.argmax(dim=1, keepdim=False) 
                top1_correct = pred1.eq(target).sum()
                test_top1 += top1_correct

                # top 2
                pred2 = (prob.topk(k=2, dim=1)[1]).T # 5 * batch_size
                target_reshaped = target.unsqueeze(0).expand_as(pred2)
                top2_correct = pred2.eq(target_reshaped).sum() 
                test_top2 += top2_correct
        
        print(f"Val loss: {sum(test_losses)/len(test_losses)}, Top1_{level}: {test_top1/test_size} "
              f"Top2_{level} : {test_top2/test_size}")

        model.train()
        train_top1 = 0
        train_top2 = 0
        train_losses = []
        for idx, (data, target_coarsest, target_coarse, target_mid, target_fine) in enumerate(train_loader):
            data = data.to(device)
            target_coarse = target_coarse.to(device)
            target_fine = target_fine.to(device)
            target_mid = target_mid.to(device)
            target_coarsest = target_coarsest.to(device)
            
            if level == 'coarsest':
                target = target_coarsest
            elif level == 'mid':
                target = target_mid
            elif level == 'coarse':
                target = target_coarse
            elif level == 'fine':
                target = target_fine

            optimizer.zero_grad()
            _, output = model(data)
            loss = criterion(output, target)
            train_losses.append(loss)
    
            loss.backward()
            optimizer.step()
            
            prob = F.softmax(output,dim=1)

            pred1 = prob.argmax(dim=1, keepdim=False) 
            top1_correct = pred1.eq(target).sum()
            train_top1 += top1_correct

            pred2 = (prob.topk(k=2, dim=1)[1]).T 
            target_reshaped = target.unsqueeze(0).expand_as(pred2)
            top2_correct = pred2.eq(target_reshaped).sum() 
            train_top2 += top2_correct
            
            if idx % 100 == 0:
                print(f"Train loss: {sum(train_losses)/len(train_losses)}, Top1_{level}: {train_top1/train_size} "
                      f"Top2_{level} : {train_top2/train_size}")

        scheduler.step()
        
        t_end = datetime.now()
        t_delta = (t_end-t_start).total_seconds()
        print(f"Epoch {epoch} takes {t_delta} sec.")
        
        wandb.log({
            "train_top1":train_top1/train_size,
            "train_top2":train_top2/train_size,
            "train_losses":sum(train_losses)/len(train_losses),
            "val_top1":test_top1/test_size,
            "val_top2":test_top2/test_size,
            "val_losses":sum(test_losses)/len(test_losses),
        })
    
    torch.save(model.state_dict(), save_dir+f"/down{task}_{level}_seed{seed}.pth")
    wandb.finish()
    
    return model

def retrieve_downstream_metrics(save_dir : str, seeds : int, device : torch.device, 
                                batch_size : int, level : str, CPCC : bool,
                                exp_name : str, num_workers : int, task : str, 
                                dataset_name : str, case : int, breeds_setting : str):

    if os.path.exists(save_dir+"/downstream_metrics.json"):
        print("retrieve_downstream_metrics: Skipped.")
        return
    
    train_loader, test_loader = make_kshot_loader(num_workers, batch_size, 1, level, 
                                                task, seeds, dataset_name, case, breeds_setting, difficulty)
    dataset = train_loader.dataset.dataset

    metrics = {
        "train_top1": [],
        "train_top2": [],
        "train_losses": [],
        "val_top1": [],
        "val_top2": [],
        "val_losses": []
    }

    criterion = torch.nn.CrossEntropyLoss().to(device)

    # print('seeds: ', seeds)
    for seed in range(seeds):
        model_path = save_dir + f"/down{task}_{level}_seed{seed}.pth"
        if level == 'fine':
            num_classes = len(dataset.fine_names)
        elif level == 'mid':
            num_classes = len(dataset.mid_names)
        elif level == 'coarse':
            num_classes = len(dataset.coarse_names)
        elif level == 'coarsest':
            num_classes = len(dataset.coarsest_names)
        train_size = len(train_loader.dataset)
        test_size = len(test_loader.dataset)
        criterion = nn.CrossEntropyLoss().to(device) # no CPCC in downstream task
        
        model = init_model(dataset_name, [num_classes], device)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        train_top1, train_top2, test_top1, test_top2 = 0, 0, 0, 0
        train_losses, test_losses = [], []

        for data, target_coarsest, target_coarse, target_mid, target_fine in train_loader:
            data = data.to(device)
            target_coarse = target_coarse.to(device)
            target_fine = target_fine.to(device)
            target_mid = target_mid.to(device)
            target_coarsest = target_coarsest.to(device)

            if level == 'coarsest':
                target = target_coarsest
            elif level == 'mid':
                target = target_mid
            elif level == 'coarse':
                target = target_coarse
            elif level == 'fine':
                target = target_fine

            _, output = model(data)
            loss = criterion(output, target)
            train_losses.append(loss.item())
            prob = F.softmax(output, dim=1)

            pred1 = prob.argmax(dim=1, keepdim=False)
            train_top1 += pred1.eq(target).sum().item()

            pred2 = (prob.topk(k=2, dim=1)[1]).T
            target_reshaped = target.unsqueeze(0).expand_as(pred2)
            train_top2 += pred2.eq(target_reshaped).sum().item()

        with torch.no_grad():
            for data, target_coarsest, target_coarse, target_mid, target_fine in test_loader:
                data = data.to(device)
                target_coarse = target_coarse.to(device)
                target_fine = target_fine.to(device)
                target_mid = target_mid.to(device)
                target_coarsest = target_coarsest.to(device)

                if level == 'coarsest':
                    target = target_coarsest
                elif level == 'mid':
                    target = target_mid
                elif level == 'coarse':
                    target = target_coarse
                elif level == 'fine':
                    target = target_fine

                _, output = model(data)
                loss = criterion(output, target)
                test_losses.append(loss.item())
                prob = F.softmax(output, dim=1)
                pred1 = prob.argmax(dim=1, keepdim=False)
                test_top1 += pred1.eq(target).sum().item()
                pred2 = (prob.topk(k=2, dim=1)[1]).T
                target_reshaped = target.unsqueeze(0).expand_as(pred2)
                test_top2 += pred2.eq(target_reshaped).sum().item()

        train_size = len(train_loader.dataset)
        test_size = len(test_loader.dataset)
        metrics["train_top1"].append(train_top1/train_size)
        metrics["train_top2"].append(train_top2/train_size)
        metrics["train_losses"].append(np.mean(train_losses))
        metrics["val_top1"].append(test_top1/test_size)
        metrics["val_top2"].append(test_top2/test_size)
        metrics["val_losses"].append(np.mean(test_losses))

    metrics_summary = {
        key: {
            "values": values,
            "mean": np.mean(values),
            "std": np.std(values)
        } for key, values in metrics.items()
    }
    with open(save_dir + "/downstream_metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=4)


def downstream_zeroshot(seeds : int , save_dir, split, task, save_name, source_train_loader, 
                        target_test_loader, levels : List[str], exp_name, device, 
                        dataset_name):
    
    # If all classes in test loader at level are already seen in train loader
    # try to use train set's label hierarchy for zero shot classification

    if os.path.exists(save_dir+f'/{save_name}.json'):
        print("downstream_zeroshot: Skipped.")
        return
    
    train_dataset = source_train_loader.dataset 
    test_dataset = target_test_loader.dataset
    
    if cpcc:
        exp_name = exp_name + 'CPCC'
    zero_shot = {'exp_name' : exp_name}
    
    for level in levels:
        
        level_res = []
        
        if level == 'fine':
            train_classes, test_classes = train_dataset.fine_names, test_dataset.fine_names
        elif level == 'mid':
            train_classes, test_classes = train_dataset.mid_names, test_dataset.mid_names
            layer_fine_map = train_dataset.mid_map
        elif level == 'coarse':
            train_classes, test_classes = train_dataset.coarse_names, test_dataset.coarse_names
            if train_on_mid:
                layer_fine_map = train_dataset.mid2coarse
            else:
                layer_fine_map = train_dataset.coarse_map
        elif level == 'coarsest':
            train_classes, test_classes = train_dataset.coarsest_names, test_dataset.coarsest_names
            if train_on_mid:
                layer_fine_map = train_dataset.mid2coarsest
            else:
                layer_fine_map = train_dataset.coarsest_map
        
        assert (train_classes == test_classes), f'Zero shot invalid for {level}.'
        
        for seed in range(seeds):
            if 'MTL' in exp_name:
                model = init_model(dataset_name, [len(train_dataset.coarse_names),len(train_dataset.fine_names)], device)
            else:
                if train_on_mid:
                    model = init_model(dataset_name, [len(train_dataset.mid_names)], device)
                elif coarse_ce:
                    model = init_model(dataset_name, [len(train_dataset.coarse_names)], device)
                else:
                    model = init_model(dataset_name, [len(train_dataset.fine_names)], device)
            model.load_state_dict(torch.load(save_dir + f'/{split}{task}_seed{seed}.pth'))
            model.eval()
            
            layer_accs = []
            with torch.no_grad():
                for (data, target_coarsest, target_coarse, target_mid, target_fine) in target_test_loader:
                    data = data.to(device)
                    target_coarsest = target_coarsest.to(device)
                    target_coarse = target_coarse.to(device)
                    target_mid = target_mid.to(device)
                    target_fine = target_fine.to(device)

                    if level == 'coarsest':
                        target_layer = target_coarsest
                    elif level == 'coarse':
                        target_layer = target_coarse
                    elif level == 'mid':
                        target_layer = target_mid
                    elif level == 'fine':
                        target_layer = target_fine
                    
                    if 'MTL' in exp_name:
                        _, _, output_one = model(data)
                    else:
                        _, output_one = model(data)
                    
                    # TODO: double check coarse case
                    prob_one = F.softmax(output_one,dim=1)
                    pred_one = prob_one.argmax(dim=1, keepdim=False)
                    if (level == 'fine' and train_on_mid == 0) or (level == 'mid' and train_on_mid == 1) or coarse_ce:
                        pred_layer = pred_one
                    else:
                        prob_layer = get_layer_prob_from_fine(prob_one, layer_fine_map)
                        pred_layer = prob_layer.argmax(dim=1, keepdim=False)
                    acc_layer = list(pred_layer.eq(target_layer).flatten().cpu().numpy())
                    layer_accs.extend(acc_layer)
            
            level_res.append(sum(layer_accs)/len(layer_accs))
        
        zero_shot[level] = {'value' : level_res, 'mean' : np.average(level_res), 'std' : np.std(level_res)}
    
    with open(save_dir+f'/{save_name}.json', 'w') as fp: 
        json.dump(zero_shot, fp, indent=4)
    print(zero_shot)
    
    return layer_accs

def plot_top_k_images(query_image, retrieved_images, query_idx, save_dir, k=5):
    """
    Plot the query image and the top-k retrieved images and save them to the specified directory.
    """
    plt.figure(figsize=(15, 5))
    plt.subplot(1, k+1, 1)
    plt.imshow(query_image.permute(1, 2, 0))
    plt.title("Query Image")
    for i, img in enumerate(retrieved_images[:k]):
        plt.subplot(1, k+1, i+2)
        plt.imshow(img.permute(1, 2, 0))
        plt.title(f"Rank {i+1}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"query_{query_idx}_top_{k}.png"))
    plt.close()

def retrieval_similarity(seeds, save_dir, split, task, task_name, train_loader, 
                               test_loader, exp_name, device, dataset_name, levels):
    """
    Function to perform image retrieval and calculate similarity at a specified level (coarse or fine).

    Parameters:
    - seeds: Number of seeds.
    - save_dir: Directory to save output.
    - split: Data split.
    - task: Task name.
    - task_name: source or train.
    - train_loader: Training data loader.
    - test_loader: Testing data loader.
    - exp_name: Experiment name.
    - device: Computation device.
    - dataset_name: Name of the dataset.
    - levels: Levels of retrieval (in ['coarse', 'mid', 'fine']).
    """
    
    # Initialization
    train_dataset = train_loader.dataset

    # Loop over seeds
    for level in levels:
        mAPs = []
        precisions = []
        recalls = []
        # Check if the level is valid
        assert level in ['coarsest', 'coarse', 'mid', 'fine'], f"Invalid level: {level}"
        for seed in range(seeds):
            # Load model based on level
            if level == 'coarsest':
                target_num = len(train_dataset.coarsest_names)
            elif level == 'coarse':
                target_num = len(train_dataset.coarse_names)
            elif level == 'mid':
                target_num = len(train_dataset.mid_names)
            else:
                assert level == 'fine'
                target_num = len(train_dataset.fine_names)

            if 'MTL' in exp_name:
                model = init_model(dataset_name, [len(train_dataset.coarse_names),len(train_dataset.fine_names)], device)
            else:
                model = init_model(dataset_name, [len(train_dataset.fine_names)], device)
            model.load_state_dict(torch.load(save_dir + f'/{split}{task}_seed{seed}.pth'))
            model.eval()

            # Initialize prototypes and other variables
            aps = []
            prototypes = {i: [] for i in range(target_num)}
            test_scores = {i: [] for i in range(target_num)}
            test_truth = {i: [] for i in range(target_num)}
            all_targets = []
            retrieval_results = []

            # Process train_loader to create prototypes
            with torch.no_grad():
                for item in train_loader:
                    data = item[0].to(device)
                    if level ==  'coarsest':
                        target = item[-4]
                    elif level ==  'coarse':
                        target = item[-3]
                    elif level == 'mid':
                        target = item[-2]
                    else:
                        assert level == 'fine'
                        target = item[-1]
                    if 'MTL' in exp_name:
                        representation, _, _ = model(data)
                    else:
                        representation, _ = model(data)
                    for it, t in enumerate(target):
                        prototypes[t.item()].append(representation[it].cpu().numpy())

                # Calculate mean of prototypes
                for i in range(target_num):
                    prototypes[i] = np.mean(np.stack(prototypes[i], axis=0), axis=0)

                # Process test_loader for retrieval
                for item in tqdm(test_loader):
                    data = item[0].to(device)
                    if level ==  'coarsest':
                        target = item[-4]
                    elif level == 'coarse':
                        target = item[-3]
                    elif level == 'mid':
                        target = item[-2]
                    else:
                        assert level == 'fine'
                        target = item[-1]
                    all_targets.extend(target.cpu().numpy())
                    if 'MTL' in exp_name:
                        test_embs, _, _ = model(data)
                    else:
                        test_embs, _ = model(data)
                    test_embs = test_embs.cpu().detach().numpy()
                    
                    for it, t in enumerate(target):
                        test_emb = test_embs[it]
                        similarities = []
                        for i in range(target_num):
                            similarity = np.dot(test_emb, prototypes[i])/(np.linalg.norm(test_emb)*np.linalg.norm(prototypes[i]))
                            similarities.append(similarity)
                            test_scores[i].append(np.dot(test_emb, prototypes[i])/(np.linalg.norm(test_emb)*np.linalg.norm(prototypes[i])))
                            test_truth[i].append(int(t.item() == i))
                        retrieval_results.append(np.argmax(similarities))
                        # similarities = [np.dot(test_emb, prot) / (np.linalg.norm(test_emb) * np.linalg.norm(prot))
                        #                 for prot in prototypes.values()]
                        # test_scores[i].append(similarities[t.item()])
                        # test_truth[i].append(int(t.item() == i))
                        # retrieval_results.append(np.argmax(similarities))

                # Compute metrics
                for i in range(target_num):
                    aps.append(average_precision_score(test_truth[i], test_scores[i]))

            # Compute average metrics
            mAP = np.mean(aps)
            precision = precision_score(all_targets, retrieval_results, average='macro')
            recall = recall_score(all_targets, retrieval_results, average='macro')

            # Append results for each seed
            mAPs.append(mAP)
            precisions.append(precision)
            recalls.append(recall)

        # Output results
        out = {
            'mAP': mAPs,
            'mean_mAP': np.mean(mAPs),
            'std_mAP': np.std(mAPs),
            'precisions': precisions,
            'mean_precision': np.mean(precisions),
            'std_precision': np.std(precisions),
            'recalls': recalls,
            'mean_recall': np.mean(recalls),
            'std_recall': np.std(recalls)
        }

        # Save results to file
        save_filename = f'/{task_name}_retrieval_evaluation_{level}.json'
        with open(save_dir + save_filename, 'w') as fp:
            json.dump(out, fp, indent=4)

    return 


def feature_extractor(dataloader : DataLoader, split : str, task : str, dataset_name : str, seed : int):
    dataset = dataloader.dataset
    if 'MTL' in exp_name:
        model = init_model(dataset_name, [len(dataset.coarse_names),len(dataset.fine_names)], device)
    elif train_on_mid:
        model = init_model(dataset_name, [len(dataset.mid_names)], device)
    elif coarse_ce:
        model = init_model(dataset_name, [len(dataset.coarse_names)], device)
    else:  
        model = init_model(dataset_name, [len(dataset.fine_names)], device)
    # model = init_model(dataset_name, [len(dataset.fine_names)], device)

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
            # target_one = item[-1]
            if train_on_mid:
                target_one = item[-2]
            elif coarse_ce:
                target_fine = item[-1]
                target_one = item[-3]
            else: 
                target_one = item[-1] # add fine target
            data = data.to(device)
            target_one = target_one.to(device)
            if coarse_ce:
                target_fine = target_fine.to(device)
            if 'MTL' in exp_name:
                feature, _, output = model(data)
            else:
                feature, output = model(data)
            prob_one = F.softmax(output,dim=1)
            probs.append(prob_one.cpu().detach().numpy())
            features.append(feature.cpu().detach().numpy())
            if coarse_ce:
                targets_one.append(target_fine.cpu().detach().numpy()) # switch coarse/fine
            else:
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

def ood_detection(seeds : int, dataset_name : str, exp_name : str, task : str, split : str):
    '''
        Set CIFAR10 as the outlier of CIFAR100.
        Credit: https://github.com/boschresearch/rince/blob/cifar/out_of_dist_detection.py
    '''
    import dataset.cifar.data
    import dataset.cifar12.data
    assert dataset_name == 'CIFAR' or dataset_name == 'CIFAR12', 'Invalid dataset for OOD detection'

    if task == "sub":
        in_train_loader, in_test_loader = dataset.cifar12.data.make_dataloader(num_workers, batch_size, 'sub_split_pretrain', difficulty)
        _, out_test_loader = dataset.cifar12.data.make_dataloader(num_workers, batch_size, 'outlier', difficulty)
    else:
        in_train_loader, in_test_loader = cifar.data.make_dataloader(num_workers, batch_size, 'full')
        _, out_test_loader = cifar.data.make_dataloader(num_workers, batch_size, 'outlier')
    oods = []
    out = {}
    for seed in range(seeds):
        # compute features
        if dataset_name == 'CIFAR12':
            split = 'split'
            task = 'sub'

        in_train_features, _, in_train_labels, _ = feature_extractor(in_train_loader, split, task, dataset_name, seed)
        in_test_features, _, _, _ = feature_extractor(in_test_loader, split, task, dataset_name, seed)
        out_test_features, _, _, _ = feature_extractor(out_test_loader, split, task, dataset_name, seed)
        print("Features successfully loaded.")

        features_outlier = np.concatenate([out_test_features, in_test_features], axis=0)
        labels = np.concatenate([np.zeros(out_test_features.shape[0], ),
                                np.ones(in_test_features.shape[0], )], axis=0)

        gms = {}
        posteriors = np.zeros((features_outlier.shape[0], len(np.unique(in_train_labels))))
        for i, label in enumerate(np.unique(in_train_labels)):
            means = np.mean(in_train_features[in_train_labels == label, :], axis=0).reshape((1, -1))
            gms[str(label)] = GaussianMixture(1, random_state=seed, means_init=means).fit(
                in_train_features[in_train_labels == label, :]) 
            posteriors[:, i] = gms[str(label)].score_samples(features_outlier)

        max_score = np.max(posteriors, axis=1)
        auroc = roc_auc_score(labels, max_score) # try different thresholds for max score
        oods.append(auroc)
    if cpcc:
        exp_name = exp_name + 'CPCC'
    out['exp_name'] = exp_name
    out['OOD'] = oods
    out['mean'] = np.average(oods)
    out['std'] = np.std(oods)
    with open(save_dir+'/OOD.json', 'w') as fp:
        json.dump(out, fp, indent=4)
    print(out)
    return oods

def retrieve_final_metrics(test_loader : DataLoader, dataset_name : str, task_name : str):
    
    def fullCPCC(dataL, targets_fineL, fine2coarse : list):
        '''
            Evaluate CPCC on full given test set.
        '''
        if os.path.exists(save_dir+f'/{task_name}_CPCC.json'):
            print(task_name, "_CPCC: Skipped.")
            return

        def poincareFn(x, y):
            eps = 1e-5
            proj_x = x * (1 - eps) / (sum(x**2)**0.5)
            proj_y = y * (1 - eps) / (sum(y**2)**0.5)
            num = 2 * sum((proj_x - proj_y)**2)
            den = (1 - (sum(proj_x**2))) * (1 - (sum(proj_y**2)))
            return np.arccosh(num/den)

        all_seed_res = {'L2-CPCC':[],'EMD-CPCC':[],'self-CPCC':[]}
        for (data, targets_fine) in zip(dataL, targets_fineL):
            df_fine = pd.concat([pd.DataFrame(data), pd.Series(targets_fine, name='target')], axis = 1)
            mean_df_fine = df_fine.groupby(['target']).mean()
            all_fine = np.unique(targets_fine)
            
            acc_tree_dist = [1 if fine2coarse[i] == fine2coarse[j] else 2 for i in range(len(fine2coarse)) for j in range(i+1,len(fine2coarse))]
            mean_tree_dist = np.average(acc_tree_dist)

            all_pairwise = cdist(data, data)
            target_indices = [np.where(targets_fine == fine)[0] for fine in all_fine]
            combidx = [(target_indices[i], target_indices[j]) for (i,j) in combinations(range(len(all_fine)),2)]
            data_slice_pairs = [(data[pair[0]],data[pair[1]]) for pair in combidx]
            dist_matrices = [all_pairwise[np.ix_(pair[0],pair[1])] for pair in combidx]

            
            # acc_l2_dist_origin = acc_l2_dist

            # if is_emd == 2:
            #     acc_l2_dist_origin = np.stack([SK.apply(M, reg, numItermax) for M in dist_matrices])
            # elif is_emd == 7:
            #     acc_l2_dist_origin = np.stack([SlicedWasserstein_np.apply(data[pair[0]], data[pair[1]], self.n_projections) for pair in combidx])
            
            

            # if cpcc_metric == 'l2':
            #     acc_l2_dist = pdist(np.array(mean_df_fine),metric='euclidean') 
            # elif cpcc_metric == 'l1':
            #     acc_l2_dist = pdist(np.array(mean_df_fine),metric='cityblock') 
            # elif cpcc_metric == 'poincare':
            #     acc_l2_dist = pdist(np.array(mean_df_fine),metric=poincareFn)
            
            # Check Gaussian
            acc_l2_dist = pdist(np.array(mean_df_fine),metric='euclidean') 
            mean_l2_dist = np.average(acc_l2_dist)
            l2_numerator = np.dot((acc_l2_dist - mean_l2_dist),(acc_tree_dist - mean_tree_dist))
            l2_denominator = (np.sum((acc_l2_dist - mean_l2_dist)**2) * np.sum((acc_tree_dist - mean_tree_dist)**2))**0.5
            all_seed_res['L2-CPCC'].append(l2_numerator/l2_denominator)

            # Check Approximation
            acc_emd_dist = np.stack([ot.emd2(np.array([]), np.array([]), M) for M in dist_matrices])
            mean_emd_dist = np.average(acc_emd_dist)
            emd_numerator = np.dot((acc_emd_dist - mean_emd_dist),(acc_tree_dist - mean_tree_dist))
            emd_denominator = (np.sum((acc_emd_dist - mean_emd_dist)**2) * np.sum((acc_tree_dist - mean_tree_dist)**2))**0.5
            all_seed_res['EMD-CPCC'].append(emd_numerator/emd_denominator)

            # Check Generalization
            if cpcc == False:
                acc_self_dist = acc_l2_dist 
            else:
                if is_emd == 0:
                    acc_self_dist = acc_l2_dist 
                elif is_emd == 1:
                    acc_self_dist = acc_emd_dist 
                elif is_emd == 2: # sinkhorn
                    acc_self_dist = np.stack([sinkhorn(M, reg, numItermax).numpy() for M in dist_matrices])
                elif is_emd == 7: # SWD
                    acc_self_dist = np.stack([ot.sliced_wasserstein_distance(X_s, X_t, n_projections=n_projections, seed=0) for X_s, X_t in data_slice_pairs])
                elif is_emd == 9: # TWD
                    acc_self_dist = np.stack([compute_flow_symmetric(X_s, X_t).numpy() for X_s, X_t in data_slice_pairs])
            mean_self_dist = np.average(acc_self_dist)
            self_numerator = np.dot((acc_self_dist - mean_self_dist),(acc_tree_dist - mean_tree_dist))
            self_denominator = (np.sum((acc_self_dist - mean_self_dist)**2) * np.sum((acc_tree_dist - mean_tree_dist)**2))**0.5
            all_seed_res['self-CPCC'].append(self_numerator/self_denominator)

        out = all_seed_res
        out['L2-mean'] = np.average(all_seed_res['L2-CPCC'])
        out['L2-std'] = np.std(all_seed_res['L2-CPCC'])
        out['EMD-mean'] = np.average(all_seed_res['EMD-CPCC'])
        out['EMD-std'] = np.std(all_seed_res['EMD-CPCC'])
        out['self-mean'] = np.average(all_seed_res['self-CPCC'])
        out['self-std'] = np.std(all_seed_res['self-CPCC'])
        with open(save_dir+f'/{task_name}_CPCC.json', 'w') as fp:
            json.dump(out, fp, indent=4)
        return out

    def silhouette(dataL, targets_coarseL):
        '''
            Use coarse label to calculate silhouette score.
        '''
        if os.path.exists(save_dir+f'/{task_name}_silhouette.json'):
            print(task_name, "_silhouette: Skipped.")
            return

        all_seed_res = []
        for (data, targets_coarse) in zip(dataL, targets_coarseL):
            res = silhouette_score(data, targets_coarse, metric='euclidean')
            all_seed_res.append(res.item())
        out = dict()
        out['silhouette'] = all_seed_res
        out['mean'] = np.average(all_seed_res)
        out['std'] = np.std(all_seed_res)
        with open(save_dir+f'/{task_name}_silhouette.json', 'w') as fp:
            json.dump(out, fp, indent=4)
        return out

    def plot_pearM(probL, targets_fineL, dataset):
        # coarse_targets_map = dataset.coarse_map
        if train_on_mid:
            coarse_targets_map = dataset.mid2coarse
            finecls2names = [str(i) for i in range(len(coarse_targets_map))]
        else:
            coarse_targets_map = dataset.coarse_map
            finecls2names = dataset.fine_names

        x_axis = []
        for i in range(len(set(coarse_targets_map))):
            x_axis.extend(list(np.where(coarse_targets_map == i)[0]))
        d = dict(zip(range(len(coarse_targets_map)),x_axis))

        fine_named_axis = [finecls2names[cls] for cls in x_axis]
        rows,cols = len(finecls2names),len(finecls2names)

        for i, (prob, targets_fine) in enumerate(zip(probL, targets_fineL)):
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(80, 80))
            seed = i
            if os.path.exists(save_dir+f"/pearM_seed{seed}.npy"):
                M = np.load(save_dir+f"/pearM_seed{seed}.npy")
            else:
                df = pd.concat([pd.DataFrame(prob), pd.Series(targets_fine, name='target')], axis = 1)
                mean_df = df.groupby(['target']).mean()
                M = np.zeros((rows,cols))
                for r in range(rows):
                    for c in range(cols):
                        if r == c:
                            M[r,c] = 0
                        else:
                            vr = np.array(mean_df.iloc[d[r],:])
                            vc = np.array(mean_df.iloc[d[c],:])
                            M[r,c] = pearsonr(vr, vc)[0]  # Use Pearson correlation
            s = sns.heatmap(M, annot=True, fmt=".3g", cmap="YlGnBu", ax=axes, cbar=False)
            s.set_xticklabels(fine_named_axis,ha='center',rotation=45)
            s.set_yticklabels(fine_named_axis,rotation=0)
            s.tick_params(left=True, right=True, bottom=True, top=True, labelright=True, labeltop=True)
            if not(os.path.exists(save_dir+f"/pearM_seed{seed}.npy")):
                np.save(save_dir+f"/pearM_seed{seed}.npy",M)
            
            plt.savefig(save_dir+f"/pearM_seed{seed}.pdf")
            plt.clf()
        return
    
    def plot_distM(dataL, targets_fineL, dataset): 
        '''
            Plot distance matrix for CIFAR. Coarse classes are grouped together,
            so that groups of distance values on the diagonal will be smaller.
        '''
        if train_on_mid:
            coarse_targets_map = dataset.mid2coarse
            finecls2names = [str(i) for i in range(len(coarse_targets_map))]
        else:
            coarse_targets_map = dataset.coarse_map
            finecls2names = dataset.fine_names

        x_axis = []
        for i in range(len(set(coarse_targets_map))):
            x_axis.extend(list(np.where(coarse_targets_map == i)[0]))
        d = dict(zip(range(len(finecls2names)),x_axis))

        fine_named_axis = [finecls2names[cls] for cls in x_axis]
        rows,cols = len(finecls2names),len(finecls2names)

        for i, (data, targets_fine) in enumerate(zip(dataL, targets_fineL)):
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(80, 80))
            seed = i
            if os.path.exists(save_dir+f"/distM_seed{seed}.npy"):
                M = np.load(save_dir+f"/distM_seed{seed}.npy")
            else:
                df = pd.concat([pd.DataFrame(data), pd.Series(targets_fine, name='target')], axis = 1)
                
                sns.set(font_scale=1.2)
                mean_df = df.groupby(['target']).mean()
                M = np.zeros((rows,cols))
                for r in range(rows):
                    for c in range(cols):
                        if r == c:
                            M[r,c] = 0
                        else:
                            vr = np.array(mean_df.iloc[d[r],:])
                            vc = np.array(mean_df.iloc[d[c],:])
                            M[r,c] = np.linalg.norm(vr-vc,ord=2)
            s = sns.heatmap(M, annot=True, fmt=".3g", cmap="YlGnBu", ax=axes, cbar=False)
            s.set_xticklabels(fine_named_axis,ha='center',rotation=45)
            s.set_yticklabels(fine_named_axis,rotation=0)
            s.tick_params(left=True, right=True, bottom=True, top=True, labelright=True, labeltop=True)
            if not(os.path.exists(save_dir+f"/distM_seed{seed}.npy")):
                np.save(save_dir+f"/distM_seed{seed}.npy",M)
            
            plt.savefig(save_dir+f"/distM_seed{seed}.pdf")
            plt.clf()
        return

    def plot_TSNE(dataL, targets_coarseL, dataset):
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(35, 30))
        for i, (data) in enumerate(dataL):
            seed = i
            tsne = TSNE(
                n_components=2,
                init="random",
                random_state=seed,
                perplexity=30, 
                learning_rate="auto",
                n_iter=1200
            )
            tsneX = tsne.fit_transform(data)
            if not(os.path.exists(save_dir+f"/TSNE_seed{seed}.npy")):
                tsneX = tsne.fit_transform(data)
                np.save(save_dir+f"/TSNE_seed{seed}.npy",tsneX)
            else:
                tsneX = np.load(save_dir+f"/TSNE_seed{seed}.npy")
            tsne_df = pd.DataFrame(data = tsneX, columns = ['TSNE1', 'TSNE2'])
            final_df = pd.concat([tsne_df, pd.Series(targets_coarseL[seed], name='target_coarse')], axis = 1)
            ax = axes
            ax.set_xticklabels([])
            ax.set_xticks([])
            ax.set_yticklabels([])
            ax.set_yticks([])
            targets = list(range(len(dataset.coarse_names)))
            mappables = []
            for target in targets:
                indicesToKeep = final_df['target_coarse'] == target
                mappable = ax.scatter(final_df.loc[indicesToKeep, 'TSNE1'],final_df.loc[indicesToKeep, 'TSNE2'])
                mappables.append(mappable)
            colormap = plt.cm.gist_ncar 
            colorst = [colormap(i) for i in np.linspace(0, 0.99, len(ax.collections))]       
            for t,j1 in enumerate(ax.collections):
                j1.set_color(colorst[t])
            ax.legend(handles=mappables, labels=dataset.coarse_names, fontsize=24, ncol=5, loc='upper center', bbox_to_anchor=(0.5, -0.01))
            ax.axis('off')
            plt.savefig(save_dir+f"/TSNE_seed{seed}.pdf")
            plt.clf()

    dataL, probL, targets_oneL, targets_coarseL = [],[],[],[]
    for seed in range(seeds):
        data, prob, targets_one, targets_coarse = feature_extractor(test_loader, split, task, dataset_name, seed)
        dataL.append(data)
        probL.append(prob)
        targets_oneL.append(targets_one)
        targets_coarseL.append(targets_coarse)
    dataset = test_loader.dataset 
    if split == 'full' or task_name == 'pretrain':
        if train_on_mid:
            out_cpcc = fullCPCC(dataL, targets_oneL, dataset.mid2coarse)
        else:
            out_cpcc = fullCPCC(dataL, targets_oneL, dataset.coarse_map)
    else:
        out_cpcc = ''
    out_silhouette = silhouette(dataL, targets_coarseL)
    
    # if (split == 'full') and (dataset_name == 'CIFAR'):
    #     plot_distM(dataL, targets_oneL, dataset)
    #     plot_TSNE(dataL, targets_coarseL, dataset)
    # plot_pearM(probL, targets_oneL, dataset)
    # print(out_cpcc, out_silhouette)
    return

def better_classification_mistakes(seeds, save_dir, split, task, device, train_loader, test_loader):
    if os.path.exists(save_dir+f'/lca_classification.json'):
        print("lca_classification: Skipped.")
        return

    lca_corrects = []
    lca_mistakes = []
    fine_res = []
    coarse_res = []

    train_dataset = train_loader.dataset
    for seed in range(seeds):
        if 'MTL' in exp_name:
            model = init_model(dataset_name, [len(train_dataset.coarse_names),len(train_dataset.fine_names)], device)
        else:
            model = init_model(dataset_name, [len(train_dataset.fine_names)], device)
        model.load_state_dict(torch.load(save_dir + f'/{split}{task}_seed{seed}.pth'))
        model.eval()

        fine_accs = []
        coarse_accs = []
        with torch.no_grad():
            lca_total = 0
            len_mistakes = 0
            for idx, (data, target_coarser, target_coarse, target_mid, target_fine) in enumerate(test_loader):
                data = data.to(device)
                target_coarse = target_coarse.to(device)
                target_fine = target_fine.to(device)
                if 'MTL' in exp_name:
                    test_representation, _, output = model(data)
                else:
                    test_representation, output = model(data)
                # simple classify
                prob_fine = F.softmax(output,dim=1)
                pred_fine = prob_fine.argmax(dim=1, keepdim=False)
                prob_coarse = get_layer_prob_from_fine(prob_fine, train_dataset.coarse_map)
                pred_coarse = prob_coarse.argmax(dim=1, keepdim=False)
                acc_fine = list(pred_fine.eq(target_fine).flatten().cpu().numpy())
                acc_coarse = list(pred_coarse.eq(target_coarse).flatten().cpu().numpy())
                fine_accs.extend(acc_fine)
                coarse_accs.extend(acc_coarse )
                # lca
                pred1 = output.argmax(dim=1, keepdim=False) 
                mistakes_target = target_fine[pred1 != target_fine]
                mistakes_pred = pred1[pred1 != target_fine]
                mistakes_coarse_target = torch.as_tensor([test_loader.dataset.coarse_map[t] for t in mistakes_target])
                mistakes_coarse_pred = torch.as_tensor([test_loader.dataset.coarse_map[t] for t in mistakes_pred])
                lca = mistakes_coarse_target.eq(mistakes_coarse_pred).sum()+(len(mistakes_target) - mistakes_coarse_target.eq(mistakes_coarse_pred).sum())*2
                lca_total += lca
                len_mistakes += len(mistakes_target)

            # include correct
            lca_correct = lca_total/len(test_loader.dataset)
            lca_mistake = lca_total/len_mistakes
            lca_corrects.append(lca_correct.item())
            lca_mistakes.append(lca_mistake.item())
        fine_res.append(sum(fine_accs)/len(fine_accs))
        coarse_res.append(sum(coarse_accs)/len(coarse_accs))
        
    out = dict()
    out['lca_mistake'] = lca_mistakes
    out['mean_lca_mistake'] = np.average(lca_mistakes)
    out['std_lca_mistake'] = np.std(lca_mistakes)
    out['lca_correct'] = lca_corrects
    out['mean_lca_correct'] = np.average(lca_corrects)
    out['std_lca_correct'] = np.std(lca_corrects)
    out['fine_acc'] = {'value' : fine_res, 'mean' : np.average(fine_res), 'std' : np.std(fine_res)}
    out['coarse_acc'] = {'value' : coarse_res, 'mean' : np.average(coarse_res), 'std' : np.std(coarse_res)}

    with open(save_dir+f'/lca_classification.json', 'w') as fp:
        json.dump(out, fp, sort_keys=True, indent=4)

def main():
    
    # Train
    for seed in range(seeds):
        seed_everything(seed)
        if split == 'split':
            # pretrain
            hyper = load_params(dataset_name, 'pre', breeds_setting=breeds_setting)
            epochs = hyper['epochs']
            
            if task == 'sub':
                train_loader, test_loader = make_dataloader(num_workers, batch_size, 'sub_split_pretrain', dataset_name, case, breeds_setting, difficulty)
            elif task == 'in':
                train_loader, test_loader = make_dataloader(num_workers, batch_size, 'in_split_pretrain', dataset_name, case, breeds_setting, difficulty)
            pretrain_objective(train_loader, test_loader, device, save_dir, seed, split, cpcc, exp_name, epochs, task, dataset_name, breeds_setting, hyper)
            
            # down
            # if dataset_name == 'CIFAR12' or dataset_name == 'CIFAR10':
            #     levels = ['fine']
            # else:
            #     levels = ['mid', 'fine']

            # only down on fine
            levels = ['fine']
                
            for level in levels: 
                hyper = load_params(dataset_name, 'down', level, breeds_setting)
                epochs = hyper['epochs']
                downstream_transfer(save_dir, seed, device, batch_size, level, cpcc, exp_name, num_workers, task, dataset_name, case, breeds_setting, hyper, epochs)
        
        elif split == 'full': 
            hyper = load_params(dataset_name, 'pre', breeds_setting=breeds_setting)
            epochs = hyper['epochs']
            train_loader, test_loader = make_dataloader(num_workers, batch_size, 'full', dataset_name, case, breeds_setting, difficulty) # full
            pretrain_objective(train_loader, test_loader, device, save_dir, seed, split, cpcc, exp_name, epochs, task, dataset_name, breeds_setting, hyper)

    if task == 'sub':
        # TODO: check levels
        if dataset_name == 'MNIST' or dataset_name == 'CIFAR12' or dataset_name == 'CIFAR10' or dataset_name == 'CIFAR' or dataset_name == 'INAT':
            levels = ['coarse', 'fine']
        elif dataset_name == 'BREEDS2':
            levels = ['coarse', 'mid', 'fine']
        else:
            levels = ['coarsest','coarse','fine'] 
        train_loader, test_loader = make_dataloader(num_workers, batch_size, 'sub_split_pretrain', dataset_name, case, breeds_setting, difficulty)
        retrieve_downstream_metrics(save_dir, seeds, device, batch_size, 'fine', cpcc, exp_name, num_workers, task, dataset_name, case, breeds_setting)
        retrieval_similarity(seeds, save_dir, split, task, 'source', train_loader, test_loader, exp_name, device, dataset_name, levels)
        better_classification_mistakes(seeds, save_dir, split, task, device, train_loader, test_loader)
        retrieve_final_metrics(test_loader, dataset_name, 'pretrain')
    
    # Eval: zero-shot/ood

    if task == 'sub':
        if dataset_name == 'MNIST' or dataset_name == 'CIFAR12' or dataset_name == 'CIFAR10' or dataset_name == 'INAT':
            levels = ['coarse']
        elif dataset_name == 'BREEDS2':
            levels = ['coarse', 'mid', 'fine']
        else:
            levels = ['coarsest','coarse'] 
        train_loader, test_loader = make_dataloader(num_workers, batch_size, f'{task}_split_zero_shot', dataset_name, case, breeds_setting, difficulty)
        retrieval_similarity(seeds, save_dir, split, task, 'target', train_loader, test_loader, exp_name, device, dataset_name, levels)
    elif task == '': # full
        if dataset_name == 'MNIST':
            if train_on_mid:
                levels = ['coarse','mid'] 
            else:
                levels = ['coarse','mid','fine'] 
        else:
            if train_on_mid:
                levels = ['coarsest','coarse','mid']
            else:
                levels = ['coarsest','coarse','mid','fine']
        train_loader, test_loader = make_dataloader(num_workers, batch_size, 'full', dataset_name, case, breeds_setting, difficulty)
    
    downstream_zeroshot(seeds, save_dir, split, task, 'zero_shot', train_loader, test_loader, levels, exp_name, device, dataset_name)
    retrieve_final_metrics(test_loader, dataset_name, 'zero_shot')

    if dataset_name == 'CIFAR12' or dataset_name == 'CIFAR':
        ood_detection(seeds, dataset_name, exp_name, task, split)
    
    return

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/common/cindy2000_sh", type=str, help='directory that you want to save your experiment results')
    parser.add_argument("--timestamp", required=True, help=r'your unique experiment id, hint: datetime.now().strftime("%m%d%Y%H%M%S")') 
    parser.add_argument("--dataset", required=True, help='MNIST/CIFAR/CIFAR12/BREEDS/BREEDS2/INAT')
    parser.add_argument("--exp_name", required=True, help='ERM/MTL/Curriculum/sumloss/HXE/soft/quad')
    parser.add_argument("--split", required=True, help='split/full')
    parser.add_argument("--task", default='', help='in/sub')
    parser.add_argument("--cpcc", required=True, type=int, help='0/1')
    parser.add_argument("--cpcc_metric", default='l2', type=str, help='distance metric in CPCC, l2/l1/poincare')
    parser.add_argument("--cpcc_list", nargs='+', default=['coarse'], help='ex: --cpcc-list mid coarse, for 3 layer cpcc')
    parser.add_argument("--group", default=0, type=int, help='0/1, grouplasso')
    parser.add_argument("--case", default=0, type=int, help='Type of MNIST, 0/1')
    parser.add_argument("--difficulty", default="medium", type=str, help='Difficulty of CIFAR12, easy/medium/hard')
    parser.add_argument("--breeds_setting", default="", type=str, help='living17, nonliving26, entity13, entity30')

    parser.add_argument("--train_on_mid", default=0, type=int, help='Train on fine or mid layer, 0/1')
    parser.add_argument("--ss_test", default=0, type=int, help='Only Source train & Source test, 0/1')
    parser.add_argument("--coarse_ce", default=0, type=int, help='Use coarse layer ce or not, 0/1')
    parser.add_argument("--emd", default=0, type=int, help='0-Euclidean distance, 1-EMD, 2-Sinkhorn, 3-SmoothOT')
    parser.add_argument("--reg", default=10, type=int, help='the regulization param of Sinkhorn or SmoothOT')
    parser.add_argument("--numItermax", default=10000, type=int, help='numIterMax to run Sinkhorn or SmoothOT')
    parser.add_argument("--n_projections", default=10, type=int, help='number of projections in SWD')

    parser.add_argument("--lamb",type=float,default=1,help='strength of CPCC regularization')
    
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--seeds", type=int,default=5)    
    parser.add_argument("--downstream", type=int, default=0, help='use max iter checkpoint as final model and do evaluation. assert seeds=1')
    
    args = parser.parse_args()
    timestamp = args.timestamp
    exp_name = args.exp_name
    dataset_name = args.dataset
    split = args.split
    task = args.task
    cpcc = args.cpcc
    cpcc_metric = args.cpcc_metric
    cpcc_layers = args.cpcc_list
    case = args.case
    group = args.group
    is_emd = args.emd
    train_on_mid = args.train_on_mid
    ss_test = args.ss_test
    coarse_ce = args.coarse_ce
    reg = args.reg
    numItermax = args.numItermax
    difficulty = args.difficulty

    num_workers = args.num_workers
    batch_size = args.batch_size
    seeds = args.seeds
    lamb = args.lamb
    n_projections = args.n_projections
    is_down = args.downstream

    root = args.root 
    
    root = f'{root}/hierarchy_results/{dataset_name}' 
    save_dir = root + '/' + timestamp 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device: ", device)

    if dataset_name == 'BREEDS' or dataset_name == 'BREEDS2':
        breeds_setting = args.breeds_setting
        assert breeds_setting in ['living17','nonliving26','entity13','entity30']
        # for breeds_setting in ['living17','nonliving26','entity13']:
        save_dir = root + '/' + timestamp + '/' + breeds_setting
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        checkpoint_dir = save_dir + '/checkpoint'
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        main()
    
    else:
        breeds_setting = None
        checkpoint_dir = save_dir + '/checkpoint'
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        main()

    