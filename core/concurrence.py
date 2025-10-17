#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 17:27:30 2024

@author: sariyanide
"""
import copy
import torch
import warnings
import numpy as np
from torch.nn import functional as F
from sklearn.metrics import balanced_accuracy_score

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from core.models import CnnDepSimpleExp
from core.dataset import DatasetFromNumpy, znorm


def compute_balanced_acc_from_dataloader(model, dataloader, scramble_labels=False, scrambler=None):
    nbatches = 0
    
    Yval_hat = []
    Yval_gt = []
    
    val_loss = 0
    
    for inputs, labels in dataloader:
        inputs = inputs.to(model.device).permute(0,2,1)
        labels = labels.to(model.device)
        
        if scramble_labels:
            labels = labels[torch.randperm(labels.shape[0], generator=scrambler),:]
        
        p, c, pred = model(inputs)
        loss2 = F.cross_entropy(pred, labels.flatten().long())
        val_loss += loss2.item()
        
        Yval_gt.append(labels.cpu().numpy())
        Yval_hat.append(pred.argmax(dim=1).cpu().flatten().numpy())
        nbatches += 1
    
    return val_loss/nbatches, balanced_accuracy_score(np.concatenate(Yval_gt), np.concatenate(Yval_hat))




def concurrence(tra_data, tes_data=None, val_data=None, params={}, model_path=None, 
                  model_type='CnnDepSimpleExp', scramble_labels=False,
                  concurrence_every_n=1, conc_thresh_early_exit=0.995, early_termination_num_iters=20):
        
    if model_type == 'CnnDepSimpleExp':
        model = CnnDepSimpleExp()
    else:
        raise Exception("Unsupported Deep Learning Architecture")
            
    # model.create_model(params, params['w'], tra_data.Nfeatures_in_dyad)
    model.create_model(params, params['w'], tra_data.Nfeats_x, tra_data.Nfeats_y)
    model.device = model.opts['device']
    model = model.to(model.opts['device'])
    
    pin_memory = True if next(model.parameters()).is_cuda else False

    tra_dataloader = DataLoader(tra_data, batch_size=model.opts['batch_size'], shuffle=False, num_workers=1, pin_memory=pin_memory)
    
    if tes_data is None:
        warnings.warn("""Test data is not provided. 
                      Concurrence will be computed on training data, which is likely to lead to a Type I error.
                      We *STRONGLY* advise the usage of an independent sample.
                      
                      The usage of tes_data is only intended for debugging purposes or using how the 
                      per-segment-concurrence-score (PSCS) varies within the training sample.
                      """)
        tes_dataloader = DataLoader(tra_data, batch_size=model.opts['batch_size'], shuffle=False, num_workers=1, pin_memory=pin_memory)
    else:
        tes_dataloader = DataLoader(tes_data, batch_size=model.opts['batch_size'], shuffle=False, num_workers=1, pin_memory=pin_memory)
        
    val_dataloader = None
    
    if val_data is not None:
        val_dataloader = DataLoader(val_data, batch_size=model.opts['batch_size'], shuffle=False, num_workers=1, pin_memory=pin_memory)
    
    hist_tra = []
    hist_tes = []
    balanced_accs_val = []
    balanced_accs_tes = []
    
    optimizer = torch.optim.Adam(model.parameters(), lr=model.opts['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, model.opts['scheduler_step_size'])
    
    scrambler = None
    
    Ntot_epochs = model.opts['num_iters']
    
    prev_max_concurrence = 0
    
    no_increase_for_n_iters = 0
    
    for n in range(0, Ntot_epochs):
        
        if scramble_labels:
            scrambler = torch.Generator()
            scrambler.manual_seed(model.opts['seed'])
        
        model.train()
        nbatches = 0
        tot_loss = 0
        for inputs, labels in tra_dataloader:
            inputs = inputs.to(model.device).permute(0,2,1)
            labels = labels.to(model.device)
                        
            if scramble_labels:
                labels = labels[torch.randperm(labels.shape[0], generator=scrambler),:]
            
            p, c, pred = model(inputs)
            loss2 = F.cross_entropy(pred, labels.flatten().long()) 
            
            tot_loss += loss2.item()
            
            optimizer.zero_grad()
            loss2.backward()
            optimizer.step()
            nbatches += 1
            
        hist_tra.append(tot_loss/nbatches)
        scheduler.step()
        
        model.eval()
        
        
        
        if n % concurrence_every_n == concurrence_every_n-1:
            with torch.no_grad():
                nbatches = 0
                
                tes_loss, balanced_acc_tes = compute_balanced_acc_from_dataloader(model, tes_dataloader, scramble_labels, scrambler)
                hist_tes.append(tes_loss)
                if val_dataloader is not None:
                    val_loss, balanced_acc_val = compute_balanced_acc_from_dataloader(model, val_dataloader, scramble_labels, scrambler)
        
            balanced_accs_tes.append(balanced_acc_tes)
            
            if val_dataloader is not None:
                balanced_accs_val.append(balanced_acc_val)
            
            if model.opts['plot_every'] != -1 and (n % model.opts['plot_every']  == model.opts['plot_every']-1):
                plt.clf()
                plt.subplot(211)
                plt.plot(hist_tra)
                plt.plot(hist_tes)
                plt.subplot(212)
                plt.plot(balanced_accs_val)
                plt.plot(balanced_accs_tes)
                plt.show(block=False)
            
            if val_dataloader is None:
                nmin = min(model.opts['num_iters'], 10)
                if n > nmin:
                    concurrence = 2*(np.mean(balanced_accs_tes[-nmin:])-0.5)
                else:
                    concurrence = 2*(np.mean(balanced_accs_tes)-0.5)
            else:
                maxacc = np.mean(np.array(balanced_accs_tes)[np.argsort(-np.array(balanced_accs_val))[:5]])
                concurrence = 2*(maxacc-0.5)
            
            if concurrence > conc_thresh_early_exit:
                break
            
            if concurrence > prev_max_concurrence:
                no_increase_for_n_iters = 0
                prev_max_concurrence = concurrence
            else:
                no_increase_for_n_iters += 1
                
            if no_increase_for_n_iters > early_termination_num_iters:
                print()
                print('Terminating early -- concurrence seems to have converged 📈')
                break
            
            if n % 1 == 0:
                print(f'\rconcurrence={concurrence:.2f} (iter={n}/{Ntot_epochs})', end='')

    
    print(flush=True)
    print(flush=True)
    print('-'*40)
    print(f'🏁 concurrence={concurrence:.2f} (final)')
    print('-'*40)
    print(flush=True)

    if model_path is not None:
        checkpoint = {
            "model_state": model.state_dict(),
            "rng_state": torch.get_rng_state(),
            "concurrence": concurrence,
            "Nfeatures_in_dyad": tra_data.Nfeatures_in_dyad,
            "balanced_accs": balanced_accs_tes
        }
        torch.save(checkpoint, model_path)
        
    return concurrence, model
    
    
    
    
    
    




def compute_PSCSs_sliding_window(file_pairs, model, params={}, step_size_dw=None):

    w = model.opts['w']
    
    if step_size_dw is None:
        step_size_dw = w//2
    
    model.eval()
    scores = []
    
    for i in range(len(file_pairs)):
        
        x = np.loadtxt(file_pairs[i][0])
        y = np.loadtxt(file_pairs[i][1])
        
        T = x.shape[0]
        if len(x.shape) == 1:
            x = x.reshape(-1,1)
        if len(y.shape) == 1:
            y = y.reshape(-1,1)
            
        cscores = []
        t0s = range(0, T-w+1, step_size_dw)
            
        if x.shape[0] != y.shape[0]:
            err_msg = f'Mismatching temporal length for files {file_pairs[i][0]} and {file_pairs[i][1]}; will fill with NaNs'
            warnings.warn(err_msg)
            for ti, t0 in enumerate(t0s):
                cscores.append(np.nan)
            scores.append(cscores)
            continue

        xy = torch.from_numpy(np.concatenate((x, y), axis=1)).float()
       
        for ti, t0 in enumerate(t0s):
            with torch.no_grad():
                xyt = xy[t0:t0+w,:].to(model.device).unsqueeze(0).permute(0,2,1)
                _, _, pred = model(xyt)
                score = pred[0][1].cpu().item()
                cscores.append(score)
       
        scores.append(cscores)
    
    return scores


