#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 14:56:21 2024

@author: sariyanide
"""
from torch.nn import functional as F

import torch
import numpy as np

from sklearn.metrics import balanced_accuracy_score

def compute_performance(model, dataloder, score_type='balanced_accuracy'):
    model.eval()
    
    with torch.no_grad():
        tot_loss = 0
        nbatches = 0
        
        Ytes_hat = []
        Ytes_gt = []
        for inputs, labels in dataloder:
            inputs = inputs.permute(0,2,1).to(model.device)
            labels = labels.to(model.device)
            # print(labels)
                
            output = model(inputs)
            if type(output) is tuple:
                output = output[-1]
            loss = F.cross_entropy(output, labels.flatten().long())
            tot_loss += loss.item() #* inputs.size(0)
            
            Ytes_gt.append(labels.cpu().numpy())
            Ytes_hat.append(output.argmax(dim=1).cpu().flatten().numpy())
            
            nbatches += 1
    # print(np.concatenate(Ytes_gt))
    # print(np.concatenate(Ytes_hat))
        
        
    avg_loss = tot_loss/nbatches
    if score_type == 'balanced_accuracy':
        score = balanced_accuracy_score(np.concatenate(Ytes_gt), np.concatenate(Ytes_hat))
    return avg_loss, score




def compute_single_loss(model, dataloader, score_type='balanced_accuracy', optimizer=None):
    
    # with torch.no_grad():
    tot_loss = 0
    nbatches = 0
    
    Ytes_hat = []
    Ytes_gt = []
    # idx = 0
    for inputs, labels in dataloader:
        # print(idx)
        # idx += 1
        if optimizer is not None:
            optimizer.zero_grad()
            
        inputs = inputs.permute(0,2,1).to(model.device)
        labels = labels.to(model.device)
        
        p, c, pred = model(inputs)
        loss2 = F.cross_entropy(pred, labels.flatten().long()) 
        
        tot_loss += loss2.item() #* inputs.size(0)
        Ytes_gt.append(labels.cpu().numpy())
        Ytes_hat.append(pred.argmax(dim=1).cpu().flatten().numpy())
        
        nbatches += 1
        if optimizer is not None:
            # loss1.backward(retain_graph=True)
            loss2.backward()
            optimizer.step()
            
    avg_loss = tot_loss/nbatches
    if score_type == 'balanced_accuracy':
        score = balanced_accuracy_score(np.concatenate(Ytes_gt), np.concatenate(Ytes_hat))
    return avg_loss, score



def compute_dependence_loss(model, dataloader, score_type='balanced_accuracy', optimizer=None):
    
    
    # with torch.no_grad():
    tot_loss = 0
    nbatches = 0
    
    Ytes_hat = []
    Ytes_gt = []
    norm1 = None
    norm2 = None
    for inputs, labels in dataloader:
        if optimizer is not None:
            optimizer.zero_grad()
            
        inputs = inputs.permute(0,2,1).to(model.device)
        labels = labels.to(model.device)
        

        pos_labels = labels == 1
        pos_inputs = inputs[pos_labels[:,0],...]
        neg_labels = labels == 0
        neg_inputs = inputs[neg_labels[:,0],...]
        # print(labels)
            
        pp, cp, pred_pos = model(pos_inputs)
        pn, cn, pred_neg = model(neg_inputs)
        
        # loss1 = -0.01*torch.clamp(torch.linalg.matrix_norm(torch.mean(torch.matmul(pp, cp.permute(0,2,1))-torch.matmul(torch.mean(pp, dim=0), torch.mean(cp, dim=0).T), dim=0), ord=1),max=10000) \
        #         +0.01*torch.clamp(torch.linalg.matrix_norm(torch.mean(torch.matmul(pn, cn.permute(0,2,1))-torch.matmul(torch.mean(pn, dim=0), torch.mean(cn, dim=0).T), dim=0), ord=1), max=10000) \
        #                 +0.00001*(0.1*torch.norm(cp)**2 + 0.1*torch.norm(pp)**2 + 0.1*torch.norm(pn)**2 + 0.1*torch.norm(cn)**2)
                        
        C1 = torch.mean(torch.matmul(pp, cp.permute(0,2,1))-torch.matmul(torch.mean(pp, dim=0), torch.mean(cp, dim=0).T), dim=0).detach().cpu()
        C2 = torch.mean(torch.matmul(pn, cn.permute(0,2,1))-torch.matmul(torch.mean(pn, dim=0), torch.mean(cn, dim=0).T), dim=0).detach().cpu()
        
        n = C1.shape[0]
        C1 = C1 * (1 - torch.eye(n, n))
        C2 = C2 * (1 - torch.eye(n, n))
        if norm1 is None:
            norm1 = C1
            norm2 = C2
        else:
            norm1 += C1
            norm2 += C2
            # + torch.mean(torch.linalg.matrix_norm(torch.matmul(torch.mean(pp, dim=0), torch.mean(cp, dim=0).T))) \
                    # - torch.mean(torch.linalg.matrix_norm(torch.matmul(torch.mean(pn, dim=0), torch.mean(cn, dim=0).T))) \
                        # +0.00001*(0.0001*torch.norm(cp)**2 + 0.0001*torch.norm(pp)**2 + 0.0001*torch.norm(pn)**2 + 0.0001*torch.norm(cn)**2)

        loss2 = F.cross_entropy(pred_pos, labels[pos_labels[:,0],...].flatten().long()) \
            + F.cross_entropy(pred_neg, labels[neg_labels[:,0],...].flatten().long())

        
        tot_loss += loss2.item() #* inputs.size(0)
        
        Ytes_gt.append(labels[pos_labels[:,0],...].cpu().numpy())
        Ytes_gt.append(labels[neg_labels[:,0],...].cpu().numpy())
        Ytes_hat.append(pred_pos.argmax(dim=1).cpu().flatten().numpy())
        Ytes_hat.append(pred_neg.argmax(dim=1).cpu().flatten().numpy())
        
        nbatches += 1
        if optimizer is not None:
            # loss1.backward(retain_graph=True)
            loss2.backward()
            optimizer.step()
            
    norm1 /= nbatches
    norm2 /= nbatches
    
    norm1 = torch.mean(torch.abs(norm1))
    norm2 = torch.mean(torch.abs(norm2))
    
    if optimizer is None:
        print(f'{norm1:.7f}; {norm2:.7f}')
    # print(np.concatenate(Ytes_gt))
    # print(np.concatenate(Ytes_hat))
        
        
    avg_loss = tot_loss/nbatches
    if score_type == 'balanced_accuracy':
        score = balanced_accuracy_score(np.concatenate(Ytes_gt), np.concatenate(Ytes_hat))
    return avg_loss, score


def compute_performance_fromtwofold(model, dataloder, score_type='balanced_accuracy'):
    model.eval()
    
    with torch.no_grad():
        tot_loss = 0
        nbatches = 0
        
        Ytes_hat = []
        Ytes_gt = []
        for i in range(3):
            for inputs, labels in dataloder:
                inputs = inputs.permute(0,2,1).to(model.device)
                labels = labels.to(model.device)
                    
                y = model(inputs)
                loss = F.cross_entropy(y[0], labels.flatten().long())
                tot_loss += loss.item() #* inputs.size(0)
                
                Ytes_gt.append(labels.cpu().numpy())
                Ytes_hat.append(y[0].argmax(dim=1).cpu().flatten().numpy())
                
                nbatches += 1
        
    avg_loss = tot_loss/nbatches
    if score_type == 'balanced_accuracy':
        score = balanced_accuracy_score(np.concatenate(Ytes_gt), np.concatenate(Ytes_hat))
    return avg_loss, score



def compute_performance_fromthreefold(model, dataloder, score_type='balanced_accuracy'):
    model.eval()
    
    with torch.no_grad():
        tot_loss = 0
        nbatches = 0
        
        Ytes_hat = []
        Ytes_hat2 = []
        Ytes_gt = []
        for inputs, labels in dataloder:
            inputs = inputs.permute(0,2,1).to(model.device)
            labels = labels.to(model.device)
                
            output, _, _ = model(inputs.clone())
            _, _, output2 = model(inputs.clone())
            loss = F.cross_entropy(output, labels.flatten().long()) + F.cross_entropy(output2, labels.flatten().long())
            tot_loss += loss.item() #* inputs.size(0)
            
            Ytes_gt.append(labels.cpu().numpy())
            Ytes_hat.append(output.argmax(dim=1).cpu().flatten().numpy())
            Ytes_hat2.append(output2.argmax(dim=1).cpu().flatten().numpy())
            
            nbatches += 1
        
    avg_loss = tot_loss/nbatches
    if score_type == 'balanced_accuracy':
        score = balanced_accuracy_score(np.concatenate(Ytes_gt), np.concatenate(Ytes_hat))
        score2 = balanced_accuracy_score(np.concatenate(Ytes_gt), np.concatenate(Ytes_hat2))
    return avg_loss, score, score2