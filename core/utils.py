#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 14:24:04 2024

@author: sariyanide
"""


import numpy as np

def textfile_rows_to_list(file_path):
    with open(file_path, 'r') as file:
        return [row.strip() for row in file]

def read_dyadic_pair(data_dir, study_id, use_diff, feat_ix0=0, feat_ixf=None):
    import torch
    try:
        np.loadtxt(f'{data_dir}/{study_id}participant')
        np.loadtxt(f'{data_dir}/{study_id}confederate')
        
        if use_diff:
            xraw = np.diff(np.loadtxt(f'{data_dir}/{study_id}participant'), axis=0).astype(np.float32)
            yraw = np.diff(np.loadtxt(f'{data_dir}/{study_id}confederate'), axis=0).astype(np.float32)
            part_data = torch.from_numpy(xraw)
            conf_data = torch.from_numpy(yraw)
        else:
            part_data = torch.from_numpy(np.loadtxt(f'{data_dir}/{study_id}participant').astype(np.float32))
            conf_data = torch.from_numpy(np.loadtxt(f'{data_dir}/{study_id}confederate').astype(np.float32)) 
    
        part_data = part_data[:,feat_ix0:feat_ixf]
        conf_data = conf_data[:,feat_ix0:feat_ixf]
    
        data = torch.cat([part_data, conf_data], dim=1)
        return data 
    except:
        return None
    
    


#correct if the population S.D. is expected to be equal for the two groups.
def cohen_d(x,y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)