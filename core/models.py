#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 17:26:37 2024

@author: sariyanide
"""
import torch
import torch.nn as nn
import warnings

    
    
    
class CnnDepSimpleExp(nn.Module):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        
        # self.add_missing_opts(default_opts)
        # we'll create model when we get the data -- we need 
        
        
    def create_model(self, opts, seq_len_, Nfeats_x, Nfeats_y):
        
        default_opts =  {'w': 180, 'seed': 1907, 'plot_every': -1 , 'use_diff': 0, 'stride1': 3, 
                    'stride2': 1, 'kernel_size': 3, 'kernel_size2': 3, 'num_blocks_pre': 3, 
                    'device': 'cuda:0', 'FPS': 1, 'tra_ptg': 0.8, 'val_ptg': 0.2, 'base_filters1': 16, 
                    'num_iters': 500, 'dropout_rate': 0.5, 'learning_rate': 1e-4, 
                    'scheduler_step_size': 500, 'early_termination_num_iters': 2000,
                    'batch_size': 64, 'segs_per_pair': 10,
                    'znorm': False, 'feat_ix0': 0, 'feat_ixf': None}

        
        self.opts = {**default_opts, **opts}
        
        
                
        self.seq_len = seq_len_#-self.opts['randomclip_tra']
        
        self.kernel_size = self.opts['kernel_size']
        self.base_filters1 = self.opts['base_filters1']
        self.dropout_rate = self.opts['dropout_rate']
        self.kernel_size2 = self.opts['kernel_size2']
        self.stride_init = self.opts['stride1']
        self.stride_later = self.opts['stride2']
        
        self.Nfeats_x = Nfeats_x
        self.Nfeats_y = Nfeats_y
        
        if self.Nfeats_x != self.Nfeats_y:
            assert self.opts['feat_ixf'] is None and self.opts['feat_ix0'] == 0
            
        tot_in_features_in_dyad = Nfeats_x + Nfeats_y
            
        tot_in_features_in_monad = tot_in_features_in_dyad//2
        if self.opts['feat_ixf'] is None and self.Nfeats_x == self.Nfeats_y:
            self.opts['feat_ixf'] = tot_in_features_in_monad
        
        # in_features = self.opts['feat_ixf']-self.opts['feat_ix0']
        # in_channels = in_features
        
        # in_channels = in_features//2
        
        # for n in range(self.opts['num_blocks_pre']):
        #     if n == 0:
        #         padding = 'valid'
        #     else:
        #         if stride == 1:
        #             padding = 'same'
        #         else:
        #             padding = 'valid'
        #     pre_layers += [nn.BatchNorm1d(in_channels),
        #                       nn.Conv1d(kernel_size=kernel_size, in_channels=in_channels,
        #                                 out_channels=out_channels, stride=stride, padding=padding),
        #                         nn.Dropout(dropout_rate),
        #                         nn.ReLU()]
        #     in_channels = out_channels
        #     out_channels //= 2
        #     stride = stride_later
        #     kernel_size = kernel_size2
        
        pre_layers = self.generate_cnn(self.Nfeats_x)
        self.net1 = nn.Sequential(*(pre_layers))
        test_model1 = nn.Sequential(*(pre_layers))
        int_dim1 = test_model1(torch.zeros(1, self.Nfeats_x, self.seq_len)).shape[1]
        int_len1 = test_model1(torch.zeros(1, self.Nfeats_x, self.seq_len)).shape[2]

        if self.Nfeats_x == self.Nfeats_y and self.opts['f_equals_g']:
            self.net2 = nn.Sequential(*(pre_layers))
            test_model2 = nn.Sequential(*(pre_layers))
            int_dim2 = int_dim1
            int_len2 = int_len1
        else:
            if self.opts['f_equals_g']:
                warnings.warn(f"""Ignoring --f_equals_g=1 and training separate functions f and g, as the dimensions 
                              of the signals x and y seem to be different ({self.Nfeatsx} vs. {self.Nfeats_y}))""")

            self.net2 = nn.Sequential(*(self.generate_cnn(self.Nfeats_y)))
            test_model2 = nn.Sequential(*(self.generate_cnn(self.Nfeats_y)))
            int_dim2 = test_model2(torch.zeros(1, self.Nfeats_y, self.seq_len)).shape[1]
            int_len2 = test_model2(torch.zeros(1, self.Nfeats_y, self.seq_len)).shape[2]
            
        self.flattener = nn.Flatten()

        
        # We currently assume that the intermediate dimensions are of the same length
        assert int_dim1 == int_dim2
        assert int_len1 == int_len2
        
        
        self.int_dim = int_dim1
        self.int_len = int_len1
        #print(f'Intermediate dim: {int_dim}')
        
        self.quad = nn.Linear(int_dim1**2, 2)
        
        del test_model1
        del test_model2
        
        
    def generate_cnn(self, in_channels):
        pre_layers = []
        
        out_channels = self.base_filters1

        
        stride = self.stride_init
        kernel_size = self.kernel_size
        
        for n in range(self.opts['num_blocks_pre']):
            if n == 0:
                padding = 'valid'
            else:
                if stride == 1:
                    padding = 'same'
                else:
                    padding = 'valid'
            pre_layers += [nn.BatchNorm1d(in_channels),
                              nn.Conv1d(kernel_size=kernel_size, in_channels=in_channels,
                                        out_channels=out_channels, stride=stride, padding=padding),
                                nn.Dropout(self.dropout_rate),
                                nn.ReLU()]
            in_channels = out_channels
            out_channels //= 2
            stride = self.stride_later
            kernel_size = self.kernel_size2
        
        return pre_layers
        
    def znorm(self, x):
        
        mean = x.mean(dim=2, keepdim=True)            # [B, D, 1]
        std = x.std(dim=2, keepdim=True) + 1e-6   # [B, D, 1]
        return (x - mean) / std
        
    def forward(self, x):
        
        # Nfeats = x.shape[1]//2
        x1 = x[:,:self.Nfeats_x,:]
        x2 = x[:,self.Nfeats_x:,:]
        
        x1 = x1[:,self.opts['feat_ix0']:self.opts['feat_ixf'],:]
        x2 = x2[:,self.opts['feat_ix0']:self.opts['feat_ixf'],:]
        
        if self.opts['use_diff']:
            x1 = x1[:,:,1:] - x1[:,:,:-1]
            x2 = x2[:,:,1:] - x2[:,:,:-1]
            
        if self.opts['znorm']:
            x1 = self.znorm(x1)
            x2 = self.znorm(x2)
        
        x1 = self.net1(x1)
        x2 = self.net2(x2)
        
        q = x1 @ x2.permute(0,2,1)-(torch.mean(x1, axis=2).unsqueeze(-1)@torch.mean(x2,axis=2).unsqueeze(1))
        
        q = self.flattener(x1 @ x2.permute(0,2,1))
        x = self.quad(q)
        
        return x1, x2, x
    


   
