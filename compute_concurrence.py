#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 17:27:30 2024

@author: sariyanide
"""
import os
import sys
import torch
import argparse
import warnings
from core.dataset import DataFromTextFilesND

parser = argparse.ArgumentParser()
parser.add_argument("xflist_traval", type=str, nargs='?')
parser.add_argument("yflist_traval", type=str, nargs='?')
parser.add_argument("xflist_tes", type=str, nargs='?', default=None)
parser.add_argument("yflist_tes", type=str, nargs='?', default=None)
parser.add_argument("--xy_pair_ids_flist_traval", type=str,  default=None)
parser.add_argument("--xy_pair_ids_flist_tes", type=str,  default=None)
parser.add_argument("--w", type=int, default=60)
parser.add_argument("--num_filters", type=int, default=512)
parser.add_argument("--kernel_size1", type=int, default=5)
parser.add_argument("--kernel_size2", type=int, default=3)
parser.add_argument("--stride1", type=int, default=3)
parser.add_argument("--stride2", type=int, default=2)
parser.add_argument("--num_iters", type=int, default=100)
parser.add_argument("--dropout_rate", type=float, default=0.25)
parser.add_argument("--num_blocks_pre", type=int, default=3)
parser.add_argument("--use_diff", type=int, default=0)
parser.add_argument("--znorm", type=int, default=1)
parser.add_argument("--feat_ix0", type=int, default=0, help='Features before feat_ix0 will be ignored')
parser.add_argument("--feat_ixf", type=int, default=None, help='Features after feat_ixf will be ignored')
parser.add_argument("--val_ptg", type=float, default=0.2)
parser.add_argument("--segs_per_pair", type=int, default=50)
parser.add_argument("--PSCS_file", type=str, default=None)
parser.add_argument("--device", type=str, default='cuda')
parser.add_argument("--randseed", type=int, default=1907)
parser.add_argument("--f_equals_g", type=int, default=1)

args = parser.parse_args()

if args.device.find('cuda') >= 0 and not torch.cuda.is_available():
    warnings.warn(f"""Chosen device ({args.device}) does not seem to be available. 
                  Consider running with the parameter --device=cpu or --device=mps (if Apple Silicon)""")
    sys.exit(1)

torch.manual_seed(args.randseed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(args.randseed)

opts = {'device': args.device, 'num_iters': args.num_iters, 'w': args.w, 'base_filters1': args.num_filters,
        'kernel_size': args.kernel_size1, 'kernel_size2': args.kernel_size2, 'use_diff': bool(args.use_diff), 
        'dropout_rate': args.dropout_rate, 'stride1': args.stride1, 'stride2': args.stride2, 
        'num_blocks_pre': args.num_blocks_pre, 'segs_per_pair': args.segs_per_pair, 'znorm': bool(args.znorm), 'batch_size': 64, 
        'scheduler_step_size': 500, 'feat_ix0': args.feat_ix0, 'feat_ixf': args.feat_ixf, 'seed': args.randseed, 'f_equals_g': args.f_equals_g}

def parse_filelist(filepath):
    if filepath is None: return []
    
    filelist = []
    with open(filepath, 'r') as f:
        filelist = f.read().splitlines()

    return filelist

xfiles_traval = parse_filelist(args.xflist_traval)
yfiles_traval = parse_filelist(args.yflist_traval)
xy_pair_ids_traval = parse_filelist(args.xy_pair_ids_flist_traval)

assert len(xfiles_traval) == len(yfiles_traval) 

xfiles_tes = parse_filelist(args.xflist_tes)
yfiles_tes = parse_filelist(args.yflist_tes)
xy_pair_ids_tes = parse_filelist(args.xy_pair_ids_flist_tes)

if len(xfiles_tes) == 0 or len(yfiles_tes) == 0:
    warnings.warn("""Valid test data is not provided. 
                  Concurrence will be computed on training data, which is likely to lead to a Type I error.
                  We *STRONGLY* advise the usage of an independent sample.
                  
                  The usage of tra_data as also tes_data is only intended for debugging purposes 
                  or analyizng how the per-segment-concurrence-score (PSCS) varies within the training sample.
                  """)
    if args.val_ptg > 0 and args.PSCS_file is not None:
        print("""Setting val_ptg to 0 since the likely goal is PSCS analysis within training sample""", flush=True)
        args.val_ptg = 0
        
else:
    assert len(xfiles_tes) == len(yfiles_tes)

Ntraval = len(xfiles_traval)
Ntes = len(xfiles_tes)

file_pairs_traval = []
file_pairs_tes = []

for i in range(Ntraval):
    file_pairs_traval.append((xfiles_traval[i], yfiles_traval[i]))

for i in range(Ntes):
    file_pairs_tes.append((xfiles_tes[i], yfiles_tes[i]))

Nval = int(Ntraval*args.val_ptg)
Ntra = Ntraval-Nval
data_params = {"time_window": opts['w'], 'segs_per_pair': opts['segs_per_pair'], 'negatives_from_same_seq': True}


if __name__ == "__main__":

    print('Reading data 💾...', flush=True)
    tra_data = DataFromTextFilesND(file_pairs_traval[:Ntra], **{**data_params})
    tes_data = None

    if Ntes > 0:
        tes_data = DataFromTextFilesND(file_pairs_tes, **{**data_params})

    val_data = None
    if Nval > 0:
        val_data = DataFromTextFilesND(file_pairs_traval[Ntra:], **{**data_params})
    print('Done ✅', flush=True)

    from core.concurrence import concurrence
    import time
    t0 = time.time()
    print('Computing Concurrence coefficient... ⏳', flush=True)
    concurrence, model = concurrence(tra_data, tes_data, val_data, opts, scramble_labels=False, model_path=None)
    tf = time.time()
    print(f"Completed in {(tf-t0):.2f}s 🚀", flush=True)

    if args.PSCS_file is not None:
        
        if len(file_pairs_tes) == 0:
            file_pairs = file_pairs_traval
            xy_pair_ids = xy_pair_ids_traval
        else:
            file_pairs = file_pairs_tes
            xy_pair_ids = xy_pair_ids_tes
        
        assert len(xy_pair_ids) == len(file_pairs)
        
        import pandas as pd
        from core.concurrence import compute_PSCSs_sliding_window
        print('Computing PSCS values ... ⏳')
        t0 = time.time()
        PSCSs_list = compute_PSCSs_sliding_window(file_pairs, model, params=opts, step_size_dw=opts['w']//2)    
        tf = time.time()
        print(f"Completed in {(tf-t0):.2f}s 🚀", flush=True)
        
        ncol = max([len(x) for x in PSCSs_list])
        cols = [f'col{i}' for i in range(ncol)]
        df = pd.DataFrame(columns=cols)
        df.index.name = 'id'
        
        for i in range(len(xy_pair_ids)):
            cur = PSCSs_list[i]
            df.loc[xy_pair_ids[i]] = cur + [pd.NA] * (ncol - len(cur))

        df.to_csv(args.PSCS_file)
        print()
        print(f'💾 Saved PSCS values to {args.PSCS_file}')

    print('✅ Program completed')


