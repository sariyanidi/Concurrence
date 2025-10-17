#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 15:25:32 2024

@author: sariyanide
"""
import os
import torch
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset

def znorm(x):
    return (x-np.mean(x))/(np.std(x)+1e-10)


def get_valid_study_ids(data_rootdir, input_study_ids):
    
    study_ids_part = set()
    study_ids_conf = set()
    
    for f in glob(f'{data_rootdir}/*participant'):
        bn = os.path.basename(f)
        study_ids_part.add(bn.split('part')[0])

    for f in glob(f'{data_rootdir}/*confederate'):
        bn = os.path.basename(f)
        study_ids_conf.add(bn.split('conf')[0])
    
    study_ids_all = study_ids_part.intersection(study_ids_conf)
    study_ids_all = study_ids_all.intersection(input_study_ids)
    
    return study_ids_all




def get_valid_study_ids_BBL(data_rootdir):
    
    study_ids = set()
    
    for f in glob(f'{data_rootdir}/*.1D'):
        bn = os.path.basename(f)
        study_ids.add(bn.split('.1D')[0])
    
    return study_ids

    


class DyadicSegmentDatasetBase(Dataset):
    
    def __init__(self,
                 data_rootdir,
                 randomize_neg_confederate=True,
                 pos_data_ratio=0.5,
                 segs_per_subject=50,
                 FPS=15,
                 partition='tra', # tra or val
                 tra_percentage=0.8,
                 time_window=4,
                 use_diff=False,
                 study_ids=None,
                 include_study_ids=None, labels = None,
                 seed=1907,
                 feat_ix0=0,
                 feat_ixf=None):  # secs
        super().__init__()

        self.pos_data_ratio = pos_data_ratio
        self.segs_per_subject = segs_per_subject
        self.randomize_neg_confederate = randomize_neg_confederate
        
        self.time_windowF = time_window*FPS
        self.data_rootdir = data_rootdir
        
        assert study_ids is not None
        
        if study_ids is not None:
            self.study_ids = study_ids
        else:
            study_ids_part = set()
            study_ids_conf = set()
            
            for f in glob(f'{data_rootdir}/*participant'):
                bn = os.path.basename(f)
                study_ids_part.add(bn.split('part')[0])

            for f in glob(f'{data_rootdir}/*confederate'):
                bn = os.path.basename(f)
                study_ids_conf.add(bn.split('conf')[0])
            
            study_ids_all = study_ids_part.intersection(study_ids_conf)
            if include_study_ids is not None:
                study_ids_all = study_ids_all.intersection(include_study_ids)
            
            study_ids_all = list(study_ids_all)
            
            random.seed(seed)
            random.shuffle(study_ids_all)
            
            self.partition = partition
            
            if self.partition == 'tra':
                self.study_ids = study_ids_all[:int(round(tra_percentage*len(study_ids_all)))]
            elif self.partition == 'val':
                self.study_ids = study_ids_all[int(round(tra_percentage*len(study_ids_all))):]
            
            self.study_ids.sort()
        
        self.num_subjs = len(self.study_ids)
        
        sample_data = np.loadtxt(f'{data_rootdir}/{self.study_ids[0]}participant')[:,feat_ix0:feat_ixf]
        
        if use_diff:
            self.part_data = [torch.from_numpy(np.diff(np.loadtxt(f'{data_rootdir}/{s}participant')[:,feat_ix0:feat_ixf], axis=0).astype(np.float32)) for s in self.study_ids]
            self.conf_data = [torch.from_numpy(np.diff(np.loadtxt(f'{data_rootdir}/{s}confederate')[:,feat_ix0:feat_ixf], axis=0).astype(np.float32)) for s in self.study_ids]
        else:            
            self.part_data = [torch.from_numpy(np.loadtxt(f'{data_rootdir}/{s}participant')[:,feat_ix0:feat_ixf].astype(np.float32)) for s in self.study_ids]
            self.conf_data = [torch.from_numpy(np.loadtxt(f'{data_rootdir}/{s}confederate')[:,feat_ix0:feat_ixf].astype(np.float32)) for s in self.study_ids]
        
            
        self.num_features = sample_data.shape[1]


    def __len__(self):
        return len(self.study_ids)*self.segs_per_subject
    

    def __getitem__(self, sample_ix):
        part_subj_ix = sample_ix % self.num_subjs
        label = torch.rand(1) > 0.5

        # x = torch.from_numpy(np.loadtxt(f'{self.data_rootdir}/{self.study_ids[part_subj_ix]}participant').astype(np.float32))
        # x = torch.from_numpy(self.part_data[part_subj_ix])
        x = self.part_data[part_subj_ix]
        
        if not label and self.randomize_neg_confederate:
            conf_subj_ix = torch.randint(self.num_subjs, (1,))
        else:
            conf_subj_ix = part_subj_ix
            
        # y = torch.from_numpy(np.loadtxt(f'{self.data_rootdir}/{self.study_ids[conf_subj_ix]}confederate').astype(np.float32))
        # y = torch.from_numpy(self.conf_data[conf_subj_ix])
        y = self.conf_data[conf_subj_ix]

        t0_part = torch.randint(0, x.shape[0]-self.time_windowF, (1,))
        if label:
            t0_conf = t0_part
        else:
            t0_conf = torch.randint(0, y.shape[0]-self.time_windowF, (1,))
        
        xslice = x[t0_part:t0_part+self.time_windowF,:].clone()
        yslice = y[t0_conf:t0_conf+self.time_windowF,:].clone()
        
        data = torch.cat([xslice, yslice], dim=1)
        return data, label.int() 
    
    
    
    
    
    
    
class DyadicSegmentDatasetSynced(DyadicSegmentDatasetBase):
    
    def __init__(self , *args, **kwargs):  # secs
        super().__init__(*args, **kwargs)

    def __len__(self):
        return len(self.study_ids)*self.segs_per_subject
    

    def __getitem__(self, sample_ix):
        part_subj_ix = sample_ix % self.num_subjs
        label = torch.rand(1) > 0.5

        # x = torch.from_numpy(np.loadtxt(f'{self.data_rootdir}/{self.study_ids[part_subj_ix]}participant').astype(np.float32))
        # x = torch.from_numpy(self.part_data[part_subj_ix])
        x = self.part_data[part_subj_ix]
        
        if not label and self.randomize_neg_confederate:
            conf_subj_ix = torch.randint(self.num_subjs, (1,))
        else:
            conf_subj_ix = part_subj_ix
            
        # y = torch.from_numpy(np.loadtxt(f'{self.data_rootdir}/{self.study_ids[conf_subj_ix]}confederate').astype(np.float32))
        # y = torch.from_numpy(self.conf_data[conf_subj_ix])
        y = self.conf_data[conf_subj_ix]

        t0_part = torch.randint(0, x.shape[0]-self.time_windowF, (1,))
        if label:
            t0_conf = t0_part
        else:
            t0_conf = torch.randint(0, y.shape[0]-self.time_windowF, (1,))
        
        xslice = x[t0_part:t0_part+self.time_windowF,:].clone()
        yslice = y[t0_conf:t0_conf+self.time_windowF,:].clone()
        
        data = torch.cat([xslice, yslice], dim=1)
        return data, label.int() 
    
    
    


class DatasetFromFiles1D(Dataset):
    
    def __init__(self, root_dir, segs_per_pair = 10, time_window=64,
                 tra_or_tes='tra', tra_ptg = 0.8):
        
        self.tra_ptg = 0.8
        self.tes_ptg = 1-self.tra_ptg 
        self.root_dir = root_dir
        self.w = time_window
        self.segs_per_pair = segs_per_pair
        
        super().__init__()
        
        self.xs = []
        self.ys = []
        
        Ntot_samples = len(glob(f'{root_dir}/x*txt'))
        
        Ntra_samples = int(Ntot_samples*self.tra_ptg)
        
        if tra_or_tes == 'tra':
            samples = list(range(0,Ntra_samples))
        elif tra_or_tes == 'tes':
            samples = list(range(Ntra_samples, Ntot_samples))
        
        self.sample_indices = samples
        
        for i in samples:
            x = np.loadtxt(f'{root_dir}/x{i:05d}.txt').reshape(-1,1)
            y = np.loadtxt(f'{root_dir}/y{i:05d}.txt').reshape(-1,1)
            
            x = znorm(x)
            y = znorm(y)
            
            self.xs.append(torch.from_numpy(x).float())
            self.ys.append(torch.from_numpy(y).float())
        
        self.Nsamples = len(self.xs)
        
                
    def __len__(self):
        return self.Nsamples*self.segs_per_pair
    
    
    def __getitem__(self, sample_ix):
        sample_ix = sample_ix % self.Nsamples
        label = torch.rand(1) > 0.5
        
        x = self.xs[sample_ix]
        y = self.ys[sample_ix]
        
        tx = torch.randint(0, x.shape[0]-self.w, (1,))
        if label:
            ty = tx
        else:
            ty = torch.randint(0, y.shape[0]-self.w, (1,))
        
        xslice = x[tx:tx+self.w,:].clone()
        yslice = y[ty:ty+self.w,:].clone()
        
        data = torch.cat([xslice, yslice], dim=1)
        return data, label
    
    
            
class DatasetFromNumpy(Dataset):
    
    def __init__(self, X, Y, segs_per_pair = 10, time_window=64, sample_indices=None,
                 negatives_from_same_seq=True):
                
        assert X.shape[0] == Y.shape[0]
        
        self.generator_for_labels = torch.Generator()
        self.generator_for_labels.manual_seed(1907)
        self.negatives_from_same_seq = negatives_from_same_seq
        
        self.w = time_window
        self.segs_per_pair = segs_per_pair
        
        super().__init__()
        
        self.xs = []
        self.ys = []
        
        Nsamples = X.shape[0]
            
        if sample_indices is None:
            self.sample_indices = list(range(0,Nsamples))
        else:
            self.sample_indices = sample_indices
        
        for i in self.sample_indices:
            x = X[i,:].reshape(-1,1)
            y = Y[i,:].reshape(-1,1)
            
            x = znorm(x)
            y = znorm(y)
            
            self.xs.append(torch.from_numpy(x).float())
            self.ys.append(torch.from_numpy(y).float())
        
        self.Nsamples = len(self.xs)
        
                
    def __len__(self):
        return self.Nsamples*self.segs_per_pair
    
    
    def __getitem__(self, sample_ix):

        sample_ix = sample_ix % self.Nsamples
        label = torch.rand(1, generator=self.generator_for_labels) > 0.5
        
        
        if not label:
            if self.negatives_from_same_seq:
                y_ix = sample_ix
            else:
                y_ix = torch.randint(self.Nsamples, (1,))
        else:
            y_ix = sample_ix
            
        x = self.xs[sample_ix]
        y = self.ys[y_ix]
        
        tx = torch.randint(0, x.shape[0]-self.w, (1,))
        if label:
            ty = tx
        else:
            ty = torch.randint(0, y.shape[0]-self.w, (1,))
        
        xslice = x[tx:tx+self.w,:].clone()
        yslice = y[ty:ty+self.w,:].clone()
        
        data = torch.cat([xslice, yslice], dim=1)
        return data, label

        




class DataFromTextFilesND(Dataset):
    
    def __init__(self, file_pairs, 
                 segs_per_pair = 10, 
                 time_window=64, 
                 # use_diff=False,
                 negatives_from_same_seq=True,
                 read_files_onthefly=False):
        
        self.generator_for_labels = torch.Generator()
        self.generator_for_labels.manual_seed(1907)
        self.negatives_from_same_seq = negatives_from_same_seq
        self.read_files_onthefly = read_files_onthefly
        # self.use_diff = use_diff
        
        self.w = time_window
        self.segs_per_pair = segs_per_pair
        
        super().__init__()
        
        self.xs = []
        self.ys = []
        
        Nsamples = len(file_pairs)
        
        self.Nfeats_x = None
        self.Nfeats_y = None
        
        if not self.read_files_onthefly:
            for i in range(Nsamples):                
                # x = znorm(x)
                # y = znorm(y)
                x = np.loadtxt(file_pairs[i][0])
                y = np.loadtxt(file_pairs[i][1])
                
                if len(x.shape) == 1:
                    x = x.reshape(-1,1)
                    
                if len(y.shape) == 1:
                    y = y.reshape(-1,1)
                
                if x.shape[0] != y.shape[0]:
                    print(f'Skipping pair--different temporal lenghts: between files {file_pairs[i][0]} and {file_pairs[i][1]}')
                    continue
                
                if self.Nfeats_x is None:
                    self.Nfeats_x = x.shape[1]
                
                if self.Nfeats_y is None:
                    self.Nfeats_y = y.shape[1]
                    
                assert x.shape[1] == self.Nfeats_x
                assert y.shape[1] == self.Nfeats_y

                x = torch.from_numpy(x).float()
                y = torch.from_numpy(y).float()
                    
                self.xs.append(x)
                self.ys.append(y)
        
        self.Nsamples = len(self.xs)
        
        assert self.Nsamples > 0 
        
        self.Nfeatures_in_dyad = self.xs[0].shape[1]+self.ys[0].shape[1]
        
                
    def __len__(self):
        return self.Nsamples*self.segs_per_pair
    
    
    def __getitem__(self, sample_ix):

        sample_ix = sample_ix % self.Nsamples
        label = torch.rand(1, generator=self.generator_for_labels) > 0.5
        
        if not label:
            if self.negatives_from_same_seq:
                y_ix = sample_ix
            else:
                y_ix = torch.randint(self.Nsamples, (1,))
        else:
            y_ix = sample_ix
            
        x = self.xs[sample_ix]
        y = self.ys[y_ix]
        
        tx = torch.randint(0, x.shape[0]-self.w, (1,))
        if label:
            ty = tx
        else:
            ty = torch.randint(0, y.shape[0]-self.w, (1,))
        
        xslice = x[tx:tx+self.w,:].clone()
        yslice = y[ty:ty+self.w,:].clone()
        
        data = torch.cat([xslice, yslice], dim=1)
        return data, label
   
    
   

#%%
if __name__ == '__main__':
    rdir = '/home/sariyanide/code/DataPairGeneration/data/CAR0.0'
    db = DatasetFromFiles1D(rdir)
    


#%%
    import numpy as np
    from scipy import ndimage, datasets
    import matplotlib.pyplot as plt
        
    plt.figure()
    h1 = np.zeros((1000,1))
    h1[200] = 50
    h1[700] = -30
    y1 = ndimage.gaussian_filter(h1, sigma=25)
    
    h2 = np.zeros((1000,1))
    h2[500] = 50
    
    y2 = ndimage.gaussian_filter(h2, sigma=30)
    y = y1+y2
    # h[600] = -100
    dy = np.diff(y,axis=0, prepend=0)

    p = np.polynomial.Polynomial([6,2,-3,1,-6])
    z1 = znorm(y/max(y))
    z2 = znorm(dy/max(dy))
    z3 = znorm(p(1*y)/max(p(y)))
    # z3 = znorm(1/(1+np.e**(-p(dy))))
    
    plt.plot(z1)
    plt.plot(z2)
    plt.legend(['x', 'y=dx'])
    # plt.plot(znorm(1/(1+np.e**(-p(dy/max(dy))))))
    """
    """
    corr=np.corrcoef(z1.T, z2.T)[1][0]
    
    print(corr)
    # print(np.corrcoef(y.T, p(dy).T))
    
    plt.figure()
    plt.scatter(z1, z2, alpha=0.4)
    plt.xlabel('x')
    plt.ylabel('dx')
    plt.title(f'Correlation={corr:.2f}')
    # db = SyntheticPair(time_window=200)
    
    
    plt.figure()
    
    plt.plot(z1)
    plt.plot(z3)
    plt.legend(['x', 'y=p(dx)'])
    
    
    plt.figure()
    plt.scatter(z1, z3, alpha=0.4)
    plt.xlabel('x')
    plt.ylabel('dx')
    plt.title(f'Correlation={np.corrcoef(z1.T, z3.T)[1][0]:.2f}')
    # db = SyntheticPair(time_window=200)
    
    g1 = np.random.randn(5000)
    g2 = np.random.randn(5000)
    plt.figure()
    plt.scatter(g1, g2, alpha=0.2)
    plt.title(f'Correlation={np.corrcoef(g1.T, g2.T)[1,0]:.2f}')
    plt.xlim((-3,3))
    plt.ylim((-3,3))

    
    x1 = np.random.rand(4000).reshape(-1,1)
    x2 = np.random.rand(4000).reshape(-1,1)
    
    x = np.concatenate((x1, x2), axis=1)
    
    
    plt.figure()
    plt.scatter(x1,x2, alpha=0.2)
    
    theta = np.pi/4
    R = np.array([[np.sin(theta), np.cos(theta)], [-np.cos(theta), np.sin(theta)]])
    xr = x@R
    plt.title(f'Correlation={np.corrcoef(x1.T, x2.T)[1,0]:.2f}')



    plt.figure()
    plt.scatter(xr[:,0], xr[:,1], alpha=0.2)
    plt.title(f'Correlation={np.corrcoef(xr[:,0:1].T, xr[:,1:2].T)[1,0]:.2f}')
    
    
    




