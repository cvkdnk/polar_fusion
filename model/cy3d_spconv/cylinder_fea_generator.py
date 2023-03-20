# -*- coding:utf-8 -*-
# author: Xinge

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numba as nb
import multiprocessing
import torch_scatter


class cylinder_fea(nn.Module):

    def __init__(self,
                 in_fea_dim=3,
                 out_pt_fea_dim=64,
                 fea_compre=False,
                 ):
        super(cylinder_fea, self).__init__()

        self.PPmodel = nn.Sequential(
            nn.BatchNorm1d(in_fea_dim),

            nn.Linear(in_fea_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, out_pt_fea_dim)
        )

        self.fea_compre = fea_compre
        kernel_size = 3
        self.local_pool_op = torch.nn.MaxPool2d(kernel_size, stride=1,
                                                padding=(kernel_size - 1) // 2,
                                                dilation=1)
        self.pool_dim = out_pt_fea_dim

        # point feature compression
        if self.fea_compre is not None:
            self.fea_compression = nn.Sequential(
                nn.Linear(self.pool_dim, self.fea_compre),
                nn.ReLU())
            self.pt_fea_dim = self.fea_compre
        else:
            self.pt_fea_dim = self.pool_dim

    def forward(self, point_feats, pt_vox_coords):
        cur_dev = point_feats[0].get_device()
        cat_pt_vox_coords = []
        for batch_idx in range(len(point_feats)):
            cat_pt_vox_coords.append(F.pad(pt_vox_coords[batch_idx], (1, 0), 'constant', value=batch_idx))

        cat_pt_fea = torch.cat(point_feats, dim=0)
        cat_pt_vox_coords = torch.cat(cat_pt_vox_coords, dim=0)
        pt_num = cat_pt_vox_coords.shape[0]

        # shuffle the data
        # shuffled_ind = torch.randperm(pt_num, device=cur_dev)
        # cat_pt_fea = cat_pt_fea[shuffled_ind, :]
        # cat_pt_vox_coords = cat_pt_vox_coords[shuffled_ind, :]

        # unique xy grid index
        unq, unq_inv, unq_cnt = torch.unique(cat_pt_vox_coords, return_inverse=True, return_counts=True, dim=0)
        unq = unq.type(torch.int64)

        # process feature
        processed_cat_pt_fea = self.PPmodel(cat_pt_fea)
        pooled_data = torch_scatter.scatter_max(processed_cat_pt_fea, unq_inv, dim=0)[0]

        if self.fea_compre:
            processed_pooled_data = self.fea_compression(pooled_data)
        else:
            processed_pooled_data = pooled_data

        return unq, processed_pooled_data
