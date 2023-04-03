# -*- coding:utf-8 -*-
# author: Xinge

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numba as nb
import multiprocessing
import torch_scatter


class PointInitFeats(nn.Module):
    def __init__(self):
        super(PointInitFeats, self).__init__()

    def forward(self, point_feats, pt_vox_coords):



