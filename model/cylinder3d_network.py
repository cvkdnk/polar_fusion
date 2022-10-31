import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.pf_base_class import PFBaseClass

def point_mlp(fea_in, fea_out):
    mlp = nn.Sequential(
        nn.Linear(fea_in, fea_out),
        nn.BatchNorm1d(fea_out),
        nn.ReLU()
    )
    return mlp


class CylinderPointMLP(nn.Module):
    def __init__(
            self, grid_size,
            in_fea_dim=3,
            mlp_channels=None,
            out_pt_fea_dim=64,
            max_pt_per_encoder=64
    ):
        super(CylinderPointMLP, self).__init__()

        if mlp_channels is None:
            mlp_channels = [in_fea_dim, 64, 128, 256, 64]

        self.mlp = nn.Sequential(nn.BatchNorm1d(mlp_channels[0]))
        for i in range(len(mlp_channels) - 1):
            self.mlp.add_module(
                f"point_mlp_{i}",
                point_mlp(mlp_channels[i], mlp_channels[i + 1])
            )
        self.mlp.add_module("linear", nn.Linear(mlp_channels[-1], out_pt_fea_dim))

        self.max_pt = max_pt_per_encoder
        self.grid_size = grid_size
        self.local_pool_op = torch.nn.MaxPool2d(kernel_size=3, stride=1,
                                                padding=1, dilation=1)

    def forward(self, point_feats_st, p2v_indices):
        # process feature
        processed_cat_pt_fea = self.mlp(point_feats_st.F)


        return unq, processed_pooled_data

