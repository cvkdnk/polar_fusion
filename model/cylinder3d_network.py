import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter


def point_mlp(fea_in, fea_out):
    mlp = nn.Sequential(
        nn.Linear(fea_in, fea_out),
        nn.BatchNorm1d(fea_out),
        nn.ReLU()
    )
    return mlp


class cylinder_fea(nn.Module):
    def __init__(
            self, grid_size,
            fea_dim=3,
            out_pt_fea_dim=64,
            max_pt_per_encoder=64
    ):
        super(cylinder_fea, self).__init__()

        self.point_module = nn.Sequential(
            nn.BatchNorm1d(fea_dim),
            point_mlp(fea_dim, 64),
            point_mlp(64, 128),
            point_mlp(128, 256),
            nn.Linear(256, out_pt_fea_dim)
        )

        self.max_pt = max_pt_per_encoder
        self.grid_size = grid_size
        self.local_pool_op = torch.nn.MaxPool2d(kernel_size=3, stride=1,
                                                padding=1, dilation=1)
        self.pool_dim = out_pt_fea_dim

    def forward(self, pt_fea, xy_ind):
        pass

