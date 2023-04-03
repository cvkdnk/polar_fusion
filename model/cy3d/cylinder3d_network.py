import torch
import torch.nn as nn
from torchsparse import SparseTensor
import torch_scatter


def point_mlp(fea_in, fea_out):
    mlp = nn.Sequential(
        nn.Linear(fea_in, fea_out),
        nn.BatchNorm1d(fea_out),
        nn.ReLU()
    )
    return mlp


class CylinderPointMLP(nn.Module):
    def __init__(
            self,
            in_fea_dim=3,
            mlp_channels=None,
            out_pt_fea_dim=64,
            fea_compre=False,
    ):
        super(CylinderPointMLP, self).__init__()

        if mlp_channels is None:
            mlp_channels = [in_fea_dim, 64, 128, 256, 64]
        else:
            mlp_channels.insert(0, in_fea_dim)

        self.bn0 = nn.BatchNorm1d(mlp_channels[0])
        self.mlp = nn.ModuleList([
            point_mlp(mlp_channels[i], mlp_channels[i+1]) for i in range(len(mlp_channels)-1)
        ])
        self.gen_feats = nn.Linear(mlp_channels[-1], out_pt_fea_dim)
        self.fea_compre = fea_compre

        if self.fea_compre is not None:
            self.fea_compression = nn.Sequential(
                nn.Linear(out_pt_fea_dim, self.fea_compre),
                nn.ReLU()
            )

    def forward(self, point_feats_st, voxel_feats_st, upsampling_index=None):
        if upsampling_index is None:
            upsampling_index = torch.unique(point_feats_st.coords, return_inverse=True, dim=0)[1]
        if isinstance(upsampling_index, list):
            batch_coords = voxel_feats_st.C[..., 3]
            batch_voxel_num = 0
            for i in range(len(upsampling_index)):
                batch_mask = batch_coords == i
                upsampling_index[i] += batch_voxel_num
                batch_voxel_num += torch.sum(batch_mask)
            upsampling_index = torch.cat(upsampling_index, dim=0)
        # process feature
        processed_pt_feats = self.bn0(point_feats_st.feats)
        skip_pt_feats = []
        for module in self.mlp:
            processed_pt_feats = module(processed_pt_feats)
            skip_pt_feats.append(processed_pt_feats)

        if self.fea_compre is not None:
            processed_pt_feats = self.fea_compression(processed_pt_feats)
        new_pt_feats_st = SparseTensor(
            feats=processed_pt_feats,
            coords=point_feats_st.coords,
        )
        # voxel feature
        pool_feats = torch_scatter.scatter_max(
            processed_pt_feats,
            upsampling_index,
            dim=0
        )[0]

        voxel_feats_st.feats = pool_feats
        return new_pt_feats_st, voxel_feats_st, skip_pt_feats


class PointWiseRefinement(nn.Module):
    def __init__(self, init_size, mlp_channels=None, num_classes=20):
        super(PointWiseRefinement, self).__init__()
        if mlp_channels is None:
            mlp_channels = [64, 128, 256, 64]
        else:
            mlp_channels.pop(0)
        mlp1 = point_mlp(4*init_size+mlp_channels[3], 8*init_size)
        mlp2 = point_mlp(8*init_size+mlp_channels[2], 4*init_size)
        mlp3 = point_mlp(4*init_size+mlp_channels[1], 2*init_size)
        mlp4 = point_mlp(2*init_size+mlp_channels[0], init_size)
        self.mlp = nn.ModuleList([mlp1, mlp2, mlp3, mlp4])
        self.gen_logits = nn.Linear(init_size, num_classes, bias=True)

    def forward(self, pt_feats, skip_pt_feats):
        for skip, mlp_module in zip(reversed(skip_pt_feats), self.mlp):
            pt_feats = torch.cat((pt_feats, skip), dim=1)
            pt_feats = mlp_module(pt_feats)
        logits = self.gen_logits(pt_feats)
        return logits
