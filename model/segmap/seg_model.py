import torch
import math
import torch_scatter
from torch import nn
from torch.nn import functional as F
import spconv.pytorch as spconv

from model.segmap.mhca import MultiHeadCrossAttentionSublayer


class SegmentorMap(nn.Module):
    def __init__(self, compress_ratio, init_size=16, num_classed=20):
        super(SegmentorMap, self).__init__()
        self.compress_ratio = compress_ratio
        self.init_size = init_size
        self.num_classed = num_classed
        self.compress_mlp = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        self.position_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.MHCA = MultiHeadCrossAttentionSublayer(4 * init_size, 4 * init_size, 4 * init_size, 128, 8)
        self.seg_head = spconv.SubMConv3d(
            128, num_classed, indice_key="seg_head",
            kernel_size=3, stride=1, padding=1, bias=True
        )

    def gen_compress_map(self, sp_feats, device, batch_size):
        """super voxel pooling"""
        compress_coords = sp_feats.indices / torch.tensor([1] + self.compress_ratio, device=device)
        compress_coords = compress_coords.type(torch.int32)
        unq, unq_inv, unq_cnt = torch.unique(compress_coords, return_inverse=True, return_counts=True, dim=0)
        unq = unq.type(torch.int32)

        compress_feats = torch_scatter.scatter_add(F.softmax(sp_feats.features, dim=1), unq_inv, dim=0)
        compress_feats = spconv.SparseConvTensor(
            compress_feats, unq, [x // y for x, y in zip(sp_feats.spatial_shape, self.compress_ratio)],
            batch_size
        )
        return compress_feats

    def forward(self, voxel_feats, batch_size):
        device = voxel_feats.features.device
        # 生成压缩特征
        compress_feats = self.gen_compress_map(voxel_feats, device, batch_size)
        compress_feats.replace_feature(self.compress_mlp(compress_feats.features))
        N, N_s = voxel_feats.indices.shape[0], compress_feats.indices.shape[0]
        # voxel_feats.indices: (N, 4), compress_feats.indices: (N_s, 4), [:, 0]是batch_idx
        position_feats = voxel_feats.indices[:, 1:].unsqueeze(1).expand(N, N_s, 3) - \
                         compress_feats.indices[:, 1:].unsqueeze(0).expand(N, N_s, 3)
        position_feats = position_feats.type(torch.float32).view(N*N_s, 3)  # (N, N_s, 3)

        position_feats = self.position_mlp(position_feats).view(N, N_s)  # (N, N_s)

        # 通过单层多头交叉注意力，向压缩地图查询特征
        feats_mhca = self.MHCA(voxel_feats, compress_feats, compress_feats, position_feats)
        logits_mhca = self.seg_head(feats_mhca)
        return logits_mhca  # SparseConvTensor


class SegMap_pooling(nn.Module):
    def __init__(self, input_dim, output_dim, compress_ratio, num_heads):
        super().__init__()
        self.mhca = MultiHeadCrossAttentionSublayer(input_dim, input_dim, input_dim, output_dim, num_heads)
        


