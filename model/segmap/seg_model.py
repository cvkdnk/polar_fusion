import torch
import math
import torch_scatter
from torch import nn
from torch.nn import functional as F
import spconv.pytorch as spconv

from model.segmap.mhca import MultiHeadCrossAttentionSublayer, SegMapAttentionSublayer
from model.segmap.pos_enc import PositionEmbeddingCoordsSine
from utils.model_utils import pcd_normalize


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


class SegMap_pooling_v0(nn.Module):
    """对Q用GS下采样、对KV用maxpool下采样，分别进行位置编码"""
    def __init__(self, input_dim, output_dim, compress_ratio, num_heads, pe_type='sine', pe_norm=True,
                 embed_type="pool", query_gs_ratio=None):
        self.num_channels = input_dim
        super().__init__()
        self.embed_type = embed_type
        self.query_gs_ratio = query_gs_ratio
        self.embed_pe = EmbeddingAndPosEnc(input_dim, pe_type, pe_norm, self.query_gs_ratio,
                                           self.embed_type, compress_ratio)
        self.mhca = MultiHeadCrossAttentionSublayer(input_dim, input_dim, input_dim, input_dim, num_heads)
        assert pe_type in ['sine', 'fourier'] or pe_type is None
        self.seg_head = spconv.SubMConv3d(
            input_dim, output_dim, indice_key="seg_head",
            kernel_size=3, stride=1, padding=1, bias=True
        )

    def forward(self, voxel_feats, batch_size):
        pool_feats, embedding_feats, pool_inv, _ = self.embed_pe(voxel_feats, batch_size)
        feats_mhca = self.mhca(pool_feats, embedding_feats, embedding_feats, None)
        if self.query_gs_ratio is not None:
            voxel_mhca_feats = feats_mhca.features[pool_inv]
        else:
            voxel_mhca_feats = feats_mhca.features
        # voxel_feats = voxel_feats.replace_feature(torch.cat([voxel_feats.features, voxel_mhca_feats], dim=1))
        voxel_feats = voxel_feats.replace_feature(voxel_feats.features+voxel_mhca_feats)
        logits = self.seg_head(voxel_feats)
        return logits, None  # SparseConvTensor


class SegMap_pooling(nn.Module):
    """对Q用GS下采样、对KV用maxpool下采样，分别进行位置编码"""
    def __init__(self, input_dim, output_dim, num_heads, sublayer_num, pe_type='sine', pe_norm=True, query_gs_ratio=None,
                 embed_type='ffs', embed_params=None):
        self.num_channels = input_dim
        super().__init__()
        self.embed_type = embed_type
        self.embed_pe = EmbeddingAndPosEnc(input_dim, pe_type, pe_norm, query_gs_ratio, embed_type, embed_params)
        self.mha_stack = nn.ModuleList([
            SegMapAttentionSublayer(input_dim, input_dim, input_dim, input_dim, num_heads)
            for _ in range(sublayer_num)
        ])
        self.seg_head = spconv.SubMConv3d(
            input_dim, output_dim, indice_key="seg_head",
            kernel_size=3, stride=1, padding=1, bias=True
        )

    def forward(self, voxel_feats, batch_size):
        pool_feats, embedding_feats, pool_inv, ffs_feats = self.embed_pe(voxel_feats, batch_size)
        feats_mhca, feats_mhsa = pool_feats, embedding_feats
        for sublayer in self.mha_stack:
            feats_mhca, feats_mhsa = sublayer(feats_mhca, feats_mhsa, None)
        if pool_inv is not None:
            voxel_mhca_feats = feats_mhca.features[pool_inv]
        else:
            voxel_mhca_feats = feats_mhca.features
        # voxel_feats = voxel_feats.replace_feature(torch.cat([voxel_feats.features, voxel_mhca_feats], dim=1))
        # voxel_feats = voxel_feats.replace_feature(voxel_feats.features+voxel_mhca_feats)
        voxel_feats = voxel_feats.replace_feature(voxel_mhca_feats)
        logits = self.seg_head(voxel_feats)
        return logits, ffs_feats  # SparseConvTensor


class EmbeddingAndPosEnc(nn.Module):
    def __init__(self, dim, pe_type='sine', pe_norm=True, query_gs_ratio=None,
                 embed_type='ffs', embed_params=None):
        """Embedding features before attention layers, and add position encoding.

        :param dim: channel number
        :param pe_type: position encoding type, 'sine' or 'fourier'
        :param pe_norm: if normalize position encoding
        :param query_gs_ratio: If None, not use grid sampling for query feats. Else, need a list like [2, 2, 2]
        :param embed_type: 'ffs' or 'pool'
        :param embed_params: if embed_type is 'ffs', need a int number. else need a list like [40, 15, 16]
        """
        super().__init__()
        '''Query Compress'''
        self.query_gs_ratio = query_gs_ratio
        self.pool_mlp = nn.Linear(dim, dim)

        '''Position Encoding'''
        assert pe_type in ['sine', 'fourier'] or pe_type is None
        self.position_encoding = PositionEmbeddingCoordsSine(
            pos_type=pe_type,
            d_pos=dim,  # encoding channels number (128)
            normalize=pe_norm
        )

        '''Embedding Sampling'''
        self.embedding_mlp = nn.Linear(dim, dim)
        if embed_type == 'ffs':
            self.embedding = self.embedding_ffs
            self.embed_params = 1024 if embed_params is None else embed_params
            self.ffs_mlp = nn.Linear(dim, self.embed_params)
        elif embed_type == 'pool':
            self.embed_params = [40, 15, 16] if embed_params is None else embed_params
            self.embedding = self.embedding_pool
        else:
            raise NotImplementedError

    def forward(self, voxel_feats, batch_size):
        device = voxel_feats.features.device
        '''Compress Query'''
        if self.query_gs_ratio is not None:
            pool_feats = self.pool_mlp(voxel_feats.features)
            pool_coords = voxel_feats.indices / torch.tensor([1]+self.query_gs_ratio, device=device)
            pool_coords = pool_coords.type(torch.int32)
            pool_coords, unq_inv, unq_cnt = torch.unique(pool_coords, return_inverse=True, return_counts=True, dim=0)
            pool_coords *= torch.tensor([1]+self.query_gs_ratio, device=device)
            pool_coords = pool_coords.type(torch.int32)
            pool_feats = torch_scatter.scatter_max(pool_feats, unq_inv, dim=0)[0]
            pool_inv = unq_inv
        else:
            pool_feats = self.pool_mlp(voxel_feats.features)
            pool_coords = voxel_feats.indices
            pool_inv = None

        embedding_feats, embedding_coords, ffs_feats = self.embedding(voxel_feats, device)
        '''Position Encoding'''
        for i in range(batch_size):
            emb_mask = embedding_coords[:, 0] == i
            pool_mask = pool_coords[:, 0] == i
            pe_coords = torch.cat([embedding_coords[emb_mask, 1:], pool_coords[pool_mask, 1:]], dim=0)
            pe_coords = self.position_encoding(pe_coords).squeeze().transpose(1, 0)
            embedding_feats[emb_mask] += pe_coords[:emb_mask.sum()]
            pool_feats[pool_mask] += pe_coords[emb_mask.sum():]
        '''Generate SparseConvTensor'''
        pool_feats = spconv.SparseConvTensor(
            pool_feats, pool_coords,
            voxel_feats.spatial_shape,
            batch_size
        )
        embedding_feats = spconv.SparseConvTensor(
            embedding_feats, embedding_coords.type(torch.int32),
            voxel_feats.spatial_shape,
            batch_size
        )
        return pool_feats, embedding_feats, pool_inv, ffs_feats

    def embedding_pool(self, voxel_feats, device):
        embedding_feats = self.embedding_mlp(voxel_feats.features)
        embedding_coords = voxel_feats.indices / torch.tensor([1] + self.embed_params, device=device)
        embedding_coords = embedding_coords.type(torch.int32)
        unq, unq_inv, unq_cnt = torch.unique(embedding_coords, return_inverse=True, return_counts=True, dim=0)
        embedding_feats, idx = torch_scatter.scatter_max(embedding_feats, unq_inv, dim=0)
        embedding_coords = embedding_coords[idx].type(torch.float32).mean(dim=1)
        embedding_coords *= torch.tensor([1] + self.embed_params, device=device)
        return embedding_feats, embedding_coords, None

    def embedding_ffs(self, voxel_feats, device):
        embedding_feats = self.embedding_mlp(voxel_feats.features)
        ffs_feats = self.ffs_mlp(voxel_feats.features)
        ffs_indices = ffs_feats.max(dim=0).indices
        ffs_return = ffs_feats[ffs_indices]
        ffs_indices = torch.unique(ffs_indices)
        embedding_feats = embedding_feats[ffs_indices]
        embedding_coords = voxel_feats.indices[ffs_indices]
        return embedding_feats, embedding_coords, ffs_return



class SegMap_pooling_history(nn.Module):
    """对Q用GS下采样、对KV用maxpool下采样，分别进行位置编码"""
    def __init__(self, input_dim, output_dim, compress_ratio, num_heads, pe_type='sine', pe_norm=True):
        self.num_channels = input_dim
        super().__init__()
        self.compress_ratio = compress_ratio
        self.query_gs_ratio = [2, 2, 4]
        self.mhca = MultiHeadCrossAttentionSublayer(input_dim, input_dim, input_dim, input_dim, num_heads)
        self.pool_mlp = nn.Linear(input_dim, input_dim)
        self.embedding_mlp = nn.Linear(input_dim, input_dim)
        assert pe_type in ['sine', 'fourier'] or pe_type is None
        self.position_encoding = PositionEmbeddingCoordsSine(
            pos_type=pe_type,
            d_pos=input_dim,  # encoding channels number (128)
            normalize=pe_norm
        )
        self.seg_head = spconv.SubMConv3d(
            input_dim*2, output_dim, indice_key="seg_head",
            kernel_size=3, stride=1, padding=1, bias=True
        )

    def forward(self, voxel_feats, batch_size):
        device = voxel_feats.features.device
        # 生成池化特征，相当于用GS下采样
        pool_feats = self.pool_mlp(voxel_feats.features)
        pool_coords = voxel_feats.indices / torch.tensor([1]+self.query_gs_ratio, device=device)
        pool_coords = pool_coords.type(torch.int32)
        pool_coords, unq_inv, unq_cnt = torch.unique(pool_coords, return_inverse=True, return_counts=True, dim=0)
        pool_coords = pool_coords.type(torch.int32)
        pool_feats = torch_scatter.scatter_max(pool_feats, unq_inv, dim=0)[0]
        pool_inv = unq_inv

        # 生成压缩嵌入特征，使用maxpool寻找最大特征，再下采样
        embedding_feats = self.embedding_mlp(voxel_feats.features)
        embedding_coords = voxel_feats.indices / torch.tensor([1] + self.compress_ratio, device=device)
        embedding_coords = embedding_coords.type(torch.int32)
        unq, unq_inv, unq_cnt = torch.unique(embedding_coords, return_inverse=True, return_counts=True, dim=0)
        embedding_feats, idx = torch_scatter.scatter_max(embedding_feats, unq_inv, dim=0)
        embedding_coords = embedding_coords[idx].type(torch.float32).mean(dim=1)  # (N_e, 3)

        # 位置编码
        for i in range(batch_size):
            emb_mask = embedding_coords[:, 0] == i
            pool_mask = pool_coords[:, 0] == i
            pe_coords = torch.cat([embedding_coords[emb_mask, 1:], pool_coords[pool_mask, 1:]], dim=0)
            pe_coords = self.position_encoding(pe_coords).squeeze().transpose(1, 0)
            embedding_feats[emb_mask] += pe_coords[:emb_mask.sum()]
            pool_feats[pool_mask] += pe_coords[emb_mask.sum():]

        # 通过单层多头交叉注意力，向压缩地图查询特征
        embedding_feats = spconv.SparseConvTensor(
            embedding_feats, embedding_coords.int(),
            [x // y for x, y in zip(voxel_feats.spatial_shape, self.compress_ratio)],
            batch_size
        )
        pool_feats = spconv.SparseConvTensor(
            pool_feats, pool_coords,
            [x // y for x, y in zip(voxel_feats.spatial_shape, self.query_gs_ratio)],
            batch_size
        )
        feats_mhca = self.mhca(pool_feats, embedding_feats, embedding_feats, None)
        voxel_mhca_feats = feats_mhca.features[pool_inv]
        voxel_feats = voxel_feats.replace_feature(torch.cat([voxel_feats.features, voxel_mhca_feats], dim=1))
        logits = self.seg_head(voxel_feats)
        return logits  # SparseConvTensor