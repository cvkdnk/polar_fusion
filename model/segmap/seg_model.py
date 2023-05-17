import torch
import math
import torch_scatter
from torch import nn
from torch.nn import functional as F
from model.cy3d_spconv.segmentator_3d_asymm_spconv import *

class Asymm_3d(nn.Module):
    def __init__(self,
                 output_shape,
                 num_input_features=128,
                 nclasses=20, init_size=16):
        super(Asymm_3d, self).__init__()
        self.nclasses = nclasses

        sparse_shape = np.array(output_shape)
        # sparse_shape[0] = 11
        self.sparse_shape = sparse_shape

        self.downCntx = ResContextBlock(num_input_features, init_size, indice_key="pre")
        self.resBlock2 = ResBlock(init_size, 2 * init_size, 0.2, height_pooling=True, indice_key="down2")
        self.resBlock3 = ResBlock(2 * init_size, 4 * init_size, 0.2, height_pooling=True, indice_key="down3")
        self.resBlock4 = ResBlock(4 * init_size, 8 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down4")
        self.resBlock5 = ResBlock(8 * init_size, 16 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down5")

        self.upBlock0 = UpBlock(16 * init_size, 16 * init_size, indice_key="up0", up_key="down5")
        self.upBlock1 = UpBlock(16 * init_size, 8 * init_size, indice_key="up1", up_key="down4")
        self.upBlock2 = UpBlock(8 * init_size, 4 * init_size, indice_key="up2", up_key="down3")
        self.upBlock3 = UpBlock(4 * init_size, 2 * init_size, indice_key="up3", up_key="down2")

        self.ReconNet = ReconBlock(2 * init_size, 2 * init_size, indice_key="recon")

        self.logits = spconv.SubMConv3d(4 * init_size, nclasses, indice_key="logit", kernel_size=3, stride=1, padding=1,
                                        bias=True)

    def forward(self, voxel_features, coors, batch_size):
        # x = x.contiguous()
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        ret = self.downCntx(ret)
        down1c, down1b = self.resBlock2(ret)
        down2c, down2b = self.resBlock3(down1c)
        down3c, down3b = self.resBlock4(down2c)
        down4c, down4b = self.resBlock5(down3c)

        up4e = self.upBlock0(down4c, down4b)
        up3e = self.upBlock1(up4e, down3b)
        up2e = self.upBlock2(up3e, down2b)
        up1e = self.upBlock3(up2e, down1b)

        up0e = self.ReconNet(up1e)

        up0e = up0e.replace_feature(torch.cat((up0e.features, up1e.features), 1))

        logits = self.logits(up0e)
        return logits, up0e


class MultiHeadAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, output_dim, num_heads):
        super().__init__()
        self.output_dim = output_dim
        self.num_heads = num_heads
        assert output_dim % num_heads == 0 # output_dim must be divisible by num_heads
        self.dim = output_dim // num_heads # dimension of each head
        self.W_q = nn.Linear(query_dim, output_dim) # query projection
        self.W_k = nn.Linear(key_dim, output_dim) # key projection
        self.W_v = nn.Linear(value_dim, output_dim) # value projection
        self.W_o = nn.Linear(output_dim, output_dim) # output projection
        self.scale = math.sqrt(self.dim) # scale factor

    def forward(self, query, key, value, pos_enc=None):
        """query, key and value should be SparseConvTensor"""
        batch_size = query.batch_size  # get the batch size from the query SparseConvTensor
        # project query, key and value
        Q = self.W_q(query.features)  # shape: (N_q, output_dim)
        K = self.W_k(key.features)  # shape: (N_k, output_dim)
        V = self.W_v(value.features)  # shape: (N_v, output_dim)
        # split Q, K and V into multiple heads
        Q = Q.view(-1, self.num_heads, self.dim).permute(1, 0, 2)  # shape: (num_heads, N_q, dim)
        K = K.view(-1, self.num_heads, self.dim).permute(1, 0, 2) # shape: (num_heads, N_k, dim)
        V = V.view(-1, self.num_heads, self.dim).permute(1, 0, 2) # shape: (num_heads, N_v, dim)
        # calculate scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale # shape: (num_heads, N_q, N_k)
        if pos_enc is not None:
            scores += pos_enc
        # create a mask to avoid attention between different batches
        mask = query.indices[:, 0].unsqueeze(-1) != key.indices[:, 0].unsqueeze(-2) # shape: (N_q, N_k)
        mask = mask.unsqueeze(0) # shape: (1, N_q, N_k)
        mask = mask.expand(self.num_heads, -1, -1) # shape: (num_heads, N_q, N_k)
        # apply mask to scores
        scores = scores.masked_fill(mask == True, -1e9) # shape: (num_heads, N_q, N_k)
        # apply softmax
        scores = F.softmax(scores, dim=-1) # shape: (num_heads, N_q, N_k)
        # apply attention
        h = torch.matmul(scores, V) # shape: (num_heads, N_q, dim)
        # concatenate heads
        h = h.permute(1, 0, 2).contiguous() # shape: (N_q, num_heads, dim)
        h = h.view(-1, self.output_dim) # shape: (N_q, output_dim)
        # apply output projection
        h = self.W_o(h) # shape: (N_q, output_dim)
        # create a SparseConvTensor for the output
        output = spconv.SparseConvTensor(h, query.indices,
                                         query.spatial_shape,
                                         query.batch_size)
        return output


class MultiHeadCrossAttentionSublayer(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, output_dim, num_heads):
        super().__init__()
        self.mhca = MultiHeadAttention(query_dim, key_dim, value_dim, output_dim, num_heads) # use the previous MultiHeadAttention class
        self.ln1 = nn.LayerNorm(output_dim) # layer normalization 1
        self.mlp = nn.Sequential( # feed-forward network
            nn.Linear(output_dim, output_dim * 4), # first linear layer
            nn.ReLU(), # activation function
            nn.Linear(output_dim * 4, output_dim) # second linear layer
        )
        self.ln2 = nn.LayerNorm(output_dim) # layer normalization 2

    def forward(self, query, key, value, pois_encoding=None):
        feats_mhca = self.mhca(query, key, value, pois_encoding) # calculate multi-head cross attention
        res0 = self.ln1(feats_mhca.features + query.features) # add residual connection and layer normalization 1
        feats_mlp = self.mlp(res0) # calculate feed-forward network
        feats_mlp = self.ln2(feats_mlp + res0) # add residual connection and layer normalization 2
        feats_mhca.replace_feature(feats_mlp) # replace the features of the multi-head cross attention with the output of the feed-forward network
        return feats_mhca


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
        self.seg_head =  spconv.SubMConv3d(
            128, num_classed, indice_key="seg_head",
            kernel_size=3, stride=1, padding=1, bias=True
        )

    def forward(self, voxel_feats, batch_size):
        device = voxel_feats.features.device
        # 生成压缩特征
        compress_feats = self.gen_compress_map(voxel_feats, device, batch_size)
        compress_feats.replace_feature(self.compress_mlp(compress_feats.features))
        N, N_s = voxel_feats.indices.shape[0], compress_feats.indices.shape[0]
        position_feats = voxel_feats.indices[:, 1:].unsqueeze(1).extend(N, N_s, 3) - \
                         compress_feats.indices[:, 1:].unsqueeze(0).extend(N, N_s, 3)
        position_feats = position_feats.type(torch.float32)
        position_feats = self.position_mlp(position_feats).squeeze()  # (N, N_s)

        # 通过单层多头交叉注意力，向压缩地图查询特征
        feats_mhca = self.MHCA(voxel_feats, compress_feats, compress_feats, position_feats)
        logits_mhca = self.seg_head(feats_mhca)
        return logits_mhca, feats_mhca  # 两个都是SparseConvTensor

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

