import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv


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
