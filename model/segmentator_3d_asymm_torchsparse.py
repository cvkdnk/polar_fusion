import numpy as np
from torchsparse import nn as spnn
from torchsparse import SparseTensor
from torchsparse.nn.utils import fapply
import torch
from torch import nn


def conv3x3x3(in_channels, out_channels, stride=1):
    return spnn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, bias=False)


def conv1x3x3(in_planes, out_planes, stride=1):
    return spnn.Conv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=stride, bias=False)


def conv1x1x3(in_planes, out_planes, stride=1):
    return spnn.Conv3d(in_planes, out_planes, kernel_size=(1, 1, 3), stride=stride, bias=False)


def conv1x3x1(in_planes, out_planes, stride=1):
    return spnn.Conv3d(in_planes, out_planes, kernel_size=(1, 3, 1), stride=stride, bias=False)


def conv3x1x1(in_planes, out_planes, stride=1):
    return spnn.Conv3d(in_planes, out_planes, kernel_size=(3, 1, 1), stride=stride, bias=False)


def conv3x1x3(in_planes, out_planes, stride=1):
    return spnn.Conv3d(in_planes, out_planes, kernel_size=(3, 1, 3), stride=stride, bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return spnn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SpnnSigmoid(nn.Sigmoid):

    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)


def conv_block(conv_func, act_func, in_channels, out_channels, stride=1):
    return nn.Sequential(
        conv_func(in_channels, out_channels, stride=stride),
        spnn.BatchNorm(out_channels),
        act_func()
    )


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pooling=True, height_pooling=False, act_func=spnn.LeakyReLU):
        super(ResBlock, self).__init__()
        self.conv_branch_1 = nn.Sequential(
            conv_block(conv3x1x3, act_func, in_channels, out_channels, stride=1),
            conv_block(conv1x3x3, act_func, out_channels, out_channels, stride=1)
        )
        self.conv_branch_2 = nn.Sequential(
            conv_block(conv1x3x3, act_func, in_channels, out_channels, stride=1),
            conv_block(conv3x1x3, act_func, out_channels, out_channels, stride=1)
        )
        self.pooling = pooling
        if pooling:
            if height_pooling:
                self.pool = spnn.Conv3d(out_channels, out_channels, kernel_size=3, stride=2, bias=False)
            else:
                self.pool = spnn.Conv3d(out_channels, out_channels, kernel_size=3, stride=(2, 2, 1), bias=False)

    def forward(self, x):
        out_1 = self.conv_branch_1(x)
        out_2 = self.conv_branch_2(x)
        out_1.feats += out_2.feats
        if self.pooling:
            out_pooling = self.pool(out_1)
            return out_pooling, out_1
        return out_1


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_func=spnn.LeakyReLU):
        super(UpBlock, self).__init__()
        self.up_block = nn.Sequential(
            conv_block(conv3x3x3, act_func, in_channels, out_channels),
            spnn.Conv3d(out_channels, out_channels, kernel_size=3, bias=False, transposed=True)
        )
        self.conv_x3 = nn.Sequential(
            conv_block(conv1x3x3, act_func, out_channels, out_channels),
            conv_block(conv3x1x3, act_func, out_channels, out_channels),
            conv_block(conv3x3x3, act_func, out_channels, out_channels),
        )

    def forward(self, x, skip):
        out = self.up_block(x)
        out.feats += skip.feats
        return self.conv_x3(out)


class DDCM(nn.Module):
    def __init__(self, in_channels, out_channels, act_func=SpnnSigmoid):
        super(DDCM, self).__init__()
        self.conv_branch_1 = conv_block(conv3x1x1, act_func, in_channels, out_channels)
        self.conv_branch_2 = conv_block(conv1x3x1, act_func, in_channels, out_channels)
        self.conv_branch_3 = conv_block(conv1x1x3, act_func, in_channels, out_channels)

    def forward(self, x):
        shortcut1 = self.conv_branch_1(x)
        shortcut2 = self.conv_branch_2(x)
        shortcut3 = self.conv_branch_3(x)

        shortcut1.feats = shortcut1.feats + shortcut2.feats + shortcut3.feats
        shortcut1.feats *= x.feats

        return shortcut1


class Asymm_3d_spconv(nn.Module):
    def __init__(self, output_shape, num_input_feats=128, num_classes=20, init_size=16):
        super(Asymm_3d_spconv, self).__init__()
        self.num_classes = num_classes
        self.sparse_shape = np.array(output_shape)

        self.init_block = ResBlock(num_input_feats, init_size, pooling=False)
        self.resBlock1 = ResBlock(init_size, 2*init_size, pooling=True, height_pooling=True)
        self.resBlock2 = ResBlock(2*init_size, 4*init_size, pooling=True, height_pooling=True)
        self.resBlock3 = ResBlock(4*init_size, 8*init_size, pooling=True, height_pooling=False)
        self.resBlock4 = ResBlock(8 * init_size, 16 * init_size, pooling=True, height_pooling=False)

        self.upBlock4 = UpBlock(16*init_size, 16*init_size)
        self.upBlock3 = UpBlock(16 * init_size, 8 * init_size)
        self.upBlock2 = UpBlock(8 * init_size, 4 * init_size)
        self.upBlock1 = UpBlock(4 * init_size, 2 * init_size)

        self.DDCM = DDCM(2*init_size, 2*init_size)
        self.logits = spnn.Conv3d(4*init_size, num_classes, kernel_size=3, stride=1, bias=True)

    def forward(self, voxel_feats_st):
        ret = self.init_block(voxel_feats_st)
        down1c, down1b = self.resBlock1(ret)
        down2c, down2b = self.resBlock2(down1c)
        down3c, down3b = self.resBlock3(down2c)
        down4c, down4b = self.resBlock4(down3c)

        up4e = self.upBlock4(down4c, down4b)
        up3e = self.upBlock3(up4e, down3b)
        up2e = self.upBlock2(up3e, down2b)
        up1e = self.upBlock1(up2e, down1b)

        up0e = self.DDCM(up1e)
        up0e.feats = torch.cat((up0e.feats, up1e.feats), 1)

        logits = self.logits(up0e)
        return logits

