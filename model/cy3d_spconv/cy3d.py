from utils.pf_base_class import PFBaseClass
from torch import nn


class cylinder_asym(nn.Module):
    def __init__(self,
                 cylin_model,
                 segmentator_spconv,
                 sparse_shape,
                 ):
        super().__init__()
        self.name = "cylinder_asym"

        self.cylinder_3d_generator = cylin_model

        self.cylinder_3d_spconv_seg = segmentator_spconv

        self.sparse_shape = sparse_shape

    def forward(self, train_pt_fea_ten, train_vox_ten, batch_size):
        """train_pt_fea_ten是点特征，train_vox_ten是点对应的体素索引"""
        coords, features_3d = self.cylinder_3d_generator(train_pt_fea_ten, train_vox_ten)

        spatial_features = self.cylinder_3d_spconv_seg(features_3d, coords, batch_size)

        return spatial_features
