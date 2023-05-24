import torch
from torch import nn
import spconv.pytorch as spconv

from utils.pf_base_class import InterfaceBase


class ModelInterface(InterfaceBase):
    REGISTER = {}

    @classmethod
    def get_model(cls, config, ckpt=None):
        name = config["Model"]
        if isinstance(name, str):
            if ckpt is not None:
                return cls.REGISTER[name].load_from_checkpoint(ckpt)
            return cls.REGISTER[name](config)
        class_type = cls.__name__.replace("Interface", "")
        raise TypeError(f"{class_type} in base.yaml should be str")


@ModelInterface.register
class Cylinder3D(nn.Module):
    @classmethod
    def gen_config_template(cls):
        config = {
            "in_fea_dim": 9,
            "out_fea_dim": 256,
            "fea_compre": 64,
            "output_shape": [480, 360, 32],
            "num_input_features": 64,
            "num_classes": 20,
            "init_size": 32,
        }
        return config

    def __init__(self, config):
        super(Cylinder3D, self).__init__()
        from model.cy3d_spconv import cylinder_fea, Asymm_3d_spconv
        self.pt_fea_gen = cylinder_fea(  # 生成点特征的模型
            config["in_fea_dim"],
            config["out_fea_dim"],
            config["fea_compre"]
        )
        self.cy3d_spconv_seg = Asymm_3d_spconv(  # 通过稀疏卷积对体素特征处理并给出预测结果
            config["output_shape"],
            config["num_input_features"],
            config["num_classes"],
            config["init_size"],
            return_logits=True
        )

    def forward(self, batch, *args):
        point_feats = batch["Voxel"]["pt_feats"]
        pt_vox_coords = batch["Voxel"]["pt_vox_coords"]
        coords, features_3d = self.pt_fea_gen(point_feats, pt_vox_coords)
        batch_size = len(batch["PointsNum"])
        logits = self.cy3d_spconv_seg(features_3d, coords, batch_size)
        return {"dense": logits}


if __name__ == "__main__":
    import time
    import torch
    import numpy as np

