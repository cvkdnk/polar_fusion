from torch import nn

from utils.pf_base_class import PFBaseClass
from model.cylinder3d_network import CylinderPointMLP, PointWiseRefinement
from model.segmentator_3d_asymm_torchsparse import Asymm_3d_spconv


class ModelLibrary(PFBaseClass):
    MODEL = {}

    @classmethod
    def gen_config_template(cls, name=None):
        assert name in cls.MODEL.keys(), f"Model {name} not found in {cls.MODEL.keys()}"

    @classmethod
    def get_model(cls, name, config):
        return cls.MODEL[name](config)

    @staticmethod
    def register(model_class):
        ModelLibrary.MODEL[model_class.__name__] = model_class
        return model_class


class ModuleBaseClass(nn.Module):
    """same as PFBaseClass"""
    default_str = "Need To Be Completed ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

    @classmethod
    def gen_config_template(cls):
        raise NotImplementedError


@ModelLibrary.register
class Cylinder3D(ModuleBaseClass):
    @classmethod
    def gen_config_template(cls):
        config = {
            "point_wise_refinement": True
            "output_shape": [480, 360, 32],
            "in_fea_dim": 9,
            "num_input_features": 16,
            "num_classes": 20,
            "use_norm": True,
            "init_size": 32,
            "out_fea_dim": 256
        }
        return config

    def __init__(self, model_config):
        super().__init__()
        self.name = "Cylinder3D"
        self.point_wise_refinement = model_config["point_wise_refinement"]
        self.cylinder_3d_generator = CylinderPointMLP(
            in_fea_dim=model_config["in_fea_dim"],
            mlp_channels=model_config["mlp_channels"],
            out_pt_fea_dim=model_config["out_fea_dim"]
        )
        self.cylinder_3d_spconv_seg = Asymm_3d_spconv(
            num_input_feats=model_config["num_input_features"],
            init_size=model_config["init_size"],
            num_classes=model_config["num_classes"]
        )
        self.cylinder_3d_generate_logits = PointWiseRefinement(
            init_size=model_config["init_size"],
            mlp_channels=model_config["mlp_channels"],
            num_classes=model_config["num_classes"]
        )

    def forward(self, data):
        voxel_feats_st, skip_pt_feats = self.cylinder_3d_generator(data["point_feats_st"], data["p2v_indices"])

        voxel_logits_st, voxel_feats_st = self.cylinder_3d_spconv_seg(voxel_feats_st)

        if self.point_wise_refinement:
            logits = self.cylinder_3d_generate_logits(voxel_feats_st.F[data["v2p_indices"]], skip_pt_feats)
        else:
            logits = voxel_logits_st.F[data["v2p_indices"]]

        return logits, voxel_logits_st
