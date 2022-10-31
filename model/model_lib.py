from torch import nn

from utils.pf_base_class import PFBaseClass
from model.segmentator_3d_asymm_spconv import Asymm_3d_spconv
from model.cylinder_fea_generator import cylinder_fea


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
            "name": cls.__name__,
            "output_shape": [480, 360, 32],
            "feature_dim": 9,
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
        self.cylinder_3d_generator = cylinder_fea(
            grid_size=model_config["output_shape"],
            fea_dim=model_config["fea_dim,"],
            out_pt_fea_dim=model_config["out_fea_dim"],
            fea_compre=model_config["num_input_features"]
        )
        self.cylinder_3d_spconv_seg = Asymm_3d_spconv(
            output_shape=model_config["output_shape"],
            use_norm=model_config["use_norm"],
            num_input_features=model_config["num_input_features"],
            init_size=model_config["init_size"],
            nclasses=model_config["num_classes"]
        )

        self.sparse_shape = model_config["output_shape"]

    def forward(self, data):

        coords, features_3d = self.cylinder_3d_generator(train_pt_fea_ten, train_vox_ten)

        spatial_features = self.cylinder_3d_spconv_seg(features_3d, coords, batch_size)

        return spatial_features
