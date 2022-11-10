from torch import nn

from utils.pf_base_class import PFBaseClass, InterfaceBase
from model.cy3d import CylinderPointMLP, PointWiseRefinement, Asymm_3d_spconv



class ModelInterface(InterfaceBase):
    REGISTER = {}


class ModuleBaseClass(nn.Module):
    """same as PFBaseClass"""
    default_str = "Need To Be Completed ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    NEED_TYPE = None

    @classmethod
    def gen_config_template(cls):
        raise NotImplementedError


@ModelInterface.register
class Cylinder3D(ModuleBaseClass):
    NEED_TYPE = "Point,Voxel"

    @classmethod
    def gen_config_template(cls):
        config = {
            "point_wise_refinement": True,
            "output_shape": [480, 360, 32],
            "in_fea_dim": 9,
            "num_input_features": 16,
            "num_classes": 20,
            "use_norm": True,
            "init_size": 32,
            "out_fea_dim": 256
        }
        return config

    def __init__(self, **model_config):
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

    def forward(self, batch):
        data = batch["Voxel"]

        voxel_feats_st, skip_pt_feats = self.cylinder_3d_generator(data["point_feats_st"], data["p2v"])

        voxel_logits_st, voxel_feats_st = self.cylinder_3d_spconv_seg(voxel_feats_st)

        if self.point_wise_refinement:
            logits = self.cylinder_3d_generate_logits(voxel_feats_st.F[data["v2p"]], skip_pt_feats)
        else:
            logits = voxel_logits_st.F[data["v2p_indices"]]

        return logits, voxel_logits_st


class CENet(ModuleBaseClass):

    @classmethod
    def gen_config_template(cls):
        return {
            "num_classes": 20,
            "aux_loss": False
        }

    def __init__(self, **config):
        super(CENet, self).__init__()
        from model.cenet import HarDNet
        self.model = HarDNet(config["num_classes"], config["aux_loss"])


if __name__ == "__main__":
    import time
    import torch
    import numpy as np
    begin_time = time.time()
    pt_features = torch.tensor(np.random.random((100000, 4))) * 100 - 50
    from dataloader.data_pipeline import DataPipelineInterface
    data = {"Point": pt_features}
    cylize = DataPipelineInterface.gen_default("Cylindrical")
    rangeproj = DataPipelineInterface.gen_default("RangeProject")
    data.update(cylize(data["Point"], None))
    data.update(rangeproj(data["Point"], None))

    for model in ModelInterface.REGISTER.keys():
        print(f"Testing {model}: ")
        begin_time = time.time()

