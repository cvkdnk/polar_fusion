import pytorch_lightning as pl
import torch
import time

from utils.pf_base_class import InterfaceBase
from utils.builders import Builder
from utils.data_utils import batch_upsampling
from model.pl_base_model import ModuleBaseClass


class ModelInterface(InterfaceBase):
    REGISTER = {}

    @classmethod
    def get_model(cls, builder, ckpt):
        name = builder.config["Model"]
        if isinstance(name, str):
            if ckpt is not None:
                return cls.REGISTER[name].load_from_checkpoint(ckpt, builder=builder)
            return cls.REGISTER[name](builder)
        class_type = cls.__name__.replace("Interface", "")
        raise TypeError(f"{class_type} in base.yaml should be str")


# class ModuleBaseClass(nn.Module, PFBaseClass):
#     NEED_TYPE = None



# class Cylinder3DTS(ModuleBaseClass):
#     NEED_TYPE = "Point,Voxel"
#
#     @classmethod
#     def gen_config_template(cls):
#         config = {
#             "point_wise_refinement": True,
#             "in_fea_dim": 9,
#             "mlp_channels": [64, 128, 256, 64],
#             "out_fea_dim": 256,
#             "fea_compre": None,
#             "output_shape": [480, 360, 32],
#             "num_input_features": 64,
#             "num_classes": 20,
#             "init_size": 32,
#         }
#         return config
#
#     def __init__(self, builder):
#         super().__init__(builder)
#         self.name = "Cylinder3D"
#         model_config = builder.config["model"]
#         self.point_wise_refinement = model_config["point_wise_refinement"]
#         from model.cy3d import CylinderPointMLP, PointWiseRefinement, Asymm_3d_spconv
#         self.cylinder_3d_generator = CylinderPointMLP(
#             in_fea_dim=model_config["in_fea_dim"],
#             mlp_channels=model_config["mlp_channels"],
#             out_pt_fea_dim=model_config["out_fea_dim"],
#             fea_compre=model_config["fea_compre"]
#         )
#         self.cylinder_3d_spconv_seg = Asymm_3d_spconv(
#             num_input_feats=model_config["num_input_features"],
#             init_size=model_config["init_size"],
#             num_classes=model_config["num_classes"]
#         )
#         self.cylinder_3d_generate_logits = PointWiseRefinement(
#             init_size=model_config["init_size"],
#             mlp_channels=model_config["mlp_channels"],
#             num_classes=model_config["num_classes"]
#         )
#
#     def forward(self, batch, *args):
#         data = batch["Voxel"]
#         point_feats_st = data["point_feats_st"]
#         voxel_feats_st = data["voxel_feats_st"]
#         upsample_inds = data["upsampling_index"]
#
#         # 生成点特征
#         point_feats_st, voxel_feats_st, skip_pt_feats = self.cylinder_3d_generator(
#             point_feats_st, voxel_feats_st, upsample_inds)
#
#         # 通过稀疏卷积对体素特征处理并给出预测结果
#         voxel_logits_st, voxel_feats_st = self.cylinder_3d_spconv_seg(voxel_feats_st)
#
#         # 处理体素预测结果
#         out_point_feats = batch_upsampling(voxel_feats_st.F, upsample_inds)
#         if self.point_wise_refinement:
#             logits = self.cylinder_3d_generate_logits(out_point_feats, skip_pt_feats)
#         else:
#             logits = batch_upsampling(voxel_logits_st.F, upsample_inds)
#
#         return logits, voxel_logits_st


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


@ModelInterface.register
class Cylinder3DSPConv(ModuleBaseClass):
    NEED_TYPE = "Point,Voxel"

    @classmethod
    def gen_config_template(cls):
        config = {
            "in_fea_dim": 9,
            # "mlp_channels": [64, 128, 256, 64],
            "out_fea_dim": 256,
            "fea_compre": 64,
            "output_shape": [480, 360, 32],
            "num_input_features": 64,
            "num_classes": 20,
            "init_size": 32,
        }
        return config

    def __init__(self, builder: Builder):
        super(Cylinder3DSPConv, self).__init__(builder)
        config = builder.config["model"]
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
            config["init_size"]
        )

    def forward(self, batch, *args):
        point_feats = batch["Voxel"]["pt_feats"]
        pt_vox_coords = batch["Voxel"]["pt_vox_coords"]
        coords, features_3d = self.pt_fea_gen(point_feats, pt_vox_coords)
        batch_size = len(batch["PointsNum"])
        logits = self.cy3d_spconv_seg(features_3d, coords, batch_size)
        y = logits.dense()
        return {"sparse": logits, "dense": y}

    def _eval(self, logits_dict, batch_data, train=True):
        logits = logits_dict["dense"]
        labels = batch_data["Voxel"]["dense_vox_labels"]
        preds = torch.argmax(logits, dim=1)
        return self._eval_func(preds, labels, train)


@ModelInterface.register
class PolarFusionBaseline(Cylinder3DSPConv):
    def __init__(self, builder):
        super(PolarFusionBaseline, self).__init__(builder)
        from model.cy3d_spconv import cylinder_fea, Asymm_3d_spconv
        config = builder.config["model"]
        self.pt_fea_gen = cylinder_fea(  # 生成点特征的模型
            config["in_fea_dim"],
            config["out_fea_dim"],
            config["fea_compre"],
            return_inverse=True
        )

    def forward(self, batch, *args):
        point_feats = batch["Voxel"]["pt_feats"]
        pt_vox_coords = batch["Voxel"]["pt_vox_coords"]
        coords, features_3d = self.pt_fea_gen(point_feats, pt_vox_coords)
        batch_size = len(batch["PointsNum"])
        logits = self.cy3d_spconv_seg(features_3d, coords, batch_size)
        y = logits.dense()
        return {"sparse": logits, "dense": y}


if __name__ == "__main__":
    import time
    import torch
    import numpy as np

    # begin_time = time.time()
    # pt_features = torch.tensor(np.random.random((100000, 4))) * 100 - 50
    # from dataloader.data_pipeline import DataPipelineInterface
    # data = {"Point": pt_features}
    # cylize = DataPipelineInterface.gen_default("Cylindrical")
    # rangeproj = DataPipelineInterface.gen_default("RangeProject")
    # data.update(cylize(data["Point"], None))
    # data.update(rangeproj(data["Point"], None))
    #
    # model = Cylinder3DTS(**Cylinder3DTS.gen_config_template())
    #
    # logits = model(data)


    # for model in ModelInterface.REGISTER.keys():
    #     print(f"Testing {model}: ")
    #     begin_time = time.time()

