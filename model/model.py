import pytorch_lightning as pl
import torch
import time

import spconv.pytorch as spconv

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
class Cy3D_SegMap_GS(ModuleBaseClass):
    NEED_TYPE = "Point,Voxel"

    @classmethod
    def gen_config_template(cls):
        config = {
            "in_fea_dim": 9,
            "mlp_channels": [64, 128, 256, 64],
            "out_fea_dim": 256,
            "fea_compre": 64,
            "output_shape": [480, 360, 32],
            "num_input_features": 64,
            "num_classes": 20,
            "init_size": 32,
            "compress_ratio": [20, 20, 4],  # 从体素空间使用GS降采样的倍率，例如[20, 20, 4]表示在x,y,z方向上分别降采样20倍，20倍，4倍
        }
        return config

    def __init__(self, builder: Builder):
        super(Cy3D_SegMap_GS, self).__init__(builder)
        from torchmetrics import JaccardIndex
        self.jaccard_raw = JaccardIndex(**builder.config["metric"])
        config = builder.config["model"]
        from model.cy3d_spconv import cylinder_fea
        from model.segmap import SegmentorMap, Asymm_3d
        self.pt_fea_gen = cylinder_fea(  # 生成点特征的模型
            config["in_fea_dim"],
            config["out_fea_dim"],
            config["fea_compre"]
        )
        self.cy3d_spconv_seg = Asymm_3d(  # 通过稀疏卷积对体素特征处理并给出预测结果
            config["output_shape"],
            config["num_input_features"],
            config["num_classes"],
            config["init_size"]
        )
        self.seg_map = SegmentorMap(
            config["compress_ratio"],
            config["init_size"],
            config["num_classes"]
        )

    def forward(self, batch, *args):
        point_feats = batch["Voxel"]["pt_feats"]
        pt_vox_coords = batch["Voxel"]["pt_vox_coords"]
        coords, features_3d = self.pt_fea_gen(point_feats, pt_vox_coords)
        batch_size = len(batch["PointsNum"])
        logits, feats = self.cy3d_spconv_seg(features_3d, coords, batch_size)
        raw_logits = spconv.SparseConvTensor(
            logits.features.clone(), logits.indices, logits.spatial_shape, batch_size
        )
        logits, feats = self.seg_map(logits, feats, batch_size)
        return {"raw_dense": raw_logits.dense(), "dense": logits.dense()}

    def _eval(self, logits_dict, batch_data, train=True):
        labels = batch_data["Voxel"]["dense_vox_labels"]
        preds = torch.argmax(logits_dict["dense"], dim=1)
        result = self._eval_func(preds, labels, train)
        if not train:
            raw_preds = torch.argmax(logits_dict["raw_dense"], dim=1)
            if len(preds.shape) != 1:
                raw_preds, labels = preds.view(-1), labels.view(-1)
            self.jaccard_raw(raw_preds, labels)
        return result

    def validation_step(self, batch_data, batch_idx):
        logits_dict = self(batch_data)
        loss = self.loss(logits_dict, batch_data)
        results = self._eval(logits_dict, batch_data, train=False)
        self._log_results_step(loss, results, train=False)
        if batch_idx == 0:
            vox_pred = torch.argmax(logits_dict["dense"][0], dim=0).cpu().numpy()
            x, y, z = batch_data["Voxel"]["pt_vox_coords"][0].cpu().numpy().T
            pt_pred = vox_pred[x, y, z]
            self._plt_sample(seq_frame=batch_data["SeqFrame"][0],
                                points=batch_data["Point"][0][..., :3].cpu().numpy(),
                                labels=batch_data["Label"][0].cpu().numpy(),
                                preds_pt=pt_pred,
                                dense_labels=batch_data["Voxel"]["dense_vox_labels"][0].cpu().numpy(),
                                dense_preds=vox_pred)
            vox_pred = torch.argmax(logits_dict["raw_dense"][0], dim=0).cpu().numpy()
            pt_pred = vox_pred[x, y, z]
            self._plt_sample(seq_frame="raw"+batch_data["SeqFrame"][0],
                                points=batch_data["Point"][0][..., :3].cpu().numpy(),
                                labels=batch_data["Label"][0].cpu().numpy(),
                                preds_pt=pt_pred,
                                dense_labels=batch_data["Voxel"]["dense_vox_labels"][0].cpu().numpy(),
                                dense_preds=vox_pred)

    def on_validation_epoch_end(self):
        # compute iou
        iou_per_class = self.jaccard.compute()
        raw_iou_per_class = self.jaccard_raw.compute()
        iou_mask = torch.ones_like(iou_per_class, dtype=torch.bool)
        iou_mask[self.builder.config["metric"]["ignore_index"]] = False
        # log iou and miou
        print("\n|==============IoU per class==============|")
        print("| Name         |    IoU |   Best |    Raw |")
        print("|-----------------------------------------|")
        for i, iou in enumerate(iou_per_class):
            if not iou_mask[i]:
                continue
            word = self.word_list[i]
            self.log(f"val_iou/{word}", iou, batch_size=self.val_bsz, on_epoch=True)
            print(f"| {word.ljust(13)}" + "| {:5.2f}% | {:5.2f}% | {:5.2f}% |".format(
                iou * 100, self.best_iou_per_class[i] * 100, raw_iou_per_class[i] * 100
            ))
        miou = torch.masked_select(iou_per_class, iou_mask).mean()
        raw_miou = torch.masked_select(raw_iou_per_class, iou_mask).mean()
        self.log("val/mIoU", miou, on_epoch=True, logger=True, batch_size=self.val_bsz)
        self.log("val/raw_mIoU", raw_miou, on_epoch=True, logger=True, batch_size=self.val_bsz)
        print("|-----------------------------------------|")
        print("| Current mIoU | {:5.2f}% | {:5.2f}% | {:5.2f}% |".format(
            miou * 100, self.best_miou * 100, raw_miou * 100
        ))
        print("|=========================================|")
        # update best iou
        if miou > self.best_miou:
            self._update_miou(miou, iou_per_class)
        # reset jaccard index
        self.jaccard.reset()
        self.jaccard_raw.reset()


# class MSegCeRes(ModuleBaseClass):
#     NEED_TYPE = "Point,Range"
#
#     @classmethod
#     def gen_config_template(cls):
#         config = {
#             "num_classes": 20,
#             "cenet_ckpt": "/home/cls2021/cvkdnk/workspace/PolarFusion/save/CENet_64x512_67_6",
#         }
#         return config
#
#     def __init__(self, builder):
#         super().__init__(builder)
#         config = builder.config["model"]
#         from model.cenet.ResNet import ResNet_34
#         self.resnet = ResNet_34(config["num_classes"], aux=False)
#         cenet_ckpt = torch.load(config["cenet_ckpt"])
#         self.resnet.load_state_dict(cenet_ckpt["state_dict"])
#
#     def forward(self, batch, *args):




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

