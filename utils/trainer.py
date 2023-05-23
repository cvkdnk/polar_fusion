import os, time
import numpy as np
import torch
import pytorch_lightning as pl
from torchmetrics import JaccardIndex
import wandb
from typing import Any

from utils.builders import Builder
from utils.pf_base_class import PFBaseClass
from utils.evaluate import mIoU
from utils.data_utils import label2word, SemKittiUtils
from utils.train_utils import AverageMeter


class BaseTrainer(pl.LightningModule):
    NEED_TYPE = None





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
            "compress_ratio": [60, 45, 32],  # 从体素空间使用GS降采样的倍率，例如[20, 20, 4]表示在x,y,z方向上分别降采样20倍，20倍，4倍
        }
        return config

    def __init__(self, builder: Builder):
        super(Cy3D_SegMap_GS, self).__init__(builder)
        from torchmetrics import JaccardIndex
        # self.jaccard_raw = JaccardIndex(**builder.config["metric"])
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
        _, feats = self.cy3d_spconv_seg(features_3d, coords, batch_size)
        # raw_logits = spconv.SparseConvTensor(
        #     logits.features.clone(), logits.indices, logits.spatial_shape, batch_size
        # )
        logits, feats = self.seg_map(feats, batch_size)
        # return {"raw_dense": raw_logits.dense(), "dense": logits.dense()}
        return {"dense": logits.dense()}

    def _eval(self, logits_dict, batch_data, train=True):
        labels = batch_data["Voxel"]["dense_vox_labels"]
        # print("logits_dict[dense].shape", logits_dict["dense"].shape)  # (B, 20, 480, 360, 32)
        preds = torch.argmax(logits_dict["dense"], dim=1)
        result = self._eval_func(preds, labels, train)
        # if not train:
        #     raw_preds = torch.argmax(logits_dict["raw_dense"], dim=1)
        #     if len(preds.shape) != 1:
        #         raw_preds, labels = preds.view(-1), labels.view(-1)
        #     self.jaccard_raw(raw_preds, labels)
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
