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


class ModuleBaseClass(pl.LightningModule, PFBaseClass):
    NEED_TYPE = None

    @classmethod
    def gen_config_template(cls):
        raise NotImplementedError

    def __init__(self, builder: Builder):
        super().__init__()
        self.builder = builder
        self.exp_dir = builder.exp_dir
        self.loss = builder.loss
        self.train_bsz = builder.config["dataloader"]["train"]["batch_size"]
        self.val_bsz = builder.config["dataloader"]["val"]["batch_size"]
        self.jaccard = JaccardIndex(**builder.config["metric"])
        os.makedirs(self.exp_dir + "/val_plt", exist_ok=True)
        self.gt_record = False
        self.best_miou = 0
        self.best_iou_per_class = [0 for i in range(20)]
        self.word_list = [str(
            label2word(i, self.builder.kitti_yaml["labels"], self.builder.kitti_yaml["learning_map_inv"])
        ) for i in range(20)]  # SemanticKITTI Only

    def forward(self, inputs, *args) -> Any:
        raise NotImplementedError

    def training_step(self, batch_data, batch_idx):
        begin_time = time.time()
        logits_dict = self(batch_data)
        end1_time = time.time()
        loss = self.loss(logits_dict, batch_data)
        end2_time = time.time()
        results = self._eval(logits_dict, batch_data)
        end3_time = time.time()
        self._log_results_step(
            loss, results, train=True
        )
        self.log("time/forward", end1_time - begin_time, logger=True, on_epoch=True, batch_size=self.train_bsz)
        self.log("time/loss", end2_time - end1_time, logger=True, on_epoch=True, batch_size=self.train_bsz)
        self.log("time/eval", end3_time - end2_time, logger=True, on_epoch=True, batch_size=self.train_bsz)
        return loss

    def on_training_epoch_end(self):
        self.log("time/epoch_end", time.time(), logger=True, on_epoch=True, batch_size=self.train_bsz)

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

    def on_validation_epoch_end(self):
        # compute iou
        iou_per_class = self.jaccard.compute()
        iou_mask = torch.ones_like(iou_per_class, dtype=torch.bool)
        iou_mask[self.builder.config["metric"]["ignore_index"]] = False
        # log iou and miou
        print("\n|=========IoU per class==========|")
        print("| Name         |    IoU |   Best |")
        print("|--------------------------------|")
        for i, iou in enumerate(iou_per_class):
            if not iou_mask[i]:
                continue
            word = self.word_list[i]
            self.log(f"val_iou/{word}", iou, batch_size=self.val_bsz, on_epoch=True)
            print(f"| {word.ljust(13)}" + "| {:5.2f}% | {:5.2f}% |".format(iou * 100, self.best_iou_per_class[i] * 100))
        miou = torch.masked_select(iou_per_class, iou_mask).mean()
        self.log("val/mIoU", miou, on_epoch=True, logger=True, batch_size=self.val_bsz)
        print("|--------------------------------|")
        print("| Current mIoU | {:5.2f}% | {:5.2f}% |".format(miou * 100, self.best_miou * 100))
        print("|================================|")
        # update best iou
        if miou > self.best_miou:
            self._update_miou(miou, iou_per_class)
        # reset jaccard index
        self.jaccard.reset()

    def on_validation_epoch_start(self) -> None:
        self.jaccard.reset()

    def configure_optimizers(self) -> Any:
        return self.builder.get_optimizer(self.parameters())

    def _log_results_step(self, loss, results, train=True):
        if train:
            self.log("train/loss", loss,
                     on_step=True, on_epoch=True, logger=True, batch_size=self.train_bsz)
            self.log("train/acc", results["acc"],
                     on_step=True, prog_bar=True, logger=True, batch_size=self.train_bsz)
        else:
            self.log("val/loss", loss, batch_size=self.val_bsz, on_epoch=True)
            self.log("val/acc", results["acc"], batch_size=self.val_bsz, on_epoch=True)

    def _plt_sample(self, seq_frame, points, labels, preds_pt, dense_labels=None, dense_preds=None):
        """input torch.Tensor"""
        save_path = self.exp_dir + "/val_plt/" + seq_frame
        # np.save(save_path + f"_{self.current_epoch}_preds.npy", preds_pt)
        # 预测错误的点云可视化
        SemKittiUtils.draw_acc(points, labels, preds_pt, save_path + f"_{self.current_epoch}_err.xyz")
        # 原始标签点云和预测点云可视化
        if not self.gt_record:
            SemKittiUtils.draw_rgb_pcd(points, labels,
                                       self.builder.kitti_yaml, save_path + "_gt.xyz")
        SemKittiUtils.draw_rgb_pcd(points, preds_pt,
                                   self.builder.kitti_yaml, save_path + f"_{self.current_epoch}_pred.xyz")
        # 体素点云可视化
        if dense_labels is not None:
            assert dense_preds is not None
            bev_idx = np.argmax(dense_labels, axis=-1)
            bev_labels = np.take_along_axis(dense_labels, bev_idx[..., None], axis=-1).squeeze(-1)
            bev_preds = np.take_along_axis(dense_preds, bev_idx[..., None], axis=-1).squeeze(-1)
            img_labels = SemKittiUtils.draw_rgb_bev(bev_labels, self.builder.kitti_yaml,
                                                    save_path + f"_{self.current_epoch}_bev_gt.png")
            img_preds = SemKittiUtils.draw_rgb_bev(bev_preds, self.builder.kitti_yaml,
                                                   save_path + f"_{self.current_epoch}_bev_pred.png")
            self.logger.experiment.log({"val_bev_gt_sample": wandb.Image(img_labels)}) if not self.gt_record else None
            self.logger.experiment.log({"val_bev_pred_sample": wandb.Image(img_preds)})
        self.gt_record = True

    def _eval(self, logits_dict, batch_data, train=True):
        """用_eval_func的方法计算评估，但是_eval_func的参数需要经过解析后传入。最终用字典返回结果，目前:acc miou iou_list"""
        raise NotImplementedError

    def _eval_func(self, preds, labels, train=True):
        """计算点或稠密体素的acc、iou等评估指标，iou仅在train=False时计算，传入torch.Tensor，且各batch stack到一起"""
        if len(preds.shape) != 1:
            preds, labels = preds.view(-1), labels.view(-1)
        if not train:
            self.jaccard(preds, labels)
        acc = torch.sum(preds == labels) / (labels.shape[0] + 1e-6)
        return {"acc": acc}

    def _update_miou(self, miou, iou_per_class):
        self.best_miou = miou
        self.best_iou_per_class = iou_per_class
        table = wandb.Table(data=[iou_per_class], columns=self.word_list)
        self.logger.experiment.log({"iou_per_class": table})
