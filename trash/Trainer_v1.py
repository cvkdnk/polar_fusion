import os
import shutil
import logging
from datetime import datetime
from typing import Any
import argparse

import torch
import numpy as np
import pytorch_lightning as pl
import torchsparse
from utils.builders import Builder
from utils.evaluate import mIoU
from utils import data_utils
from utils.data_utils import label2word, batch_upsampling
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint


# define the LightningModule
class AutoModel(pl.LightningModule):
    def __init__(self, builder: Builder, exp_dir):
        super().__init__()
        self.builder = builder
        self.exp_dir = exp_dir
        self.model = builder.model
        self.loss, self.loss_weight = builder.loss, builder.loss_weight
        self.train_bsz = builder.config["dataloader"]["train"]["batch_size"]
        self.val_bsz = builder.config["dataloader"]["val"]["batch_size"]
        self.val_idx = 0
        os.makedirs(exp_dir+"/val_plt", exist_ok=True)

    def forward(self, inputs, *args) -> Any:
        logits = self.model(inputs)
        return logits

    def training_step(self, batch_data, batch_idx):
        self.train()
        # device = batch_data["Point"].get_device()
        # self.loss, _ = builder.get_loss(device)
        labels = batch_data["Label"]
        vox_labels = batch_upsampling(labels, batch_data["Voxel"]["sampling_index"])
        logits_dict = self.model(batch_data)
        if isinstance(self.loss, list):
            loss = 0
            for l, w in zip(self.loss, self.loss_weight):
                loss += l(logits, labels) * w
                # loss += l(vox_logits_st.F, vox_labels) * w
        else:
            # loss = self.loss(logits, labels) + self.loss(vox_logits_st.F, vox_labels)
            loss = self.loss(logits, labels)
        self.log("train/loss", loss,
                 on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.train_bsz)
        acc = torch.sum(torch.argmax(logits, dim=1) == labels).float() / labels.shape[0]
        self.log("train/acc", acc,
                 on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.train_bsz)
        return loss

    def configure_optimizers(self):
        return self.builder.optimizer

    def validation_step(self, batch_data, batch_idx):
        # this is the validation loop
        self.train()
        labels = batch_data["Label"]
        vox_labels = batch_upsampling(labels, batch_data["Voxel"]["sampling_index"])
        logits, vox_logits_st = self.model(batch_data)
        if isinstance(self.loss, list):
            val_loss = 0
            for l, w in zip(self.loss, self.loss_weight):
                val_loss += l(logits, labels) * w
                # val_loss += l(vox_logits_st.F, vox_labels) * w
        else:
            val_loss = self.loss(logits, labels)  # + self.loss(vox_logits_st.F, vox_labels)

        miou, iou_list, _ = mIoU(np.argmax(logits.cpu().numpy(), axis=1), labels.cpu().numpy(),
                                 class_num=self.builder.config["model"]["num_classes"])
        acc = torch.sum(torch.argmax(logits, dim=1) == labels).float() / labels.shape[0]
        self.log("val/loss", val_loss, batch_size=self.val_bsz, on_epoch=True)
        self.log("val/mIoU", miou, batch_size=self.val_bsz, on_epoch=True)
        self.log("val/acc", acc, batch_size=self.val_bsz, on_epoch=True)
        iou_dict = {}
        for i, iou in enumerate(iou_list):
            word = label2word(i, self.builder.kitti_yaml["labels"], self.builder.kitti_yaml["learning_map_inv"])
            self.log(f"val/iou_{word}", iou, batch_size=self.val_bsz, on_epoch=True)
            iou_dict[str(word)] = iou
        # iou_df = pd.DataFrame([iou_dict])
        # iou_tbl = wandb.Table(data=iou_df)
        # self.log("val/iou_table", iou_tbl)

        # Debug绘图
        if batch_idx == 0:
            pts0_num = batch_data["PointsNum"][0]
            points0 = batch_data["Point"][:pts0_num, :3]
            labels0 = batch_data["Label"][:pts0_num]
            preds = torch.argmax(logits, dim=1)[:pts0_num]
            # 画错误预测的点云，错误点为红色
            pred_err = torch.zeros((pts0_num, 1), dtype=torch.float32)
            pred_err[preds!=labels0] = 255
            pc_plt = torch.cat((points0.cpu(), pred_err), dim=1).numpy()
            seq_frame = batch_data["SeqFrame"][0]
            save_path = self.exp_dir+"/val_plt/"+seq_frame
            np.save(save_path+f"_{self.val_idx}_pt.npy", points0.cpu().numpy())
            np.save(save_path+f"_{self.val_idx}_gt.npy", labels0.cpu().numpy())
            np.save(save_path+f"_{self.val_idx}_pred.npy", preds.cpu().numpy())
            np.save(save_path+f"_{self.val_idx}_logits.npy", logits.cpu().numpy()[:pts0_num])
            np.savetxt(save_path+f"_{self.val_idx}.xyz", pc_plt, fmt="%f", delimiter=",")
            # 画出原始点云和预测点云
            data_utils.SemKittiUtils.draw_rgb_pcd(points0.cpu().numpy(), labels0.cpu().numpy(),
                                                  self.builder.kitti_yaml, save_path+f"_{self.val_idx}_gt.xyz")
            data_utils.SemKittiUtils.draw_rgb_pcd(points0.cpu().numpy(), preds.cpu().numpy(),
                                                  self.builder.kitti_yaml, save_path+f"_{self.val_idx}_pred.xyz")
            # 画出体素化后的点云
            vox_coords = vox_logits_st.C[vox_logits_st.C[..., 3] == 0][..., :3]
            vox_preds = vox_logits_st.F[vox_logits_st.C[..., 3] == 0]
            vox_labels0 = vox_labels[vox_logits_st.C[..., 3] == 0]
            data_utils.SemKittiUtils.draw_rgb_pcd(vox_coords.cpu().numpy(), vox_labels0.cpu().numpy(),
                                                  self.builder.kitti_yaml, save_path+f"_{self.val_idx}_gt_vox.xyz")
            data_utils.SemKittiUtils.draw_rgb_pcd(vox_coords.cpu().numpy(), vox_preds.cpu().numpy(),
                                                    self.builder.kitti_yaml, save_path+f"_{self.val_idx}_pred_vox.xyz")

    def validation_epoch_end(self, outputs):
        self.val_idx += 1
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="./experiments/cylinder3d")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--exp_name", type=str, default="Debug")
    parser.add_argument("--checkpoint", type=str, default="")
    exp_name = parser.parse_args().exp_name
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    work_dir = parser.parse_args().work_dir
    if parser.parse_args().config != "":
        config_path = parser.parse_args().config
    else:
        config_path = work_dir + "/config.yaml"
    builder = Builder(config_path, "cuda:0")
    experiment_dir = os.path.join(work_dir, exp_name+"_"+datetime.now().strftime("%m-%d_%H-%M"))
    os.makedirs(experiment_dir, exist_ok=True)
    shutil.copy(work_dir + "/config.yaml", experiment_dir + "/config.yaml")
    wandb_logger = WandbLogger(save_dir=experiment_dir, project="cy3d", name=exp_name)
    checkpoint_callback = ModelCheckpoint(monitor="val/mIoU",
                                          save_last=True, save_top_k=3, mode="max")
    if parser.parse_args().checkpoint != "":
        checkpoint = parser.parse_args().checkpoint
        auto_model = AutoModel.load_from_checkpoint(checkpoint, builder=builder, exp_dir=experiment_dir)
    else:
        auto_model = AutoModel(builder, experiment_dir)
    wandb_logger.watch(auto_model, log="all", log_freq=500)
    train_loader, val_loader = builder.train_loader, builder.val_loader
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        # strategy="ddp",
        # sync_batchnorm=True,
        logger=wandb_logger,
        max_epochs=40,
        default_root_dir=experiment_dir,
        check_val_every_n_epoch=1,
        # val_check_interval=600,
        callbacks=checkpoint_callback,
        # move_metrics_to_cpu=True
    )
    trainer.fit(model=auto_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("Training complete!")
