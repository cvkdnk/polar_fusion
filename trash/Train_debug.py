import os
import shutil
import time
from datetime import datetime
from typing import Any
import pandas as pd
import wandb

import torch
import numpy as np
import pytorch_lightning as pl
from utils.builders import Builder
from utils.evaluate import mIoU
from sklearn.metrics import confusion_matrix
from utils.data_utils import label2word
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint


class AutoModel(pl.LightningModule):
    def __init__(self, builder: Builder, exp_dir):
        super().__init__()
        self.builder = builder
        self.exp_dir = exp_dir
        self.model = builder.model
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.train_bsz = builder.config["dataloader"]["train"]["batch_size"]
        self.val_bsz = builder.config["dataloader"]["val"]["batch_size"]
        self.val_idx = 0

    def forward(self, inputs, *args) -> Any:
        logits, vox_logits_st = self.model(inputs)
        return logits, vox_logits_st

    def training_step(self, batch_data, batch_idx):
        # device = batch_data["Point"].get_device()
        # self.loss, _ = builder.get_loss(device)
        labels = batch_data["Label"]
        vox_labels = labels[batch_data["Voxel"]["p2v"]]
        begin_time = time.time()
        logits, vox_logits_st = self.model(batch_data)
        vox_loss = self.loss(vox_logits_st.F, vox_labels)
        self.log("train/vox_loss", vox_loss,
                 on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.train_bsz)
        vox_labels = vox_labels.cpu().numpy()
        vox_acc = len((vox_logits_st.F.cpu().numpy() == vox_labels).nonzero()) / (len(vox_labels) + 0.01)
        self.log("train/vox_acc", vox_acc,
                 on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.train_bsz)
        self.log("train/time", time.time()-begin_time,
                 on_step=True, prog_bar=True, logger=True, batch_size=self.train_bsz)
        return vox_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.02)

    def validation_step(self, batch_data, batch_idx):
        # this is the validation loop
        # device = batch_data["Point"].get_device()
        # self.loss, _ = builder.get_loss(device)
        labels = batch_data["Label"]
        vox_labels = labels[batch_data["Voxel"]["p2v"]]
        logits, vox_logits_st = self.model(batch_data)
        vox_preds = torch.argmax(vox_logits_st.F, dim=1)
        val_vox_loss = self.loss(vox_logits_st.F, vox_labels)
        confusion = confusion_matrix(vox_labels.cpu().numpy(), vox_preds.cpu().numpy(), labels=np.arange(0, 20, 1))
        self.log("val/vox_loss", val_vox_loss, batch_size=self.val_bsz)
        return torch.tensor(confusion, dtype=torch.int32)

    def validation_epoch_end(self, outputs):
        confusion = torch.sum(torch.stack(outputs), dim=0)
        gt_classes = torch.sum(confusion, dim=1)
        positive_classes = torch.sum(confusion, dim=0)
        true_positive_classes = torch.diag(confusion)
        iou = torch.zeros(20, dtype=torch.float32)
        acc = torch.zeros(20, dtype=torch.float32)
        for i in range(20):
            iou = true_positive_classes[i] / (gt_classes[i] + positive_classes[i] - true_positive_classes[i] + 0.001)
            acc = true_positive_classes[i] / (gt_classes[i] + 0.001)
        mean_iou = torch.mean(iou)
        mean_acc = torch.mean(acc)
        all_acc = torch.sum(true_positive_classes) / (torch.sum(gt_classes) + 0.001)
        self.log("val/mIoU", mean_iou, batch_size=self.val_bsz, on_epoch=True)
        self.log("val/mAcc", mean_acc, batch_size=self.val_bsz, on_epoch=True)
        self.log("val/aAcc", all_acc, batch_size=self.val_bsz, on_epoch=True)


if __name__ == "__main__":
    work_dir = "../experiments/cylinder3d"
    builder = Builder(work_dir, "cuda:0")
    experiment_dir = os.path.join(work_dir, datetime.now().strftime("%m-%d_%H-%M"))
    os.makedirs(experiment_dir, exist_ok=True)
    shutil.copy(work_dir + "/config.yaml", experiment_dir + "/config.yaml")
    wandb_logger = WandbLogger(save_dir=experiment_dir, project="cy3d", name="Testin3g")
    checkpoint_callback = ModelCheckpoint(monitor="val/mIoU",
                                          save_last=True, save_top_k=3, mode="max")
    auto_model = AutoModel(builder, experiment_dir)
    wandb_logger.watch(auto_model, log="all", log_freq=500)
    train_loader, val_loader = builder.train_loader, builder.val_loader
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        # strategy="ddp",
        # sync_batchnorm=True,
        logger=wandb_logger,
        max_epochs=60,
        default_root_dir=experiment_dir,
        check_val_every_n_epoch=1,
        callbacks=checkpoint_callback,
        # move_metrics_to_cpu=True
    )
    trainer.fit(model=auto_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("Training complete!")

