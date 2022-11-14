import os
import torch
from torch import optim, nn, utils, Tensor
import numpy as np
import pytorch_lightning as pl
from utils.builders import Builder
from utils.evaluate import mIoU
from utils.data_utils import label2word
from pytorch_lightning.loggers import WandbLogger




# define the LightningModule
class AutoModel(pl.LightningModule):
    def __init__(self, builder: Builder):
        super().__init__()
        self.builder = builder
        self.model = builder.model
        self.loss, self.loss_weight = builder.loss, builder.loss_weight

    def training_step(self, batch_data, batch_idx):
        labels = batch_data["Label"]
        vox_labels = labels[batch_data["Voxel"]["p2v"]]
        logits, vox_logits_st = self.model(batch_data)
        if isinstance(self.loss, list):
            loss = 0
            for l, w in zip(self.loss, self.loss_weight):
                loss += l(logits, labels) * w
                loss += l(vox_logits_st.F, vox_labels) * w
        else:
            loss = self.loss(logits, labels) + self.loss(vox_logits_st.F, vox_labels)
        return loss

    def configure_optimizers(self):
        return self.builder.optimizer

    def validation_step(self, batch_data, batch_idx):
        # this is the validation loop
        labels = batch_data["Label"]
        vox_labels = labels[batch_data["Voxel"]["p2v"]]
        logits, vox_logits_st = self.model(batch_data)
        if isinstance(self.loss, list):
            val_loss = 0
            for l, w in zip(self.loss, self.loss_weight):
                val_loss += l(logits, labels) * w
                val_loss += l(vox_logits_st.F, vox_labels) * w
        else:
            val_loss = self.loss(logits, labels) + self.loss(vox_logits_st.F, vox_labels)

        miou, iou_list, _ = mIoU(np.argmax(logits.cpu().numpy(), axis=1), labels.cpu().numpy(),
                                 class_num=self.builder.config["model"]["num_classes"])
        self.log("val_loss", val_loss)
        self.log("val_mIoU", miou)
        for i, iou in enumerate(iou_list):
            word = label2word(i, self.builder.kitti_yaml["labels"], self.builder.kitti_yaml["learning_map_inv"])
            self.log(f"val_iou_{word}", iou)


if __name__ == "__main__":
    builder = Builder("./experiments/cy3d")
    wandb_logger = WandbLogger(project="cy3d")
    auto_model = AutoModel(builder)
    wandb_logger.watch(auto_model)
    train_loader, val_loader = builder.train_loader, builder.val_loader
    trainer = pl.Trainer(
        max_epochs=40,
        accelerator="gpu",
        default_root_dir="./experiments/cy3d"
    )
    trainer.fit(model=auto_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
