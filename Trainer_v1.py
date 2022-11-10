import os
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
        self.model = builder.get_model()
        self.loss, self.loss_weight = builder.get_loss()

    def training_step(self, batch_data, batch_idx):
        labels = batch_data["Label"]
        vox_labels = labels[batch_data["Voxel"]["p2v"]]
        logits, vox_logits_st = self.model(batch_data, labels)
        if isinstance(self.loss, list):
            loss = 0
            for l, w in zip(self.loss, self.loss_weight):
                loss += l(logits, labels) * w
                loss += l(vox_logits_st.F, vox_labels) * w
        else:
            loss = self.loss(logits, labels) + self.loss(vox_logits_st.F, vox_labels)
        return loss

    def configure_optimizers(self):
        optimizer = self.builder.get_optimizer()
        return optimizer

    def validation_step(self, batch_data, batch_idx):
        # this is the validation loop
        labels = batch_data["Label"]
        vox_labels = labels[batch_data["Voxel"]["p2v"]]
        logits, vox_logits_st = self.model(batch_data, labels)
        if isinstance(self.loss, list):
            val_loss = 0
            for l, w in zip(self.loss, self.loss_weight):
                val_loss += l(logits, labels) * w
                val_loss += l(vox_logits_st.F, vox_labels) * w
        else:
            val_loss = self.loss(logits, labels) + self.loss(vox_logits_st.F, vox_labels)

        miou, iou_list, _ = mIoU(np.argmax(logits.numpy(), axis=1), labels,
                                 class_num=self.builder.config["model"]["num_classes"])
        self.log("val_loss", val_loss)
        self.log("val_mIoU", miou)
        for i, iou in enumerate(iou_list):
            word = label2word(i, self.builder.kitti_yaml["labels"], self.builder.kitti_yaml["learning_map_inv"])
            self.log(f"val_iou_{word}", iou)


if __name__ == "__main__":
    builder = Builder("./config/test/total.yaml")
    wandb_logger = WandbLogger(project="test_cenet")
    auto_model = AutoModel(builder)
    wandb_logger.watch(auto_model)
    train_loader, *_ = builder.get_dataloader()
    trainer = pl.Trainer(
        max_epochs=40,
        devices=2, accelerator="gpu", strategy="ddp",
        default_root_dir="./test_cenet"
    )
    trainer.fit(model=auto_model, train_dataloaders=train_loader)
