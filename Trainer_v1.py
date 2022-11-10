import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
from utils.builders import Builder


# define the LightningModule
class AutoModel(pl.LightningModule):
    def __init__(self, builder: Builder):
        super().__init__()
        self.builder = builder
        self.model = builder.get_model()
        self.loss, self.loss_weight = builder.get_loss()

    def training_step(self, batch_data, batch_idx):
        data = batch_data["Voxel"]
        labels = batch_data["Label"]
        vox_labels = labels[data["p2v_indices"]]
        logits, vox_logits_st = self.model(data, labels)
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


if __name__ == "__main__":
    builder = Builder("./config/test/total.yaml")
    auto_model = AutoModel(builder)
    train_loader, *_ = builder.get_dataloader()
    trainer = pl.Trainer()
    trainer.fit(model=auto_model, train_dataloaders=train_loader)
