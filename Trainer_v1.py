import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
from utils.builders import Builder


# define the LightningModule
class LitAutoEncoder(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.builder = Builder("./config/test")


    def training_step(self, batch_data, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward

        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return self.builder.optimizer


# init the autoencoder
autoencoder = LitAutoEncoder(encoder, decoder)
os.makedirs()