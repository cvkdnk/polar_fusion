import os, yaml
import argparse
import shutil
import logging
from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from model import ModelInterface
from utils.builders import Builder


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("work_dir", type=str, help="work directory, e.g. ./experiments/spconv")
    parser.add_argument("exp_name", type=str, help="experiment name, e.g. test")
    parser.add_argument("--config", type=str, help="if not set, use config.yaml in work_dir")
    parser.add_argument("--checkpoint", type=str, help="checkpoint path")
    parser.add_argument("--project_name", type=str, default="polar_fusion", help="wandb project name")
    parser.add_argument("--debug", action="store_true", help="debug mode")

    return parser.parse_args()


class Trainer:
    def __init__(self, args, exp_dir):
        self.train_config = yaml.safe_load(open("./config/train.yaml", 'r'))
        self.args = args
        # process exp environment
        self.config_path = args.config if args.config else os.path.join(args.work_dir, "config.yaml")
        self.exp_dir = exp_dir
        os.makedirs(self.exp_dir, exist_ok=True)
        shutil.copy(self.config_path, self.exp_dir + "/config.yaml")
        if args.debug:
            wandb_logger = None
        else:
            wandb_logger = WandbLogger(
                save_dir=self.exp_dir, project=args.project_name, name=args.exp_name, offline=args.debug
            )
        checkpoint_callback = ModelCheckpoint(monitor="val/mIoU",
                                              save_last=True, save_top_k=3, mode="max")
        self.profiler = PyTorchProfiler(dirpath=self.exp_dir, filename="profile", export_to_chrome=True)
        self.builder = Builder(self.config_path, self.exp_dir, self.train_config["device"])
        self.builder.debug = args.debug
        self.trainer = pl.Trainer(
            accelerator=self.train_config["device"],
            devices=1,
            # strategy="ddp",
            # sync_batchnorm=True,
            # precision=self.train_config["precision"],
            logger=wandb_logger,
            callbacks=[checkpoint_callback],
            max_epochs=self.train_config["epochs"],
            default_root_dir=self.exp_dir,
            # resume_from_checkpoint=args.checkpoint,
            # profiler=self.profiler
        )
        ############################
        # init model
        self.pl_model = ModelInterface.get_model(self.builder, args.checkpoint)
        for param in self.pl_model.parameters():
            param = param.float()
        if not args.debug:
            wandb_logger.watch(self.pl_model)
        ############################
        # init dataloader
        self.train_loader, self.val_loader, self.test_loader = self.builder.get_dataloader()
        self.upload_config(wandb_logger)

    def fit(self):
        self.trainer.fit(
            model=self.pl_model,
            train_dataloaders=self.train_loader,
            val_dataloaders=self.val_loader
        )

    def upload_config(self, wandb_logger):
        config = self.builder.config.copy()
        config["exp_dir"] = self.exp_dir
        config.update(self.train_config)
        if not args.debug:
            wandb_logger.experiment.config.update(config)


if __name__ == "__main__":
    args = get_parser()
    exp_dir = os.path.join(
        args.work_dir,
        args.exp_name+"_"+datetime.now().strftime("%m%d_%H%M")
    )
    try:
        trainer = Trainer(args, exp_dir)
    except Exception as e:
        shutil.rmtree(exp_dir)
        raise e
    trainer.fit()
    # print(trainer.profiler.describe())
