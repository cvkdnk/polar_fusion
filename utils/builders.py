import os

import torch
import yaml
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from dataloader import DatasetInterface, DataPipelineInterface
from loss import LossInterface
from utils.optimizer import OptimizerInterface
from model import ModelInterface
from utils.data_utils import custom_collate_fn
from utils.config_utils import load_config


class Builder:
    def __init__(self, config_path, exp_dir, device='cpu'):
        self.debug = None
        self.config = load_config(config_path)
        self.ignore = self.config["dataset"]["ignore"]
        # 生成JaccardMetric参数
        self.config["metric"] = {
            "task": "multiclass",
            "num_classes": self.config["model"]["num_classes"],
            "ignore_index": self.ignore,
            "average": "none"
        }
        self.exp_dir = exp_dir
        self.kitti_yaml = yaml.safe_load(open("./config/semantic-kitti.yaml", 'r'))
        self.loss = self.get_loss(device=device)
        self._process_config()

    def get_dataloader(self):
        """ 生成dataloader

        :return: (train_loader, val_loader, test_loader)
        """
        dataflow = DatasetInterface.get(self.config["Dataset"], self.config["dataset"])

        class DataPipeline(Dataset):
            def __init__(self, dataset, data_pipeline_list, data_pipeline_config):
                self.data_pipeline = DataPipelineInterface.get(data_pipeline_list, data_pipeline_config)
                self.dataset = dataset

            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, item):
                data = self.dataset[item]
                for dp in self.data_pipeline:
                    data.update(dp(data))
                return data

        loader_tuple = ()
        for mode in ["train", "val", "test"]:
            dataset = DataPipeline(
                dataflow[mode],
                self.config["DataPipeline"][mode],
                self.config["pipeline"][mode]
            )
            loader_tuple += (DataLoader(
                dataset,
                collate_fn=custom_collate_fn,
                **self.config["dataloader"][mode]
            ), )
        return loader_tuple

    # def get_model(self):
    #     model = ModelInterface.get(self.config["Model"], self.config["model"])
    #     return model

    def get_loss(self, device):
        loss_name = self.config["Loss"]
        if isinstance(loss_name, str):
            return LossInterface.get(loss_name, self.config["loss"], device, self.ignore)
        else:
            raise RuntimeError("loss_name should be str")

    def get_optimizer(self, params):
        optimizer = OptimizerInterface.get(self.config["Optimizer"], self.config["optimizer"], params)
        return optimizer

    def _process_config(self):
        for config_name in ["DataPipeline", "pipeline", "dataloader"]:
            for mode in ["train", "val", "test"]:
                tmp = self.config[config_name]["base"].copy()
                if mode in self.config[config_name]:
                    if isinstance(self.config[config_name][mode], dict):
                        tmp.update(self.config[config_name][mode])
                    else:
                        tmp += self.config[config_name][mode]
                    self.config[config_name][mode] = tmp
                else:
                    self.config[config_name][mode] = tmp
        train_config = yaml.safe_load(open("./config/train.yaml", 'r'))
        self.config["dataset"]["data_root"] = train_config.pop("data_root")
        self.config["train"] = train_config

