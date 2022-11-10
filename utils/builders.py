from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from dataloader import DatasetInterface, DataPipelineInterface
from loss import LossInterface
from utils.optimizer import OptimizerInterface
from model import ModelInterface
from utils.data_utils import custom_collate_fn
from process_config import load_config


class Builder:
    def __init__(self, config_path):
        self.config = load_config(config_path)

    def build(self):
        train_loader, val_loader, test_loader = self.get_dataloader()
        model = self.get_model()
        loss = self.get_loss()
        optimizer = self.get_optimizer()
        return train_loader, val_loader, test_loader, model, loss, optimizer

    def get_dataloader(self):
        dataflow = DatasetInterface.get_dataflow(self.config["Dataset"], self.config["dataset"])

        class DataPipeline(Dataset):
            def __init__(self, dataset, data_pipeline_list, data_pipeline_config):
                self.data_pipeline = DataPipelineInterface.get_pipeline(data_pipeline_list, data_pipeline_config)
                self.dataset = dataset

            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, item):
                data = self.dataset[item]
                for dp in self.data_pipeline:
                    data.update(dp(data["Point"], data["Label"]))
                return data

        train_dataset = DataPipeline(dataflow["train"],
                                     self.config["DataPipeline"]["train"],
                                     self.config["pipeline"]["train"])
        val_dataset = DataPipeline(dataflow["val"],
                                   self.config["DataPipeline"]["val"],
                                   self.config["pipeline"]["val"])
        test_dataset = DataPipeline(dataflow["test"],
                                    self.config["DataPipeline"]["test"],
                                    self.config["pipeline"]["test"])

        train_loader = DataLoader(train_dataset, collate_fn=custom_collate_fn,
                                  **self.config["dataloader"]["train"])
        val_loader = DataLoader(val_dataset, shuffle=False, collate_fn=custom_collate_fn,
                                **self.config["dataloader"]["val"])
        test_loader = DataLoader(test_dataset, shuffle=False, collate_fn=custom_collate_fn,
                                 **self.config["dataloader"]["test"])

        return train_loader, val_loader, test_loader

    def get_model(self):
        model = ModelInterface.get_model(self.config["Model"], self.config["model"])
        return model

    def get_loss(self):
        loss_name = self.config["Loss"]
        loss_weight = self.config["loss"]["loss_weight"]
        if isinstance(loss_name, list):
            loss = []
            for l in loss_name:
                loss.append(LossInterface.get_loss(l, **self.config["loss"]))
            return loss, loss_weight
        elif isinstance(loss_name, str):
            return LossInterface.get_loss(loss_name, **self.config["loss"]), None

    def get_optimizer(self):
        optimizer = OptimizerInterface.get_optimizer(self.config["Optimizer"], **self.config["optimizer"])
        return optimizer
