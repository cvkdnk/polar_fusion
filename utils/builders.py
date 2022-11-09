from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from dataloader import DatasetInterface, DataPipelineInterface
from loss import LossInterface
from utils.optimizer import OptimizerInterface
from model import ModelInterface
from utils.pf_base_class import PFBaseClass
from dataloader.data_utils import custom_collate_fn
from process_config import load_config

class Builder:
    def __init__(self, config_path):
        self.config = load_config(config_path)

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

        train_dataset = DataPipeline(dataflow["train"], self.config["DataPipeline"]["train"], datapipeline_config["train"])
        val_dataset = DataPipeline(dataflow["val"], data_pipeline["val"], datapipeline_config["val"])
        test_dataset = DataPipeline(dataflow["test"], data_pipeline["test"], datapipeline_config["test"])

        train_loader = DataLoader(train_dataset, collate_fn=custom_collate_fn, **dataloader_config["train"])
        val_loader = DataLoader(val_dataset, shuffle=False, collate_fn=custom_collate_fn, **dataloader_config["val"])
        test_loader = DataLoader(test_dataset, shuffle=False, collate_fn=custom_collate_fn, **dataloader_config["test"])

        return train_loader, val_loader, test_loader



# class ModelBuilder(PFBaseClass):
#     MODEL = ModelInterface.MODEL
#
#     @classmethod
#     def gen_config_template(cls, model_name=None):
#         return ModelInterface.gen_config_template(model_name)
#
#     @staticmethod
#     def __call__(model_name, model_config):
#         return ModelInterface.get_model(model_name, model_config)


# class LossBuilder(PFBaseClass):
#     LOSS = LossInterface.LOSS
#
#     @classmethod
#     def gen_config_template(cls, loss=None):
#         return LossInterface.gen_config_template(loss)
#
#     @staticmethod
#     def __call__(loss_name, loss_config):
#         return LossInterface.get_loss(loss_name, loss_config)


# class OptimizerBuilder(PFBaseClass):
#     OPTIMIZER = OptimizerInterface.OPTIMIZER
#
#     @classmethod
#     def gen_config_template(cls, optimizer=None):
#         return cls.OPTIMIZER.gen_config_template(optimizer)
#
#     @staticmethod
#     def __call__(optimizer_name, optimizer_config):
#         return OptimizerInterface.get_optimizer(optimizer_name, optimizer_config)
