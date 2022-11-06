from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from dataloader import DatasetLibrary, DataPipelineLibrary
from loss import LossLibrary
from utils.pf_base_class import PFBaseClass
from dataloader.data_utils import custom_collate_fn
from model.model_lib import ModelLibrary


class DataBuilder(PFBaseClass):
    DATASET = DatasetLibrary.DATASET
    PIPELINE = DataPipelineLibrary.PIPELINE

    @classmethod
    def gen_config_template(cls, dataset_name=None, data_pipeline=None):
        return {
            "dataset": DatasetLibrary.gen_config_template(dataset_name),
            "pipeline": {
                "train": {dp: DataPipelineLibrary.gen_config_template(dp) for dp in data_pipeline["train"]},
                "val": {dp: DataPipelineLibrary.gen_config_template(dp) for dp in data_pipeline["val"]},
                "test": {dp: DataPipelineLibrary.gen_config_template(dp) for dp in data_pipeline["test"]}
            },
            "dataloader": {
                "train": {"batch_size": 4, "shuffle": True, "num_workers": 4, "pin_memory": True, "drop_last": False},
                "val": {"batch_size": 4, "num_workers": 4, "pin_memory": True},
                "test": {"batch_size": 1, "num_workers": 4, "pin_memory": True}
            }
        }

    def __call__(self, dataset_name, data_pipeline, dataset_config, dataloader_config, datapipeline_config):
        dataflow = DatasetLibrary.get_dataflow(dataset_name, dataset_config)

        class DataPipeline(Dataset):
            def __init__(self, dataset, data_pipeline_list, data_pipeline_config):
                self.data_pipeline = [DataPipelineLibrary.get_pipeline(dp, data_pipeline_config[dp])
                                      for dp in data_pipeline_list]
                self.dataset = dataset

            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, item):
                data = self.dataset[item]
                for dp in self.data_pipeline:
                    data.update(dp(data["Point"], data["Label"]))
                return data

        train_dataset = DataPipeline(dataflow["train"], data_pipeline["train"], datapipeline_config["train"])
        val_dataset = DataPipeline(dataflow["val"], data_pipeline["val"], datapipeline_config["val"])
        test_dataset = DataPipeline(dataflow["test"], data_pipeline["test"], datapipeline_config["test"])

        train_loader = DataLoader(train_dataset, collate_fn=custom_collate_fn, **dataloader_config["train"])
        val_loader = DataLoader(val_dataset, shuffle=False, collate_fn=custom_collate_fn, **dataloader_config["val"])
        test_loader = DataLoader(test_dataset, shuffle=False, collate_fn=custom_collate_fn, **dataloader_config["test"])

        return train_loader, val_loader, test_loader


class ModelBuilder(PFBaseClass):
    MODEL = ModelLibrary.MODEL

    @classmethod
    def gen_config_template(cls, model_name=None):
        return ModelLibrary.gen_config_template(model_name)

    @staticmethod
    def __call__(model_name, model_config):
        return ModelLibrary.get_model(model_name, model_config)


class LossBuilder(PFBaseClass):
    LOSS = LossLibrary.LOSS

    @classmethod
    def gen_config_template(cls, loss=None):
        return_dict = {"ignore": 0}
        if isinstance(loss, list):
            for l in loss:
                return_dict.update(LossLibrary.gen_config_template(l))
            return return_dict
        elif isinstance(loss, str):
            return LossLibrary.gen_config_template(loss)
        else:
            raise TypeError("loss should be list or str")

    @staticmethod
    def __call__(loss_name, loss_config):
        return LossLibrary.get_loss(loss_name, loss_config)


class OptimizerBuilder(PFBaseClass):
    OPTIMIZER = {}

    @classmethod
    def gen_config_template(cls, optimizer=None):
        return cls.OPTIMIZER.gen_config_template(optimizer)

    @staticmethod
    def __call__(optimizer_name, optimizer_config):
        return OptimizerLibrary.get(optimizer_name, optimizer_config)
