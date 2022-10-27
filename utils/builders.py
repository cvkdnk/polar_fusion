from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from utils.dataloader import DatasetBuilder
from utils.data_pipeline import DataPipelineBuilder
from utils.pf_base_class import PFBaseClass
from utils.data_utils import custom_collate_fn
from model.model_lib import ModelLibrary


class DataBuilders(PFBaseClass):
    @classmethod
    def gen_config_template(cls):
        raise NotImplementedError

    def __init__(self, dataset_name, data_pipeline):
        self.dataset_name = dataset_name
        self.data_pipeline = data_pipeline

    def __call__(self, dataset_config, dataloader_config):
        dataflow = DatasetBuilder.get_dataflow(self.dataset_name, dataset_config)

        class DataPipeline(Dataset):
            def __init__(self, dataset, data_pipeline_list, data_pipeline_config):
                self.data_pipeline = [DataPipelineBuilder.get_pipeline(dp, data_pipeline_config[dp])
                                      for dp in data_pipeline_list]
                self.dataset = dataset

            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, item):
                data = self.dataset[item]
                for dp in self.data_pipeline:
                    data.update(dp(data["Point"], data["Label"]))
                return data

        train_dataset = DataPipeline(dataflow["train"], self.data_pipeline["train"], dataloader_config["train"])
        val_dataset = DataPipeline(dataflow["val"], self.data_pipeline["val"], dataloader_config["val"])
        test_dataset = DataPipeline(dataflow["test"], self.data_pipeline["test"], dataloader_config["test"])

        train_loader = DataLoader(train_dataset, collate_fn=custom_collate_fn, **dataloader_config["train"])
        val_loader = DataLoader(val_dataset, collate_fn=custom_collate_fn, **dataloader_config["val"])
        test_loader = DataLoader(test_dataset, collate_fn=custom_collate_fn, **dataloader_config["test"])

        return train_loader, val_loader, test_loader


class ModelBuilder(PFBaseClass):
    @classmethod
    def gen_config_template(cls):
        pass

    def __init__(self, model_name):
        self.model_name = model_name

    def __call__(self, model_config):
        return ModelLibrary.get_model(self.model_name, model_config)


