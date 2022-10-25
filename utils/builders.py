from utils.dataloader import DATASET


def data_builder(dataset_name):
    return DATASET[dataset_name]



