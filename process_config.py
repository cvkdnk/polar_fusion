import os, yaml
import argparse
import datetime
import shutil
from dataloader import DatasetLibrary, DataPipelineLibrary
from model import ModelLibrary
from utils import PFBaseClass


DATA_ROOT = None


def parse_args():
    parser = argparse.ArgumentParser(description="Process config yaml files")
    parser.add_argument("--gen_from_base", "-g", action="store_true", help="Generate config files from base.yaml")
    parser.add_argument("--update", "-u", action="store_true", help="Update base config file")
    return parser.parse_args()


def update_base_config():
    assert DatasetLibrary.DATASET, "ERROR, ./dataloader/dataloader.py -> DatasetLibrary.DATASET " + \
                                   "is EMPTY, need to register dataset first"
    assert DataPipelineLibrary.PIPELINE, "ERROR, ./dataloader/data_pipeline.py -> DataPipelineLibrary.PIPELINE " + \
                                         "is EMPTY, need to register data pipeline first"
    assert ModelLibrary.MODEL, "ERROR, ./model/model_lib.py -> ModelLibrary.MODEL " + \
                               "is EMPTY, need to register model first"

    with open("./config/base.yaml", 'w', encoding='utf-8') as f:
        f.write("# Base config to generate config dir\n")
        f.write("# Last updated: " + str(datetime.datetime.now()) + "\n\n")
        f.write("# The param is where to store the generated config files, \n" + \
                "# base path is {}\n".format(os.getcwd() + "/config"))
        f.write("Dirname: /path/to/store/configs\n\n")
        for strline in yield_line_every5(DatasetLibrary.DATASET.keys()):
            f.write(strline)
        f.write("Dataset: SemanticKITTI\n\n")
        for strline in yield_line_every5(DataPipelineLibrary.PIPELINE.keys()):
            f.write(strline)
        f.write("DataPipeline:\n")
        f.write("  - PointAugmentor\n\n")
        for strline in yield_line_every5(ModelLibrary.MODEL.keys()):
            f.write(strline)
        f.write("Model: Cylinder3D\n\n")
        f.write("# Complete the file and run [python process_config.py -b]")
    print("Update base config file successfully.")


def examine_config(config):
    need_type = ModelLibrary.MODEL[config["Model"]].NEED_TYPE
    assert need_type, "ERROR, the model have not define NEED_TYPE"
    return_type_list = []
    for data_pipeline in config["DataPipeline"]:
        return_type = DataPipelineLibrary.PIPELINE[data_pipeline].RETURN_TYPE
        assert return_type, "ERROR, the data pipeline have not define RETURN_TYPE"
        return_type_list.append(return_type)
    for need_type_i in need_type.split(","):
        assert need_type_i in return_type_list, "ERROR, the need type is not satisfied"


def yield_line_every5(l):
    str_line = "#"
    for idx, l_i in enumerate(l):
        if idx % 5 == 0 and idx != 0:
            yield str_line + "\n"
            str_line = "#"
        str_line += f" {l_i}"
    yield str_line


def scan_config(config, cfg_type):
    for k, v in config.item():
        if v == PFBaseClass.default_str:
            print(cfg_type, k, "is not set.")


def gen_from_base():
    if not os.path.exists("config/base.yaml"):
        update_base_config()
        print("base.yaml is not found. Create a new one.")
        print("Need to complete the base config file and run [python process_config.py -b] again.")
    with open("./config/base.yaml", 'r') as f:
        base_config = yaml.load(f, Loader=yaml.FullLoader)
    if os.path.exists(base_config["Dirname"]):
        print("ERROR, the directory to store configs is already exist.")
        return
    os.makedirs(base_config["Dirname"])
    work_path = os.path.join(os.path.join(os.getcwd(), "config", base_config["Dirname"]))
    shutil.copy(
        "./config/base.yaml",
        os.path.join(work_path, "base.yaml")
    )
    os.makedirs(work_path + "/dataset")
    os.makedirs(work_path + "/data_pipeline")
    os.makedirs(work_path + "/model")
    dataset_config = DatasetLibrary.DATASET[base_config["Dataset"]].gen_config_template()
    if DATA_ROOT is not None:
        dataset_config["data_root"] = DATA_ROOT
    scan_config(dataset_config, "Dataset")
    with open(work_path + "/dataset/" + base_config["Dataset"] + ".yaml", 'w') as f:
        yaml.dump(dataset_config, f)
    for data_pipeline in base_config["DataPipeline"]:
        data_pipeline_config = DataPipelineLibrary.PIPELINE[data_pipeline].gen_config_template()
        scan_config(data_pipeline_config, "DataPipeline." + data_pipeline)
        with open(work_path + "/data_pipeline/" + data_pipeline + ".yaml", 'w') as f:
            yaml.dump(data_pipeline_config, f)
    model_config = ModelLibrary.MODEL[base_config["Model"]].gen_config_template()
    scan_config(model_config, "Model")
    with open(work_path + "/model/" + base_config["Model"] + ".yaml", 'w') as f:
        yaml.dump(model_config, f)
    print("Generate config files successfully.")


def main(args):
    if args.gen_from_base:
        gen_from_base()
    if args.update:
        update_base_config()

