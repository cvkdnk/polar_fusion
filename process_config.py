import os, yaml
import argparse
import datetime
import shutil
from utils.builders import DataBuilder, ModelBuilder, LossBuilder
from utils import PFBaseClass


def parse_args():
    parser = argparse.ArgumentParser(description="Process config yaml files")
    parser.add_argument("--gen_from_base", "-g", action="store_true", help="Generate config files from base.yaml")
    parser.add_argument("--update", "-u", action="store_true", help="Update base config file")
    parser.add_argument("--force", "-f", action="store_true", help="Cover the existed config directory")
    return parser.parse_args()


def update_base_config():
    """rewrite ./config/base.yaml"""
    assert DataBuilder.DATASET, "ERROR, ./dataloader/dataloader.py -> DatasetLibrary.DATASET " + \
                                "is EMPTY, need to register dataset first"
    assert DataBuilder.PIPELINE, "ERROR, ./dataloader/data_pipeline.py -> DataPipelineLibrary.PIPELINE " + \
                                 "is EMPTY, need to register data pipeline first"
    assert ModelBuilder.MODEL, "ERROR, ./model/model_lib.py -> ModelLibrary.MODEL " + \
                               "is EMPTY, need to register model first"
    assert LossBuilder.LOSS, "ERROR, ./loss/loss_lib.py -> LossLibrary.LOSS " + \
                             "is EMPTY, need to register loss first"

    with open("./config/base.yaml", 'w', encoding='utf-8') as f:
        f.write("# Base config to generate config dir\n")
        f.write("# Last updated: " + str(datetime.datetime.now()) + "\n\n")
        f.write("# The param is where to store the generated config files, \n" + \
                "# base path is {}\n".format(os.getcwd() + "/config"))
        f.write("Dirname: /path/to/store/configs\n\n")
        for strline in yield_line_every5(DataBuilder.DATASET.keys()):
            f.write(strline)
        f.write("Dataset: SemanticKITTI\n\n")
        for strline in yield_line_every5(DataBuilder.PIPELINE.keys()):
            f.write(strline)
        f.write("DataPipeline:\n")
        for mode in ["train", "val", "test"]:
            f.write(f"    - {mode}:\n")
            f.write("         - PointAugmentor\n\n")
        for strline in yield_line_every5(ModelBuilder.MODEL.keys()):
            f.write(strline)
        f.write("Model: Cylinder3D\n\n")
        for strline in yield_line_every5(LossBuilder.LOSS.keys()):
            f.write(strline)
        f.write("Loss:\n    - CrossEntropyLoss\n    - LovaszSoftmax\n\n")
        f.write("# Complete the file and run [python process_config.py -b]")
    print("Update base config file successfully.")


def examine_config(config):
    """check if the config is valid (between model and data pipeline)."""
    need_type = ModelBuilder.MODEL[config["Model"]].NEED_TYPE
    assert need_type, "ERROR, the model have not define NEED_TYPE"
    return_type_list = []
    for data_pipeline in config["DataPipeline"]:
        return_type = DataBuilder.PIPELINE[data_pipeline].RETURN_TYPE
        assert return_type, "ERROR, the data pipeline have not define RETURN_TYPE"
        return_type_list.append(return_type)
    for need_type_i in need_type.split(","):
        assert need_type_i in return_type_list, "ERROR, the need type is not satisfied"


def yield_line_every5(l):
    """yield a string line every 5 elements"""
    str_line = "#"
    for idx, l_i in enumerate(l):
        if idx % 5 == 0 and idx != 0:
            yield str_line + "\n"
            str_line = "#"
        str_line += f" [{l_i}]"
    yield str_line + "\n"


def scan_config(config, cfg_type):
    """When generating config files, scan the config file and print which params need to set"""
    for k, v in config.items():
        if isinstance(v, dict):
            scan_config(v, cfg_type)
        elif v == PFBaseClass.default_str:
            print(cfg_type, k, "is not set.")


def gen_from_base():
    if not os.path.exists("config/base.yaml"):
        update_base_config()
        print("base.yaml is not found. Create a new one.")
        raise FileNotFoundError("Need to complete the base config file and run [python process_config.py -b] again.")
    with open("./config/base.yaml", 'r') as f:
        base_config = yaml.load(f, Loader=yaml.FullLoader)
    work_path = os.path.join(os.path.join("./config", base_config["Dirname"]))
    os.makedirs(work_path, exist_ok=args.force)
    shutil.copy(
        "./config/base.yaml",
        os.path.join(work_path, "base.yaml")
    )

    def create_3mode(path):
        os.makedirs(os.path.join(path, "train"), exist_ok=args.force)
        os.makedirs(os.path.join(path, "val"), exist_ok=args.force)
        os.makedirs(os.path.join(path, "test"), exist_ok=args.force)

    os.makedirs(os.path.join(work_path, "data", "dataset"), exist_ok=args.force)
    create_3mode(os.path.join(work_path, "data", "pipeline"))
    create_3mode(os.path.join(work_path, "data", "dataloader"))
    os.makedirs(os.path.join(work_path, "model"), exist_ok=args.force)
    os.makedirs(os.path.join(work_path, "loss"), exist_ok=args.force)

    data_config = DataBuilder.gen_config_template(base_config["Dataset"], base_config["DataPipeline"])
    dataset_config = data_config["dataset"]
    pipeline_config = data_config["pipeline"]
    dataloader_config = data_config["dataloader"]
    scan_config(dataset_config, "dataset")
    scan_config(pipeline_config, "pipeline")
    scan_config(dataloader_config, "dataloader")
    with open(os.path.join(work_path, "data", "dataset", base_config["Dataset"]+".yaml"), 'w') as f:
        yaml.dump(dataset_config, f)
    for mode in ["train", "val", "test"]:
        with open(os.path.join(work_path, "data", "pipeline", mode, mode+"_pipeline.yaml"), 'w') as f:
            yaml.dump(pipeline_config[mode], f)
        with open(os.path.join(work_path, "data", "dataloader", mode, mode+"_dataloader.yaml"), 'w') as f:
            yaml.dump(dataloader_config[mode], f)
    model_config = ModelBuilder.gen_config_template(base_config["Model"])
    scan_config(model_config, "Model")
    with open(work_path + "/model/" + base_config["Model"] + ".yaml", 'w') as f:
        yaml.dump(model_config, f)
    loss_config = LossBuilder.gen_config_template(base_config["Loss"])
    with open(work_path + "/loss/loss.yaml", 'w') as f:
        yaml.dump(loss_config, f)
    print("Generate config files successfully.")


def load_config(config_path):
    """load config file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError("ERROR, config file not found.")
    if os.path.isfile(config_path):
        with open(config_path, 'r') as i:
            config = yaml.load(i, Loader=yaml.FullLoader)
        return config
    if os.path.isdir(config_path):
        config = {}
        for i in os.listdir(config_path):
            i_path = os.path.join(config_path, i)
            if i.endswith(".yaml"):
                with open(i_path, 'r') as f:
                    config.update(yaml.load(f, Loader=yaml.FullLoader))
            elif os.path.isdir(i_path):
                config[i] = load_config(i_path)
        return config


def main(args):
    if args.gen_from_base:
        gen_from_base()
    if args.update:
        update_base_config()


if __name__ == "__main__":
    args = parse_args()
    main(args)
