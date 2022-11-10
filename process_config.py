import os, yaml
import argparse
import datetime
import shutil
from dataloader import DatasetInterface, DataPipelineInterface
from model import ModelInterface
from loss import LossInterface
from utils.optimizer import OptimizerInterface
from utils import PFBaseClass


def parse_args():
    parser = argparse.ArgumentParser(description="Process config yaml files")
    parser.add_argument("--gen_from_base", "-g", action="store_true", help="Generate config files from base.yaml")
    parser.add_argument("--update", "-u", action="store_true", help="Update base config file")
    parser.add_argument("--force", "-f", action="store_true", help="Cover the existed config directory")
    parser.add_argument("--gen_total", "-t", default=None, type=str, help="Generate total config file")
    parser.add_argument("--gen_from_total", "-T", default=None, type=str, help="Generate config files from total.yaml")
    return parser.parse_args()


def update_base_config():
    """rewrite ./config/base.yaml"""
    assert DatasetInterface.REGISTER, "ERROR, ./dataloader/dataloader.py -> DatasetInterface.DATASET " + \
                                     "is EMPTY, need to register dataset first"
    assert DataPipelineInterface.REGISTER, "ERROR, ./dataloader/data_pipeline.py -> DataPipelineInterface.PIPELINE " + \
                                           "is EMPTY, need to register data pipeline first"
    assert ModelInterface.REGISTER, "ERROR, ./model/model.py -> ModelInterface.MODEL " + \
                                 "is EMPTY, need to register model first"
    assert LossInterface.REGISTER, "ERROR, ./loss/loss.py -> LossInterface.LOSS " + \
                               "is EMPTY, need to register loss first"
    assert OptimizerInterface.REGISTER, "ERROR, ./utils/optimizer.py -> OptimizerInterface.OPTIMIZER " + \
                                         "is EMPTY, need to register optimizer first"

    with open("./config/base.yaml", 'w', encoding='utf-8') as f:
        f.write("# Base config to generate config dir\n")
        f.write("# Last updated: " + str(datetime.datetime.now()) + "\n\n")
        f.write("# The param is where to store the generated config files, \n" + \
                "# base path is {}\n".format(os.getcwd() + "/config"))
        f.write("Dirname: /path/to/store/configs\n\n")

        def write_tips(file, _dict):
            for strline in yield_line_every5(_dict.keys()):
                file.write(strline)

        write_tips(f, DatasetInterface.REGISTER)
        f.write("Dataset: SemanticKITTI\n\n")
        write_tips(f, DataPipelineInterface.REGISTER)
        f.write("DataPipeline:\n")
        for mode in ["train", "val", "test"]:
            f.write(f"    {mode}:\n")
            f.write("    - Cylindrical\n    - RangeProject\n\n")
        write_tips(f, ModelInterface.REGISTER)
        f.write("Model: Cylinder3D\n\n")
        write_tips(f, LossInterface.REGISTER)
        f.write("Loss:\n- CrossEntropyLoss\n- LovaszSoftmax\n\n")
        write_tips(f, OptimizerInterface.REGISTER)
        f.write("Optimizer: Adam\n\n")
        f.write("# Complete the file and run [python process_config.py -g]")
    print("Update base config file successfully.")


def examine_config(config):
    """check if the config is valid (between model and data pipeline)."""
    need_type = ModelInterface.REGISTER[config["Model"]].NEED_TYPE
    assert need_type, "ERROR, the model have not define NEED_TYPE"
    return_type_list = []
    for mode in ['train', 'val', 'test']:
        for data_pipeline in config["DataPipeline"][mode]:
            return_type = DataPipelineInterface.REGISTER[data_pipeline].RETURN_TYPE
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


def scan_config(config, path):
    """When generating config files, scan the config file and print which params need to set"""
    for k, v in config.items():
        if isinstance(v, dict):
            scan_config(v, path+"/"+k)
        elif v == PFBaseClass.default_str:
            print(k+" in "+path+" is not set.")


def gen_from_base():
    if not os.path.exists("config/base.yaml"):
        update_base_config()
        print("base.yaml is not found. Create a new one.")
        raise FileNotFoundError("Need to complete the base config file and run [python process_config.py -b] again.")
    with open("./config/base.yaml", 'r') as f:
        base_config = yaml.load(f, Loader=yaml.FullLoader)
    examine_config(base_config)
    work_path = os.path.join(os.path.join("./config", base_config["Dirname"]))
    os.makedirs(work_path, exist_ok=args.force)
    shutil.copy(
        "./config/base.yaml",
        os.path.join(work_path, "base.yaml")
    )

    def create_3mode_dirs(path):
        os.makedirs(os.path.join(path, "train"), exist_ok=args.force)
        os.makedirs(os.path.join(path, "val"), exist_ok=args.force)
        os.makedirs(os.path.join(path, "test"), exist_ok=args.force)

    os.makedirs(os.path.join(work_path, "dataset"), exist_ok=args.force)
    create_3mode_dirs(os.path.join(work_path, "pipeline"))
    create_3mode_dirs(os.path.join(work_path, "dataloader"))
    os.makedirs(os.path.join(work_path, "model"), exist_ok=args.force)
    os.makedirs(os.path.join(work_path, "loss"), exist_ok=args.force)
    os.makedirs(os.path.join(work_path, "optimizer"), exist_ok=args.force)
    # os.makedirs(os.path.join(work_path, "scheduler"), exist_ok=args.force)

    config_template = {
        "dataset": DatasetInterface.gen_config_template(base_config["Dataset"]),
        "pipeline": {
            "train": DataPipelineInterface.gen_config_template(base_config["DataPipeline"]["train"]),
            "val": DataPipelineInterface.gen_config_template(base_config["DataPipeline"]["val"]),
            "test": DataPipelineInterface.gen_config_template(base_config["DataPipeline"]["test"])
        },
        "dataloader": {
            "train": {"batch_size": 4, "shuffle": True, "num_workers": 4, "pin_memory": True, "drop_last": False},
            "val": {"batch_size": 4, "num_workers": 4, "pin_memory": True},
            "test": {"batch_size": 1, "num_workers": 4, "pin_memory": True}
        },
        "model": ModelInterface.gen_config_template(base_config["Model"]),
        "loss": LossInterface.gen_config_template(base_config["Loss"]),
        "optimizer": OptimizerInterface.gen_config_template(base_config["Optimizer"]),
        # "scheduler": SchedulerInterface.gen_config_template(base_config["Scheduler"])
    }
    save_tree(work_path, config_template, base_config)
    gen_total_config(work_path, config=config_template)
    scan_config(config_template, "")
    # with open(work_path + "/scheduler/"+ base_config["Scheduler"] +".yaml", 'w') as f:
    #     yaml.dump(config_template["scheduler"], f)
    print("Generate config files successfully.")


def save_tree(config_path, config, base_config):
    with open(os.path.join(config_path, "dataset", base_config["Dataset"] + ".yaml"), 'w') as f:
        yaml.dump(config["dataset"], f, sort_keys=False)
    for mode in ["train", "val", "test"]:
        with open(os.path.join(config_path, "pipeline", mode, mode + "_pipeline.yaml"), 'w') as f:
            yaml.dump(config["pipeline"][mode], f, sort_keys=False)
        with open(os.path.join(config_path, "dataloader", mode, mode + "_dataloader.yaml"), 'w') as f:
            yaml.dump(config["dataloader"][mode], f, sort_keys=False)
    with open(config_path + "/model/" + base_config["Model"] + ".yaml", 'w') as f:
        yaml.dump(config["model"], f, sort_keys=False)
    with open(config_path + "/loss/loss.yaml", 'w') as f:
        yaml.dump(config["loss"], f, sort_keys=False)
    with open(config_path + "/optimizer/" + base_config["Optimizer"] + ".yaml", 'w') as f:
        yaml.dump(config["optimizer"], f, sort_keys=False)


def load_config(config_path):
    """load config file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError("ERROR, config file not found.")
    if os.path.isfile(config_path):
        with open(config_path, 'r') as i:
            config = yaml.load(i, Loader=yaml.FullLoader)
        if os.path.basename(config_path) == "total.yaml":
            with open(config_path.replace("total", "base"), 'r') as i:
                config.update(yaml.load(i, Loader=yaml.FullLoader))
        return config
    if os.path.isdir(config_path):
        config = {}
        total_config = None
        for i in os.listdir(config_path):
            if i == "total.yaml":
                total_config = yaml.load(open(config_path+"/total.yaml", 'r'), Loader=yaml.FullLoader)
            i_path = os.path.join(config_path, i)
            if i.endswith(".yaml"):
                with open(i_path, 'r') as f:
                    config.update(yaml.load(f, Loader=yaml.FullLoader))
            elif os.path.isdir(i_path):
                config[i] = load_config(i_path)
        if total_config is not None:
            if total_config != config:
                print("[WARNING]: total.yaml is not equal to the tree's config files. Loading the tree's config files.")
        return config


def total_cover_tree(config_path):
    with open(config_path + "total.yaml", 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    with open(config_path + "base.yaml", 'r') as f:
        base_config = yaml.load(f, Loader=yaml.FullLoader)
    save_tree(config_path, config, base_config)
    print("Cover config files successfully.")


def gen_total_config(config_path, config=None):
    if config is None:
        config = load_config(config_path)
    with open(config_path + "/total.yaml", 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    print("Generate total config file successfully.")


def main(args):
    if args.gen_from_base:
        gen_from_base()
    if args.update:
        update_base_config()
    if args.gen_total is not None:
        gen_total_config(args.gen_total)
    elif args.gen_from_total is not None:
        total_cover_tree(args.gen_from_total)


if __name__ == "__main__":
    args = parse_args()
    main(args)
