import os, yaml
import argparse
import datetime
import shutil
from dataloader import DatasetInterface, DataPipelineInterface
from model import ModelInterface
from loss import LossInterface
from utils.optimizer import OptimizerInterface
from utils.config_utils import *


def parse_args():
    parser = argparse.ArgumentParser(description="Process config yaml files")
    parser.add_argument("--gen_from_base", "-g", action="store_true", help="Generate config files from base.yaml")
    parser.add_argument("--update", "-u", action="store_true", help="Update base config file")
    parser.add_argument("--force", "-f", action="store_true", help="Cover the existed config directory")
    parser.add_argument("--diy", "-d", action="store_true",
                        help="Generate config files by user interface! Recommend!")
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
                "# base path is {}\n".format(os.getcwd() + "/experiments"))
        f.write("Dirname: \n\n")

        def write_tips(file, _dict):
            for strline in yield_line_every5(_dict.keys()):
                file.write(strline)

        write_tips(f, DatasetInterface.REGISTER)
        f.write("Dataset: SemanticKITTI\n\n")
        write_tips(f, DataPipelineInterface.REGISTER)
        f.write("DataPipeline:\n")
        f.write("    base:\n")
        f.write("    - Cylindrical\n    - RangeProject\n\n")
        f.write("    train:\n")
        f.write("    - PointAugmentor  # cover the base pipeline\n\n")
        write_tips(f, ModelInterface.REGISTER)
        f.write("Model: \n\n")
        write_tips(f, LossInterface.REGISTER)
        f.write("Loss: \n\n")
        write_tips(f, OptimizerInterface.REGISTER)
        f.write("Optimizer: Adam\n\n")
        f.write("# Complete the file and run [python main.py -g]\n\n")
    print("Update base config file successfully.")
    print("Please complete ./config/base.yaml and run [python main.py -g].")


def gen_from_base():
    if not os.path.exists("config/base.yaml"):
        update_base_config()
        print("base.yaml is not found. Create a new one.")
        raise FileNotFoundError("Need to complete the base config file and run [python main.py -b] again.")
    with open("./config/base.yaml", 'r') as f:
        base_config = yaml.load(f, Loader=yaml.FullLoader)
    examine_config(base_config)
    # 生成保存配置文件的路径并创建
    work_path = os.path.join(os.path.join("./experiments", base_config["Dirname"]))
    try:
        os.makedirs(work_path, exist_ok=args.force)
    except FileExistsError:
        print("ERROR, the config dir is existed, use [-f] to force cover it.")
        raise FileExistsError("ERROR, the config dir is existed, use [-f] to force cover it.")
    shutil.copy(
        "./config/base.yaml",
        os.path.join(work_path, "config.yaml")
    )

    # 生成config默认模板
    config_template = {
        "dataset": DatasetInterface.gen_config_template(base_config["Dataset"]),
        "pipeline": {
            "base": DataPipelineInterface.gen_config_template(base_config["DataPipeline"]["base"]),
        },
        "dataloader": {
            "base": {"batch_size": 4, "num_workers": 4, "pin_memory": True},
            "train": {"shuffle": True, "drop_last": False},
        },
        "model": ModelInterface.gen_config_template(base_config["Model"]),
        "loss": LossInterface.gen_config_template(base_config["Loss"]),
        "optimizer": OptimizerInterface.gen_config_template(base_config["Optimizer"]),
        # "scheduler": SchedulerInterface.gen_config_template(base_config["Scheduler"])
    }
    for mode in ["train", "val", "test"]:
        if mode in base_config["DataPipeline"]:
            config_template["pipeline"][mode] = DataPipelineInterface.gen_config_template(
                base_config["DataPipeline"]["train"])
    # 分类生成配置文件的代码，已弃用
    # save_tree(work_path, config_template, base_config)
    gen_config(work_path, config=config_template)
    # scan_config(config_template, "")
    # with open(work_path + "/scheduler/"+ base_config["Scheduler"] +".yaml", 'w') as f:
    #     yaml.dump(config_template["scheduler"], f)
    print("Generate config files successfully, in {}".format(work_path+"/config.yaml"))


def gen_from_interface():
    """generate config files from interface, code very ugly"""
    print("======================Config DIY==========================")
    print("experiment directory list: ", os.listdir("./experiments/"))
    print("input your expdir: ./experiments/", end="")
    exp_dir = "./experiments/" + input()
    if not os.path.exists(exp_dir):
        print("expdir not exists, mkdir")
        os.mkdir(exp_dir)

    def diy_module(mode: str):
        if mode == "dataset":
            interface = DatasetInterface
        elif mode == "model":
            interface = ModelInterface
        elif mode == "loss":
            interface = LossInterface
        elif mode == "optimizer":
            interface = OptimizerInterface
        else:
            raise RuntimeError("mode should be dataset, model, loss or optimizer")
        print("choose a " + mode + ": ", interface.REGISTER.keys())
        print("input: ", end="")
        name = input()
        config = interface.gen_config_template(name)
        print(yaml.dump(config, sort_keys=False, indent=4, default_flow_style=False))
        print("change the config by use dict format, input nothing to skip")
        print("for example: {\"base\": {\"batch_size\": 4}}")
        print("input: ", end="")
        change = input()
        if change:
            config.update(yaml.load(change, Loader=yaml.FullLoader))
        return name, config

    def diy_datapipeline():
        print("Data pipeline is used to process the data from dataset's __getitem__ in order.")
        print("Firstly, you need to choose pipelines used in train, val and test")
        print("If you dont want to use a pipeline in a mode, just input nothing")
        print("choose pipelines for base(split by space): ", DataPipelineInterface.REGISTER.keys())
        print("input: ", end="")
        pipelines = input().split()
        if pipelines:
            pipelines = {"base": pipelines}
        else:
            pipelines = {}
        for mode in ["train", "val", "test"]:
            print("choose pipelines for " + mode + ": ", DataPipelineInterface.REGISTER.keys())
            print("input: ", end="")
            pipelines_list = input().split()
            if pipelines_list:
                pipelines[mode] = pipelines_list
        config = {}
        for mode in pipelines:
            config[mode] = {}
            for pipeline in pipelines[mode]:
                config[mode][pipeline] = DataPipelineInterface.gen_config_template(pipeline)
        print(yaml.dump(config, sort_keys=False, indent=4, default_flow_style=False))
        print("change the config by use dict format, input nothing to skip")
        print("for example: {\"base\": {\"batch_size\": 4}}")
        print("input: ", end="")
        change = input()
        if change:
            config.update(yaml.load(change, Loader=yaml.FullLoader))
        return pipelines, config

    # 生成配置文件，其中涉及到字典添加顺序以及配置文件添加注释的问题，因此代码比较繁琐
    config = {"Dirname": exp_dir.split("/")[-1]}
    detail_config = {}

    def mode_config(mode):
        name, mode_config = diy_module(mode)
        mode_up = mode[0].upper() + mode[1:]
        config[mode_up] = name
        detail_config[mode] = mode_config

    mode_config("dataset")
    pipelines, pipeline_config = diy_datapipeline()
    config["DataPipeline"] = pipelines
    detail_config["dataloader"] = {
            "base": {"batch_size": 4, "num_workers": 4, "pin_memory": True},
            "train": {"shuffle": True, "drop_last": False},
        }
    detail_config["pipeline"] = pipeline_config
    for mode in ["model", "loss", "optimizer"]:
        mode_config(mode)
    examine_config(config)

    with open(exp_dir + "/config.yaml", 'w') as f:
        yaml.dump(config, f)
        f.write("##################################\n")
        f.write("# Detail Config:##################\n")
        for k in detail_config:
            f.write("# " + k + "====================\n")
            yaml.dump({k: detail_config[k]}, f, sort_keys=False)
    print("Generate config files successfully, in {}".format(exp_dir + "/config.yaml"))


def gen_config(config_path, config=None):
    if config is None:
        config = load_config(config_path)
    with open(config_path + "/config.yaml", 'a') as f:
        f.write("##################################\n")
        f.write("# Detail Config:##################\n")
        for k in config:
            f.write("# " + k + "====================\n")
            yaml.dump({k: config[k]}, f, sort_keys=False)
    print("Generate detail config file successfully.")


def examine_config(config):
    """check if the config is valid (between model and data pipeline)."""
    need_type = ModelInterface.REGISTER[config["Model"]].NEED_TYPE
    assert need_type, "ERROR, the model have not define NEED_TYPE"
    return_type_list = []
    for mode in ['base', 'train', 'val', 'test']:
        if mode in config["DataPipeline"]:
            for data_pipeline in config["DataPipeline"][mode]:
                return_type = DataPipelineInterface.REGISTER[data_pipeline].RETURN_TYPE
                assert return_type, "ERROR, the data pipeline have not define RETURN_TYPE"
                return_type_list.append(return_type)
    for need_type_i in need_type.split(","):
        assert need_type_i in return_type_list, "ERROR, the need type is not satisfied"


def main(args):
    if args.gen_from_base:
        gen_from_base()
    if args.update:
        update_base_config()
    if args.gen_total is not None:
        gen_config(args.gen_total)
    if args.diy is not None:
        gen_from_interface()
    # elif args.gen_from_total is not None:
    #     total_cover_tree(args.gen_from_total)


if __name__ == "__main__":
    args = parse_args()
    main(args)
