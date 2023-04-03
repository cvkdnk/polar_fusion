import os, yaml
from utils import PFBaseClass


def load_config(config_path):
    """读取config文件，可以单个文件，也可以是文件夹，也支持base.yaml和detail.yaml的组合（同时读取）"""
    if not os.path.exists(config_path):
        raise FileNotFoundError("ERROR, config file not found.")
    if os.path.isfile(config_path):
        with open(config_path, 'r') as i:
            config = yaml.load(i, Loader=yaml.FullLoader)
        if os.path.basename(config_path) == "detail.yaml":
            with open(config_path.replace("detail", "base"), 'r') as i:
                config.update(yaml.load(i, Loader=yaml.FullLoader))
        if os.path.basename(config_path) == "base.yaml":
            with open(config_path.replace("base", "detail"), 'r') as i:
                config.update(yaml.load(i, Loader=yaml.FullLoader))
        return config
    if os.path.isdir(config_path):
        config = load_config(config_path+"/config.yaml")
        return config


def yield_line_every5(l):
    """yield a string line every 5 elements"""
    str_line = "#"
    for idx, l_i in enumerate(l):
        if idx % 5 == 0 and idx != 0:
            yield str_line + "\n"
            str_line = "#"
        str_line += f" [{l_i}]"
    yield str_line + "\n"


# def scan_config(config, path):
#     """When generating config files, scan the config file and print which params need to set"""
#     for k, v in config.items():
#         if isinstance(v, dict):
#             scan_config(v, path+"/"+k)
#         elif v == PFBaseClass.default_str:
#             print(k+" in "+path+" is not set.")


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
