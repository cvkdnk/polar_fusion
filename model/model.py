import pytorch_lightning as pl
import torch
import time

import spconv.pytorch as spconv

from utils.pf_base_class import InterfaceBase
from utils.train_utils import AverageMeter
from utils.data_utils import batch_upsampling
from model.pl_base_model import ModuleBaseClass


class ModelInterface(InterfaceBase):
    REGISTER = {}

    @classmethod
    def get_model(cls, config, ckpt=None):
        name = config["Model"]
        if isinstance(name, str):
            if ckpt is not None:
                return cls.REGISTER[name].load_from_checkpoint(ckpt)
            return cls.REGISTER[name](config)
        class_type = cls.__name__.replace("Interface", "")
        raise TypeError(f"{class_type} in base.yaml should be str")


if __name__ == "__main__":
    import time
    import torch
    import numpy as np

