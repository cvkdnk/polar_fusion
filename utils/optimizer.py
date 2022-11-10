from torch import optim

from utils import PFBaseClass, InterfaceBase


class OptimizerInterface(InterfaceBase):
    REGISTER = {}


@OptimizerInterface.register
class Adam(PFBaseClass):
    @classmethod
    def gen_config_template(cls):
        return {
            "lr": 0.001,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0,
            "amsgrad": False

        }

    @staticmethod
    def __call__(self, **kwargs):
        kwargs["betas"] = tuple(kwargs["betas"])
        return optim.Adam(**kwargs)


@OptimizerInterface.register
class SGD(PFBaseClass):
    @classmethod
    def gen_config_template(cls):
        return {
            'lr': 0.01,
            'momentum': 0,
            'dampening': 0,
            'weight_decay': 0,
            'nesterov': False
        }

    @staticmethod
    def __call__(self, **kwargs):
        return optim.SGD(**kwargs)





