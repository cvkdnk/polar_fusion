from torch import optim

from utils import PFBaseClass, InterfaceBase


class OptimizerInterface(InterfaceBase):
    REGISTER = {}

    @classmethod
    def get(cls, name, config: dict, params=None):
        if "name" in config.keys():
            config.pop("name")
        if isinstance(name, str):
            return cls.REGISTER[name]()(params, **config)
        raise TypeError


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

    def __call__(self, params, **kwargs):
        kwargs["betas"] = tuple(kwargs["betas"])
        return optim.Adam(params, **kwargs)


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

    def __call__(self, params, **kwargs):
        return optim.SGD(params, **kwargs)





