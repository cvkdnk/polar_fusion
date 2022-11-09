from torch import optim

from utils import PFBaseClass


class OptimizerInterface(PFBaseClass):
    OPTIMIZER = {}

    @classmethod
    def gen_config_template(cls, name=None):
        assert name in cls.OPTIMIZER.keys(), f"Model {name} not found in {cls.OPTIMIZER.keys()}"
        return cls.OPTIMIZER[name].gen_config_template()

    @classmethod
    def get_optimizer(cls, name, config):
        return cls.OPTIMIZER[name](config)

    @staticmethod
    def register(optimizer_class):
        OptimizerInterface.OPTIMIZER[optimizer_class.__name__] = optimizer_class
        return optimizer_class


class Adam(PFBaseClass):
    @classmethod
    def gen_config_template(cls):
        return {
            "lr": 0.001,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0,
            "amsgrad": False

        }

    @staticmethod
    def __call__(self, **kwargs):
        return optim.Adam(**kwargs)


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





