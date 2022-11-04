from torch import nn

from utils import PFBaseClass
from loss.lovasz import lovasz_softmax


class LossLibrary(PFBaseClass):
    LOSS = {}

    @classmethod
    def gen_config_template(cls, name=None):
        assert name in cls.LOSS.keys(), f"Loss {name} not found in {cls.LOSS.keys()}"
        return cls.LOSS[name].gen_config_template()

    @classmethod
    def get_loss(cls, name, config):
        return cls.LOSS[name](config)

    @staticmethod
    def register(loss_class):
        LossLibrary.LOSS[loss_class.__name__] = loss_class
        return loss_class


class BaseLoss(PFBaseClass):
    @classmethod
    def gen_config_template(cls):
        raise NotImplementedError

    def __init__(self, config):
        self.config = config

    def __call__(self, pred, target):
        raise NotImplementedError


@LossLibrary.register
class CrossEntropyLoss(BaseLoss):
    @classmethod
    def gen_config_template(cls):
        return {
            "weight": None,
        }

    def __init__(self, config):
        super().__init__(config)
        self.loss = nn.CrossEntropyLoss(
            weight=self.config["weight"],
            ignore_index=self.config["ignore"]
        )

    def __call__(self, pred, target):
        return self.loss(pred, target)


@LossLibrary.register
class LovaszSoftmax(BaseLoss):
    @classmethod
    def gen_config_template(cls):
        return {}

    def __init__(self, config):
        super().__init__(config)
        self.ignore = config["ignore"]

    def __call__(self, pred, target):
        return lovasz_softmax(
            pred, target,
            ignore=self.ignore
        )