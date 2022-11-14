from torch import nn

from utils import PFBaseClass, InterfaceBase
from loss.lovasz import lovasz_softmax


class LossInterface(InterfaceBase):
    REGISTER = {}

    @classmethod
    def gen_config_template(cls, loss=None):
        return_dict = {"ignore": 0}
        if isinstance(loss, list):
            for l in loss:
                return_dict.update(LossInterface.gen_config_template(l))
            return_dict["loss_weight"] = [1 for i in range(len(loss))]
            return return_dict
        elif isinstance(loss, str):
            return LossInterface.REGISTER[loss].gen_config_template()
        else:
            raise TypeError("loss should be list or str")

    @classmethod
    def get(cls, name, config):
        if isinstance(name, str):
            return cls.REGISTER[name](**config)
        elif isinstance(name, list):
            return [cls.REGISTER[n](**config) for n in name]
        class_type = cls.__name__.replace("Interface", "")
        raise TypeError(f"{class_type} in base.yaml should be str or list")


class BaseLoss(PFBaseClass):
    @classmethod
    def gen_config_template(cls):
        raise NotImplementedError

    def __init__(self, **config):
        self.config = config

    def __call__(self, pred, target):
        raise NotImplementedError


@LossInterface.register
class CrossEntropyLoss(BaseLoss):
    @classmethod
    def gen_config_template(cls):
        return {
            "weight": None,
        }

    def __init__(self, **config):
        super().__init__(**config)
        self.loss = nn.CrossEntropyLoss(
            weight=self.config["weight"],
            ignore_index=self.config["ignore"]
        )

    def __call__(self, pred, target):
        return self.loss(pred, target)


@LossInterface.register
class LovaszSoftmax(BaseLoss):
    @classmethod
    def gen_config_template(cls):
        return {}

    def __init__(self, **config):
        super().__init__(**config)
        self.ignore = config["ignore"]

    def __call__(self, pred, target):
        return lovasz_softmax(
            nn.functional.softmax(pred), target,
            ignore=self.ignore
        )