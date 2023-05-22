import numpy as np
import torch
from torch import nn

from utils import PFBaseClass, InterfaceBase
from loss.lovasz import lovasz_softmax


class LossInterface(InterfaceBase):
    REGISTER = {}

    @classmethod
    def gen_config_template(cls, loss=None):
        return_dict = {"loss_name": loss}
        if isinstance(loss, list):
            for l in loss:
                return_dict.update(LossInterface.gen_config_template(l))
            return_dict["loss_weight"] = [1 for i in range(len(loss))]
            return return_dict
        elif isinstance(loss, str):
            return_dict.update(LossInterface.REGISTER[loss].gen_config_template())
            return return_dict
        else:
            raise TypeError("loss should be list or str")

    @classmethod
    def get(cls, name, config, device=None, ignore=0):
        if isinstance(name, str):
            return cls.REGISTER[name](device=device, ignore=ignore, **config)
        class_type = cls.__name__.replace("Interface", "")
        raise TypeError(f"{class_type} in base.yaml should be str")


class BaseLoss(PFBaseClass):
    @classmethod
    def gen_config_template(cls):
        return {}

    def __init__(self, **config):
        self.config = config

    def __call__(self, logits_dict, batch_data):
        raise NotImplementedError


class WeightBaseLoss(BaseLoss):
    def __call__(self, logits_dict, batch_data):
        raise NotImplementedError

    @classmethod
    def gen_config_template(cls):
        return {
            "weight_npy_path": None,
        }

    def __init__(self, **config):
        super().__init__(**config)
        self.weight = self.config["weight_npy_path"]
        if self.weight is not None:
            device = self.config["device"]
            if isinstance(self.weight, list):
                self.weight = np.array(self.weight, dtype=np.float32)
            elif isinstance(self.weight, str):
                self.weight = np.load(self.weight)
            self.weight = torch.tensor(self.weight, dtype=torch.float32, device=device)


@LossInterface.register
class VoxelWCE(WeightBaseLoss):
    def __init__(self, **config):
        super().__init__(**config)
        self.loss = nn.CrossEntropyLoss(
            weight=self.weight,
            ignore_index=self.config["ignore"]
        )

    def __call__(self, logits_dict, batch_data):
        if "dense" in logits_dict:
            pred = logits_dict["dense"]
            target = batch_data["Voxel"]["dense_vox_labels"]
            loss = self.loss(pred, target)
        else:
            raise RuntimeError("No dense prediction in logits_dict")
        return loss


@LossInterface.register
class VoxelLovasz(BaseLoss):
    def __init__(self, **config):
        super().__init__(**config)

    def __call__(self, logits_dict, batch_data):
        if "dense" in logits_dict:
            pred = logits_dict["dense"]
            target = batch_data["Voxel"]["dense_vox_labels"]
        else:
            raise RuntimeError("No dense prediction in logits_dict")
        return lovasz_softmax(
            nn.functional.softmax(pred), target, ignore=self.config["ignore"]
        )


@LossInterface.register
class VoxelWCELovasz(VoxelWCE):
    def __call__(self, logits_dict, batch_data):
        if "dense" in logits_dict:
            pred = logits_dict["dense"]
            target = batch_data["Voxel"]["dense_vox_labels"]
        else:
            raise RuntimeError("No dense prediction in logits_dict")
        loss = self.loss(pred, target) + lovasz_softmax(
            nn.functional.softmax(pred, dim=1), target, ignore=self.config["ignore"])
        return loss


@LossInterface.register
class MultiLogitsLoss(VoxelWCELovasz):
    def __call__(self, logits_dict, batch_data):
        loss = 0
        for logits in logits_dict:
            loss += super().__call__({"dense": logits_dict[logits]}, batch_data)
        return loss


@LossInterface.register
class PointWCELovasz(BaseLoss):
    def __call__(self, logits_dict, batch_data):
        if "point" in logits_dict:
            pred = logits_dict["point"]
            target = batch_data["Label"]
        else:
            raise RuntimeError("No point prediction in logits_dict")
        return lovasz_softmax(
            nn.functional.softmax(pred), target, ignore=self.config["ignore"]
        )
