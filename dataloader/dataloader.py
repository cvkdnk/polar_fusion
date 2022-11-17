import numpy as np
import yaml

from utils.data_utils import SemKittiUtils, label_mapping
from utils.pf_base_class import PFBaseClass, InterfaceBase


class DatasetInterface(InterfaceBase):
    REGISTER = {}

    @classmethod
    def get(cls, name, config):
        dataflow = {
            "train": cls.REGISTER[name](mode="train", **config),
            "val": cls.REGISTER[name](mode="val", **config),
            "test": cls.REGISTER[name](mode="test", **config)
        }
        return dataflow


class BaseDataset(PFBaseClass):
    """A base class for dataset"""

    def __getitem__(self, item):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    @classmethod
    def gen_config_template(cls):
        raise NotImplementedError

    def yield_data(self):
        for i in range(len(self)):
            yield self[i]

    def comp_dataset_norm_info(self, num_classes):
        """Compute mean, std and proportion of each class"""
        mean = np.zeros(self[0][0].shape[1])
        proportion = np.zeros(num_classes, dtype=np.float32)
        count = 0
        for data, labels in self.yield_data():
            mean += np.sum(data, axis=0)
            count += data.shape[0]
            proportion += np.bincount(labels, minlength=num_classes)
        mean = mean / count
        proportion = proportion / count
        std = np.zeros(self[0][0].shape[1])
        for data, labels in self.yield_data():
            std += np.sum((data - mean) ** 2, axis=0)
        std = np.sqrt(std / count)
        np.save(f"./utils/{self.__name__}_mean.npy", mean)
        np.save(f"./utils/{self.__name__}_std.npy", std)
        np.save(f"./utils/{self.__name__}_proportion.npy", proportion)
        return mean, std, proportion


@DatasetInterface.register
class SemanticKITTI(BaseDataset):
    test_config = {
        'data_root': '/data/semantickitti/sequences',
        'return_rem': True,
        'kitti_yaml': "./config/semantic-kitti.yaml"
    }

    def __init__(self, mode='train', return_ins_label=False, **ds_config):
        super(SemanticKITTI, self).__init__()
        self.return_ref = ds_config["return_rem"]
        self.return_ins_label = return_ins_label
        self.mode = mode
        with open(ds_config["kitti_yaml"], 'r') as f:
            self.kitti_config = yaml.safe_load(f)
        if mode == "train":
            split = self.kitti_config["split"]["train"]
        elif mode == "val":
            split = self.kitti_config["split"]["valid"]
        elif mode == "test":
            split = self.kitti_config["split"]["test"]
        elif mode == "train+val":
            split = self.kitti_config["split"]["train"] + self.kitti_config["split"]["valid"]
        else:
            raise Exception("dataset_type should be one of [train, val, test, train+val]")

        self.data_root = ds_config["data_root"]
        self.data_path = SemKittiUtils.load_data_path(self.data_root, split)

    def __getitem__(self, item):
        test_mode = False if self.mode != 'test' else True
        pt_features, sem_labels, ins_labels, seq_frame = SemKittiUtils.load_data(self.data_path[item],
                                                                                 self.return_ins_label,
                                                                                 test_mode)
        sem_labels = label_mapping(sem_labels, self.kitti_config['learning_map'])
        if not self.return_ref:
            pt_features = pt_features[:, :3]
        points_num = sem_labels.shape[0]
        return_dict = {"Point": pt_features, "Label": sem_labels, "PointsNum": points_num, "SeqFrame": seq_frame}
        if self.return_ins_label:
            return_dict["InstanceLabel"] = ins_labels
        return return_dict

    def __len__(self):
        return len(self.data_path)

    @classmethod
    def gen_config_template(cls):
        cfg_struct = {
            'name': 'SemanticKITTI',
            'data_root': '/data/semkitti/sequences',
            'return_rem': True,
            'kitti_yaml': "./config/semantic-kitti.yaml"
        }
        return cfg_struct


@DatasetInterface.register
class NuScenes(BaseDataset): # TODO: complete nuscenes dataset
    def __init__(self):
        super().__init__()

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass

    @classmethod
    def gen_config_template(cls):
        raise NotImplementedError



