import os
from numba import jit
import numpy as np


class SemKittiUtils:
    """A set of utils used to process SemantickKITTI dataset"""
    @staticmethod
    @jit(nopython=True)
    def load_data_path(data_path, split):
        """return sorted semkitti data path list"""
        data_path_list = []
        for sequence in split:
            velodyne_dir = os.path.join(data_path, str(sequence).zfill(2), "velodyne")
            bin_list = sorted(os.listdir(velodyne_dir))
            for filename in bin_list:
                data_path_list.append(os.path.join(velodyne_dir, filename))
        return data_path_list

    @staticmethod
    def load_data(bin_path: str, return_ins_label=False, test_mode=False):
        pt_features = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        if test_mode:
            sem_labels =
            return pt_features, None
        labels = np.fromfile(bin_path.replace("velodyne", "labels")[:-3]+"label", np.uint32)
        sem_labels = labels & 0xFFFF
        ins_labels = labels >> 16
        if return_ins_label:
            return pt_features, sem_labels, ins_labels
        else:
            return pt_features, sem_labels

    @staticmethod
    def label_mapping(self, sem_labels, kitti_config):
        return np.vectorize(kitti_config["learning_map"].__getitem__)(sem_labels)

    @staticmethod
    def label2word(labels, word_mapping, learning_map_inv=None):
        """If learning_map_inv is not None, it should give the dict."""
        if learning_map_inv is not None:
            labels = np.vectorize(learning_map_inv.__getitem__)(labels)
        words = np.vectorize(word_mapping.__getitem__)(labels)
        return words

