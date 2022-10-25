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
            sem_labels = np.zeros(pt_features.shape[0], dtype=np.int32)
            return pt_features, None
        labels = np.fromfile(bin_path.replace("velodyne", "labels")[:-3]+"label", np.uint32)
        sem_labels = labels & 0xFFFF
        ins_labels = None
        if return_ins_label:
            ins_labels = labels >> 16
        if return_ins_label:
            return pt_features, sem_labels, ins_labels
        else:
            return pt_features, sem_labels, ins_labels


class NuScenesUtils:
    """A set of utils used to process NuScenes dataset"""


def label_mapping(labels, label_map):
    return np.vectorize(label_map.__getitem__)(labels)


def label2word(labels, word_mapping, learning_map_inv=None):
    """If learning_map_inv is not None, it should give a dict."""
    map_labels = np.copy(labels)
    if learning_map_inv is not None:
        map_labels = np.vectorize(learning_map_inv.__getitem__)(labels)
    words = np.vectorize(word_mapping.__getitem__)(map_labels)
    return words

