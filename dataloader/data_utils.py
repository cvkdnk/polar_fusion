import os
import numba as nb
import torch
from numba import jit
import numpy as np
from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate


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


def cart2polar(input_xyz):
    rho = np.sqrt(input_xyz[..., 0] ** 2 + input_xyz[..., 1] ** 2)
    phi = np.arctan2(input_xyz[..., 1], input_xyz[..., 0])
    return np.stack((rho, phi, input_xyz[..., 2:]), axis=-1)


def polar2cart(input_xyz_polar):
    # print(input_xyz_polar.shape)
    x = input_xyz_polar[..., 0] * np.cos(input_xyz_polar[..., 1])
    y = input_xyz_polar[..., 0] * np.sin(input_xyz_polar[..., 1])
    return np.stack((x, y, input_xyz_polar[..., 2:]), axis=-1)


# def cart2polar3d(input_xyz):
#     rho = np.sqrt(input_xyz[..., 0] ** 2 + input_xyz[..., 1] ** 2)
#     phi = np.arctan2(input_xyz[..., 1], input_xyz[..., 0])
#     theta = np.arctan2(input_xyz[..., 2], rho)
#     return np.stack((rho, phi, theta), axis=-1)
#
#
# def polar2cart3d(input_xyz):
#     x = input_xyz[..., 0] * np.cos(input_xyz[..., 1]) * np.sin(input_xyz[..., 2])
#     y = input_xyz[..., 0] * np.sin(input_xyz[..., 1]) * np.sin(input_xyz[..., 2])
#     z = input_xyz[..., 0] * np.cos(input_xyz[..., 2])
#     return np.stack((x, y, z), axis=-1)


def cart2spherical(input_xyz):
    """return rho, phi, theta; also known as depth, yaw, pitch"""
    depth = np.sqrt(input_xyz[..., 0] ** 2 + input_xyz[..., 1] ** 2 + input_xyz[..., 2] ** 2)
    yaw = np.arctan2(input_xyz[..., 1], input_xyz[..., 0])
    pitch = np.arcsin(input_xyz[..., 2] / depth)
    return np.stack((depth, yaw, pitch), axis=-1)


def spherical2cart(input_dyp):
    x = input_dyp[..., 0] * np.cos(input_dyp[..., 2]) * np.cos(input_dyp[..., 1])
    y = input_dyp[..., 0] * np.cos(input_dyp[..., 2]) * np.sin(input_dyp[..., 1])
    z = input_dyp[..., 0] * np.sin(input_dyp[..., 2])
    return np.stack((x, y, z), axis=-1)


@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True, parallel=False)
def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    return processed_label


def custom_collate_fn(inputs):  # TODO: complete collate function
    """Custom collate function to deal with batches that have different
    numbers of samples per gpu.
    """
    if isinstance(inputs[0], dict):
        output = {}
        for name in inputs[0].keys():
            if isinstance(inputs[0][name], dict):
                output[name] = custom_collate_fn(
                    [input[name] for input in inputs])
            elif "list" in name or "Point" == name or "Label" == name:
                output[name] = [input[name] for input in inputs]
            elif isinstance(inputs[0][name], np.ndarray):
                output[name] = torch.stack(
                    [torch.tensor(input[name]) for input in inputs], dim=0)
            elif isinstance(inputs[0][name], torch.Tensor):
                output[name] = torch.stack([input[name] for input in inputs],
                                           dim=0)
            elif isinstance(inputs[0][name], SparseTensor):
                output[name] = sparse_collate([input[name] for input in inputs])
            else:
                output[name] = [input[name] for input in inputs]
        return output
    else:
        return inputs
