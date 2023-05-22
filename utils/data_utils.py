import os

import cv2
import numba as nb
import torch
import logging
from numba import jit
import numpy as np
import yaml

with open("./config/collate.yaml", 'r') as f:
    collate_dict = yaml.safe_load(f)


class SemKittiUtils:
    """A set of utils used to process SemantickKITTI dataset"""
    @staticmethod
    def load_data_path(data_path, split):
        """return sorted semkitti data path list"""
        data_path_list = []
        for sequence in split:
            velodyne_dir = data_path + "/" + str(sequence).zfill(2) + "/velodyne"
            bin_list = sorted(os.listdir(velodyne_dir))
            for filename in bin_list:
                data_path_list.append(velodyne_dir + "/" + filename)
        return data_path_list

    @staticmethod
    def load_data(bin_path: str, return_ins_label=False, test_mode=False):
        pt_features = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        if test_mode:
            # sem_labels = np.zeros(pt_features.shape[0], dtype=np.int32)
            return pt_features, None
        labels = np.fromfile(bin_path.replace("velodyne", "labels")[:-3]+"label", np.uint32)
        sem_labels = labels & 0xFFFF
        ins_labels = None
        frame = os.path.splitext(os.path.basename(bin_path))[0]
        seq = os.path.dirname(os.path.dirname(bin_path))[-2:]
        seq_frame = seq+"_"+frame
        if return_ins_label:
            ins_labels = labels >> 16
        if return_ins_label:
            return pt_features, sem_labels, ins_labels, seq_frame
        else:
            return pt_features, sem_labels, ins_labels, seq_frame

    @staticmethod
    def draw_rgb_pcd(points, labels, kitti_yaml, save_path):
        """如果是Tensor，转换成ndarray处理"""
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()
        if len(labels.shape) == 2 and labels.shape[1] != 1:
            labels = np.argmax(labels, axis=1)
        labels = np.vectorize(kitti_yaml["learning_map_inv"].__getitem__)(labels)
        colors = np.array(list(map(kitti_yaml["color_map"].__getitem__, labels)))
        save_array = np.concatenate((points, colors), axis=1)
        np.savetxt(save_path, save_array, fmt="%f %f %f %d %d %d")

    @staticmethod
    def draw_rgb_bev(img, kitti_yaml, save_path):
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        img_color = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                img_color[i, j] = kitti_yaml["color_map"][kitti_yaml["learning_map_inv"][img[i, j]]]
        cv2.imwrite(save_path, img_color)
        return img_color

    def draw_acc(points, labels, preds, save_path):
        """只接受ndarray类型，preds可以是one-hot编码，也可以是单个类别"""
        if len(preds.shape) == 2 and preds.shape[1] != 1:
            preds = np.argmax(preds, axis=1)
        pred_err = np.zeros((labels.shape[0], 1), dtype=np.int32)
        pred_err[preds != labels] = 255
        pc_plt = np.concatenate((points, pred_err), axis=1)
        np.savetxt(save_path, pc_plt, fmt="%f %f %f %d", delimiter=" ")


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
    """return rho phi z ..."""
    rho = np.sqrt(input_xyz[..., 0] ** 2 + input_xyz[..., 1] ** 2)
    phi = np.arctan2(input_xyz[..., 1], input_xyz[..., 0])
    return np.hstack((rho.reshape(-1, 1), phi.reshape(-1, 1), input_xyz[..., 2:]))


def polar2cart(input_xyz_polar):
    # print(input_xyz_polar.shape)
    x = input_xyz_polar[..., 0] * np.cos(input_xyz_polar[..., 1])
    y = input_xyz_polar[..., 0] * np.sin(input_xyz_polar[..., 1])
    return np.hstack((x.reshape(-1, 1), y.reshape(-1, 1), input_xyz_polar[..., 2:]))


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


def custom_collate_fn(inputs):
    """Custom collate function to deal with batches that have different numbers of samples per gpu.

    Args:
        inputs (list): a list of dictionaries or other types of data.

    Returns:
        output (dict or other types): a dictionary that contains the collated data for each key in the input
        dictionaries, or the original inputs if they are not dictionaries.

    Notes:
        This function recursively calls itself if the values in the input dictionaries are also dictionaries.
        This function uses a global variable collate_dict to specify how to collate different keys in the input
        dictionaries.
        This function tries to stack, hstack or tensorize the values in the input dictionaries according to their types
        and shapes. If it fails, it returns a list of values instead.
    """
    if isinstance(inputs[0], dict):
        output = {}
        for name in inputs[0].keys():
            if isinstance(inputs[0][name], dict):
                output[name] = custom_collate_fn(
                    [input[name] for input in inputs])
            elif "list" in name or name in collate_dict["list"]:
                output[name] = [torch.tensor(input[name]) for input in inputs]
            elif name in collate_dict["stack"]:
                output[name] = torch.stack(
                    [torch.tensor(input[name]) for input in inputs], dim=0)
            elif name in collate_dict["hstack"]:
                output[name] = torch.hstack(
                    [torch.tensor(input[name]) for input in inputs])
            elif isinstance(inputs[0][name], np.ndarray):
                try:
                    output[name] = torch.stack(
                        [torch.tensor(input[name]) for input in inputs], dim=0)
                except RuntimeError:
                    output[name] = [torch.tensor(input[name]) for input in inputs]
            elif isinstance(inputs[0][name], torch.Tensor):
                try:
                    output[name] = torch.stack([input[name] for input in inputs],
                                               dim=0)
                except RuntimeError:
                    output[name] = [input[name] for input in inputs]
            # elif isinstance(inputs[0][name], SparseTensor):
            #     output[name] = sparse_collate([input[name] for input in inputs])
            else:
                output[name] = [input[name] for input in inputs]
        return output
    else:
        return inputs


def batch_upsampling(voxel_feats, upsampling_inds):
    point_feats = []
    for unsample_index in upsampling_inds:
        point_feats.append(voxel_feats[unsample_index])
    if isinstance(point_feats[0], torch.Tensor):
        point_feats = torch.cat(point_feats, dim=0)
    elif isinstance(point_feats[0], np.ndarray):
        point_feats = np.concatenate(point_feats, axis=0)
    return point_feats




