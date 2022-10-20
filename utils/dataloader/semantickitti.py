import numpy as np
import torch, yaml, os
from numba import jit


from utils.data_utils import SemKittiUtils
from utils.builders import DatasetBuilder


class PointDataset(DatasetBuilder):
    """Load raw points' data and semantic labels with label_mapping only. No data augment."""
    def __init__(self, data_root, mode='train', return_ref=True,
                 kitti_yaml="./config/semantic-kitti.yaml"):
        super(PointDataset, self).__init__()
        self.return_ref = return_ref
        with open(kitti_yaml, 'r') as f:
            self.kitti_config = yaml.safe_load(f)
        if mode == "train":
            split = self.kitti_config["split"]["train"]
        elif mode == "val":
            split = self.kitti_config["split"]["valid"]
        elif mode == "test":
            split = self.kitti_config["split"]["test"]
        else:
            raise Exception("dataset_type should be one of [train, val, test]")

        self.data_root = data_root
        self.data_path = SemKittiUtils.load_data_path(data_root, split)

    def __getitem__(self, item):
        pt_features, sem_labels = SemKittiUtils.load_data(self.data_path[item])
        sem_labels = np.vectorize(self.kitti_config["learning_map"].__getitem__)(sem_labels)
        if self.return_ref:
            return pt_features, sem_labels
        else:
            return pt_features[:, :3], sem_labels

    def __len__(self):
        return len(self.data_path)


class PointDatasetWithInsLabel(PointDataset):
    """Load raw points' data semantic labels with label_mapping only. No data augment."""
    def __getitem__(self, item):
        pt_features, sem_labels, ins_labels = SemKittiUtils.load_data(self.data_path[item], return_ins_label=True)
        sem_labels = np.vectorize(self.kitti_config["learning_map"].__getitem__)(sem_labels)
        if self.return_ref:
            return pt_features, sem_labels, ins_labels
        else:
            return pt_features[:, :3], sem_labels, ins_labels

    def __len__(self):
        return len(self.data_path)


class AugmentPointDataset(PointDataset):
    """Load points and do random data augment, which include rotate, jitter, scale and flip."""
    def __init__(self, data_root, mode='train', return_ref=True, kitti_yaml="./config/semantic-kitti.yaml",
                 rotate_aug=False, jitter_aug=False, scale_aug=False, flip_aug=False):
        super(AugmentPointDataset, self).__init__(data_root, mode, return_ref, kitti_yaml)
        self.rotate_aug = rotate_aug
        self.jitter_aug = jitter_aug
        self.scale_aug = scale_aug
        self.flip_aug = flip_aug

    def __getitem__(self, item):
        pt_features, sem_labels = SemKittiUtils.load_data(self.data_path[item])
        pt_features = self.augment(pt_features)
        sem_labels = np.vectorize(self.kitti_config["learning_map"].__getitem__)(sem_labels)
        if self.return_ref:
            return pt_features, sem_labels
        else:
            return pt_features[:, :3], sem_labels

    def augment(self, pt_features):
        aug_features = pt_features.copy()
        if self.rotate_aug:
            aug_features[:, :3] = self.rotate_point_cloud(pt_features[:, :3])
        if self.jitter_aug:
            aug_features[:, :3] = self.jitter_point_cloud(pt_features[:, :3])
        if self.scale_aug:
            aug_features[:, :3] = self.scale_point_cloud(pt_features[:, :3])
        if self.flip_aug:
            aug_features[:, :3] = self.flip_point_cloud(pt_features[:, :3])
        return aug_features

    @staticmethod
    def rotate_point_cloud(points, rotation_range=180):
        """ Randomly rotate the point clouds to augument the dataset
            rotation_range: Range of rotation in degree
            Return:
              rotated_points: Rotated point clouds
        """
        rotated_points = np.empty(points.shape, dtype=np.float32)
        rotation_angle = np.random.uniform() * rotation_range
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, sinval],
                                    [-sinval, cosval]])
        rotated_points[:, :2] = np.dot(points[:, :2].reshape((-1, 2)), rotation_matrix)
        rotated_points[:, 2:] = points[:, 2:]
        return rotated_points

    @staticmethod
    def jitter_point_cloud(points, sigma=0.01, clip=0.05):
        """ Randomly jitter points. jittering is per point.
            Input:
              Nx3 array, original point clouds
            Return:
              Nx3 array, jittered point clouds
        """
        N, C = points.shape
        assert (clip > 0)
        jittered_data = np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
        jittered_data += points
        return jittered_data

    @staticmethod
    def scale_point_cloud(points, scale_low=0.8, scale_high=1.25):
        """ Randomly scale the point clouds. Scale is per point cloud.
            Input:
              Nx3 array, original point clouds
            Return:
              Nx3 array, scaled point clouds
        """
        scaled_data = np.random.uniform(scale_low, scale_high)
        scaled_data *= points
        return scaled_data

    @staticmethod
    def flip_point_cloud(points, flip_axis=0):
        """ Randomly flip the point clouds.
            Input:
              Nx3 array, original point clouds
            Return:
              Nx3 array, flipped point clouds
        """
        flipped_points = np.copy(points)
        flipped_points[:, flip_axis] *= -1
        return flipped_points


class VoxelDataset(DatasetBuilder):
    def __init__(self, pt_dataset, mode='train', voxel_size=0.05, rotate_aug=False, flip_aug=False, ignore_label=255,
                 return_test=False, fixed_volume_space=False, max_volume_space=[50, 50, 1.5],
                 min_volume_space=[-50, -50, -3]):
        super(VoxelDataset, self).__init__()
        self.pt_dataset = pt_dataset
        self.mode = mode
        self.kitti_config = pt_dataset.kitti_config
        self.voxel_size = voxel_size
        self.rotate_aug = rotate_aug
        self.ignore_label = ignore_label
        self.return_test = return_test
        self.flip_aug = flip_aug
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space

    def __getitem__(self, item):
        pt_features, sem_labels = self.pt_dataset[item]
        voxels, coordinates, num_points = self.sparse_voxel_generator(pt_features, sem_labels, self.voxel_size)
        return voxels, coordinates, num_points, sem_labels

    def sparse_voxel_generator(self, pt_features, label, voxel_size):
        coords = pt_features[:, :3]
        coords -= np.min(coords, axis=0, keepdims=True)






