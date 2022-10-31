import numpy as np
import torch
from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize

from utils.pf_base_class import PFBaseClass
from utils.data_utils import cart2polar, polar2cart, cart2spherical, spherical2cart


class DataPipelineBuilder(PFBaseClass):
    PIPELINE = {}

    @classmethod
    def get_pipeline(cls, name, pipeline_config):
        return cls.PIPELINE[name](pipeline_config)

    @classmethod
    def gen_config_template(cls, name=None):
        assert name in cls.PIPELINE.keys(), f"Pipeline {name} not found in {cls.PIPELINE.keys()}"
        return cls.PIPELINE[name].gen_config_template()

    @staticmethod
    def register(pipeline_class):
        DataPipelineBuilder.PIPELINE[pipeline_class.__name__] = pipeline_class
        return pipeline_class


class DataPipelineBaseClass(PFBaseClass):
    RETURN_TYPE = None

    @classmethod
    def gen_config_template(cls):
        raise NotImplementedError

    def __init__(self):
        if self.RETURN_TYPE is None:
            raise NotImplementedError

    def __call__(self, pt_features, labels):
        raise NotImplementedError


class PointExtendChannel(DataPipelineBaseClass):
    RETURN_TYPE = "Point"

    def __init__(self, config):
        super().__init__()
        self.extend_channels = config["extend_channels"]

    @classmethod
    def gen_config_template(cls):
        config = {
            "dist": True,
            "pitch": True,
            "yaw": False,
        }
        return config

    def __call__(self, pt_features, labels):
        data = pt_features
        xyz_sphere = cart2spherical(pt_features)
        if self.extend_channels["dist"]:
            data = np.concatenate([pt_features, np.linalg.norm(pt_features[..., :2], axis=1, keepdims=True)], axis=1)
        if self.extend_channels["pitch"]:
            data = np.concatenate([data, xyz_sphere[..., 2]], axis=1)
        if self.extend_channels["yaw"]:
            data = np.concatenate([data, xyz_sphere[..., 1]], axis=1)
        return {"Point": data}


@DataPipelineBuilder.register
class PointAugmentor(DataPipelineBaseClass):
    """Point Augmentor, which include rotate, jitter, scale and flip """
    RETURN_TYPE = "Point"

    def __init__(self, config):
        super(PointAugmentor, self).__init__()
        self.rotate = config["rotate"]
        self.jitter = config["jitter"]
        self.scale = config["scale"]
        self.flip = config["flip"]

    @classmethod
    def gen_config_template(cls):
        return {
            "rotate": {'inuse': True, 'rotation_range': 180},
            "jitter": {'inuse': True, 'sigma': 0.01, 'clip': 0.05},
            "scale": {'inuse': True, 'scale_range': 0.1},
            "flip": {'inuse': True, 'flip_axis': 0}
        }

    def __call__(self, pt_features, labels):
        aug_features = pt_features.copy()
        if self.rotate["inuse"]:
            aug_features[:, :3] = self.rotate_point_cloud(pt_features[:, :3], self.rotate["rotation_range"])
        if self.jitter["inuse"]:
            aug_features[:, :3] = self.jitter_point_cloud(pt_features[:, :3], self.jitter["sigma"], self.jitter["clip"])
        if self.scale["inuse"]:
            aug_features[:, :3] = self.scale_point_cloud(pt_features[:, :3], self.scale["scale_range"])
        if self.flip["inuse"]:
            aug_features[:, :3] = self.flip_point_cloud(pt_features[:, :3], self.flip["flip_axis"])
        return {"Point": aug_features}

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


@DataPipelineBuilder.register
class InsAugPointAugmentor(DataPipelineBaseClass):  # TODO: Complete this class
    RETURN_TYPE = "Point"

    def __init__(self, config):
        super(InsAugPointAugmentor, self).__init__()
        self.ins_aug_max_num = config["ins_aug_max_num"]
        self.ins_aug_scale = config["ins_aug_scale"]
        self.ins_aug_classes = config["ins_aug_classes"]
        self.ins_aug_num = config["ins_aug_num"]

    @classmethod
    def gen_config_template(cls):
        return {
            "ins_aug_max_num": 100,
            "ins_aug_scale": 0.05,
            "ins_aug_classes": cls.default_str + ", list type",
            "ins_aug_num": cls.default_str + ", list type"
        }

    def __call__(self, pt_features, labels):
        raise NotImplementedError


@DataPipelineBuilder.register
class Voxel(PFBaseClass):
    RETURN_TYPE = "Voxel"

    def __init__(self, config):
        super(Voxel, self).__init__()
        self.voxel_size = config["voxel_size"]
        self.fixed_volume_space = config["fixed_volume_space"]  # {inuse, max, min}
        self.max_voxel_num = config["max_voxel_num"]

    @classmethod
    def gen_config_template(cls):
        return {
            "voxel_size": 0.05,
            "fixed_volume_space": {
                "inuse": False,
                "max_volume_space": [50, 50, 1.5],
                "min_volume_space": [-50, -50, -3]
            }
        }

    def __call__(self, pt_features, labels):
        coords = pt_features[..., :3]
        min_bound = np.min(coords, axis=0) // self.voxel_size
        if self.fixed_volume_space["inuse"]:
            coords = np.clip(coords,
                             self.fixed_volume_space["min_volume_space"],
                             self.fixed_volume_space["max_volume_space"])
            min_bound = np.array(self.fixed_volume_space["min_volume_space"]) // self.voxel_size
        coords_pol = cart2polar(coords)

        voxel_idx, v2p_indices, p2v_indices = sparse_quantize(coords,
                                                               voxel_size=self.voxel_size,
                                                               return_index=True,
                                                               return_inverse=True)

        vox_centers = voxel_idx * self.voxel_size + self.voxel_size / 2 + min_bound
        return_xyz = coords - vox_centers[v2p_indices]
        return_xyz = np.concatenate([return_xyz, coords_pol, coords[..., :2]], axis=-1)

        if pt_features.shape[1] == 3:
            return_fea = return_xyz
        else:
            return_fea = np.concatenate((return_xyz, pt_features[:, 3:]), axis=-1)
        return_fea = torch.tensor(return_fea, dtype=torch.float32)
        point_feats_st = SparseTensor(feats=return_fea, coords=voxel_idx)

        return {"Voxel": {
            "point_feats_st": point_feats_st,
            "p2v_indices_list": p2v_indices,
            "v2p_indices_list": v2p_indices,
        }}


@DataPipelineBuilder.register
class Cylindrical(PFBaseClass):
    RETURN_TYPE = "Voxel"

    def __init__(self, config):
        super(Cylindrical, self).__init__()
        self.voxel_size = config["voxel_size"]
        self.fixed_volume_space = config["fixed_volume_space"]
        self.grid_shape = config["grid_shape"]

    @classmethod
    def gen_config_template(cls):
        return {
            "fixed_volume_space": {
                "inuse": False,
                "max_volume_space": [50, np.pi, 2],
                "min_volume_space": [-50, -np.pi, -4]
            },
            "grid_shape": [480, 360, 32]
        }

    def __call__(self, pt_features, labels):
        """return: point_features, voxel_start_coords, p2v, v2p"""
        xyz = pt_features[..., :3]
        xyz_pol = cart2polar(xyz)
        max_bound_r = np.percentile(xyz_pol[..., 0], 100, axis=-2)
        min_bound_r = np.percentile(xyz_pol[..., 0], 0, axis=-2)
        max_bound = np.max(xyz_pol[..., 1:], axis=-2)
        min_bound = np.min(xyz_pol[..., 1:], axis=-2)
        max_bound = np.concatenate(([max_bound_r], max_bound))
        min_bound = np.concatenate(([min_bound_r], min_bound))
        if self.fixed_volume_space["inuse"]:
            max_bound = np.asarray(self.fixed_volume_space["max_volume_space"])
            min_bound = np.asarray(self.fixed_volume_space["min_volume_space"])
            xyz_pol = np.clip(xyz_pol, min_bound, max_bound)
            xyz = polar2cart(xyz_pol)
        # get grid index
        crop_range = max_bound - min_bound
        grid_size = crop_range / self.grid_shape
        if (grid_size == 0).any():
            print("Zero interval!")
        voxel_idx, v2p_indices, p2v_indices = sparse_quantize(
            xyz_pol-min_bound, grid_size, return_index=True, return_inverse=True
        )

        voxel_coords_pol = voxel_idx * grid_size + min_bound
        # center data on each voxel for PTnet
        voxel_centers_pol = voxel_coords_pol + 0.5 * grid_size

        return_xyz = xyz_pol - voxel_centers_pol[v2p_indices]
        # maybe not use regression xy
        # return_xyz = np.concatenate([
        #         return_xyz,
        #         xyz[..., :2]-polar2cart(voxel_centers_pol)[..., :2][v2p_indices]
        #     ], axis=-1)
        return_xyz = np.concatenate((xyz, xyz_pol[..., :2], return_xyz), axis=-1)

        if pt_features.shape[1] == 3:
            return_fea = return_xyz
        else:
            return_fea = np.concatenate((return_xyz, pt_features[:, 3:]), axis=-1)

        point_feats_st = SparseTensor(
            feats=torch.tensor(return_fea, dtype=torch.float32),
            coords=voxel_idx[v2p_indices]
        )
        voxel_feats_st = SparseTensor(
            feats=torch.tensor(return_fea[p2v_indices], dtype=torch.float32),
            coords=voxel_idx
        )

        return {"Voxel": {
            "point_feats_st": point_feats_st,
            "voxel_feats_st": voxel_feats_st,
            "p2v_indices_list": p2v_indices,
            "v2p_indices_list": v2p_indices,
        }}

    def draw(self, data):
        pass


@DataPipelineBuilder.register
class RangeProject(DataPipelineBaseClass):
    RETURN_TYPE = "Range"

    def __init__(self, config):
        super(RangeProject, self).__init__()
        self.proj_H = config["proj_H"]
        self.proj_W = config["proj_W"]
        self.proj_fov_up = config["proj_fov_up"] / 180.0 * np.pi
        self.proj_fov_down = config["proj_fov_down"] / 180.0 * np.pi

    @classmethod
    def gen_config_template(cls):
        return {
            "proj_H": 64,
            "proj_W": 1024,
            "proj_fov_up": 3,
            "proj_fov_down": -25
        }

    def __call__(self, pt_features, labels):
        coords = pt_features[..., :3]
        coords_sph = cart2spherical(coords)
        coords_sph[..., 2] = np.clip(coords_sph[..., 2], self.proj_fov_down, self.proj_fov_up)
        fov = self.proj_fov_up - self.proj_fov_down
        depth = coords_sph[..., 0]
        yaw = coords_sph[..., 1]
        pitch = coords_sph[..., 2]

        # project to image
        proj_x = 0.5 * (yaw / np.pi + 1.0) * self.proj_W
        proj_y = (1.0 - (pitch - self.proj_fov_down) / fov) * self.proj_H

        # round and clamp for use as index
        proj_x = np.floor(proj_x)
        proj_x = np.clip(proj_x, 0, self.proj_W - 1)

        proj_y = np.floor(proj_y)
        proj_y = np.clip(proj_y, 0, self.proj_H - 1)

        unproj_range = np.copy(depth)
        # order in decreasing depth
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        indices = order
        order_pt_features = pt_features[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        # mapping to range image and inverse mapping
        p2r_indices = np.zeros((pt_features.shape[0], 2), dtype=np.int32)
        r2p_indices = np.zeros((self.proj_H, self.proj_W), dtype=np.int32)
        p2r_indices[..., 0] = proj_y
        p2r_indices[..., 1] = proj_x
        r2p_indices[proj_y, proj_x] = indices
        range_image = np.full((self.proj_H, self.proj_W, 5), -1, dtype=np.float32)
        range_image[proj_y, proj_x, 0] = depth
        range_image[proj_y, proj_x, 1:] = order_pt_features
        range_mask = np.zeros((self.proj_H, self.proj_W), dtype=np.int32)
        range_mask[proj_y, proj_x] = 1

        return {"Range": {
                "range_image_ten": range_image,
                "range_mask_ten": range_mask,
                "p2r_indices_list": p2r_indices,
                "r2p_indices_ten": r2p_indices,
            }}


@DataPipelineBuilder.register
class BevProject(DataPipelineBaseClass):  # TODO: complete this
    RETURN_TYPE = "Bev"

    def __init__(self, config):
        super(BevProject, self).__init__()

    @classmethod
    def gen_config_template(cls):
        return {}

    def __call__(self, pt_features, labels):

        return {"Bev": pt_features[..., 1:3]}


@DataPipelineBuilder.register
class PolarBevProject(DataPipelineBaseClass):  # TODO: complete this
    RETURN_TYPE = "Bev"

    def __init__(self, config):
        super(PolarBevProject, self).__init__()

    @classmethod
    def gen_config_template(cls):
        return {}

    def __call__(self, pt_features, labels):
        return {"Bev": pt_features[..., 1:]}
