import numpy as np

from utils.pf_base_class import PFBaseClass


class DataPipelineBuilder:
    PIPELINE = {}

    def get_pipeline(self, name, pipeline_config):
        return self.PIPELINE[name](pipeline_config)

    @staticmethod
    def register(pipeline_class):
        DataPipelineBuilder.PIPELINE[pipeline_class.__name__] = pipeline_class
        return pipeline_class


@DataPipelineBuilder.register
class PointAugmentor(PFBaseClass):
    """Point Augmentor, which include rotate, jitter, scale and flip """
    def __init__(self, config):
        super(PointAugmentor, self).__init__()
        self.rotate = config["rotate"]
        self.jitter = config["jitter"]
        self.scale = config["scale"]
        self.flip = config["flip"]

    def gen_config_template(cls):
        return {
            "rotate": {'inuse': True, 'rotation_range': 180},
            "jitter": {'inuse': True, 'sigma': 0.01, 'clip': 0.05},
            "scale": {'inuse': True, 'scale_range': 0.1},
            "flip": {'inuse': True, 'flip_axis': 0}
        }

    def __call__(self, pt_features):
        aug_features = pt_features.copy()
        if self.rotate["inuse"]:
            aug_features[:, :3] = self.rotate_point_cloud(pt_features[:, :3], self.rotate["rotation_range"])
        if self.jitter["inuse"]:
            aug_features[:, :3] = self.jitter_point_cloud(pt_features[:, :3], self.jitter["sigma"], self.jitter["clip"])
        if self.scale["inuse"]:
            aug_features[:, :3] = self.scale_point_cloud(pt_features[:, :3], self.scale["scale_range"])
        if self.flip["inuse"]:
            aug_features[:, :3] = self.flip_point_cloud(pt_features[:, :3], self.flip["flip_axis"])
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


@DataPipelineBuilder.register
class InsAugPointAugmentor(PFBaseClass):  # TODO: Complete this class
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
            "ins_aug_classes": cls.default_str+", list type",
            "ins_aug_num": cls.default_str+", list type"
        }

    def __call__(self, pt_features):
        raise NotImplementedError


@DataPipelineBuilder.register
class Voxelize(PFBaseClass):
    def __init__(self, config):
        super(Voxelize, self).__init__()
        self.voxel_size = config["voxel_size"]
        self.
        self.max_points_in_voxel = config["max_points_in_voxel"]
        self.max_voxel_num = config["max_voxel_num"]

    @classmethod
    def gen_config_template(cls):
        return {
            "voxel_size": 0.05,
            "":,
            "max_voxel_num": cls.default_int
        }
