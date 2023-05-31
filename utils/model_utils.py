import torch


def pcd_normalize(xyz: torch.Tensor, dst_range=None) -> torch.Tensor:
    """Normalize point cloud to [0, 1] or dst_range.

    Args:
        :param xyz: (N, 3) point cloud,  (torch.Tensor).
        :param dst_range: (2, 3) dst range, (torch.Tensor) or list[torch.Tensor, torch.Tensor].
        :return: (N, 3) normalized point cloud, (torch.Tensor).
    """

    if dst_range is None:
        dst_range = [
            torch.zeros(3, device=xyz.device),
            torch.ones(3, device=xyz.device),
        ]

    min_xyz = xyz.min(dim=0)[0][None, :]
    max_xyz = xyz.max(dim=0)[0][None, :]
    dst_min, dst_max = dst_range[0][None, :], dst_range[1][None, :]

    norm_xyz = (xyz - min_xyz) / (max_xyz - min_xyz)
    norm_xyz = norm_xyz * (dst_max - dst_min) + dst_min

    return norm_xyz
