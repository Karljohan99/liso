from typing import List, Union

import torch
from torch import nn
from torch.nn.modules.utils import _pair


class Voxelization(nn.Module):
    """Convert kitti points(N, >=3) to voxels.

    Please refer to `Point-Voxel CNN for Efficient 3D Deep Learning
    <https://arxiv.org/abs/1907.03739>`_ for more details.

    Args:
        voxel_size (tuple or float): The size of voxel with the shape of [3].
        point_cloud_range (tuple or float): The coordinate range of voxel with
            the shape of [6].
        max_num_points (int): maximum points contained in a voxel. if
            max_points=-1, it means using dynamic_voxelize.
        max_voxels (int, optional): maximum voxels this function create.
            for second, 20000 is a good choice. Users should shuffle points
            before call this function because max_voxels may drop points.
            Default: 20000.
    """

    def __init__(self,
                 voxel_size: List,
                 point_cloud_range: List,
                 max_num_points: int,
                 max_voxels: Union[tuple, int] = 20000,
                 deterministic: bool = True):
        """
        Args:
            voxel_size (list): list [x, y, z] size of three dimension
            point_cloud_range (list):
                [x_min, y_min, z_min, x_max, y_max, z_max]
            max_num_points (int): max number of points per voxel
            max_voxels (tuple or int): max number of voxels in
                (training, testing) time
            deterministic: bool. whether to invoke the non-deterministic
                version of hard-voxelization implementations. non-deterministic
                version is considerablly fast but is not deterministic. only
                affects hard voxelization. default True. for more information
                of this argument and the implementation insights, please refer
                to the following links:
                https://github.com/open-mmlab/mmdetection3d/issues/894
                https://github.com/open-mmlab/mmdetection3d/pull/904
                it is an experimental feature and we will appreciate it if
                you could share with us the failing cases.
        """
        super().__init__()

        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.max_num_points = max_num_points
        if isinstance(max_voxels, tuple):
            self.max_voxels = max_voxels
        else:
            self.max_voxels = _pair(max_voxels)
        self.deterministic = deterministic

        point_cloud_range = torch.tensor(
            point_cloud_range, dtype=torch.float32)
        voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        grid_size = (
            point_cloud_range[3:] -  # type: ignore
            point_cloud_range[:3]) / voxel_size  # type: ignore
        grid_size = torch.round(grid_size).long()
        input_feat_shape = grid_size[:2]
        self.grid_size = grid_size
        # the origin shape is as [x-len, y-len, z-len]
        # [w, h, d] -> [d, h, w]
        self.pcd_shape = [*input_feat_shape, 1][::-1]

    def dynamic_voxelize_forward(self, points, voxel_size, coors_range, NDim=3):
        """
        Compute voxel coordinates for each valid point.

        Args:
            points (torch.Tensor): Tensor of shape (N, D) where the first NDim columns 
                                are the spatial coordinates.
            voxel_size (list[float] or torch.Tensor): Voxel size per dimension.
            coors_range (list[float] or torch.Tensor): Coordinate range as 
                [min_0, ..., min_{NDim-1}, max_0, ..., max_{NDim-1}].
            NDim (int): Number of spatial dimensions (default: 3).

        Returns:
            coors (torch.Tensor): Tensor of shape (M, NDim) containing the voxel indices.
        """
        # Convert constant lists to tensors using new_tensor to avoid tracing warnings.
        if not torch.is_tensor(voxel_size):
            voxel_size = points.new_tensor(voxel_size)
        if not torch.is_tensor(coors_range):
            coors_range = points.new_tensor(coors_range)

        min_range = coors_range[:NDim]
        max_range = coors_range[NDim:]
        
        valid_mask = (points[:, :NDim] >= min_range) & (points[:, :NDim] < max_range)
        valid_mask = valid_mask.all(dim=1)
        valid_points = points[valid_mask]

        # Compute voxel coordinates without converting to Python lists.
        coors = ((valid_points[:, :NDim] - min_range) / voxel_size).floor().int()
        return coors


    def hard_voxelize_forward(self, points, voxel_size, coors_range, max_points, max_voxels, NDim=3, deterministic=True):
        """
        Group points into voxels and limit the number of points per voxel.

        Args:
            points (torch.Tensor): Tensor of shape (N, D) containing points (first NDim columns are coordinates).
            voxel_size (list[float] or torch.Tensor): Voxel size per dimension.
            coors_range (list[float] or torch.Tensor): Coordinate range as 
                [min_0, ..., min_{NDim-1}, max_0, ..., max_{NDim-1}].
            max_points (int): Maximum number of points to keep per voxel.
            max_voxels (int): Maximum number of voxels.
            NDim (int): Number of spatial dimensions (default: 3).
            deterministic (bool): Whether to use a deterministic ordering.

        Returns:
            voxels (torch.Tensor): Tensor of shape (M, max_points, D) containing the points per voxel.
            coors (torch.Tensor): Tensor of shape (M, NDim) with the voxel indices.
            num_points_per_voxel (torch.Tensor): Tensor of shape (M,) with the number of points in each voxel.
            voxel_num (int): Total number of voxels (M).
        """
        # Convert constant lists to tensors with new_tensor.
        if not torch.is_tensor(voxel_size):
            voxel_size = points.new_tensor(voxel_size)
        if not torch.is_tensor(coors_range):
            coors_range = points.new_tensor(coors_range)

        min_range = coors_range[:NDim]
        max_range = coors_range[NDim:]
        
        valid_mask = (points[:, :NDim] >= min_range) & (points[:, :NDim] < max_range)
        valid_mask = valid_mask.all(dim=1)
        valid_points = points[valid_mask]

        voxel_coords = ((valid_points[:, :NDim] - min_range) / voxel_size).floor().int()
        
        # Group points by voxel coordinates using torch.unique.
        unique_coords, inverse = torch.unique(voxel_coords, return_inverse=True, dim=0)
        
        # If too many voxels, keep only the first max_voxels.
        if unique_coords.shape[0].item() > max_voxels:
            unique_coords = unique_coords[:max_voxels]
        
        voxel_num = unique_coords.shape[0]
        D = points.shape[1]
        voxels = torch.zeros((voxel_num, max_points, D), device=points.device, dtype=points.dtype)
        coors = unique_coords  # (voxel_num, NDim)
        num_points_per_voxel = torch.zeros((voxel_num,), device=points.device, dtype=torch.int32)

        # For each unique voxel, gather points that fall in it.
        for i in range(voxel_num):
            # Use tensor boolean indexing instead of converting to a Python list.
            indices = (inverse == i).nonzero(as_tuple=True)[0]
            if indices.numel() > max_points:
                indices = indices[:max_points]
            num = indices.numel()
            num_points_per_voxel[i] = num
            pts = valid_points[indices]
            voxels[i, :num] = pts

        return voxels, coors, num_points_per_voxel, voxel_num

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if self.training:
            max_voxels = self.max_voxels[0]
        else:
            max_voxels = self.max_voxels[1]

        if self.max_num_points == -1 or max_voxels == -1:
            #coors = torch.zeros((points.size(0), 3), dtype=torch.int, device=points.device)
            coors = self.dynamic_voxelize_forward(points, self.voxel_size, self.point_cloud_range, NDim=3)
            return coors
        else:
            #voxels = torch.zeros((max_voxels, self.max_num_points, points.size(1)), device=points.device)
            #coors = torch.zeros((max_voxels, 3), dtype=torch.int, device=points.device)
            #num_points_per_voxel = torch.zeros((max_voxels,), dtype=torch.int, device=points.device)
            #voxel_num = torch.zeros((), dtype=torch.long, device=points.device)

            # hard_voxelize_forward(self, points, voxel_size, coors_range, max_points, max_voxels, NDim=3, deterministic=True)
            voxels, coors, num_points_per_voxel, voxel_num = self.hard_voxelize_forward(
                points,
                self.voxel_size,
                self.point_cloud_range,
                #voxels,
                #coors,
                #num_points_per_voxel,
                #voxel_num,
                max_points=self.max_num_points,
                max_voxels=max_voxels,
                NDim=3,
                deterministic=self.deterministic
            )

            # Select the valid voxels
            return voxels[:voxel_num], coors[:voxel_num], num_points_per_voxel[:voxel_num]