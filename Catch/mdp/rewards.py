# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul
from isaaclab.utils.math import matrix_from_quat
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

#抬升奖励
def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float, 
    object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)

#夹爪中心接近目标物体奖励（基于距离）
def object_ee_distance(
    env: ManagerBasedRLEnv,
    threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: FrameTransformer = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(cube_pos_w - ee_w, dim=1, p=2)
    reward = 1.0 / (1.0 + distance**2)
    reward = torch.pow(reward, 2)
    return torch.where(distance <= threshold, 2 * reward, reward)

# def grasp_object_orientation(
#     env: ManagerBasedRLEnv,
#     std:float,
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
#     robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# )-> torch.Tensor:
#     # extract the asset (to enable type hinting)
#     object: RigidObject = env.scene[object_cfg.name]
#     robot: RigidObject = env.scene[robot_cfg.name]
#     # obtain the desired and current orientations
#     des_quat_w = object.data.body_quat_w[:, 0, :4]  # [num_envs, 4]
#     curr_quat_w = robot.data.body_quat_w[:, 10,:4]
#     obj_quat_mat  = matrix_from_quat(des_quat_w)

def grasp_object_orientation(
    env: ManagerBasedRLEnv,
    std:float,
    object_frame_cfg: SceneEntityCfg = SceneEntityCfg("object_frame"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
)-> torch.Tensor:
    # extract the asset (to enable type hinting)
    obj_frame: FrameTransformer = env.scene[object_frame_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # obtain the desired and current orientations
    des_quat_w = obj_frame.data.target_quat_w[..., 0, :]
    curr_quat_w = ee_frame.data.target_quat_w[..., 0, :]
    obj_quat_mat  = matrix_from_quat(des_quat_w)
    handle_mat = matrix_from_quat(curr_quat_w)
    handle_x, handle_y = handle_mat[..., 0], handle_mat[..., 1]
    # get current x and z direction of the gripper
    obj_x,obj_y = obj_quat_mat[..., 0], obj_quat_mat[..., 1]
    align_y = torch.bmm(obj_y.unsqueeze(1), handle_y.unsqueeze(1)).squeeze(1).squeeze(1)
    align_x = torch.bmm(obj_x.unsqueeze(1), handle_x.unsqueeze(1)).squeeze(1).squeeze(1)
    return 0.5 * (torch.sign(align_y) * align_y ** 2 + torch.sign(align_x) * align_x ** 2)

def object_is_dropped(
    env: ManagerBasedRLEnv,
    minimal_height: float, 
    object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] < minimal_height, 1.0, 0.0)


def approach_gripper_handle(
        env: ManagerBasedRLEnv,
        offset: float = 0.04
) -> torch.Tensor:
    """Reward the robot's gripper reaching the drawer handle with the right pose.

    This function returns the distance of fingertips to the handle when the fingers are in a grasping orientation
    (i.e., the left finger is above the handle and the right finger is below the handle). Otherwise, it returns zero.
    """
    # Target object position: (num_envs, 3)
    handle_pos = env.scene["object_frame"].data.target_pos_w[..., 0, :]
    # Fingertips position: (num_envs, n_fingertips, 3)
    ee_fingertips_w = env.scene["handle_frame"].data.target_pos_w[..., 0:, :]
    lfinger_pos = ee_fingertips_w[..., 0, :]
    rfinger_pos = ee_fingertips_w[..., 1, :]

    # Compute the distance of each finger from the handle
    lfinger_dist = torch.abs(lfinger_pos[:, 2] - handle_pos[:, 2])
    rfinger_dist = torch.abs(rfinger_pos[:, 2] - handle_pos[:, 2])

    # Check if hand is in a graspable pose
    is_graspable = (rfinger_pos[:, 2] < handle_pos[:, 2]) & (lfinger_pos[:, 2] > handle_pos[:, 2])

    return is_graspable * ((offset - lfinger_dist) + (offset - rfinger_dist))

def align_grasp_around_handle(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Bonus for correct hand orientation around the handle.

    The correct hand orientation is when the left finger is above the handle and the right finger is below the handle.
    """
    # Target object position: (num_envs, 3)
    handle_pos = env.scene["object_frame"].data.target_pos_w[..., 0, :]
    # Fingertips position: (num_envs, n_fingertips, 3)
    ee_fingertips_w = env.scene["handle_frame"].data.target_pos_w[..., 0:, :]
    lfinger_pos = ee_fingertips_w[..., 0, :]
    rfinger_pos = ee_fingertips_w[..., 1, :]

    # Check if hand is in a graspable pose
    is_graspable = (rfinger_pos[:, 2] < handle_pos[:, 2]) & (lfinger_pos[:, 2] > handle_pos[:, 2])

    # bonus if left finger is above the drawer handle and right below
    return is_graspable


#===============================================================================

