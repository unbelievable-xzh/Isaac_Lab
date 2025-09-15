# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from dask.array import average
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul
from isaaclab.utils.math import matrix_from_quat
from statsmodels.sandbox.gam import Offset

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def open_drawer_bonus(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Bonus for opening the drawer given by the joint position of the drawer.

    The bonus is given when the drawer is open. If the grasp is around the handle, the bonus is doubled.
    """
    drawer_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids[0]]
    is_graspable = aligin_gripper_around_object(env).float()

    return (is_graspable + 1.0) * drawer_pos


# #抬升奖励
# def object_is_lifted(
#     env: ManagerBasedRLEnv,
#     initial_height: float,
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object")
# ) -> torch.Tensor:
#     object: RigidObject = env.scene[object_cfg.name]
#     is_graspable =  aligin_gripper_around_object(env).float()
#     object_height = object.data.root_pos_w[:,2]
#     object_lift = object_height - initial_height
#     return (1+is_graspable) * object_lift

# the distance of object and end_effort
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
    reward = 1.0 / (1.0 + distance ** 2)
    reward = torch.pow(reward, 2)
    return torch.where(distance <= threshold, 2 * reward, reward)


#modify the orientation of end_effort and object
def grasp_object_orientation(
        env: ManagerBasedRLEnv,
        std: float,
        object_frame_cfg: SceneEntityCfg = SceneEntityCfg("object_frame"),
        gripper_frame_cfg: SceneEntityCfg = SceneEntityCfg("gripper_frame"),
        finger_ids=(0, 1, 2, 3),
) -> torch.Tensor:
    # extract the asset (to enable type hinting)
    object: FrameTransformer = env.scene[object_frame_cfg.name]
    gripper: FrameTransformer = env.scene[gripper_frame_cfg.name]
    # obtain the desired and gripper orientations
    # 取目标与各指的四元数 (N,4) 与 (N,K,4)
    obj_q = object.data.target_quat_w[..., 0, :]  # (N,4)
    grip_q_all = gripper.data.target_quat_w  # (N,*,4)
    finger_q = grip_q_all[..., finger_ids, :]  # (N,K,4)
    ##############采用点积或者用四元数差值#####
    eps = 1e-8
    obj_q = obj_q / (obj_q.norm(dim=-1, keepdim=True) + eps)
    finger_q = finger_q / (finger_q.norm(dim=-1, keepdim=True) + eps)

    dots = torch.sum(finger_q * obj_q.unsqueeze(-2), dim=-1).abs().clamp(0.0, 1.0)
    theta2 = 4.0 * (1.0 - dots * dots)
    sigma2 = std * std
    per_finger = torch.exp(-theta2 / (2.0 * sigma2))
    reward = per_finger.mean(dim=-1)  # (N,)
    return reward


###若夹爪处于正确姿态给予布尔奖励
def align_gripper_around_object(
        env: ManagerBasedRLEnv,
        object_frame_cfg: SceneEntityCfg = SceneEntityCfg("object_frame"),
        gripper_frame_cfg: SceneEntityCfg = SceneEntityCfg("gripper_frame"),
        finger_ids=(0, 1, 2, 3),
) -> torch.Tensor:
    object: FrameTransformer = env.scene[object_frame_cfg.name]
    gripper: FrameTransformer = env.scene[gripper_frame_cfg.name]
    obj_pos = object.data.target_pos_w[..., 0, :]
    finger_1_pos = gripper.data.target_pos_w[..., 0, :]
    finger_3_pos = gripper.data.target_pos_w[..., 1, :]
    finger_4_pos = gripper.data.target_pos_w[..., 2, :]
    finger_6_pos = gripper.data.target_pos_w[..., 3, :]
    is_graspable = (finger_1_pos[:, 2] > obj_pos[:, 2]) & (finger_3_pos[:, 2] > obj_pos[:, 2]) & (
                finger_4_pos[:, 2] < obj_pos[:, 2]) & (finger_6_pos[:, 2] < obj_pos[:, 2])
    return is_graspable


def object_is_dropped(
        env: ManagerBasedRLEnv,
        minimal_height: float,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] < minimal_height, 1.0, 0.0)


def approach_gripper_object(
        env: ManagerBasedRLEnv,
        object_frame_cfg: SceneEntityCfg = SceneEntityCfg("object_frame"),
        gripper_frame_cfg: SceneEntityCfg = SceneEntityCfg("gripper_frame"),
        offset: float = 0.1
) -> torch.Tensor:
    object: FrameTransformer = env.scene[object_frame_cfg.name]
    gripper: FrameTransformer = env.scene[gripper_frame_cfg.name]
    """Reward the robot's gripper reaching the drawer handle with the right pose.

    This function returns the distance of fingertips to the handle when the fingers are in a grasping orientation
    (i.e., the left finger is above the handle and the right finger is below the handle). Otherwise, it returns zero.
    """
    # Target object position: (num_envs, 3)
    obj_pos = object.data.target_pos_w[..., 0, :]
    finger_1_pos = gripper.data.target_pos_w[..., 0, :]
    finger_3_pos = gripper.data.target_pos_w[..., 1, :]
    finger_4_pos = gripper.data.target_pos_w[..., 2, :]
    finger_6_pos = gripper.data.target_pos_w[..., 3, :]

    finger_1_dist = torch.abs(finger_1_pos[:, 2] - obj_pos[:, 2])
    finger_3_dist = torch.abs(finger_3_pos[:, 2] - obj_pos[:, 2])
    finger_4_dist = torch.abs(finger_4_pos[:, 2] - obj_pos[:, 2])
    finger_6_dist = torch.abs(finger_6_pos[:, 2] - obj_pos[:, 2])

    is_graspable = (finger_1_pos[:, 2] > obj_pos[:, 2]) & (finger_3_pos[:, 2] > obj_pos[:, 2]) & (
                finger_4_pos[:, 2] < obj_pos[:, 2]) & (finger_6_pos[:, 2] < obj_pos[:, 2])
    average_dist = (offset - finger_1_dist) + (offset - finger_3_dist) + (offset - finger_4_dist) + (
                offset - finger_6_dist)

    return is_graspable * average_dist


def grasp_object(
        env: ManagerBasedRLEnv,
        threshold: float,
        open_joint_pos: float,
        asset_cfg: SceneEntityCfg,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward for closing the fingers when being close to the handle.

    The :attr:`threshold` is the distance from the handle at which the fingers should be closed.
    The :attr:`open_joint_pos` is the joint position when the fingers are open.

    Note:
        It is assumed that zero joint position corresponds to the fingers being closed.
    """
    object: FrameTransformer = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    gripper_pos = ee_frame.data.target_pos_w[..., 0, :]
    obj_pos = object.data.target_pos_w[..., 0, :]

    gripper_joint_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]

    distance = torch.norm(obj_pos - gripper_pos, dim=-1, p=2)
    is_close = distance <= threshold

    return is_close * torch.sum(open_joint_pos - gripper_joint_pos, dim=-1)

#===============================================================================
