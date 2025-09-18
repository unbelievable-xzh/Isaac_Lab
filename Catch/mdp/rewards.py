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
from isaaclab.utils.math import matrix_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# def open_drawer_bonus(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
#     """Bonus for opening the drawer given by the joint position of the drawer.

#     The bonus is given when the drawer is open. If the grasp is around the handle, the bonus is doubled.
#     """
#     drawer_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids[0]]
#     is_graspable = aligin_gripper_around_object(env).float()

#     return (is_graspable + 1.0) * drawer_pos


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
    object: RigidObject = env.scene[object_cfg.name]
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




def align_ee_object(
    env: ManagerBasedRLEnv,
    object_frame_cfg: SceneEntityCfg = SceneEntityCfg("object_frame"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward for aligning the end-effector with the handle.

    The reward is based on the alignment of the gripper with the handle. It is computed as follows:

    .. math::

        reward = 0.5 * (align_z^2 + align_x^2)

    where :math:`align_z` is the dot product of the z direction of the gripper and the -x direction of the handle
    and :math:`align_x` is the dot product of the x direction of the gripper and the -y direction of the handle.
    """
    object: FrameTransformer = env.scene[object_frame_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_quat = ee_frame.data.target_quat_w[..., 0, :]
    object_quat = object.data.target_quat_w[..., 0, :]
    ee_frame_rot_mat = matrix_from_quat(ee_frame_quat)
    object_mat = matrix_from_quat(object_quat)

    # get current x and y direction of the handle
    obj_x, obj_y,obj_z = object_mat[..., 0], object_mat[..., 1], object_mat[..., 2]
    # get current x and z direction of the gripper
    ee_frame_x, ee_frame_y,ee_frame_z = ee_frame_rot_mat[..., 0], ee_frame_rot_mat[..., 1], ee_frame_rot_mat[..., 2]

    # make sure gripper aligns with the handle
    # in this case, the z direction of the gripper should be close to the -x direction of the handle
    # and the x direction of the gripper should be close to the -y direction of the handle
    # dot product of z and x should be large
    align_y = torch.bmm(ee_frame_y.unsqueeze(1), obj_y.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    align_x = torch.bmm(ee_frame_x.unsqueeze(1), obj_x.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    align_z = torch.bmm(ee_frame_z.unsqueeze(1), obj_z.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    return 1 * (torch.sign(align_y) * align_y**2 + torch.sign(align_x) * align_x**2 + torch.sign(align_z) * align_z**2 + torch.sign(align_z) * align_z**2)


###若夹爪处于正确姿态给予布尔奖励
def align_gripper_around_object(
        env: ManagerBasedRLEnv,
        object_frame_cfg: SceneEntityCfg = SceneEntityCfg("object_frame"),
        gripper_frame_cfg: SceneEntityCfg = SceneEntityCfg("gripper_frame"),
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
        threshold: float = 0.02,
) -> torch.Tensor:
    object: FrameTransformer = env.scene[object_frame_cfg.name]
    gripper: FrameTransformer = env.scene[gripper_frame_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]
    obj_pos = object.data.target_pos_w[..., 0, :]
    finger_1_pos = gripper.data.target_pos_w[..., 0, :]
    finger_3_pos = gripper.data.target_pos_w[..., 1, :]
    finger_4_pos = gripper.data.target_pos_w[..., 2, :]
    finger_6_pos = gripper.data.target_pos_w[..., 3, :]
    distance = torch.norm(obj_pos - ee_pos, dim=-1, p=2)

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
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
        offset: float = 0.04,
        threshold: float = 0.01,
) -> torch.Tensor:
    object: FrameTransformer = env.scene[object_frame_cfg.name]
    gripper: FrameTransformer = env.scene[gripper_frame_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    """Reward the robot's gripper reaching the drawer handle with the right pose.

    This function returns the distance of fingertips to the handle when the fingers are in a grasping orientation
    (i.e., the left finger is above the handle and the right finger is below the handle). Otherwise, it returns zero.
    """
    # Target object position: (num_envs, 3)
    obj_pos = object.data.target_pos_w[..., 0, :]
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]
    distance = torch.norm(obj_pos - ee_pos, dim=-1, p=2)
    finger_1_pos = gripper.data.target_pos_w[..., 0, :]
    finger_3_pos = gripper.data.target_pos_w[..., 1, :]
    finger_4_pos = gripper.data.target_pos_w[..., 2, :]
    finger_6_pos = gripper.data.target_pos_w[..., 3, :]

    finger_1_dist = torch.abs(finger_1_pos[:, 2] - obj_pos[:, 2])
    finger_3_dist = torch.abs(finger_3_pos[:, 2] - obj_pos[:, 2])
    finger_4_dist = torch.abs(finger_4_pos[:, 2] - obj_pos[:, 2])
    finger_6_dist = torch.abs(finger_6_pos[:, 2] - obj_pos[:, 2])

    is_graspable = (finger_1_pos[:, 2] > obj_pos[:, 2]) & (finger_3_pos[:, 2] > obj_pos[:, 2]) & (
                finger_4_pos[:, 2] < obj_pos[:, 2]) & (finger_6_pos[:, 2] < obj_pos[:, 2]) & (distance < threshold)
    average_dist = (offset - finger_1_dist) + (offset - finger_3_dist) + (offset - finger_4_dist) + (
                offset - finger_6_dist)

    return is_graspable * average_dist


def grasp_object(
        env: ManagerBasedRLEnv,
        threshold: float,
        open_joint_pos: float,
        asset_cfg: SceneEntityCfg,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        object_frame_cfg: SceneEntityCfg = SceneEntityCfg("object_frame"),
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
        gripper_frame_cfg: SceneEntityCfg = SceneEntityCfg("gripper_frame"),
) -> torch.Tensor:
    """Reward for closing the fingers when being close to the handle.

    The :attr:`threshold` is the distance from the handle at which the fingers should be closed.
    The :attr:`open_joint_pos` is the joint position when the fingers are open.

    Note:
        It is assumed that zero joint position corresponds to the fingers being closed.
    """
    object: FrameTransformer = env.scene[object_frame_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    gripper: FrameTransformer = env.scene[gripper_frame_cfg.name]
    gripper_pos = ee_frame.data.target_pos_w[..., 0, :]
    obj_pos = object.data.target_pos_w[..., 0, :]

    gripper_joint_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]

    distance = torch.norm(obj_pos - gripper_pos, dim=-1, p=2)
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]
    distance = torch.norm(obj_pos - ee_pos, dim=-1, p=2)
    finger_1_pos = gripper.data.target_pos_w[..., 0, :]
    finger_3_pos = gripper.data.target_pos_w[..., 1, :]
    finger_4_pos = gripper.data.target_pos_w[..., 2, :]
    finger_6_pos = gripper.data.target_pos_w[..., 3, :]
    is_graspable = (finger_1_pos[:, 2] > obj_pos[:, 2]) & (finger_3_pos[:, 2] > obj_pos[:, 2]) & (
                finger_4_pos[:, 2] < obj_pos[:, 2]) & (finger_6_pos[:, 2] < obj_pos[:, 2]) & (distance < threshold)

    return is_graspable * torch.sum(open_joint_pos - gripper_joint_pos, dim=-1)

#===============================================================================
