# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.assets import Articulation
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
        near_radius: float= 0.3,                      # 例如 0.30
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """距离奖励：到达 near_radius 后不再增长（封顶）"""
    # 提取量
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cube_pos_w = object.data.root_pos_w                  # (N,3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]         # (N,3)

    # 距离与基础奖励
    distance = torch.norm(cube_pos_w - ee_w, dim=1, p=2) # (N,)
    base = (1.0 / (1.0 + distance**2)).pow(2)            # (N,)

    # 封顶值：等于在 near_radius 处的奖励
    cap = (1.0 / (1.0 + near_radius * near_radius)) ** 2

    # 近距离内不再变化：等价于 torch.clamp_max(base, cap)
    reward = torch.minimum(base, torch.full_like(base, cap))
    return reward





def align_ee_object(
    env: ManagerBasedRLEnv,
    object_frame_cfg: SceneEntityCfg = SceneEntityCfg("object_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward for aligning the end-effector with the handle.

    The reward is based on the alignment of the gripper with the handle. It is computed as follows:

    .. math::

        reward = 0.5 * (align_z^2 + align_x^2)

    where :math:`align_z` is the dot product of the z direction of the gripper and the -x direction of the handle
    and :math:`align_x` is the dot product of the x direction of the gripper and the -y direction of the handle.
    """
    object_frame: FrameTransformer = env.scene[object_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    #姿态
    ee_frame_quat = ee_frame.data.target_quat_w[..., 0, :]
    object_quat = object_frame.data.target_quat_w[..., 0, :]
    ee_frame_rot_mat = matrix_from_quat(ee_frame_quat)
    object_mat = matrix_from_quat(object_quat)
    
    cube_pos_w = object.data.root_pos_w          # (N,3)
    ee_w       = ee_frame.data.target_pos_w[..., 0, :]  # (N,3)

    dpos   = cube_pos_w - ee_w                   # (N,3)
    zdist  = dpos[:, 2].abs()                    # 纵向距离 |Δz|, 形状 (N,)
    xydist = torch.norm(dpos[:, :2], dim=1)      # XY 平面距离 √(dx^2+dy^2), 形状 (N,)
    ori_thresh  = 1.3  
    z_thresh    = 0.03   

    obj_x, obj_y,obj_z = object_mat[..., 0], object_mat[..., 1], object_mat[..., 2]
    # get current x and z direction of the gripper
    ee_frame_x, ee_frame_y,ee_frame_z = ee_frame_rot_mat[..., 0], ee_frame_rot_mat[..., 1], ee_frame_rot_mat[..., 2]

    align_y = torch.bmm(ee_frame_y.unsqueeze(1), obj_y.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    align_x = torch.bmm(ee_frame_x.unsqueeze(1), obj_z.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    align_z = torch.bmm(ee_frame_z.unsqueeze(1), -obj_x.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    
    ori_accuracy = torch.sign(align_y) * align_y**2 + torch.sign(align_x) * align_x**2 + torch.sign(align_z) * align_z**2
    
    # 阶段开关（0/1）
    ori_ready = (ori_accuracy >= ori_thresh)         
    z_ready   = (zdist   <= z_thresh)           
    w_z  = ori_ready.float()                     
    w_xy = (ori_ready & z_ready).float() 
    # 距离项建议用平滑核，避免梯度过小/爆炸
    def kernel(d, scale):
        return 1.0 / (1.0 + (d / scale) ** 2)

    z_scale  = 0.05
    xy_scale = 0.05
    r_z  = kernel(zdist,  z_scale)   # (N,)
    r_xy = kernel(xydist, xy_scale)  # (N,)       

    return ori_accuracy + w_z * r_z + w_xy * r_xy


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
        offset: float = 0.08,
        threshold: float = 0.03,
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
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
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
    object: RigidObject = env.scene[object_cfg.name]
    robot:  Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    gripper: FrameTransformer = env.scene[gripper_frame_cfg.name]
    gripper_pos = ee_frame.data.target_pos_w[..., 0, :]
    obj_pos= object.data.root_pos_w

    gripper_joint_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]

    distance = torch.norm(obj_pos - gripper_pos, dim=-1, p=2)
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]
    distance = torch.norm(obj_pos - ee_pos, dim=-1, p=2)
    finger_1_pos = gripper.data.target_pos_w[..., 0, :]
    finger_3_pos = gripper.data.target_pos_w[..., 1, :]
    finger_4_pos = gripper.data.target_pos_w[..., 2, :]
    finger_6_pos = gripper.data.target_pos_w[..., 3, :]
    is_graspable = (finger_1_pos[:, 2] > obj_pos[:, 2]) & (finger_3_pos[:, 2] > obj_pos[:, 2]) & (
                finger_4_pos[:, 2] < obj_pos[:, 2]) & (finger_6_pos[:, 2] < obj_pos[:, 2]) & (distance < threshold)
    
    gap = (open_joint_pos - gripper_joint_pos).clamp(min=0.0)
    close_frac = (gap / open_joint_pos).mean(dim=-1)
    r_close = is_graspable * close_frac

    return 2 * r_close
#===============================================================================
