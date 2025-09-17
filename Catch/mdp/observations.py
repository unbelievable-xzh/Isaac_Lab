# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.sensors import FrameTransformer
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

#目标物体在机械臂的跟坐标系下的相对位置和旋转角度
def object_position_in_robot_root_frame_pos(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.body_pos_w[:, 0, :3]
    object_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w)
    return object_pos_b

def object_position_in_robot_root_frame_quat(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.body_pos_w[:, 0, :3]
    object_quat_w = object.data.body_quat_w[:,0, :4]
    _, object_quat_b= subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w, object_quat_w)
    return object_quat_b

def object_gripper_distance(
        env: ManagerBasedRLEnv,
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
    return distance.unsqueeze(-1)
