# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.sensors import FrameTransformer
import isaaclab.utils.math as math_utils
from isaaclab.sensors import FrameTransformerData
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat
from isaaclab.utils.math import subtract_frame_transforms
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

#目标物体在机械臂的跟坐标系下的相对位置和旋转角度
def object_gripper_relative_pose(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    obj_pos = object.data.root_pos_w[..., 0, :]
    ee_frame_pos = ee_frame.data.target_pos_w[..., 0, :]
    return obj_pos - ee_frame_pos

def fingertips_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The position of the fingertips relative to the environment origins."""
    gripper_frame: FrameTransformerData = env.scene["gripper_frame"].data
    fingertips_pos = gripper_frame.target_pos_w[..., :, :] - env.scene.env_origins.unsqueeze(1)

    return fingertips_pos.view(env.num_envs, -1)

def ee_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The position of the end-effector relative to the environment origins."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    ee_pos = ee_tf_data.target_pos_w[..., 0, :] - env.scene.env_origins

    return ee_pos
def ee_quat(env: ManagerBasedRLEnv, make_quat_unique: bool = True) -> torch.Tensor:
    """The orientation of the end-effector in the environment frame.

    If :attr:`make_quat_unique` is True, the quaternion is made unique by ensuring the real part is positive.
    """
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    ee_quat = ee_tf_data.target_quat_w[..., 0, :]
    # make first element of quaternion positive
    return math_utils.quat_unique(ee_quat) if make_quat_unique else ee_quat

def ee_object_rel_pose_obs(
    env: ManagerBasedRLEnv,
    object_frame_cfg: SceneEntityCfg = SceneEntityCfg("object_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """末端坐标系下的相对位置 Δp_ee 与 三个对齐余弦（或改为 9 维 R_rel）。"""
    object_frame: FrameTransformer = env.scene[object_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    # 位姿
    p_obj = object.data.root_pos_w
    p_ee  = ee_frame.data.target_pos_w[..., 0, :]
    q_obj = object_frame.data.target_quat_w[..., 0, :]
    q_ee  = ee_frame.data.target_quat_w[..., 0, :]

    R_obj = matrix_from_quat(q_obj)              # (N,3,3)
    R_ee  = matrix_from_quat(q_ee)               # (N,3,3)

    # Δp in ee frame
    dpos_w  = p_obj - p_ee                       # (N,3)
    dpos_ee = torch.bmm(R_ee.transpose(-1, -2), dpos_w.unsqueeze(-1)).squeeze(-1)  # (N,3)

    # 3个对齐余弦（与你奖励一致的映射）
    def unit(v): return v / (v.norm(dim=-1, keepdim=True).clamp_min(1e-8))
    ex, ey, ez = unit(R_ee[..., 0]), unit(R_ee[..., 1]), unit(R_ee[..., 2])
    ox, oy, oz = unit(R_obj[..., 0]), unit(R_obj[..., 1]), unit(R_obj[..., 2])
    # ay = ey·oy, ax = ex·oz, az = ez·(-ox)
    ay = (ey * oy).sum(-1, keepdim=True)
    ax = (ex * oz).sum(-1, keepdim=True)
    az = (ez * (-ox)).sum(-1, keepdim=True)

    # 拼接：Δp_ee(3) + cos(3) -> (N,6)
    obs = torch.cat([dpos_ee, ax, ay, az], dim=-1)
    return obs
