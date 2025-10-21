# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms
from dataclasses import dataclass
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w)
    return object_pos_b
# 在原文件同一位置新增
def object_position_in_robot_root_frame_noisy(
    env: ManagerBasedRLEnv,
    pos_sigma: float = 0.003,  # 位置白噪声标准差（米），例如 3mm
    use_episode_bias: bool = True,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """物体在机器人根坐标系下的位置（带观测噪声，仅给actor用）"""
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    # 世界系真值
    object_pos_w = obj.data.root_pos_w[:, :3]
    # 转到机器人根坐标系（与你原逻辑一致）
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w
    )  # (N,3)

    # 白噪声（零均值，按米计）
    if pos_sigma > 0.0:
        object_pos_b = object_pos_b + torch.randn_like(object_pos_b) * pos_sigma

    return object_pos_b  # (N,3)

#带有KF滤波后的观测
def object_position_in_robot_root_frame_kf(env: ManagerBasedRLEnv, pos_sigma: float = 0.01,
    q_pos: float = 1e-5, q_vel: float = 1e-4, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    # 量测=带噪位置（同上）
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    object_pos_w = obj.data.root_pos_w[:, :3]
    z_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w)
    # 初始化/持有 KF
    if not hasattr(env, "_objpos_kf"):
        env._objpos_kf = BatchPosVelKF(
            num_envs=env.num_envs, dt=float(env.step_dt), pos_sigma=pos_sigma,
            q_pos=q_pos, q_vel=q_vel, device=env.device
        )
    # 回合重置（根据你的 done 标志改）
    done_mask = (env.episode_length_buf == 0) if hasattr(env, "episode_length_buf") else None
    env._objpos_kf.reset(done_mask)
    # 步进KF，返回滤波后的估计（同维度）
    p_est = env._objpos_kf.step(z=z_b + torch.randn_like(z_b) * pos_sigma)  # 若你已经在 policy 里加了噪声，这里可直接 z=z_b
    return p_est