# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING
from isaaclab.assets import Articulation
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import matrix_from_quat
from isaaclab.utils.math import combine_frame_transforms
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
    """姿态连续打分 + 靠近奖励按对齐度渐进；错姿态贴近会被温和惩罚。"""
    eps = 1e-8
    def unit(v):  # (N,3) -> (N,3)
        return v / (v.norm(dim=-1, keepdim=True).clamp_min(eps))

    object_frame: FrameTransformer = env.scene[object_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    # 姿态矩阵
    ee_q  = ee_frame.data.target_quat_w[..., 0, :]
    obj_q = object_frame.data.target_quat_w[..., 0, :]
    R_ee  = matrix_from_quat(ee_q)      # (N,3,3)
    R_obj = matrix_from_quat(obj_q)     # (N,3,3)

    # 位置
    obj_pos = object.data.root_pos_w                    # (N,3)
    ee_pos  = ee_frame.data.target_pos_w[..., 0, :]    # (N,3)
    dpos    = obj_pos - ee_pos
    zdist   = dpos[:, 2].abs()
    xydist  = torch.norm(dpos[:, :2], dim=1)

    # 轴向（单位化）
    obj_x, obj_y, obj_z = unit(R_obj[..., 0]), unit(R_obj[..., 1]), unit(R_obj[..., 2])
    ee_x,  ee_y,  ee_z  = unit(R_ee[..., 0]), unit(R_ee[..., 1]), unit(R_ee[..., 2])

    # 对齐余弦（你的映射：ey↔oy, ex↔oz, ez↔-ox）
    ay = (ee_y * obj_y).sum(-1).clamp(-1+1e-8, 1-1e-8)
    ax = (ee_x * obj_z).sum(-1).clamp(-1+1e-8, 1-1e-8)
    az = (ee_z * (-obj_x)).sum(-1).clamp(-1+1e-8, 1-1e-8)

    # 姿态连续分数 s_ori ∈ [0,1]（用平方强化高对齐区间）
    sx = (ax + 1) * 0.5
    sy = (ay + 1) * 0.5
    sz = (az + 1) * 0.5
    s_ori = (sx**2 + sy**2 + sz**2) / 3.0  # 对齐越好越接近 1

    # 距离核（平滑、可微）
    def kernel(d, scale):
        return 1.0 / (1.0 + (d / scale) ** 2)

    z_scale  = 0.05
    xy_scale = 0.05
    r_z  = kernel(zdist, z_scale)   # (0,1]
    r_xy = kernel(xydist, xy_scale) # (0,1]

    # —— 关键1：靠近奖励按对齐度渐进 —— 
    # gamma 越大越“姿态先行”；可做 curriculum：前期小、后期大
    gamma = 3.0
    gate  = s_ori.pow(gamma)        # [0,1]，姿态差时≈0，姿态好时≈1

    # —— 关键2：反靠近惩罚（错姿态时贴近会被扣分）——
    # 当 s_ori 小时（1 - s_ori 大），离得越近（xydist, zdist 小）扣得越多
    repel_xy = torch.exp(-(xydist / 0.06) ** 2)  # 近距离≈1，远距离→0
    repel_z  = torch.exp(-(zdist  / 0.04) ** 2)
    anti_approach = (1.0 - s_ori) * (0.5 * repel_xy + 0.5 * repel_z)  # ∈[0,1]

    # 项权重（同量纲，易调）
    w_ori   = 1.0   # 姿态本身奖励（鼓励一直对齐）
    w_close = 1.0   # 靠近奖励的强度
    w_rep   = 0.5   # 反靠近惩罚强度

    # 最终奖励：姿态 + （按姿态门缩放的靠近） - 错姿态贴近惩罚
    reward = (
        w_ori   * s_ori +
        w_close * gate * (0.5 * r_z + 0.5 * r_xy) -
        w_rep   * anti_approach
    )
    return reward


# def align_ee_object(
#     env: ManagerBasedRLEnv,
#     object_frame_cfg: SceneEntityCfg = SceneEntityCfg("object_frame"),
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
#     ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
# ) -> torch.Tensor:
#     """Reward for aligning the end-effector with the handle.

#     The reward is based on the alignment of the gripper with the handle. It is computed as follows:

#     .. math::

#         reward = 0.5 * (align_z^2 + align_x^2)

#     where :math:`align_z` is the dot product of the z direction of the gripper and the -x direction of the handle
#     and :math:`align_x` is the dot product of the x direction of the gripper and the -y direction of the handle.
#     """
#     eps = 1e-8
#     def unit(v):  # (N,3) -> (N,3)
#         return v / (v.norm(dim=-1, keepdim=True).clamp_min(eps))
#     object_frame: FrameTransformer = env.scene[object_frame_cfg.name]
#     object: RigidObject = env.scene[object_cfg.name]
#     ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

#     #姿态
#     ee_frame_quat = ee_frame.data.target_quat_w[..., 0, :]
#     object_quat = object_frame.data.target_quat_w[..., 0, :]
#     ee_frame_rot_mat = matrix_from_quat(ee_frame_quat)
#     object_mat = matrix_from_quat(object_quat)
    
#     cube_pos_w = object.data.root_pos_w          # (N,3)
#     ee_w       = ee_frame.data.target_pos_w[..., 0, :]  # (N,3)

#     dpos   = cube_pos_w - ee_w                   # (N,3)
#     zdist  = dpos[:, 2].abs()                    # 纵向距离 |Δz|, 形状 (N,)
#     xydist = torch.norm(dpos[:, :2], dim=1)      # XY 平面距离 √(dx^2+dy^2), 形状 (N,)
#     ori_thresh  = 2.7  
#     z_thresh    = 0.05   

#     obj_x, obj_y,obj_z = object_mat[..., 0], object_mat[..., 1], object_mat[..., 2]
#     # get current x and z direction of the gripper
#     ee_frame_x, ee_frame_y,ee_frame_z = ee_frame_rot_mat[..., 0], ee_frame_rot_mat[..., 1], ee_frame_rot_mat[..., 2]
    
#     obj_x, obj_y, obj_z = unit(obj_x), unit(obj_y), unit(obj_z)
#     ee_frame_x, ee_frame_y, ee_frame_z = unit(ee_frame_x), unit(ee_frame_y), unit(ee_frame_z)

#     align_y = torch.bmm(ee_frame_y.unsqueeze(1), obj_y.unsqueeze(-1)).squeeze(-1).squeeze(-1)
#     align_x = torch.bmm(ee_frame_x.unsqueeze(1), obj_z.unsqueeze(-1)).squeeze(-1).squeeze(-1)
#     align_z = torch.bmm(ee_frame_z.unsqueeze(1), -obj_x.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    
#     ori_accuracy = torch.sign(align_y) * align_y**2 + torch.sign(align_x) * align_x**2 + torch.sign(align_z) * align_z**2
    
#     # 阶段开关（0/1）
#     ori_ready = (ori_accuracy >= ori_thresh)         
#     z_ready   = (zdist   <= z_thresh)           
#     w_z  = ori_ready.float()                     
#     w_xy = (ori_ready & z_ready).float() 
#     # 距离项建议用平滑核，避免梯度过小/爆炸
#     def kernel(d, scale):
#         return 1.0 / (1.0 + (d / scale) ** 2)

#     z_scale  = 0.05
#     xy_scale = 0.05
#     r_z  = kernel(zdist,  z_scale)   # (N,)
#     r_xy = kernel(xydist, xy_scale)  # (N,)       

#     return ori_accuracy + w_z * r_z + w_xy * r_xy


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


# def approach_gripper_object(
#         env: ManagerBasedRLEnv,
#         object_frame_cfg: SceneEntityCfg = SceneEntityCfg("object_frame"),
#         gripper_frame_cfg: SceneEntityCfg = SceneEntityCfg("gripper_frame"),
#         ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
#         offset: float = 0.08,
#         threshold: float = 0.03,
# ) -> torch.Tensor:
#     object: FrameTransformer = env.scene[object_frame_cfg.name]
#     gripper: FrameTransformer = env.scene[gripper_frame_cfg.name]
#     ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
#     """Reward the robot's gripper reaching the drawer handle with the right pose.

#     This function returns the distance of fingertips to the handle when the fingers are in a grasping orientation
#     (i.e., the left finger is above the handle and the right finger is below the handle). Otherwise, it returns zero.
#     """
#     # Target object position: (num_envs, 3)
#     obj_pos = object.data.target_pos_w[..., 0, :]
#     ee_pos = ee_frame.data.target_pos_w[..., 0, :]

#     distance = torch.norm(obj_pos - ee_pos, dim=-1, p=2)

#     finger_1_pos = gripper.data.target_pos_w[..., 0, :]
#     finger_3_pos = gripper.data.target_pos_w[..., 1, :]
#     finger_4_pos = gripper.data.target_pos_w[..., 2, :]
#     finger_6_pos = gripper.data.target_pos_w[..., 3, :]

#     finger_1_dist = torch.abs(finger_1_pos[:, 2] - obj_pos[:, 2])
#     finger_3_dist = torch.abs(finger_3_pos[:, 2] - obj_pos[:, 2])
#     finger_4_dist = torch.abs(finger_4_pos[:, 2] - obj_pos[:, 2])
#     finger_6_dist = torch.abs(finger_6_pos[:, 2] - obj_pos[:, 2])

#     is_graspable = (finger_1_pos[:, 2] > obj_pos[:, 2]) & (finger_3_pos[:, 2] > obj_pos[:, 2]) & (
#                 finger_4_pos[:, 2] < obj_pos[:, 2]) & (finger_6_pos[:, 2] < obj_pos[:, 2]) & (distance < threshold)
#     average_dist = (offset - finger_1_dist) + (offset - finger_3_dist) + (offset - finger_4_dist) + (
#                 offset - finger_6_dist)

#     return is_graspable * average_dist
def approach_gripper_object(
    env: ManagerBasedRLEnv,
    object_frame_cfg: SceneEntityCfg = SceneEntityCfg("object_frame"),
    gripper_frame_cfg: SceneEntityCfg = SceneEntityCfg("gripper_frame"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    offset: float = 0.08,
    threshold: float = 0.05,   # 放宽或改软门
) -> torch.Tensor:
    object: FrameTransformer  = env.scene[object_frame_cfg.name]
    gripper: FrameTransformer = env.scene[gripper_frame_cfg.name]
    ee_frame: FrameTransformer= env.scene[ee_frame_cfg.name]

    obj_pos = object.data.target_pos_w[..., 0, :]         # (N,3)
    ee_pos  = ee_frame.data.target_pos_w[..., 0, :]       # (N,3)
    distance = torch.norm(obj_pos - ee_pos, dim=-1)

    # 指尖世界位置（确保这些帧就是指尖）
    f1 = gripper.data.target_pos_w[..., 0, :]
    f3 = gripper.data.target_pos_w[..., 1, :]
    f4 = gripper.data.target_pos_w[..., 2, :]
    f6 = gripper.data.target_pos_w[..., 3, :]

    # 取物体系的“上下”轴 u（单位向量）
    obj_q   = object.data.target_quat_w[..., 0, :]        # (N,4)
    R_obj   = matrix_from_quat(obj_q)                     # (N,3,3)
    u       = R_obj[..., 2]                               # 例：obj_z；按你的抓取定义可换

    # 计算沿 u 的有符号高度（>0 为“上”，<0 为“下”）
    def height_along_u(p):  # (N,3)->(N,)
        return torch.sum((p - obj_pos) * u, dim=-1)

    h1, h3 = height_along_u(f1), height_along_u(f3)
    h4, h6 = height_along_u(f4), height_along_u(f6)

    # 硬条件（可留作 gate），但建议配合软门
    is_graspable = (h1 > 0) & (h3 > 0) & (h4 < 0) & (h6 < 0) & (distance < threshold)

    # 软门：姿态/高度满足度（让奖励不至于全零）
    sharp_h = 60.0
    up_ok   = torch.sigmoid(sharp_h *  h1) * torch.sigmoid(sharp_h *  h3)
    down_ok = torch.sigmoid(sharp_h * -h4) * torch.sigmoid(sharp_h * -h6)
    gate_h  = (up_ok * down_ok).clamp(0, 1)

    # 距离软门（越近越接近 1）
    sharp_d = 40.0
    gate_d  = torch.sigmoid(sharp_d * (threshold - distance))

    # === 期望高度：上指 +offset，下指 -offset ===
    h1_star =  offset
    h3_star =  offset
    h4_star = -offset
    h6_star = -offset

    # 每指匹配度（高斯核 or Huber 都行）
    sigma_h = 0.02  # 高度匹配的尺度，可调
    def score(h, h_star):
        return torch.exp(-((h - h_star) / sigma_h) ** 2)  # ∈(0,1]

    s1 = score(h1, h1_star)
    s3 = score(h3, h3_star)
    s4 = score(h4, h4_star)
    s6 = score(h6, h6_star)
    # 上/下两指的“对称性”平衡（鼓励两上指高度接近、两下指接近）
    tips_geom = (s1 * s3 * s4 * s6) ** 0.25  
    def balance(a, b, sigma=0.01):
        return torch.exp(-((a - b) / sigma) ** 2)  # 差越小越接近1

    bal_up   = balance(h1, h3, sigma=0.01)
    bal_down = balance(h4, h6, sigma=0.01)
    tips_balance = (bal_up * bal_down) ** 0.5

    # 整体（夹具中心）接近：用 ee 与物体中心距离的核
    sigma_c = 0.06
    r_center = torch.exp(- (distance / sigma_c) ** 2)  # ∈(0,1]

    # 软门（你已有的 gate_h/gate_d），把“姿态未就绪”时的抓取奖励压低
    gate = (gate_h * gate_d).clamp(0, 1)

    # 最终“同时到位且整体接近”的抓取接近奖励
    w_center = 0.4   # 整体接近权重
    w_tips   = 0.6   # 四指到位权重（更关键）
    reward_approach = gate * (w_center * r_center + w_tips * (tips_geom * tips_balance))

    return reward_approach

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
def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))
