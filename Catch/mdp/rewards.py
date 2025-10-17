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
# def object_ee_distance(
#         env: ManagerBasedRLEnv,
#         near_radius: float= 0.3,                      # 例如 0.30
#         object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
#         ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
# ) -> torch.Tensor:
#     """距离奖励：到达 near_radius 后不再增长（封顶）"""
#     # 提取量
#     object: RigidObject = env.scene[object_cfg.name]
#     ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
#     cube_pos_w = object.data.root_pos_w                  # (N,3)
#     ee_w = ee_frame.data.target_pos_w[..., 0, :]         # (N,3)

#     # 距离与基础奖励
#     distance = torch.norm(cube_pos_w - ee_w, dim=1, p=2) # (N,)
#     base = (1.0 / (1.0 + distance**2)).pow(2)            # (N,)

#     # 封顶值：等于在 near_radius 处的奖励
#     cap = (1.0 / (1.0 + near_radius * near_radius)) ** 2

#     # 近距离内不再变化：等价于 torch.clamp_max(base, cap)
#     reward = torch.minimum(base, torch.full_like(base, cap))
#     return reward
def object_ee_distance(
    env: ManagerBasedRLEnv,
    near_radius: float = 0.30,   # ← 交接点 = 0.3 m
    sigma_far:  float = 0.6,     # 远处拉力更“长”
    sigma_near: float = 0.08,    # 近处更敏感
    softcap_k:  float = 6.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    p_obj = object.data.root_pos_w
    p_ee  = ee_frame.data.target_pos_w[..., 0, :]
    d     = torch.norm(p_obj - p_ee, dim=1)  # (N,)

    # 统一 dtype/device
    device = d.device
    dtype  = d.dtype
    sigma_far_t   = torch.as_tensor(sigma_far,  device=device, dtype=dtype)
    sigma_near_t  = torch.as_tensor(sigma_near, device=device, dtype=dtype)
    softcap_k_t   = torch.as_tensor(softcap_k,  device=device, dtype=dtype)
    near_t        = torch.full_like(d, fill_value=near_radius)  # (N,)

    # 核函数（入参都为 Tensor）
    def k_far(x: torch.Tensor)  -> torch.Tensor: return torch.exp(- x / sigma_far_t)
    def k_near(x: torch.Tensor) -> torch.Tensor: return torch.exp(- (x / sigma_near_t) ** 2)

    base = 0.5 * k_far(d) + 0.5 * k_near(d)               # (N,)
    cap  = 0.5 * k_far(near_t) + 0.5 * k_near(near_t)     # (N,)

    # 软封顶：softmin(base, cap)
    # softcap = cap - (1/k) * log(1 + exp(k*(cap - base)))
    softcap = cap - (1.0 / softcap_k_t) * torch.log1p(torch.exp(softcap_k_t * (cap - base)))

    return softcap



def align_ee_object(
    env: ManagerBasedRLEnv,
    object_frame_cfg: SceneEntityCfg = SceneEntityCfg("object_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """
    三阶段结构保持不变：
      1) 姿态引导（远处也有梯度，近处更重）
      2) 姿态就绪后对齐 Z
      3) 近距内 XY 靠近 + Z 保持
    变化点：
      - 姿态分 s_ori 使用“角度高斯核（粗） × 角度高斯核（细）”
      - 当“最大轴角误差”进入微小阈值时，启用“精度放大 + 精度奖励”以逼近 0 误差
    """
    eps = 1e-8
    def unit(v):  # (N,3)->(N,3)
        return v / (v.norm(dim=-1, keepdim=True).clamp_min(eps))

    # --- 取姿态/位置 ---
    object_frame: FrameTransformer = env.scene[object_frame_cfg.name]
    object: RigidObject           = env.scene[object_cfg.name]
    ee_frame: FrameTransformer    = env.scene[ee_frame_cfg.name]

    ee_q  = ee_frame.data.target_quat_w[..., 0, :]
    obj_q = object_frame.data.target_quat_w[..., 0, :]
    R_ee  = matrix_from_quat(ee_q)     # (N,3,3)
    R_obj = matrix_from_quat(obj_q)    # (N,3,3)

    obj_pos = object.data.root_pos_w                  # (N,3)
    ee_pos  = ee_frame.data.target_pos_w[..., 0, :]  # (N,3)
    dpos    = obj_pos - ee_pos
    zdist   = dpos[:, 2].abs()                       # |Δz|
    xydist  = torch.norm(dpos[:, :2], dim=1)         # √(dx^2+dy^2)

    # --- 轴向对齐（ey↔oy, ex↔oz, ez↔-ox）---
    ox, oy, oz = unit(R_obj[..., 0]), unit(R_obj[..., 1]), unit(R_obj[..., 2])
    ex, ey, ez = unit(R_ee[..., 0]),  unit(R_ee[..., 1]),  unit(R_ee[..., 2])

    # 余弦
    ay = (ey * oy).sum(-1).clamp(-1 + 1e-8, 1 - 1e-8)
    ax = (ex * oz).sum(-1).clamp(-1 + 1e-8, 1 - 1e-8)
    az = (ez * (-ox)).sum(-1).clamp(-1 + 1e-8, 1 - 1e-8)
    # 角度（弧度）
    thx = torch.acos(ax)
    thy = torch.acos(ay)
    thz = torch.acos(az)
    theta_max = torch.max(torch.stack([thx, thy, thz], dim=-1), dim=-1).values  # 三轴最大角误差（精度门用）

    # --- 训练进度（0→1）---
    prog_val = getattr(env, "train_progress", 0.0)
    # 把标量进度变成张量并对齐 dtype/device
    prog = torch.clamp(torch.as_tensor(prog_val, device=ax.device, dtype=ax.dtype), 0.0, 1.0)

    # ===================== 核心：双尺度角度核 + 精度门 =====================
    deg = 3.141592653589793 / 180.0

    # 粗尺度（远场/中场）：60° → 12°
    theta0 = (60.0 - 48.0 * prog) * deg
    sx0 = torch.exp(- (thx / theta0) ** 2)
    sy0 = torch.exp(- (thy / theta0) ** 2)
    sz0 = torch.exp(- (thz / theta0) ** 2)
    s_coarse = (sx0 * sy0 * sz0).pow(1.0 / 3.0).clamp(0.0, 1.0)

    # 细尺度（近场“显微镜”）：6° → 1.5°
    theta1 = (6.0 - 4.5 * prog).clamp_min(1.5) * deg
    sx1 = torch.exp(- (thx / theta1) ** 2)
    sy1 = torch.exp(- (thy / theta1) ** 2)
    sz1 = torch.exp(- (thz / theta1) ** 2)
    s_fine = (sx1 * sy1 * sz1).pow(1.0 / 3.0).clamp(0.0, 1.0)

    # 精度门：当最大轴角进入小阈值时，加大细尺度权重（平滑）
    # 阈值：5° → 2°；陡峭度：20 → 40
    th_gate = (5.0 - 3.0 * prog) * deg
    k_gate  = 20.0 + 20.0 * prog
    g_prec  = torch.sigmoid(k_gate * (th_gate - theta_max))  # ∈(0,1)，角度越小越接近 1

    # 融合：远/中距用 s_coarse，靠近后逐步转向 s_fine
    s_ori = ((1.0 - g_prec) * s_coarse + g_prec * s_fine).clamp(0.0, 1.0)

    # 精度奖励：当所有轴都很小（max 角误差<θ*），给一个小但尖锐的 bonus，逼近 0 误差
    # θ*：3° → 1°；bonus 强度随进度线性上升
    th_star = (3.0 - 2.0 * prog) * deg
    over    = (th_star - theta_max).clamp_min(0.0)          # 只在阈值内为正
    r_prec  = (over / th_star.clamp_min(1e-6)) ** 2         # 平滑二次井
    r_prec  = (0.2 + 0.6 * prog) * r_prec                   # 进度越晚，bonus 越强
    # ======================================================================

    # --- 软门：姿态、Z、距离(0.3m处开启精调) ---
    # 姿态门：门槛略收紧（你原来 0.65 → 0.97 太严，这里 0.60→0.92）
    c_ori = 0.60 + 0.32 * prog
    k_ori = 6.0  + 19.0 * prog
    g_ori = torch.sigmoid(k_ori * (s_ori - c_ori))

    # Z 门：0.15→0.03(m)，陡峭度 10→60
    z_ok  = 0.15 - 0.12 * prog
    k_z   = 10.0 + 50.0 * prog
    g_z   = torch.sigmoid(k_z * (z_ok - zdist))

    # 距离门：在 0.30m 内显著开启精调
    g_dist = torch.sigmoid(20.0 * (0.30 - xydist))

    # XY 阶段门：需 姿态就绪 & Z 就绪 & 到近距离
    g_xy = (g_ori * g_z * g_dist).clamp(0, 1)

    # --- 平滑核 ---
    def invquad(d, s):  # 1/(1+(d/s)^2)
        return 1.0 / (1.0 + (d / s) ** 2)
    def gauss(d, s):    # exp(-(d/s)^2)
        return torch.exp(- (d / s) ** 2)

    # --- 阶段2：Z 对齐（姿态门开后）---
    r_z_align = gauss(zdist.abs(), s=0.03)

    # --- 阶段3：XY 靠近 + Z 保持（后期更严）---
    r_xy     = invquad(xydist, s=(0.12 - 0.04 * prog))                     # 远处也有拉力，后期收紧
    r_z_hold = gauss(zdist.abs(),  s=(0.020 - 0.006 * prog)).clamp_min(1e-8)

    # 姿态回退惩罚（仅在 XY 阶段生效）
    pen_ori_back = g_xy * (1.0 - s_ori)

    # --- 早期暖启动：姿态引导下的靠近（随进度衰减）---
    w_warm = 0.50 * (1.0 - prog)   # 强一点，确保早期有正梯度
    warm_close = w_warm * s_ori * invquad(xydist, s=0.12) * gauss(zdist.abs(), s=0.06)

    # --- 反靠近：仅近距离触发，防错姿态贴脸（随进度增强）---
    near_gate   = torch.sigmoid(40.0 * (0.12 - xydist))  # ~12cm 内明显
    repel_near  = gauss(xydist, s=0.08) * gauss(zdist, s=0.06)
    w_anti = 0.03 + 0.57 * prog
    anti_approach = w_anti * (1.0 - g_ori) * near_gate * repel_near

    # --- 距离处的姿态权重：远处也鼓励朝向正确，近处权重更高 ---
    R_ori = (0.5 + 0.5 * g_dist) * s_ori + r_prec  # 把精度 bonus 融进去

    # --- 汇总权重（可按需微调） ---
    w_ori, w_z_align, w_xy, w_z_hold, w_back = 1.2, 0.7, 1.0, 1.0, 0.5  # 略提 w_ori，突出姿态精度

    reward = (
        w_ori * R_ori                                  # 姿态（近处更重要，含精度奖励）
      + g_ori * w_z_align * r_z_align                  # 阶段2：Z 对齐
      + g_xy  * (w_xy * r_xy + w_z_hold * r_z_hold)    # 阶段3：XY 靠近 + Z 保持
      - g_xy  * (w_back * pen_ori_back)                # 阶段3：姿态回退惩罚
      + warm_close                                     # 早期暖启动
      - anti_approach                                  # 近距错姿态反靠近
    )
    return reward



# def align_ee_object(
#     env: "ManagerBasedRLEnv",
#     object_frame_cfg: "SceneEntityCfg" = SceneEntityCfg("object_frame"),
#     object_cfg: "SceneEntityCfg" = SceneEntityCfg("object"),
#     ee_frame_cfg: "SceneEntityCfg" = SceneEntityCfg("ee_frame"),
# ) -> torch.Tensor:
#     """
#     三阶段 + 抓取窗口内的超精细姿态微调（无角速度惩罚版本）
#     """
#     eps = 1e-8
#     def unit(v):  # (N,3)->(N,3)
#         return v / (v.norm(dim=-1, keepdim=True).clamp_min(eps))

#     # --- 取姿态/位置 ---
#     object_frame: "FrameTransformer" = env.scene[object_frame_cfg.name]
#     object: "RigidObject"            = env.scene[object_cfg.name]
#     ee_frame: "FrameTransformer"     = env.scene[ee_frame_cfg.name]

#     ee_q  = ee_frame.data.target_quat_w[..., 0, :]
#     obj_q = object_frame.data.target_quat_w[..., 0, :]
#     R_ee  = matrix_from_quat(ee_q)     # (N,3,3)
#     R_obj = matrix_from_quat(obj_q)    # (N,3,3)

#     obj_pos = object.data.root_pos_w                  # (N,3)
#     ee_pos  = ee_frame.data.target_pos_w[..., 0, :]  # (N,3)
#     dpos    = obj_pos - ee_pos
#     zdist   = dpos[:, 2].abs()
#     xydist  = torch.norm(dpos[:, :2], dim=1)

#     # --- 轴向对齐（ey↔oy, ex↔oz, ez↔-ox）---
#     ox, oy, oz = unit(R_obj[..., 0]), unit(R_obj[..., 1]), unit(R_obj[..., 2])
#     ex, ey, ez = unit(R_ee[..., 0]),  unit(R_ee[..., 1]),  unit(R_ee[..., 2])

#     ay = (ey * oy).sum(-1).clamp(-1 + 1e-8, 1 - 1e-8)
#     ax = (ex * oz).sum(-1).clamp(-1 + 1e-8, 1 - 1e-8)
#     az = (ez * (-ox)).sum(-1).clamp(-1 + 1e-8, 1 - 1e-8)

#     thx = torch.acos(ax)
#     thy = torch.acos(ay)
#     thz = torch.acos(az)
#     theta_max = torch.max(torch.stack([thx, thy, thz], dim=-1), dim=-1).values

#     # --- 训练进度（0→1）---
#     prog_val = getattr(env, "train_progress", 0.0)
#     prog = torch.clamp(torch.as_tensor(prog_val, device=ax.device, dtype=ax.dtype), 0.0, 1.0)

#     # ===================== 双尺度角度核 + 精度门（整体姿态） =====================
#     deg = 3.141592653589793 / 180.0

#     # 粗尺度：60° → 12°
#     theta0 = (60.0 - 48.0 * prog) * deg
#     s_coarse = (torch.exp(-(thx/theta0)**2) *
#                 torch.exp(-(thy/theta0)**2) *
#                 torch.exp(-(thz/theta0)**2)).pow(1/3).clamp(0,1)

#     # 细尺度：6° → 1.5°
#     theta1 = (6.0 - 4.5 * prog).clamp_min(1.5) * deg
#     s_fine = (torch.exp(-(thx/theta1)**2) *
#               torch.exp(-(thy/theta1)**2) *
#               torch.exp(-(thz/theta1)**2)).pow(1/3).clamp(0,1)

#     # 精度门（5° → 2°）
#     th_gate = (5.0 - 3.0 * prog) * deg
#     k_gate  = 20.0 + 20.0 * prog
#     g_prec  = torch.sigmoid(k_gate * (th_gate - theta_max))

#     s_ori = ((1.0 - g_prec) * s_coarse + g_prec * s_fine).clamp(0.0, 1.0)

#     # 整体精度奖励（3° → 1°）
#     th_star = (3.0 - 2.0 * prog) * deg
#     over    = (th_star - theta_max).clamp_min(0.0)
#     r_prec  = (over / th_star.clamp_min(1e-6)) ** 2
#     r_prec  = (0.2 + 0.6 * prog) * r_prec

#     # --- 软门：姿态、Z、距离 ---
#     c_ori = 0.60 + 0.32 * prog
#     k_ori = 6.0  + 19.0 * prog
#     g_ori = torch.sigmoid(k_ori * (s_ori - c_ori))

#     z_ok  = 0.15 - 0.12 * prog
#     k_z   = 10.0 + 50.0 * prog
#     g_z   = torch.sigmoid(k_z * (z_ok - zdist))

#     g_dist = torch.sigmoid(20.0 * (0.30 - xydist))
#     g_xy   = (g_ori * g_z * g_dist).clamp(0, 1)

#     # --- 平滑核 ---
#     def invquad(d, s):  # 1/(1+(d/s)^2)
#         return 1.0 / (1.0 + (d / s) ** 2)
#     def gauss(d, s):    # exp(-(d/s)^2)
#         return torch.exp(- (d / s) ** 2)

#     # --- 阶段2：Z 对齐（姿态门开后）---
#     r_z_align = gauss(zdist.abs(), s=0.03)

#     # --- 阶段3：XY 靠近 + Z 保持（后期更严）---
#     r_xy     = invquad(xydist, s=(0.12 - 0.04 * prog))
#     r_z_hold = gauss(zdist.abs(),  s=(0.020 - 0.006 * prog)).clamp_min(1e-8)

#     # 姿态回退惩罚（仅在 XY 阶段生效）
#     pen_ori_back = g_xy * (1.0 - s_ori)

#     # 早期暖启动
#     w_warm = 0.50 * (1.0 - prog)
#     warm_close = w_warm * s_ori * invquad(xydist, s=0.12) * gauss(zdist.abs(), s=0.06)

#     # 近距错姿态反靠近
#     near_gate   = torch.sigmoid(40.0 * (0.12 - xydist))
#     repel_near  = gauss(xydist, s=0.08) * gauss(zdist, s=0.06)
#     w_anti = 0.03 + 0.57 * prog
#     anti_approach = w_anti * (1.0 - g_ori) * near_gate * repel_near

#     # 距离处的姿态权重
#     R_ori = (0.5 + 0.5 * g_dist) * s_ori + r_prec

#     # ====== 抓取窗口内的“超精细姿态微调”（无角速度项） ======
#     g_refine = g_xy * torch.sigmoid(30.0 * (0.10 - xydist)) * torch.sigmoid(30.0 * (0.03 - zdist))

#     theta_ultra = (2.0 - 1.7 * prog).clamp_min(0.3) * deg
#     s_ultra = torch.exp(- (theta_max / theta_ultra) ** 2)

#     th_zero = (1.5 - 1.0 * prog).clamp_min(0.5) * deg
#     over0   = (th_zero - theta_max).clamp_min(0.0)
#     r_ultra = (over0 / th_zero.clamp_min(1e-6)) ** 2

#     w_ultra, w_zero = 1.2, 0.8
#     R_refine = g_refine * (w_ultra * s_ultra + w_zero * r_ultra)

#     # --- 汇总 ---
#     w_ori, w_z_align, w_xy, w_z_hold, w_back = 1.2, 0.7, 1.0, 1.0, 0.5
#     reward = (
#         w_ori * (R_ori + R_refine)
#       + g_ori * w_z_align * r_z_align
#       + g_xy  * (w_xy * r_xy + w_z_hold * r_z_hold)
#       - g_xy  * (w_back * pen_ori_back)
#       + warm_close
#       - anti_approach
#     )
#     return reward




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
    open_joint_pos: float,             # 若有每指 open/closed，仍按此签名传入标量即可；下方做广播
    asset_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_frame_cfg: SceneEntityCfg = SceneEntityCfg("object_frame"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    gripper_frame_cfg: SceneEntityCfg = SceneEntityCfg("gripper_frame"),
) -> torch.Tensor:
    """
    连续就绪门控：就绪后奖励闭合；未就绪早闭轻罚。用物体系轴判“上/下”，
    并按每指上下限归一化闭合度，避免 scale/符号问题。
    """
    eps = 1e-6
    object: RigidObject         = env.scene[object_cfg.name]
    robot:  Articulation        = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer  = env.scene[ee_frame_cfg.name]
    gripper: FrameTransformer   = env.scene[gripper_frame_cfg.name]

    # --- 目标与末端位置（尽量用“夹持中心”，此处沿用 ee_frame） ---
    obj_pos = object.data.root_pos_w                          # (N,3) 实际物体位置
    ee_pos  = ee_frame.data.target_pos_w[..., 0, :]           # (N,3) 末端位置
    dist    = torch.norm(obj_pos - ee_pos, dim=-1)            # (N,)

    # --- 距离软门：越近越接近 1 ---
    sharp_d = 40.0
    gate_d  = torch.sigmoid(sharp_d * (threshold - dist))     # (N,)

    # --- 用物体系轴判“上/下”（避免世界系 z 歧义） ---
    obj_q   = getattr(object.data, "root_quat_w", object.data.root_quat_w)[..., 0, :]  # (N,4)
    R_obj   = matrix_from_quat(obj_q)                         # (N,3,3)
    u       = R_obj[..., 2]                                   # 抓取法向：obj_z（如有需要换 obj_x/obj_y）

    f1 = gripper.data.target_pos_w[..., 0, :]                 # (N,3) 指尖/指端帧
    f3 = gripper.data.target_pos_w[..., 1, :]
    f4 = gripper.data.target_pos_w[..., 2, :]
    f6 = gripper.data.target_pos_w[..., 3, :]

    def signed_height(p):                                     # 沿 u 的有符号高度
        return ((p - obj_pos) * u).sum(dim=-1)

    h1, h3, h4, h6 = map(signed_height, (f1, f3, f4, f6))

    sharp_h = 60.0
    up_ok   = torch.sigmoid(sharp_h *  h1) * torch.sigmoid(sharp_h *  h3)   # 两上指在“上”
    down_ok = torch.sigmoid(sharp_h * -h4) * torch.sigmoid(sharp_h * -h6)   # 两下指在“下”
    gate_h  = (up_ok * down_ok).clamp(0, 1)
    gate    = (gate_h * gate_d).clamp(0, 1)                                  # 综合就绪度 ∈ (0,1)

    # --- 闭合度：按每指上下限归一化到 [0,1]（1=完全闭合） ---
    q          = robot.data.joint_pos[:, asset_cfg.joint_ids]                # (N,J)
    open_pos   = torch.as_tensor(open_joint_pos, device=q.device, dtype=q.dtype)
    if open_pos.ndim == 0:                                                   # 标量 → 广播
        open_pos = open_pos.expand_as(q)
    closed_pos = torch.zeros_like(q)                                         # 若 0≠闭合，请改成真实 closed_pos
    denom      = (open_pos - closed_pos).abs().clamp_min(eps)
    open_frac  = ((q - closed_pos) / denom).clamp(0.0, 1.0)                  # 0=闭, 1=开
    close_frac = 1.0 - open_frac                                             # 1=闭
    close_frac = close_frac.mean(dim=-1)                                     # (N,)

    # --- 对称性（可选）：鼓励两上/两下高度接近，抑制单侧先夹 ---
    def balance(a, b, sigma=0.01):                                           # ∈(0,1]
        return torch.exp(-((a - b) / sigma) ** 2)
    tips_balance = (balance(h1, h3) * balance(h4, h6)).sqrt()

    # --- 就绪后奖励闭合；未就绪早闭轻罚（避免“到位也不闭”） ---
    w_grasp = 1.0
    w_early = 0.3
    w_sym   = 0.2

    grasp_intent    = gate * close_frac
    early_close_pen = (1.0 - gate) * close_frac

    reward = (
        w_grasp * grasp_intent
        - w_early * early_close_pen
        + w_sym   * gate * tips_balance
    )
    return reward


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

#===============================================================================test
#===============================================================================
# def grasp_object(
#     env: ManagerBasedRLEnv,
#     threshold: float,                 # 未用，占位
#     open_joint_pos: float,            # 夹爪“完全张开”的关节位置
#     asset_cfg: SceneEntityCfg,
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
#     robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     object_frame_cfg: SceneEntityCfg = SceneEntityCfg("object_frame"),
#     ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
#     gripper_frame_cfg: SceneEntityCfg = SceneEntityCfg("gripper_frame"),
#     target_open_frac: float = 0.20,   # 目标开口比例（0=闭,1=开）
# ) -> torch.Tensor:
#     """
#     极简测试：奖励=跟踪目标开口比例（越接近 target_open_frac 越高）。
#     """
#     eps = 1e-6
#     robot: Articulation = env.scene[robot_cfg.name]
#     grip_ids = asset_cfg.joint_ids
#     q_g = robot.data.joint_pos[:, grip_ids]            # (N, Jg)

#     open_pos = torch.as_tensor(open_joint_pos, device=q_g.device, dtype=q_g.dtype)
#     if open_pos.ndim == 0:
#         open_pos = open_pos.expand_as(q_g)
#     closed_pos = torch.zeros_like(q_g)                 # 若 0 不是闭合位置请替换
#     denom = (open_pos - closed_pos).abs().clamp_min(eps)
#     open_frac = ((q_g - closed_pos) / denom).clamp(0.0, 1.0)   # (N, Jg)

#     # 误差（对所有夹爪 DOF 取平均）
#     err = (open_frac - float(target_open_frac)).mean(dim=-1)   # (N,)
#     # 用高斯核把误差转为奖励（0~1），误差±0.1 仍有 ~0.61 的得分
#     sigma = 0.10
#     reward = torch.exp(- (err / sigma) ** 2)
#     return reward
