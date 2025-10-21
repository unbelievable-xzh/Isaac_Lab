# obs_kf.py
from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms
from typing import Optional
from isaaclab.envs import ManagerBasedRLEnv
# ===== 批量卡尔曼滤波：x=[p(3), v(3)]，量测 z=p(3) =====
class BatchPosVelKF:
    """
    批量KF：状态 x=[p(3), v(3)], 量测 z=p(3)。
    仅依赖带噪观测 z_t，不用真值。支持 num_envs 并行、GPU 运行。
    """
    def __init__(self, num_envs, dt, pos_sigma, q_pos=1e-5, q_vel=1e-4, device="cuda"):
        self.N = num_envs
        self.dt = dt
        self.device = device

        I3 = torch.eye(3, device=device)

        # 状态转移 A (6x6)，量测矩阵 H (3x6)
        self.A = torch.block_diag(I3, I3)
        self.A[:3, 3:] = I3 * dt
        self.H = torch.cat([torch.eye(3, device=device), torch.zeros(3,3, device=device)], dim=1)

        # 协方差
        self.Q = torch.block_diag(I3 * q_pos, I3 * q_vel)           # 过程噪声
        self.R = torch.eye(3, device=device) * (pos_sigma ** 2)     # 量测噪声

        # 批量状态与协方差
        self.x = torch.zeros(self.N, 6, device=device)               # 初始 p,v=0
        self.P = torch.eye(6, device=device)[None].repeat(self.N,1,1) * 1e-2

    @torch.no_grad()
    def reset(self, mask: Optional[torch.Tensor] = None):
        """对 done 的 env 重置滤波器（mask: (N,) 的bool）。"""
        if mask is None:
            self.x.zero_()
            self.P.copy_(torch.eye(6, device=self.device)[None].repeat(self.N,1,1) * 1e-2)
            return
        if mask.any():
            idx = mask.nonzero(as_tuple=False).squeeze(-1)
            self.x[idx] = 0.0
            self.P[idx] = torch.eye(6, device=self.device) * 1e-2

    @torch.no_grad()
    def step(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (N,3) 带噪位置观测
        返回：滤波后的估计位置 (N,3)
        """
        A, H, Q, R = self.A, self.H, self.Q, self.R
        x, P = self.x, self.P

        # 预测
        x_pred = x @ A.T                              # (N,6)
        P_pred = A @ P @ A.T + Q                      # (N,6,6)

        # 卡尔曼增益
        S = H @ P_pred @ H.T + R                      # (N,3,3)
        S_inv = torch.inverse(S)
        K = P_pred @ H.T @ S_inv                      # (N,6,3)

        # 更新
        y = z - (x_pred @ H.T)                        # (N,3)
        x_new = x_pred + (K @ y.unsqueeze(-1)).squeeze(-1)
        I6 = torch.eye(6, device=z.device)
        P_new = (I6 - K @ H) @ P_pred

        self.x = x_new
        self.P = P_new
        return x_new[:, :3]

# ===== 观测函数：Actor 用 noisy，Critic 用 KF =====
def object_position_in_robot_root_frame_noisy_simple(
    env: ManagerBasedRLEnv,
    pos_sigma: float = 0.01,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """极简带噪：训练加高斯噪，评估自动关；返回 (N,3)"""
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    object_pos_w = obj.data.root_pos_w[:, :3]
    clean_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w)

    # 评估/回放关噪（要做带噪评估就注释掉下面两行）
    if getattr(env, "is_testing", False) or getattr(env, "is_playing", False):
        noisy = clean_b
    else:
        noisy = clean_b + torch.randn_like(clean_b) * pos_sigma

    # （可选）把这一步的“测量”缓存，KF 可复用，避免双重加噪
    setattr(env, "_objpos_meas_b", noisy)
    return noisy

def object_position_in_robot_root_frame_kf(
    env: ManagerBasedRLEnv,
    pos_sigma: float = 0.01,
    q_pos: float = 1e-5,
    q_vel: float = 1e-4,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """KF 滤波后的 (N,3) 位置（只用带噪观测，不用真值）"""
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    object_pos_w = obj.data.root_pos_w[:, :3]
    z_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w)

    # 先用 Actor 缓存的“测量”；若没有，就自己加噪
    meas = getattr(env, "_objpos_meas_b", None)
    if meas is None:
        meas = z_b + torch.randn_like(z_b) * pos_sigma

    # 取/建 KF（避免 IDE 红线，用 getattr/setattr）
    kf: Optional[BatchPosVelKF] = getattr(env, "_objpos_kf", None)
    if kf is None:
        dt = float(getattr(env, "step_dt", 0.02))  # 你的 env 步长
        kf = BatchPosVelKF(num_envs=env.num_envs, dt=dt, pos_sigma=pos_sigma,
                           q_pos=q_pos, q_vel=q_vel, device=env.device)
        setattr(env, "_objpos_kf", kf)

    # 回合重置：按你任务中的 done 标志选一个
    if hasattr(env, "reset_buf"):
        done_mask = env.reset_buf.bool()
    else:
        done_mask = (env.episode_length_buf == 0) if hasattr(env, "episode_length_buf") else None
    kf.reset(done_mask)

    # KF 更新
    p_est = kf.step(meas)
    return p_est
