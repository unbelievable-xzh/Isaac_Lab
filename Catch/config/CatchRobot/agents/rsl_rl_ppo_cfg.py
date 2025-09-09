# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class CatchCubePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # rollout 更长一点
    num_steps_per_env = 24          # 从 24 -> 64
    max_iterations = 1500           # 先不改；是否增加看你要的总步数
    save_interval = 50
    experiment_name = "catch_cube"
    empirical_normalization = True  # 开启经验归一化（更稳）

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 128],   # 稍加宽；若算力紧张可回到 [256,128,64]
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # 长时程更需要探索；前期略大，后期可调小
        entropy_coef=0.008,          # 0.006 -> 0.008（可在课程2/3阶段降回 0.004~0.006）
        num_learning_epochs=5,
        num_mini_batches=4,          # 保持；确保 batch 能整除
        learning_rate=1.0e-4,
        schedule="adaptive",
        # ★关键：折扣与时间尺度（50 Hz，每步 0.02 s）
        # γ=0.98 => 1/(1-0.98)=50 步 ≈ 1.0 s（太短）
        # γ=0.998=> 500 步 ≈ 10 s；γ=0.999=> 1000 步 ≈ 20 s
        gamma=0.998,                 # 推荐 0.998；若轨迹特别长可试 0.999
        lam=0.97,                    # 从 0.95 -> 0.97，长回合优势估计更稳
        desired_kl=0.01,             # 自适应 LR 的目标 KL，保持
        max_grad_norm=1.0,
    )
