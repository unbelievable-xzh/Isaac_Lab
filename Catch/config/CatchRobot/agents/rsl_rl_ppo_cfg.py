# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class CatchCubePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # rollout 更长一点
    num_steps_per_env = 24         # 从 24 -> 64
    max_iterations = 1500          # 先不改；是否增加看你要的总步数
    save_interval = 100
    experiment_name = "catch_cube"
    empirical_normalization = True  # 开启经验归一化（更稳）

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=3.0,
        actor_hidden_dims=[256, 256, 128],   # 稍加宽；若算力紧张可回到 [256,128,64]
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=1e-3,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.02,
        max_grad_norm=1.0,
    )
