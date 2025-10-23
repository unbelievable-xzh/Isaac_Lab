# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    # 关键：引入循环策略配置
    RslRlPpoActorCriticRecurrentCfg,
)

@configclass
class LiftCubePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # RNN 建议更长的序列步长，便于捕捉时序
    num_steps_per_env = 24              # ← 原来 24，建议 48~128 之间
    max_iterations = 2000
    save_interval = 200
    experiment_name = "franka_lift_lstm"
    empirical_normalization = True      # RNN 更建议开启归一化

    # ------ 改成 LSTM 策略 ------
    policy = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=0.5,             # RNN 初期探索不必太大
        actor_hidden_dims=[256],        # 共享前馈层 -> LSTM -> 双头
        critic_hidden_dims=[256],
        activation="elu",
        rnn_type="lstm",                # 也可 "gru"
        rnn_hidden_dim=128,             # 128/256 常用
        rnn_num_layers=1,
    )

    # ------ PPO 超参微调（更稳） ------
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.5,            # 价值更稳一些
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,             # RNN 略小的熵，防止过度随机
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.5e-4,           # RNN 稍降学习率有助稳定
        schedule="cosine",              # "adaptive"/"fixed"/"cosine" 均可
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=0.8,              # RNN 建议更紧的梯度裁剪
    )
