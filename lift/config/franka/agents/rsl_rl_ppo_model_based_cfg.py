# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL configuration for the model-based (plan C) lift task."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticRecurrentCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class LiftCubeModelBasedRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Runner settings tuned for the model-based variant."""

    num_steps_per_env = 32
    max_iterations = 1200
    save_interval = 50
    experiment_name = "franka_lift_plan_c"
    empirical_normalization = True

    policy = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=0.4,
        actor_hidden_dims=[256],
        critic_hidden_dims=[256],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=128,
        rnn_num_layers=1,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.2,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=6,
        num_mini_batches=4,
        learning_rate=2.0e-4,
        schedule="cosine",
        gamma=0.985,
        lam=0.95,
        desired_kl=0.015,
        max_grad_norm=0.7,
    )
