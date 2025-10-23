"""Observation helpers for the lift task.

This module exposes three flavours of the object pose observation that are used by
the policy and value networks:

* :func:`object_position_in_robot_root_frame` returns the ground-truth pose in the
  robot root frame.  This is primarily consumed by the critic during training.
* :func:`object_position_in_robot_root_frame_noisy` adds Gaussian noise through a
  reusable noise generator, emulating a corrupted sensor measurement.
* :func:`object_position_in_robot_root_frame_kf` filters the noisy measurement
  using a batched Kalman filter to produce a smoothed estimate.

The concrete noise and Kalman filter implementations live in
``noise_generators.py`` and ``kalman_filters.py`` respectively.  Both helpers are
cached on the environment instance so that they maintain temporal state across
simulation steps and across parallel environments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

from .kalman_filters import BatchPosVelKalmanFilter
from .noise_generators import GaussianPositionNoise

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _resolve_scene_entities(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
) -> tuple[RigidObject, RigidObject]:
    """Returns the robot and object assets from the scene."""

    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    return robot, obj


def _object_position_in_robot_frame(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Computes the ground-truth object position in the robot root frame."""

    robot, obj = _resolve_scene_entities(env, robot_cfg, object_cfg)
    object_pos_w = obj.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w
    )
    return object_pos_b


def _get_noise_generator(
    env: "ManagerBasedRLEnv",
    pos_sigma: float,
) -> GaussianPositionNoise:
    """Fetches (or creates) the cached Gaussian noise generator."""

    generator: Optional[GaussianPositionNoise] = getattr(env, "_objpos_noise_gen", None)
    if generator is None:
        generator = GaussianPositionNoise(pos_sigma)
        setattr(env, "_objpos_noise_gen", generator)
    else:
        generator.sigma = pos_sigma
    return generator


def _get_kalman_filter(
    env: "ManagerBasedRLEnv",
    pos_sigma: float,
    q_pos: float,
    q_vel: float,
) -> BatchPosVelKalmanFilter:
    """Fetches (or creates) the cached batched Kalman filter instance."""

    kf: Optional[BatchPosVelKalmanFilter] = getattr(env, "_objpos_kf", None)
    if kf is None:
        dt = float(getattr(env, "step_dt", 0.02))
        kf = BatchPosVelKalmanFilter(
            num_envs=env.num_envs,
            dt=dt,
            pos_sigma=pos_sigma,
            q_pos=q_pos,
            q_vel=q_vel,
            device=env.device,
        )
        setattr(env, "_objpos_kf", kf)
    else:
        kf.update_process_noise(pos_sigma=pos_sigma, q_pos=q_pos, q_vel=q_vel)
    return kf


def _env_is_in_eval_mode(env: "ManagerBasedRLEnv") -> bool:
    """Returns ``True`` when the environment is running in evaluation mode."""

    return bool(getattr(env, "is_testing", False) or getattr(env, "is_playing", False))


def _reset_kalman_filter_if_needed(
    env: "ManagerBasedRLEnv", kf: BatchPosVelKalmanFilter
) -> None:
    """Resets individual filters for environments that terminated this step."""

    done_mask: Optional[torch.Tensor]
    if hasattr(env, "reset_buf"):
        done_mask = env.reset_buf.bool()
    elif hasattr(env, "episode_length_buf"):
        done_mask = env.episode_length_buf == 0
    else:
        done_mask = None
    kf.reset(done_mask)


# -----------------------------------------------------------------------------
# Observation functions
# -----------------------------------------------------------------------------

def object_position_in_robot_root_frame(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Ground-truth object position expressed in the robot root frame."""

    return _object_position_in_robot_frame(env, robot_cfg, object_cfg)


def object_position_in_robot_root_frame_noisy(
    env: "ManagerBasedRLEnv",
    pos_sigma: float = 0.003,
    use_during_evaluation: bool = False,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Noisy object position in the robot root frame.

    The helper corrupts the ground-truth measurement using a Gaussian noise
    generator.  The generated noisy measurement is cached on the environment and
    subsequently consumed by the Kalman-filtered observation to ensure both
    observation heads see the exact same measurement sample.
    """

    clean_pos = _object_position_in_robot_frame(env, robot_cfg, object_cfg)
    if pos_sigma <= 0 or (_env_is_in_eval_mode(env) and not use_during_evaluation):
        noisy_pos = clean_pos
    else:
        noise_gen = _get_noise_generator(env, pos_sigma)
        noisy_pos = noise_gen(clean_pos)

    setattr(env, "_objpos_meas_b", noisy_pos)
    return noisy_pos


def object_position_in_robot_root_frame_kf(
    env: "ManagerBasedRLEnv",
    pos_sigma: float = 0.003,
    q_pos: float = 1e-5,
    q_vel: float = 1e-4,
    use_during_evaluation: bool = False,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Kalman-filtered object position in the robot root frame."""

    measurement = getattr(env, "_objpos_meas_b", None)
    if measurement is None:
        clean_pos = _object_position_in_robot_frame(env, robot_cfg, object_cfg)
        if pos_sigma <= 0 or (_env_is_in_eval_mode(env) and not use_during_evaluation):
            measurement = clean_pos
        else:
            noise_gen = _get_noise_generator(env, pos_sigma)
            measurement = noise_gen(clean_pos)

    kf = _get_kalman_filter(env, pos_sigma, q_pos, q_vel)
    _reset_kalman_filter_if_needed(env, kf)
    filtered_pos = kf.step(measurement)
    setattr(env, "_objpos_meas_b", measurement)
    return filtered_pos

