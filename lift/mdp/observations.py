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
    *,
    bias_sigma: float = 0.0,
    drift_sigma: float = 0.0,
    rot_bias_deg: float = 0.0,
    scale_err: float = 0.0,
    quant_step: float = 0.0,
    dropout_p: float = 0.0,
    delay_steps: int = 0,
    jitter_range: int = 0,
    low_fps_hold: int = 1,
) -> GaussianPositionNoise:
    """Fetches (or creates) the cached position noise generator."""

    generator: Optional[GaussianPositionNoise] = getattr(env, "_objpos_noise_gen", None)
    if generator is None:
        generator = GaussianPositionNoise(
            env,
            pos_sigma=pos_sigma,
            bias_sigma=bias_sigma,
            drift_sigma=drift_sigma,
            rot_bias_deg=rot_bias_deg,
            scale_err=scale_err,
            quant_step=quant_step,
            dropout_p=dropout_p,
            delay_steps=delay_steps,
            jitter_range=jitter_range,
            low_fps_hold=low_fps_hold,
        )
        setattr(env, "_objpos_noise_gen", generator)
    else:
        generator.update_parameters(
            pos_sigma=pos_sigma,
            bias_sigma=bias_sigma,
            drift_sigma=drift_sigma,
            rot_bias_deg=rot_bias_deg,
            scale_err=scale_err,
            quant_step=quant_step,
            dropout_p=dropout_p,
            delay_steps=delay_steps,
            jitter_range=jitter_range,
            low_fps_hold=low_fps_hold,
        )
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
    bias_sigma: float = 0.0,
    drift_sigma: float = 0.0,
    rot_bias_deg: float = 0.0,
    scale_err: float = 0.0,
    quant_step: float = 0.0,
    dropout_p: float = 0.0,
    delay_steps: int = 0,
    jitter_range: int = 0,
    low_fps_hold: int = 1,
    use_during_evaluation: bool = False,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Noisy object position in the robot root frame.

    The helper corrupts the ground-truth measurement using a configurable noise
    generator.  In addition to zero-mean Gaussian jitter, the generator can
    inject per-episode bias, random-walk drift, rotation and scale errors,
    quantisation, dropouts, delays and low frame-rate behaviour.  The generated
    noisy measurement is cached on the environment and subsequently consumed by
    the Kalman-filtered observation to ensure both observation heads see the
    exact same measurement sample.
    """

    clean_pos = _object_position_in_robot_frame(env, robot_cfg, object_cfg)
    noise_disabled = (
        pos_sigma <= 0
        and bias_sigma <= 0
        and drift_sigma <= 0
        and rot_bias_deg == 0
        and scale_err == 0
        and quant_step == 0
        and dropout_p <= 0
        and delay_steps == 0
        and jitter_range == 0
        and low_fps_hold <= 1
    )
    if noise_disabled or (_env_is_in_eval_mode(env) and not use_during_evaluation):
        noisy_pos = clean_pos
    else:
        noise_gen = _get_noise_generator(
            env,
            pos_sigma,
            bias_sigma=bias_sigma,
            drift_sigma=drift_sigma,
            rot_bias_deg=rot_bias_deg,
            scale_err=scale_err,
            quant_step=quant_step,
            dropout_p=dropout_p,
            delay_steps=delay_steps,
            jitter_range=jitter_range,
            low_fps_hold=low_fps_hold,
        )
        noisy_pos = noise_gen(clean_pos)

    setattr(env, "_objpos_measurement", noisy_pos)
    return noisy_pos


def object_position_in_robot_root_frame_kf(
    env: "ManagerBasedRLEnv",
    pos_sigma: float = 0.003,
    bias_sigma: float = 0.0,
    drift_sigma: float = 0.0,
    rot_bias_deg: float = 0.0,
    scale_err: float = 0.0,
    quant_step: float = 0.0,
    dropout_p: float = 0.0,
    delay_steps: int = 0,
    jitter_range: int = 0,
    low_fps_hold: int = 1,
    q_pos: float = 1e-5,
    q_vel: float = 1e-4,
    use_during_evaluation: bool = False,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Kalman-filtered object position in the robot root frame."""

    measurement = getattr(env, "_objpos_measurement", None)
    if measurement is None:
        clean_pos = _object_position_in_robot_frame(env, robot_cfg, object_cfg)
        noise_disabled = (
            pos_sigma <= 0
            and bias_sigma <= 0
            and drift_sigma <= 0
            and rot_bias_deg == 0
            and scale_err == 0
            and quant_step == 0
            and dropout_p <= 0
            and delay_steps == 0
            and jitter_range == 0
            and low_fps_hold <= 1
        )
        if noise_disabled or (
            _env_is_in_eval_mode(env) and not use_during_evaluation
        ):
            measurement = clean_pos
        else:
            noise_gen = _get_noise_generator(
                env,
                pos_sigma,
                bias_sigma=bias_sigma,
                drift_sigma=drift_sigma,
                rot_bias_deg=rot_bias_deg,
                scale_err=scale_err,
                quant_step=quant_step,
                dropout_p=dropout_p,
                delay_steps=delay_steps,
                jitter_range=jitter_range,
                low_fps_hold=low_fps_hold,
            )
            measurement = noise_gen(clean_pos)

    kf = _get_kalman_filter(env, pos_sigma, q_pos, q_vel)
    _reset_kalman_filter_if_needed(env, kf)
    filtered_pos = kf.step(measurement)
    setattr(env, "_objpos_measurement", measurement)
    return filtered_pos

