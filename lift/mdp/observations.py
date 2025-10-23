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
from .dynamics_models import DynamicsEnsemble, create_ensemble, get_plan_c_dataset

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


# -----------------------------------------------------------------------------
# Model-based observation helpers (Plan C)
# -----------------------------------------------------------------------------


def _infer_action_tensor(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Best-effort extraction of the last applied action for each environment."""

    action: Optional[torch.Tensor] = None
    if hasattr(env, "action_manager"):
        manager = getattr(env, "action_manager")
        getter = getattr(manager, "get_last_actions", None)
        if callable(getter):
            try:
                action = getter()
            except TypeError:
                action = getter(env)  # type: ignore[misc]
    if action is None and hasattr(env, "_last_actions"):
        cached = getattr(env, "_last_actions")
        if isinstance(cached, torch.Tensor):
            action = cached
    if action is None:
        num_envs = getattr(env, "num_envs", 1)
        device = getattr(env, "device", torch.device("cpu"))
        action_dim = 0
        for name in ("num_actions", "action_dim", "action_size"):
            value = getattr(env, name, None)
            if isinstance(value, int) and value > 0:
                action_dim = value
                break
        if action_dim == 0 and hasattr(env, "action_manager"):
            manager = getattr(env, "action_manager")
            for name in ("num_actions", "action_dim", "action_size"):
                value = getattr(manager, name, None)
                if isinstance(value, int) and value > 0:
                    action_dim = value
                    break
        if action_dim == 0:
            action_dim = 1
        action = torch.zeros(num_envs, action_dim, device=device)
    return action


def _get_state_history(env: "ManagerBasedRLEnv", *, history_len: int, state_dim: int, device: torch.device) -> torch.Tensor:
    history: Optional[torch.Tensor] = getattr(env, "_plan_c_state_history", None)
    if history is None or history.shape[0] != env.num_envs or history.shape[1] != history_len:
        history = torch.zeros(env.num_envs, history_len, state_dim, device=device)
        setattr(env, "_plan_c_state_history", history)
    return history


def _get_dynamics_ensemble(
    env: "ManagerBasedRLEnv",
    *,
    state_dim: int,
    action_dim: int,
    history_len: int,
    ensemble_size: int,
    hidden_dim: int,
    learning_rate: float,
    momentum: float,
) -> DynamicsEnsemble:
    ensemble: Optional[DynamicsEnsemble] = getattr(env, "_plan_c_dynamics", None)
    if ensemble is None:
        ensemble = create_ensemble(
            state_dim=state_dim,
            action_dim=action_dim,
            history_len=history_len,
            ensemble_size=ensemble_size,
            device=env.device,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            momentum=momentum,
        )
        setattr(env, "_plan_c_dynamics", ensemble)
    return ensemble


def model_rollout_features(
    env: "ManagerBasedRLEnv",
    *,
    horizon: int = 5,
    history_len: int = 4,
    ensemble_size: int = 5,
    hidden_dim: int = 64,
    learning_rate: float = 5e-3,
    momentum: float = 0.05,
    soft_update: float = 0.1,
    log_transition: bool = True,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Returns model-based rollout features used by the plan-C policy."""

    current_state = object_position_in_robot_root_frame_kf(env, robot_cfg=robot_cfg, object_cfg=object_cfg)
    action = _infer_action_tensor(env).to(device=current_state.device)
    state_dim = current_state.shape[1]
    history = _get_state_history(env, history_len=history_len, state_dim=state_dim, device=current_state.device)
    prev_state: Optional[torch.Tensor] = getattr(env, "_plan_c_prev_state", None)
    prev_history = history.clone()

    dataset = get_plan_c_dataset(env)

    if prev_state is not None:
        if log_transition:
            dataset.push(prev_state, action, prev_history, current_state)
        ensemble = _get_dynamics_ensemble(
            env,
            state_dim=state_dim,
            action_dim=action.shape[1],
            history_len=history_len,
            ensemble_size=ensemble_size,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            momentum=momentum,
        )
        rollout_states, rollout_actions = ensemble.rollout(
            current_state, action, history, horizon=horizon, soft_update=soft_update
        )
        kf_params = ensemble.export_kf_parameters().repeat(env.num_envs, 1)
    else:
        rollout_states = current_state.unsqueeze(1).repeat(1, horizon, 1)
        rollout_actions = action.unsqueeze(1).repeat(1, horizon, 1)
        kf_params = torch.zeros(env.num_envs, state_dim * 3, device=current_state.device)

    history = torch.roll(history, shifts=1, dims=1)
    history[:, 0, :] = current_state
    setattr(env, "_plan_c_state_history", history)
    setattr(env, "_plan_c_prev_state", current_state.detach())

    states_flat = rollout_states.reshape(rollout_states.shape[0], -1)
    actions_flat = rollout_actions.reshape(rollout_actions.shape[0], -1)
    return torch.cat((states_flat, actions_flat, kf_params), dim=-1)

