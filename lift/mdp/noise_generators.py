"""Noise generators used by lift MDP observations."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch


class GaussianPositionNoise:
    """Stateful position noise generator supporting multiple corruption modes.

    The generator mimics a simple sensor model with a collection of optional
    error sources such as additive bias, drift, random dropouts, quantisation,
    delay and frame-hold behaviour.  All internal buffers are cached on the
    environment instance so that the same noise process is shared between
    different observation heads (e.g. raw vs. filtered).
    """

    def __init__(
        self,
        env,
        *,
        pos_sigma: float = 0.0,
        bias_sigma: float = 0.0,
        drift_sigma: float = 0.0,
        rot_bias_deg: float = 0.0,
        scale_err: float = 0.0,
        quant_step: float = 0.0,
        dropout_p: float = 0.0,
        delay_steps: int = 0,
        jitter_range: int = 0,
        low_fps_hold: int = 1,
    ) -> None:
        self._env = env
        self.update_parameters(
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

    def update_parameters(
        self,
        *,
        pos_sigma: float,
        bias_sigma: float,
        drift_sigma: float,
        rot_bias_deg: float,
        scale_err: float,
        quant_step: float,
        dropout_p: float,
        delay_steps: int,
        jitter_range: int,
        low_fps_hold: int,
    ) -> None:
        self.sigma = pos_sigma
        self.bias_sigma = bias_sigma
        self.drift_sigma = drift_sigma
        self.rot_bias_deg = rot_bias_deg
        self.scale_err = scale_err
        self.quant_step = quant_step
        self.dropout_p = dropout_p
        self.delay_steps = max(0, int(delay_steps))
        self.jitter_range = max(0, int(jitter_range))
        self.low_fps_hold = max(1, int(low_fps_hold))

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _get_buffer(self, num_envs: int, device: torch.device) -> Dict[str, Any]:
        env = self._env
        buf: Optional[Dict[str, Any]] = getattr(env, "_obs_noise_buf", None)
        if buf is not None:
            bias = buf.get("bias")
            if (
                bias is None
                or bias.shape[0] != num_envs
                or bias.device != device
            ):
                buf = None
        if buf is None:
            buf = {
                "bias": torch.zeros(num_envs, 3, device=device),
                "drift": torch.zeros(num_envs, 3, device=device),
                "delay_fifo": [],  # type: List[torch.Tensor]
                "frame_count": torch.zeros(num_envs, dtype=torch.long, device=device),
                "last_dropout_mask": torch.zeros(num_envs, dtype=torch.bool, device=device),
            }
            setattr(env, "_obs_noise_buf", buf)
        return buf

    def _reset_if_needed(self, buf: Dict[str, Any], device: torch.device) -> None:
        env = self._env
        done = getattr(env, "reset_buf", None)
        if done is None and hasattr(env, "episode_length_buf"):
            done = env.episode_length_buf == 0
        if isinstance(done, torch.Tensor):
            done = done.to(device=device)
            if done.any():
                if self.bias_sigma > 0:
                    buf["bias"][done] = (
                        torch.randn(done.sum(), 3, device=device) * self.bias_sigma
                    )
                buf["drift"][done] = 0.0
                buf["frame_count"][done] = 0
                buf["last_dropout_mask"][done] = False

    # ------------------------------------------------------------------
    # main interface
    # ------------------------------------------------------------------
    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        """Returns ``values`` corrupted using the configured noise process."""

        env = self._env
        if self._all_disabled():
            setattr(env, "_objpos_meas_b", values)
            return values

        num_envs, device = values.shape[0], values.device
        buf = self._get_buffer(num_envs, device)
        self._reset_if_needed(buf, device)

        if self.drift_sigma > 0:
            buf["drift"] += torch.randn_like(buf["drift"]) * self.drift_sigma

        noisy = values.clone()

        if self.rot_bias_deg != 0.0:
            theta = math.radians(self.rot_bias_deg)
            rot = torch.tensor(
                [
                    [math.cos(theta), -math.sin(theta), 0.0],
                    [math.sin(theta), math.cos(theta), 0.0],
                    [0.0, 0.0, 1.0],
                ],
                device=device,
            )
            noisy = noisy @ rot.T
        if self.scale_err != 0.0:
            noisy = noisy * (1.0 + self.scale_err)

        noisy = noisy + buf["bias"] + buf["drift"]

        if self.dropout_p > 0.0:
            mask = torch.rand(num_envs, device=device) < self.dropout_p
            if buf["delay_fifo"]:
                prev = buf["delay_fifo"][-1]
                noisy[mask] = prev[mask]
            else:
                noisy[mask] = 0.0
            buf["last_dropout_mask"] = mask

        if self.sigma > 0:
            noisy = noisy + torch.randn_like(noisy) * self.sigma

        if self.quant_step > 0.0:
            noisy = torch.round(noisy / self.quant_step) * self.quant_step

        if self.low_fps_hold > 1:
            buf["frame_count"] += 1
            hold_mask = (buf["frame_count"] % self.low_fps_hold) != 0
            if buf["delay_fifo"]:
                prev = buf["delay_fifo"][-1]
                noisy[hold_mask] = prev[hold_mask]

        delay = self.delay_steps
        if self.jitter_range > 0:
            jitter = int(torch.randint(0, self.jitter_range + 1, (), device=device).item())
            delay += jitter

        buf["delay_fifo"].append(noisy.detach())
        max_hist = max(self.delay_steps + self.jitter_range, self.low_fps_hold) + 1
        if len(buf["delay_fifo"]) > max_hist:
            buf["delay_fifo"] = buf["delay_fifo"][-max_hist:]
        if delay > 0 and len(buf["delay_fifo"]) > delay:
            noisy = buf["delay_fifo"][-1 - delay]

        setattr(env, "_objpos_meas_b", noisy)
        return noisy

    def _all_disabled(self) -> bool:
        return (
            self.sigma <= 0
            and self.bias_sigma <= 0
            and self.drift_sigma <= 0
            and self.rot_bias_deg == 0
            and self.scale_err == 0
            and self.quant_step == 0
            and self.dropout_p <= 0
            and self.delay_steps == 0
            and self.jitter_range == 0
            and self.low_fps_hold <= 1
        )


def get_last_dropout_mask(env) -> Optional[torch.Tensor]:
    """Returns the dropout mask generated during the previous noise step."""

    buf = getattr(env, "_obs_noise_buf", None)
    if buf is None:
        return None
    return buf.get("last_dropout_mask")