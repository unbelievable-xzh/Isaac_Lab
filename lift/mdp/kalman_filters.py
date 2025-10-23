"""Kalman filter utilities for lift task observations."""

from __future__ import annotations

from typing import Optional

import torch


class BatchPosVelKalmanFilter:
    """Batched Kalman filter for position/velocity estimation.

    The filter models a six-dimensional state vector ``x = [p(3), v(3)]`` and a
    position-only measurement ``z = p``.  All operations are vectorised across
    ``num_envs`` to efficiently filter observations in parallel across multiple
    simulation instances.
    """

    def __init__(
        self,
        *,
        num_envs: int,
        dt: float,
        pos_sigma: float,
        q_pos: float,
        q_vel: float,
        device: torch.device | str,
    ) -> None:
        self.N = num_envs
        self.dt = dt
        self.device = torch.device(device)

        self.A = torch.eye(6, device=self.device)
        self.A[:3, 3:] = torch.eye(3, device=self.device) * dt
        self.A_T = self.A.transpose(0, 1)

        self.H = torch.cat(
            (torch.eye(3, device=self.device), torch.zeros(3, 3, device=self.device)), dim=1
        )
        self.H_T = self.H.transpose(0, 1)

        self.Q = torch.zeros(6, 6, device=self.device)
        self.R = torch.eye(3, device=self.device)
        self.update_process_noise(pos_sigma=pos_sigma, q_pos=q_pos, q_vel=q_vel)

        self.x = torch.zeros(self.N, 6, device=self.device)
        self.P = torch.eye(6, device=self.device).unsqueeze(0).repeat(self.N, 1, 1) * 1e-2

        self._I6 = torch.eye(6, device=self.device)

    @torch.no_grad()
    def update_process_noise(self, *, pos_sigma: float, q_pos: float, q_vel: float) -> None:
        """Updates process and measurement noise covariances in-place."""

        self.Q[:3, :3] = torch.eye(3, device=self.device) * q_pos
        self.Q[3:, 3:] = torch.eye(3, device=self.device) * q_vel
        self.Q[:3, 3:] = 0.0
        self.Q[3:, :3] = 0.0
        self.R = torch.eye(3, device=self.device) * (pos_sigma**2)

    @torch.no_grad()
    def reset(self, mask: Optional[torch.Tensor] = None) -> None:
        """Resets the filter for the environments marked in ``mask``."""

        if mask is None:
            self.x.zero_()
            self.P.copy_(self._I6.unsqueeze(0) * 1e-2)
            return

        if mask.dtype != torch.bool:
            mask = mask.to(dtype=torch.bool, device=self.device)
        else:
            mask = mask.to(device=self.device)

        if not mask.any():
            return

        idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        self.x[idx] = 0.0
        self.P[idx] = self._I6 * 1e-2

    @torch.no_grad()
    def step(self, measurement: torch.Tensor) -> torch.Tensor:
        """Performs a single Kalman filter update using ``measurement``."""

        measurement = measurement.to(self.device)
        x_pred = torch.matmul(self.x, self.A_T)
        P_pred = self.A @ self.P @ self.A_T + self.Q.unsqueeze(0)

        innovation = measurement - torch.matmul(self.H, x_pred.transpose(0, 1)).transpose(0, 1)
        S = torch.matmul(torch.matmul(self.H, P_pred), self.H_T) + self.R.unsqueeze(0)
        S_inv = torch.inverse(S)
        K = torch.matmul(torch.matmul(P_pred, self.H_T), S_inv)

        x_new = x_pred + torch.matmul(K, innovation.unsqueeze(-1)).squeeze(-1)
        
        KH   = torch.matmul(K, self.H)
        I_KH = self._I6 - KH
        P_new = I_KH @ P_pred @ I_KH.transpose(-1, -2) + K @ self.R @ K.transpose(-1, -2)
        P_new = 0.5 * (P_new + P_new.transpose(-1, -2))

        self.x = x_new
        self.P = P_new
        return x_new[:, :3]