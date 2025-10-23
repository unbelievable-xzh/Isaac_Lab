"""Online dynamics models used by the plan-C variant of the lift task.

The module implements a light-weight ensemble of neural networks that learn to
predict clean object state increments from partially observed inputs.  The
ensemble is designed for on-policy updates within the simulation loop: each
member receives the filtered object pose, the previous action and a short
history of filtered poses.  The predicted deltas are then rolled out to provide
look-ahead features for the actor and uncertainty estimates for the critic.

The models are intentionally kept compact – a single hidden layer MLP is more
than enough for the quasi-linear cube lifting dynamics.  The forward path is
TorchScript-compatible so that the predictions can be exported or executed in a
background thread without Python overhead.  Training remains in eager mode to
avoid the complexity of stateful optimisers inside TorchScript.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


class _DynamicsNet(nn.Module):
    """Small MLP that predicts a delta in the filtered object position.

    Parameters
    ----------
    state_dim: int
        Dimension of the filtered object state (typically 3 for position).
    action_dim: int
        Dimension of the low-level action applied to the robot arm.
    history_len: int
        Number of previous filtered states concatenated to the input.
    hidden_dim: int
        Width of the hidden layer.
    """

    def __init__(self, state_dim: int, action_dim: int, history_len: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.history_len = history_len
        input_dim = state_dim + action_dim + state_dim * history_len
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, state: Tensor, action: Tensor, history: Tensor) -> Tensor:  # noqa: D401
        hist_flat = history.reshape(history.shape[0], -1)
        stacked = torch.cat((state, action, hist_flat), dim=-1)
        return self.net(stacked)


@dataclass
class _ErrorStats:
    """Tracks running statistics of the model prediction errors."""

    pos_sigma_sq: Tensor
    q_pos: Tensor
    q_vel: Tensor

    @classmethod
    def create(cls, device: torch.device) -> "_ErrorStats":
        zero = torch.zeros(1, device=device)
        return cls(pos_sigma_sq=zero.clone(), q_pos=zero.clone(), q_vel=zero.clone())

    def update(self, delta: Tensor, prev_delta: Tensor, residual: Tensor, momentum: float) -> None:
        avg = (1.0 - momentum)
        self.pos_sigma_sq = avg * self.pos_sigma_sq + momentum * residual.pow(2).mean(dim=0, keepdim=True)
        self.q_pos = avg * self.q_pos + momentum * delta.pow(2).mean(dim=0, keepdim=True)
        self.q_vel = avg * self.q_vel + momentum * (delta - prev_delta).pow(2).mean(dim=0, keepdim=True)

    def export(self) -> Tuple[Tensor, Tensor, Tensor]:
        pos_sigma = torch.sqrt(torch.clamp(self.pos_sigma_sq, min=1e-12))
        return pos_sigma, self.q_pos.clamp(min=1e-12), self.q_vel.clamp(min=1e-12)


class DynamicsModel:
    """Single ensemble member that wraps the neural network and optimiser."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        history_len: int,
        *,
        device: torch.device,
        hidden_dim: int = 64,
        learning_rate: float = 5e-3,
        momentum: float = 0.05,
    ) -> None:
        self.module = _DynamicsNet(state_dim, action_dim, history_len, hidden_dim=hidden_dim).to(device)
        self._script_module: Optional[torch.jit.ScriptModule] = None
        self.optimizer = torch.optim.Adam(self.module.parameters(), lr=learning_rate)
        self.momentum = momentum
        self.error_stats = _ErrorStats.create(device)

    # ------------------------------------------------------------------
    # Training utilities
    # ------------------------------------------------------------------
    def update(self, prev_state: Tensor, prev_action: Tensor, prev_history: Tensor, next_state: Tensor) -> None:
        """Performs a single optimisation step using the latest transition."""

        delta = next_state - prev_state
        if prev_history.shape[1] > 1:
            prev_delta = prev_state - prev_history[:, 1, :]
        else:
            prev_delta = torch.zeros_like(delta)

        self.optimizer.zero_grad(set_to_none=True)
        # Observation computations typically run under ``torch.no_grad`` to avoid
        # polluting the autodiff graph of the RL algorithm.  Since the ensemble
        # updates happen inside that phase we must temporarily re-enable gradient
        # tracking, otherwise the loss would be detached and ``backward()`` would
        # fail.  Wrapping the optimisation step in ``torch.enable_grad`` keeps the
        # rest of the observation pipeline free from gradient side-effects.
        self.module.train()
        with torch.enable_grad():
            pred_delta = self.module(prev_state, prev_action, prev_history)
            loss = F.mse_loss(pred_delta, delta)
            loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            residual = delta - pred_delta
            self.error_stats.update(delta, prev_delta, residual, self.momentum)

    # ------------------------------------------------------------------
    # Inference utilities
    # ------------------------------------------------------------------
    def _get_script_module(self) -> torch.jit.ScriptModule:
        if self._script_module is None:
            self._script_module = torch.jit.script(self.module)
        else:
            self._script_module.load_state_dict(self.module.state_dict())
        return self._script_module

    @torch.no_grad()
    def predict_delta(self, state: Tensor, action: Tensor, history: Tensor) -> Tensor:
        scripted = self._get_script_module()
        return scripted(state, action, history)

    def export_kf_params(self) -> Tuple[Tensor, Tensor, Tensor]:
        return self.error_stats.export()


class DynamicsEnsemble:
    """Maintains multiple ``DynamicsModel`` instances to capture epistemic uncertainty."""

    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        history_len: int,
        ensemble_size: int = 5,
        device: torch.device,
        hidden_dim: int = 64,
        learning_rate: float = 5e-3,
        momentum: float = 0.05,
    ) -> None:
        self.members: List[DynamicsModel] = [
            DynamicsModel(
                state_dim,
                action_dim,
                history_len,
                device=device,
                hidden_dim=hidden_dim,
                learning_rate=learning_rate,
                momentum=momentum,
            )
            for _ in range(ensemble_size)
        ]
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.history_len = history_len

    # ------------------------------------------------------------------
    # Training utilities
    # ------------------------------------------------------------------
    def update(self, prev_state: Tensor, prev_action: Tensor, prev_history: Tensor, next_state: Tensor) -> None:
        for member in self.members:
            member.update(prev_state, prev_action, prev_history, next_state)

    # ------------------------------------------------------------------
    # Prediction utilities
    # ------------------------------------------------------------------
    @torch.no_grad()
    def rollout(
        self,
        state: Tensor,
        action: Tensor,
        history: Tensor,
        *,
        horizon: int,
        soft_update: float = 0.1,
    ) -> Tuple[Tensor, Tensor]:
        """Predicts a sequence of clean states and associated actions."""

        traj_states: List[Tensor] = []
        traj_actions: List[Tensor] = []
        current_state = state
        current_action = action
        hist = history.clone()

        for _ in range(horizon):
            deltas = [member.predict_delta(current_state, current_action, hist) for member in self.members]
            stacked = torch.stack(deltas, dim=0)
            mean_delta = stacked.mean(dim=0)
            current_state = current_state + mean_delta
            traj_states.append(current_state)
            traj_actions.append(current_action)

            hist = torch.roll(hist, shifts=1, dims=1)
            hist[:, 0, :] = current_state
            current_action = current_action.lerp(torch.zeros_like(current_action), soft_update)

        states_tensor = torch.stack(traj_states, dim=1)
        actions_tensor = torch.stack(traj_actions, dim=1)
        return states_tensor, actions_tensor

    @torch.no_grad()
    def export_kf_parameters(self) -> Tensor:
        """Aggregates the per-member Kalman parameters into a feature vector."""

        pos_sigmas = []
        q_pos = []
        q_vel = []
        for member in self.members:
            sigma, q_p, q_v = member.export_kf_params()
            pos_sigmas.append(sigma)
            q_pos.append(q_p)
            q_vel.append(q_v)

        pos_sigma = torch.stack(pos_sigmas, dim=0).mean(dim=0)
        q_pos_mean = torch.stack(q_pos, dim=0).mean(dim=0)
        q_vel_mean = torch.stack(q_vel, dim=0).mean(dim=0)
        return torch.cat((pos_sigma, q_pos_mean, q_vel_mean), dim=-1)


def create_ensemble(
    *,
    state_dim: int,
    action_dim: int,
    history_len: int,
    ensemble_size: int,
    device: torch.device,
    hidden_dim: int = 64,
    learning_rate: float = 5e-3,
    momentum: float = 0.05,
) -> DynamicsEnsemble:
    """Factory helper used by the observation function."""

    return DynamicsEnsemble(
        state_dim=state_dim,
        action_dim=action_dim,
        history_len=history_len,
        ensemble_size=ensemble_size,
        device=device,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        momentum=momentum,
    )