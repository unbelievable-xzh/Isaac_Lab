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

from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from threading import Event, Lock, Thread
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
        avg = 1.0 - momentum
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
        # Observation computations typically run under ``torch.no_grad`` (or even
        # ``torch.inference_mode``) to avoid polluting the autodiff graph of the RL
        # algorithm.  Since the ensemble updates happen inside that phase we must
        # temporarily re-enable gradient tracking, otherwise the loss would be
        # detached and ``backward()`` would fail.  Wrapping the optimisation step in
        # ``torch.enable_grad`` keeps the rest of the observation pipeline free from
        # gradient side-effects.  Additionally, some call-sites enable
        # ``torch.inference_mode`` globally, in which case ``torch.enable_grad``
        # alone is insufficient; explicitly disabling inference mode ensures the
        # autograd engine is active while the optimiser step executes.
        self.module.train()
        inference_ctx = getattr(torch.autograd.grad_mode, "inference_mode", None)
        grad_guard = inference_ctx(False) if inference_ctx is not None else nullcontext()
        with grad_guard:
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


class PlanCTransitionDataset:
    """Thread-safe queue that buffers transitions for the dynamics ensemble."""

    def __init__(self, capacity: int = 512) -> None:
        self._buffer: deque[tuple[Tensor, Tensor, Tensor, Tensor]] = deque(maxlen=capacity)
        self._lock = Lock()
        self._has_data = Event()

    def __len__(self) -> int:  # pragma: no cover - trivial
        with self._lock:
            return len(self._buffer)

    def push(self, prev_state: Tensor, action: Tensor, history: Tensor, next_state: Tensor) -> None:
        transition = (
            prev_state.detach().to(device="cpu"),
            action.detach().to(device="cpu"),
            history.detach().to(device="cpu"),
            next_state.detach().to(device="cpu"),
        )
        with self._lock:
            self._buffer.append(transition)
            self._has_data.set()

    def pop_many(self, max_items: int) -> List[tuple[Tensor, Tensor, Tensor, Tensor]]:
        with self._lock:
            if not self._buffer:
                self._has_data.clear()
                return []
            count = min(max_items, len(self._buffer))
            items = [self._buffer.popleft() for _ in range(count)]
            if not self._buffer:
                self._has_data.clear()
            return items

    def wait_for_data(self, timeout: Optional[float] = None) -> bool:
        return self._has_data.wait(timeout)

    def notify_all(self) -> None:
        self._has_data.set()


def get_plan_c_dataset(env) -> PlanCTransitionDataset:
    """Returns the cached transition dataset attached to the environment."""

    dataset: Optional[PlanCTransitionDataset] = getattr(env, "_plan_c_dataset", None)
    if dataset is None:
        dataset = PlanCTransitionDataset()
        setattr(env, "_plan_c_dataset", dataset)
    return dataset


def _locate_plan_c_env(root) -> object:
    """Best-effort lookup of the environment hosting the plan-C buffers."""

    visited: set[int] = set()
    stack: List[object] = [root]
    while stack:
        current = stack.pop()
        obj_id = id(current)
        if obj_id in visited:
            continue
        visited.add(obj_id)

        if hasattr(current, "_plan_c_dataset") or hasattr(current, "_plan_c_state_history"):
            return current

        for attr_name in ("env", "_env", "unwrapped", "vec_env", "_vec_env", "envs"):
            if not hasattr(current, attr_name):
                continue
            candidate = getattr(current, attr_name)
            if isinstance(candidate, (list, tuple)):
                stack.extend(candidate)
            elif candidate is not None:
                stack.append(candidate)

    return root


class PlanCDynamicsAsyncTrainer:
    """Background worker that performs deferred dynamics updates."""

    def __init__(
        self,
        env,
        *,
        device: Optional[torch.device | str] = None,
        max_updates_per_cycle: int = 4,
        poll_interval: float = 0.05,
    ) -> None:
        self._root_env = env
        self._base_env = _locate_plan_c_env(env)
        self._device = torch.device(device) if device is not None else getattr(self._base_env, "device", torch.device("cpu"))
        self._max_updates = max(1, int(max_updates_per_cycle))
        self._poll_interval = max(0.0, float(poll_interval))
        self._stop_event = Event()
        self._thread: Optional[Thread] = None
        self._last_error: Optional[BaseException] = None

        # ensure dataset exists so that the worker can block on it immediately
        get_plan_c_dataset(self._base_env)

    def start(self) -> None:
        if self._thread is not None:
            return

        self._stop_event.clear()
        self._thread = Thread(target=self._worker_loop, name="plan_c_dynamics_trainer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return

        self._stop_event.set()
        get_plan_c_dataset(self._base_env).notify_all()
        self._thread.join()
        self._thread = None

        if self._last_error is not None:
            error = self._last_error
            self._last_error = None
            raise error

    def update_now(self, max_updates: Optional[int] = None) -> int:
        """Runs a bounded number of optimisation steps on the calling thread."""

        if max_updates is None:
            max_updates = self._max_updates

        dataset = get_plan_c_dataset(self._base_env)
        transitions = dataset.pop_many(max_updates)
        if not transitions:
            return 0

        ensemble: Optional[DynamicsEnsemble] = getattr(self._base_env, "_plan_c_dynamics", None)
        if ensemble is None:
            # If the observation group never created the ensemble we simply drop the data.
            return 0

        updates = 0
        for prev_state, action, history, next_state in transitions:
            prev_state = prev_state.to(self._device)
            action = action.to(self._device)
            history = history.to(self._device)
            next_state = next_state.to(self._device)
            ensemble.update(prev_state, action, history, next_state)
            updates += 1
        return updates

    def _worker_loop(self) -> None:
        dataset = get_plan_c_dataset(self._base_env)
        try:
            while not self._stop_event.is_set():
                if not dataset.wait_for_data(timeout=self._poll_interval):
                    continue
                if self.update_now(self._max_updates) == 0:
                    # No work was performed. Loop again after waiting for new data.
                    continue
        except BaseException as exc:  # noqa: BLE001
            self._last_error = exc
            self._stop_event.set()
            dataset.notify_all()

