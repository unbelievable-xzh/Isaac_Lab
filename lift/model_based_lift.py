"""Minimal model-based reinforcement learning demo for the lift task.

This example shows how to collect a dataset from the Isaac Lab lift environment,
fit a simple neural network dynamics model, and use it for action selection via
random shooting model predictive control (MPC).

The implementation is intentionally compact: it focuses on clarity and keeps
all logic in a single file so that it is easy to adapt for experimentation.

Usage
-----
Run the module directly from the repository root. The script will (1) gather a
random dataset from the Lift environment, (2) train the learned dynamics model,
and (3) evaluate the MPC controller:

.. code-block:: bash

    python -m lift.model_based_lift --headless

Commonly-adjusted arguments include ``--dataset-size`` (number of transitions),
``--epochs`` (model training epochs), and ``--mpc-horizon`` (planning depth).

The script works out-of-the-box; no other project files need modification.
Use ``python -m lift.model_based_lift --show-usage`` to print the same
instructions at the command line.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Sequence

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


# -----------------------------------------------------------------------------
# Data utilities
# -----------------------------------------------------------------------------


@dataclass
class Transition:
    """Container for a single environment transition."""

    obs: np.ndarray
    action: np.ndarray
    reward: float
    next_obs: np.ndarray
    done: bool


class TransitionDataset(Dataset):
    """PyTorch dataset for environment transitions."""

    def __init__(self, transitions: Sequence[Transition]):
        self._observations = torch.as_tensor(np.stack([t.obs for t in transitions]), dtype=torch.float32)
        self._actions = torch.as_tensor(np.stack([t.action for t in transitions]), dtype=torch.float32)
        self._rewards = torch.as_tensor([t.reward for t in transitions], dtype=torch.float32).unsqueeze(-1)
        self._next_observations = torch.as_tensor(
            np.stack([t.next_obs for t in transitions]), dtype=torch.float32
        )

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self._observations.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self._observations[index],
            self._actions[index],
            self._rewards[index],
            self._next_observations[index],
        )


def collect_transitions(
    env: gym.Env,
    num_samples: int,
    max_episode_steps: int,
) -> List[Transition]:
    """Collect transitions using a random policy."""

    transitions: List[Transition] = []
    obs, _ = env.reset()
    steps_remaining = max_episode_steps

    while len(transitions) < num_samples:
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        transitions.append(Transition(obs=obs, action=action, reward=reward, next_obs=next_obs, done=terminated))

        steps_remaining -= 1
        done = terminated or truncated or steps_remaining <= 0
        if done:
            obs, _ = env.reset()
            steps_remaining = max_episode_steps
        else:
            obs = next_obs

    return transitions


# -----------------------------------------------------------------------------
# Dynamics model
# -----------------------------------------------------------------------------


class DynamicsModel(nn.Module):
    """Simple multi-layer perceptron dynamics model."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim + 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, action], dim=-1)
        output = self.net(x)
        delta_obs = output[..., : self.obs_dim]
        reward = output[..., self.obs_dim : self.obs_dim + 1]
        return obs + delta_obs, reward


def train_dynamics_model(
    model: DynamicsModel,
    dataset: Dataset,
    batch_size: int,
    epochs: int,
    device: torch.device,
) -> None:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for obs, action, reward, next_obs in dataloader:
            obs = obs.to(device)
            action = action.to(device)
            reward = reward.to(device)
            next_obs = next_obs.to(device)

            pred_next_obs, pred_reward = model(obs, action)
            loss = loss_fn(pred_next_obs, next_obs) + loss_fn(pred_reward, reward)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


# -----------------------------------------------------------------------------
# Model predictive control
# -----------------------------------------------------------------------------


@dataclass
class MPCConfig:
    horizon: int = 5
    num_action_sequences: int = 256


def plan_action(
    model: DynamicsModel,
    obs: np.ndarray,
    action_space: gym.spaces.Box,
    config: MPCConfig,
    device: torch.device,
) -> np.ndarray:
    """Random shooting MPC planner."""

    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

    low = torch.as_tensor(action_space.low, dtype=torch.float32, device=device)
    high = torch.as_tensor(action_space.high, dtype=torch.float32, device=device)
    action_dim = low.shape[0]

    best_return = -float("inf")
    best_first_action = None

    model.eval()
    with torch.no_grad():
        for _ in range(config.num_action_sequences):
            actions = torch.rand((config.horizon, action_dim), device=device) * (high - low) + low
            rollout_obs = obs_t.clone()
            total_reward = torch.zeros(1, device=device)

            for action in actions:
                rollout_obs, reward = model(rollout_obs, action.unsqueeze(0))
                total_reward += reward.squeeze(0)

            rollout_return = float(total_reward.item())
            if rollout_return > best_return:
                best_return = rollout_return
                best_first_action = actions[0]

    assert best_first_action is not None  # safety
    return best_first_action.cpu().numpy()


# -----------------------------------------------------------------------------
# Evaluation helpers
# -----------------------------------------------------------------------------


def evaluate_policy(
    env: gym.Env,
    model: DynamicsModel,
    episodes: int,
    max_episode_steps: int,
    mpc_cfg: MPCConfig,
    device: torch.device,
) -> List[float]:
    rewards: List[float] = []
    for _ in range(episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        for _ in range(max_episode_steps):
            action = plan_action(model, obs, env.action_space, mpc_cfg, device)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        rewards.append(total_reward)
    return rewards


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Model-based RL demo for the lift task")
    parser.add_argument("--env-id", type=str, default="Isaac-Lift-Cube-Franka-v0", help="Gym environment id")
    parser.add_argument("--dataset-size", type=int, default=2000, help="Number of random transitions to collect")
    parser.add_argument("--max-episode-steps", type=int, default=200, help="Cutoff for episode length")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs for the dynamics model")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    parser.add_argument("--mpc-horizon", type=int, default=5, help="Planning horizon for MPC")
    parser.add_argument("--mpc-samples", type=int, default=256, help="Number of sampled action sequences")
    parser.add_argument("--headless", action="store_true", help="Run the simulator without rendering")
    parser.add_argument(
        "--show-usage",
        action="store_true",
        help="Print step-by-step instructions for running the demo and exit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.show_usage:
        print(
            "Model-based Lift demo usage:\n"
            "1. From the repository root run: python -m lift.model_based_lift --headless\n"
            "   (omit --headless to render).\n"
            "2. Adjust --dataset-size, --epochs, or MPC flags to change training/planning.\n"
            "3. The script collects data, trains the dynamics model, and evaluates it automatically.\n"
            "No other files need modification to try the demo."
        )
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(args.env_id, headless=args.headless)

    try:
        transitions = collect_transitions(
            env=env,
            num_samples=args.dataset_size,
            max_episode_steps=args.max_episode_steps,
        )

        dataset = TransitionDataset(transitions)
        obs_dim = dataset[0][0].shape[0]
        action_dim = dataset[0][1].shape[0]

        model = DynamicsModel(obs_dim=obs_dim, action_dim=action_dim).to(device)
        train_dynamics_model(model, dataset, args.batch_size, args.epochs, device)

        mpc_cfg = MPCConfig(horizon=args.mpc_horizon, num_action_sequences=args.mpc_samples)
        rewards = evaluate_policy(
            env=env,
            model=model,
            episodes=3,
            max_episode_steps=args.max_episode_steps,
            mpc_cfg=mpc_cfg,
            device=device,
        )

        avg_reward = float(np.mean(rewards))
        print(f"Average reward across {len(rewards)} evaluation episodes: {avg_reward:.3f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()

