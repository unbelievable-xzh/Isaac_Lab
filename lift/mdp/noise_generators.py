"""Noise generators used by lift MDP observations."""

from __future__ import annotations

import torch


class GaussianPositionNoise:
    """Simple additive Gaussian noise generator for position measurements."""

    def __init__(self, sigma: float) -> None:
        self.sigma = sigma

    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        """Returns ``values`` corrupted with zero-mean Gaussian noise."""

        if self.sigma <= 0:
            return values

        noise = torch.randn_like(values)
        return values + noise * self.sigma

