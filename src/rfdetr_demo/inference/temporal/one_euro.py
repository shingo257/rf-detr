# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""One Euro low-pass filtering for temporal coordinates."""

from __future__ import annotations

import numpy as np


class OneEuroFilter:
    """Speed-adaptive low-pass filter for smooth, low-latency tracking."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.05, d_cutoff: float = 1.0) -> None:
        """Initialize the filter thresholds and empty state."""
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev: float | None = None
        self.dx_prev: float = 0.0

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        """Return the smoothing coefficient for a cutoff and time step."""
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x: float, dt: float) -> float:
        """Filter one scalar sample."""
        if dt <= 0:
            return x
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            return x

        dx = (x - self.x_prev) / dt
        alpha_dx = self._alpha(self.d_cutoff, dt)
        dx_hat = alpha_dx * dx + (1.0 - alpha_dx) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        alpha = self._alpha(cutoff, dt)
        x_hat = alpha * x + (1.0 - alpha) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


__all__ = ["OneEuroFilter"]
