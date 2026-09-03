"""Damped secondary motion for mascot rig controls."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from rfdetr_demo.animation.retarget import MascotRigPose


@dataclass(frozen=True, slots=True)
class SpringProfile:
    """Frequency-like stiffness and exponential velocity damping."""

    stiffness: float
    damping: float


_PROFILES = {
    "root_x_px": SpringProfile(150.0, 21.0),
    "root_y_px": SpringProfile(170.0, 23.0),
    "body_angle_deg": SpringProfile(120.0, 18.0),
    "body_scale_x": SpringProfile(130.0, 20.0),
    "body_scale_y": SpringProfile(130.0, 20.0),
    "spine_curve_px": SpringProfile(95.0, 15.0),
    "head_angle_deg": SpringProfile(105.0, 15.0),
    "left_arm_angle_deg": SpringProfile(75.0, 11.0),
    "right_arm_angle_deg": SpringProfile(75.0, 11.0),
    "left_leg_angle_deg": SpringProfile(115.0, 18.0),
    "right_leg_angle_deg": SpringProfile(115.0, 18.0),
    "left_foot_angle_deg": SpringProfile(80.0, 12.0),
    "right_foot_angle_deg": SpringProfile(80.0, 12.0),
    "left_foot_lift_px": SpringProfile(135.0, 19.0),
    "right_foot_lift_px": SpringProfile(135.0, 19.0),
    "twist_deg": SpringProfile(100.0, 16.0),
    "mouth_open": SpringProfile(145.0, 22.0),
}

_LIMITS = {
    "root_x_px": (-90.0, 90.0),
    "root_y_px": (-55.0, 55.0),
    "body_angle_deg": (-18.0, 18.0),
    "body_scale_x": (0.86, 1.02),
    "body_scale_y": (0.88, 1.12),
    "spine_curve_px": (-20.0, 20.0),
    "head_angle_deg": (-22.0, 22.0),
    "left_arm_angle_deg": (-40.0, 40.0),
    "right_arm_angle_deg": (-40.0, 40.0),
    "left_leg_angle_deg": (-25.0, 25.0),
    "right_leg_angle_deg": (-25.0, 25.0),
    "left_foot_angle_deg": (-20.0, 20.0),
    "right_foot_angle_deg": (-20.0, 20.0),
    "left_foot_lift_px": (0.0, 48.0),
    "right_foot_lift_px": (0.0, 48.0),
    "twist_deg": (-48.0, 48.0),
    "mouth_open": (0.0, 1.0),
}


class MascotMotionDynamics:
    """Apply per-control damped springs and optional 1€ Filter while preserving the first pose."""

    def __init__(
        self,
        *,
        fps: float,
        use_one_euro: bool = True,
        min_cutoff: float = 1.2,
        beta: float = 0.08,
    ) -> None:
        self.dt = 1.0 / max(float(fps), 1.0)
        self.use_one_euro = use_one_euro
        self.min_cutoff = min_cutoff
        self.beta = beta
        self._values: dict[str, float] | None = None
        self._velocities: dict[str, float] = {}
        self._filters_x: dict[str, float | None] = {}
        self._filters_dx: dict[str, float] = {}

    def reset(self) -> None:
        self._values = None
        self._velocities.clear()
        self._filters_x.clear()
        self._filters_dx.clear()

    def _one_euro_filter(self, name: float, val: float, name_str: str) -> float:
        x_prev = self._filters_x.get(name_str)
        dx_prev = self._filters_dx.get(name_str, 0.0)
        if x_prev is None:
            self._filters_x[name_str] = val
            self._filters_dx[name_str] = 0.0
            return val

        dt = self.dt
        dx = (val - x_prev) / dt
        alpha_dx = 1.0 / (1.0 + (1.0 / (2.0 * np.pi * 1.0)) / dt)
        dx_hat = alpha_dx * dx + (1.0 - alpha_dx) * dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        alpha = 1.0 / (1.0 + (1.0 / (2.0 * np.pi * cutoff)) / dt)
        x_hat = alpha * val + (1.0 - alpha) * x_prev

        self._filters_x[name_str] = x_hat
        self._filters_dx[name_str] = dx_hat
        return x_hat

    def apply(self, target: MascotRigPose) -> MascotRigPose:
        """Advance one frame toward ``target`` and return the filtered pose."""
        target_values = {name: float(value) for name, value in asdict(target).items()}
        if self._values is None:
            self._values = target_values.copy()
            self._velocities = {name: 0.0 for name in _PROFILES}
            return target

        output = self._values.copy()
        for name, profile in _PROFILES.items():
            value = self._values[name]
            velocity = self._velocities[name]
            target_val = target_values[name]
            if self.use_one_euro:
                target_val = self._one_euro_filter(name, target_val, name)

            velocity += (target_val - value) * profile.stiffness * self.dt
            velocity *= float(np.exp(-profile.damping * self.dt))
            value += velocity * self.dt
            lower, upper = _LIMITS[name]
            clipped = float(np.clip(value, lower, upper))
            if clipped != value:
                velocity = 0.0
            self._values[name] = clipped
            self._velocities[name] = velocity
            output[name] = clipped

        output["confidence"] = target.confidence
        output["left_foot_contact"] = target.left_foot_contact
        output["right_foot_contact"] = target.right_foot_contact
        self._values["confidence"] = target.confidence
        self._values["left_foot_contact"] = target.left_foot_contact
        self._values["right_foot_contact"] = target.right_foot_contact
        return MascotRigPose(**output)



