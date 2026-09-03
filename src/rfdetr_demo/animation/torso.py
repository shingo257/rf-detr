"""Derive animation-oriented torso controls from COCO 17 keypoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

PointSource = Literal["detected", "derived", "interpolated", "model3d"]

COCO17_KEYPOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
COCO17_INDEX = {name: index for index, name in enumerate(COCO17_KEYPOINT_NAMES)}

TORSO_CONTROL_NAMES = (
    "neck_base", "sternum", "left_lower_rib", "right_lower_rib", "navel",
    "left_waist", "right_waist", "pelvis_center", "spine_upper", "spine_mid",
    "spine_lower", "chest_center", "hip_axis_center",
)


@dataclass(frozen=True, slots=True)
class ControlPoint:
    """One animation control point with provenance and confidence."""

    xy: np.ndarray
    confidence: float
    visible: bool
    source: PointSource = "derived"

    def as_dict(self) -> dict[str, object]:
        return {
            "xy": [float(self.xy[0]), float(self.xy[1])],
            "confidence": float(self.confidence),
            "visible": bool(self.visible),
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class TorsoControls:
    """Torso control points and normalized animation parameters."""

    points: dict[str, ControlPoint]
    parameters: dict[str, float]
    confidence: float

    def as_dict(self) -> dict[str, object]:
        return {
            "points": {name: point.as_dict() for name, point in self.points.items()},
            "parameters": {name: float(value) for name, value in self.parameters.items()},
            "confidence": float(self.confidence),
        }


def _lerp(start: np.ndarray, end: np.ndarray, amount: float) -> np.ndarray:
    return start + amount * (end - start)


def _angle(first: np.ndarray, second: np.ndarray) -> float:
    delta = second - first
    return float(np.degrees(np.arctan2(delta[1], delta[0])))


def _wrap_degrees(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def _geometric_confidence(values: list[float]) -> float:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    return float(np.prod(clipped) ** (1.0 / len(clipped)))


def derive_torso_controls(
    xy: np.ndarray,
    *,
    confidence: np.ndarray | None = None,
    visible: np.ndarray | None = None,
    min_confidence: float = 0.15,
    max_curve_fraction: float = 0.10,
) -> TorsoControls:
    """Derive torso controls from one person's COCO 17 keypoints.

    The S-curve is an animation proxy inferred from shoulder/pelvis counter
    rotation. It is not an observed anatomical spine position.
    """
    coordinates = np.asarray(xy, dtype=np.float64)
    if coordinates.shape != (17, 2):
        raise ValueError(f"xy must have shape (17, 2), got {coordinates.shape}")
    scores = np.ones(17, dtype=np.float64) if confidence is None else np.asarray(confidence, dtype=np.float64)
    visibility = np.ones(17, dtype=bool) if visible is None else np.asarray(visible, dtype=bool)
    if scores.shape != (17,):
        raise ValueError(f"confidence must have shape (17,), got {scores.shape}")
    if visibility.shape != (17,):
        raise ValueError(f"visible must have shape (17,), got {visibility.shape}")

    indices = [COCO17_INDEX[name] for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")]
    left_shoulder, right_shoulder, left_hip, right_hip = (coordinates[index] for index in indices)
    valid = visibility & np.isfinite(coordinates).all(axis=1) & (scores >= min_confidence)
    torso_visible = all(bool(valid[index]) for index in indices)
    torso_confidence = _geometric_confidence([scores[index] for index in indices]) if torso_visible else 0.0

    neck = (left_shoulder + right_shoulder) * 0.5
    pelvis = (left_hip + right_hip) * 0.5
    torso_axis = pelvis - neck
    torso_height = float(np.linalg.norm(torso_axis))
    lateral = (
        np.array([-torso_axis[1], torso_axis[0]], dtype=np.float64) / torso_height
        if torso_height >= 1e-6
        else np.array([1.0, 0.0], dtype=np.float64)
    )
    shoulder_angle = _angle(left_shoulder, right_shoulder)
    pelvis_angle = _angle(left_hip, right_hip)
    counter_rotation = _wrap_degrees(shoulder_angle - pelvis_angle)
    curve_fraction = float(np.clip(np.sin(np.radians(counter_rotation)) * 0.5, -max_curve_fraction, max_curve_fraction))
    curve_offset = lateral * torso_height * curve_fraction

    def control(value: np.ndarray) -> ControlPoint:
        return ControlPoint(value.copy(), torso_confidence, torso_visible, "derived")

    left_rib = _lerp(left_shoulder, left_hip, 0.56)
    right_rib = _lerp(right_shoulder, right_hip, 0.56)
    left_waist = _lerp(left_shoulder, left_hip, 0.78)
    right_waist = _lerp(right_shoulder, right_hip, 0.78)
    points = {
        "neck_base": control(neck),
        "sternum": control(_lerp(neck, pelvis, 0.30) + curve_offset),
        "left_lower_rib": control(left_rib),
        "right_lower_rib": control(right_rib),
        "navel": control(_lerp(neck, pelvis, 0.68) - curve_offset),
        "left_waist": control(left_waist),
        "right_waist": control(right_waist),
        "pelvis_center": control(pelvis),
        "spine_upper": control(_lerp(neck, pelvis, 0.25) + curve_offset),
        "spine_mid": control(_lerp(neck, pelvis, 0.50)),
        "spine_lower": control(_lerp(neck, pelvis, 0.75) - curve_offset),
        "chest_center": control((left_rib + right_rib) * 0.5),
        "hip_axis_center": control(pelvis),
    }

    left_side = float(np.linalg.norm(left_hip - left_shoulder))
    right_side = float(np.linalg.norm(right_hip - right_shoulder))
    mean_side = max((left_side + right_side) * 0.5, 1e-6)
    vertical = max(abs(torso_axis[1]), 1e-6)
    parameters = {
        "body_lean_deg": float(np.degrees(np.arctan2(torso_axis[0], vertical))),
        "shoulder_roll_deg": shoulder_angle,
        "pelvis_roll_deg": pelvis_angle,
        "torso_twist_2d_deg": counter_rotation,
        "spine_curvature": curve_fraction,
        "waist_compression_left": 1.0 - left_side / mean_side,
        "waist_compression_right": 1.0 - right_side / mean_side,
        "torso_height_px": torso_height,
        "shoulder_width_px": float(np.linalg.norm(right_shoulder - left_shoulder)),
        "pelvis_width_px": float(np.linalg.norm(right_hip - left_hip)),
    }
    return TorsoControls(points, parameters, torso_confidence)
