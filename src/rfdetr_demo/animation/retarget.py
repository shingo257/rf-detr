"""Retarget tracked human keypoints to a compact mascot rig pose."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


def _point(person: dict[str, Any], name: str) -> np.ndarray:
    return np.asarray(person["coco17"][name]["xy"], dtype=np.float64)


def _angle(first: np.ndarray, second: np.ndarray) -> float:
    delta = second - first
    return float(np.degrees(np.arctan2(delta[1], delta[0])))


def _angle_delta(current: float, reference: float) -> float:
    return (current - reference + 180.0) % 360.0 - 180.0


def _limb_angle(person: dict[str, Any], start: str, end: str) -> float:
    return _angle(_point(person, start), _point(person, end))


@dataclass(frozen=True, slots=True)
class MascotRigPose:
    """Frame-local controls consumed by the layered mascot renderer."""

    root_x_px: float = 0.0
    root_y_px: float = 0.0
    body_angle_deg: float = 0.0
    body_scale_x: float = 1.0
    body_scale_y: float = 1.0
    spine_curve_px: float = 0.0
    head_angle_deg: float = 0.0
    left_arm_angle_deg: float = 0.0
    right_arm_angle_deg: float = 0.0
    left_leg_angle_deg: float = 0.0
    right_leg_angle_deg: float = 0.0
    left_foot_angle_deg: float = 0.0
    right_foot_angle_deg: float = 0.0
    left_foot_lift_px: float = 0.0
    right_foot_lift_px: float = 0.0
    left_foot_contact: float = 1.0
    right_foot_contact: float = 1.0
    twist_deg: float = 0.0
    mouth_open: float = 0.0
    confidence: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in asdict(self).items()}


class FukkachanRetargeter:
    """Map human motion deltas onto the proportions of the mascot rig."""

    def __init__(
        self,
        reference_person: dict[str, Any],
        *,
        character_torso_height_px: float = 230.0,
        motion_scale: float = 0.65,
    ) -> None:
        self.reference = reference_person
        self.character_torso_height_px = character_torso_height_px
        self.motion_scale = motion_scale
        self._reference_pelvis = np.asarray(reference_person["points"]["pelvis_center"]["xy"], dtype=np.float64)
        self._reference_torso_height = max(float(reference_person["parameters"]["torso_height_px"]), 1e-6)
        self._reference_parameters = reference_person["parameters"]
        self._reference_ankle_relative = {
            "left": _point(reference_person, "left_ankle") - self._reference_pelvis,
            "right": _point(reference_person, "right_ankle") - self._reference_pelvis,
        }
        self._angles = {
            "head": _angle(_point(reference_person, "left_eye"), _point(reference_person, "right_eye")),
            "left_arm": _limb_angle(reference_person, "left_shoulder", "left_wrist"),
            "right_arm": _limb_angle(reference_person, "right_shoulder", "right_wrist"),
            "left_leg": _limb_angle(reference_person, "left_hip", "left_ankle"),
            "right_leg": _limb_angle(reference_person, "right_hip", "right_ankle"),
            "left_foot": _limb_angle(reference_person, "left_knee", "left_ankle"),
            "right_foot": _limb_angle(reference_person, "right_knee", "right_ankle"),
        }

    def apply(self, person: dict[str, Any]) -> MascotRigPose:
        """Return a bounded mascot pose relative to the reference frame."""
        pelvis = np.asarray(person["points"]["pelvis_center"]["xy"], dtype=np.float64)
        root_delta = (pelvis - self._reference_pelvis) / self._reference_torso_height
        root_delta *= self.character_torso_height_px * self.motion_scale

        parameters = person["parameters"]
        torso_ratio = float(parameters["torso_height_px"]) / self._reference_torso_height
        twist_delta = _angle_delta(
            float(parameters["torso_twist_2d_deg"]),
            float(self._reference_parameters["torso_twist_2d_deg"]),
        )
        curvature_delta = float(parameters["spine_curvature"]) - float(self._reference_parameters["spine_curvature"])

        def foot_lift(side: str) -> float:
            relative = _point(person, f"{side}_ankle") - pelvis
            lift_fraction = (self._reference_ankle_relative[side][1] - relative[1]) / self._reference_torso_height
            return float(np.clip(lift_fraction * self.character_torso_height_px * 0.55, 0.0, 45.0))

        left_lift = foot_lift("left")
        right_lift = foot_lift("right")

        def limb(name: str, start: str, end: str, limit: float, gain: float) -> float:
            delta = _angle_delta(_limb_angle(person, start, end), self._angles[name])
            return float(np.clip(delta * gain, -limit, limit))

        left_arm = limb("left_arm", "left_shoulder", "left_wrist", 35.0, 0.50)
        right_arm = limb("right_arm", "right_shoulder", "right_wrist", 35.0, 0.50)
        left_leg = limb("left_leg", "left_hip", "left_ankle", 22.0, 0.45)
        right_leg = limb("right_leg", "right_hip", "right_ankle", 22.0, 0.45)
        motion_energy = (
            (abs(left_arm) + abs(right_arm)) / 70.0 * 0.40
            + (abs(left_leg) + abs(right_leg)) / 44.0 * 0.15
            + (left_lift + right_lift) / 90.0 * 0.30
            + min(abs(root_delta[1]) / 35.0, 1.0) * 0.15
        )

        return MascotRigPose(
            root_x_px=float(np.clip(root_delta[0], -85.0, 85.0)),
            root_y_px=float(np.clip(root_delta[1], -50.0, 50.0)),
            body_angle_deg=float(
                np.clip(
                    (float(parameters["body_lean_deg"]) - float(self._reference_parameters["body_lean_deg"])) * 0.7,
                    -16.0,
                    16.0,
                )
            ),
            body_scale_x=float(np.clip(1.0 - abs(twist_delta) / 90.0 * 0.12, 0.88, 1.0)),
            body_scale_y=float(np.clip(1.0 + (torso_ratio - 1.0) * 0.45, 0.90, 1.10)),
            spine_curve_px=float(np.clip(curvature_delta * self.character_torso_height_px * 0.8, -18.0, 18.0)),
            head_angle_deg=float(
                np.clip(
                    _angle_delta(_angle(_point(person, "left_eye"), _point(person, "right_eye")), self._angles["head"]),
                    -20.0,
                    20.0,
                )
            ),
            left_arm_angle_deg=left_arm,
            right_arm_angle_deg=right_arm,
            left_leg_angle_deg=left_leg,
            right_leg_angle_deg=right_leg,
            left_foot_angle_deg=limb("left_foot", "left_knee", "left_ankle", 18.0, 0.30),
            right_foot_angle_deg=limb("right_foot", "right_knee", "right_ankle", 18.0, 0.30),
            left_foot_lift_px=left_lift,
            right_foot_lift_px=right_lift,
            left_foot_contact=1.0 if left_lift <= 3.0 else 0.0,
            right_foot_contact=1.0 if right_lift <= 3.0 else 0.0,
            twist_deg=float(np.clip(twist_delta, -45.0, 45.0)),
            mouth_open=float(np.clip((motion_energy - 0.18) / 0.62, 0.0, 1.0)),
            confidence=float(person["confidence"]),
        )
