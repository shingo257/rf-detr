# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Pure mesh-field calculations for mascot rendering."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from rfdetr_demo.animation.puppet_assets import Float32Array
from rfdetr_demo.animation.retarget import MascotRigPose


def mesh_weight(
    grid_x: Float32Array,
    grid_y: Float32Array,
    *,
    width: int,
    height: int,
    pivot: tuple[float, float],
    radius_x: float,
    radius_y: float,
    side: str | None = None,
    lower_bias: float | None = None,
    direction_bias: tuple[float, float] | None = None,
) -> Float32Array:
    """Calculate a smoothly bounded influence field for one mesh handle."""
    px, py = pivot
    distance = ((grid_x - px) / radius_x) ** 2 + ((grid_y - py) / radius_y) ** 2
    weight = np.exp(-0.5 * distance).astype(np.float32)

    if direction_bias is not None:
        dx, dy = direction_bias
        vec_x = grid_x - px
        vec_y = grid_y - py
        norm = np.sqrt(vec_x**2 + vec_y**2) + 1e-6
        dot = (vec_x * dx + vec_y * dy) / norm
        direction_factor = np.clip(1.0 - dot * 1.6, 0.1, 1.0)
        weight *= direction_factor

    if side == "left":
        weight *= np.clip((width * 0.58 - grid_x) / (width * 0.18), 0.0, 1.0)
    elif side == "right":
        weight *= np.clip((grid_x - width * 0.42) / (width * 0.18), 0.0, 1.0)
    if lower_bias is not None:
        weight *= np.clip((grid_y - height * lower_bias) / (height * 0.10), 0.0, 1.0)
    return weight


def add_mesh_rotation(
    displacement_x: Float32Array,
    displacement_y: Float32Array,
    *,
    grid_x: Float32Array,
    grid_y: Float32Array,
    pivot: tuple[float, float],
    angle_deg: float,
    weight: Float32Array,
) -> None:
    """Add a weighted rotation field to displacement arrays in place."""
    radians = np.radians(angle_deg)
    cosine = float(np.cos(radians))
    sine = float(np.sin(radians))
    rel_x = grid_x - pivot[0]
    rel_y = grid_y - pivot[1]
    displacement_x += ((cosine - 1.0) * rel_x - sine * rel_y) * weight
    displacement_y += (sine * rel_x + (cosine - 1.0) * rel_y) * weight


def anchor_angle_deg(anchor: str, pose: MascotRigPose, mesh_profile: dict[str, Any]) -> float:
    """Resolve the continuous-mesh joint angle for a rigid-prop anchor."""
    arm_gain = float(mesh_profile.get("arm_angle_gain", 1.0))
    leg_gain = float(mesh_profile.get("leg_angle_gain", 1.0))
    if anchor in {"Left Arm", "Left Hand"}:
        return float(pose.right_arm_angle_deg) * arm_gain
    if anchor in {"Right Arm", "Right Hand"}:
        return float(pose.left_arm_angle_deg) * arm_gain
    if anchor in {"Left Leg", "Left Foot"}:
        return float(pose.right_leg_angle_deg + pose.right_foot_angle_deg * 0.35) * leg_gain
    if anchor in {"Right Leg", "Right Foot"}:
        return float(pose.left_leg_angle_deg + pose.left_foot_angle_deg * 0.35) * leg_gain
    if anchor == "Head":
        return float(
            (pose.head_angle_deg + pose.body_angle_deg * 0.20) * float(mesh_profile.get("head_angle_gain", 1.0))
        )
    if anchor in {"Body", "Bell"}:
        return float(pose.body_angle_deg) * float(mesh_profile.get("body_angle_gain", 1.0))
    return 0.0


def apply_rigid_prop_constraints(
    displacement_x: Float32Array,
    displacement_y: Float32Array,
    *,
    grid_x: Float32Array,
    grid_y: Float32Array,
    width: int,
    height: int,
    rigid_props: list[dict[str, Any]],
    pose: MascotRigPose,
    mesh_profile: dict[str, Any],
) -> None:
    """Replace soft prop-region fields with shape-preserving rigid motion."""
    for prop in rigid_props:
        center = prop.get("center", [0.5, 0.5])
        radius = prop.get("radius", [0.08, 0.08])
        feather = max(float(prop.get("feather", 0.12)), 1e-3)
        source_x = int(np.clip(width * float(center[0]), 0, width - 1))
        source_y = int(np.clip(height * float(center[1]), 0, height - 1))
        radius_x = width * float(radius[0])
        radius_y = height * float(radius[1])
        pad = 1.0 + feather
        x1 = max(0, int(source_x - radius_x * pad - 4))
        x2 = min(width, int(source_x + radius_x * pad + 5))
        y1 = max(0, int(source_y - radius_y * pad - 4))
        y2 = min(height, int(source_y + radius_y * pad + 5))
        if x2 <= x1 or y2 <= y1:
            continue

        local_grid_x = grid_x[y1:y2, x1:x2]
        local_grid_y = grid_y[y1:y2, x1:x2]
        soft_tx = float(displacement_x[source_y, source_x])
        soft_ty = float(displacement_y[source_y, source_x])
        rigid_dx = np.full(local_grid_x.shape, soft_tx, dtype=np.float32)
        rigid_dy = np.full(local_grid_y.shape, soft_ty, dtype=np.float32)

        anchor = str(prop.get("anchor", ""))
        if bool(prop.get("rotate", True)) and anchor:
            angle_scale = float(prop.get("angle_scale", 0.35))
            radians = np.radians(anchor_angle_deg(anchor, pose, mesh_profile) * angle_scale)
            cosine = float(np.cos(radians))
            sine = float(np.sin(radians))
            rel_x = local_grid_x - float(source_x)
            rel_y = local_grid_y - float(source_y)
            rigid_dx = rigid_dx + ((cosine - 1.0) * rel_x - sine * rel_y)
            rigid_dy = rigid_dy + (sine * rel_x + (cosine - 1.0) * rel_y)

        distance = ((local_grid_x - source_x) / radius_x) ** 2 + ((local_grid_y - source_y) / radius_y) ** 2
        prop_weight = np.clip((1.0 + feather - distance) / feather, 0.0, 1.0)
        prop_weight = cv2.GaussianBlur(prop_weight.astype(np.float32), (0, 0), 1.2)
        region_x = displacement_x[y1:y2, x1:x2]
        region_y = displacement_y[y1:y2, x1:x2]
        region_x[:] = region_x * (1.0 - prop_weight) + rigid_dx * prop_weight
        region_y[:] = region_y * (1.0 - prop_weight) + rigid_dy * prop_weight
