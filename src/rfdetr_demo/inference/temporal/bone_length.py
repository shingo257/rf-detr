# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Bone-length constraints for COCO keypoint tracks."""

from __future__ import annotations

import numpy as np
import supervision as sv

COCO_BONE_PAIRS: tuple[tuple[int, int], ...] = (
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (5, 6),
    (11, 12),
    (5, 11),
    (6, 12),
)


def apply_bone_length_constraint(
    key_points: sv.KeyPoints,
    detection_index: int,
    reference_lengths: dict[tuple[int, int], float],
    max_deviation: float,
) -> int:
    """Constrain COCO bone lengths and return the number of corrected pairs."""
    xy = key_points.xy[detection_index]
    visible = key_points.visible[detection_index] if key_points.visible is not None else None
    corrections = 0

    for parent, child in COCO_BONE_PAIRS:
        if parent >= len(xy) or child >= len(xy):
            continue
        if visible is not None and (not visible[parent] or not visible[child]):
            continue
        parent_position = xy[parent].astype(np.float64)
        child_position = xy[child].astype(np.float64)
        if np.allclose(parent_position, 0) or np.allclose(child_position, 0):
            continue

        current_length = float(np.linalg.norm(child_position - parent_position))
        if current_length < 1e-4:
            continue

        pair = (parent, child)
        if pair not in reference_lengths:
            reference_lengths[pair] = current_length
            continue

        reference_length = reference_lengths[pair]
        reference_lengths[pair] = 0.98 * reference_length + 0.02 * current_length
        target_length = reference_lengths[pair]
        deviation_ratio = abs(current_length - target_length) / max(target_length, 1e-4)
        if deviation_ratio <= max_deviation:
            continue

        direction = (child_position - parent_position) / current_length
        corrected_length = target_length + np.sign(current_length - target_length) * (target_length * max_deviation)
        delta = (current_length - corrected_length) * direction * 0.5
        key_points.xy[detection_index, parent] += delta
        key_points.xy[detection_index, child] -= delta
        corrections += 1

    return corrections


__all__ = ["COCO_BONE_PAIRS", "apply_bone_length_constraint"]
