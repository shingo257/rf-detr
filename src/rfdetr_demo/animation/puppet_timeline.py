# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Pure timeline helpers for mascot pose animation."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from rfdetr_demo.animation.retarget import MascotRigPose


def person_for_track(frame: dict[str, Any], track_id: int) -> dict[str, Any] | None:
    """Return the person matching ``track_id`` in one control frame."""
    return next((person for person in frame["people"] if int(person["track_id"]) == track_id), None)


def interpolate_pose(start: MascotRigPose, end: MascotRigPose, amount: float) -> MascotRigPose:
    """Interpolate rig controls, using shortest paths for angular fields."""
    t = float(np.clip(amount, 0.0, 1.0))
    start_values = asdict(start)
    end_values = asdict(end)
    output: dict[str, float] = {}
    for name, start_value_raw in start_values.items():
        start_value = float(start_value_raw)
        end_value = float(end_values[name])
        if name in {"left_foot_contact", "right_foot_contact"}:
            output[name] = start_value if t < 0.5 else end_value
        elif name.endswith("_angle_deg") or name == "twist_deg":
            delta = (end_value - start_value + 180.0) % 360.0 - 180.0
            output[name] = start_value + delta * t
        else:
            output[name] = start_value + (end_value - start_value) * t
    return MascotRigPose(**output)


def resampled_targets(
    keyframes: list[tuple[int, MascotRigPose]],
    *,
    source_frame_count: int,
) -> list[tuple[int, MascotRigPose]]:
    """Expand sparse keyframes to every source frame with linear interpolation."""
    if not keyframes:
        return []
    final_frame = max(source_frame_count, keyframes[-1][0] + 1)
    output: list[tuple[int, MascotRigPose]] = []
    segment = 0
    for frame_index in range(keyframes[0][0], final_frame):
        while segment + 1 < len(keyframes) and frame_index > keyframes[segment + 1][0]:
            segment += 1
        start_index, start_pose = keyframes[segment]
        if segment + 1 >= len(keyframes):
            output.append((frame_index, start_pose))
            continue
        end_index, end_pose = keyframes[segment + 1]
        amount = (frame_index - start_index) / max(end_index - start_index, 1)
        output.append((frame_index, interpolate_pose(start_pose, end_pose, amount)))
    return output
