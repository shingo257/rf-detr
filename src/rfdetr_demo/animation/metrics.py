"""Sequence-level quality metrics for torso animation controls."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rfdetr_demo.animation.torso import TorsoControls


@dataclass(frozen=True, slots=True)
class TorsoSequenceMetrics:
    """Compact quality summary for one tracked torso sequence."""

    frame_count: int
    dropout_rate: float
    jitter_rms_fraction: float
    mean_confidence: float


def evaluate_torso_sequence(
    frames: list[TorsoControls],
    *,
    point_name: str = "pelvis_center",
) -> TorsoSequenceMetrics:
    """Measure dropout and second-difference jitter normalized by torso height."""
    if not frames:
        return TorsoSequenceMetrics(0, 0.0, 0.0, 0.0)
    if any(point_name not in frame.points for frame in frames):
        raise ValueError(f"point_name {point_name!r} is missing from one or more frames")

    visible_count = sum(frame.points[point_name].visible for frame in frames)
    jitter_samples: list[float] = []
    for previous, current, following in zip(frames, frames[1:], frames[2:]):
        points = [item.points[point_name] for item in (previous, current, following)]
        if not all(point.visible for point in points):
            continue
        acceleration = points[2].xy - 2.0 * points[1].xy + points[0].xy
        torso_height = max(current.parameters.get("torso_height_px", 0.0), 1e-6)
        jitter_samples.append(float(np.linalg.norm(acceleration)) / torso_height)
    jitter_rms = float(np.sqrt(np.mean(np.square(jitter_samples)))) if jitter_samples else 0.0
    return TorsoSequenceMetrics(
        frame_count=len(frames),
        dropout_rate=1.0 - visible_count / len(frames),
        jitter_rms_fraction=jitter_rms,
        mean_confidence=float(np.mean([frame.confidence for frame in frames])),
    )
