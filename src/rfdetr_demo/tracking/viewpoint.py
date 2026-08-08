# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Heuristic camera-viewpoint (overhead vs eye-level) estimation for FlowCount.

Distinguishes overhead/high-mounted cameras from eye-level cameras using two
signals derived from a sample of person bounding boxes, per the heuristic
outlined in ``FlowCount/README.md``:

- **box aspect ratio**: bird's-eye views show near-square person silhouettes;
  eye-level views show tall/narrow silhouettes.
- **size vs. vertical-position correlation**: in eye-level (perspective) shots,
  people lower in the frame (closer to the camera) have larger boxes; in
  overhead shots, box size is roughly independent of frame position.

Calibration note (from real footage, RF-DETR large): aspect ratio alone is
not a reliable discriminator — production detectors keep regressing "tall"
boxes even for people seen from directly above (median ratios of 1.7-1.9 on
a genuine bird's-eye scramble-crossing clip, vs. 2.8-2.9 for eye-level/
moderately-elevated clips — tall in both cases, just less so overhead).
Correlation tracked the expected pattern far more cleanly (~0.8+ for
eye-level/moderate vs. ~0.2 for true overhead) and is therefore treated as
the primary signal: it alone decides the viewpoint, while aspect ratio only
raises or lowers confidence depending on whether it agrees.

The resulting estimate maps to one of the FlowCount presets so a caller can
auto-select ``overhead`` or ``eye-level`` instead of guessing.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Literal, Sequence

from rfdetr_demo.inference.types import COCO_PERSON_CLASS_ID

CameraViewpoint = Literal["overhead", "eye_level"]

_DEFAULT_ASPECT_RATIO_THRESHOLD = 2.2
_DEFAULT_CORRELATION_THRESHOLD = 0.35
_MIN_SAMPLE_COUNT = 2


@dataclass(frozen=True)
class ViewpointEstimate:
    """Result of :func:`estimate_camera_viewpoint`.

    Attributes:
        viewpoint: Estimated camera viewpoint.
        confidence: Agreement between the aspect-ratio and correlation signals,
            in ``{0.0, 0.5, 1.0}`` (0.0 when there were too few boxes to judge).
        median_aspect_ratio: Median height/width ratio of the sampled boxes.
        size_position_correlation: Pearson correlation between box area and
            normalized vertical center (0.0 when it could not be computed).
        sample_count: Number of boxes used (after discarding degenerate ones).
    """

    viewpoint: CameraViewpoint
    confidence: float
    median_aspect_ratio: float
    size_position_correlation: float
    sample_count: int


def estimate_camera_viewpoint(
    boxes: Sequence[tuple[float, float, float, float]],
    frame_height: float,
    *,
    aspect_ratio_threshold: float = _DEFAULT_ASPECT_RATIO_THRESHOLD,
    correlation_threshold: float = _DEFAULT_CORRELATION_THRESHOLD,
) -> ViewpointEstimate:
    """Estimate whether person boxes come from an overhead or eye-level camera.

    Args:
        boxes: Person bounding boxes as ``(x1, y1, x2, y2)`` tuples, ideally
            sampled across several frames of a clip.
        frame_height: Height of the video frame in pixels, used to normalize
            box vertical position.
        aspect_ratio_threshold: Median height/width ratio above which boxes
            are considered "tall" (an eye-level signal; secondary — see
            module docstring).
        correlation_threshold: Size-vs-position correlation above which boxes
            are considered "perspective-scaled" (an eye-level signal;
            primary — decides the viewpoint on its own).

    Returns:
        A :class:`ViewpointEstimate` with the classified viewpoint and the
        underlying signal values.
    """
    aspect_ratios: list[float] = []
    positions: list[float] = []
    areas: list[float] = []

    for x1, y1, x2, y2 in boxes:
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            continue
        aspect_ratios.append(height / width)
        areas.append(width * height)
        positions.append(((y1 + y2) / 2) / frame_height if frame_height > 0 else 0.0)

    sample_count = len(aspect_ratios)
    if sample_count < _MIN_SAMPLE_COUNT:
        return ViewpointEstimate(
            viewpoint="overhead",
            confidence=0.0,
            median_aspect_ratio=aspect_ratios[0] if aspect_ratios else 0.0,
            size_position_correlation=0.0,
            sample_count=sample_count,
        )

    median_aspect_ratio = statistics.median(aspect_ratios)
    try:
        correlation = statistics.correlation(positions, areas)
    except statistics.StatisticsError:
        correlation = 0.0

    # Correlation is the primary signal (see module docstring): it alone
    # decides the viewpoint. Aspect ratio only confirms or disputes it,
    # raising confidence to 1.0 when both agree and dropping it to 0.5 when
    # they disagree rather than being allowed to override the decision.
    eye_level_from_correlation = correlation > correlation_threshold
    eye_level_from_aspect = median_aspect_ratio > aspect_ratio_threshold
    viewpoint: CameraViewpoint = "eye_level" if eye_level_from_correlation else "overhead"
    confidence = 1.0 if eye_level_from_aspect == eye_level_from_correlation else 0.5

    return ViewpointEstimate(
        viewpoint=viewpoint,
        confidence=confidence,
        median_aspect_ratio=median_aspect_ratio,
        size_position_correlation=correlation,
        sample_count=sample_count,
    )


def estimate_viewpoint_from_frames(
    frames: Sequence[Any],
    model: Any,
    *,
    threshold: float = 0.4,
) -> ViewpointEstimate:
    """Run a person-detection model over sampled frames and estimate the camera viewpoint.

    Uses the same detection model class as the real counting pipeline (e.g.
    ``build_detection_model``), not the keypoint-preview model: its box
    regression is what the rest of FlowCount actually sees, and in practice
    its statistics track the overhead/eye-level split more cleanly (see the
    calibration note in the module docstring).

    Args:
        frames: RGB frames (``H x W x 3`` arrays) sampled from a clip.
        model: A detection model exposing ``predict(frame, threshold=..., include_source_image=...)``
            and returning an ``sv.Detections``-like object with ``xyxy``/``class_id``.
        threshold: Detection confidence threshold passed to the model.

    Returns:
        A :class:`ViewpointEstimate` aggregated over every person box found
        across all frames.
    """
    boxes: list[tuple[float, float, float, float]] = []
    frame_height = 0.0
    for frame in frames:
        frame_height = max(frame_height, float(frame.shape[0]))
        detections = model.predict(frame, threshold=threshold, include_source_image=False)
        if isinstance(detections, list) or detections.class_id is None:
            continue
        person_mask = detections.class_id == COCO_PERSON_CLASS_ID
        for x1, y1, x2, y2 in detections.xyxy[person_mask]:
            boxes.append((float(x1), float(y1), float(x2), float(y2)))
    return estimate_camera_viewpoint(boxes, frame_height=frame_height)


def preset_for_viewpoint(estimate: ViewpointEstimate) -> Literal["overhead", "eye-level"]:
    """Map a :class:`ViewpointEstimate` to a FlowCount preset name.

    Args:
        estimate: Result of :func:`estimate_camera_viewpoint`.

    Returns:
        ``"overhead"`` or ``"eye-level"``, matching the preset names documented
        in ``FlowCount/README.md``.
    """
    return "overhead" if estimate.viewpoint == "overhead" else "eye-level"
