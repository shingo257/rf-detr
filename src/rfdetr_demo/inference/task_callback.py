# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Assemble per-task inference callbacks for the video runner."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import supervision as sv

from rfdetr_demo.inference.callbacks import (
    make_detection_callback,
    make_keypoint_callback,
    make_segmentation_callback,
)
from rfdetr_demo.inference.models import (
    build_detection_model,
    build_keypoint_model,
    build_segmentation_model,
)
from rfdetr_demo.inference.overlays.keypoint import KeypointOverlaySettings
from rfdetr_demo.inference.temporal import MotionPlausibilitySettings
from rfdetr_demo.inference.types import KeypointUncertaintyStyle, ModelSize, TaskName


def default_motion_settings() -> MotionPlausibilitySettings:
    """Build motion settings from tuning defaults."""
    from rfdetr_demo.tuning.auto_tune import DEFAULT_PARAMETERS

    defaults = DEFAULT_PARAMETERS
    return MotionPlausibilitySettings(
        enabled=defaults.motion_filter_enabled,
        max_speed_fraction_per_sec=defaults.motion_max_speed_fraction,
        ema_alpha=defaults.motion_ema_alpha,
        suppress_oscillation=defaults.motion_oscillation_enabled,
    )


def build_task_callback(
    *,
    task: TaskName,
    model_size: ModelSize,
    threshold: float,
    person_only: bool,
    stats: dict[str, int],
    frame_width: int,
    frame_height: int,
    video_fps: float,
    frame_stride: int,
    keypoint_threshold: float,
    keypoint_uncertainty_style: KeypointUncertaintyStyle,
    keypoint_uncertainty_enabled: bool,
    ellipse_sigma: float,
    max_ellipse_axis: float | None,
    heatmap_opacity: float,
    heatmap_decay: float,
    vertex_radius: int,
    motion_settings: Any | None,
    tune_cache: Any | None,
) -> Callable[[Any, int], Any]:
    """Return the per-frame callback for ``task``."""
    if task == "keypoint":
        model = build_keypoint_model()
        if motion_settings is None:
            motion_settings = default_motion_settings()
        overlay_style: KeypointUncertaintyStyle = (
            keypoint_uncertainty_style
            if keypoint_uncertainty_enabled and keypoint_uncertainty_style != "none"
            else "none"
        )
        overlay_settings = KeypointOverlaySettings(
            keypoint_threshold=keypoint_threshold,
            uncertainty_enabled=keypoint_uncertainty_enabled,
            uncertainty_style=overlay_style,
            ellipse_sigma=ellipse_sigma,
            max_ellipse_axis=max_ellipse_axis,
            heatmap_opacity=heatmap_opacity,
            heatmap_decay=heatmap_decay,
            vertex_radius=vertex_radius,
            frame_width=frame_width,
            frame_height=frame_height,
            motion=motion_settings,
        )
        temporal_filter = None
        person_track_pipeline = None
        if motion_settings is not None and motion_settings.enabled:
            from rfdetr_demo.inference.temporal import KeypointTemporalFilter

            temporal_filter = KeypointTemporalFilter(
                motion_settings,
                frame_width=frame_width,
                frame_height=frame_height,
                fps=video_fps,
                frame_stride=frame_stride,
            )
        from rfdetr_demo.tracking.pipeline import PersonTrackPipeline
        from rfdetr_demo.tracking.types import PersonTrackSettings, is_person_track_enabled

        if is_person_track_enabled():
            person_track_pipeline = PersonTrackPipeline.from_env(
                frame_width=frame_width,
                frame_height=frame_height,
                temporal_filter=temporal_filter,
                base_settings=PersonTrackSettings(enabled=True),
            )
            temporal_filter = None
        return make_keypoint_callback(
            model,
            threshold,
            overlay_settings,
            stats,
            tune_cache=tune_cache,
            temporal_filter=temporal_filter,
            person_track_pipeline=person_track_pipeline,
        )

    if task == "segment":
        model = build_segmentation_model(model_size)
        return make_segmentation_callback(
            model,
            threshold,
            person_only,
            sv.MaskAnnotator(),
            sv.LabelAnnotator(),
            stats,
            tune_cache=tune_cache,
        )

    model = build_detection_model(model_size)
    return make_detection_callback(
        model,
        threshold,
        person_only,
        sv.BoxAnnotator(),
        sv.LabelAnnotator(),
        stats,
        tune_cache=tune_cache,
    )


__all__ = ["build_task_callback", "default_motion_settings"]
