# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Core video inference runner (model load, callback wiring, I/O)."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from threading import Event
from typing import Any

from rfdetr_demo.inference.overlays.keypoint import resolve_uncertainty_max_axis
from rfdetr_demo.inference.progress import compose_preview_callback, compose_progress_callback
from rfdetr_demo.inference.task_callback import build_task_callback
from rfdetr_demo.inference.types import (
    KeypointUncertaintyStyle,
    ModelSize,
    PreviewCallback,
    ProgressCallback,
    TaskName,
    VideoProcessingCancelledError,
)
from rfdetr_demo.inference.video_io import (
    cleanup_partial_video,
    finalize_video_path,
    partial_video_path,
    probe_video_size,
    process_video,
)
from rfdetr_demo.media.frame_audit import (
    ConfidentialFrameAuditLogger,
    is_frame_audit_enabled,
    wrap_callback_with_frame_audit,
)

logger = logging.getLogger(__name__)


def run_demo(
    source_path: Path,
    target_path: Path,
    task: TaskName,
    model_size: ModelSize,
    threshold: float,
    frame_stride: int,
    max_frames: int | None,
    person_only: bool,
    keypoint_threshold: float = 0.25,
    keypoint_uncertainty_style: KeypointUncertaintyStyle = "none",
    ellipse_sigma: float = 1.5,
    max_ellipse_axis: float | None = None,
    progress_callback: ProgressCallback | None = None,
    preview_callback: PreviewCallback | None = None,
    preview_stride: int | None = None,
    preview_min_interval_sec: float = 0.12,
    preview_max_width: int = 400,
    cancel_event: Event | None = None,
    progress_file: Path | None = None,
    max_source_seconds: float | None = None,
    start_source_seconds: float | None = None,
    tune_cache: Any | None = None,
    heatmap_opacity: float = 0.38,
    heatmap_decay: float = 3.0,
    vertex_radius: int = 4,
    keypoint_uncertainty_enabled: bool = True,
    motion_settings: Any | None = None,
    frame_audit_log_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Process ``source_path`` and write annotated video to ``target_path``."""
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Source video not found: {source_path}. "
            "Place a video under confidential/media/input/ or pass --source."
        )
    if frame_stride < 1:
        raise ValueError(f"frame_stride must be >= 1, got {frame_stride}")
    if max_source_seconds is not None and max_source_seconds <= 0:
        raise ValueError(f"max_source_seconds must be > 0, got {max_source_seconds}")
    if start_source_seconds is not None and start_source_seconds < 0:
        raise ValueError(f"start_source_seconds must be >= 0, got {start_source_seconds}")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = partial_video_path(target_path)

    stats: dict[str, int] = {
        "processed_frames": 0,
        "total_detections": 0,
        "frame_raw_detections": 0,
        "frame_nms_detections": 0,
        "frame_active_tracks": 0,
        "frame_ghost_tracks": 0,
        "frame_live_tracks": 0,
        "motion_speed_rejections": 0,
        "motion_covariance_rejections": 0,
        "motion_oscillation_corrections": 0,
        "motion_smoothed_joints": 0,
    }
    frame_width, frame_height, video_fps = probe_video_size(source_path)
    resolved_max_axis = (
        resolve_uncertainty_max_axis(
            max_ellipse_axis,
            frame_width=frame_width,
            frame_height=frame_height,
            style=keypoint_uncertainty_style,
        )
        if task == "keypoint" and keypoint_uncertainty_style != "none"
        else None
    )

    callback = build_task_callback(
        task=task,
        model_size=model_size,
        threshold=threshold,
        person_only=person_only,
        stats=stats,
        frame_width=frame_width,
        frame_height=frame_height,
        video_fps=video_fps,
        frame_stride=frame_stride,
        keypoint_threshold=keypoint_threshold,
        keypoint_uncertainty_style=keypoint_uncertainty_style,
        keypoint_uncertainty_enabled=keypoint_uncertainty_enabled,
        ellipse_sigma=ellipse_sigma,
        max_ellipse_axis=max_ellipse_axis,
        heatmap_opacity=heatmap_opacity,
        heatmap_decay=heatmap_decay,
        vertex_radius=vertex_radius,
        motion_settings=motion_settings,
        tune_cache=tune_cache,
    )

    frame_audit: ConfidentialFrameAuditLogger | None = None
    if is_frame_audit_enabled():
        frame_audit = ConfidentialFrameAuditLogger(source_path=source_path, task=task)

        def _on_frame_audited(record: Any) -> None:
            if frame_audit_log_callback is None:
                return
            line = (
                f"[機密] #{record.processed_count:02d} "
                f"src={record.frame_index} 検出={record.instance_count} "
                f"→ {record.image_relpath}"
            )
            frame_audit_log_callback(line)

        callback = wrap_callback_with_frame_audit(
            callback,
            audit=frame_audit,
            stats=stats,
            on_frame_logged=_on_frame_audited if frame_audit_log_callback else None,
        )

    started = time.perf_counter()
    if partial_path.exists():
        cleanup_partial_video(partial_path)

    combined_progress = compose_progress_callback(progress_callback, progress_file)
    preview_throttle = compose_preview_callback(
        preview_callback,
        preview_stride=preview_stride,
        preview_min_interval_sec=preview_min_interval_sec,
        preview_max_width=preview_max_width,
        frame_stride=frame_stride,
    )

    try:
        process_video(
            source_path=source_path,
            target_path=partial_path,
            callback=callback,
            frame_stride=frame_stride,
            max_frames=max_frames,
            max_source_seconds=max_source_seconds,
            start_source_seconds=start_source_seconds,
            stats=stats,
            progress_callback=combined_progress,
            preview_throttle=preview_throttle,
            cancel_event=cancel_event,
        )
        finalize_video_path(partial_path, target_path)
    except VideoProcessingCancelledError:
        cleanup_partial_video(partial_path)
        raise
    except Exception:
        cleanup_partial_video(partial_path)
        raise
    elapsed_sec = time.perf_counter() - started

    summary: dict[str, Any] = {
        "source": str(source_path.resolve()),
        "target": str(target_path.resolve()),
        "task": task,
        "model_size": model_size if task in {"detect", "segment"} else "keypoint-preview",
        "threshold": threshold,
        "frame_stride": frame_stride,
        "person_only": person_only if task in {"detect", "segment"} else None,
        "keypoint_threshold": keypoint_threshold if task == "keypoint" else None,
        "keypoint_uncertainty_style": keypoint_uncertainty_style if task == "keypoint" else None,
        "ellipse_sigma": ellipse_sigma if task == "keypoint" else None,
        "max_ellipse_axis": resolved_max_axis if task == "keypoint" else None,
        "heatmap_opacity": heatmap_opacity if task == "keypoint" else None,
        "heatmap_decay": heatmap_decay if task == "keypoint" else None,
        "vertex_radius": vertex_radius if task == "keypoint" else None,
        "motion_speed_rejections": stats.get("motion_speed_rejections", 0),
        "motion_oscillation_corrections": stats.get("motion_oscillation_corrections", 0),
        "processed_frames": stats["processed_frames"],
        "total_detections": stats["total_detections"],
        "elapsed_sec": round(elapsed_sec, 2),
    }
    if max_source_seconds is not None:
        summary["max_source_seconds"] = max_source_seconds
        summary["tune_preview"] = True
    if start_source_seconds is not None and start_source_seconds > 0:
        summary["start_source_seconds"] = start_source_seconds
    if stats["processed_frames"] > 0:
        summary["avg_detections_per_frame"] = round(
            stats["total_detections"] / stats["processed_frames"],
            2,
        )
        summary["avg_fps"] = round(stats["processed_frames"] / elapsed_sec, 2)
    if frame_audit is not None:
        audit_summary = frame_audit.finalize()
        if audit_summary is not None:
            summary["frame_audit"] = {
                "run_id": audit_summary.run_id,
                "run_dir_relpath": audit_summary.run_dir_relpath,
                "frames_logged": audit_summary.frames_logged,
                "evaluation": audit_summary.evaluation,
            }
    return summary


__all__ = ["run_demo"]
