# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Run confidential center-person tracking audits over a video."""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path

import cv2

from rfdetr_demo.inference.models import build_keypoint_model
from rfdetr_demo.inference.overlays.keypoint import KeypointOverlaySettings, render_keypoint_overlay
from rfdetr_demo.inference.temporal import KeypointTemporalFilter, MotionPlausibilitySettings
from rfdetr_demo.inference.video_io import probe_video_size
from rfdetr_demo.media.audit.common import (
    append_jsonl,
    build_base_audit_line,
    make_run_id,
    repo_relpath,
    save_audit_image,
    write_summary_json,
)
from rfdetr_demo.media.audit.tracking_annotate import (
    annotate_track_labels,
    center_x_range,
    find_center_track,
    raw_diagnostics,
)
from rfdetr_demo.media.audit.tracking_eval import evaluate_center_tracking
from rfdetr_demo.media.audit.tracking_types import (
    DEFAULT_CENTER_X_FRACTION,
    TrackingAuditSummary,
    TrackingFrameRecord,
)
from rfdetr_demo.paths import CONFIDENTIAL_AUDIT, resolve_default_source
from rfdetr_demo.tracking.bbox import nms_detection_indices
from rfdetr_demo.tracking.pipeline import PersonTrackPipeline
from rfdetr_demo.tracking.types import PersonTrackSettings

logger = logging.getLogger(__name__)

_TRACKING_AUDIT_ROOT = CONFIDENTIAL_AUDIT / "tracking-runs"
_TRACKING_AUDIT_JSONL = CONFIDENTIAL_AUDIT / "tracking-audit.jsonl"


def run_center_tracking_audit(
    *,
    source_path: Path | None = None,
    sample_interval: int = 20,
    threshold: float = 0.6,
    keypoint_threshold: float = 0.25,
    max_frames: int | None = None,
    save_on_center_event: bool = True,
    run_id: str | None = None,
) -> TrackingAuditSummary:
    """Process a video and log confidential tracking diagnostics every ``sample_interval`` frames."""
    source = (source_path or resolve_default_source()).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Source video not found: {source}")

    frame_width, frame_height, fps = probe_video_size(source)
    x_lane = center_x_range(frame_width)

    audit_run_id = run_id or make_run_id()
    run_dir = _TRACKING_AUDIT_ROOT / audit_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    model = build_keypoint_model()
    motion = MotionPlausibilitySettings(enabled=True)
    temporal_filter = KeypointTemporalFilter(
        motion,
        frame_width=frame_width,
        frame_height=frame_height,
        fps=fps,
        frame_stride=1,
    )
    track_pipeline = PersonTrackPipeline.from_env(
        frame_width=frame_width,
        frame_height=frame_height,
        temporal_filter=temporal_filter,
        base_settings=PersonTrackSettings(enabled=True),
    )
    overlay_settings = KeypointOverlaySettings(
        keypoint_threshold=keypoint_threshold,
        uncertainty_enabled=True,
        uncertainty_style="heatmap",
        frame_width=frame_width,
        frame_height=frame_height,
        motion=motion,
    )

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {source}")

    records: list[TrackingFrameRecord] = []
    images_saved = 0
    processed_count = 0
    prev_center_present: bool | None = None
    frame_index = 0
    source_rel = repo_relpath(source)

    try:
        while True:
            if max_frames is not None and processed_count >= max_frames:
                break
            ok, frame_bgr = capture.read()
            if not ok:
                break

            processed_count += 1
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            raw_key_points = model.predict(
                frame_rgb,
                threshold=threshold,
                include_source_image=False,
            )
            raw_rows = raw_diagnostics(raw_key_points, frame_width)
            nms_keep = nms_detection_indices(
                raw_key_points,
                track_pipeline.settings.nms_iou_threshold,
            )

            stabilized = track_pipeline.apply(raw_key_points, frame_index)
            key_points = stabilized.key_points

            center_raw = any(row.in_center_lane for row in raw_rows)
            center_track = find_center_track(stabilized.diagnostics, frame_width)
            center_present = center_track is not None
            center_is_ghost = bool(center_track and center_track.is_ghost)

            center_changed = (
                prev_center_present is not None and center_present != prev_center_present
            )
            should_save = (
                frame_index % sample_interval == 0
                or (save_on_center_event and center_changed)
                or not center_present
            )

            image_relpath: str | None = None
            if should_save:
                annotated = render_keypoint_overlay(frame_bgr, key_points, overlay_settings)
                annotated = annotate_track_labels(annotated, stabilized.diagnostics)
                cv2.rectangle(
                    annotated,
                    (int(x_lane[0]), 0),
                    (int(x_lane[1]), frame_height - 1),
                    (255, 128, 0),
                    1,
                )
                cv2.putText(
                    annotated,
                    f"frame={frame_index} raw={len(raw_key_points)} active={len(key_points)}",
                    (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                image_name = f"frame_{processed_count:04d}_src{frame_index:05d}.jpg"
                image_path = run_dir / image_name
                if save_audit_image(image_path, annotated):
                    image_relpath = repo_relpath(image_path)
                    images_saved += 1

            record = TrackingFrameRecord(
                frame_index=frame_index,
                processed_count=processed_count,
                raw_count=len(raw_key_points),
                nms_count=len(nms_keep),
                active_count=len(stabilized.key_points),
                ghost_count=stabilized.stats.ghost_count,
                center_present_raw=center_raw,
                center_present_stabilized=center_present,
                center_track_id=center_track.track_id if center_track else None,
                center_is_ghost=center_is_ghost,
                center_confidence=center_track.confidence if center_track else None,
                image_saved=image_relpath is not None,
                image_relpath=image_relpath,
                raw_detections=raw_rows,
                stabilized_tracks=list(stabilized.diagnostics),
            )
            records.append(record)

            append_jsonl(
                _TRACKING_AUDIT_JSONL,
                build_base_audit_line(
                    audit_kind="tracking",
                    run_id=audit_run_id,
                    source_relpath=source_rel,
                    frame_index=frame_index,
                    processed_count=processed_count,
                    image_relpath=image_relpath,
                    raw_count=record.raw_count,
                    nms_count=record.nms_count,
                    active_count=record.active_count,
                    ghost_count=record.ghost_count,
                    center_present_raw=center_raw,
                    center_present_stabilized=center_present,
                    center_track_id=record.center_track_id,
                    center_is_ghost=center_is_ghost,
                    center_lane_x=[x_lane[0], x_lane[1]],
                    raw_centroids_x=[round(row.cx, 1) for row in raw_rows],
                    track_diagnostics=[asdict(row) for row in stabilized.diagnostics],
                ),
            )

            prev_center_present = center_present
            frame_index += 1
    finally:
        capture.release()

    evaluation = evaluate_center_tracking(records)
    summary = TrackingAuditSummary(
        run_id=audit_run_id,
        source_relpath=source_rel,
        frames_processed=len(records),
        frames_logged=len(records),
        images_saved=images_saved,
        run_dir_relpath=repo_relpath(run_dir),
        evaluation=evaluation,
        records=records,
    )

    write_summary_json(
        run_dir / "summary.json",
        {
            "run_id": summary.run_id,
            "source_relpath": summary.source_relpath,
            "audit_kind": "tracking",
            "sample_interval": sample_interval,
            "threshold": threshold,
            "center_lane_x_fraction": list(DEFAULT_CENTER_X_FRACTION),
            "frames_processed": summary.frames_processed,
            "images_saved": summary.images_saved,
            "evaluation": evaluation,
            "frames": [
                {
                    "frame_index": row.frame_index,
                    "processed_count": row.processed_count,
                    "raw_count": row.raw_count,
                    "nms_count": row.nms_count,
                    "active_count": row.active_count,
                    "ghost_count": row.ghost_count,
                    "center_present_raw": row.center_present_raw,
                    "center_present_stabilized": row.center_present_stabilized,
                    "center_track_id": row.center_track_id,
                    "center_is_ghost": row.center_is_ghost,
                    "center_confidence": row.center_confidence,
                    "image_relpath": row.image_relpath,
                    "raw_centroids_x": [round(det.cx, 1) for det in row.raw_detections],
                    "track_ids": [track.track_id for track in row.stabilized_tracks],
                }
                for row in records
            ],
        },
    )
    logger.info(
        "tracking_audit_complete run_id=%s frames=%s images=%s verdict=%s",
        audit_run_id,
        summary.frames_processed,
        summary.images_saved,
        evaluation.get("verdict"),
    )
    return summary
