# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Analyze keypoint inference on a short video clip and report quality issues."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from rfdetr.visualize.keypoints import key_points_for_display
from rfdetr_demo.inference.models import build_keypoint_model
from rfdetr_demo.inference.overlays.keypoint import KeypointOverlaySettings, render_keypoint_overlay
from rfdetr_demo.inference.temporal import KeypointTemporalFilter, MotionPlausibilitySettings
from rfdetr_demo.inference.uncertainty import COCO17_KEYPOINT_NAMES
from rfdetr_demo.inference.video_io import probe_video_size
from rfdetr_demo.paths import resolve_default_source
from rfdetr_demo.tracking.keypoints_ops import attach_track_ids
from rfdetr_demo.tracking.person_associator import PersonAssociator
from rfdetr_demo.tuning.analyze_clip_issues import derive_issues
from rfdetr_demo.tuning.analyze_clip_types import ClipAnalysis, FrameMetric, JointFrameMetric


def _covariance_trace(key_points: object, det: int, joint: int) -> float | None:
    cov_raw = key_points.data.get("covariance")  # type: ignore[attr-defined]
    if cov_raw is None:
        return None
    cov = np.asarray(cov_raw, dtype=np.float64)
    matrix = cov[det, joint]
    if not np.isfinite(matrix).all():
        return None
    return float(matrix[0, 0] + matrix[1, 1])


def analyze_clip(
    source_path: Path,
    *,
    max_source_seconds: float,
    frame_stride: int,
    threshold: float,
    keypoint_threshold: float,
    motion_settings: MotionPlausibilitySettings,
) -> ClipAnalysis:
    """Run keypoint inference on a short clip and return quality metrics."""
    width, height, fps = probe_video_size(source_path)
    diagonal = float(np.hypot(width, height))
    max_speed_px = motion_settings.max_speed_fraction_per_sec * diagonal

    model = build_keypoint_model()
    temporal_filter = KeypointTemporalFilter(
        motion_settings,
        frame_width=width,
        frame_height=height,
        fps=fps,
        frame_stride=frame_stride,
    )
    associator = PersonAssociator()
    overlay = KeypointOverlaySettings(
        keypoint_threshold=keypoint_threshold,
        uncertainty_enabled=True,
        uncertainty_style="heatmap",
        frame_width=width,
        frame_height=height,
        motion=motion_settings,
    )

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {source_path}")

    max_source_frames = max(1, int(max_source_seconds * fps))
    frame_metrics: list[FrameMetric] = []
    prev_xy: dict[tuple[int, int], np.ndarray] = {}
    prev_frame_index: int | None = None
    person_counts: list[int] = []

    frame_index = 0
    while frame_index < max_source_frames:
        success, frame_bgr = capture.read()
        if not success:
            break
        if frame_index % frame_stride != 0:
            frame_index += 1
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        key_points = model.predict(frame_rgb, threshold=threshold, include_source_image=False)
        raw_count = len(key_points)
        track_ids = associator.assign(key_points)
        key_points = attach_track_ids(key_points, track_ids)
        key_points = temporal_filter.apply(key_points, frame_index)
        display = key_points_for_display(key_points, keypoint_threshold=keypoint_threshold)
        person_counts.append(len(display))

        dt_sec = (
            (frame_index - prev_frame_index) / fps
            if prev_frame_index is not None
            else frame_stride / fps
        )
        dt_sec = max(dt_sec, 1e-6)

        joints: list[JointFrameMetric] = []
        max_speed = 0.0
        low_conf = 0
        for det_index in range(len(display)):
            conf_arr = display.keypoint_confidence
            visible = display.visible
            for joint_index, (x, y) in enumerate(display.xy[det_index]):
                if visible is not None and not visible[det_index, joint_index]:
                    continue
                if np.allclose((x, y), 0):
                    continue
                conf = float(conf_arr[det_index, joint_index]) if conf_arr is not None else 0.0
                if conf < 0.5:
                    low_conf += 1
                displacement = None
                speed = None
                key = (det_index, joint_index)
                current = np.array([x, y], dtype=np.float64)
                if key in prev_xy:
                    displacement = float(np.linalg.norm(current - prev_xy[key]))
                    speed = displacement / dt_sec
                    max_speed = max(max_speed, speed)
                prev_xy[key] = current.copy()
                joints.append(
                    JointFrameMetric(
                        joint_index=joint_index,
                        joint_name=COCO17_KEYPOINT_NAMES[joint_index],
                        x=float(x),
                        y=float(y),
                        confidence=conf,
                        speed_px_per_sec=speed,
                        displacement_px=displacement,
                        covariance_trace=_covariance_trace(display, det_index, joint_index),
                    ),
                )

        frame_metrics.append(
            FrameMetric(
                frame_index=frame_index,
                person_count=raw_count,
                joints=joints,
                max_joint_speed=max_speed if joints else None,
                low_confidence_joints=low_conf,
            ),
        )
        prev_frame_index = frame_index
        render_keypoint_overlay(frame_bgr, key_points, overlay)
        frame_index += 1

    capture.release()

    analysis = ClipAnalysis(
        source=str(source_path.resolve()),
        duration_sec=max_source_seconds,
        fps=fps,
        frame_stride=frame_stride,
        frames_analyzed=len(frame_metrics),
        avg_persons=float(np.mean(person_counts)) if person_counts else 0.0,
        motion_speed_rejections=temporal_filter.stats.speed_rejections,
        motion_oscillation_corrections=temporal_filter.stats.oscillation_corrections,
        frame_metrics=frame_metrics,
    )
    analysis.issues = derive_issues(analysis, max_speed_px=max_speed_px, fps=fps)
    return analysis


def main(argv: list[str] | None = None) -> int:
    """CLI entry for short-clip analysis (also used by ``rfdetr-demo analyze-clip``)."""
    parser = argparse.ArgumentParser(description="Analyze keypoint quality on a short clip.")
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--keypoint-threshold", type=float, default=0.0)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    source = args.source if args.source is not None else resolve_default_source()
    motion = MotionPlausibilitySettings(enabled=True)
    analysis = analyze_clip(
        source,
        max_source_seconds=args.seconds,
        frame_stride=args.frame_stride,
        threshold=args.threshold,
        keypoint_threshold=args.keypoint_threshold,
        motion_settings=motion,
    )

    report = {
        "source": analysis.source,
        "duration_sec": analysis.duration_sec,
        "fps": analysis.fps,
        "frames_analyzed": analysis.frames_analyzed,
        "avg_persons": round(analysis.avg_persons, 2),
        "motion_speed_rejections": analysis.motion_speed_rejections,
        "motion_oscillation_corrections": analysis.motion_oscillation_corrections,
        "issues": analysis.issues,
        "peak_speeds_by_frame": [
            {"frame": f.frame_index, "max_speed_px_s": f.max_joint_speed, "persons": f.person_count}
            for f in analysis.frame_metrics
            if f.max_joint_speed is not None
        ],
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
