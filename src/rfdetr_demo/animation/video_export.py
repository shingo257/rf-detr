"""Export stabilized torso controls and a diagnostic overlay from video."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from rfdetr_demo.animation.metrics import evaluate_torso_sequence
from rfdetr_demo.animation.pipeline import TorsoControlPipeline, TrackedTorsoControls, torso_frame_as_dict
from rfdetr_demo.animation.torso import TORSO_CONTROL_NAMES, TorsoControls


def render_torso_controls(
    frame_bgr: np.ndarray,
    people: list[TrackedTorsoControls],
) -> np.ndarray:
    """Draw spine, rib, waist, and pelvis controls on a diagnostic frame."""
    annotated = frame_bgr.copy()
    for person in people:
        controls = person.controls
        if person.is_ghost or not controls.points["pelvis_center"].visible:
            continue

        def pixel(name: str) -> tuple[int, int]:
            xy = controls.points[name].xy
            return int(round(float(xy[0]))), int(round(float(xy[1])))

        spine = np.asarray(
            [pixel(name) for name in ("neck_base", "spine_upper", "spine_mid", "spine_lower", "pelvis_center")],
            dtype=np.int32,
        )
        cv2.polylines(annotated, [spine], False, (255, 220, 0), 3, cv2.LINE_AA)
        cv2.line(annotated, pixel("left_lower_rib"), pixel("right_lower_rib"), (0, 220, 255), 2, cv2.LINE_AA)
        cv2.line(annotated, pixel("left_waist"), pixel("right_waist"), (255, 80, 220), 2, cv2.LINE_AA)
        for name in TORSO_CONTROL_NAMES:
            point = controls.points[name]
            if point.visible:
                cv2.circle(annotated, pixel(name), 4, (70, 255, 120), -1, cv2.LINE_AA)
        label_point = pixel("neck_base")
        label = f"ID {person.track_id} twist {controls.parameters['torso_twist_2d_deg']:+.1f}"
        cv2.putText(
            annotated,
            label,
            (label_point[0] + 8, max(18, label_point[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (70, 255, 120),
            1,
            cv2.LINE_AA,
        )
    return annotated


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.stem}.partial{path.suffix}")
    partial.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if path.exists():
        path.unlink()
    partial.replace(path)


def run_torso_animation_export(
    *,
    source_path: Path,
    json_path: Path,
    overlay_path: Path | None = None,
    threshold: float = 0.5,
    keypoint_threshold: float = 0.15,
    frame_stride: int = 1,
    max_frames: int | None = None,
    model: Any | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    """Run RF-DETR keypoints and export animation controls for one video."""
    if not source_path.is_file():
        raise FileNotFoundError(f"Source video not found: {source_path}")
    if frame_stride < 1:
        raise ValueError("frame_stride must be >= 1")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be >= 1")

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video source: {source_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    planned_processed = (source_frame_count + frame_stride - 1) // frame_stride if source_frame_count else 0
    if max_frames is not None:
        planned_processed = min(planned_processed, max_frames) if planned_processed else max_frames
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Invalid video dimensions: {width}x{height}")

    if model is None:
        from rfdetr_demo.inference.models import build_keypoint_model

        model = build_keypoint_model()

    from rfdetr_demo.inference.overlays.keypoint import KeypointOverlaySettings, render_keypoint_overlay
    from rfdetr_demo.inference.temporal import KeypointTemporalFilter, MotionPlausibilitySettings
    from rfdetr_demo.inference.video_io import cleanup_partial_video, finalize_video_path, partial_video_path
    from rfdetr_demo.tracking.pipeline import PersonTrackPipeline
    from rfdetr_demo.tracking.types import PersonTrackSettings

    motion_filter = KeypointTemporalFilter(
        MotionPlausibilitySettings(),
        frame_width=width,
        frame_height=height,
        fps=fps,
        frame_stride=frame_stride,
    )
    tracking = PersonTrackPipeline(
        settings=PersonTrackSettings(enabled=True),
        frame_width=width,
        frame_height=height,
        _temporal_filter=motion_filter,
    )
    torso_pipeline = TorsoControlPipeline(min_confidence=keypoint_threshold)
    overlay_settings = KeypointOverlaySettings(
        keypoint_threshold=keypoint_threshold,
        uncertainty_enabled=False,
        uncertainty_style="none",
        frame_width=width,
        frame_height=height,
    )

    writer: cv2.VideoWriter | None = None
    overlay_partial: Path | None = None
    if overlay_path is not None:
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_partial = partial_video_path(overlay_path)
        cleanup_partial_video(overlay_partial)
        writer = cv2.VideoWriter(
            str(overlay_partial),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"Failed to open video writer: {overlay_partial}")

    frames: list[dict[str, object]] = []
    controls_by_track: dict[int, list[TorsoControls]] = defaultdict(list)
    frame_index = 0
    processed = 0
    last_overlay: np.ndarray | None = None
    try:
        while True:
            success, frame_bgr = capture.read()
            if not success:
                break
            if frame_index % frame_stride == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                raw_key_points = model.predict(frame_rgb, threshold=threshold, include_source_image=False)
                tracked_key_points = tracking.apply(raw_key_points, frame_index).key_points
                people = torso_pipeline.process(tracked_key_points)
                frames.append(torso_frame_as_dict(people, frame_index=frame_index, timestamp_sec=frame_index / fps))
                for person in people:
                    if not person.is_ghost:
                        controls_by_track[person.track_id].append(person.controls)
                last_overlay = render_keypoint_overlay(frame_bgr, tracked_key_points, overlay_settings)
                last_overlay = render_torso_controls(last_overlay, people)
                processed += 1
                if progress_callback is not None:
                    progress_callback(processed, planned_processed)
            if writer is not None:
                writer.write(last_overlay if last_overlay is not None else frame_bgr)
            frame_index += 1
            if max_frames is not None and processed >= max_frames:
                break
    except Exception:
        if overlay_partial is not None:
            cleanup_partial_video(overlay_partial)
        raise
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    if overlay_path is not None and overlay_partial is not None:
        finalize_video_path(overlay_partial, overlay_path)

    metrics = {
        str(track_id): asdict(evaluate_torso_sequence(track_frames))
        for track_id, track_frames in controls_by_track.items()
    }
    payload: dict[str, object] = {
        "schema_version": 1,
        "source": str(source_path.resolve()),
        "fps": fps,
        "width": width,
        "height": height,
        "frame_stride": frame_stride,
        "source_frame_count": source_frame_count,
        "source_frames_read": frame_index,
        "complete_source": source_frame_count <= 0 or frame_index >= source_frame_count,
        "control_point_names": list(TORSO_CONTROL_NAMES),
        "frames": frames,
        "metrics_by_track": metrics,
    }
    _write_json_atomic(json_path, payload)
    return {
        "source": str(source_path.resolve()),
        "json": str(json_path.resolve()),
        "overlay": str(overlay_path.resolve()) if overlay_path is not None else None,
        "processed_frames": processed,
        "track_count": len(controls_by_track),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export torso animation controls from an RF-DETR keypoint video.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--keypoint-threshold", type=float, default=0.15)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = run_torso_animation_export(
            source_path=args.source,
            json_path=args.json,
            overlay_path=args.overlay,
            threshold=args.threshold,
            keypoint_threshold=args.keypoint_threshold,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
