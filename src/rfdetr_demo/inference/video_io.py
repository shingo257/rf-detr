# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Video file I/O helpers for the demo pipeline."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from threading import Event

import cv2
import numpy as np

from rfdetr_demo.inference.progress import PreviewThrottle
from rfdetr_demo.inference.types import ProgressCallback, VideoProcessingCancelledError


def probe_video_size(source_path: Path) -> tuple[int, int, float]:
    """Return width, height, and fps for a video file."""
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video source: {source_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    capture.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video dimensions for {source_path}: {width}x{height}")
    return width, height, float(fps)


def finalize_video_path(partial_path: Path, target_path: Path) -> None:
    """Atomically promote a finished partial MP4 to the final output path."""
    if not partial_path.is_file() or partial_path.stat().st_size == 0:
        raise RuntimeError(f"Partial video output is missing or empty: {partial_path}")
    if target_path.exists():
        target_path.unlink()
    partial_path.replace(target_path)


def cleanup_partial_video(partial_path: Path) -> None:
    """Remove a partial MP4 after a failed export."""
    if partial_path.is_file():
        partial_path.unlink()


def partial_video_path(target_path: Path) -> Path:
    """Return a sidecar path used while encoding is in progress."""
    return target_path.with_name(f"{target_path.stem}.partial{target_path.suffix}")


def effective_source_frame_limit(
    total_frames: int,
    fps: float,
    max_source_seconds: float | None,
    *,
    start_source_seconds: float | None = None,
) -> int:
    """Return absolute end frame index (exclusive) for the clip window.

    When ``start_source_seconds`` is set, the window starts there and
    ``max_source_seconds`` is treated as clip duration (not absolute end time).
    Without a duration limit, returns ``total_frames`` (or 0 if unknown).
    """
    start_frame, end_frame = resolve_clip_window(
        total_frames,
        fps,
        start_source_seconds=start_source_seconds,
        max_source_seconds=max_source_seconds,
    )
    return end_frame


def resolve_clip_window(
    total_frames: int,
    fps: float,
    *,
    start_source_seconds: float | None = None,
    max_source_seconds: float | None = None,
) -> tuple[int, int]:
    """Return ``(start_frame, end_frame_exclusive)`` for the analysis window."""
    if start_source_seconds is not None and start_source_seconds < 0:
        msg = f"start_source_seconds must be >= 0, got {start_source_seconds}"
        raise ValueError(msg)
    if max_source_seconds is not None and max_source_seconds <= 0:
        msg = f"max_source_seconds must be > 0, got {max_source_seconds}"
        raise ValueError(msg)

    start_frame = 0
    if start_source_seconds is not None and start_source_seconds > 0:
        start_frame = max(0, int(start_source_seconds * fps))

    if total_frames > 0 and start_frame >= total_frames:
        start_frame = max(0, total_frames - 1)

    if max_source_seconds is not None:
        duration_frames = max(1, int(max_source_seconds * fps))
        end_frame = start_frame + duration_frames
    elif total_frames > 0:
        end_frame = total_frames
    else:
        # Unknown length and no duration cap: no hard end (caller breaks on EOF).
        end_frame = 0

    if total_frames > 0:
        end_frame = min(end_frame, total_frames)
        start_frame = min(start_frame, total_frames)

    return start_frame, end_frame


def count_inference_targets(
    total_frames: int,
    frame_stride: int,
    max_frames: int | None,
    max_source_seconds: float | None = None,
    fps: float = 30.0,
    *,
    start_source_seconds: float | None = None,
) -> int:
    """Estimate how many frames will run through the inference callback."""
    start_frame, end_frame = resolve_clip_window(
        total_frames,
        fps,
        start_source_seconds=start_source_seconds,
        max_source_seconds=max_source_seconds,
    )
    if end_frame > start_frame:
        window_frames = end_frame - start_frame
    elif max_source_seconds is not None and total_frames <= 0:
        window_frames = max(1, int(max_source_seconds * fps))
        start_frame = (
            max(0, int(start_source_seconds * fps))
            if start_source_seconds and start_source_seconds > 0
            else 0
        )
    else:
        return 0

    if window_frames <= 0 or frame_stride < 1:
        return 0
    # Absolute indices in [start_frame, start_frame + window_frames)
    rem = start_frame % frame_stride
    first = start_frame if rem == 0 else start_frame + (frame_stride - rem)
    last_exclusive = start_frame + window_frames
    if first >= last_exclusive:
        inferred = 0
    else:
        inferred = ((last_exclusive - 1 - first) // frame_stride) + 1
    if max_frames is not None:
        return min(max_frames, inferred)
    return inferred


def _seek_to_frame(capture: cv2.VideoCapture, start_frame: int) -> int:
    """Seek so the next ``read()`` returns ``start_frame``. Returns that index."""
    if start_frame <= 0:
        return 0
    if capture.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame)):
        return start_frame
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0.0)
    for skipped in range(start_frame):
        success, _ = capture.read()
        if not success:
            return skipped
    return start_frame


def process_video(
    source_path: Path,
    target_path: Path,
    callback: Callable[[np.ndarray, int], np.ndarray],
    frame_stride: int,
    max_frames: int | None,
    stats: dict[str, int] | None = None,
    progress_callback: ProgressCallback | None = None,
    preview_throttle: PreviewThrottle | None = None,
    cancel_event: Event | None = None,
    max_source_seconds: float | None = None,
    start_source_seconds: float | None = None,
) -> None:
    """Decode video, run ``callback`` on selected frames, and write annotated MP4."""
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video source: {source_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Invalid video dimensions for {source_path}: {width}x{height}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    start_frame, end_frame = resolve_clip_window(
        total_frames,
        fps,
        start_source_seconds=start_source_seconds,
        max_source_seconds=max_source_seconds,
    )
    inference_targets = count_inference_targets(
        total_frames,
        frame_stride,
        max_frames,
        max_source_seconds=max_source_seconds,
        fps=fps,
        start_source_seconds=start_source_seconds,
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(target_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Failed to open video writer: {target_path}")

    frame_index = _seek_to_frame(capture, start_frame)
    processed = 0
    last_annotated: np.ndarray | None = None
    progress_stats = stats if stats is not None else {}
    has_end_limit = end_frame > start_frame or max_source_seconds is not None

    try:
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise VideoProcessingCancelledError("Video export cancelled by user.")

            if has_end_limit and end_frame > 0 and frame_index >= end_frame:
                break

            success, frame_bgr = capture.read()
            if not success:
                break

            if frame_index % frame_stride == 0:
                last_annotated = callback(frame_bgr, frame_index)
                processed += 1
                if preview_throttle is not None and last_annotated is not None:
                    preview_throttle.maybe_emit(last_annotated, frame_index, processed)
                if progress_callback is not None:
                    total = inference_targets if inference_targets > 0 else processed
                    progress_callback(processed, total, progress_stats)

            output_frame = last_annotated if last_annotated is not None else frame_bgr
            writer.write(output_frame)
            frame_index += 1

            if max_frames is not None and processed >= max_frames:
                break
    finally:
        capture.release()
        writer.release()


# Backward-compatible aliases
_probe_video_size = probe_video_size
_finalize_video_path = finalize_video_path
_cleanup_partial_video = cleanup_partial_video
_partial_video_path = partial_video_path
_process_video = process_video
