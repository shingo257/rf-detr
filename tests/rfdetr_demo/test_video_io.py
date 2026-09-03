# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for video I/O helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from rfdetr_demo.inference.video_io import (
    count_inference_targets,
    probe_video_size,
    process_video,
    resolve_clip_window,
)


def test_probe_video_size_returns_dimensions(tmp_path: Path) -> None:
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake")

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: {3: 640.0, 4: 480.0, 5: 30.0}.get(prop, 0.0)

    with patch("rfdetr_demo.inference.video_io.cv2.VideoCapture", return_value=mock_cap):
        width, height, fps = probe_video_size(video_path)

    assert width == 640
    assert height == 480
    assert fps == 30.0
    mock_cap.release.assert_called_once()


def test_resolve_clip_window_treats_max_seconds_as_duration_after_start() -> None:
    assert resolve_clip_window(
        total_frames=300,
        fps=30.0,
        start_source_seconds=2.0,
        max_source_seconds=3.0,
    ) == (60, 150)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        pytest.param({"start_source_seconds": -0.1}, "start_source_seconds", id="negative-start"),
        pytest.param({"max_source_seconds": 0.0}, "max_source_seconds", id="zero-duration"),
    ],
)
def test_resolve_clip_window_rejects_invalid_limits(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_clip_window(total_frames=300, fps=30.0, **kwargs)


@pytest.mark.parametrize(
    ("start_seconds", "frame_stride", "max_frames", "expected"),
    [
        pytest.param(1.0, 6, None, 5, id="aligned-start"),
        pytest.param(1.1, 6, None, 5, id="unaligned-start"),
        pytest.param(1.0, 6, 3, 3, id="max-frames-cap"),
    ],
)
def test_count_inference_targets_uses_absolute_source_indices(
    start_seconds: float,
    frame_stride: int,
    max_frames: int | None,
    expected: int,
) -> None:
    assert (
        count_inference_targets(
            total_frames=300,
            frame_stride=frame_stride,
            max_frames=max_frames,
            max_source_seconds=1.0,
            fps=30.0,
            start_source_seconds=start_seconds,
        )
        == expected
    )


def test_process_video_seeks_and_preserves_absolute_frame_indices(tmp_path: Path) -> None:
    source_path = tmp_path / "source.mp4"
    target_path = tmp_path / "target.mp4"
    source_path.write_bytes(b"fake")

    capture = MagicMock()
    capture.isOpened.return_value = True
    capture.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FPS: 2.0,
        cv2.CAP_PROP_FRAME_WIDTH: 3.0,
        cv2.CAP_PROP_FRAME_HEIGHT: 2.0,
        cv2.CAP_PROP_FRAME_COUNT: 10.0,
    }.get(prop, 0.0)
    capture.set.return_value = True
    frames = [np.full((2, 3, 3), value, dtype=np.uint8) for value in range(4, 8)]
    capture.read.side_effect = [(True, frame) for frame in frames]

    writer = MagicMock()
    writer.isOpened.return_value = True
    callback_indices: list[int] = []

    def callback(frame: np.ndarray, frame_index: int) -> np.ndarray:
        callback_indices.append(frame_index)
        return frame

    with (
        patch("rfdetr_demo.inference.video_io.cv2.VideoCapture", return_value=capture),
        patch("rfdetr_demo.inference.video_io.cv2.VideoWriter", return_value=writer),
        patch("rfdetr_demo.inference.video_io.cv2.VideoWriter_fourcc", return_value=1234),
    ):
        process_video(
            source_path=source_path,
            target_path=target_path,
            callback=callback,
            frame_stride=2,
            max_frames=None,
            start_source_seconds=2.0,
            max_source_seconds=2.0,
        )

    capture.set.assert_any_call(cv2.CAP_PROP_POS_FRAMES, 4.0)
    assert callback_indices == [4, 6]
    assert writer.write.call_count == 4
    capture.release.assert_called_once()
    writer.release.assert_called_once()
