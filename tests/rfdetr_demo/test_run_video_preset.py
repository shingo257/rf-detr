# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the video CLI --preset flag wiring."""

from __future__ import annotations

from unittest.mock import patch

from rfdetr_demo.cli import run_video


def _summary() -> dict[str, object]:
    return {"target": "out.mp4", "task": "detect", "processed_frames": 1, "total_detections": 1, "elapsed_sec": 0.1}


def test_preset_overhead_sets_documented_flags() -> None:
    with patch("rfdetr_demo.cli.run_video.run_demo", return_value=_summary()) as run_demo:
        code = run_video.main(
            ["--task", "detect", "--person-only", "--track", "--preset", "overhead", "--source", "x.mp4"],
        )
    assert code == 0
    kwargs = run_demo.call_args.kwargs
    assert kwargs["model_resolution"] == 960
    assert kwargs["threshold"] == 0.25
    assert kwargs["tile_size"] == 640
    assert kwargs["tile_overlap"] == 256
    assert kwargs["pose_topk"] == 8
    assert kwargs["reid_enabled"] is True


def test_preset_eye_level_sets_documented_flags() -> None:
    with patch("rfdetr_demo.cli.run_video.run_demo", return_value=_summary()) as run_demo:
        run_video.main(
            ["--task", "detect", "--person-only", "--track", "--preset", "eye-level", "--source", "x.mp4"],
        )
    kwargs = run_demo.call_args.kwargs
    assert kwargs["threshold"] == 0.4
    assert kwargs["tile_size"] == 0
    assert kwargs["pose_topk"] == 3
    assert kwargs["reid_enabled"] is True


def test_preset_respects_an_explicit_reid_model() -> None:
    with patch("rfdetr_demo.cli.run_video.run_demo", return_value=_summary()) as run_demo:
        run_video.main(
            [
                "--task",
                "detect",
                "--person-only",
                "--track",
                "--preset",
                "eye-level",
                "--reid-model",
                "reid.onnx",
                "--source",
                "x.mp4",
            ],
        )
    kwargs = run_demo.call_args.kwargs
    assert kwargs["reid_enabled"] is True
    assert kwargs["reid_model"] == "reid.onnx"


def test_preset_fast_sets_documented_flags() -> None:
    with patch("rfdetr_demo.cli.run_video.run_demo", return_value=_summary()) as run_demo:
        run_video.main(["--task", "detect", "--person-only", "--track", "--preset", "fast", "--source", "x.mp4"])
    kwargs = run_demo.call_args.kwargs
    assert kwargs["threshold"] == 0.3
    assert kwargs["tile_size"] == 0
    assert kwargs["pose_topk"] == 0


def test_preset_auto_resolves_viewpoint_and_applies_matching_preset() -> None:
    with patch("rfdetr_demo.cli.run_video._resolve_auto_preset", return_value="eye-level") as resolver:
        with patch("rfdetr_demo.cli.run_video.run_demo", return_value=_summary()) as run_demo:
            code = run_video.main(
                ["--task", "detect", "--person-only", "--track", "--preset", "auto", "--source", "x.mp4"],
            )
    assert code == 0
    resolver.assert_called_once()
    kwargs = run_demo.call_args.kwargs
    assert kwargs["threshold"] == 0.4
    assert kwargs["tile_size"] == 0
    assert kwargs["pose_topk"] == 3


def test_no_preset_leaves_explicit_flags_untouched() -> None:
    with patch("rfdetr_demo.cli.run_video.run_demo", return_value=_summary()) as run_demo:
        run_video.main(
            ["--task", "detect", "--person-only", "--track", "--threshold", "0.33", "--source", "x.mp4"],
        )
    kwargs = run_demo.call_args.kwargs
    assert kwargs["threshold"] == 0.33
    assert kwargs["tile_size"] == 0
