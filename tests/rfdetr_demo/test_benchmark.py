# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Unit tests for rfdetr_demo.benchmark helpers (no full inference)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rfdetr_demo.benchmark.jobs import (
    BENCHMARK_JOBS,
    ModelSize,
    job_label,
    output_path_for,
    resolve_jobs,
)
from rfdetr_demo.benchmark.report import format_duration, render_markdown
from rfdetr_demo.inference.types import TaskName
from rfdetr_demo.paths import REPO_ROOT


class TestJobLabel:
    """job_label coverage."""

    @pytest.mark.parametrize(
        ("task", "model_size", "expected"),
        [
            pytest.param("detect", "nano", "detect-nano", id="detect-nano"),
            pytest.param("detect", "large", "detect-large", id="detect-large"),
            pytest.param("keypoint", None, "keypoint-preview", id="keypoint"),
        ],
    )
    def test_job_label(
        self,
        task: TaskName,
        model_size: ModelSize | None,
        expected: str,
    ) -> None:
        """Map task/size pairs to stable job labels."""
        assert job_label(task, model_size) == expected


class TestResolveJobs:
    """resolve_jobs selection and skip behavior."""

    def test_default_includes_all_benchmark_jobs(self) -> None:
        """Default selection returns the full BENCHMARK_JOBS tuple."""
        assert resolve_jobs(None, set()) == BENCHMARK_JOBS

    def test_only_selects_requested_labels(self) -> None:
        """``--only`` keeps jobs in the requested order."""
        jobs = resolve_jobs(["detect-nano", "keypoint-preview"], set())
        assert jobs == (("detect", "nano"), ("keypoint", None))

    def test_skip_removes_labels(self) -> None:
        """``--skip`` removes matching labels from the default set."""
        jobs = resolve_jobs(None, {"detect-nano", "keypoint-preview"})
        labels = {job_label(task, size) for task, size in jobs}
        assert "detect-nano" not in labels
        assert "keypoint-preview" not in labels
        assert "detect-small" in labels

    def test_unknown_only_raises(self) -> None:
        """Unknown ``--only`` labels raise ValueError."""
        with pytest.raises(ValueError, match="Unknown job"):
            resolve_jobs(["not-a-job"], set())


class TestOutputPathFor:
    """output_path_for naming."""

    def test_detect_suffix(self) -> None:
        """Detect jobs use ``detected_<size>`` suffix under artifacts/demo."""
        path = output_path_for(Path("sample/mzoo.mov"), "detect", "nano")
        assert path == REPO_ROOT / "artifacts" / "demo" / "mzoo_detected_nano.mp4"

    def test_keypoint_suffix(self) -> None:
        """Keypoint jobs use the ``keypoints`` suffix."""
        path = output_path_for(Path("sample/mzoo.mov"), "keypoint", None)
        assert path.name == "mzoo_keypoints.mp4"


class TestFormatDuration:
    """format_duration formatting."""

    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            pytest.param(45, "0:45", id="under-one-minute"),
            pytest.param(125, "2:05", id="minutes"),
            pytest.param(3723, "1:02:03", id="hours"),
        ],
    )
    def test_format_duration(self, seconds: float, expected: str) -> None:
        """Format wall-clock seconds as H:MM:SS or M:SS."""
        assert format_duration(seconds) == expected


class TestRenderMarkdown:
    """render_markdown structure (no inference)."""

    def test_contains_environment_and_reproduce_cmd(self) -> None:
        """Markdown includes environment table and the package entry point."""
        report = {
            "started_at_utc": "2026-08-04T00:00:00+00:00",
            "finished_at_utc": "2026-08-04T00:01:00+00:00",
            "total_wall_clock_sec": 60.0,
            "environment": {
                "hostname": "host",
                "platform": "Linux",
                "python_version": "3.12.0",
                "pytorch_version": "2.5.0",
                "device": "CPU only",
                "cuda_available": False,
                "cpu_name": "cpu",
                "cpu_logical_cores": 4,
                "cpu_physical_cores": 2,
                "ram_total_gb": 8.0,
            },
            "source_video": {
                "path": "/tmp/mzoo.mov",
                "width": 1920,
                "height": 1080,
                "frame_count": 100,
                "fps": 30.0,
                "duration_sec": 3.33,
                "file_size_mb": 1.5,
            },
            "settings": {
                "threshold": 0.5,
                "frame_stride": 1,
                "max_frames": None,
                "person_only": True,
                "full_video": True,
            },
            "results": [
                {
                    "job": "detect-nano",
                    "status": "success",
                    "processed_frames": 10,
                    "total_detections": 20,
                    "elapsed_sec": 1.5,
                    "avg_fps": 6.67,
                    "output": "/tmp/out.mp4",
                    "output_size_mb": 0.5,
                },
            ],
        }
        markdown = render_markdown(report)
        assert "# mzoo.mov 動画解析ベンチマーク" in markdown
        assert "`detect-nano`" in markdown
        assert "uv run rfdetr-mzoo-benchmark" in markdown
        assert "1:00" in markdown
