# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Video-demo model benchmark jobs and path helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from rfdetr_demo.inference.types import TaskName
from rfdetr_demo.paths import REPO_ROOT

ModelSize = Literal["nano", "small", "medium", "large"]

DETECTION_MODELS: tuple[ModelSize, ...] = ("nano", "small", "medium", "large")
BENCHMARK_JOBS: tuple[tuple[TaskName, ModelSize | None], ...] = (
    *((("detect", size) for size in DETECTION_MODELS)),
    ("keypoint", None),
)


def output_path_for(source_path: Path, task: TaskName, model_size: ModelSize | None) -> Path:
    """Build a distinct output filename per model."""
    if task == "keypoint":
        suffix = "keypoints"
    else:
        suffix = f"detected_{model_size}"
    return REPO_ROOT / "artifacts" / "demo" / f"{source_path.stem}_{suffix}.mp4"


def job_label(task: TaskName, model_size: ModelSize | None) -> str:
    """Return a human-readable job name."""
    if task == "keypoint":
        return "keypoint-preview"
    return f"detect-{model_size}"


def resolve_jobs(
    only: list[str] | None,
    skip: set[str],
) -> tuple[tuple[TaskName, ModelSize | None], ...]:
    """Select benchmark jobs from ``--only`` or the default list, applying ``--skip``."""
    if only:
        label_to_job = {job_label(task, size): (task, size) for task, size in BENCHMARK_JOBS}
        selected: list[tuple[TaskName, ModelSize | None]] = []
        for label in only:
            if label not in label_to_job:
                supported = ", ".join(sorted(label_to_job))
                raise ValueError(f"Unknown job {label!r}. Choose from: {supported}")
            selected.append(label_to_job[label])
        jobs = tuple(selected)
    else:
        jobs = BENCHMARK_JOBS
    return tuple((task, size) for task, size in jobs if job_label(task, size) not in skip)


__all__ = [
    "BENCHMARK_JOBS",
    "DETECTION_MODELS",
    "ModelSize",
    "job_label",
    "output_path_for",
    "resolve_jobs",
]
