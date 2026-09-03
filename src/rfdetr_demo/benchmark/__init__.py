# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""MZoo RF-DETR video-demo benchmark package."""

from rfdetr_demo.benchmark.cli import main
from rfdetr_demo.benchmark.jobs import BENCHMARK_JOBS, job_label, resolve_jobs
from rfdetr_demo.benchmark.report import format_duration, render_markdown

__all__ = [
    "BENCHMARK_JOBS",
    "format_duration",
    "job_label",
    "main",
    "render_markdown",
    "resolve_jobs",
]
