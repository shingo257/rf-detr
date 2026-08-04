# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Execute video-demo benchmark jobs and collect timing summaries."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

from rfdetr_demo.benchmark.environment import collect_environment, probe_source_video
from rfdetr_demo.benchmark.jobs import ModelSize, job_label, output_path_for
from rfdetr_demo.inference.runner import run_demo
from rfdetr_demo.inference.types import TaskName

logger = logging.getLogger(__name__)


def run_all_benchmarks(
    source_path: Path,
    threshold: float,
    frame_stride: int,
    max_frames: int | None,
    jobs: tuple[tuple[TaskName, ModelSize | None], ...],
) -> dict[str, Any]:
    """Execute every configured model and collect timing summaries."""
    started_at = datetime.now(timezone.utc)
    environment = collect_environment()
    source_info = probe_source_video(source_path)

    results: list[dict[str, Any]] = []
    total_elapsed_sec = 0.0

    for task, model_size in jobs:
        label = job_label(task, model_size)
        target_path = output_path_for(source_path, task, model_size)
        detect_size: ModelSize = model_size if model_size is not None else "nano"
        logger.info("Starting benchmark job: %s -> %s", label, target_path.name)

        job_started = time.perf_counter()
        try:
            summary = run_demo(
                source_path=source_path,
                target_path=target_path,
                task=task,
                model_size=detect_size,
                threshold=threshold,
                frame_stride=frame_stride,
                max_frames=max_frames,
                person_only=True,
            )
            status = "success"
            error_message: str | None = None
        except Exception as exc:
            logger.exception("Benchmark job failed: %s", label)
            status = "failed"
            error_message = str(exc)
            summary = {
                "target": str(target_path.resolve()),
                "processed_frames": 0,
                "total_detections": 0,
                "elapsed_sec": round(time.perf_counter() - job_started, 2),
            }

        job_elapsed = float(summary.get("elapsed_sec", 0.0))
        total_elapsed_sec += job_elapsed

        output_size_mb: float | None = None
        target = Path(str(summary.get("target", target_path)))
        if status == "success" and target.is_file():
            output_size_mb = round(target.stat().st_size / (1024**2), 2)

        results.append(
            {
                "job": label,
                "task": task,
                "model_size": model_size if task == "detect" else "keypoint-preview",
                "status": status,
                "error": error_message,
                "output": str(target.resolve()) if target.exists() else str(target),
                "output_size_mb": output_size_mb,
                **{key: summary[key] for key in summary if key not in {"source", "target"}},
            },
        )

    finished_at = datetime.now(timezone.utc)
    return {
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "total_wall_clock_sec": round(total_elapsed_sec, 2),
        "environment": environment,
        "source_video": source_info,
        "settings": {
            "threshold": threshold,
            "frame_stride": frame_stride,
            "max_frames": max_frames,
            "person_only": True,
            "full_video": frame_stride == 1 and max_frames is None,
        },
        "results": results,
    }


def build_cached_result(
    source_path: Path,
    task: TaskName,
    model_size: ModelSize | None,
    target_path: Path,
    elapsed_sec: float,
    total_detections: int | None = None,
) -> dict[str, Any]:
    """Build a result row from an existing output file (reuse prior run)."""
    del source_path
    label = job_label(task, model_size)
    capture = cv2.VideoCapture(str(target_path))
    processed_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if capture.isOpened() else 0
    capture.release()
    output_size_mb = round(target_path.stat().st_size / (1024**2), 2) if target_path.is_file() else None
    row: dict[str, Any] = {
        "job": label,
        "task": task,
        "model_size": model_size if task == "detect" else "keypoint-preview",
        "status": "success (cached)",
        "error": None,
        "output": str(target_path.resolve()),
        "output_size_mb": output_size_mb,
        "threshold": 0.5,
        "frame_stride": 1,
        "person_only": True if task == "detect" else None,
        "processed_frames": processed_frames,
        "total_detections": total_detections if total_detections is not None else "—",
        "elapsed_sec": round(elapsed_sec, 2),
    }
    if processed_frames > 0 and elapsed_sec > 0:
        row["avg_fps"] = round(processed_frames / elapsed_sec, 2)
        if isinstance(total_detections, int):
            row["avg_detections_per_frame"] = round(total_detections / processed_frames, 2)
    return row


__all__ = ["build_cached_result", "run_all_benchmarks"]
