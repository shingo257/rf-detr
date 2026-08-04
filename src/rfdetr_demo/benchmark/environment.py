# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Environment and source-video probing for benchmark reports."""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import cv2
import torch

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None


def collect_environment() -> dict[str, Any]:
    """Gather hardware and software context for the benchmark report."""
    cpu_name = platform.processor() or "unknown"
    if platform.system() == "Windows":
        try:
            completed = subprocess.run(
                ["wmic", "cpu", "get", "Name"],
                capture_output=True,
                text=True,
                check=False,
                timeout=15,
            )
            lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
            if len(lines) >= 2:
                cpu_name = lines[1]
        except (OSError, subprocess.TimeoutExpired):
            pass

    cuda_available = torch.cuda.is_available()
    device_label = "CPU only"
    gpu_name: str | None = None
    if cuda_available:
        device_label = "CUDA GPU"
        gpu_name = torch.cuda.get_device_name(0)

    if psutil is not None:
        cpu_logical_cores = psutil.cpu_count(logical=True)
        cpu_physical_cores = psutil.cpu_count(logical=False)
        ram_total_gb = round(psutil.virtual_memory().total / (1024**3), 2)
    else:
        cpu_logical_cores = os.cpu_count()
        cpu_physical_cores = None
        ram_total_gb = None

    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_name": cpu_name,
        "cpu_logical_cores": cpu_logical_cores,
        "cpu_physical_cores": cpu_physical_cores,
        "ram_total_gb": ram_total_gb,
        "pytorch_version": torch.__version__,
        "device": device_label,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
    }


def probe_source_video(source_path: Path) -> dict[str, Any]:
    """Read basic metadata from the input video."""
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Source video not found or unreadable: {source_path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()

    duration_sec = round(frame_count / fps, 2) if fps > 0 else None
    return {
        "path": str(source_path.resolve()),
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "fps": round(fps, 3),
        "duration_sec": duration_sec,
        "file_size_mb": round(source_path.stat().st_size / (1024**2), 2),
    }


__all__ = ["collect_environment", "probe_source_video"]
