# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Vast.ai safety limits loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from rfdetr_demo.paths import REPO_ROOT

_DEFAULT_LEASE_PATH = REPO_ROOT / "artifacts" / "vast" / "vast-job-lease.local.json"


@dataclass(frozen=True)
class VastSafetySettings:
    """Safety limits aligned with FlashFind ``gpu_pod_lease`` defaults."""

    max_session_sec: float = 7_200.0
    max_execute_sec: float = 7_200.0
    boot_timeout_sec: float = 900.0
    destroy_retries: int = 3
    destroy_retry_delay_sec: float = 5.0
    instance_label_prefix: str = "rfdetr-demo"
    auto_cleanup_orphans_on_start: bool = True
    lease_path: Path = _DEFAULT_LEASE_PATH

    @classmethod
    def from_env(cls) -> VastSafetySettings:
        """Load settings from ``RFDETR_VAST_*`` environment variables."""
        return cls(
            max_session_sec=_env_float("RFDETR_VAST_MAX_SESSION_HOURS", 2.0) * 3600.0,
            max_execute_sec=_env_float("RFDETR_VAST_MAX_EXECUTE_HOURS", 2.0) * 3600.0,
            boot_timeout_sec=_env_float("RFDETR_VAST_BOOT_TIMEOUT_SEC", 900.0),
            destroy_retries=_env_int("RFDETR_VAST_DESTROY_RETRIES", 3),
            destroy_retry_delay_sec=_env_float("RFDETR_VAST_DESTROY_RETRY_DELAY_SEC", 5.0),
            instance_label_prefix=os.environ.get("RFDETR_VAST_INSTANCE_LABEL_PREFIX", "rfdetr-demo"),
            auto_cleanup_orphans_on_start=_env_bool("RFDETR_VAST_AUTO_CLEANUP_ORPHANS", True),
            lease_path=Path(os.environ.get("RFDETR_VAST_LEASE_PATH", str(_DEFAULT_LEASE_PATH))),
        )


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return int(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}
