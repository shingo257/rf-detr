#!/usr/bin/env python3
"""Deprecated shim — prefer ``uv run rfdetr-demo`` or canonical ``rfdetr_demo.vast`` imports."""

from __future__ import annotations

import warnings

warnings.warn(
    "scripts/vast_ai_runner.py is deprecated; import from rfdetr_demo.vast.cli / "
    "rfdetr_demo.vast.offers / rfdetr_demo.vast.video_job instead.",
    DeprecationWarning,
    stacklevel=1,
)

from rfdetr_demo.vast.api_config import resolve_vast_api_key  # noqa: E402
from rfdetr_demo.vast.cli import (  # noqa: E402
    ensure_vast_cli_or_raise,
    is_vast_cli_available,
)
from rfdetr_demo.vast.offers import search_gpu_offers  # noqa: E402
from rfdetr_demo.vast.types import (  # noqa: E402
    DEFAULT_VAST_IMAGE,
    REMOTE_JOB_DIR,
    REMOTE_OUTPUT_NAME,
    REMOTE_PROGRESS_PATH,
    VAST_CLI_DOCS_URL,
    VAST_DOCS_URL,
    VastGpuOffer,
    VastLogCallback,
    VastPhase,
    VastPhaseCallback,
    VastRunnerCancelledError,
    VastRunnerError,
    VastVideoJobConfig,
)
from rfdetr_demo.vast.video_job import run_video_demo_on_vast  # noqa: E402

__all__ = [
    "DEFAULT_VAST_IMAGE",
    "REMOTE_JOB_DIR",
    "REMOTE_OUTPUT_NAME",
    "REMOTE_PROGRESS_PATH",
    "VAST_CLI_DOCS_URL",
    "VAST_DOCS_URL",
    "VastGpuOffer",
    "VastLogCallback",
    "VastPhase",
    "VastPhaseCallback",
    "VastRunnerCancelledError",
    "VastRunnerError",
    "VastVideoJobConfig",
    "ensure_vast_cli_or_raise",
    "is_vast_cli_available",
    "resolve_vast_api_key",
    "run_video_demo_on_vast",
    "search_gpu_offers",
]
