#!/usr/bin/env python3
"""Destroy orphaned rf-detr Vast.ai instances (deprecated wrapper)."""

from __future__ import annotations

import sys
import warnings

warnings.warn(
    "scripts/vast_cleanup_orphans.py is deprecated; use `uv run rfdetr-vast-cleanup` instead.",
    DeprecationWarning,
    stacklevel=1,
)

from rfdetr_demo.vast.cleanup_cli import main

if __name__ == "__main__":
    sys.exit(main())
