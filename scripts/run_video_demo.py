#!/usr/bin/env python3
"""Deprecated: use ``rfdetr-demo`` or ``rfdetr_demo.cli``."""

from __future__ import annotations

import sys
import warnings

warnings.warn(
    "scripts/run_video_demo.py is deprecated; use `uv run rfdetr-demo` instead.",
    DeprecationWarning,
    stacklevel=1,
)

from rfdetr_demo.cli import main

if __name__ == "__main__":
    sys.exit(main())
