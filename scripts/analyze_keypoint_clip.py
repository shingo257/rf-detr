#!/usr/bin/env python3
"""Deprecated: use ``uv run rfdetr-demo analyze-clip``."""

from __future__ import annotations

import sys
import warnings

warnings.warn(
    "scripts/analyze_keypoint_clip.py is deprecated; use `uv run rfdetr-demo analyze-clip` instead.",
    DeprecationWarning,
    stacklevel=1,
)

from rfdetr_demo.cli.main import main


if __name__ == "__main__":
    clip_argv = ["analyze-clip", *sys.argv[1:]]
    raise SystemExit(main(clip_argv))
