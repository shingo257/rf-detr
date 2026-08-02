#!/usr/bin/env python3
"""Deprecated: use ``uv run rfdetr-demo probe-count``."""

from __future__ import annotations

import sys
import warnings

warnings.warn(
    "scripts/probe_person_count.py is deprecated; use `uv run rfdetr-demo probe-count` instead.",
    DeprecationWarning,
    stacklevel=1,
)

from rfdetr_demo.cli.main import main


if __name__ == "__main__":
    probe_argv = ["probe-count", *sys.argv[1:]]
    raise SystemExit(main(probe_argv))
