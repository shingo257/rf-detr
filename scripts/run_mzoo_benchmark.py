#!/usr/bin/env python3
# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Deprecated thin launcher for the MZoo RF-DETR video-demo model benchmark.

Prefer::

    uv run rfdetr-mzoo-benchmark

or::

    python -m rfdetr_demo.benchmark
"""

from __future__ import annotations

import warnings

from rfdetr_demo.benchmark.cli import main

if __name__ == "__main__":
    warnings.warn(
        "scripts/run_mzoo_benchmark.py is deprecated; use `uv run rfdetr-mzoo-benchmark` "
        "or `python -m rfdetr_demo.benchmark.cli`.",
        DeprecationWarning,
        stacklevel=1,
    )
    raise SystemExit(main())
