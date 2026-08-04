#!/usr/bin/env python3
"""Deprecated shim — prefer ``rfdetr_demo.inference.uncertainty``."""

from __future__ import annotations

import warnings

warnings.warn(
    "scripts/keypoint_uncertainty_viz.py is deprecated; "
    "import from rfdetr_demo.inference.uncertainty instead.",
    DeprecationWarning,
    stacklevel=1,
)

from rfdetr_demo.inference.uncertainty import *  # noqa: E402, F403
