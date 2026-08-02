#!/usr/bin/env python3
"""Deprecated: use ``rfdetr-demo-gui`` or ``rfdetr_demo.gui.app``."""

from __future__ import annotations

import sys
import warnings

warnings.warn(
    "scripts/video_demo_gui.py is deprecated; use `uv run rfdetr-demo-gui` instead.",
    DeprecationWarning,
    stacklevel=1,
)

from rfdetr_demo.gui.main_window import main

if __name__ == "__main__":
    sys.exit(main())
