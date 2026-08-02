#!/usr/bin/env python3
"""Deprecated: use ``uv run rfdetr-demo audit-tracking``."""

from __future__ import annotations

import sys
import warnings

warnings.warn(
    "scripts/audit_center_tracking.py is deprecated; use `uv run rfdetr-demo audit-tracking` instead.",
    DeprecationWarning,
    stacklevel=1,
)

from rfdetr_demo.cli.main import main


if __name__ == "__main__":
    audit_argv = ["audit-tracking", *sys.argv[1:]]
    raise SystemExit(main(audit_argv))
