# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""RF-DETR fork demo layer: video inference, GUI, Vast.ai, and tuning.

Stable public symbols are defined in :mod:`rfdetr_demo.public` and re-exported
here. Prefer::

    from rfdetr_demo import run_demo, PersonTrackPipeline
    # or
    from rfdetr_demo.public import PUBLIC_API, run_demo
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rfdetr_demo.public import PUBLIC_API

if TYPE_CHECKING:
    from rfdetr_demo.inference.runner import run_demo as run_demo
    from rfdetr_demo.media.audit.frame import (
        ConfidentialFrameAuditLogger as ConfidentialFrameAuditLogger,
    )
    from rfdetr_demo.media.audit.tracking import (
        run_center_tracking_audit as run_center_tracking_audit,
    )
    from rfdetr_demo.tracking.pipeline import PersonTrackPipeline as PersonTrackPipeline
    from rfdetr_demo.tuning.auto_tune import DEFAULT_PARAMETERS as DEFAULT_PARAMETERS
    from rfdetr_demo.vast.cli import run_vast_cli as run_vast_cli

__all__ = [
    "PUBLIC_API",
    "ConfidentialFrameAuditLogger",
    "DEFAULT_PARAMETERS",
    "PersonTrackPipeline",
    "__version__",
    "run_center_tracking_audit",
    "run_demo",
    "run_vast_cli",
]

__version__ = "0.2.0"


def __getattr__(name: str) -> Any:
    """Lazily resolve public exports without loading heavy dependencies at import time."""
    if name == "PUBLIC_API":
        return PUBLIC_API
    if name in PUBLIC_API:
        from importlib import import_module

        module = import_module(PUBLIC_API[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}. "
        f"Stable public API: {', '.join(sorted(PUBLIC_API))}",
    )


def __dir__() -> list[str]:
    """Return module attributes including lazily resolved public exports."""
    return sorted(set(globals()) | set(__all__))
