# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""RF-DETR fork demo layer: video inference, GUI, Vast.ai, and tuning."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rfdetr_demo.inference.runner import run_demo as run_demo

__all__ = ["__version__", "run_demo"]

__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    """Lazily resolve public exports without loading inference dependencies at import time."""
    if name == "run_demo":
        from rfdetr_demo.inference.runner import run_demo

        globals()[name] = run_demo
        return run_demo
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return module attributes including lazily resolved public exports."""
    return sorted(set(globals()) | set(__all__))
