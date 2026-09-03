# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Stable public API allowlist for ``rfdetr_demo`` (Phase 13).

Import from this module (or the package root) for supported symbols.
Deep imports under ``rfdetr_demo.*`` remain available for advanced use, but
only the names listed in :data:`PUBLIC_API` are covered by semver stability
guarantees for the demo layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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

# Maps public attribute name → fully-qualified module that defines it.
PUBLIC_API: dict[str, str] = {
    "run_demo": "rfdetr_demo.inference.runner",
    "PersonTrackPipeline": "rfdetr_demo.tracking.pipeline",
    "ConfidentialFrameAuditLogger": "rfdetr_demo.media.audit.frame",
    "run_center_tracking_audit": "rfdetr_demo.media.audit.tracking",
    "run_vast_cli": "rfdetr_demo.vast.cli",
    "DEFAULT_PARAMETERS": "rfdetr_demo.tuning.auto_tune",
}

__all__ = [
    "PUBLIC_API",
    "ConfidentialFrameAuditLogger",
    "DEFAULT_PARAMETERS",
    "PersonTrackPipeline",
    "run_center_tracking_audit",
    "run_demo",
    "run_vast_cli",
]


def __getattr__(name: str) -> Any:
    """Lazily resolve allowlisted public symbols."""
    if name not in PUBLIC_API:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}. "
            f"Stable public API: {', '.join(sorted(PUBLIC_API))}",
        )
    module_name = PUBLIC_API[name]
    from importlib import import_module

    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include allowlisted public names in ``dir()``."""
    return sorted(set(globals()) | set(__all__))
