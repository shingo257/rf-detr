# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Backward-compatible facade for :mod:`rfdetr_demo.media.audit.tracking`."""

from rfdetr_demo.media.audit.tracking import (
    DEFAULT_CENTER_X_FRACTION,
    RawDetectionDiagnostic,
    TrackingAuditSummary,
    TrackingFrameRecord,
    center_x_range,
    evaluate_center_tracking,
    run_center_tracking_audit,
)

__all__ = [
    "DEFAULT_CENTER_X_FRACTION",
    "RawDetectionDiagnostic",
    "TrackingAuditSummary",
    "TrackingFrameRecord",
    "center_x_range",
    "evaluate_center_tracking",
    "run_center_tracking_audit",
]
