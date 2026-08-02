# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Backward-compatible facade for :mod:`rfdetr_demo.media.audit.frame`."""

from rfdetr_demo.media.audit.frame import (
    DEFAULT_FRAME_AUDIT_COUNT,
    ConfidentialFrameAuditLogger,
    FrameAuditRecord,
    FrameAuditSummary,
    evaluate_frame_audit,
    is_frame_audit_enabled,
    wrap_callback_with_frame_audit,
)

__all__ = [
    "DEFAULT_FRAME_AUDIT_COUNT",
    "ConfidentialFrameAuditLogger",
    "FrameAuditRecord",
    "FrameAuditSummary",
    "evaluate_frame_audit",
    "is_frame_audit_enabled",
    "wrap_callback_with_frame_audit",
]
