# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Unified confidential audit package (frame + tracking)."""

from rfdetr_demo.media.audit.common import (
    CLASSIFICATION,
    append_jsonl,
    build_base_audit_line,
    make_run_id,
    repo_relpath,
    save_audit_image,
    write_summary_json,
)
from rfdetr_demo.media.audit.frame import (
    DEFAULT_FRAME_AUDIT_COUNT,
    ConfidentialFrameAuditLogger,
    FrameAuditRecord,
    FrameAuditSummary,
    evaluate_frame_audit,
    is_frame_audit_enabled,
    wrap_callback_with_frame_audit,
)
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
    "CLASSIFICATION",
    "DEFAULT_CENTER_X_FRACTION",
    "DEFAULT_FRAME_AUDIT_COUNT",
    "ConfidentialFrameAuditLogger",
    "FrameAuditRecord",
    "FrameAuditSummary",
    "RawDetectionDiagnostic",
    "TrackingAuditSummary",
    "TrackingFrameRecord",
    "append_jsonl",
    "build_base_audit_line",
    "center_x_range",
    "evaluate_center_tracking",
    "evaluate_frame_audit",
    "is_frame_audit_enabled",
    "make_run_id",
    "repo_relpath",
    "run_center_tracking_audit",
    "save_audit_image",
    "wrap_callback_with_frame_audit",
    "write_summary_json",
]
