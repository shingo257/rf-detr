# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Vast.ai safety facade — prefer ``safety_settings`` / ``safety_lease`` / ``safety_guardrails``."""

from __future__ import annotations

from rfdetr_demo.vast.safety_guardrails import (
    VastJobGuard,
    cleanup_orphan_instances,
    destroy_instance_with_retry,
    install_emergency_handlers,
    list_labeled_instances,
)
from rfdetr_demo.vast.safety_lease import VastJobLease, VastJobLeaseState
from rfdetr_demo.vast.safety_settings import VastSafetySettings

__all__ = [
    "VastJobGuard",
    "VastJobLease",
    "VastJobLeaseState",
    "VastSafetySettings",
    "cleanup_orphan_instances",
    "destroy_instance_with_retry",
    "install_emergency_handlers",
    "list_labeled_instances",
]
