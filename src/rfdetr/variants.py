# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Concrete RF-DETR model variant classes.

All classes inherit from :class:`~rfdetr.detr.RFDETR` which remains defined in ``rfdetr.detr``. Backward-compatible
access from ``rfdetr.detr`` is provided via lazy ``__getattr__`` re-exports, so importing ``rfdetr.variants`` no longer
depends on a fragile eager ``detr -> variants`` import sequence.
"""

from __future__ import annotations

__all__ = [
    "RFDETRKeypointPreview",
    "RFDETRNano",
    "RFDETRSmall",
    "RFDETRMedium",
    "RFDETRLarge",
    "RFDETRSeg",
    "RFDETRSegNano",
    "RFDETRSegSmall",
    "RFDETRSegMedium",
    "RFDETRSegLarge",
    "RFDETRSegXLarge",
    "RFDETRSeg2XLarge",
]

from typing import Any

from rfdetr.config import (
    KeypointTrainConfig,
    ModelConfig,
    RFDETRKeypointPreviewConfig,
    RFDETRLargeConfig,
    RFDETRMediumConfig,
    RFDETRNanoConfig,
    RFDETRSeg2XLargeConfig,
    RFDETRSegLargeConfig,
    RFDETRSegMediumConfig,
    RFDETRSegNanoConfig,
    RFDETRSegSmallConfig,
    RFDETRSegXLargeConfig,
    RFDETRSmallConfig,
    SegmentationTrainConfig,
)
from rfdetr.detr import RFDETR


class RFDETRNano(RFDETR):
    """Train an RF-DETR Nano model.

    Training accepts custom square integer ``resolution`` values. The value must be divisible by ``patch_size *
    num_windows``.
    """

    size = "rfdetr-nano"
    _model_config_class = RFDETRNanoConfig


class RFDETRKeypointPreview(RFDETR):
    """Train or run inference with the RF-DETR keypoint preview model.

    Training accepts custom square integer ``resolution`` values. The value must be divisible by ``patch_size *
    num_windows``.
    """

    size = "rfdetr-keypoint-preview"
    _model_config_class = RFDETRKeypointPreviewConfig
    _train_config_class = KeypointTrainConfig


class RFDETRSmall(RFDETR):
    """Train an RF-DETR Small model.

    Training accepts custom square integer ``resolution`` values. The value must be divisible by ``patch_size *
    num_windows``.
    """

    size = "rfdetr-small"
    _model_config_class = RFDETRSmallConfig


class RFDETRMedium(RFDETR):
    """Train an RF-DETR Medium model.

    Training accepts custom square integer ``resolution`` values. The value must be divisible by ``patch_size *
    num_windows``.
    """

    size = "rfdetr-medium"
    _model_config_class = RFDETRMediumConfig


class RFDETRLarge(RFDETR):
    """Train an RF-DETR Large model.

    Training accepts custom square integer ``resolution`` values. The value must be divisible by ``patch_size *
    num_windows``.
    """

    size = "rfdetr-large"

    def get_model_config(self, **kwargs: Any) -> ModelConfig:
        return RFDETRLargeConfig(**kwargs)


class RFDETRSeg(RFDETR):
    """Base class for all RF-DETR segmentation models.

    Training accepts custom square integer ``resolution`` values. Most segmentation variants use multiples of 24;
    ``RFDETRSegNano`` uses multiples of 12.
    """

    _train_config_class = SegmentationTrainConfig


class RFDETRSegNano(RFDETRSeg):
    """Train an RF-DETR Segmentation Nano model.

    Training accepts custom square integer ``resolution`` values. The value must be divisible by ``patch_size *
    num_windows``; this variant uses multiples of 12.
    """

    size = "rfdetr-seg-nano"
    _model_config_class = RFDETRSegNanoConfig


class RFDETRSegSmall(RFDETRSeg):
    """Train an RF-DETR Segmentation Small model.

    Training accepts custom square integer ``resolution`` values. The value must be divisible by ``patch_size *
    num_windows``; this variant uses multiples of 24.
    """

    size = "rfdetr-seg-small"
    _model_config_class = RFDETRSegSmallConfig


class RFDETRSegMedium(RFDETRSeg):
    """Train an RF-DETR Segmentation Medium model.

    Training accepts custom square integer ``resolution`` values. The value must be divisible by ``patch_size *
    num_windows``; this variant uses multiples of 24.
    """

    size = "rfdetr-seg-medium"
    _model_config_class = RFDETRSegMediumConfig


class RFDETRSegLarge(RFDETRSeg):
    """Train an RF-DETR Segmentation Large model.

    Training accepts custom square integer ``resolution`` values. The value must be divisible by ``patch_size *
    num_windows``; this variant uses multiples of 24.
    """

    size = "rfdetr-seg-large"
    _model_config_class = RFDETRSegLargeConfig


class RFDETRSegXLarge(RFDETRSeg):
    """Train an RF-DETR Segmentation XLarge model.

    Training accepts custom square integer ``resolution`` values. The value must be divisible by ``patch_size *
    num_windows``; this variant uses multiples of 24.
    """

    size = "rfdetr-seg-xlarge"
    _model_config_class = RFDETRSegXLargeConfig


class RFDETRSeg2XLarge(RFDETRSeg):
    """Train an RF-DETR Segmentation 2XLarge model.

    Training accepts custom square integer ``resolution`` values. The value must be divisible by ``patch_size *
    num_windows``; this variant uses multiples of 24.
    """

    size = "rfdetr-seg-2xlarge"
    _model_config_class = RFDETRSeg2XLargeConfig
