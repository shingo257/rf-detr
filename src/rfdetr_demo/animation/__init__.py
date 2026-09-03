"""Animation retargeting helpers for RF-DETR keypoint tracks."""

from rfdetr_demo.animation.dynamics import MascotMotionDynamics
from rfdetr_demo.animation.pipeline import TorsoControlPipeline, TrackedTorsoControls
from rfdetr_demo.animation.retarget import FukkachanRetargeter, MascotRigPose
from rfdetr_demo.animation.temporal import TorsoTemporalFilter, TorsoTemporalSettings
from rfdetr_demo.animation.torso import (
    COCO17_KEYPOINT_NAMES,
    TORSO_CONTROL_NAMES,
    ControlPoint,
    TorsoControls,
    derive_torso_controls,
)

__all__ = [
    "COCO17_KEYPOINT_NAMES",
    "TORSO_CONTROL_NAMES",
    "ControlPoint",
    "FukkachanRetargeter",
    "MascotRigPose",
    "MascotMotionDynamics",
    "TorsoControls",
    "TorsoControlPipeline",
    "TorsoTemporalFilter",
    "TorsoTemporalSettings",
    "TrackedTorsoControls",
    "derive_torso_controls",
]
