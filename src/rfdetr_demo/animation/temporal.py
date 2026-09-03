"""Temporal stabilization and short-gap filling for torso controls."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rfdetr_demo.animation.torso import ControlPoint, TorsoControls


@dataclass(frozen=True, slots=True)
class TorsoTemporalSettings:
    """Settings for confidence-adaptive torso smoothing."""

    ema_alpha: float = 0.55
    min_confidence: float = 0.15
    max_gap_frames: int = 4
    gap_confidence_decay: float = 0.75


@dataclass
class _PointState:
    xy: np.ndarray
    confidence: float
    gap_frames: int = 0


@dataclass
class _TrackState:
    points: dict[str, _PointState] = field(default_factory=dict)
    parameters: dict[str, float] = field(default_factory=dict)


class TorsoTemporalFilter:
    """Maintain independent torso histories for stable person track ids."""

    def __init__(self, settings: TorsoTemporalSettings | None = None) -> None:
        self.settings = settings or TorsoTemporalSettings()
        if not 0.0 < self.settings.ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        if self.settings.max_gap_frames < 0:
            raise ValueError("max_gap_frames must be non-negative")
        self._tracks: dict[int, _TrackState] = {}

    def reset(self, track_id: int | None = None) -> None:
        """Clear one track, or all tracks when ``track_id`` is omitted."""
        if track_id is None:
            self._tracks.clear()
        else:
            self._tracks.pop(track_id, None)

    def apply(self, controls: TorsoControls, *, track_id: int = 0) -> TorsoControls:
        """Smooth valid controls and hold recent values across short gaps."""
        state = self._tracks.setdefault(track_id, _TrackState())
        output_points: dict[str, ControlPoint] = {}
        for name, current in controls.points.items():
            previous = state.points.get(name)
            valid = current.visible and current.confidence >= self.settings.min_confidence
            if valid:
                if previous is None:
                    output_xy = current.xy.copy()
                else:
                    alpha = self.settings.ema_alpha + (1.0 - self.settings.ema_alpha) * current.confidence
                    output_xy = alpha * current.xy + (1.0 - alpha) * previous.xy
                state.points[name] = _PointState(output_xy.copy(), current.confidence)
                output_points[name] = ControlPoint(output_xy, current.confidence, True, current.source)
            elif previous is not None and previous.gap_frames < self.settings.max_gap_frames:
                previous.gap_frames += 1
                held_confidence = previous.confidence * self.settings.gap_confidence_decay**previous.gap_frames
                output_points[name] = ControlPoint(previous.xy.copy(), held_confidence, True, "interpolated")
            else:
                output_points[name] = ControlPoint(current.xy.copy(), 0.0, False, current.source)

        output_parameters: dict[str, float] = {}
        for name, value in controls.parameters.items():
            previous_value = state.parameters.get(name)
            if previous_value is None:
                output_value = value
            elif controls.confidence < self.settings.min_confidence:
                output_value = previous_value
            elif name.endswith("_deg"):
                delta = (value - previous_value + 180.0) % 360.0 - 180.0
                output_value = previous_value + self.settings.ema_alpha * delta
            else:
                output_value = self.settings.ema_alpha * value + (1.0 - self.settings.ema_alpha) * previous_value
            state.parameters[name] = float(output_value)
            output_parameters[name] = float(output_value)

        visible_confidences = [point.confidence for point in output_points.values() if point.visible]
        output_confidence = float(np.mean(visible_confidences)) if visible_confidences else 0.0
        return TorsoControls(output_points, output_parameters, output_confidence)
