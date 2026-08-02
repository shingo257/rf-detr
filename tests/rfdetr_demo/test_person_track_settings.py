# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for person track settings elevation (env / defaults)."""

from __future__ import annotations

import pytest

from rfdetr_demo.tracking.types import PersonTrackSettings, person_track_settings_from_env


def test_person_track_settings_from_env_max_missed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RFDETR_MAX_MISSED", "5")
    settings = person_track_settings_from_env()
    assert settings.max_missed == 5


def test_person_track_settings_from_env_sticky(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RFDETR_MAX_MISSED", raising=False)
    monkeypatch.setenv("RFDETR_STICKY_CENTER_TRACK", "1")
    monkeypatch.setenv("RFDETR_STICKY_MAX_MISSED", "6")
    settings = person_track_settings_from_env(base=PersonTrackSettings(max_missed=2))
    assert settings.sticky_center_track is True
    assert settings.sticky_max_missed == 6
    assert settings.max_missed == 2


def test_person_track_settings_from_env_disables_sticky(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RFDETR_STICKY_CENTER_TRACK", "0")
    settings = person_track_settings_from_env(
        base=PersonTrackSettings(sticky_center_track=True),
    )
    assert settings.sticky_center_track is False
