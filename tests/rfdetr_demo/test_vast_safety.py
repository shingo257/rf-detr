# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for Vast.ai safety settings, lease persistence, and orphan cleanup."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from rfdetr_demo.vast.safety_guardrails import (
    VastJobGuard,
    cleanup_orphan_instances,
    destroy_instance_with_retry,
)
from rfdetr_demo.vast.safety_lease import VastJobLease
from rfdetr_demo.vast.safety_settings import VastSafetySettings


def test_vast_safety_settings_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    lease_path = tmp_path / "lease.json"
    monkeypatch.setenv("RFDETR_VAST_MAX_SESSION_HOURS", "1.5")
    monkeypatch.setenv("RFDETR_VAST_MAX_EXECUTE_HOURS", "0.5")
    monkeypatch.setenv("RFDETR_VAST_BOOT_TIMEOUT_SEC", "120")
    monkeypatch.setenv("RFDETR_VAST_DESTROY_RETRIES", "2")
    monkeypatch.setenv("RFDETR_VAST_AUTO_CLEANUP_ORPHANS", "0")
    monkeypatch.setenv("RFDETR_VAST_INSTANCE_LABEL_PREFIX", "demo-test")
    monkeypatch.setenv("RFDETR_VAST_LEASE_PATH", str(lease_path))

    settings = VastSafetySettings.from_env()
    assert settings.max_session_sec == pytest.approx(1.5 * 3600.0)
    assert settings.max_execute_sec == pytest.approx(0.5 * 3600.0)
    assert settings.boot_timeout_sec == pytest.approx(120.0)
    assert settings.destroy_retries == 2
    assert settings.auto_cleanup_orphans_on_start is False
    assert settings.instance_label_prefix == "demo-test"
    assert settings.lease_path == lease_path


def test_vast_job_lease_persists_and_restores(tmp_path: Path) -> None:
    lease_path = tmp_path / "lease.json"
    settings = VastSafetySettings(lease_path=lease_path, max_session_sec=3600.0)
    lease = VastJobLease(settings=settings)
    lease.mark_started(instance_id=42, label="rfdetr-demo-1")
    assert lease_path.is_file()
    payload = json.loads(lease_path.read_text(encoding="utf-8"))
    assert payload["instance_id"] == 42
    assert payload["started_by_app"] is True

    restored = VastJobLease(settings=settings)
    restored.load_from_disk()
    assert restored.instance_id == 42
    assert restored.started_by_app is True
    assert restored.snapshot()["label"] == "rfdetr-demo-1"


def test_vast_job_lease_auto_stop_after_max_session(tmp_path: Path) -> None:
    settings = VastSafetySettings(lease_path=tmp_path / "lease.json", max_session_sec=10.0)
    lease = VastJobLease(settings=settings)
    lease.mark_started(instance_id=7)
    lease._state.started_at = lease._state.started_at - 11.0
    should_stop, reason = lease.should_auto_stop_now()
    assert should_stop is True
    assert reason == "max_session_duration"


def test_destroy_instance_with_retry_returns_false_without_cli() -> None:
    settings = VastSafetySettings(destroy_retries=1)
    with patch("rfdetr_demo.vast.cli.is_vast_cli_available", return_value=False):
        assert (
            destroy_instance_with_retry(
                99,
                api_key="key",
                settings=settings,
                reason="test",
            )
            is False
        )


def test_cleanup_orphan_instances_uses_lease_candidate(tmp_path: Path) -> None:
    lease_path = tmp_path / "lease.json"
    settings = VastSafetySettings(
        lease_path=lease_path,
        destroy_retries=1,
        instance_label_prefix="rfdetr-demo",
    )
    lease = VastJobLease(settings=settings)
    lease.mark_started(instance_id=55, label="rfdetr-demo-x")

    with (
        patch("rfdetr_demo.vast.safety_guardrails.list_labeled_instances", return_value=[]),
        patch(
            "rfdetr_demo.vast.safety_guardrails.destroy_instance_with_retry",
            return_value=True,
        ) as destroy,
    ):
        destroyed = cleanup_orphan_instances(api_key="key", settings=settings)

    assert destroyed == [55]
    destroy.assert_called_once()
    assert not lease_path.is_file()


def test_vast_job_guard_check_runtime_limits(tmp_path: Path) -> None:
    settings = VastSafetySettings(
        lease_path=tmp_path / "lease.json",
        max_session_sec=0.0,
        max_execute_sec=1.0,
    )
    guard = VastJobGuard(api_key="key", settings=settings, destroy_on_finish=False)
    should_stop, reason = guard.check_runtime_limits(execute_started_at=__import__("time").time() - 2.0)
    assert should_stop is True
    assert reason == "max_execute_duration"


def test_vast_job_guard_destroy_if_needed_skips_when_disabled(tmp_path: Path) -> None:
    settings = VastSafetySettings(lease_path=tmp_path / "lease.json")
    guard = VastJobGuard(api_key="key", settings=settings, destroy_on_finish=False)
    with patch("rfdetr_demo.vast.safety_guardrails.destroy_instance_with_retry") as destroy:
        assert guard.destroy_if_needed(reason="done") is False
    destroy.assert_not_called()
