# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Vast.ai destroy retries, orphan cleanup, and emergency teardown handlers."""

from __future__ import annotations

import atexit
import contextlib
import logging
import signal
import threading
import time
from typing import Any

from rfdetr_demo.vast.safety_lease import VastJobLease
from rfdetr_demo.vast.safety_settings import VastSafetySettings

logger = logging.getLogger(__name__)

_guard_lock = threading.Lock()
_active_guard: VastJobGuard | None = None
_emergency_handlers_installed = False


class VastJobGuard:
    """Ensures a Vast.ai instance is destroyed on normal exit, cancel, or process termination."""

    def __init__(
        self,
        *,
        api_key: str,
        settings: VastSafetySettings,
        destroy_on_finish: bool = True,
    ) -> None:
        self._api_key = api_key
        self._settings = settings
        self._destroy_on_finish = destroy_on_finish
        self._instance_id: int | None = None
        self._lease = VastJobLease(settings=settings)
        self._destroyed = False

    def attach_instance(self, instance_id: int, *, label: str | None = None) -> None:
        """Bind this guard to a running instance and register emergency handlers."""
        self._instance_id = instance_id
        self._lease.mark_started(instance_id=instance_id, label=label)
        self._register_global()

    def heartbeat(self) -> None:
        """Forward a heartbeat to the persisted lease."""
        self._lease.heartbeat()

    def snapshot(self) -> dict[str, Any]:
        """Return the current lease snapshot."""
        return self._lease.snapshot()

    def deactivate(self) -> None:
        """Unregister this guard as the process-global active guard."""
        with _guard_lock:
            global _active_guard
            if _active_guard is self:
                _active_guard = None

    def destroy_if_needed(self, *, reason: str = "job_finished") -> bool:
        """Destroy the attached instance when configured to tear down on finish."""
        if not self._destroy_on_finish or self._destroyed or self._instance_id is None:
            self._lease.clear()
            self.deactivate()
            return False
        success = destroy_instance_with_retry(
            self._instance_id,
            api_key=self._api_key,
            settings=self._settings,
            reason=reason,
        )
        self._destroyed = True
        self._lease.clear()
        self.deactivate()
        return success

    def emergency_destroy(self, *, reason: str = "emergency") -> None:
        """Best-effort destroy used by atexit / signal handlers."""
        if self._destroyed or self._instance_id is None or not self._destroy_on_finish:
            return
        logger.warning("vast_emergency_destroy instance_id=%s reason=%s", self._instance_id, reason)
        destroy_instance_with_retry(
            self._instance_id,
            api_key=self._api_key,
            settings=self._settings,
            reason=reason,
        )
        self._destroyed = True
        self._lease.clear()

    def check_runtime_limits(self, execute_started_at: float) -> tuple[bool, str]:
        """Return whether session or execute duration limits have been exceeded."""
        should_stop, reason = self._lease.should_auto_stop_now()
        if should_stop:
            return True, reason
        if self._settings.max_execute_sec > 0:
            elapsed = time.time() - execute_started_at
            if elapsed >= self._settings.max_execute_sec:
                return True, "max_execute_duration"
        return False, ""

    def _register_global(self) -> None:
        global _active_guard
        with _guard_lock:
            _active_guard = self
        install_emergency_handlers()


def destroy_instance_with_retry(
    instance_id: int,
    *,
    api_key: str,
    settings: VastSafetySettings,
    reason: str,
) -> bool:
    """Destroy a Vast.ai instance with retries (FlashFind-style best-effort teardown)."""
    from rfdetr_demo.vast.cli import is_vast_cli_available, run_vast_cli
    from rfdetr_demo.vast.types import VastRunnerError

    if not is_vast_cli_available():
        logger.error("vast_destroy_skipped instance_id=%s reason=%s detail=no_cli", instance_id, reason)
        return False

    last_error: Exception | None = None
    for attempt in range(1, settings.destroy_retries + 1):
        try:
            run_vast_cli(["destroy", "instance", str(instance_id)], api_key=api_key)
            logger.warning(
                "vast_instance_destroyed instance_id=%s reason=%s attempt=%s",
                instance_id,
                reason,
                attempt,
            )
            return True
        except VastRunnerError as error:
            last_error = error
            logger.error(
                "vast_destroy_failed instance_id=%s reason=%s attempt=%s error=%s",
                instance_id,
                reason,
                attempt,
                error,
            )
            if attempt < settings.destroy_retries:
                time.sleep(settings.destroy_retry_delay_sec)

    logger.critical(
        "vast_destroy_exhausted instance_id=%s reason=%s — "
        "MANUAL ACTION REQUIRED: vastai destroy instance %s",
        instance_id,
        reason,
        instance_id,
    )
    if last_error is not None:
        logger.critical("vast_destroy_last_error=%s", last_error)
    return False


def list_labeled_instances(
    *,
    api_key: str,
    label_prefix: str,
) -> list[dict[str, Any]]:
    """Return Vast.ai instances whose label starts with ``label_prefix``."""
    from rfdetr_demo.vast.cli import is_vast_cli_available, parse_json_output, run_vast_cli

    if not is_vast_cli_available():
        return []
    try:
        completed = run_vast_cli(["show", "instances", "--raw"], api_key=api_key)
    except Exception as error:
        logger.warning("vast_list_instances_failed error=%s", error)
        return []

    payload = parse_json_output(completed.stdout)
    instances: list[dict[str, Any]] = []
    if isinstance(payload, list):
        instances = [item for item in payload if isinstance(item, dict)]
    elif isinstance(payload, dict):
        nested = payload.get("instances")
        if isinstance(nested, list):
            instances = [item for item in nested if isinstance(item, dict)]

    matched: list[dict[str, Any]] = []
    for item in instances:
        label = str(item.get("label") or "")
        if label.startswith(label_prefix):
            matched.append(item)
    return matched


def cleanup_orphan_instances(
    *,
    api_key: str,
    settings: VastSafetySettings | None = None,
) -> list[int]:
    """Destroy orphaned rf-detr instances from a stale lease or matching labels."""
    resolved_settings = settings or VastSafetySettings.from_env()
    destroyed: list[int] = []
    candidate_ids: set[int] = set()

    lease = VastJobLease(settings=resolved_settings)
    lease.load_from_disk()
    should_stop, stop_reason = lease.should_auto_stop_now()
    if lease.instance_id is not None and (lease.started_by_app or should_stop):
        candidate_ids.add(lease.instance_id)
        logger.info(
            "vast_orphan_candidate_from_lease instance_id=%s should_stop=%s reason=%s",
            lease.instance_id,
            should_stop,
            stop_reason,
        )

    for item in list_labeled_instances(api_key=api_key, label_prefix=resolved_settings.instance_label_prefix):
        instance_id = item.get("id")
        if instance_id is not None:
            candidate_ids.add(int(instance_id))

    for instance_id in sorted(candidate_ids):
        if destroy_instance_with_retry(
            instance_id,
            api_key=api_key,
            settings=resolved_settings,
            reason="orphan_cleanup",
        ):
            destroyed.append(instance_id)

    if destroyed:
        lease.clear()
    return destroyed


def install_emergency_handlers() -> None:
    """Register atexit and signal handlers once per process (GUI / CLI safety net)."""
    global _emergency_handlers_installed
    if _emergency_handlers_installed:
        return
    _emergency_handlers_installed = True
    atexit.register(_emergency_cleanup)

    def _handler(signum: int, _frame: object | None) -> None:
        logger.warning("vast_signal_received signum=%s", signum)
        _emergency_cleanup()
        raise SystemExit(128 + signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(ValueError, OSError):
            signal.signal(sig, _handler)
    if hasattr(signal, "SIGBREAK"):
        with contextlib.suppress(ValueError, OSError):
            signal.signal(signal.SIGBREAK, _handler)  # type: ignore[attr-defined]


def _emergency_cleanup() -> None:
    with _guard_lock:
        guard = _active_guard
    if guard is not None:
        guard.emergency_destroy(reason="process_exit")
