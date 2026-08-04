# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Persisted lease state for ephemeral Vast.ai jobs."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from rfdetr_demo.vast.safety_settings import VastSafetySettings

logger = logging.getLogger(__name__)


@dataclass
class VastJobLeaseState:
    """In-memory lease fields persisted to disk while a job is active."""

    started_by_app: bool = False
    instance_id: int | None = None
    started_at: float = 0.0
    last_heartbeat_at: float = 0.0
    label: str | None = None


@dataclass
class VastJobLease:
    """Persisted lease for ephemeral rf-detr Vast.ai jobs (sync variant of FlashFind lease)."""

    settings: VastSafetySettings = field(default_factory=VastSafetySettings.from_env)
    _state: VastJobLeaseState = field(default_factory=VastJobLeaseState)

    def mark_started(self, *, instance_id: int, label: str | None = None) -> None:
        """Record that this process started a labeled Vast instance."""
        now = time.time()
        self._state.started_by_app = True
        self._state.instance_id = instance_id
        self._state.started_at = now
        self._state.last_heartbeat_at = now
        self._state.label = label
        logger.info("vast_job_lease_started instance_id=%s", instance_id)
        self._persist()

    def heartbeat(self) -> None:
        """Refresh the lease heartbeat timestamp when the job is still alive."""
        if not self._state.started_by_app:
            return
        self._state.last_heartbeat_at = time.time()
        self._persist()

    def clear(self) -> None:
        """Clear in-memory state and delete the on-disk lease file."""
        self._state = VastJobLeaseState()
        path = self.settings.lease_path
        if path.is_file():
            path.unlink(missing_ok=True)

    def load_from_disk(self) -> None:
        """Restore lease state from disk when present."""
        path = self.settings.lease_path
        if not path.is_file():
            return
        try:
            raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            logger.warning("vast_job_lease_load_failed path=%s error=%s", path, error)
            return
        self._state.started_by_app = bool(raw.get("started_by_app", False))
        instance_id = raw.get("instance_id")
        self._state.instance_id = int(instance_id) if instance_id is not None else None
        self._state.started_at = float(raw.get("started_at", 0.0))
        self._state.last_heartbeat_at = float(raw.get("last_heartbeat_at", 0.0))
        self._state.label = raw.get("label")
        if self._state.started_by_app:
            logger.info(
                "vast_job_lease_restored instance_id=%s age_sec=%.0f",
                self._state.instance_id,
                time.time() - self._state.started_at if self._state.started_at else 0.0,
            )

    def should_auto_stop_now(self) -> tuple[bool, str]:
        """Return whether the max session duration has elapsed."""
        if not self._state.started_by_app:
            return False, ""
        if self.settings.max_session_sec <= 0:
            return False, ""
        if self._state.started_at <= 0:
            return False, ""
        if time.time() - self._state.started_at >= self.settings.max_session_sec:
            return True, "max_session_duration"
        return False, ""

    def snapshot(self) -> dict[str, Any]:
        """Return a GUI/CLI-friendly view of the active lease."""
        now = time.time()
        session_remaining: float | None = None
        if self._state.started_by_app and self._state.started_at > 0 and self.settings.max_session_sec > 0:
            session_remaining = max(0.0, self.settings.max_session_sec - (now - self._state.started_at))
        return {
            "started_by_app": self._state.started_by_app,
            "instance_id": self._state.instance_id,
            "max_session_sec": self.settings.max_session_sec,
            "session_remaining_sec": session_remaining,
            "label": self._state.label,
        }

    @property
    def instance_id(self) -> int | None:
        return self._state.instance_id

    @property
    def started_by_app(self) -> bool:
        """Return whether this process marked the lease as started."""
        return self._state.started_by_app

    def _persist(self) -> None:
        if not self._state.started_by_app:
            self.clear()
            return
        path = self.settings.lease_path
        payload = {
            "started_by_app": self._state.started_by_app,
            "instance_id": self._state.instance_id,
            "started_at": self._state.started_at,
            "last_heartbeat_at": self._state.last_heartbeat_at,
            "label": self._state.label,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
