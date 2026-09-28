# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Repository-wide pytest collection hooks."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import pytest

# rfdetr_demo.gui (and the one vast module it reaches into) import tkinter at module scope, some
# for widget base-class inheritance, which `from __future__ import annotations` cannot defer. So
# --doctest-plus collection, which imports every module under src/ to look for doctests, fails
# outright on interpreters whose tkinter extension isn't built — e.g. some uv-managed
# python-build-standalone 3.12 releases on Linux. Which specific files need tkinter shifts as the
# GUI is refactored (state/ and controllers/ deliberately don't), so this probes each candidate by
# import rather than hard-coding a file list.
_TKINTER_DEPENDENT_PREFIXES = (
    Path("src/rfdetr_demo/gui"),
    Path("src/rfdetr_demo/vast/start_progress.py"),
)

_TKINTER_IMPORTABLE = importlib.util.find_spec("tkinter") is not None


def _module_name_for(relpath: Path) -> str:
    """Convert a ``src``-relative file path to its importable dotted module name.

    Args:
        relpath: Path to a ``.py`` file, relative to the repository root.

    Returns:
        The dotted module name pytest's import machinery would use to import it.
    """
    parts = relpath.with_suffix("").parts[1:]  # drop the leading "src" path segment
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    """Skip collecting a GUI module that fails to import solely because tkinter is missing.

    Args:
        collection_path: Candidate file or directory pytest is deciding whether to collect.
        config: The running pytest session's config, used to resolve the repository root.

    Returns:
        ``True`` to skip the path, or ``None`` to leave the decision to pytest's defaults.
    """
    if _TKINTER_IMPORTABLE or collection_path.suffix != ".py":
        return None
    try:
        relpath = collection_path.relative_to(config.rootpath)
    except ValueError:
        return None
    if not any(relpath == prefix or prefix in relpath.parents for prefix in _TKINTER_DEPENDENT_PREFIXES):
        return None
    try:
        importlib.import_module(_module_name_for(relpath))
    except ModuleNotFoundError as exc:
        if exc.name in ("tkinter", "_tkinter"):
            return True
    return None
