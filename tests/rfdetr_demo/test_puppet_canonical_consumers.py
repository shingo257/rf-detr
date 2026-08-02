# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests that keep mascot animation consumers on canonical modules."""

from __future__ import annotations

import ast
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_MODULE = "rfdetr_demo.animation.puppet_render"
CANONICAL_CLI_MODULE = "rfdetr_demo.animation.puppet_cli"
CANONICAL_VIDEO_MODULE = "rfdetr_demo.animation.puppet_video"


def test_kirby_notebooks_import_video_renderer_from_canonical_module() -> None:
    """Require every Kirby notebook consumer to bypass the legacy facade."""
    notebook_paths = sorted((REPO_ROOT / "docs" / "cookbooks").glob("kirby_keypoint_tracking*.ipynb"))
    assert notebook_paths

    legacy_consumers: list[str] = []
    missing_canonical_import: list[str] = []
    for notebook_path in notebook_paths:
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code"
        )
        if LEGACY_MODULE in code:
            legacy_consumers.append(notebook_path.name)
        if "render_puppet_video" in code and CANONICAL_VIDEO_MODULE not in code:
            missing_canonical_import.append(notebook_path.name)

    assert legacy_consumers == []
    assert missing_canonical_import == []


def test_animation_docs_use_canonical_cli_module() -> None:
    """Keep executable documentation on the canonical CLI entry point."""
    doc_paths = [
        REPO_ROOT / "docs" / "ja" / "kirby-mesh-animation.md",
        REPO_ROOT / "docs" / "ja" / "torso-animation-controls.md",
    ]
    legacy_consumers = [
        str(path.relative_to(REPO_ROOT))
        for path in doc_paths
        if f"python -m {LEGACY_MODULE}" in path.read_text(encoding="utf-8")
    ]
    missing_canonical_command = [
        str(path.relative_to(REPO_ROOT))
        for path in doc_paths
        if f"python -m {CANONICAL_CLI_MODULE}" not in path.read_text(encoding="utf-8")
    ]

    assert legacy_consumers == []
    assert missing_canonical_command == []


def test_compact_colab_builder_requires_canonical_video_module() -> None:
    """Validate the runtime bundle through its canonical video module."""
    builder_path = REPO_ROOT / "scripts" / "build_kirby_compact_colab_notebook.py"
    builder = builder_path.read_text(encoding="utf-8")

    assert "/ 'puppet_render.py'" not in builder
    assert "/ 'puppet_video.py'" in builder


def test_production_modules_do_not_import_legacy_puppet_facade() -> None:
    """Prevent canonical production modules from depending on the facade."""
    animation_dir = REPO_ROOT / "src" / "rfdetr_demo" / "animation"
    legacy_consumers: list[str] = []
    for path in sorted(animation_dir.glob("*.py")):
        if path.name == "puppet_render.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports_legacy_facade = any(
            (isinstance(node, ast.ImportFrom) and node.module == LEGACY_MODULE)
            or (isinstance(node, ast.Import) and any(alias.name == LEGACY_MODULE for alias in node.names))
            for node in ast.walk(tree)
        )
        if imports_legacy_facade:
            legacy_consumers.append(path.name)

    assert legacy_consumers == []
