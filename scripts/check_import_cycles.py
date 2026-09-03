"""Detect circular imports and layer-boundary violations in rfdetr_demo."""

from __future__ import annotations

import ast
import sys
from collections import defaultdict
from pathlib import Path

# GUI may use the Vast safety *facade*, but must not reach into split internals.
_GUI_FORBIDDEN_VAST_INTERNALS: tuple[str, ...] = (
    "rfdetr_demo.vast.safety_guardrails",
    "rfdetr_demo.vast.safety_lease",
    "rfdetr_demo.vast.safety_settings",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _module_name_for_path(package_root: Path, path: Path) -> str:
    relative = path.relative_to(package_root.parent)
    parts = list(relative.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][:-3]
    return ".".join(parts)


def _collect_imports(path: Path, package_prefix: str) -> set[str]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if name == package_prefix or name.startswith(f"{package_prefix}."):
                    imports.add(name)
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            module = node.module
            if node.level and node.level > 0:
                continue
            if module == package_prefix or module.startswith(f"{package_prefix}."):
                imports.add(module)
    return imports


def _build_graph(package_root: Path, package_prefix: str) -> dict[str, set[str]]:
    graph: dict[str, set[str]] = defaultdict(set)
    for path in sorted(package_root.rglob("*.py")):
        module = _module_name_for_path(package_root, path)
        for imported in _collect_imports(path, package_prefix):
            graph[module].add(imported)
    return graph


def _find_cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    cycles: list[list[str]] = []
    visited: set[str] = set()
    stack: list[str] = []
    on_stack: set[str] = set()

    def visit(node: str) -> None:
        if node in on_stack:
            start = stack.index(node)
            cycles.append(stack[start:] + [node])
            return
        if node in visited:
            return
        visited.add(node)
        on_stack.add(node)
        stack.append(node)
        for neighbor in sorted(graph.get(node, ())):
            visit(neighbor)
        stack.pop()
        on_stack.remove(node)

    for node in sorted(graph):
        visit(node)
    return cycles


def _find_gui_vast_internal_imports(graph: dict[str, set[str]]) -> list[str]:
    """Return GUI modules that import Vast safety internals (bypass facade)."""
    violations: list[str] = []
    for module, deps in sorted(graph.items()):
        if not module.startswith("rfdetr_demo.gui"):
            continue
        for forbidden in _GUI_FORBIDDEN_VAST_INTERNALS:
            if forbidden in deps or any(dep.startswith(f"{forbidden}.") for dep in deps):
                violations.append(f"{module} -> {forbidden}")
    return violations


def main() -> int:
    root = _repo_root()
    package_root = root / "src" / "rfdetr_demo"
    package_prefix = "rfdetr_demo"
    if not package_root.is_dir():
        print(f"Package not found: {package_root}", file=sys.stderr)
        return 1

    graph = _build_graph(package_root, package_prefix)
    cycles = _find_cycles(graph)
    boundary_violations = _find_gui_vast_internal_imports(graph)
    module_count = len({node for node in graph} | {n for deps in graph.values() for n in deps})

    print(f"Scanned {module_count} rfdetr_demo modules")
    exit_code = 0

    if cycles:
        exit_code = 1
        print(f"Found {len(cycles)} cycle(s):")
        for index, cycle in enumerate(cycles, start=1):
            print(f"  {index}. {' -> '.join(cycle)}")
    else:
        print("No import cycles detected.")

    if boundary_violations:
        exit_code = 1
        print(f"Found {len(boundary_violations)} GUI→Vast safety-internal import(s):")
        for item in boundary_violations:
            print(f"  - {item}")
        print("Use rfdetr_demo.vast.safety facade from GUI code instead.")
    else:
        print("No GUI→Vast safety-internal boundary violations.")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
