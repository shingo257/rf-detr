"""Detect circular imports within the rfdetr_demo package (stdlib only)."""

from __future__ import annotations

import ast
import sys
from collections import defaultdict
from pathlib import Path


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


def main() -> int:
    root = _repo_root()
    package_root = root / "src" / "rfdetr_demo"
    package_prefix = "rfdetr_demo"
    if not package_root.is_dir():
        print(f"Package not found: {package_root}", file=sys.stderr)
        return 1

    graph = _build_graph(package_root, package_prefix)
    cycles = _find_cycles(graph)
    module_count = len({node for node in graph} | {n for deps in graph.values() for n in deps})

    print(f"Scanned {module_count} rfdetr_demo modules")
    if not cycles:
        print("No import cycles detected.")
        return 0

    print(f"Found {len(cycles)} cycle(s):")
    for index, cycle in enumerate(cycles, start=1):
        print(f"  {index}. {' -> '.join(cycle)}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
