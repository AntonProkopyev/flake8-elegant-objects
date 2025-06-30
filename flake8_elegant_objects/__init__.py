"""Flake8 plugin for Elegant Objects violations.

This plugin detects violations of the Elegant Objects principles including
"-er" entities, null usage, mutable objects, static methods, and more.

Based on Yegor Bugayenko's principles from elegantobjects.org
"""

import ast
from collections.abc import Iterator
from typing import Any

from .base import ElegantObjectsCore


class ElegantObjectsPlugin:
    """Flake8 plugin to check for Elegant Objects violations."""

    name = "flake8-elegant-objects"
    version = "1.1.1"

    def __init__(self, tree: ast.AST) -> None:
        self.tree = tree
        self._core = ElegantObjectsCore(tree)

    def run(self) -> Iterator[tuple[int, int, str, type["ElegantObjectsPlugin"]]]:
        """Run the checker and yield errors."""
        violations = self._core.check_violations()
        for violation in violations:
            yield (violation.line, violation.column, violation.message, type(self))


# Entry point for flake8 plugin registration
def factory(_app: Any) -> type[ElegantObjectsPlugin]:
    """Factory function for flake8 plugin."""
    return ElegantObjectsPlugin
