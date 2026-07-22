"""Flake8 plugin for Elegant Objects violations.

This plugin detects violations of the Elegant Objects principles including
"-er" entities, null usage, mutable objects, static methods, and more.

Based on Yegor Bugayenko's principles from elegantobjects.org
"""

import ast
from collections.abc import Iterator
from typing import Any, final

from .base import ElegantObjectsCore


@final
class ElegantObjectsPlugin:
    """Flake8 plugin to check for Elegant Objects violations."""

    name = "flake8-elegant-objects"
    version = "2.0.0"

    def __init__(self, tree: ast.AST) -> None:
        self.tree = tree

    def run(self) -> Iterator[tuple[int, int, str, type["ElegantObjectsPlugin"]]]:  # noqa: EO011
        """Run the checker and yield errors."""
        for violation in ElegantObjectsCore(self.tree).check_violations():
            yield (violation.line, violation.column, violation.message, type(self))  # noqa: EO010


# Entry point for flake8 plugin registration
def factory(_app: Any) -> type[ElegantObjectsPlugin]:  # noqa: EO009
    """Factory function for flake8 plugin."""
    return ElegantObjectsPlugin
