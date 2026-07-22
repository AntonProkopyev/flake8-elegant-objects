"""No type discrimination principle checker for Elegant Objects violations."""

import ast
from typing import ClassVar

from .base import ErrorCodes, Source, Violations, violation


class NoTypeDiscrimination:
    """Checks for type discrimination violations (EO010)."""

    FORBIDDEN_CALLS: ClassVar[set[str]] = {
        "callable",
        "cast",
        "delattr",
        "getattr",
        "hasattr",
        "isinstance",
        "issubclass",
        "setattr",
        "type",
    }

    FORBIDDEN_ATTRIBUTES: ClassVar[set[str]] = {
        "__bases__",
        "__class__",
        "__mro__",
        "__subclasses__",
    }

    def check(self, source: Source) -> Violations:
        """Check source for type discrimination violations."""
        node = source.node

        if isinstance(node, ast.Call):
            return self._check_call(node)

        if isinstance(node, ast.Attribute):
            return self._check_attribute(node)

        if isinstance(node, ast.MatchClass):
            return violation(node, ErrorCodes.EO010)

        return []

    def _check_call(self, node: ast.Call) -> Violations:
        """Check for isinstance, type casting, or reflection calls."""
        if self._callee(node.func) in self.FORBIDDEN_CALLS:
            return violation(node, ErrorCodes.EO010)
        return []

    def _check_attribute(self, node: ast.Attribute) -> Violations:
        """Check for reflection through dunder type attributes."""
        if node.attr in self.FORBIDDEN_ATTRIBUTES:
            return violation(node, ErrorCodes.EO010)
        return []

    def _callee(self, func: ast.expr) -> str:
        """Resolve the trailing name of a called expression."""
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
        return ""
