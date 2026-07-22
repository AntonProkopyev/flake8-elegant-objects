"""No type discrimination principle checker for Elegant Objects violations."""

import ast
from typing import ClassVar, final

from .base import EO010, REPORT, Instance, Principle, Source, Violations

ATTRIBUTE = Instance(ast.Attribute)
CALL = Instance(ast.Call)
MATCH_CLASS = Instance(ast.MatchClass)
NAME = Instance(ast.Name)


@final
class NoTypeDiscrimination(Principle):
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

        if CALL.covers(node):
            return self._check_call(node)

        if ATTRIBUTE.covers(node):
            return self._check_attribute(node)

        if MATCH_CLASS.covers(node):
            return REPORT.of(node, EO010)

        return []

    def _check_call(self, node: ast.Call) -> Violations:
        """Check for isinstance, type casting, or reflection calls."""
        if self._callee(node.func) in self.FORBIDDEN_CALLS:
            return REPORT.of(node, EO010)
        return []

    def _check_attribute(self, node: ast.Attribute) -> Violations:
        """Check for reflection through dunder type attributes."""
        if node.attr in self.FORBIDDEN_ATTRIBUTES:
            return REPORT.of(node, EO010)
        return []

    def _callee(self, func: ast.expr) -> str:
        """Resolve the trailing name of a called expression."""
        if NAME.covers(func):
            return func.id
        if ATTRIBUTE.covers(func):
            return func.attr
        return ""
