"""No static methods principle checker for Elegant Objects violations."""

import ast
from typing import ClassVar

from .base import ErrorCodes, Source, Violations, violation


class NoStatic:
    """Checks for static method violations (EO009)."""

    STATIC_DECORATORS: ClassVar[set[str]] = {
        "abstractclassmethod",
        "abstractstaticmethod",
        "classmethod",
        "staticmethod",
    }

    def check(self, source: Source) -> Violations:
        """Check source for static method violations."""
        node = source.node

        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            return self._check_static_methods(node)

        return []

    def _check_static_methods(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> Violations:
        """Check for static methods violations."""
        # Check for @staticmethod decorator, plain or reached through a module
        for decorator in node.decorator_list:
            if self._decorator_name(decorator) in self.STATIC_DECORATORS:
                return violation(node, ErrorCodes.EO009.format(name=node.name))
        return []

    def _decorator_name(self, decorator: ast.expr) -> str:
        """Resolve the trailing name of a decorator expression."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        if isinstance(decorator, ast.Attribute):
            return decorator.attr
        return ""
