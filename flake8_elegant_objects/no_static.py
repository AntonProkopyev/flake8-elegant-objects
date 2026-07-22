"""No static methods principle checker for Elegant Objects violations."""

import ast
from typing import ClassVar, final

from .base import EO009, REPORT, Instance, Principle, Source, Violations

ATTRIBUTE = Instance(ast.Attribute)
FUNCTION: Instance[ast.FunctionDef | ast.AsyncFunctionDef] = Instance((
    ast.FunctionDef,
    ast.AsyncFunctionDef,
))
MODULE = Instance(ast.Module)
NAME = Instance(ast.Name)


@final
class NoStatic(Principle):
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

        if FUNCTION.covers(node):
            return self._check_static_methods(node)

        if MODULE.covers(node):
            return self._check_utility_functions(node)

        return []

    def _check_utility_functions(self, node: ast.Module) -> Violations:
        """Check for module level functions, the Python static method."""
        violations: Violations = []
        for statement in node.body:
            if FUNCTION.covers(statement):
                if statement.name.startswith("_"):
                    continue
                violations.extend(
                    REPORT.of(statement, EO009.format(name=statement.name))
                )
        return violations

    def _check_static_methods(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> Violations:
        """Check for static methods violations."""
        # Check for @staticmethod decorator, plain or reached through a module
        for decorator in node.decorator_list:
            if self._decorator_name(decorator) in self.STATIC_DECORATORS:
                return REPORT.of(node, EO009.format(name=node.name))
        return []

    def _decorator_name(self, decorator: ast.expr) -> str:
        """Resolve the trailing name of a decorator expression."""
        if NAME.covers(decorator):
            return decorator.id
        if ATTRIBUTE.covers(decorator):
            return decorator.attr
        return ""
