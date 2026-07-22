"""Additional specific violation pattern detectors."""

import ast
from typing import final

from ..base import EO026, EO027, REPORT, Instance, Violations

ASSIGN = Instance(ast.Assign)
ATTRIBUTE = Instance(ast.Attribute)
NAME = Instance(ast.Name)
RETURN = Instance(ast.Return)


@final
class MutablePatterns:
    """Collection of specific mutable pattern detectors."""

    def detect_aliasing_violations(  # noqa: EO011
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> Violations:
        """Detect aliasing that can lead to external mutation."""
        violations = []

        for stmt in ast.walk(node):
            if RETURN.covers(stmt) and stmt.value:
                if (
                    ATTRIBUTE.covers(stmt.value)
                    and NAME.covers(stmt.value.value)
                    and stmt.value.value.id == "self"
                ):
                    # Only flag if attribute name suggests it's mutable data
                    # Skip private attributes (starting with _) as they're often used for immutable storage
                    attr_name = stmt.value.attr
                    if not attr_name.startswith("_"):
                        lowered = attr_name.lower()
                        mutable_attr_patterns = {
                            "data",
                            "items",
                            "list",
                            "dict",
                            "set",
                            "collection",
                            "values",
                            "cache",
                        }
                        if any(pattern in lowered for pattern in mutable_attr_patterns):
                            violations.extend(
                                REPORT.of(
                                    stmt,
                                    EO026.format(
                                        name=f"returning internal mutable state 'self.{stmt.value.attr}'"
                                    ),
                                )
                            )

        return violations

    def detect_defensive_copy_missing(  # noqa: EO011
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> Violations:
        """Detect missing defensive copies in constructors."""
        if node.name != "__init__":
            return []

        return self._check_init_defensive_copies(node)

    def _check_init_defensive_copies(
        self,
        init_node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> Violations:
        """Check __init__ method for missing defensive copies."""
        violations = []
        param_names = [arg.arg for arg in init_node.args.args[1:]]

        for stmt in ast.walk(init_node):
            if ASSIGN.covers(stmt):
                violations.extend(
                    self._check_assignment_defensive_copy(stmt, param_names)
                )

        return violations

    def _check_assignment_defensive_copy(
        self, stmt: ast.Assign, param_names: list[str]
    ) -> Violations:
        """Check assignment for missing defensive copy."""
        violations = []

        for target in stmt.targets:
            if self._is_self_attribute(target):
                if self._is_param_assignment(stmt.value, param_names):
                    assert NAME.covers(stmt.value)  # Type narrowing for mypy
                    violations.extend(
                        REPORT.of(
                            stmt,
                            EO027.format(
                                name=f"possible mutable parameter '{stmt.value.id}' assigned without defensive copy"
                            ),
                        )
                    )

        return violations

    def _is_self_attribute(self, target: ast.expr) -> bool:
        """Check if target is a self attribute."""
        return (
            ATTRIBUTE.covers(target)
            and NAME.covers(target.value)
            and target.value.id == "self"
        )

    def _is_param_assignment(self, value: ast.expr, param_names: list[str]) -> bool:
        """Check if value is a parameter assignment."""
        # Only flag if it's a direct parameter assignment AND likely mutable
        if NAME.covers(value) and value.id in param_names:
            # Don't flag assignments of simple types (strings, numbers, etc.)
            # Only flag if parameter name suggests it could be mutable (lists, data, items, etc.)
            mutable_param_patterns = {
                "data",
                "items",
                "list",
                "dict",
                "set",
                "collection",
                "values",
            }
            return any(
                pattern in value.id.lower() for pattern in mutable_param_patterns
            )
        return False
