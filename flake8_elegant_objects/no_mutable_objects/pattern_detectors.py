"""Additional specific violation pattern detectors."""

import ast

from ..base import ErrorCodes, Violations, violation


class MutablePatternDetectors:
    """Collection of specific mutable pattern detectors."""

    @staticmethod
    def detect_aliasing_violations(
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> Violations:
        """Detect aliasing that can lead to external mutation."""
        violations = []

        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Return) and stmt.value:
                if (
                    isinstance(stmt.value, ast.Attribute)
                    and isinstance(stmt.value.value, ast.Name)
                    and stmt.value.value.id == "self"
                ):
                    # Only flag if attribute name suggests it's mutable data
                    # Skip private attributes (starting with _) as they're often used for immutable storage
                    attr_name = stmt.value.attr
                    if not attr_name.startswith("_"):
                        attr_lower = attr_name.lower()
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
                        if any(
                            pattern in attr_lower for pattern in mutable_attr_patterns
                        ):
                            violations.extend(
                                violation(
                                    stmt,
                                    ErrorCodes.EO026.format(
                                        name=f"returning internal mutable state 'self.{stmt.value.attr}'"
                                    ),
                                )
                            )

        return violations

    @staticmethod
    def detect_defensive_copy_missing(
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> Violations:
        """Detect missing defensive copies in constructors."""
        if node.name != "__init__":
            return []

        return MutablePatternDetectors._check_init_defensive_copies(node)

    @staticmethod
    def _check_init_defensive_copies(
        init_node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> Violations:
        """Check __init__ method for missing defensive copies."""
        violations = []
        param_names = [arg.arg for arg in init_node.args.args[1:]]

        for stmt in ast.walk(init_node):
            if isinstance(stmt, ast.Assign):
                violations.extend(
                    MutablePatternDetectors._check_assignment_defensive_copy(
                        stmt, param_names
                    )
                )

        return violations

    @staticmethod
    def _check_assignment_defensive_copy(
        stmt: ast.Assign, param_names: list[str]
    ) -> Violations:
        """Check assignment for missing defensive copy."""
        violations = []

        for target in stmt.targets:
            if MutablePatternDetectors._is_self_attribute(target):
                if MutablePatternDetectors._is_param_assignment(
                    stmt.value, param_names
                ):
                    assert isinstance(stmt.value, ast.Name)  # Type narrowing for mypy
                    violations.extend(
                        violation(
                            stmt,
                            ErrorCodes.EO027.format(
                                name=f"possible mutable parameter '{stmt.value.id}' assigned without defensive copy"
                            ),
                        )
                    )

        return violations

    @staticmethod
    def _is_self_attribute(target: ast.expr) -> bool:
        """Check if target is a self attribute."""
        return (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
        )

    @staticmethod
    def _is_param_assignment(value: ast.expr, param_names: list[str]) -> bool:
        """Check if value is a parameter assignment."""
        # Only flag if it's a direct parameter assignment AND likely mutable
        if isinstance(value, ast.Name) and value.id in param_names:
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
