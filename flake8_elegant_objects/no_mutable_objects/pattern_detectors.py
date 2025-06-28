"""Additional specific violation pattern detectors."""

import ast

from ..base import ErrorCodes, Violations, violation


class MutablePatternDetectors:
    """Collection of specific mutable pattern detectors."""

    @staticmethod
    def detect_aliasing_violations(node: ast.FunctionDef) -> Violations:
        """Detect aliasing that can lead to external mutation."""
        violations = []

        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Return) and stmt.value:
                if (
                    isinstance(stmt.value, ast.Attribute)
                    and isinstance(stmt.value.value, ast.Name)
                    and stmt.value.value.id == "self"
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
    def detect_defensive_copy_missing(node: ast.FunctionDef) -> Violations:
        """Detect missing defensive copies in constructors."""
        violations = []

        if node.name == "__init__":
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if (
                            isinstance(target, ast.Attribute)
                            and isinstance(target.value, ast.Name)
                            and target.value.id == "self"
                        ):
                            if isinstance(stmt.value, ast.Name) and stmt.value.id in [
                                arg.arg for arg in node.args.args[1:]
                            ]:
                                violations.extend(
                                    violation(
                                        stmt,
                                        ErrorCodes.EO027.format(
                                            name=f"possible mutable parameter '{stmt.value.id}' assigned without defensive copy"
                                        ),
                                    )
                                )

        return violations
