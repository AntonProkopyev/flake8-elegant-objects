"""Copy-on-write checker for proper immutability patterns."""

import ast

from ..base import ErrorCodes, Violations, violation


class CopyOnWriteChecker:
    """Checks for proper copy-on-write patterns for immutability."""

    def check_copy_on_write(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, class_name: str
    ) -> Violations:
        """Check if mutations properly implement copy-on-write."""
        violations: Violations = []

        if node.name == "__init__":
            return violations

        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                    ):
                        if not self._returns_new_instance(node, class_name):
                            violations.extend(
                                violation(
                                    stmt,
                                    ErrorCodes.EO025.format(
                                        name=f"mutation in {node.name} without returning new instance"
                                    ),
                                )
                            )

        return violations

    @staticmethod
    def _returns_new_instance(
        func: ast.FunctionDef | ast.AsyncFunctionDef, class_name: str
    ) -> bool:
        """Check if function returns a new instance."""
        for node in ast.walk(func):
            if isinstance(node, ast.Return) and node.value:
                if CopyOnWriteChecker._is_class_constructor_call(
                    node.value, class_name
                ):
                    return True
        return False

    @staticmethod
    def _is_class_constructor_call(call_node: ast.expr, class_name: str) -> bool:
        """Check if call is a constructor for the given class."""
        return (
            isinstance(call_node, ast.Call)
            and isinstance(call_node.func, ast.Name)
            and call_node.func.id == class_name
        )
