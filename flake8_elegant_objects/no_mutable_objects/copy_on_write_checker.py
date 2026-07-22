"""Copy-on-write checker for proper immutability patterns."""

import ast
from typing import final

from ..base import ErrorCodes, Instance, Violations, violation

ASSIGN = Instance(ast.Assign)
ATTRIBUTE = Instance(ast.Attribute)
CALL = Instance(ast.Call)
NAME = Instance(ast.Name)
RETURN = Instance(ast.Return)


@final
class CopyOnWrite:
    """Checks for proper copy-on-write patterns for immutability."""

    def check_copy_on_write(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, class_name: str
    ) -> Violations:
        """Check if mutations properly implement copy-on-write."""
        violations: Violations = []

        if node.name == "__init__":
            return violations

        for stmt in ast.walk(node):
            if ASSIGN.covers(stmt):
                for target in stmt.targets:
                    if (
                        ATTRIBUTE.covers(target)
                        and NAME.covers(target.value)
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

    def _returns_new_instance(
        self, func: ast.FunctionDef | ast.AsyncFunctionDef, class_name: str
    ) -> bool:
        """Check if function returns a new instance."""
        for node in ast.walk(func):
            if RETURN.covers(node) and node.value:
                if self._is_class_constructor_call(node.value, class_name):
                    return True
        return False

    def _is_class_constructor_call(self, call_node: ast.expr, class_name: str) -> bool:
        """Check if call is a constructor for the given class."""
        return (
            CALL.covers(call_node)
            and NAME.covers(call_node.func)
            and call_node.func.id == class_name
        )
