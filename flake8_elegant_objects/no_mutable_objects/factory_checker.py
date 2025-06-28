"""Factory method checker for immutable object creation."""

import ast

from ..base import ErrorCodes, Violations, violation


class FactoryMethodChecker:
    """Checks that objects are created immutably through factory methods."""

    def check_factory_pattern(self, node: ast.ClassDef) -> Violations:
        """Check if class follows immutable factory pattern."""
        violations = []
        has_mutable_init = False
        has_factory_methods = False

        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                if item.name == "__init__":
                    for stmt in ast.walk(item):
                        if isinstance(stmt, ast.Assign):
                            for target in stmt.targets:
                                if (
                                    isinstance(target, ast.Attribute)
                                    and isinstance(target.value, ast.Name)
                                    and target.value.id == "self"
                                    and self._is_mutable_init(stmt.value)
                                ):
                                    has_mutable_init = True

                elif self._returns_new_instance(item, node.name):
                    has_factory_methods = True

        if has_mutable_init and not has_factory_methods:
            violations.extend(
                violation(
                    node,
                    ErrorCodes.EO022.format(
                        name=f"class {node.name} with mutable state but no immutable factory methods"
                    ),
                )
            )

        return violations

    def _is_mutable_init(self, node: ast.AST) -> bool:
        """Check if initialization creates mutable state."""
        return isinstance(node, ast.List | ast.Dict | ast.Set)

    def _returns_new_instance(self, func: ast.FunctionDef, class_name: str) -> bool:
        """Check if function returns a new instance of the class."""
        for node in ast.walk(func):
            if isinstance(node, ast.Return) and node.value:
                if isinstance(node.value, ast.Call):
                    if (
                        isinstance(node.value.func, ast.Name)
                        and node.value.func.id == class_name
                    ):
                        return True
        return False
