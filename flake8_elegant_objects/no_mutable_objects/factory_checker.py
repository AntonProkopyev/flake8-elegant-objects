"""Factory method checker for immutable object creation."""

import ast

from ..base import ErrorCodes, Violations, violation


class FactoryMethodChecker:
    """Checks that objects are created immutably through factory methods."""

    def check_factory_pattern(self, node: ast.ClassDef) -> Violations:
        """Check if class follows immutable factory pattern."""
        violations = []
        has_mutable_init = self._has_mutable_init(node)
        has_factory_methods = self._has_factory_methods(node)

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

    def _has_mutable_init(self, node: ast.ClassDef) -> bool:
        """Check if class has mutable initialization."""
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                return self._check_init_mutations(item)
        return False

    def _check_init_mutations(self, init_method: ast.FunctionDef) -> bool:
        """Check if __init__ method has mutable assignments."""
        for stmt in ast.walk(init_method):
            if isinstance(stmt, ast.Assign) and self._has_mutable_self_assignment(stmt):
                return True
        return False

    def _has_mutable_self_assignment(self, stmt: ast.Assign) -> bool:
        """Check if assignment is a mutable self attribute."""
        for target in stmt.targets:
            if self._is_mutable_self_target(target, stmt.value):
                return True
        return False

    def _is_mutable_self_target(self, target: ast.expr, value: ast.expr) -> bool:
        """Check if target is a mutable self attribute assignment."""
        return (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and self._is_mutable_init(value)
        )

    def _has_factory_methods(self, node: ast.ClassDef) -> bool:
        """Check if class has factory methods."""
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and self._returns_new_instance(
                item, node.name
            ):
                return True
        return False

    @staticmethod
    def _is_mutable_init(node: ast.AST) -> bool:
        """Check if initialization creates mutable state."""
        return isinstance(node, ast.List | ast.Dict | ast.Set)

    @staticmethod
    def _returns_new_instance(func: ast.FunctionDef, class_name: str) -> bool:
        """Check if function returns a new instance of the class."""
        for node in ast.walk(func):
            if isinstance(node, ast.Return) and node.value:
                if FactoryMethodChecker._is_class_constructor_call(
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
