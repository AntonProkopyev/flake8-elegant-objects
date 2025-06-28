"""Immutability contract checker for proper immutability declarations."""

import ast

from ..base import ErrorCodes, Violations, violation


class ImmutabilityContractChecker:
    """Checks that classes properly declare and enforce immutability contracts."""

    def check_immutability_contract(self, node: ast.ClassDef) -> Violations:
        """Check if class has proper immutability declarations."""
        violations = []

        has_slots = any(
            isinstance(item, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "__slots__"
                for target in item.targets
            )
            for item in node.body
        )

        has_setattr_override = any(
            isinstance(item, ast.FunctionDef) and item.name == "__setattr__"
            for item in node.body
        )

        has_mutable_attrs = self._has_mutable_attributes(node)

        if has_mutable_attrs and not (has_slots or has_setattr_override):
            violations.extend(
                violation(
                    node,
                    ErrorCodes.EO024.format(
                        name=f"class {node.name} with mutable attributes but no immutability enforcement"
                    ),
                )
            )

        return violations

    def _has_mutable_attributes(self, node: ast.ClassDef) -> bool:
        """Check if class has mutable attributes in __init__."""
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                return self._check_init_for_mutable_attrs(item)
        return False

    def _check_init_for_mutable_attrs(self, init_method: ast.FunctionDef) -> bool:
        """Check if __init__ method has mutable attribute assignments."""
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
            and self._is_mutable_value(value)
        )

    @staticmethod
    def _is_mutable_value(node: ast.AST) -> bool:
        """Check if a value is mutable."""
        return isinstance(node, ast.List | ast.Dict | ast.Set)
