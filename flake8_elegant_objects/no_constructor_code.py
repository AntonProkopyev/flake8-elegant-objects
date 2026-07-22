"""No constructor code principle checker for Elegant Objects violations."""

import ast
from typing import final

from .base import ErrorCodes, Source, Violations, is_method, violation


@final
class NoConstructorCode:
    """Checks for code in constructors beyond parameter assignments (EO006)."""

    def check(self, source: Source) -> Violations:
        """Check source for constructor code violations."""
        node = source.node
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            return []
        return self._check_constructor_code(node)

    def _check_constructor_code(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> Violations:
        """Check for code in constructors beyond parameter assignments."""
        if node.name != "__init__" or not is_method(node):
            return []

        violations = []
        # Constructors should only contain assignments to self.attribute = parameter
        for stmt in node.body:
            if not self._is_assembly(stmt):
                violations.extend(violation(stmt, ErrorCodes.EO006))

        return violations

    def _is_assembly(self, stmt: ast.stmt) -> bool:
        """Check if a constructor statement merely assembles the object."""
        if isinstance(stmt, ast.Pass) or self._is_docstring(stmt):
            return True
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            return self._is_super_call(stmt.value)
        if isinstance(stmt, ast.Assign):
            return self._is_parameter_assignment(stmt)
        return False

    def _is_parameter_assignment(self, stmt: ast.Assign) -> bool:
        """Check if an assignment binds a parameter to a single self attribute."""
        if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Attribute):
            return False
        target = stmt.targets[0]
        if not isinstance(target.value, ast.Name) or target.value.id != "self":
            return False
        return isinstance(stmt.value, ast.Name)

    def _is_docstring(self, stmt: ast.stmt) -> bool:
        """Check if a statement is a docstring rather than executable code."""
        return (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        )

    def _is_super_call(self, call: ast.Call) -> bool:
        """Check if a call is a super() call."""
        if isinstance(call.func, ast.Name) and call.func.id == "super":
            return True
        if isinstance(call.func, ast.Attribute) and isinstance(
            call.func.value, ast.Call
        ):
            return self._is_super_call(call.func.value)
        return False
