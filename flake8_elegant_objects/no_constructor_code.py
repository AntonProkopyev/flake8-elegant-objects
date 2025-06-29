"""No constructor code principle checker for Elegant Objects violations."""

import ast

from .base import ErrorCodes, Source, Violations, is_method, violation


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
            if isinstance(stmt, ast.Assign):
                # Check if it's a simple self.attr = param assignment
                if len(stmt.targets) == 1 and isinstance(
                    stmt.targets[0], ast.Attribute
                ):
                    target = stmt.targets[0]
                    if isinstance(target.value, ast.Name) and target.value.id == "self":
                        # This is a self.attr assignment, check if value is a simple name
                        if not isinstance(stmt.value, ast.Name):
                            violations.extend(violation(stmt, ErrorCodes.EO006))
                    else:
                        violations.extend(violation(stmt, ErrorCodes.EO006))
                else:
                    violations.extend(violation(stmt, ErrorCodes.EO006))
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                if self._is_super_call(stmt.value):
                    continue
                violations.extend(violation(stmt, ErrorCodes.EO006))
            elif not isinstance(stmt, ast.Pass):  # Allow pass statements
                violations.extend(violation(stmt, ErrorCodes.EO006))

        return violations

    def _is_super_call(self, call: ast.Call) -> bool:
        """Check if a call is a super() call."""
        if isinstance(call.func, ast.Name) and call.func.id == "super":
            return True
        if isinstance(call.func, ast.Attribute) and isinstance(
            call.func.value, ast.Call
        ):
            return self._is_super_call(call.func.value)
        return False
