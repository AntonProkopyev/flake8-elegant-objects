"""No impure tests principle checker for Elegant Objects violations."""

import ast
from typing import final

from .base import ErrorCodes, Source, Violations, violation


@final
class NoImpureTests:
    """Checks for impure test methods violations (EO012)."""

    def check(self, source: Source) -> Violations:
        """Check source for impure test method violations."""
        node = source.node

        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            return self._check_test_methods(node)

        return []

    def _check_test_methods(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> Violations:
        """Check that test methods only contain single assertion statements."""
        if not node.name.startswith("test_"):
            return []

        violations = []
        assertions = []

        body = self._without_docstring(node.body)
        for position, stmt in enumerate(body):
            violation_found, is_assertion = self._analyze_statement(stmt, node.name)

            if violation_found:
                violations.extend(violation_found)

            if is_assertion:
                assertions.append(position)

        violations.extend(self._validate_assertions(assertions, body, node))
        return violations

    def _without_docstring(self, body: list[ast.stmt]) -> list[ast.stmt]:
        """Drop the leading docstring, which is documentation and not code."""
        if not body:
            return body
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            return body[1:]
        return body

    def _analyze_statement(
        self, stmt: ast.stmt, test_name: str
    ) -> tuple[Violations, bool]:
        """Analyze a statement and return violations and whether it's an assertion."""
        if isinstance(stmt, ast.Pass):
            return [], False

        if isinstance(stmt, ast.Assert):
            return [], True

        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            return self._handle_expression_call(stmt, test_name)

        if isinstance(stmt, ast.With):
            return self._handle_with_statement(stmt, test_name)

        return violation(stmt, ErrorCodes.EO012.format(name=test_name)), False

    def _handle_expression_call(
        self, stmt: ast.Expr, test_name: str
    ) -> tuple[Violations, bool]:
        """Handle expression call statements."""
        if isinstance(stmt.value, ast.Call) and self._is_assertion_call(stmt.value):
            return [], True
        return violation(stmt, ErrorCodes.EO012.format(name=test_name)), False

    def _handle_with_statement(
        self, stmt: ast.With, test_name: str
    ) -> tuple[Violations, bool]:
        """Handle with statement for assertion context managers."""
        if self._is_assertion_context_manager(stmt):
            return [], True
        return violation(stmt, ErrorCodes.EO012.format(name=test_name)), False

    def _validate_assertions(
        self,
        assertions: list[int],
        body: list[ast.stmt],
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> Violations:
        """Validate that a test holds one assertion and closes with it."""
        if len(assertions) != 1:
            return violation(node, ErrorCodes.EO012_COUNT.format(name=node.name))
        if assertions[0] != len(body) - 1:
            return violation(node, ErrorCodes.EO012_ORDER.format(name=node.name))
        return []

    def _is_assertion_call(self, call: ast.Call) -> bool:
        """Check if a call is an assertion."""
        # Check for unittest style assertions (self.assertEqual, self.assertTrue, etc.)
        if isinstance(call.func, ast.Attribute):
            if call.func.attr.startswith("assert"):
                return True
            # Check for chained assertions like assertThat(...).isEqualTo(...)
            if self._contains_assertion_in_chain(call):
                return True

        # Check for standalone assertion functions
        if isinstance(call.func, ast.Name):
            if call.func.id.startswith("assert") or call.func.id == "assertThat":
                return True

        return False

    def _contains_assertion_in_chain(self, call: ast.Call) -> bool:
        """Check if assertion exists anywhere in the call chain."""
        current = call
        while isinstance(current, ast.Call):
            if isinstance(current.func, ast.Name):
                if (
                    current.func.id.startswith("assert")
                    or current.func.id == "assertThat"
                ):
                    return True
            elif isinstance(current.func, ast.Attribute):
                if (
                    current.func.attr.startswith("assert")
                    or current.func.attr == "assertThat"
                ):
                    return True
                # Move to the next level in the chain
                if isinstance(current.func.value, ast.Call):
                    current = current.func.value
                else:
                    break
            else:
                break
        return False

    def _is_assertion_context_manager(self, with_stmt: ast.With) -> bool:
        """Check if with statement is for assertions like pytest.raises."""
        for item in with_stmt.items:
            if isinstance(item.context_expr, ast.Call):
                if isinstance(item.context_expr.func, ast.Attribute):
                    # Check for pytest.raises, unittest.assertRaises, etc.
                    if item.context_expr.func.attr in {"raises", "assertRaises"}:
                        return True
                elif isinstance(item.context_expr.func, ast.Name):
                    if item.context_expr.func.id in {"raises", "assertRaises"}:
                        return True
        return False
