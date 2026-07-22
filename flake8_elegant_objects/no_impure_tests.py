"""No impure tests principle checker for Elegant Objects violations."""

import ast
from typing import final

from .base import ErrorCodes, Instance, Principle, Source, Violations, violation

FUNCTION: Instance[ast.FunctionDef | ast.AsyncFunctionDef] = Instance((
    ast.FunctionDef,
    ast.AsyncFunctionDef,
))
ASSERT = Instance(ast.Assert)
ATTRIBUTE = Instance(ast.Attribute)
CALL = Instance(ast.Call)
CONSTANT = Instance(ast.Constant)
EXPR = Instance(ast.Expr)
NAME = Instance(ast.Name)
PASS = Instance(ast.Pass)
WITH = Instance(ast.With)
STRING = Instance(str)


@final
class NoImpureTests(Principle):
    """Checks for impure test methods violations (EO012)."""

    def check(self, source: Source) -> Violations:
        """Check source for impure test method violations."""
        node = source.node

        if FUNCTION.covers(node):
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
            EXPR.covers(first)
            and CONSTANT.covers(first.value)
            and STRING.covers(first.value.value)
        ):
            return body[1:]
        return body

    def _analyze_statement(
        self, stmt: ast.stmt, test_name: str
    ) -> tuple[Violations, bool]:
        """Analyze a statement and return violations and whether it's an assertion."""
        if PASS.covers(stmt):
            return [], False

        if ASSERT.covers(stmt):
            return [], True

        if EXPR.covers(stmt) and CALL.covers(stmt.value):
            return self._handle_expression_call(stmt, test_name)

        if WITH.covers(stmt):
            return self._handle_with_statement(stmt, test_name)

        return violation(stmt, ErrorCodes.EO012.format(name=test_name)), False

    def _handle_expression_call(
        self, stmt: ast.Expr, test_name: str
    ) -> tuple[Violations, bool]:
        """Handle expression call statements."""
        if CALL.covers(stmt.value) and self._is_assertion_call(stmt.value):
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
        if ATTRIBUTE.covers(call.func):
            if call.func.attr.startswith("assert"):
                return True
            # Check for chained assertions like assertThat(...).isEqualTo(...)
            if self._contains_assertion_in_chain(call):
                return True

        # Check for standalone assertion functions
        if NAME.covers(call.func):
            if call.func.id.startswith("assert") or call.func.id == "assertThat":
                return True

        return False

    def _contains_assertion_in_chain(self, call: ast.Call) -> bool:
        """Check if assertion exists anywhere in the call chain."""
        current = call
        while CALL.covers(current):
            if NAME.covers(current.func):
                if (
                    current.func.id.startswith("assert")
                    or current.func.id == "assertThat"
                ):
                    return True
            elif ATTRIBUTE.covers(current.func):
                if (
                    current.func.attr.startswith("assert")
                    or current.func.attr == "assertThat"
                ):
                    return True
                # Move to the next level in the chain
                if CALL.covers(current.func.value):
                    current = current.func.value
                else:
                    break
            else:
                break
        return False

    def _is_assertion_context_manager(self, with_stmt: ast.With) -> bool:
        """Check if with statement is for assertions like pytest.raises."""
        for item in with_stmt.items:
            if CALL.covers(item.context_expr):
                if ATTRIBUTE.covers(item.context_expr.func):
                    # Check for pytest.raises, unittest.assertRaises, etc.
                    if item.context_expr.func.attr in {"raises", "assertRaises"}:
                        return True
                elif NAME.covers(item.context_expr.func):
                    if item.context_expr.func.id in {"raises", "assertRaises"}:
                        return True
        return False
