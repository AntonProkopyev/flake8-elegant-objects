"""No constructor code principle checker for Elegant Objects violations."""

import ast
from typing import final

from .base import EO006, METHOD, REPORT, Instance, Principle, Source, Violations

FUNCTION: Instance[ast.FunctionDef | ast.AsyncFunctionDef] = Instance((
    ast.FunctionDef,
    ast.AsyncFunctionDef,
))
ASSIGN = Instance(ast.Assign)
ATTRIBUTE = Instance(ast.Attribute)
CALL = Instance(ast.Call)
CONSTANT = Instance(ast.Constant)
EXPR = Instance(ast.Expr)
NAME = Instance(ast.Name)
PASS = Instance(ast.Pass)
STRING = Instance(str)


@final
class NoConstructorCode(Principle):
    """Checks for code in constructors beyond parameter assignments (EO006)."""

    def check(self, source: Source) -> Violations:
        """Check source for constructor code violations."""
        node = source.node
        if not FUNCTION.covers(node):
            return []
        return self._check_constructor_code(node)

    def _check_constructor_code(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> Violations:
        """Check for code in constructors beyond parameter assignments."""
        if node.name != "__init__" or not METHOD.covers(node):
            return []

        violations = []
        # Constructors should only contain assignments to self.attribute = parameter
        for stmt in node.body:
            if not self._is_assembly(stmt):
                violations.extend(REPORT.of(stmt, EO006))

        return violations

    def _is_assembly(self, stmt: ast.stmt) -> bool:
        """Check if a constructor statement merely assembles the object."""
        if PASS.covers(stmt) or self._is_docstring(stmt):
            return True
        if EXPR.covers(stmt) and CALL.covers(stmt.value):
            return self._is_super_call(stmt.value)
        if ASSIGN.covers(stmt):
            return self._is_parameter_assignment(stmt)
        return False

    def _is_parameter_assignment(self, stmt: ast.Assign) -> bool:
        """Check if an assignment binds a parameter to a single self attribute."""
        if len(stmt.targets) != 1 or not ATTRIBUTE.covers(stmt.targets[0]):
            return False
        target = stmt.targets[0]
        if not NAME.covers(target.value) or target.value.id != "self":
            return False
        return NAME.covers(stmt.value)

    def _is_docstring(self, stmt: ast.stmt) -> bool:
        """Check if a statement is a docstring rather than executable code."""
        return (
            EXPR.covers(stmt)
            and CONSTANT.covers(stmt.value)
            and STRING.covers(stmt.value.value)
        )

    def _is_super_call(self, call: ast.Call) -> bool:
        """Check if a call is a super() call."""
        if NAME.covers(call.func) and call.func.id == "super":
            return True
        if ATTRIBUTE.covers(call.func) and CALL.covers(call.func.value):
            return self._is_super_call(call.func.value)
        return False
