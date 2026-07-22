"""No null principle checker for Elegant Objects violations."""

import ast
from collections.abc import Iterator
from types import NoneType
from typing import final

from .base import ErrorCodes, Instance, Principle, Source, Violations, violation

ANN_ASSIGN = Instance(ast.AnnAssign)
ARG = Instance(ast.arg)
CONSTANT = Instance(ast.Constant)
NOTHING = Instance(NoneType)
FUNCTION: Instance[ast.FunctionDef | ast.AsyncFunctionDef] = Instance((
    ast.FunctionDef,
    ast.AsyncFunctionDef,
))
RETURN = Instance(ast.Return)
SCOPE: Instance[ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda] = Instance((
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.Lambda,
))


@final
class NoNull(Principle):
    """Checks for None usage violations (EO005)."""

    def check(self, source: Source) -> Violations:
        """Check source for None usage violations."""
        node = source.node
        if FUNCTION.covers(node):
            return self._check_implicit_returns(node)
        if CONSTANT.covers(node) and NOTHING.covers(node.value):
            # Skip None in type annotations
            if self._is_in_type_annotation(node, source.tree):
                return []
            return violation(node, ErrorCodes.EO005)
        return []

    def _check_implicit_returns(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> Violations:
        """Check for bare returns in a function that returns values elsewhere."""
        returns = list(self._returns(node))
        if not any(each.value for each in returns):
            return []

        violations: Violations = []
        for each in returns:
            if NOTHING.covers(each.value):
                violations.extend(violation(each, ErrorCodes.EO005))
        return violations

    def _returns(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> Iterator[ast.Return]:
        """Yield the return statements belonging to this function only."""
        for child in ast.iter_child_nodes(node):
            if SCOPE.covers(child):
                continue
            for inner in ast.walk(child):
                if RETURN.covers(inner):
                    yield inner

    def _is_in_type_annotation(
        self, target_node: ast.AST, tree: ast.AST | None
    ) -> bool:
        """Check if the target node is within a type annotation context."""
        if not tree:
            return False

        # Find all annotation contexts in the tree
        for node in ast.walk(tree):
            # Function return annotations
            if FUNCTION.covers(node) and node.returns:
                if self._node_in_subtree(target_node, node.returns):
                    return True
            # Parameter annotations
            elif ARG.covers(node) and node.annotation:
                if self._node_in_subtree(target_node, node.annotation):
                    return True
            # Variable annotations
            elif ANN_ASSIGN.covers(node) and node.annotation:
                if self._node_in_subtree(target_node, node.annotation):
                    return True

        return False

    def _node_in_subtree(self, target: ast.AST, tree: ast.AST) -> bool:
        """Check if target node is within the tree."""
        return any(child is target for child in ast.walk(tree))
