"""No null principle checker for Elegant Objects violations."""

import ast
from collections.abc import Iterator
from types import NoneType
from typing import final

from .base import NOTHING as ABSENT
from .base import EO005, REPORT, Instance, Parents, Principle, Source, Violations

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
            if self._annotated(node, source.parents):
                return []
            return REPORT.of(node, EO005)
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
                violations.extend(REPORT.of(each, EO005))
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

    def _annotated(self, node: ast.AST, parents: Parents) -> bool:
        """Whether this None sits inside a type annotation.

        The question used to be asked downwards: walk the whole tree,
        collect every annotation, look for the node inside it. That is the
        tree once per None, and a file with hundreds of them pays for the
        tree hundreds of times. Asked upwards, the answer costs the depth
        of the nesting and nothing more.
        """
        held = node
        above = parents.above(held)
        while above is not ABSENT:
            if FUNCTION.covers(above) and above.returns is held:
                return True
            if ARG.covers(above) and above.annotation is held:
                return True
            if ANN_ASSIGN.covers(above) and above.annotation is held:
                return True
            held, above = above, parents.above(above)
        return False
