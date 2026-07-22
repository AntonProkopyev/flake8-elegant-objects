"""Base classes and utilities for mutable object checkers."""

import ast
from typing import final

from ..base import Instance

CALL = Instance(ast.Call)
NAME = Instance(ast.Name)
MUTABLE_LITERAL: Instance[ast.List | ast.Dict | ast.Set] = Instance((
    ast.List,
    ast.Dict,
    ast.Set,
))
MUTABLE_COMPREHENSION: Instance[ast.ListComp | ast.DictComp | ast.SetComp] = Instance((
    ast.ListComp,
    ast.DictComp,
    ast.SetComp,
))


@final
class MutableType:
    """A syntax node, asked whether it builds a mutable value."""

    def covers(self, node: ast.AST) -> bool:
        """Answer whether the node represents a mutable type."""
        if MUTABLE_LITERAL.covers(node):
            return True

        if CALL.covers(node) and NAME.covers(node.func):
            mutable_types = {"list", "dict", "set", "bytearray", "deque", "defaultdict"}
            return node.func.id in mutable_types

        return bool(MUTABLE_COMPREHENSION.covers(node))


MUTABLE_TYPE = MutableType()
