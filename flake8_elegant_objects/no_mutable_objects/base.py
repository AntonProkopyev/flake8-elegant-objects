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
class MutableState:
    """Tracks mutable state across class definitions."""

    def __init__(self) -> None:
        self.instance_attrs: dict[str, set[str]] = {}
        self.mutable_attrs: dict[str, set[str]] = {}

    def add_instance_attr(
        self, class_name: str, attr_name: str, is_mutable: bool
    ) -> None:
        """Track instance attribute and whether it's mutable."""
        if class_name not in self.instance_attrs:
            self.instance_attrs[class_name] = set()
            self.mutable_attrs[class_name] = set()

        self.instance_attrs[class_name].add(attr_name)
        if is_mutable:
            self.mutable_attrs[class_name].add(attr_name)

    def is_mutable_attr(self, class_name: str, attr_name: str) -> bool:
        """Check if an attribute is known to be mutable."""
        return attr_name in self.mutable_attrs.get(class_name, set())


def is_mutable_type(node: ast.AST) -> bool:
    """Check if a node represents a mutable type."""
    if MUTABLE_LITERAL.covers(node):
        return True

    if CALL.covers(node) and NAME.covers(node.func):
        mutable_types = {"list", "dict", "set", "bytearray", "deque", "defaultdict"}
        return node.func.id in mutable_types

    return bool(MUTABLE_COMPREHENSION.covers(node))
