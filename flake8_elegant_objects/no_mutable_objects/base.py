"""Base classes and utilities for mutable object checkers."""

import ast
from typing import Any

from ..base import Violations


class MutabilityVisitor(ast.NodeVisitor):
    """Helper visitor to track parent nodes for better mutation detection."""

    def __init__(self, checker: Any, current_class: ast.ClassDef | None) -> None:
        self.checker = checker
        self.current_class = current_class
        self.violations: Violations = []

    def visit(self, node: ast.AST) -> None:
        """Visit nodes and set parent references."""
        for child in ast.iter_child_nodes(node):
            setattr(child, "_parent", node)  # noqa: B010
        self.generic_visit(node)


class MutableStateTracker:
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
    if isinstance(node, ast.List | ast.Dict | ast.Set):
        return True

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        mutable_types = {"list", "dict", "set", "bytearray", "deque", "defaultdict"}
        return node.func.id in mutable_types

    return bool(isinstance(node, ast.ListComp | ast.DictComp | ast.SetComp))
