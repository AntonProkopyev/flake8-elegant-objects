"""Deep mutability checker for complex mutation patterns."""

import ast
from typing import final

from ..base import CLASS_DEF, ErrorCodes, Instance, Violations, violation

ASSIGN = Instance(ast.Assign)
ATTRIBUTE = Instance(ast.Attribute)
CALL = Instance(ast.Call)
FUNCTION_DEF = Instance(ast.FunctionDef)
NAME = Instance(ast.Name)
MUTABLE_LITERAL: Instance[ast.List | ast.Dict | ast.Set] = Instance((
    ast.List,
    ast.Dict,
    ast.Set,
))


@final
class DeepMutability:
    """Enhanced checker for deep mutability patterns."""

    def check_deep_mutations(self, tree: ast.AST) -> Violations:
        """Check for deep mutation patterns across the entire tree."""
        return Mutation(tree).violations()


@final
class Mutation:
    """Detects various mutation patterns."""

    def __init__(self, tree: ast.AST) -> None:
        self.tree = tree

    def violations(self) -> Violations:
        """Find every chained mutation in the tree."""
        return self._scan(self.tree, False, "")

    def _scan(self, node: ast.AST, in_class: bool, function: str) -> Violations:
        """Scan a node and its children, carrying the enclosing context."""
        found = self._chained_mutation(node, in_class, function)
        inner_class = in_class or CLASS_DEF.covers(node)
        inner_function = node.name if FUNCTION_DEF.covers(node) else function
        for child in ast.iter_child_nodes(node):
            found.extend(self._scan(child, inner_class, inner_function))
        return found

    def _chained_mutation(
        self, node: ast.AST, in_class: bool, function: str
    ) -> Violations:
        """Report a node that mutates state through a chained call."""
        if not in_class or function == "__init__":
            return []
        if CALL.covers(node) and self._is_chained_mutation(node):
            return violation(node, ErrorCodes.EO021.format(name="chained mutation"))
        return []

    def _is_chained_mutation(self, node: ast.Call) -> bool:
        """Detect chained mutations like self.dict.get('key', []).append()."""
        if ATTRIBUTE.covers(node.func):
            if node.func.attr in {"append", "extend", "add", "update", "remove"}:
                if CALL.covers(node.func.value):
                    inner_call = node.func.value
                    if (
                        ATTRIBUTE.covers(inner_call.func)
                        and ATTRIBUTE.covers(inner_call.func.value)
                        and NAME.covers(inner_call.func.value.value)
                        and inner_call.func.value.value.id == "self"
                    ):
                        return True
        return False
