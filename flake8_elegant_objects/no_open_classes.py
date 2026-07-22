"""No open classes principle checker for Elegant Objects violations.

Classes are final and hold one to four attributes. A class left open invites
implementation inheritance, and a class holding more than four attributes has
stopped being one object.
"""

import ast
from typing import ClassVar, final

from .base import ErrorCodes, Instance, Source, Violations, violation

MAX_ATTRIBUTES = 4

ANN_ASSIGN = Instance(ast.AnnAssign)
ASSIGN = Instance(ast.Assign)
ATTRIBUTE = Instance(ast.Attribute)
CLASS_DEF = Instance(ast.ClassDef)
FUNCTION: Instance[ast.FunctionDef | ast.AsyncFunctionDef] = Instance((
    ast.FunctionDef,
    ast.AsyncFunctionDef,
))
NAME = Instance(ast.Name)


@final
class NoOpenClasses:
    """Checks for non-final classes (EO028) and overweight classes (EO029)."""

    CONTRACT_BASES: ClassVar[set[str]] = {"ABC", "ABCMeta", "Protocol"}

    def check(self, source: Source) -> Violations:
        """Check source for open or overweight classes."""
        node = source.node
        if not CLASS_DEF.covers(node):
            return []

        violations: Violations = []
        violations.extend(self._check_final(node))
        violations.extend(self._check_attributes(node))
        return violations

    def _check_final(self, node: ast.ClassDef) -> Violations:
        """Check that a class is closed for inheritance."""
        if self._is_contract(node) or self._is_test(node):
            return []
        if any(self._trailing_name(each) == "final" for each in node.decorator_list):
            return []
        return violation(node, ErrorCodes.EO028.format(name=node.name))

    def _check_attributes(self, node: ast.ClassDef) -> Violations:
        """Check that a class holds no more than four attributes."""
        if self._is_contract(node) or self._is_test(node):
            return []

        names = self._attributes(node)
        if len(names) <= MAX_ATTRIBUTES:
            return []
        return violation(
            node,
            ErrorCodes.EO029.format(name=f"class {node.name} holds {len(names)}"),
        )

    def _attributes(self, node: ast.ClassDef) -> set[str]:
        """Collect the attribute names a class declares."""
        names: set[str] = set()
        for statement in node.body:
            if ANN_ASSIGN.covers(statement) and NAME.covers(statement.target):
                names.add(statement.target.id)
            elif ASSIGN.covers(statement):
                for target in statement.targets:
                    if NAME.covers(target):
                        names.add(target.id)
            elif FUNCTION.covers(statement):
                names.update(self._assigned_attributes(statement))
        return names

    def _assigned_attributes(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> set[str]:
        """Collect the self attributes a method assigns."""
        names: set[str] = set()
        for inner in ast.walk(node):
            targets: list[ast.expr] = []
            if ASSIGN.covers(inner):
                targets = list(inner.targets)
            elif ANN_ASSIGN.covers(inner):
                targets = [inner.target]
            for target in targets:
                if (
                    ATTRIBUTE.covers(target)
                    and NAME.covers(target.value)
                    and target.value.id == "self"
                ):
                    names.add(target.attr)
        return names

    def _is_contract(self, node: ast.ClassDef) -> bool:
        """Check if a class is a contract, which must stay open."""
        for base in node.bases:
            name = self._trailing_name(base)
            if name in self.CONTRACT_BASES or name.endswith("Protocol"):
                return True
        return any(keyword.arg == "metaclass" for keyword in node.keywords)

    def _is_test(self, node: ast.ClassDef) -> bool:
        """Check if a class is a test suite rather than an object."""
        return node.name.startswith("Test")

    def _trailing_name(self, node: ast.expr) -> str:
        """Resolve the trailing name of an expression."""
        if NAME.covers(node):
            return node.id
        if ATTRIBUTE.covers(node):
            return node.attr
        return ""
