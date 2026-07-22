"""No implementation inheritance principle checker for Elegant Objects violations."""

import ast
from typing import final

from .base import ErrorCodes, Instance, Source, Violations, violation

ATTRIBUTE = Instance(ast.Attribute)
CLASS_DEF = Instance(ast.ClassDef)
NAME = Instance(ast.Name)


@final
class NoImplementationInheritance:
    """Checks for implementation inheritance violations (EO014)."""

    def check(self, source: Source) -> Violations:
        """Check source for implementation inheritance violations."""
        node = source.node

        if CLASS_DEF.covers(node):
            return self._check_implementation_inheritance(node, source.tree)

        return []

    def _is_contract_in_tree(self, name: str, tree: ast.AST | None) -> bool:
        """Check if a base class is a contract declared in this file."""
        if not tree:
            return False
        for node in ast.walk(tree):
            if CLASS_DEF.covers(node) and node.name == name:
                return any(
                    self._base_name(base) in {"ABC", "ABCMeta", "Protocol"}
                    for base in node.bases
                )
        return False

    def _base_name(self, base: ast.expr) -> str:
        """Resolve the trailing name of a base class expression."""
        if NAME.covers(base):
            return base.id
        if ATTRIBUTE.covers(base):
            return base.attr
        return ""

    def _check_implementation_inheritance(
        self, node: ast.ClassDef, tree: ast.AST | None
    ) -> Violations:
        """Check for implementation inheritance violations."""
        for base in node.bases:
            is_abstract_base = False

            if NAME.covers(base):
                # Allow inheritance from abstract base classes and common patterns
                allowed_bases = {
                    # Abstract bases
                    "ABC",
                    "Protocol",
                    # Exception hierarchy (standard pattern)
                    "Exception",
                    "BaseException",
                    "ValueError",
                    "TypeError",
                    "RuntimeError",
                    "AttributeError",
                    "KeyError",
                    "IndexError",
                    "ImportError",
                    "OSError",
                    # Standard library abstract bases
                    "Enum",
                    "IntEnum",
                    "Flag",
                    "IntFlag",
                    # Generic object (unavoidable in Python)
                    "object",
                }
                is_abstract_base = base.id in allowed_bases

            elif ATTRIBUTE.covers(base):
                # Check for module.AbstractBase patterns
                if base.attr in {"Protocol", "ABC"}:
                    is_abstract_base = True
                elif NAME.covers(base.value) and base.value.id in {
                    "abc",
                    "typing",
                    "collections",
                    "enum",
                }:
                    is_abstract_base = True
                # Check for imported ABC/Protocol
                elif NAME.covers(base.value) and base.attr in {
                    "ABC",
                    "abstractmethod",
                    "Protocol",
                }:
                    is_abstract_base = True

            # A contract declared in this same file is subtyping, not inheritance
            if not is_abstract_base:
                is_abstract_base = self._is_contract_in_tree(
                    self._base_name(base), tree
                )

            # If not an abstract base, it's implementation inheritance
            if not is_abstract_base:
                return violation(node, ErrorCodes.EO014.format(name=node.name))

        return []
