"""No implementation inheritance principle checker for Elegant Objects violations."""

import ast
from typing import final

from .base import EO014, REPORT, Instance, Principle, Source, Violations

ATTRIBUTE = Instance(ast.Attribute)
CLASS_DEF = Instance(ast.ClassDef)
IMPORT: Instance[ast.Import | ast.ImportFrom] = Instance((ast.Import, ast.ImportFrom))
NAME = Instance(ast.Name)


@final
class NoImplementationInheritance(Principle):
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

    def _origin_name(self, base: ast.expr) -> str:
        """Resolve the leftmost name a base class expression starts from."""
        if NAME.covers(base):
            return base.id
        if ATTRIBUTE.covers(base):
            return self._origin_name(base.value)
        return ""

    def _is_imported(self, name: str, tree: ast.AST | None) -> bool:
        """Check if a name enters this file through an import statement."""
        if not name or not tree:
            return False
        for node in ast.walk(tree):
            if IMPORT.covers(node):
                for alias in node.names:
                    if (alias.asname or alias.name.split(".")[0]) == name:
                        return True
        return False

    def _is_declared_here(self, name: str, tree: ast.AST | None) -> bool:
        """Check if a class of that name is defined in this file."""
        if not name or not tree:
            return False
        return any(
            CLASS_DEF.covers(node) and node.name == name for node in ast.walk(tree)
        )

    def _is_opaque(self, base: ast.expr, tree: ast.AST | None) -> bool:
        """Check if a base comes from another module and cannot be read here.

        A Protocol is commonly declared in one module and implemented in
        another, and flake8 hands this plugin a single file at a time. With
        the definition out of sight there is no evidence either way, so such
        a base is left alone rather than accused.
        """
        return self._is_imported(
            self._origin_name(base), tree
        ) and not self._is_declared_here(self._base_name(base), tree)

    def _is_abstract_base(self, base: ast.expr) -> bool:
        """Check if a base class expression names an abstract base."""
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
            return base.id in allowed_bases

        if ATTRIBUTE.covers(base):
            # Check for module.AbstractBase patterns
            if base.attr in {"Protocol", "ABC"}:
                return True
            if NAME.covers(base.value) and base.value.id in {
                "abc",
                "typing",
                "collections",
                "enum",
            }:
                return True
            # Check for imported ABC/Protocol
            if NAME.covers(base.value) and base.attr in {
                "ABC",
                "abstractmethod",
                "Protocol",
            }:
                return True

        return False

    def _check_implementation_inheritance(
        self, node: ast.ClassDef, tree: ast.AST | None
    ) -> Violations:
        """Check for implementation inheritance violations."""
        for base in node.bases:
            if self._is_abstract_base(base):
                continue
            # A contract declared in this same file is subtyping, not inheritance
            if self._is_contract_in_tree(self._base_name(base), tree):
                continue
            # A base from a sibling module is unreadable here, so unjudgeable
            if self._is_opaque(base, tree):
                continue
            # If not an abstract base, it's implementation inheritance
            return REPORT.of(node, EO014.format(name=node.name))

        return []
