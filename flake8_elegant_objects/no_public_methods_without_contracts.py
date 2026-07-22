"""No public methods without contracts principle checker for Python."""

import ast
from typing import final

from .base import (
    ErrorCodes,
    Instance,
    Principle,
    Source,
    Violations,
    is_method,
    violation,
)

ATTRIBUTE = Instance(ast.Attribute)
CLASS_DEF = Instance(ast.ClassDef)
FUNCTION: Instance[ast.FunctionDef | ast.AsyncFunctionDef] = Instance((
    ast.FunctionDef,
    ast.AsyncFunctionDef,
))
IMPORT: Instance[ast.Import | ast.ImportFrom] = Instance((ast.Import, ast.ImportFrom))
NAME = Instance(ast.Name)


@final
class NoPublicMethodsWithoutContracts(Principle):
    """Check that public methods are defined by contracts (Protocol/ABC)."""

    def check(self, source: Source) -> Violations:
        """Check for public methods without contracts."""
        violations: Violations = []

        if not FUNCTION.covers(source.node):
            return violations

        if not source.current_class or not is_method(source.node):
            return violations

        if source.node.name.startswith("_"):
            return violations

        if self._is_test(source.node, source.current_class):
            return violations

        if self._has_opaque_base(source.current_class, source.tree):
            return violations

        if self._class_has_contract(source.current_class, source.tree):
            if not self._method_from_contract(
                source.node.name, source.current_class, source.tree
            ):
                violations.extend(
                    violation(
                        source.node, ErrorCodes.EO011.format(name=source.node.name)
                    )
                )
        else:
            violations.extend(
                violation(source.node, ErrorCodes.EO011.format(name=source.node.name))
            )

        return violations

    def _is_test(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        class_node: ast.ClassDef,
    ) -> bool:
        """Check if a method belongs to a test suite rather than a contract."""
        return node.name.startswith("test") or class_node.name.startswith("Test")

    def _class_has_contract(
        self, class_node: ast.ClassDef, tree: ast.AST | None
    ) -> bool:
        """Check if class implements any Protocol or ABC."""
        if not class_node.bases:
            return False

        for base in class_node.bases:
            base_name = self._get_base_name(base)
            if not base_name:
                continue

            if self._is_protocol_or_abc(base_name, tree):
                return True

        return False

    def _method_from_contract(
        self, method_name: str, class_node: ast.ClassDef, tree: ast.AST | None
    ) -> bool:
        """Check if method is defined in any of the class's contracts."""
        for base in class_node.bases:
            base_name = self._get_base_name(base)
            if not base_name or not self._is_protocol_or_abc(base_name, tree):
                continue

            contracts = self._class_defs(base_name, tree)
            if not contracts:
                return True

            if any(self._has_method(each, method_name) for each in contracts):
                return True

        return False

    def _has_opaque_base(self, class_node: ast.ClassDef, tree: ast.AST | None) -> bool:
        """Check if any base of the class comes from beyond this file.

        A Protocol is commonly declared in one module and implemented in
        another, and flake8 hands this plugin a single file at a time. With
        the definition out of sight there is no evidence either way, so the
        methods of such a class are left alone rather than accused.
        """
        return any(self._is_opaque(base, tree) for base in class_node.bases)

    def _is_opaque(self, base: ast.expr, tree: ast.AST | None) -> bool:
        """Check if a base is imported and has no definition in this file."""
        return self._is_imported(
            self._origin_name(base), tree
        ) and not self._class_defs(self._get_base_name(base), tree)

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

    def _get_base_name(self, base: ast.expr) -> str:
        """Extract base class name from AST node."""
        if NAME.covers(base):
            return base.id
        elif ATTRIBUTE.covers(base):
            return base.attr
        return ""

    def _origin_name(self, base: ast.expr) -> str:
        """Extract the leftmost name a base class expression starts from."""
        if NAME.covers(base):
            return base.id
        elif ATTRIBUTE.covers(base):
            return self._origin_name(base.value)
        return ""

    def _is_contract_name(self, class_name: str) -> bool:
        """Check if a name is that of a Protocol or an ABC by itself."""
        return class_name in {"Protocol", "ABC", "ABCMeta"} or class_name.endswith((
            "Protocol",
            "ABC",
        ))

    def _is_protocol_or_abc(self, class_name: str, tree: ast.AST | None) -> bool:
        """Check if a class is a Protocol or ABC."""
        if self._is_contract_name(class_name):
            return True

        for class_def in self._class_defs(class_name, tree):
            for base in class_def.bases:
                if self._is_contract_name(self._get_base_name(base)):
                    return True

        return False

    def _class_defs(self, class_name: str, tree: ast.AST | None) -> list[ast.ClassDef]:
        """Find the class definitions of that name in the AST tree."""
        if not class_name or not tree:
            return []

        return [
            node
            for node in ast.walk(tree)
            if CLASS_DEF.covers(node) and node.name == class_name
        ]

    def _has_method(self, class_node: ast.ClassDef, method_name: str) -> bool:
        """Check if class has a method with given name."""
        for node in class_node.body:
            if FUNCTION.covers(node):
                if node.name == method_name:
                    return True
        return False
