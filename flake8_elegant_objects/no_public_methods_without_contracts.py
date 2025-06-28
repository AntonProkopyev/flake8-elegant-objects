"""No public methods without contracts principle checker for Python."""

import ast

from .base import ErrorCodes, Principle, Source, Violations, is_method, violation


class NoPublicMethodsWithoutContracts(Principle):
    """Check that public methods are defined by contracts (Protocol/ABC)."""

    def check(self, source: Source) -> Violations:
        """Check for public methods without contracts."""
        violations: Violations = []

        if not isinstance(source.node, ast.FunctionDef):
            return violations

        if not source.current_class or not is_method(source.node):
            return violations

        if source.node.name.startswith("_"):
            return violations

        if source.node.name.startswith("__") and source.node.name.endswith("__"):
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
            if not base_name:
                continue

            base_class = self._find_class_def(base_name, tree)
            if not base_class:
                if self._is_protocol_or_abc(base_name, tree):
                    return True
                continue

            if self._has_method(base_class, method_name):
                if self._is_protocol_or_abc(base_name, tree):
                    return True

        return False

    def _get_base_name(self, base: ast.expr) -> str | None:
        """Extract base class name from AST node."""
        if isinstance(base, ast.Name):
            return base.id
        elif isinstance(base, ast.Attribute):
            return base.attr
        return None

    def _is_protocol_or_abc(self, class_name: str, tree: ast.AST | None) -> bool:
        """Check if a class is a Protocol or ABC."""
        if class_name in {"Protocol", "ABC", "ABCMeta"}:
            return True

        if class_name.endswith("Protocol") or class_name.endswith("ABC"):
            return True

        if tree:
            class_def = self._find_class_def(class_name, tree)
            if class_def:
                for base in class_def.bases:
                    base_name = self._get_base_name(base)
                    if base_name and self._is_protocol_or_abc(base_name, None):
                        return True

        return False

    def _find_class_def(
        self, class_name: str, tree: ast.AST | None
    ) -> ast.ClassDef | None:
        """Find class definition in the AST tree."""
        if not tree:
            return None

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                return node

        return None

    def _has_method(self, class_node: ast.ClassDef, method_name: str) -> bool:
        """Check if class has a method with given name."""
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                if node.name == method_name:
                    return True
        return False
