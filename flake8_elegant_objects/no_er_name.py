"""Naming violations checker for Elegant Objects principles."""

import ast
import re
from typing import ClassVar

from .base import ErrorCodes, Source, Violations, is_method, violation


class NoErName:
    """Checks for naming violations in classes, methods, variables, and functions."""

    # The principle names readers, parsers, controllers, sorters "and so on",
    # so the suffix is matched, not a closed list of words.
    ER_SUFFIXES: ClassVar[tuple[str, ...]] = ("ers", "ors", "er", "or")

    # Allowed exceptions (common patterns that are OK)
    ALLOWED_EXCEPTIONS: ClassVar[set[str]] = {
        "anchor",
        "author",
        "behavior",
        "border",
        "buffer",
        "center",
        "chapter",
        "character",
        "cluster",
        "color",
        "container",
        "corner",
        "counter",
        "cover",
        "cursor",
        "delimiter",
        "diameter",
        "doctor",
        "elder",
        "error",
        "factor",
        "finger",
        "folder",
        "footer",
        "gender",
        "header",
        "humor",
        "identifier",
        "integer",
        "interior",
        "ladder",
        "layer",
        "ledger",
        "letter",
        "major",
        "matter",
        "member",
        "meter",
        "mirror",
        "neighbor",
        "number",
        "offer",
        "order",
        "other",
        "owner",
        "paper",
        "parameter",
        "partner",
        "perimeter",
        "pointer",
        "power",
        "quarter",
        "register",
        "remainder",
        "river",
        "sector",
        "semester",
        "server",
        "shoulder",
        "silver",
        "sister",
        "spider",
        "summer",
        "super",
        "tenor",
        "terror",
        "theater",
        "timer",
        "tower",
        "trailer",
        "transfer",
        "tumor",
        "upper",
        "user",
        "vapor",
        "vector",
        "water",
        "weather",
        "winter",
        "wonder",
    }

    def _is_allowed(self, name: str) -> bool:
        """Check if a name ends in an ordinary noun rather than an actor."""
        return self._last_word(name) in self.ALLOWED_EXCEPTIONS

    def _last_word(self, name: str) -> str:
        """Split a snake_case or camelCase name and return its final word."""
        words: list[str] = []
        for part in name.split("_"):
            spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", part)
            words.extend(word.lower() for word in spaced.split() if word)
        return words[-1] if words else ""

    def check(self, source: Source) -> Violations:
        """Check source for naming violations."""
        violations = []
        node = source.node

        if isinstance(node, ast.ClassDef):
            violations.extend(self._check_class_name(node))
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            violations.extend(self._check_function_name(node))
        elif isinstance(node, ast.Assign):
            violations.extend(self._check_variable_assignment(node))
        elif isinstance(node, ast.AnnAssign):
            violations.extend(self._check_annotated_assignment(node))

        return violations

    def _check_class_name(self, node: ast.ClassDef) -> Violations:
        """Check if class name violates -er principle."""
        name = node.name.lower()

        # Skip if it's an allowed exception
        if self._is_allowed(node.name):
            return []

        # Check for -er suffixes (the hall of shame)
        for suffix in self.ER_SUFFIXES:
            if name.endswith(suffix):
                return violation(node, ErrorCodes.EO001.format(name=node.name))

        return []

    def _check_function_name(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> Violations:
        """Check if function/method name violates -er principle."""
        # Skip special methods (__init__, __str__, etc.)
        if node.name.startswith("_"):
            return []

        name = node.name.lower()

        # Skip if it's an allowed exception
        if self._is_allowed(node.name):
            return []

        # Check for -er suffixes; verbs are legitimate method names
        for suffix in self.ER_SUFFIXES:
            if name.endswith(suffix):
                error_code = ErrorCodes.EO002 if is_method(node) else ErrorCodes.EO004
                return violation(node, error_code.format(name=node.name))

        return []

    def _check_variable_assignment(self, node: ast.Assign) -> Violations:
        """Check variable names in assignments."""
        violations = []
        for target in node.targets:
            if isinstance(target, ast.Name):
                violations.extend(self._check_variable_name(target))
        return violations

    def _check_annotated_assignment(self, node: ast.AnnAssign) -> Violations:
        """Check variable names in annotated assignments."""
        if isinstance(node.target, ast.Name):
            return self._check_variable_name(node.target)
        return []

    def _check_variable_name(self, node: ast.Name) -> Violations:
        """Check if variable name violates -er principle."""
        # Skip private variables and common patterns
        if node.id.startswith("_") or node.id.isupper():
            return []

        name = node.id.lower()

        # Skip if it's an allowed exception
        if self._is_allowed(node.id):
            return []

        # Check for -er suffixes
        for suffix in self.ER_SUFFIXES:
            if name.endswith(suffix):
                return violation(node, ErrorCodes.EO003.format(name=node.id))

        return []
