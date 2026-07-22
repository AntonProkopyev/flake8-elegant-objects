"""Naming violations checker for Elegant Objects principles."""

import ast
import re
from typing import ClassVar, final

from .base import (
    EO001,
    EO002,
    EO003,
    EO004,
    METHOD,
    REPORT,
    Instance,
    Principle,
    Source,
    Violations,
)

ANN_ASSIGN = Instance(ast.AnnAssign)
ASSIGN = Instance(ast.Assign)
CLASS_DEF = Instance(ast.ClassDef)
FUNCTION: Instance[ast.FunctionDef | ast.AsyncFunctionDef] = Instance((
    ast.FunctionDef,
    ast.AsyncFunctionDef,
))
NAME = Instance(ast.Name)


@final
class NoErName(Principle):
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
        word = self._last_word(name)
        if word in self.ALLOWED_EXCEPTIONS:
            return True
        # The plural of an ordinary noun is an ordinary noun
        return word.endswith("s") and word[:-1] in self.ALLOWED_EXCEPTIONS

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

        if CLASS_DEF.covers(node):
            violations.extend(self._check_class_name(node))
        elif FUNCTION.covers(node):
            violations.extend(self._check_function_name(node))
        elif ASSIGN.covers(node):
            violations.extend(self._check_variable_assignment(node))
        elif ANN_ASSIGN.covers(node):
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
                return REPORT.of(node, EO001.format(name=node.name))

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
                error_code = EO002 if METHOD.covers(node) else EO004
                return REPORT.of(node, error_code.format(name=node.name))

        return []

    def _check_variable_assignment(self, node: ast.Assign) -> Violations:
        """Check variable names in assignments."""
        violations = []
        for target in node.targets:
            if NAME.covers(target):
                violations.extend(self._check_variable_name(target))
        return violations

    def _check_annotated_assignment(self, node: ast.AnnAssign) -> Violations:
        """Check variable names in annotated assignments."""
        if NAME.covers(node.target):
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
                return REPORT.of(node, EO003.format(name=node.id))

        return []
