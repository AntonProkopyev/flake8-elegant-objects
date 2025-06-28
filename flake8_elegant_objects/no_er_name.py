"""Naming violations checker for Elegant Objects principles."""

import ast
import re
from typing import ClassVar

from .base import ErrorCodes, Source, Violations, is_method, violation


class NoErName:
    """Checks for naming violations in classes, methods, variables, and functions."""

    # Hall of shame: common -er suffixes (from elegantobjects.org)
    ER_SUFFIXES: ClassVar[set[str]] = {
        "accumulator",
        "adapter",
        "aggregator",
        "analyzer",
        "broker",
        "builder",
        "calculator",
        "checker",
        "collector",
        "compiler",
        "compressor",
        "consumer",
        "controller",
        "converter",
        "coordinator",
        "creator",
        "decoder",
        "decompressor",
        "deserializer",
        "dispatcher",
        "displayer",
        "encoder",
        "evaluator",
        "executor",
        "exporter",
        "factory",
        "fetcher",
        "filter",
        "finder",
        "formatter",
        "generator",
        "handler",
        "helper",
        "importer",
        "interpreter",
        "joiner",
        "listener",
        "loader",
        "manager",
        "mediator",
        "merger",
        "monitor",
        "observer",
        "orchestrator",
        "organizer",
        "parser",
        "printer",
        "processor",
        "producer",
        "provider",
        "reader",
        "renderer",
        "reporter",
        "router",
        "runner",
        "saver",
        "scanner",
        "scheduler",
        "serializer",
        "sorter",
        "splitter",
        "supplier",
        "synchronizer",
        "tracker",
        "transformer",
        "validator",
        "worker",
        "wrapper",
        "writer",
    }

    # Common procedural verbs that should be nouns
    PROCEDURAL_VERBS: ClassVar[set[str]] = {
        "accumulate",
        "add",
        "aggregate",
        "analyze",
        "append",
        "build",
        "calculate",
        "change",
        "check",
        "clean",
        "clear",
        "close",
        "collect",
        "compile",
        "compress",
        "control",
        "convert",
        "create",
        "decode",
        "decompress",
        "delete",
        "deserialize",
        "dispatch",
        "display",
        "do",
        "encode",
        "evaluate",
        "execute",
        "export",
        "fetch",
        "filter",
        "find",
        "format",
        "generate",
        "get",
        "handle",
        "hide",
        "import",
        "insert",
        "interpret",
        "join",
        "load",
        "manage",
        "merge",
        "modify",
        "open",
        "organize",
        "parse",
        "pause",
        "prepend",
        "print",
        "process",
        "put",
        "read",
        "receive",
        "refresh",
        "remove",
        "render",
        "reset",
        "resume",
        "retrieve",
        "route",
        "run",
        "save",
        "schedule",
        "search",
        "send",
        "serialize",
        "set",
        "show",
        "sort",
        "split",
        "start",
        "stop",
        "store",
        "transform",
        "transmit",
        "update",
        "validate",
        "write",
    }

    # Allowed exceptions (common patterns that are OK)
    ALLOWED_EXCEPTIONS: ClassVar[set[str]] = {
        "buffer",
        "character",
        "cluster",
        "container",
        "counter",
        "error",
        "footer",
        "header",
        "identifier",
        "number",
        "order",
        "owner",
        "parameter",
        "pointer",
        "register",
        "server",
        "timer",
        "user",
    }

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
        if name in self.ALLOWED_EXCEPTIONS:
            return []

        # Check for -er suffixes (the hall of shame)
        for suffix in self.ER_SUFFIXES:
            if name.endswith(suffix):
                return violation(node, ErrorCodes.EO001.format(name=node.name))

        # Check for procedural patterns in compound names
        if self._contains_procedural_pattern(name):
            return violation(node, ErrorCodes.EO001.format(name=node.name))

        return []

    def _check_function_name(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> Violations:
        """Check if function/method name violates -er principle."""
        # Skip special methods (__init__, __str__, etc.)
        if node.name.startswith("_"):
            return []

        # Skip common property patterns
        if node.name in {"property", "getter", "setter"}:
            return []

        # Check for procedural verbs
        if self._starts_with_procedural_verb(node.name):
            # Determine if it's a method or standalone function
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
        if name in self.ALLOWED_EXCEPTIONS:
            return []

        # Check for -er suffixes
        for suffix in self.ER_SUFFIXES:
            if name.endswith(suffix):
                return violation(node, ErrorCodes.EO003.format(name=node.id))

        # Check for procedural verbs as variable names
        if self._starts_with_procedural_verb(name):
            return violation(node, ErrorCodes.EO003.format(name=node.id))

        return []

    def _contains_procedural_pattern(self, name: str) -> bool:
        """Check if name contains procedural patterns."""
        # Split camelCase/snake_case into words
        words = re.findall(r"[a-z]+", name)

        # Check if any word is a procedural verb
        return any(word in self.PROCEDURAL_VERBS for word in words)

    def _starts_with_procedural_verb(self, name: str) -> bool:
        """Check if name starts with a procedural verb."""
        # Split camelCase/snake_case and check first word
        # First split on underscores, then on camelCase boundaries
        words: list[str] = []
        for part in name.split("_"):
            # Split camelCase: insert space before uppercase letters
            camel_split = re.sub(r"([a-z])([A-Z])", r"\1 \2", part)
            words.extend(word.lower() for word in camel_split.split() if word)

        if not words:
            return False

        first_word = words[0]
        return first_word in self.PROCEDURAL_VERBS
