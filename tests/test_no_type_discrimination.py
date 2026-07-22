"""Unit tests for no type discrimination principle (NoTypeDiscrimination)."""

import ast

from flake8_elegant_objects.base import Source
from flake8_elegant_objects.no_type_discrimination import NoTypeDiscrimination


class TestNoTypeDiscrimination:
    """Test cases for type discrimination violations detection."""

    def _check_code(self, code: str) -> list[str]:
        """Helper to check code and return violation messages."""
        tree = ast.parse(code)
        checker = NoTypeDiscrimination()
        violations = []

        def visit(node: ast.AST, current_class: ast.ClassDef | None = None) -> None:
            if isinstance(node, ast.ClassDef):
                current_class = node
            source = Source(node, current_class, tree)
            violations.extend(checker.check(source))
            for child in ast.iter_child_nodes(node):
                visit(child, current_class)

        visit(tree)
        return [v.message for v in violations]

    def test_isinstance_violation(self) -> None:
        """Test detection of isinstance usage."""
        code = """
def check_data(data):
    if isinstance(data, str):
        return True
"""
        violations = self._check_code(code)
        assert len(violations) == 1
        assert "EO010" in violations[0]

    def test_reflection_violation(self) -> None:
        """Test detection of reflection usage."""
        code = """
def inspect_object(obj):
    return hasattr(obj, 'attribute')
"""
        violations = self._check_code(code)
        assert len(violations) == 1
        assert "EO010" in violations[0]

    def test_type_function_violation(self) -> None:
        """Test detection of type() function usage."""
        code = """
def get_type(obj):
    return type(obj)
"""
        violations = self._check_code(code)
        assert len(violations) == 1
        assert "EO010" in violations[0]

    def test_dotted_calls_violation(self) -> None:
        """Test detection of casting and introspection reached through a module."""
        code = """
import builtins
import typing

def check(value):
    typing.cast(int, value)
    return builtins.isinstance(value, int)
"""
        violations = self._check_code(code)
        assert len(violations) == 2
        assert all("EO010" in v for v in violations)

    def test_class_attribute_violation(self) -> None:
        """Test detection of type introspection via __class__."""
        code = """
def check(value):
    return value.__class__ is int
"""
        violations = self._check_code(code)
        assert len(violations) == 1
        assert "EO010" in violations[0]

    def test_match_class_pattern_violation(self) -> None:
        """Test detection of type discrimination through match statements."""
        code = """
def check(value):
    match value:
        case int():
            return 1
        case str():
            return 2
"""
        violations = self._check_code(code)
        assert len(violations) == 2
        assert all("EO010" in v for v in violations)

    def test_valid_code(self) -> None:
        """Test that code without type discrimination is valid."""
        code = """
def process_data(data):
    return data.upper()
"""
        violations = self._check_code(code)
        assert len(violations) == 0
