"""Unit tests for no implementation inheritance principle (NoImplementationInheritance)."""

import ast

from flake8_elegant_objects.base import Source
from flake8_elegant_objects.no_implementation_inheritance import (
    NoImplementationInheritance,
)


class TestNoImplementationInheritanceContracts:
    """Test cases for contracts declared alongside their implementations."""

    def _check_code(self, code: str) -> list[str]:
        """Helper to check code and return violation messages."""
        tree = ast.parse(code)
        checker = NoImplementationInheritance()
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

    def test_local_protocol_is_subtyping(self) -> None:
        """Test that implementing a Protocol from the same file is allowed."""
        code = """
from typing import Protocol

class Money(Protocol):
    def cents(self) -> int: ...

class Dollars(Money):
    def cents(self) -> int:
        return 100
"""
        violations = self._check_code(code)
        assert len(violations) == 0

    def test_local_concrete_base_is_inheritance(self) -> None:
        """Test that a concrete base from the same file is still a violation."""
        code = """
class Money:
    def cents(self) -> int:
        return 0

class Dollars(Money):
    def cents(self) -> int:
        return 100
"""
        violations = self._check_code(code)
        assert len(violations) == 1
        assert "EO014" in violations[0]


class TestNoImplementationInheritance:
    """Test cases for implementation inheritance violations detection."""

    def _check_code(self, code: str) -> list[str]:
        """Helper to check code and return violation messages."""
        tree = ast.parse(code)
        checker = NoImplementationInheritance()
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

    def test_implementation_inheritance_violation(self) -> None:
        """Test detection of implementation inheritance."""
        code = """
class ConcreteBase:
    def method(self):
        pass

class Child(ConcreteBase):
    pass
"""
        violations = self._check_code(code)
        assert len(violations) == 1
        assert "EO014" in violations[0]

    def test_valid_abstract_inheritance(self) -> None:
        """Test that abstract inheritance is allowed."""
        code = """
from abc import ABC, abstractmethod

class AbstractBase(ABC):
    @abstractmethod
    def method(self):
        pass

class Child(abc.ABC):
    def method(self):
        pass
"""
        violations = self._check_code(code)
        assert len(violations) == 0

    def test_exception_inheritance_valid(self) -> None:
        """Test that exception inheritance is allowed."""
        code = """
class CustomError(ValueError):
    pass
"""
        violations = self._check_code(code)
        assert len(violations) == 0

    def test_object_inheritance_valid(self) -> None:
        """Test that inheriting from object is allowed."""
        code = """
class MyClass(object):
    pass
"""
        violations = self._check_code(code)
        assert len(violations) == 0
