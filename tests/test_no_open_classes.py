"""Unit tests for NoOpenClasses principle."""

import ast

from flake8_elegant_objects.base import Source
from flake8_elegant_objects.no_open_classes import NoOpenClasses


class TestNoOpenClasses:
    """Test cases for non-final and overweight class detection."""

    def _check_code(self, code: str) -> list[str]:
        """Helper to check code and return violation messages."""
        tree = ast.parse(code)
        checker = NoOpenClasses()
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

    def test_open_class_violation(self) -> None:
        """Test detection of a class left open for inheritance."""
        code = """
class Money:
    def __init__(self, cents):
        self.cents = cents
"""
        violations = self._check_code(code)
        assert len(violations) == 1
        assert "EO028" in violations[0]

    def test_final_class_valid(self) -> None:
        """Test that a final class is accepted."""
        code = """
from typing import final

@final
class Money:
    def __init__(self, cents):
        self.cents = cents
"""
        violations = self._check_code(code)
        assert len(violations) == 0

    def test_dotted_final_decorator_valid(self) -> None:
        """Test that @typing.final is recognised."""
        code = """
import typing

@typing.final
class Money:
    def __init__(self, cents):
        self.cents = cents
"""
        violations = self._check_code(code)
        assert len(violations) == 0

    def test_protocol_stays_open(self) -> None:
        """Test that contracts are exempt, since they exist to be implemented."""
        code = """
from typing import Protocol

class Money(Protocol):
    def cents(self) -> int: ...
"""
        violations = self._check_code(code)
        assert len(violations) == 0

    def test_abc_stays_open(self) -> None:
        """Test that abstract base classes are exempt."""
        code = """
from abc import ABC

class Money(ABC):
    def cents(self) -> int: ...
"""
        violations = self._check_code(code)
        assert len(violations) == 0

    def test_test_class_is_exempt(self) -> None:
        """Test that test suites are not objects under these rules."""
        code = """
class TestMoney:
    def test_cents(self):
        assert True
"""
        violations = self._check_code(code)
        assert len(violations) == 0

    def test_too_many_attributes_violation(self) -> None:
        """Test detection of a class holding more than four attributes."""
        code = """
from typing import final

@final
class Invoice:
    def __init__(self, number, date, customer, total, tax):
        self.number = number
        self.date = date
        self.customer = customer
        self.total = total
        self.tax = tax
"""
        violations = self._check_code(code)
        assert len(violations) == 1
        assert "EO029" in violations[0]
        assert "holds 5" in violations[0]

    def test_four_attributes_valid(self) -> None:
        """Test that four attributes are within the limit."""
        code = """
from typing import final

@final
class Invoice:
    def __init__(self, number, date, customer, total):
        self.number = number
        self.date = date
        self.customer = customer
        self.total = total
"""
        violations = self._check_code(code)
        assert len(violations) == 0

    def test_class_level_attributes_are_counted(self) -> None:
        """Test that class attributes count towards the limit."""
        code = """
from typing import final

@final
class Invoice:
    kind: str = "invoice"
    currency: str = "EUR"
    rate = 1

    def __init__(self, number, total):
        self.number = number
        self.total = total
"""
        violations = self._check_code(code)
        assert len(violations) == 1
        assert "EO029" in violations[0]
