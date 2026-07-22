"""Unit tests for NoNull principle."""

import ast

from flake8_elegant_objects.base import Parents, Source
from flake8_elegant_objects.no_null import NoNull


class TestNoNullPrinciple:
    """Test cases for None usage violations detection."""

    def _check_code(self, code: str) -> list[str]:
        """Helper to check code and return violation messages."""
        tree = ast.parse(code)
        checker = NoNull()
        violations = []
        held: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                held[child] = node
        parents = Parents(held)

        def visit(node: ast.AST) -> None:
            source = Source(node, None, tree, parents)
            violations.extend(checker.check(source))
            for child in ast.iter_child_nodes(node):
                visit(child)

        visit(tree)
        return [v.message for v in violations]

    def test_none_usage_violation(self) -> None:
        """Test detection of None usage."""
        code = """
def get_user():
    return None

value = None

result = None if condition else data
"""
        violations = self._check_code(code)
        assert len(violations) == 3
        assert all("EO005" in v for v in violations)
        assert all("None" in v for v in violations)

    def test_none_in_function_arguments(self) -> None:
        """Test detection of None in function arguments."""
        code = """
def process_data(data=None):
    pass

def handle_request(request, user=None):
    pass
"""
        violations = self._check_code(code)
        assert len(violations) == 2
        assert all("EO005" in v for v in violations)

    def test_none_in_class_attributes(self) -> None:
        """Test detection of None in class attributes."""
        code = """
class User:
    name = None
    email = None

    def __init__(self):
        self.data = None
"""
        violations = self._check_code(code)
        assert len(violations) == 3
        assert all("EO005" in v for v in violations)

    def test_none_in_comparison(self) -> None:
        """Test detection of None in comparisons."""
        code = """
if user is None:
    pass

if data == None:
    pass

result = value is not None
"""
        violations = self._check_code(code)
        assert len(violations) == 3
        assert all("EO005" in v for v in violations)

    def test_none_in_list_comprehension(self) -> None:
        """Test detection of None in list comprehensions."""
        code = """
values = [None for _ in range(10)]
filtered = [x for x in data if x is not None]
"""
        violations = self._check_code(code)
        assert len(violations) == 2
        assert all("EO005" in v for v in violations)

    def test_bare_return_violation(self) -> None:
        """Test detection of a bare return, which yields None implicitly."""
        code = """
def value(flag):
    if flag:
        return 1
    return
"""
        violations = self._check_code(code)
        assert len(violations) == 1
        assert "EO005" in violations[0]

    def test_bare_return_in_command_is_valid(self) -> None:
        """Test that a command returning nothing everywhere is not a null return."""
        code = """
def send(flag):
    if flag:
        return
    print("sent")
"""
        violations = self._check_code(code)
        assert len(violations) == 0

    def test_valid_code_without_none(self) -> None:
        """Test that code without None doesn't trigger violations."""
        code = """
def get_user():
    return []

value = []

def process_data(data=[]):
    pass

class User:
    name = ""
    email = ""
"""
        violations = self._check_code(code)
        assert len(violations) == 0

    def test_none_in_annotations(self) -> None:
        """Test that None in type annotations doesn't trigger violations."""
        code = """
from typing import Optional

def get_user() -> Optional[str]:
    return ""

def process(data: str | None = "") -> None:
    pass
"""
        violations = self._check_code(code)
        # Type annotations with None should not trigger violations
        # Only actual None values should
        assert len(violations) == 0
