"""Unit tests for no ORM principle (NoOrm)."""

import ast

from flake8_elegant_objects.base import Source
from flake8_elegant_objects.no_orm import NoOrm


class TestNoOrmCollections:
    """Test cases for collection methods that share a name with ORM calls."""

    def _check_code(self, code: str) -> list[str]:
        """Helper to check code and return violation messages."""
        tree = ast.parse(code)
        checker = NoOrm()
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

    def test_collection_methods_are_not_orm(self) -> None:
        """Test that ordinary collection and string methods are left alone."""
        code = """
def collect(names, mapping, parts, items):
    names.update(mapping)
    mapping.get("key")
    ", ".join(parts)
    items.insert(0, "first")
    "text".find("x")
"""
        violations = self._check_code(code)
        assert len(violations) == 0


class TestNoOrm:
    """Test cases for ORM pattern violations detection."""

    def _check_code(self, code: str) -> list[str]:
        """Helper to check code and return violation messages."""
        tree = ast.parse(code)
        checker = NoOrm()
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

    def test_orm_save_violation(self) -> None:
        """Test detection of ORM save pattern."""
        code = """
user = User()
user.save()
"""
        violations = self._check_code(code)
        assert len(violations) == 1
        assert "EO013" in violations[0]

    def test_orm_query_violation(self) -> None:
        """Test detection of ORM query pattern."""
        code = """
users = User.objects.filter(name="John")
"""
        violations = self._check_code(code)
        assert len(violations) == 1
        assert "EO013" in violations[0]

    def test_built_in_methods_valid(self) -> None:
        """Test that built-in methods are valid."""
        code = """
my_list = list([1, 2, 3])
length = len(my_list)
"""
        violations = self._check_code(code)
        assert len(violations) == 0

    def test_string_methods_valid(self) -> None:
        """Test that string methods are valid."""
        code = """
text = "hello"
result = text.upper()
"""
        violations = self._check_code(code)
        assert len(violations) == 0
