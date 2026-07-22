"""Unit tests for suppression comments and the one type check that remains."""

import ast

from flake8_elegant_objects.base import Instance
from flake8_elegant_objects.noqa import Noqa


class TestNoqa:
    """Test cases for noqa handling outside flake8."""

    def test_named_code_is_suppressed(self) -> None:
        """A code named in a noqa comment is suppressed on that line."""
        noqa = Noqa("x = 1\ny = isinstance(x, int)  # noqa: EO010\n")
        assert not noqa.allows(2, "EO010")

    def test_other_codes_survive_a_named_noqa(self) -> None:
        """A noqa naming one code leaves the others reported."""
        noqa = Noqa("x = 1\ny = isinstance(x, int)  # noqa: EO010\n")
        assert noqa.allows(2, "EO005")

    def test_bare_noqa_suppresses_everything(self) -> None:
        """A noqa without codes silences the whole line."""
        noqa = Noqa("value = None  # noqa\n")
        assert not noqa.allows(1, "EO005")

    def test_several_codes_in_one_comment(self) -> None:
        """A noqa may name more than one code."""
        noqa = Noqa("value = None  # noqa: EO005, EO010\n")
        assert not noqa.allows(1, "EO010")

    def test_line_without_comment_is_untouched(self) -> None:
        """A line carrying no comment suppresses nothing."""
        noqa = Noqa("value = None\n")
        assert noqa.allows(1, "EO005")

    def test_line_beyond_the_source_is_untouched(self) -> None:
        """A line number outside the file suppresses nothing."""
        noqa = Noqa("value = None\n")
        assert noqa.allows(99, "EO005")


class TestInstance:
    """Test cases for the single place type discrimination is allowed."""

    def test_covers_a_matching_node(self) -> None:
        """A node of the given type is covered."""
        assert Instance(ast.ClassDef).covers(ast.parse("class A: pass").body[0])

    def test_rejects_a_different_node(self) -> None:
        """A node of another type is not covered."""
        assert not Instance(ast.ClassDef).covers(ast.parse("x = 1").body[0])

    def test_covers_a_union_of_types(self) -> None:
        """A union of types is covered by any of its members."""
        kinds = Instance(ast.FunctionDef | ast.AsyncFunctionDef)
        assert kinds.covers(ast.parse("def f(): pass").body[0])

    def test_covers_a_tuple_of_types(self) -> None:
        """A tuple of types is covered by any of its members."""
        kinds = Instance((ast.Assign, ast.AnnAssign))
        assert kinds.covers(ast.parse("x = 1").body[0])
