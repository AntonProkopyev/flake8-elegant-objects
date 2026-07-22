"""Unit tests for no impure tests principle (NoImpureTests)."""

import ast

from flake8_elegant_objects.base import Source
from flake8_elegant_objects.no_impure_tests import NoImpureTests


class TestNoImpureTests:
    """Test cases for impure test method violations detection."""

    def _check_code(self, code: str) -> list[str]:
        """Helper to check code and return violation messages."""
        tree = ast.parse(code)
        checker = NoImpureTests()
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

    def test_valid_unittest_assertion(self) -> None:
        """Test that unittest style assertions are valid."""
        code = """
def test_generates_array_of_correct_size(self):
    self.assertEqual(
        len(ArrayFromRandom(random.Random(42)).to_array(25)),
        25
    )
"""
        violations = self._check_code(code)
        assert len(violations) == 0

    def test_valid_pytest_assertion(self) -> None:
        """Test that pytest style assertions are valid."""
        code = """
def test_valid_user_is_valid(self):
    assert User(name="Alice", age=30).is_valid()
"""
        violations = self._check_code(code)
        assert len(violations) == 0

    def test_valid_pytest_raises(self) -> None:
        """Test that pytest.raises is valid."""
        code = """
def test_invalid_age_raises_exception(self):
    with pytest.raises(ValueError):
        User(name="Alice", age="not a number")
"""
        violations = self._check_code(code)
        assert len(violations) == 0

    def test_multiple_assertions_violation(self) -> None:
        """Test that multiple assertions are detected as violations."""
        code = """
def test_multiple_user_properties(self):
    user = User(name="Alice", age=30)
    self.assertEqual(user.name, "Alice")
    self.assertEqual(user.age, 30)
    self.assertTrue(user.is_valid())
"""
        violations = self._check_code(code)
        assert len(violations) == 2  # 1 assignment + 1 for multiple assertions

    def test_assignment_statement_violation(self) -> None:
        """Test that assignment statements are detected as violations."""
        code = """
def test_array_properties_with_assignment(self):
    arr = ArrayFromRandom(random.Random(42)).to_array(10)
    self.assertEqual(len(arr), 10)
"""
        violations = self._check_code(code)
        assert len(violations) == 1  # Assignment statement violates purity

    def test_loop_with_assertions_violation(self) -> None:
        """Test that loops with assertions are detected as violations."""
        code = """
def test_array_properties_with_loop(self):
    arr = ArrayFromRandom(random.Random(42)).to_array(10)
    self.assertEqual(len(arr), 10)
    for num in arr:
        self.assertIsInstance(num, int)
        self.assertGreaterEqual(num, 0)
"""
        violations = self._check_code(code)
        # Assignment, loop, and an assertion that does not close the method
        assert len(violations) == 3

    def test_no_assertion_violation(self) -> None:
        """Test that methods without assertions are detected as violations."""
        code = """
def test_without_assertion(self):
    user = User(name="Alice", age=30)
    print(f"Created user: {user}")
"""
        violations = self._check_code(code)
        assert len(violations) == 3  # Assignment + print + no assertion

    def test_valid_complex_assertion(self) -> None:
        """Test that complex single assertions are valid."""
        code = """
def test_array_contains_only_integers(self):
    self.assertTrue(
        all(isinstance(n, int)
            for n in ArrayFromRandom(random.Random(42)).to_array(10))
    )
"""
        violations = self._check_code(code)
        assert len(violations) == 0

    def test_valid_assertion_with_expression(self) -> None:
        """Test that assertions with expressions are valid."""
        code = """
def test_negative_age_is_invalid(self):
    assert User(name="Bob", age=-5).is_valid() is False
"""
        violations = self._check_code(code)
        assert len(violations) == 0

    def test_non_test_method_ignored(self) -> None:
        """Test that non-test methods are ignored."""
        code = """
def regular_method(self):
    x = 5
    y = x + 3
    return y
"""
        violations = self._check_code(code)
        assert len(violations) == 0

    def test_test_with_pass_valid(self) -> None:
        """Test that test methods with pass are valid."""
        code = """
def test_example(self):
    pass
"""
        violations = self._check_code(code)
        assert len(violations) == 1  # No assertion is a violation

    def test_assertThat_chain_valid(self) -> None:
        """Test that assertThat chained calls are valid."""
        code = """
def test_assertThat_example(self):
    assertThat(calculate(5, 3)).isEqualTo(8)
"""
        violations = self._check_code(code)
        assert len(violations) == 0

    def test_docstring_is_not_a_statement(self) -> None:
        """Test that a docstring does not count as code in a test."""
        code = '''
def test_cents_are_counted(self):
    """Cents are counted."""
    assert money.cents() == 100
'''
        violations = self._check_code(code)
        assert len(violations) == 0

    def test_assertion_must_be_last(self) -> None:
        """Test that nothing may follow the assertion."""
        code = """
def test_cents_are_counted(self):
    assert money.cents() == 100
    pass
"""
        violations = self._check_code(code)
        assert len(violations) == 1
        assert "last statement" in violations[0]

    def test_assertion_last_is_valid(self) -> None:
        """Test that an assertion closing the method is accepted."""
        code = """
def test_cents_are_counted(self):
    pass
    assert money.cents() == 100
"""
        violations = self._check_code(code)
        assert len(violations) == 0

    def test_two_assertions_report_the_count(self) -> None:
        """Test that the message names the real fault when there are two."""
        code = """
def test_cents_are_counted(self):
    assert money.cents() == 100
    assert money.cents() == 200
"""
        violations = self._check_code(code)
        assert len(violations) == 1
        assert "one assertion" in violations[0]

    def test_call_on_a_call_terminates(self) -> None:
        """A chain headed by a plain call must not loop forever.

        `helper().clear()` walks down to a Name that is not an assertion.
        The walk once had no way out of that case and hung the whole run,
        so this test is here to keep the exit.
        """
        code = """
def test_cents_are_counted(self):
    _loaded_tools().clear()
    assert money.cents() == 100
"""
        violations = self._check_code(code)
        assert len(violations) == 1
