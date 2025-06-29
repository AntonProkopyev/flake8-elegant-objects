"""Unit tests for NoConstructorCode principle."""

import ast

from flake8_elegant_objects.base import Source
from flake8_elegant_objects.no_constructor_code import NoConstructorCode


class TestNoConstructorCodePrinciple:
    """Test cases for constructor code violations detection."""

    def _check_code(self, code: str) -> list[str]:
        """Helper to check code and return violation messages."""
        tree = ast.parse(code)
        checker = NoConstructorCode()
        violations = []

        def visit(node: ast.AST) -> None:
            source = Source(node, None, tree)
            violations.extend(checker.check(source))
            for child in ast.iter_child_nodes(node):
                visit(child)

        visit(tree)
        return [v.message for v in violations]

    def test_constructor_with_code_violation(self) -> None:
        """Test detection of code in constructors beyond parameter assignments."""
        code = """
class User:
    def __init__(self, name):
        self.name = name
        self.data = []  # This should trigger violation
        self.processed_name = name.upper()  # This should trigger violation
"""
        violations = self._check_code(code)
        assert len(violations) == 2
        assert all("EO006" in v for v in violations)

    def test_constructor_with_computation_violation(self) -> None:
        """Test detection of computation in constructor assignments."""
        code = """
class Calculator:
    def __init__(self, value):
        self.value = value
        self.squared = value * value  # This should trigger violation
        self.formatted = f"Value: {value}"  # This should trigger violation
"""
        violations = self._check_code(code)
        assert len(violations) == 2
        assert all("EO006" in v for v in violations)

    def test_constructor_with_method_calls_violation(self) -> None:
        """Test detection of method calls in constructors."""
        code = """
class Logger:
    def __init__(self, level):
        self.level = level
        self.setup_logging()  # This should trigger violation
        print("Logger initialized")  # This should trigger violation
"""
        violations = self._check_code(code)
        assert len(violations) == 2
        assert all("EO006" in v for v in violations)

    def test_valid_constructor_simple_assignments(self) -> None:
        """Test that simple parameter assignments don't trigger violations."""
        code = """
class User:
    def __init__(self, name, email, age):
        self.name = name
        self.email = email
        self.age = age
"""
        violations = self._check_code(code)
        assert len(violations) == 0

    def test_valid_constructor_with_pass(self) -> None:
        """Test that constructors with pass statements are valid."""
        code = """
class Empty:
    def __init__(self):
        pass

class Simple:
    def __init__(self, value):
        self.value = value
        pass
"""
        violations = self._check_code(code)
        assert len(violations) == 0

    def test_non_constructor_methods_ignored(self) -> None:
        """Test that non-constructor methods are ignored."""
        code = """
class Processor:
    def process_data(self):
        self.data = []
        self.result = self.compute()
        print("Processing complete")

    def compute(self):
        return sum(self.data)
"""
        violations = self._check_code(code)
        assert len(violations) == 0

    def test_static_method_init_ignored(self) -> None:
        """Test that static methods named __init__ are ignored."""
        code = """
class Factory:
    @staticmethod
    def __init__(data):
        processed = data.upper()
        return processed
"""
        violations = self._check_code(code)
        # This should not trigger constructor code violation since it's static
        constructor_violations = [v for v in violations if "EO006" in v]
        assert len(constructor_violations) == 0

    def test_constructor_with_complex_assignment_violation(self) -> None:
        """Test detection of complex assignments in constructors."""
        code = """
class DataContainer:
    def __init__(self, items):
        self.items = items
        self.count = len(items)  # This should trigger violation
        self.first = items[0] if items else None  # This should trigger violation
"""
        violations = self._check_code(code)
        assert len(violations) == 2
        assert all("EO006" in v for v in violations)

    def test_valid_constructor_with_super_call(self) -> None:
        """Test that super() calls in constructors are allowed."""
        code = """
class Parent:
    def __init__(self, name):
        self.name = name

class Child(Parent):
    def __init__(self, name, age):
        super().__init__(name)
        self.age = age

class GrandChild(Child):
    def __init__(self, name, age, grade):
        super().__init__(name, age)
        self.grade = grade
"""
        violations = self._check_code(code)
        assert len(violations) == 0

    def test_valid_constructor_with_super_call_variations(self) -> None:
        """Test various forms of super() calls are allowed."""
        code = """
class Base:
    def __init__(self):
        pass

class Derived(Base):
    def __init__(self):
        super().__init__()

class DerivedWithArgs(Base):
    def __init__(self, value):
        super().__init__()
        self.value = value

class ExplicitSuper(Base):
    def __init__(self):
        super(ExplicitSuper, self).__init__()
"""
        violations = self._check_code(code)
        assert len(violations) == 0

    def test_constructor_with_super_and_invalid_code(self) -> None:
        """Test that super() calls are allowed but other code violations are still detected."""
        code = """
class Parent:
    def __init__(self, name):
        self.name = name

class Child(Parent):
    def __init__(self, name, items):
        super().__init__(name)
        self.items = items
        self.count = len(items)  # This should trigger violation
        print("Child initialized")  # This should trigger violation
"""
        violations = self._check_code(code)
        assert len(violations) == 2
        assert all("EO006" in v for v in violations)
