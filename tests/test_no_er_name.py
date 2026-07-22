"""Unit tests for naming principle (NoErNamePrinciple)."""

import ast

from flake8_elegant_objects.base import Source
from flake8_elegant_objects.no_er_name import NoErName


class TestNamingPrinciple:
    """Test cases for naming violations detection."""

    def _check_code(self, code: str) -> list[str]:
        """Helper to check code and return violation messages."""
        tree = ast.parse(code)
        checker = NoErName()
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

    def test_er_class_name_violation(self) -> None:
        """Test detection of -er class names."""
        code = """
class Manager:
    pass

class Controller:
    pass

class Helper:
    pass
"""
        violations = self._check_code(code)
        assert len(violations) == 3
        assert any("Manager" in v and "EO001" in v for v in violations)
        assert any("Controller" in v and "EO001" in v for v in violations)
        assert any("Helper" in v and "EO001" in v for v in violations)

    def test_procedural_function_name_violation(self) -> None:
        """Test detection of procedural function names."""
        code = """
def parser():
    pass

def request_handler():
    pass

def validator():
    pass
"""
        violations = self._check_code(code)
        assert len(violations) == 3
        assert any("parser" in v and "EO004" in v for v in violations)
        assert any("request_handler" in v and "EO004" in v for v in violations)
        assert any("validator" in v and "EO004" in v for v in violations)

    def test_er_method_name_violation(self) -> None:
        """Test detection of -er method names."""
        code = """
class Text:
    def parser(self):
        pass

    def formatter(self):
        pass
"""
        violations = self._check_code(code)
        method_violations = [v for v in violations if "EO002" in v]
        assert len(method_violations) == 2
        assert any("parser" in v for v in method_violations)
        assert any("formatter" in v for v in method_violations)

    def test_verb_method_names_are_valid(self) -> None:
        """Test that verbs are allowed, since methods are nouns or verbs."""
        code = """
class Text:
    def print(self):
        pass

    def save(self):
        pass

    def analyze(self):
        pass
"""
        violations = self._check_code(code)
        assert len(violations) == 0

    def test_procedural_variable_name_violation(self) -> None:
        """Test detection of procedural variable names."""
        code = """
manager = DataManager()
processor = DataProcessor()
handler = RequestHandler()
"""
        violations = self._check_code(code)
        assert len(violations) == 3
        assert any("manager" in v and "EO003" in v for v in violations)
        assert any("processor" in v and "EO003" in v for v in violations)
        assert any("handler" in v and "EO003" in v for v in violations)

    def test_allowed_exceptions(self) -> None:
        """Test that allowed exceptions don't trigger violations."""
        code = """
class User:
    pass

class Order:
    pass

class Buffer:
    pass

def _private_method():
    pass

def __special_method__(self):
    pass

SERVER = "localhost"
"""
        violations = self._check_code(code)
        # Should not have violations for User, Order, Buffer, or private methods
        assert len(violations) == 0

    def test_compound_names_with_er_suffixes(self) -> None:
        """Test detection of compound names with -er suffixes."""
        code = """
class UserManager:
    pass

class DataProcessor:
    pass

class RequestHandler:
    pass
"""
        violations = self._check_code(code)
        assert len(violations) == 3
        assert any("UserManager" in v and "EO001" in v for v in violations)
        assert any("DataProcessor" in v and "EO001" in v for v in violations)
        assert any("RequestHandler" in v and "EO001" in v for v in violations)

    def test_er_suffix_is_open_ended(self) -> None:
        """Test that the principle covers -er and -or names beyond a fixed list."""
        code = """
class Iterator:
    pass

class Visitor:
    pass

class Simulator:
    pass

class Handlers:
    pass
"""
        violations = self._check_code(code)
        assert len(violations) == 4
        assert all("EO001" in v for v in violations)

    def test_ordinary_nouns_are_allowed(self) -> None:
        """Test that ordinary nouns ending in -er survive, alone and compounded."""
        code = """
class ImmutableUser:
    pass

class TaskCounter:
    pass

class HttpHeader:
    pass

class OrderNumber:
    pass
"""
        violations = self._check_code(code)
        assert len(violations) == 0

    def test_camel_case_er_names(self) -> None:
        """Test detection of camelCase -er names."""
        code = """
def dataParser():
    pass

def requestHandler():
    pass

class DataProcessor:
    def textFormatter(self):
        pass
"""
        violations = self._check_code(code)
        assert len(violations) >= 3
        assert any("dataParser" in v and "EO004" in v for v in violations)
        assert any("requestHandler" in v and "EO004" in v for v in violations)
        assert any("textFormatter" in v and "EO002" in v for v in violations)
