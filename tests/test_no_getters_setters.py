"""Unit tests for NoGettersSetters principle."""

import ast

from flake8_elegant_objects.base import Source
from flake8_elegant_objects.no_getters_setters import NoGettersSetters


class TestNoGettersSettersPrinciple:
    """Test cases for getter/setter violations detection."""

    def _check_code(self, code: str) -> list[str]:
        """Helper to check code and return violation messages."""
        tree = ast.parse(code)
        checker = NoGettersSetters()
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

    def test_getter_setter_violation(self) -> None:
        """Test detection of getter/setter methods."""
        code = """
class User:
    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name

    def getName(self):
        return self.name

    def setEmail(self, email):
        self.email = email
"""
        violations = self._check_code(code)
        assert len(violations) == 4
        assert all("EO007" in v for v in violations)
        assert any("get_name" in v for v in violations)
        assert any("set_name" in v for v in violations)
        assert any("getName" in v for v in violations)
        assert any("setEmail" in v for v in violations)

    def test_simple_get_set_methods_violation(self) -> None:
        """Test detection of simple get/set methods."""
        code = """
class Data:
    def get(self):
        return self.value

    def set(self, value):
        self.value = value
"""
        violations = self._check_code(code)
        assert len(violations) == 2
        assert all("EO007" in v for v in violations)
        assert any("get" in v for v in violations)
        assert any("set" in v for v in violations)

    def test_camel_case_getters_setters_violation(self) -> None:
        """Test detection of camelCase getter/setter patterns."""
        code = """
class Person:
    def getFirstName(self):
        return self.first_name

    def setFirstName(self, name):
        self.first_name = name

    def getAge(self):
        return self.age

    def setAge(self, age):
        self.age = age
"""
        violations = self._check_code(code)
        assert len(violations) == 4
        assert all("EO007" in v for v in violations)

    def test_private_methods_ignored(self) -> None:
        """Test that private methods starting with _ are ignored."""
        code = """
class User:
    def _get_internal_data(self):
        return self._data

    def _set_internal_state(self, state):
        self._state = state

    def __get_private(self):
        return self.__private
"""
        violations = self._check_code(code)
        assert len(violations) == 0

    def test_valid_method_names(self) -> None:
        """Test that valid method names don't trigger violations."""
        code = """
class User:
    def name(self):
        return self._name

    def email(self):
        return self._email

    def update_profile(self, data):
        self.profile = data

    def process_data(self):
        pass

    def calculate_total(self):
        return sum(self.items)
"""
        violations = self._check_code(code)
        # Note: update_profile, process_data, calculate_total might trigger
        # other violations (naming) but not getter/setter violations
        getter_setter_violations = [v for v in violations if "EO007" in v]
        assert len(getter_setter_violations) == 0

    def test_property_decorators_do_not_excuse_getters(self) -> None:
        """Test that @property does not turn a getter into something else."""
        code = """
class User:
    @property
    def get_name(self):
        return self._name

    @property
    def getName(self):
        return self._name
"""
        violations = self._check_code(code)
        getter_setter_violations = [v for v in violations if "EO007" in v]
        assert len(getter_setter_violations) == 2

    def test_functions_not_methods_ignored(self) -> None:
        """Test that standalone functions are ignored by this principle."""
        code = """
def get_user_data():
    return data

def set_global_config(config):
    global_config = config

def getDataFromAPI():
    return api_data
"""
        violations = self._check_code(code)
        # These might trigger naming violations but not getter/setter violations
        getter_setter_violations = [v for v in violations if "EO007" in v]
        assert len(getter_setter_violations) == 0

    def test_mixed_valid_invalid_methods(self) -> None:
        """Test class with mix of valid and invalid method names."""
        code = """
class DataProcessor:
    def name(self):  # Valid
        return self._name

    def get_data(self):  # Invalid - getter
        return self._data

    def process(self):  # Valid
        pass

    def set_config(self, config):  # Invalid - setter
        self._config = config

    def _get_internal(self):  # Valid - private
        return self._internal
"""
        violations = self._check_code(code)
        getter_setter_violations = [v for v in violations if "EO007" in v]
        assert len(getter_setter_violations) == 2
        assert any("get_data" in v for v in getter_setter_violations)
        assert any("set_config" in v for v in getter_setter_violations)

    def test_property_getter_violation(self) -> None:
        """Test detection of getters written as properties."""
        code = """
class User:
    @property
    def name(self):
        return self._name
"""
        violations = self._check_code(code)
        assert len(violations) == 1
        assert "EO007" in violations[0]

    def test_property_with_behaviour_is_valid(self) -> None:
        """Test that a property doing real work is not a getter."""
        code = """
class User:
    @property
    def greeting(self):
        return "hello, " + self._name.upper()
"""
        violations = self._check_code(code)
        assert len(violations) == 0

    def test_property_setter_violation(self) -> None:
        """Test detection of setters written as property setters."""
        code = """
class User:
    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value
"""
        violations = self._check_code(code)
        # The property getter and its setter are both violations
        assert len(violations) == 2
        assert all("EO007" in v for v in violations)
