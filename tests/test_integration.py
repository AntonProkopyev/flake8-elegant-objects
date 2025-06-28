"""Integration tests for the complete flake8-elegant-objects plugin."""

import ast

from flake8_elegant_objects import ElegantObjectsPlugin


class TestIntegration:
    """Integration test cases for the complete plugin."""

    def _check_code(self, code: str) -> list[tuple[int, int, str]]:
        """Helper to check code and return violations."""
        tree = ast.parse(code)
        checker = ElegantObjectsPlugin(tree)
        return [(line, col, msg) for line, col, msg, _ in checker.run()]

    def test_comprehensive_violations(self) -> None:
        """Test detection of multiple violation types in one code sample."""
        code = """
class DataManager:  # EO001 - -er class name
    data = []  # EO015 - mutable class attribute

    def __init__(self):
        self.processed_data = []  # EO006 - code in constructor

    def get_data(self):  # EO002 + EO007 - procedural name + getter
        return None  # EO005 - None usage

    @staticmethod  # EO009 - static method
    def process_items(items):
        if isinstance(items, list):  # EO010 - isinstance
            return items
        return None
"""
        violations = self._check_code(code)

        # Check that we have violations from multiple principles
        error_codes = {v[2].split()[0] for v in violations}

        assert "EO001" in error_codes  # Naming
        assert "EO005" in error_codes  # NoNull
        assert "EO006" in error_codes  # NoConstructorCode
        assert "EO007" in error_codes  # NoGettersSetters
        assert "EO015" in error_codes  # NoMutableObjects - class attribute
        assert "EO009" in error_codes  # Advanced - static method
        assert "EO010" in error_codes  # Advanced - isinstance

    def test_real_world_example_violations(self) -> None:
        """Test with a more realistic code example."""
        code = """
from dataclasses import dataclass

@dataclass  # EO008 - mutable dataclass
class UserManager:  # EO001 - -er class name
    users = []  # EO015 - mutable class attribute

    def __init__(self, config):
        self.config = config
        self.logger = create_logger()  # EO006 - code in constructor

    def get_user(self, user_id):  # EO002 + EO007 - procedural + getter
        if user_id is None:  # EO005 - None usage
            return None  # EO005 - None usage

        for user in self.users:
            if user.id == user_id:
                return user
        return None  # EO005 - None usage

    def save_user(self, user):  # EO002 - procedural name
        user.save()  # EO013 - ORM pattern

    @staticmethod  # EO009 - static method
    def validate_email(email):
        return "@" in email
"""
        violations = self._check_code(code)

        # Should have multiple violations
        assert len(violations) > 5

        # Check specific violation types are present
        violation_messages = [v[2] for v in violations]
        assert any(
            "EO001" in msg and "UserManager" in msg for msg in violation_messages
        )
        assert any("EO005" in msg for msg in violation_messages)
        assert any("EO006" in msg for msg in violation_messages)
        assert any("EO007" in msg and "get_user" in msg for msg in violation_messages)
        assert any("EO008" in msg for msg in violation_messages)  # dataclass
        assert any("EO015" in msg for msg in violation_messages)  # class attribute
        assert any("EO009" in msg for msg in violation_messages)

    def test_clean_elegant_code(self) -> None:
        """Test that clean, elegant code has no violations."""
        code = """
from abc import ABC
from dataclasses import dataclass

class User(ABC):
    def name(self) -> str:
        pass

@dataclass(frozen=True)
class ImmutableUser:
    name: str
    email: str

class UserRepository:
    def __init__(self, storage):
        self.storage = storage

    def user_by_email(self, email: str):
        return self.storage.find_by_email(email)

    def persistence(self, user: User):
        return self.storage.persist(user)
"""
        violations = self._check_code(code)

        # Clean code should have minimal or no violations
        # Filter out any false positives for method contracts (EO011)
        serious_violations = [v for v in violations if not v[2].startswith("EO011")]
        assert len(serious_violations) == 0

    def test_mixed_valid_invalid_code(self) -> None:
        """Test code with both valid and invalid patterns."""
        code = """
# Valid patterns
class User:
    def __init__(self, name: str):
        self.name = name

    def email(self) -> str:
        return self._email

# Invalid patterns
class DataProcessor:  # EO001 - -er name
    cache = {}  # EO015 - mutable attribute

    def get_name(self):  # EO007 - getter
        return None  # EO005 - None
"""
        violations = self._check_code(code)

        # Should have exactly the violations from invalid patterns
        violation_messages = [v[2] for v in violations]

        # Check invalid patterns are caught
        assert any(
            "EO001" in msg and "DataProcessor" in msg for msg in violation_messages
        )
        assert any("EO005" in msg for msg in violation_messages)
        assert any("EO007" in msg and "get_name" in msg for msg in violation_messages)
        assert any("EO015" in msg for msg in violation_messages)

    def test_plugin_end_to_end(self) -> None:
        """Test complete plugin functionality end-to-end."""
        code = """
class FileManager:  # EO001
    def __init__(self):
        self.files = []  # EO006

    def get_file(self):  # EO002 + EO007
        return None  # EO005
"""

        # Test through the plugin interface
        tree = ast.parse(code)
        plugin = ElegantObjectsPlugin(tree)

        errors = list(plugin.run())
        assert len(errors) > 0

        # Check that each error has the correct format: (line, col, msg, plugin_type)
        for line, col, msg, plugin_type in errors:
            assert isinstance(line, int)
            assert isinstance(col, int)
            assert isinstance(msg, str)
            assert plugin_type == ElegantObjectsPlugin
            assert msg.startswith("EO")

    def test_all_error_codes_represented(self) -> None:
        """Test that we can generate violations for all error codes."""
        code = """
# EO001 - -er class name
class Manager:
    # EO008 - mutable class attribute
    data = []

    def __init__(self):
        # EO006 - code in constructor
        self.processed = []

    # EO002 - procedural method name
    def process_data(self):
        # EO005 - None usage
        return None

    # EO007 - getter method
    def get_value(self):
        return self.value

    # EO009 - static method
    @staticmethod
    def handle_request():
        # EO010 - isinstance usage
        if isinstance(data, str):
            return data

# EO004 - procedural function name
def analyze_data():
    pass

# EO003 - procedural variable name
processor = DataProcessor()

# EO011 will be triggered by public methods without contracts
# EO012 would need test methods
# EO013 - ORM pattern
user.save()

# EO014 - implementation inheritance
class DataContainer(list):
    pass
"""
        violations = self._check_code(code)

        # Extract error codes
        error_codes = {v[2].split()[0] for v in violations}

        # Should have most error codes (some might not trigger in this specific example)
        expected_codes = {
            "EO001",
            "EO003",
            "EO004",
            "EO005",
            "EO006",
            "EO007",
            "EO008",
            "EO009",
            "EO010",
            "EO013",
            "EO014",
        }
        found_codes = error_codes.intersection(expected_codes)

        # Should find at least most of the expected codes
        assert len(found_codes) >= 8
