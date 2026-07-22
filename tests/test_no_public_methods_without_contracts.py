"""Unit tests for no public methods without contracts principle (NoPublicMethodsWithoutContracts)."""

import ast

from flake8_elegant_objects.base import Source
from flake8_elegant_objects.no_public_methods_without_contracts import (
    NoPublicMethodsWithoutContracts,
)


class TestNoPublicMethodsWithoutContracts:
    """Test cases for public methods without contracts violations detection."""

    def _check_code(self, code: str) -> list[str]:
        """Helper to check code and return violation messages."""
        tree = ast.parse(code)
        checker = NoPublicMethodsWithoutContracts()
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

    def test_public_method_without_contract_violation(self) -> None:
        """Test detection of public methods without contracts."""
        code = """
class Service:
    def public_method(self):
        pass
"""
        violations = self._check_code(code)
        assert len(violations) == 1
        assert "EO011" in violations[0]

    def test_private_method_valid(self) -> None:
        """Test that private methods are valid."""
        code = """
class Service:
    def _private_method(self):
        pass
"""
        violations = self._check_code(code)
        assert len(violations) == 0

    def test_protocol_class_valid(self) -> None:
        """Test that Protocol classes are valid."""
        code = """
from typing import Protocol

class ServiceProtocol(Protocol):
    def public_method(self):
        pass
"""
        violations = self._check_code(code)
        assert len(violations) == 0

    def test_abc_class_valid(self) -> None:
        """Test that ABC classes are valid."""
        code = """
from abc import ABC, abstractmethod

class AbstractService(ABC):
    @abstractmethod
    def public_method(self):
        pass
"""
        violations = self._check_code(code)
        assert len(violations) == 0

    def test_class_implementing_protocol_valid(self) -> None:
        """Test that classes implementing Protocol are valid."""
        code = """
from typing import Protocol

class FileProtocol(Protocol):
    def read(self) -> bytes: ...
    def write(self, data: bytes) -> int: ...

class GoodDiskFile(FileProtocol):
    def read(self) -> bytes:
        return b"content"

    def write(self, data: bytes) -> int:
        return len(data)
"""
        violations = self._check_code(code)
        assert len(violations) == 0

    def test_class_implementing_abc_valid(self) -> None:
        """Test that classes implementing ABC are valid."""
        code = """
from abc import ABC, abstractmethod

class StorageABC(ABC):
    @abstractmethod
    def save(self, key: str, value: str) -> None:
        pass

    @abstractmethod
    def load(self, key: str) -> str:
        pass

class GoodMemoryStorage(StorageABC):
    def __init__(self):
        self._data = {}

    def save(self, key: str, value: str) -> None:
        self._data[key] = value

    def load(self, key: str) -> str:
        return self._data[key]
"""
        violations = self._check_code(code)
        assert len(violations) == 0

    def test_multiple_public_methods_without_contract(self) -> None:
        """Test detection of multiple public methods without contracts."""
        code = """
class DiskFile:
    def read(self) -> bytes:
        return b"content"

    def write(self, data: bytes) -> int:
        return len(data)
"""
        violations = self._check_code(code)
        assert len(violations) == 2
        assert all("EO011" in v for v in violations)
        assert any("read" in v for v in violations)
        assert any("write" in v for v in violations)

    def test_mixed_contract_and_non_contract_methods(self) -> None:
        """Test partial compliance - some methods from contract, some not."""
        code = """
from abc import ABC, abstractmethod

class StorageABC(ABC):
    @abstractmethod
    def save(self, key: str, value: str) -> None:
        pass

    @abstractmethod
    def load(self, key: str) -> str:
        pass

class PartiallyGoodStorage(StorageABC):
    def save(self, key: str, value: str) -> None:
        pass

    def load(self, key: str) -> str:
        return ""

    def delete(self, key: str) -> None:
        pass
"""
        violations = self._check_code(code)
        assert len(violations) == 1
        assert "delete" in violations[0]
        assert "EO011" in violations[0]

    def test_special_methods_valid(self) -> None:
        """Test that special methods (dunder methods) are valid."""
        code = """
from typing import Protocol

class FileProtocol(Protocol):
    def read(self) -> bytes: ...

class FileWithPrivateMethods(FileProtocol):
    def read(self) -> bytes:
        return b"content"

    def __str__(self) -> str:
        return "File"

    def __len__(self) -> int:
        return 10
"""
        violations = self._check_code(code)
        assert len(violations) == 0

    def test_private_and_protected_methods_valid(self) -> None:
        """Test that private and protected methods are valid."""
        code = """
from typing import Protocol

class FileProtocol(Protocol):
    def read(self) -> bytes: ...
    def write(self, data: bytes) -> int: ...

class FileWithPrivateMethods(FileProtocol):
    def read(self) -> bytes:
        return self._read_internal()

    def write(self, data: bytes) -> int:
        return self._write_internal(data)

    def _read_internal(self) -> bytes:
        return b"internal"

    def _write_internal(self, data: bytes) -> int:
        return len(data)
"""
        violations = self._check_code(code)
        assert len(violations) == 0

    def test_multiple_protocols_valid(self) -> None:
        """Test that classes implementing multiple protocols are valid."""
        code = """
from typing import Protocol

class ReadableProtocol(Protocol):
    def read(self) -> str: ...

class WritableProtocol(Protocol):
    def write(self, data: str) -> None: ...

class ReadWriteFile(ReadableProtocol, WritableProtocol):
    def read(self) -> str:
        return "data"

    def write(self, data: str) -> None:
        pass
"""
        violations = self._check_code(code)
        assert len(violations) == 0

    def test_custom_protocol_naming_convention(self) -> None:
        """Test that custom protocols with Protocol suffix are recognized."""
        code = """
from typing import Protocol

class HTTPClientProtocol(Protocol):
    def get(self, url: str) -> str: ...
    def post(self, url: str, data: dict) -> str: ...

class SimpleHTTPClient(HTTPClientProtocol):
    def get(self, url: str) -> str:
        return "response"

    def post(self, url: str, data: dict) -> str:
        return "response"
"""
        violations = self._check_code(code)
        assert len(violations) == 0

    def test_bad_memory_storage_without_contract(self) -> None:
        """Test detection of class without any contract."""
        code = """
class BadMemoryStorage:
    def save(self, key: str, value: str) -> None:
        self._data[key] = value

    def load(self, key: str) -> str:
        return self._data[key]
"""
        violations = self._check_code(code)
        assert len(violations) == 2
        assert all("EO011" in v for v in violations)
        assert any("save" in v for v in violations)
        assert any("load" in v for v in violations)

    def test_empty_class_valid(self) -> None:
        """Test that empty classes don't trigger violations."""
        code = """
class EmptyClass:
    pass
"""
        violations = self._check_code(code)
        assert len(violations) == 0

    def test_test_methods_are_exempt(self) -> None:
        """Test that test methods are not contract implementations."""
        code = """
class TestMoney:
    def test_cents_are_counted(self):
        assert Money(100).cents() == 100
"""
        violations = self._check_code(code)
        assert len(violations) == 0

    def test_relative_import_base_is_unjudged(self) -> None:
        """Test that a base imported from a sibling module is not accused."""
        code = """
from .contracts import Money

class Dollars(Money):
    def cents(self) -> int:
        return 100
"""
        violations = self._check_code(code)
        assert len(violations) == 0

    def test_absolute_import_base_is_unjudged(self) -> None:
        """Test that a base imported by package path is not accused."""
        code = """
from mypackage.contracts import Money

class Dollars(Money):
    def cents(self) -> int:
        return 100
"""
        violations = self._check_code(code)
        assert len(violations) == 0

    def test_aliased_import_base_is_unjudged(self) -> None:
        """Test that a base imported under another name is not accused."""
        code = """
from .contracts import Money as Currency

class Dollars(Currency):
    def cents(self) -> int:
        return 100
"""
        violations = self._check_code(code)
        assert len(violations) == 0

    def test_module_attribute_base_is_unjudged(self) -> None:
        """Test that a base reached through an imported module is not accused."""
        code = """
import mypackage.contracts as contracts

class Dollars(contracts.Money):
    def cents(self) -> int:
        return 100
"""
        violations = self._check_code(code)
        assert len(violations) == 0

    def test_shadowed_import_still_judged_by_local_definition(self) -> None:
        """Test that a base defined here is judged even when a name is imported."""
        code = """
from .contracts import Money

class Money:
    def cents(self) -> int:
        return 0

class Dollars(Money):
    def cents(self) -> int:
        return 100
"""
        violations = self._check_code(code)
        assert len(violations) == 2
        assert all("EO011" in v for v in violations)

    def test_class_without_bases_is_still_judged(self) -> None:
        """Test that imports do not excuse a class that inherits nothing."""
        code = """
from .contracts import Money

class Dollars:
    def cents(self) -> int:
        return 100
"""
        violations = self._check_code(code)
        assert len(violations) == 1
        assert "EO011" in violations[0]

    def test_async_methods_are_checked(self) -> None:
        """Test that async public methods are checked like the rest."""
        code = """
class Money:
    async def cents(self):
        return 100
"""
        violations = self._check_code(code)
        assert len(violations) == 1
        assert "EO011" in violations[0]
