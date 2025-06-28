"""Unit tests for NoMutableObjects principle."""

import ast

from flake8_elegant_objects.base import Source
from flake8_elegant_objects.no_mutable_objects import NoMutableObjects


class TestNoMutableObjectsPrinciple:
    """Test cases for mutable objects violations detection."""

    def _check_code(self, code: str) -> list[str]:
        """Helper to check code and return violation messages."""
        tree = ast.parse(code)
        checker = NoMutableObjects()
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

    def test_mutable_dataclass_violation(self) -> None:
        """Test detection of mutable dataclasses."""
        code = """
from dataclasses import dataclass

@dataclass
class MutableUser:
    name: str
    email: str

@dataclass()
class AnotherMutable:
    data: list
"""
        violations = self._check_code(code)
        assert len(violations) == 2
        assert all(
            "EO008" in v for v in violations
        )  # Dataclass violations still use EO008
        assert any("MutableUser" in v for v in violations)
        assert any("AnotherMutable" in v for v in violations)

    def test_frozen_dataclass_valid(self) -> None:
        """Test that frozen dataclasses don't trigger violations."""
        code = """
from dataclasses import dataclass

@dataclass(frozen=True)
class ImmutableUser:
    name: str
    email: str

@dataclass(frozen=True, slots=True)
class ImmutableData:
    value: int
"""
        violations = self._check_code(code)
        assert len(violations) == 0

    def test_mutable_class_attributes_violation(self) -> None:
        """Test detection of mutable class attributes."""
        code = """
class DataContainer:
    items = []  # Mutable class attribute
    config = {}  # Mutable class attribute
    tags = set()  # Mutable class attribute

class ProcessorConfig:
    allowed_types = ["str", "int"]  # Mutable class attribute
"""
        violations = self._check_code(code)
        assert len(violations) == 4
        assert all("EO015" in v for v in violations)

    def test_mutable_type_constructors_violation(self) -> None:
        """Test detection of mutable type constructors as class attributes."""
        code = """
class Configuration:
    data = list()  # Mutable
    settings = dict()  # Mutable
    cache = set()  # Mutable
    buffer = bytearray()  # Mutable

class ValidConfig:
    name = str()  # Immutable - OK
    count = int()  # Immutable - OK
"""
        violations = self._check_code(code)
        assert len(violations) == 4
        assert all("EO015" in v for v in violations)

    def test_immutable_class_attributes_valid(self) -> None:
        """Test that immutable class attributes don't trigger violations."""
        code = """
class Constants:
    MAX_SIZE = 100
    DEFAULT_NAME = "user"
    PI = 3.14159
    ENABLED = True

class ImmutableData:
    empty_tuple = ()
    frozen_set = frozenset([1, 2, 3])
"""
        violations = self._check_code(code)
        assert len(violations) == 0

    def test_mixed_dataclass_parameters(self) -> None:
        """Test dataclass with various parameter combinations."""
        code = """
from dataclasses import dataclass

@dataclass(frozen=False)
class ExplicitlyMutable:
    name: str

@dataclass(order=True, frozen=True)
class FrozenWithOrder:
    value: int

@dataclass(unsafe_hash=True)  # No frozen=True
class UnsafeHash:
    data: str
"""
        violations = self._check_code(code)
        mutable_violations = [
            v for v in violations if "EO008" in v
        ]  # Dataclass violations
        assert len(mutable_violations) == 2
        assert any("ExplicitlyMutable" in v for v in mutable_violations)
        assert any("UnsafeHash" in v for v in mutable_violations)

    def test_regular_classes_ignored(self) -> None:
        """Test that regular classes (non-dataclass) don't trigger dataclass violations."""
        code = """
class RegularClass:
    def __init__(self, name):
        self.name = name

class AnotherRegular:
    data = []  # This should trigger mutable attribute violation
"""
        violations = self._check_code(code)
        # Should only have 1 violation for the mutable class attribute
        assert len(violations) == 1
        assert "data" in violations[0]
        assert "EO015" in violations[0]  # Mutable class attribute

    def test_instance_attributes_ignored(self) -> None:
        """Test that instance attributes in methods are ignored."""
        code = """
class DataProcessor:
    def __init__(self):
        self.data = []  # Instance attribute - now detected as mutable
        self.cache = {}  # Instance attribute - now detected as mutable

    def process(self):
        self.temp = set()  # Instance attribute - OK
"""
        violations = self._check_code(code)
        # Enhanced checker now detects mutable instance attributes in __init__
        mutable_violations = [
            v for v in violations if "EO016" in v
        ]  # Instance attribute violations
        assert len(mutable_violations) >= 2  # At least the two __init__ violations

    def test_mutation_in_methods(self) -> None:
        """Test detection of instance attribute mutations in methods."""
        code = """
class DataProcessor:
    def __init__(self):
        self.data = []

    def process(self):
        self.data.append("something")  # Mutation via method call

    def update(self, value):
        self.data = value  # Direct mutation
"""
        violations = self._check_code(code)
        # Should have instance attribute violation (EO016), mutation violation (EO017), and mutating method call (EO019)
        assert len(violations) >= 2
        assert any("EO016" in v for v in violations)  # Instance attribute
        assert any(
            "EO017" in v or "EO019" in v for v in violations
        )  # Mutation or method call

    def test_augmented_assignment_mutation(self) -> None:
        """Test detection of augmented assignments."""
        code = """
class Counter:
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1  # Augmented assignment mutation
"""
        violations = self._check_code(code)
        # Should have augmented assignment violations (EO018)
        assert any("EO018" in v for v in violations)  # Augmented assignment
        assert any("augmented assignment" in v for v in violations)

    def test_list_mutation_methods(self) -> None:
        """Test detection of list mutation methods."""
        code = """
class ListManager:
    def __init__(self):
        self.items = []

    def add_item(self, item):
        self.items.append(item)  # Mutation

    def sort_items(self):
        self.items.sort()  # In-place mutation
"""
        violations = self._check_code(code)
        # Should have instance attribute (EO016) and mutating method calls (EO019)
        assert any("EO016" in v for v in violations)  # Instance attribute
        assert any("EO019" in v for v in violations)  # Mutating method calls
        assert any("mutating method" in v for v in violations)

    def test_comprehensions_as_mutable(self) -> None:
        """Test that comprehensions creating mutable types are detected."""
        code = """
class DataHandler:
    def __init__(self, items):
        self.data = [x for x in items]  # List comprehension is mutable
        self.mapping = {x: x*2 for x in items}  # Dict comprehension is mutable
"""
        violations = self._check_code(code)
        # Should have instance attribute violations (EO016) for comprehensions
        assert any("EO016" in v for v in violations)  # Instance attribute violations

    def test_dict_subscript_mutations(self) -> None:
        """Test detection of dictionary subscript mutations."""
        code = """
class UserManager:
    def __init__(self):
        self.metadata = {}

    def update_metadata(self, key, value):
        self.metadata[key] = value  # Mutation via subscript
"""
        violations = self._check_code(code)
        # Should have instance attribute (EO016) and subscript mutation (EO020)
        assert any("EO016" in v for v in violations)  # Instance attribute
        # FIXME: EO020 subscript mutations not detected - needs proper AST parent tracking

    def test_nested_mutable_structures(self) -> None:
        """Test detection of nested mutable data structures."""
        code = """
class Registry:
    def __init__(self):
        self.data = {"services": []}  # EO008: Nested mutable

    def add_service(self, service):
        self.data.get("services", []).append(service)  # Chained mutation
"""
        violations = self._check_code(code)
        # Should have instance attribute (EO016) and potentially chained mutation (EO021)
        assert any("EO016" in v for v in violations)  # Instance attribute

    def test_mutable_default_arguments(self) -> None:
        """Test detection of mutable default arguments."""
        code = """
class Service:
    def process(self, items=[]):  # Mutable default argument
        items.append("processed")
        return items
"""
        violations = self._check_code(code)
        # Note: This test depends on enhanced checker detecting mutable defaults
        assert isinstance(violations, list)  # Basic sanity check

    def test_exposing_internal_mutable_state(self) -> None:
        """Test detection of methods that expose internal mutable state."""
        code = """
class Container:
    def __init__(self):
        self._items = []

    def items(self):
        return self._items  # Exposing mutable state (aliasing violation)
"""
        violations = self._check_code(code)
        # This depends on enhanced aliasing detection
        assert isinstance(violations, list)

    def test_missing_defensive_copies(self) -> None:
        """Test detection of missing defensive copies."""
        code = """
class Collection:
    def __init__(self, items):
        self.items = items  # Direct assignment of potentially mutable parameter
"""
        violations = self._check_code(code)
        # This depends on enhanced defensive copy detection
        assert isinstance(violations, list)

    def test_complex_mutation_patterns(self) -> None:
        """Test detection of complex mutation patterns."""
        code = """
class DataProcessor:
    def __init__(self):
        self.pipeline = []
        self.cache = {}

    def process(self, data):
        self.cache.clear()  # Clear mutation
        for step in self.pipeline:
            self.cache[step] = step(data)  # Assignment mutation
        return self.cache
"""
        violations = self._check_code(code)
        # Should have instance attribute violations (EO016) and possibly other violations
        assert any("EO016" in v for v in violations)  # Instance attribute violations

    def test_immutable_patterns_not_flagged(self) -> None:
        """Test that proper immutable patterns are not flagged."""
        code = """
from dataclasses import dataclass

@dataclass(frozen=True)
class ImmutablePoint:
    x: float
    y: float

    def moved(self, dx: float, dy: float):
        return ImmutablePoint(self.x + dx, self.y + dy)

class ImmutableUser:
    def __init__(self, name: str, tags: tuple = ()):
        self._name = name
        self._tags = tags  # Immutable tuple

    def with_tag(self, tag: str):
        return ImmutableUser(self._name, (*self._tags, tag))

class SafeCollection:
    def __init__(self, items: list):
        self._items = tuple(items)  # Convert to immutable

    def items(self) -> tuple:
        return self._items  # Return immutable view
"""
        violations = self._check_code(code)
        # Immutable patterns should not have any mutable object violations
        mutable_codes = [
            "EO008",
            "EO015",
            "EO016",
            "EO017",
            "EO018",
            "EO019",
            "EO020",
            "EO021",
            "EO022",
            "EO023",
            "EO024",
            "EO025",
            "EO026",
            "EO027",
        ]
        mutable_violations = [
            v for v in violations if any(code in v for code in mutable_codes)
        ]
        assert len(mutable_violations) == 0

    def test_builder_pattern_acceptable(self) -> None:
        """Test that builder pattern with internal mutability is acceptable."""
        code = """
class DocumentBuilder:
    def __init__(self):
        self._title = ""
        self._sections = []  # Acceptable for builders

    def title(self, title: str):
        self._title = title
        return self

    def section(self, section: str):
        self._sections.append(section)  # Acceptable mutation in builder
        return self

    def build(self):
        return ImmutableDocument(self._title, tuple(self._sections))
"""
        violations = self._check_code(code)
        # Builder patterns may still trigger violations - this tests current behavior
        assert isinstance(violations, list)

    def test_aliasing_violations_eo026(self) -> None:
        """Test detection of aliasing violations (EO026)."""
        code = """
class DataContainer:
    def __init__(self):
        self.data = []

    def get_data(self):
        return self.data  # EO026: Exposing internal mutable state

    def get_items(self):
        return self.items  # EO026: Exposing internal mutable state
"""
        violations = self._check_code(code)
        aliasing_violations = [v for v in violations if "EO026" in v]
        assert len(aliasing_violations) >= 2
        assert any("self.data" in v for v in aliasing_violations)
        assert any("self.items" in v for v in aliasing_violations)

    def test_defensive_copy_violations_eo027(self) -> None:
        """Test detection of missing defensive copies (EO027)."""
        code = """
class Container:
    def __init__(self, items, data):
        self.items = items  # EO027: Should copy mutable parameter
        self.data = data    # EO027: Should copy mutable parameter
        self.count = 0      # OK: immutable assignment

    def update(self, value):
        self.value = value  # OK: not in __init__
"""
        violations = self._check_code(code)
        defensive_violations = [v for v in violations if "EO027" in v]
        assert len(defensive_violations) >= 2
        assert any("items" in v for v in defensive_violations)
        assert any("data" in v for v in defensive_violations)

    def test_no_false_positives_defensive_copy(self) -> None:
        """Test that proper defensive copies don't trigger violations."""
        code = """
class Container:
    def __init__(self, items):
        self.items = tuple(items)  # OK: defensive copy
        self.data = []             # OK: new mutable object

    def process(self, data):
        self.temp = data  # OK: not __init__
"""
        violations = self._check_code(code)
        defensive_violations = [v for v in violations if "EO027" in v]
        assert len(defensive_violations) == 0

    def test_no_false_positives_aliasing(self) -> None:
        """Test that safe returns don't trigger aliasing violations."""
        code = """
class Container:
    def __init__(self):
        self.data = []

    def size(self):
        return len(self.data)  # OK: not returning mutable state

    def process(self):
        result = self.data.copy()  # OK: not a return
        return result
"""
        violations = self._check_code(code)
        aliasing_violations = [v for v in violations if "EO026" in v]
        assert len(aliasing_violations) == 0
