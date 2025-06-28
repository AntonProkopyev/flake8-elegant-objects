# Flake8 ElegantObjects Plugin

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](https://github.com/AntonProkopyev/flake8-elegant-objects)
[![Coverage](https://img.shields.io/badge/coverage-90%25-brightgreen.svg)](https://github.com/AntonProkopyev/flake8-elegant-objects)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type_checked-mypy-blue.svg)](https://mypy.readthedocs.io/)
[![Flake8](https://img.shields.io/badge/flake8-plugin-orange.svg)](https://flake8.pycqa.org/)

Detects violations of [Elegant Objects principles](https://www.elegantobjects.org/) including the "-er" naming principle, null usage, mutable objects, code in constructors, and getter/setter patterns.

## Error Codes

### Naming Violations (EO001-EO004)
- `EO001`: Class name violates -er principle
- `EO002`: Method name violates -er principle
- `EO003`: Variable name violates -er principle
- `EO004`: Function name violates -er principle

### Object Behavior (EO005-EO007)
- `EO005`: Null (None) usage violates EO principle
- `EO006`: Code in constructor violates EO principle
- `EO007`: Getter/setter method violates EO principle

### Mutable Object Violations (EO008, EO015-EO027)
- `EO008`: Mutable dataclass violation
- `EO015`: Mutable class attribute violation
- `EO016`: Mutable instance attribute violation
- `EO017`: Instance attribute mutation violation
- `EO018`: Augmented assignment mutation violation
- `EO019`: Mutating method call violation
- `EO020`: Subscript assignment mutation violation
- `EO021`: Chained mutation violation
- `EO022`: Missing factory methods violation
- `EO023`: Mutable default argument violation
- `EO024`: Missing immutability enforcement violation
- `EO025`: Copy-on-write violation
- `EO026`: Aliasing violation (exposing mutable state)
- `EO027`: Defensive copy violation

### Design and Architecture (EO009-EO014)
- `EO009`: Static method violates EO principle (no static methods allowed)
- `EO010`: isinstance/type casting violates EO principle (avoid type discrimination)
- `EO011`: Public method without contract (Protocol/ABC) violates EO principle
- `EO012`: Test method contains non-assertThat statements (only assertThat allowed)
- `EO013`: ORM/ActiveRecord pattern violates EO principle
- `EO014`: Implementation inheritance violates EO principle

## Installation

```bash
pip install flake8-elegant-objects
```

## Usage

**Standalone:**

```bash
python -m flake8_elegant_objects path/to/files/*.py
python -m flake8_elegant_objects --show-source path/to/files/*.py
```

**As flake8 plugin:**

```bash
flake8 --select=EO path/to/files/
```

The plugin is automatically registered when the package is installed.

## Philosophy

Based on [Yegor Bugayenko](https://www.yegor256.com/)'s [Elegant Objects principles](https://www.elegantobjects.org/), this plugin enforces object-oriented design that treats objects as living, thinking entities rather than data containers or procedure executors.

### 1. No "-er" Entities (EO001-EO004)

**Why?** Names ending in "-er" describe what objects _do_ rather than what they _are_, reducing them to mechanical task performers instead of equal partners in your design.

- ❌ `class DataProcessor` → ✅ `class ProcessedData`
- ❌ `def analyze()` → ✅ `def analysis()`
- ❌ `parser = ArgumentParser()` → ✅ `arguments = ArgumentParser()`

### 2. No Null/None (EO005)

**Why?** Null references break object-oriented thinking by representing "absence of object" - but absence cannot participate in object interactions. They lead to defensive programming and unclear contracts.

- ❌ `return None` → ✅ Return null objects with safe default behavior
- ❌ `if user is None:` → ✅ Use null object pattern or throw exceptions

### 3. No Code in Constructors (EO006)

**Why?** Constructors should be about object assembly, not computation. Complex logic in constructors violates the principle that objects should be lazy and do work only when asked.

- ❌ `self.name = name.upper()` → ✅ `self.name = name` (transform on access)
- ❌ `self.items = [process(x) for x in data]` → ✅ `self.data = data` (process lazily)

### 4. No Getters/Setters (EO007)

**Why?** Getters and setters expose internal state, breaking encapsulation. They encourage "tell, don't ask" violations and treat objects as data containers rather than behavioral entities.

- ❌ `def get_value()` / `def set_value()` → ✅ Objects should expose behavior, not data
- ❌ `user.getName()` → ✅ `user.introduce_yourself()` or `user.greet(visitor)`

### 5. No Mutable Objects (EO008, EO015-EO027)

**Why?** Mutable objects introduce temporal coupling and make reasoning about code difficult. Immutable objects are thread-safe, predictable, and easier to test. This plugin provides comprehensive detection of various mutability patterns.

**Basic Mutability Issues:**
- ❌ `@dataclass class Data` → ✅ `@dataclass(frozen=True) class Data` *(EO008)*
- ❌ `items = []` (class attribute) → ✅ `items: tuple = ()` *(EO015)*
- ❌ `self.data = []` (instance attribute) → ✅ `self.data: tuple = ()` *(EO016)*

**Mutation Patterns:**
- ❌ `self.items.append(x)` → ✅ `self.items = (*self.items, x)` *(EO019)*
- ❌ `self.count += 1` → ✅ `return Counter(self.count + 1)` *(EO018)*
- ❌ `self.data[key] = value` → ✅ Use immutable data structures *(EO020)*
- ❌ `self.data = new_value` (after init) → ✅ Return new instance *(EO017)*

**Advanced Patterns:**
- ❌ `def items=[]:` (mutable defaults) → ✅ `def items=None:` + null object *(EO023)*
- ❌ `return self._items` (exposing mutable state) → ✅ `return tuple(self._items)` *(EO026)*
- ❌ `self.items = items` (no defensive copy) → ✅ `self.items = tuple(items)` *(EO027)*
- ❌ Class with mutable state but no factory methods → ✅ Provide immutable factory methods *(EO022)*

### 6. No Static Methods (EO009)

**Why?** Static methods belong to classes, not objects, breaking object-oriented design. They can't be overridden, can't be mocked easily, and promote procedural thinking. Every static method is a candidate for a new class.

- ❌ `@staticmethod def process()` → ✅ Create dedicated objects for behavior
- ❌ `Math.sqrt(x)` → ✅ `SquareRoot(x).value()`

### 7. No Type Discrimination (EO010)

**Why?** Using `isinstance`, type casting, or reflection is a form of object discrimination. It violates polymorphism by treating objects unequally based on their type rather than their behavior contracts.

- ❌ `isinstance(obj, str)` → ✅ Design common interfaces and use polymorphism
- ❌ `if type(x) == int:` → ✅ Let objects decide how to behave

### 8. No Public Methods Without Contracts (EO011)

**Why?** Public methods without explicit contracts (Protocol/ABC) create implicit dependencies and unclear expectations. Contracts make object collaboration explicit and testable.

- ❌ `class Service:` with ad-hoc public methods → ✅ `class Service(Protocol):` with defined contracts
- ❌ Implicit interfaces → ✅ Explicit protocols that can be tested and verified

### 9. Test Methods: Only assertThat (EO012)

**Why?** Test methods should contain only one assertion statement (preferably `assertThat`). Multiple statements create complex tests that are hard to understand and maintain. Each test should verify one specific behavior.

- ❌ `x = 5; y = calculate(x); assert y > 0` → ✅ `assertThat(calculate(5), is_(greater_than(0)))`
- ❌ Multiple assertions per test → ✅ One focused assertion per test

### 10. No ORM/ActiveRecord (EO013)

**Why?** ORM and ActiveRecord patterns mix data persistence concerns with business logic, violating single responsibility. They create anemic domain models and tight coupling to databases.

- ❌ `user.save()`, `Model.find()` → ✅ Separate repository objects
- ❌ Mixing persistence with business logic → ✅ Clean separation of concerns

### 11. No Implementation Inheritance (EO014)

**Why?** Implementation inheritance creates tight coupling between parent and child classes, making code fragile and hard to test. It violates composition over inheritance and creates deep hierarchies that are difficult to understand.

- ❌ `class UserList(list):` → ✅ `class UserList:` with composition
- ❌ Inheriting concrete implementations → ✅ Inherit only from abstractions (ABC/Protocol)

The plugin detects the "hall of shame" naming patterns: Manager, Controller, Helper, Handler, Writer, Reader, Converter, Validator, Router, Dispatcher, Observer, Listener, Sorter, Encoder, Decoder, Analyzer, etc.

## Configuration

The plugin is integrated with flake8. Add to your `.flake8` config:

```ini
[flake8]
select = E,W,F,EO
per-file-ignores =
    tests/*:EO012  # Allow non-assertThat in tests if needed
```

## Development

### Testing

Run all tests:

```bash
python -m pytest tests/ -v
```

### Code Quality

```bash
# Type checking
mypy flake8_elegant_objects/

# Linting and formatting
ruff check flake8_elegant_objects/
ruff format flake8_elegant_objects/
```

### Project Structure

```
flake8_elegant_objects/
├── __init__.py              # Main plugin entry point
├── __main__.py              # CLI interface
├── base.py                  # Core types, error codes, and base classes
├── no_constructor_code.py   # EO006: No code in constructors
├── no_er_name.py           # EO001-EO004: No -er naming violations
├── no_getters_setters.py   # EO007: No getter/setter methods
├── no_implementation_inheritance.py  # EO014: No implementation inheritance
├── no_impure_tests.py      # EO012: Test methods with single assertions
├── no_null.py              # EO005: No None/null usage
├── no_orm.py               # EO013: No ORM/ActiveRecord patterns
├── no_public_methods_without_contracts.py  # EO011: Methods need contracts
├── no_static.py            # EO009: No static methods
├── no_type_discrimination.py  # EO010: No isinstance/type casting
└── no_mutable_objects/     # EO008, EO015-EO027: Comprehensive mutability detection
    ├── __init__.py         # Package initialization
    ├── base.py             # Shared utilities and state tracking
    ├── core.py             # Main orchestrator for all mutable object checks
    ├── contract_checker.py # EO024: Immutability contract enforcement
    ├── copy_on_write_checker.py  # EO025: Copy-on-write pattern validation
    ├── deep_checker.py     # Cross-class mutation analysis
    ├── factory_checker.py  # EO022: Factory method pattern validation
    ├── pattern_detectors.py # EO026-EO027: Aliasing and defensive copy detection
    └── shared_state_checker.py  # EO023: Shared mutable state detection
```
