# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2024-06-28

### 🚀 Major Enhancements

#### Enhanced Mutable Objects Detection

- **Expanded EO008**: Complete rewrite from single checker to comprehensive modular package
- **12 New Error Codes**: Added EO015-EO027 for granular mutability detection
  - `EO015`: Mutable class attribute violations
  - `EO016`: Mutable instance attribute violations
  - `EO017`: Instance attribute mutation violations
  - `EO018`: Augmented assignment mutation violations
  - `EO019`: Mutating method call violations
  - `EO020`: Subscript assignment mutation violations
  - `EO021`: Chained mutation violations
  - `EO022`: Missing factory methods violations
  - `EO023`: Mutable default argument violations
  - `EO024`: Missing immutability enforcement violations
  - `EO025`: Copy-on-write violations
  - `EO026`: Aliasing violations (exposing mutable state)
  - `EO027`: Defensive copy violations

#### Modular Architecture

- **6 Specialized Checkers**: Organized mutable objects detection into focused modules
  - `core.py`: Main orchestrator coordinating all sub-checkers
  - `contract_checker.py`: Immutability contract enforcement
  - `copy_on_write_checker.py`: Copy-on-write pattern validation
  - `deep_checker.py`: Cross-class mutation analysis
  - `factory_checker.py`: Factory method pattern validation
  - `pattern_detectors.py`: Aliasing and defensive copy detection
  - `shared_state_checker.py`: Shared mutable state detection

### 🔧 Code Quality & Complexity Rules

#### Ruff Integration

- **McCabe Complexity**: Added `C90` with max-complexity = 10
- **Nested Blocks**: Added `PLR1702` with max-nested-blocks = 5
- **Set Literals**: Added `PLR6201` enforcement (tuple → set membership tests)
- **Preview Mode**: Enabled ruff preview for latest pylint rules

#### Code Refactoring

- **Reduced Complexity**: Refactored methods from 6-7 nesting levels to ≤5
- **Extract Methods**: Improved maintainability through helper method extraction
- **Static Methods**: Added `@staticmethod` decorators where appropriate
- **Type Safety**: Enhanced mypy compliance with proper type narrowing

### 📈 Testing & Coverage

#### Coverage Improvements

- **90% Coverage**: Increased from 86% to 90% (+5% improvement)
- **Pattern Detectors**: From 0% to 99% coverage
- **New Test Cases**: Added 4 comprehensive test cases for EO026/EO027
- **Integration Tests**: Enhanced end-to-end testing for new checkers

#### Test Quality

- **100 Tests**: All tests pass with enhanced coverage
- **False Positive Prevention**: Added tests ensuring clean code doesn't trigger violations
- **Real-world Scenarios**: Added aliasing and defensive copy test cases

### 🏗️ Architecture Improvements

#### Type Safety

- **MyPy Clean**: Fixed all type errors with proper AsyncFunctionDef support
- **Type Assertions**: Added strategic type narrowing for complex AST patterns
- **Null Safety**: Enhanced null checking for optional attributes

#### Documentation

- **Updated README**: Comprehensive error code documentation with examples
- **Project Structure**: Updated to reflect new modular architecture
- **Coverage Badge**: Updated to reflect improved test coverage (90%)

### 🐛 Bug Fixes

- **Pattern Detector Integration**: Fixed unused pattern detectors (0% → 99% coverage)
- **Type Compatibility**: Resolved mypy errors with union types
- **Whitespace Handling**: Consistent code formatting across all files

### ⚠️ Breaking Changes

- **Mutable Objects**: EO008 now focuses specifically on dataclass violations
- **New Error Codes**: Code that was previously missed may now trigger EO015-EO027

### 🔄 Internal Changes

- **Ruff Configuration**: Added comprehensive linting rules with complexity enforcement
- **Parent Tracking**: Enhanced AST analysis with parent node mapping
- **Modular Imports**: Reorganized imports for better dependency management

---

## [1.0.0] - 2024-06-21

### 🎉 Initial Release

This is the first release of `flake8-elegant-objects`, a comprehensive Flake8 plugin that enforces the core principles of Elegant Objects programming philosophy by Yegor Bugayenko.

### ✨ Features

#### Error Codes Implemented

- **EO001-EO004**: No "-er" naming principle violations

  - `EO001`: Class names ending in "-er" (Manager, Controller, Handler, etc.)
  - `EO002`: Method names ending in "-er" (process → processing)
  - `EO003`: Variable names ending in "-er" (parser → arguments)
  - `EO004`: Function names ending in "-er" (analyzer → analysis)

- **EO005**: No null (None) usage - prevents defensive programming and unclear contracts

- **EO006**: No code in constructors - constructors should only assemble objects, not execute logic

- **EO007**: No getters/setters - prevents data container anti-pattern, promotes behavioral objects

- **EO008**: No mutable objects - enforces immutability for thread-safety and predictability

- **EO009**: No static methods - promotes proper object-oriented design over procedural programming

- **EO010**: No type discrimination - prevents isinstance/type casting that violates polymorphism

- **EO011**: No public methods without contracts - requires Protocol/ABC contracts for public methods

- **EO012**: Test methods should only contain assertThat statements - enforces focused, single-assertion tests

- **EO013**: No ORM/ActiveRecord patterns - prevents mixing persistence with business logic

- **EO014**: No implementation inheritance - promotes composition over inheritance

#### 🔧 Technical Details

##### Requirements

- Python 3.10+
- AST-based analysis for accurate code inspection
- Zero runtime dependencies beyond Python standard library

##### Code Quality

- **86% Test Coverage**: Comprehensive test suite with real-world examples
- **Type Safety**: Full mypy type checking compliance
- **Code Style**: Ruff formatting and linting
- **Documentation**: Extensive examples and principle explanations

##### Performance

- **Efficient AST Traversal**: Optimized single-pass analysis
- **Minimal Memory Usage**: Lightweight violation tracking
- **Fast Execution**: Suitable for large codebases

#### 🧪 Testing

- **14 Test Modules**: One dedicated test file per principle
- **Real-world Examples**: Test cases based on actual code patterns
- **Integration Tests**: End-to-end plugin functionality verification
- **Continuous Testing**: Automated test execution with coverage reporting

#### 🚀 Usage Examples

```bash
# Install
pip install flake8-elegant-objects

# Use as flake8 plugin
flake8 --select=EO your_code/

# Run standalone
python -m flake8_elegant_objects your_code/

# Configuration
echo "[flake8]\nselect = E,W,F,EO" > .flake8
```
