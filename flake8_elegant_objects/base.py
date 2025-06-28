"""Base classes and protocols for Elegant Objects checkers."""

import ast
from typing import Protocol


class ErrorCodes:
    """Centralized error message definitions."""

    # Naming violations (EO001-EO004)
    EO001 = "EO001 Class name '{name}' violates -er principle (describes what it does, not what it is)"
    EO002 = (
        "EO002 Method name '{name}' violates -er principle (should be noun, not verb)"
    )
    EO003 = (
        "EO003 Variable name '{name}' violates -er principle (should be noun, not verb)"
    )
    EO004 = (
        "EO004 Function name '{name}' violates -er principle (should be noun, not verb)"
    )
    EO005 = "EO005 Null (None) usage violates EO principle (avoid None)"
    EO006 = "EO006 Code in constructor violates EO principle (constructors should only assign parameters)"
    EO007 = "EO007 Getter/setter method '{name}' violates EO principle (avoid getters/setters)"
    EO008 = "EO008 Mutable dataclass violation: {name}"
    EO009 = (
        "EO009 Static method '{name}' violates EO principle (no static methods allowed)"
    )
    EO010 = "EO010 isinstance/type casting violates EO principle (avoid type discrimination)"
    EO011 = "EO011 Public method '{name}' without contract (Protocol/ABC) violates EO principle"
    EO012 = "EO012 Test method '{name}' contains non-assertThat statements (only assertThat allowed)"
    EO013 = "EO013 ORM/ActiveRecord pattern '{name}' violates EO principle"
    EO014 = "EO014 Implementation inheritance violates EO principle (class '{name}' inherits from non-abstract class)"
    EO015 = "EO015 Mutable class attribute violation: {name}"
    EO016 = "EO016 Mutable instance attribute violation: {name}"
    EO017 = "EO017 Instance attribute mutation violation: {name}"
    EO018 = "EO018 Augmented assignment mutation violation: {name}"
    EO019 = "EO019 Mutating method call violation: {name}"
    EO020 = "EO020 Subscript assignment mutation violation: {name}"
    EO021 = "EO021 Chained mutation violation: {name}"
    EO022 = "EO022 Missing factory methods violation: {name}"
    EO023 = "EO023 Mutable default argument violation: {name}"
    EO024 = "EO024 Missing immutability enforcement violation: {name}"
    EO025 = "EO025 Copy-on-write violation: {name}"
    EO026 = "EO026 Aliasing violation (exposing mutable state): {name}"
    EO027 = "EO027 Defensive copy violation: {name}"


class Violation:
    """Represents a detected violation."""

    def __init__(self, line: int, column: int, message: str) -> None:
        self._line = line
        self._column = column
        self._message = message

    @property
    def line(self) -> int:
        return self._line

    @property
    def column(self) -> int:
        return self._column

    @property
    def message(self) -> str:
        return self._message


Violations = list[Violation]


class Source:
    """Aggregation of AST node and current class context."""

    def __init__(
        self,
        node: ast.AST,
        current_class: ast.ClassDef | None = None,
        tree: ast.AST | None = None,
    ) -> None:
        self._node = node
        self._current_class = current_class
        self._tree = tree

    @property
    def node(self) -> ast.AST:
        return self._node

    @property
    def current_class(self) -> ast.ClassDef | None:
        return self._current_class

    @property
    def tree(self) -> ast.AST | None:
        return self._tree


class Principle(Protocol):
    """Protocol for Elegant Objects principles analysis."""

    def check(self, source: Source) -> Violations:
        """Check source for violations and return list of detected violations."""
        ...


def violation(node: ast.AST, message: str) -> Violations:
    """Create a violation if node has location information."""
    if hasattr(node, "lineno") and hasattr(node, "col_offset"):
        return [Violation(node.lineno, node.col_offset, message)]
    return []


def is_method(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Check if function is a method (has self parameter)."""
    if not node.args.args:
        return False
    return node.args.args[0].arg in {"self", "cls"}


def get_all_principles() -> list[Principle]:
    """Get all available Elegant Objects principle checkers."""
    # Import here to avoid circular imports
    from .no_constructor_code import NoConstructorCode
    from .no_er_name import NoErName
    from .no_getters_setters import NoGettersSetters
    from .no_implementation_inheritance import NoImplementationInheritance
    from .no_impure_tests import NoImpureTests
    from .no_mutable_objects import NoMutableObjects
    from .no_null import NoNull
    from .no_orm import NoOrm
    from .no_public_methods_without_contracts import NoPublicMethodsWithoutContracts
    from .no_static import NoStatic
    from .no_type_discrimination import NoTypeDiscrimination

    return [
        NoErName(),
        NoNull(),
        NoConstructorCode(),
        NoGettersSetters(),
        NoMutableObjects(),
        NoStatic(),
        NoTypeDiscrimination(),
        NoPublicMethodsWithoutContracts(),
        NoImpureTests(),
        NoOrm(),
        NoImplementationInheritance(),
    ]


class ElegantObjectsCore:
    """Core analyzer for Elegant Objects violations."""

    def __init__(self, tree: ast.AST) -> None:
        self.tree = tree
        self._parent_map: dict[ast.AST, ast.AST | None] = {}
        self._build_parent_map(tree, None)

    def _build_parent_map(self, node: ast.AST, parent: ast.AST | None) -> None:
        """Build a map of nodes to their parents for better context."""
        if parent:
            setattr(node, "_parent", parent)  # noqa: B010
        for child in ast.iter_child_nodes(node):
            self._build_parent_map(child, node)

    def check_violations(self) -> list[Violation]:
        """Check for all violations in the AST tree."""
        violations = []
        for violation in self._visit(self.tree, None):
            violations.append(violation)
        return violations

    def _visit(
        self, node: ast.AST, current_class: ast.ClassDef | None = None
    ) -> list[Violation]:
        """Visit AST nodes and check for violations."""
        violations = []

        if isinstance(node, ast.ClassDef):
            current_class = node

        violations.extend(self._check_principles(node, current_class))

        for child in ast.iter_child_nodes(node):
            violations.extend(self._visit(child, current_class))

        return violations

    def _check_principles(
        self, node: ast.AST, current_class: ast.ClassDef | None
    ) -> list[Violation]:
        """Check all principles against the given node."""
        violations = []
        source = Source(node, current_class, self.tree)
        principles = get_all_principles()

        for principle in principles:
            principle_violations = principle.check(source)
            violations.extend(principle_violations)

        return violations
