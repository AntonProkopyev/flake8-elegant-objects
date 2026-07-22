"""Base classes and protocols for Elegant Objects checkers."""

import ast
from dataclasses import dataclass, field
from typing import Generic, Protocol, TypeGuard, TypeVar, final

from .links import LINKS

Kind = TypeVar("Kind")


# Error messages, one per code
# Naming violations (EO001-EO004)
EO001 = "EO001 Class name '{name}' violates -er principle (describes what it does, not what it is)"
EO002 = "EO002 Method name '{name}' violates -er principle (should be noun, not verb)"
EO003 = "EO003 Variable name '{name}' violates -er principle (should be noun, not verb)"
EO004 = "EO004 Function name '{name}' violates -er principle (should be noun, not verb)"
EO005 = "EO005 Null (None) usage violates EO principle (avoid None)"
EO006 = "EO006 Code in constructor violates EO principle (constructors should only assign parameters)"
EO007 = (
    "EO007 Getter/setter method '{name}' violates EO principle (avoid getters/setters)"
)
EO008 = "EO008 Mutable dataclass violation: {name}"
EO009 = "EO009 Static method '{name}' violates EO principle (no static methods allowed)"
EO010 = (
    "EO010 isinstance/type casting violates EO principle (avoid type discrimination)"
)
EO011 = (
    "EO011 Public method '{name}' without contract (Protocol/ABC) violates EO principle"
)
EO012 = "EO012 Test method '{name}' contains non-assertThat statements (only assertThat allowed)"
EO012_COUNT = (
    "EO012 Test method '{name}' must hold exactly one assertion, verifying"
    " one behaviour"
)
EO012_ORDER = "EO012 Test method '{name}' must hold its assertion as the last statement"
EO013 = "EO013 ORM/ActiveRecord pattern '{name}' violates EO principle"
EO014 = "EO014 Implementation inheritance violates EO principle (class '{name}' inherits from non-abstract class)"
EO015 = "EO015 Mutable class attribute violation: {name}"
EO016 = "EO016 Mutable instance attribute violation: {name}"
EO017 = "EO017 Instance attribute mutation violation: {name}"
EO018 = "EO018 Augmented assignment mutation violation: {name}"
EO019 = "EO019 Mutating method call violation: {name}"
EO020 = "EO020 Subscript assignment mutation violation: {name}"
EO021 = "EO021 Chained mutation violation: {name}"
EO023 = "EO023 Mutable default argument violation: {name}"
EO025 = "EO025 Copy-on-write violation: {name}"
EO026 = "EO026 Aliasing violation (exposing mutable state): {name}"
EO027 = "EO027 Defensive copy violation: {name}"
EO028 = "EO028 Class '{name}' is not final, which invites implementation inheritance"
EO029 = "EO029 Too many attributes: {name}, more than four attributes is not one object"


@final
class Instance(Generic[Kind]):  # noqa: EO014
    """A type, asked whether something is one of it.

    Type discrimination cannot leave a linter that reads syntax trees: the
    nodes of ast carry no polymorphic answer to "are you a name or a call".
    What it can do is live in one place. This is that place, and the single
    suppression below is the whole of it.

    The answer is a TypeGuard so that callers keep the narrowing they had
    from isinstance, which a plain bool would have taken away.
    """

    def __init__(self, kind: type[Kind] | tuple[type[Kind], ...]) -> None:
        self._kind = kind

    def covers(self, node: object) -> TypeGuard[Kind]:  # noqa: EO011
        """Answer whether the given object is of this type."""
        return isinstance(node, self._kind)  # noqa: EO010


@final
@dataclass(frozen=True)
class Violation:
    """One violation, at the place it was found.

    The three readers this used to carry were properties over private
    fields, which is the getter shape EO007 exists to refuse. Frozen
    fields say the same thing without the ceremony.
    """

    line: int
    column: int
    message: str


Violations = list[Violation]

CLASS_DEF = Instance(ast.ClassDef)


@final
class Parents:
    """Who holds whom in one syntax tree.

    ast gives a node no way back to the node containing it, and the answer
    used to be an attribute stitched onto the nodes themselves. Borrowed
    objects were being written to, which is exactly what this plugin tells
    everyone else not to do.
    """

    def __init__(self, of: dict[ast.AST, ast.AST]) -> None:
        self._of = of

    def above(self, node: ast.AST) -> ast.AST | None:  # noqa: EO011
        """Answer the node holding this one, or nothing at the root."""
        return self._of.get(node)


# The one absence in the package. A null object cannot stand in for the
# enclosing class: checkers ask ast for its name, bases and body, and a
# hollow ClassDef would answer them with lies rather than with nothing.
NOTHING: None = None  # noqa: EO005


@final
@dataclass(frozen=True)
class Source:
    """One node, with the context a principle needs to judge it."""

    node: ast.AST
    current_class: ast.ClassDef | None = NOTHING
    tree: ast.AST | None = NOTHING
    parents: Parents = field(default_factory=lambda: Parents({}))


class Principle(Protocol):
    """Protocol for Elegant Objects principles analysis."""

    def check(self, source: Source) -> Violations:
        """Check source for violations and return list of detected violations."""
        ...


@final
class Report:
    """Violations, raised where a node has a place to point at."""

    def of(self, node: ast.AST, message: str) -> Violations:  # noqa: EO011
        """Report a violation, unless the node carries no position."""
        if hasattr(node, "lineno") and hasattr(node, "col_offset"):  # noqa: EO010
            return [Violation(node.lineno, node.col_offset, LINKS.behind(message))]
        return []


@final
class Method:
    """A function, asked whether it belongs to an object."""

    def covers(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:  # noqa: EO011
        """Answer whether the function takes self or cls first."""
        if not node.args.args:
            return False
        return node.args.args[0].arg in {"self", "cls"}


REPORT = Report()
METHOD = Method()


@final
class AllPrinciples:
    """Every principle this plugin knows."""

    def each(self) -> list[Principle]:  # noqa: EO011
        """Assemble one checker per principle."""
        # Import here to avoid circular imports
        from .no_constructor_code import NoConstructorCode
        from .no_er_name import NoErName
        from .no_getters_setters import NoAccessMethods
        from .no_implementation_inheritance import NoImplementationInheritance
        from .no_impure_tests import NoImpureTests
        from .no_mutable_objects import NoMutableObjects
        from .no_null import NoNull
        from .no_open_classes import NoOpenClasses
        from .no_orm import NoOrm
        from .no_public_methods_without_contracts import NoPublicMethodsWithoutContracts
        from .no_static import NoStatic
        from .no_type_discrimination import NoTypeDiscrimination

        return [
            NoErName(),
            NoNull(),
            NoConstructorCode(),
            NoAccessMethods(),
            NoMutableObjects(),
            NoStatic(),
            NoTypeDiscrimination(),
            NoPublicMethodsWithoutContracts(),
            NoImpureTests(),
            NoOrm(),
            NoImplementationInheritance(),
            NoOpenClasses(),
        ]


@final
class ElegantObjectsCore:
    """Core analyzer for Elegant Objects violations."""

    def __init__(self, tree: ast.AST) -> None:
        self.tree = tree

    def _parents(self) -> Parents:
        """Map every node of the tree to the node holding it."""
        of: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(self.tree):
            for child in ast.iter_child_nodes(node):
                of[child] = node
        return Parents(of)

    def check_violations(self) -> list[Violation]:  # noqa: EO011
        """Check for all violations in the AST tree."""
        # The principles are assembled once per tree, not once per node
        return self._visit(
            self.tree, NOTHING, tuple(AllPrinciples().each()), self._parents()
        )

    def _visit(
        self,
        node: ast.AST,
        current_class: ast.ClassDef | None,
        principles: tuple[Principle, ...],
        parents: Parents,
    ) -> list[Violation]:
        """Visit AST nodes and check for violations."""
        violations = []

        if CLASS_DEF.covers(node):
            current_class = node

        violations.extend(
            self._check_principles(node, current_class, principles, parents)
        )

        for child in ast.iter_child_nodes(node):
            violations.extend(self._visit(child, current_class, principles, parents))

        return violations

    def _check_principles(
        self,
        node: ast.AST,
        current_class: ast.ClassDef | None,
        principles: tuple[Principle, ...],
        parents: Parents,
    ) -> list[Violation]:
        """Check all principles against the given node."""
        violations = []
        source = Source(node, current_class, self.tree, parents)

        for principle in principles:
            principle_violations = principle.check(source)
            violations.extend(principle_violations)

        return violations
