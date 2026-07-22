"""Core NoMutableObjects checker that orchestrates all sub-checkers."""

import ast
from typing import final

from ..base import (
    ErrorCodes,
    Instance,
    Parents,
    Principle,
    Source,
    Violations,
    violation,
)
from .base import MUTABLE_TYPE
from .copy_on_write_checker import CopyOnWrite
from .deep_checker import DeepMutability
from .pattern_detectors import MutablePatterns
from .shared_state_checker import SharedMutableState

CLASS_DEF = Instance(ast.ClassDef)
FUNCTION: Instance[ast.FunctionDef | ast.AsyncFunctionDef] = Instance((
    ast.FunctionDef,
    ast.AsyncFunctionDef,
))
ASSIGN = Instance(ast.Assign)
AUG_ASSIGN = Instance(ast.AugAssign)
CALL = Instance(ast.Call)
SUBSCRIPT = Instance(ast.Subscript)
MODULE = Instance(ast.Module)
NAME = Instance(ast.Name)
ATTRIBUTE = Instance(ast.Attribute)
CONSTANT = Instance(ast.Constant)


@final
class NoMutableObjects(Principle):
    """Checks for mutable object violations (EO008) with enhanced detection."""

    def check(self, source: Source) -> Violations:
        """Check source for mutable object violations with enhanced detection."""
        node = source.node
        violations = []

        if CLASS_DEF.covers(node):
            violations.extend(self._check_mutable_class(node))
            violations.extend(SharedMutableState().check_shared_state(node))
        elif FUNCTION.covers(node):
            violations.extend(
                self._check_mutable_assignments(node, source.current_class)
            )
            violations.extend(MutablePatterns().detect_aliasing_violations(node))
            violations.extend(MutablePatterns().detect_defensive_copy_missing(node))
            if source.current_class:
                violations.extend(
                    CopyOnWrite().check_copy_on_write(node, source.current_class.name)
                )
        elif ASSIGN.covers(node):
            violations.extend(
                self._check_assignment_mutation(
                    node, source.current_class, source.parents
                )
            )
        elif AUG_ASSIGN.covers(node):
            violations.extend(
                self._check_augmented_assignment(node, source.current_class)
            )
        elif CALL.covers(node):
            violations.extend(
                self._check_mutating_method_call(node, source.current_class)
            )
        elif SUBSCRIPT.covers(node):
            violations.extend(
                self._check_subscript_mutation(
                    node, source.current_class, source.parents
                )
            )

        if source.tree and MODULE.covers(source.node):
            violations.extend(DeepMutability().check_deep_mutations(source.tree))

        return violations

    def _check_mutable_class(self, node: ast.ClassDef) -> Violations:
        """Check for mutable class violations."""
        violations: Violations = []

        violations.extend(self._check_dataclass_mutability(node))
        violations.extend(self._check_class_attributes(node))

        return violations

    def _check_dataclass_mutability(self, node: ast.ClassDef) -> Violations:
        """Check if dataclass is properly frozen."""
        has_dataclass, has_frozen = self._analyze_dataclass_decorators(
            node.decorator_list
        )

        if has_dataclass and not has_frozen:
            return violation(
                node, ErrorCodes.EO008.format(name=f"@dataclass class {node.name}")
            )
        return []

    def _analyze_dataclass_decorators(
        self, decorators: list[ast.expr]
    ) -> tuple[bool, bool]:
        """Analyze decorators for dataclass and frozen status."""
        has_dataclass = False
        has_frozen = False

        for decorator in decorators:
            if NAME.covers(decorator) and decorator.id == "dataclass":
                has_dataclass = True
            elif CALL.covers(decorator) and self._is_dataclass_call(decorator):
                has_dataclass = True
                has_frozen = self._check_frozen_keyword(decorator.keywords)

        return has_dataclass, has_frozen

    def _is_dataclass_call(self, decorator: ast.Call) -> bool:
        """Check if decorator call is a dataclass."""
        return NAME.covers(decorator.func) and decorator.func.id == "dataclass"

    def _check_frozen_keyword(self, keywords: list[ast.keyword]) -> bool:
        """Check if frozen=True is set in dataclass keywords."""
        for keyword in keywords:
            if (
                keyword.arg == "frozen"
                and CONSTANT.covers(keyword.value)
                and keyword.value.value is True
            ):
                return True
        return False

    def _check_class_attributes(self, node: ast.ClassDef) -> Violations:
        """Check for mutable class attributes."""
        violations: Violations = []

        for stmt in node.body:
            if ASSIGN.covers(stmt):
                violations.extend(self._check_assignment_targets(stmt))

        return violations

    def _check_assignment_targets(self, stmt: ast.Assign) -> Violations:
        """Check assignment targets for mutable types."""
        violations: Violations = []

        for target in stmt.targets:
            if NAME.covers(target) and MUTABLE_TYPE.covers(stmt.value):
                violations.extend(
                    violation(
                        stmt,
                        ErrorCodes.EO015.format(name=f"class attribute '{target.id}'"),
                    )
                )

        return violations

    def _check_mutable_assignments(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        current_class: ast.ClassDef | None,
    ) -> Violations:
        """Check for mutable instance attribute assignments in methods."""
        violations: Violations = []

        if not current_class:
            return violations

        if node.name == "__init__":
            for stmt in ast.walk(node):
                if ASSIGN.covers(stmt):
                    for target in stmt.targets:
                        if (
                            ATTRIBUTE.covers(target)
                            and NAME.covers(target.value)
                            and target.value.id == "self"
                            and MUTABLE_TYPE.covers(stmt.value)
                        ):
                            violations.extend(
                                violation(
                                    stmt,
                                    ErrorCodes.EO016.format(
                                        name=f"instance attribute 'self.{target.attr}'"
                                    ),
                                )
                            )

        return violations

    def _check_assignment_mutation(
        self,
        node: ast.Assign,
        current_class: ast.ClassDef | None,
        parents: Parents,
    ) -> Violations:
        """Check for mutations of instance attributes."""
        violations: Violations = []

        if not current_class:
            return violations

        for target in node.targets:
            if (
                ATTRIBUTE.covers(target)
                and NAME.covers(target.value)
                and target.value.id == "self"
            ):
                parent: ast.AST | None = node
                while parent:
                    if FUNCTION.covers(parent):
                        if parent.name != "__init__":
                            violations.extend(
                                violation(
                                    node,
                                    ErrorCodes.EO017.format(
                                        name=f"mutation of 'self.{target.attr}'"
                                    ),
                                )
                            )
                        break
                    parent = parents.above(parent)

        return violations

    def _check_augmented_assignment(
        self, node: ast.AugAssign, current_class: ast.ClassDef | None
    ) -> Violations:
        """Check for augmented assignments (+=, -=, etc.) to instance attributes."""
        violations: Violations = []

        if not current_class:
            return violations

        # Check for self.attr += value
        if (
            ATTRIBUTE.covers(node.target)
            and NAME.covers(node.target.value)
            and node.target.value.id == "self"
        ):
            violations.extend(
                violation(
                    node,
                    ErrorCodes.EO018.format(
                        name=f"augmented assignment to 'self.{node.target.attr}'"
                    ),
                )
            )

        return violations

    def _check_mutating_method_call(
        self, node: ast.Call, current_class: ast.ClassDef | None
    ) -> Violations:
        """Check for calls to mutating methods on instance attributes."""
        violations: Violations = []

        if not current_class:
            return violations

        mutating_methods = {
            "append",
            "extend",
            "insert",
            "remove",
            "pop",
            "clear",
            "add",
            "discard",
            "update",
            "popitem",
            "setdefault",
            "sort",
            "reverse",
        }

        if (
            ATTRIBUTE.covers(node.func)
            and node.func.attr in mutating_methods
            and ATTRIBUTE.covers(node.func.value)
            and NAME.covers(node.func.value.value)
            and node.func.value.value.id == "self"
        ):
            violations.extend(
                violation(
                    node,
                    ErrorCodes.EO019.format(
                        name=f"call to mutating method 'self.{node.func.value.attr}.{node.func.attr}()'"
                    ),
                )
            )

        return violations

    def _check_subscript_mutation(
        self,
        node: ast.Subscript,
        current_class: ast.ClassDef | None,
        parents: Parents,
    ) -> Violations:
        """Check for subscript mutations like self.data[0] = value."""
        violations: Violations = []

        if not current_class:
            return violations

        parent = parents.above(node)
        if parent and ASSIGN.covers(parent):
            if (
                ATTRIBUTE.covers(node.value)
                and NAME.covers(node.value.value)
                and node.value.value.id == "self"
            ):
                violations.extend(
                    violation(
                        parent,
                        ErrorCodes.EO020.format(
                            name=f"subscript assignment to 'self.{node.value.attr}[...]'"
                        ),
                    )
                )

        return violations
