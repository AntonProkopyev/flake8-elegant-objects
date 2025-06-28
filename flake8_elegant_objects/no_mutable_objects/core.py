"""Core NoMutableObjects checker that orchestrates all sub-checkers."""

import ast

from ..base import ErrorCodes, Source, Violations, violation
from .base import is_mutable_type
from .contract_checker import ImmutabilityContractChecker
from .copy_on_write_checker import CopyOnWriteChecker
from .deep_checker import DeepMutabilityChecker
from .factory_checker import FactoryMethodChecker
from .pattern_detectors import MutablePatternDetectors
from .shared_state_checker import SharedMutableStateChecker


class NoMutableObjects:
    """Checks for mutable object violations (EO008) with enhanced detection."""

    def __init__(self) -> None:
        self.deep_checker = DeepMutabilityChecker()
        self.factory_checker = FactoryMethodChecker()
        self.shared_state_checker = SharedMutableStateChecker()
        self.contract_checker = ImmutabilityContractChecker()
        self.copy_on_write_checker = CopyOnWriteChecker()

    def check(self, source: Source) -> Violations:
        """Check source for mutable object violations with enhanced detection."""
        node = source.node
        violations = []

        if isinstance(node, ast.ClassDef):
            violations.extend(self._check_mutable_class(node))
            violations.extend(self.factory_checker.check_factory_pattern(node))
            violations.extend(self.shared_state_checker.check_shared_state(node))
            violations.extend(self.contract_checker.check_immutability_contract(node))
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            violations.extend(
                self._check_mutable_assignments(node, source.current_class)
            )
            violations.extend(MutablePatternDetectors.detect_aliasing_violations(node))
            violations.extend(
                MutablePatternDetectors.detect_defensive_copy_missing(node)
            )
            if source.current_class:
                violations.extend(
                    self.copy_on_write_checker.check_copy_on_write(
                        node, source.current_class.name
                    )
                )
        elif isinstance(node, ast.Assign):
            violations.extend(
                self._check_assignment_mutation(node, source.current_class)
            )
        elif isinstance(node, ast.AugAssign):
            violations.extend(
                self._check_augmented_assignment(node, source.current_class)
            )
        elif isinstance(node, ast.Call):
            violations.extend(
                self._check_mutating_method_call(node, source.current_class)
            )
        elif isinstance(node, ast.Subscript):
            violations.extend(
                self._check_subscript_mutation(node, source.current_class)
            )

        if source.tree and isinstance(source.node, ast.Module):
            violations.extend(self.deep_checker.check_deep_mutations(source.tree))

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
            if isinstance(decorator, ast.Name) and decorator.id == "dataclass":
                has_dataclass = True
            elif isinstance(decorator, ast.Call) and self._is_dataclass_call(decorator):
                has_dataclass = True
                has_frozen = self._check_frozen_keyword(decorator.keywords)

        return has_dataclass, has_frozen

    def _is_dataclass_call(self, decorator: ast.Call) -> bool:
        """Check if decorator call is a dataclass."""
        return isinstance(decorator.func, ast.Name) and decorator.func.id == "dataclass"

    def _check_frozen_keyword(self, keywords: list[ast.keyword]) -> bool:
        """Check if frozen=True is set in dataclass keywords."""
        for keyword in keywords:
            if (
                keyword.arg == "frozen"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
            ):
                return True
        return False

    def _check_class_attributes(self, node: ast.ClassDef) -> Violations:
        """Check for mutable class attributes."""
        violations: Violations = []

        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                violations.extend(self._check_assignment_targets(stmt))

        return violations

    def _check_assignment_targets(self, stmt: ast.Assign) -> Violations:
        """Check assignment targets for mutable types."""
        violations: Violations = []

        for target in stmt.targets:
            if isinstance(target, ast.Name) and is_mutable_type(stmt.value):
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
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if (
                            isinstance(target, ast.Attribute)
                            and isinstance(target.value, ast.Name)
                            and target.value.id == "self"
                            and is_mutable_type(stmt.value)
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
        self, node: ast.Assign, current_class: ast.ClassDef | None
    ) -> Violations:
        """Check for mutations of instance attributes."""
        violations: Violations = []

        if not current_class:
            return violations

        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                parent: ast.AST | None = node
                while parent:
                    if isinstance(parent, ast.FunctionDef | ast.AsyncFunctionDef):
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
                    parent = getattr(parent, "_parent", None)

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
            isinstance(node.target, ast.Attribute)
            and isinstance(node.target.value, ast.Name)
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
            isinstance(node.func, ast.Attribute)
            and node.func.attr in mutating_methods
            and isinstance(node.func.value, ast.Attribute)
            and isinstance(node.func.value.value, ast.Name)
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
        self, node: ast.Subscript, current_class: ast.ClassDef | None
    ) -> Violations:
        """Check for subscript mutations like self.data[0] = value."""
        violations: Violations = []

        if not current_class:
            return violations

        parent = getattr(node, "_parent", None)
        if parent and isinstance(parent, ast.Assign):
            if (
                isinstance(node.value, ast.Attribute)
                and isinstance(node.value.value, ast.Name)
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
