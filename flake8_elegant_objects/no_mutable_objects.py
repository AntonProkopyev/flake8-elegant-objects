"""No mutable objects principle checker for Elegant Objects violations."""

import ast

from .base import ErrorCodes, Source, Violations, violation


class NoMutableObjects:
    """Checks for mutable object violations (EO008) with enhanced detection."""

    def __init__(self):
        self.deep_checker = DeepMutabilityChecker()
        self.factory_checker = FactoryMethodChecker()
        self.shared_state_checker = SharedMutableStateChecker()
        self.contract_checker = ImmutabilityContractChecker()
        self.copy_on_write_checker = CopyOnWriteChecker()

    def check(self, source: Source) -> Violations:
        """Check source for mutable object violations with enhanced detection."""
        node = source.node
        violations = []

        # Run basic checks
        if isinstance(node, ast.ClassDef):
            violations.extend(self._check_mutable_class(node))
            # Run additional enhanced checks for classes
            violations.extend(self.factory_checker.check_factory_pattern(node))
            violations.extend(self.shared_state_checker.check_shared_state(node))
            violations.extend(self.contract_checker.check_immutability_contract(node))
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            violations.extend(
                self._check_mutable_assignments(node, source.current_class)
            )
            # Run copy-on-write checks for functions
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

        # Run deep analysis on the entire tree if available
        if source.tree and isinstance(source.node, ast.Module):
            violations.extend(self.deep_checker.check_deep_mutations(source.tree))

        return violations

    def _check_mutable_class(self, node: ast.ClassDef) -> Violations:
        """Check for mutable class violations."""
        violations = []

        # Look for @dataclass decorator without frozen=True
        has_dataclass = False
        has_frozen = False

        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id == "dataclass":
                has_dataclass = True
            elif isinstance(decorator, ast.Call):
                if (
                    isinstance(decorator.func, ast.Name)
                    and decorator.func.id == "dataclass"
                ):
                    has_dataclass = True
                    # Check for frozen=True
                    for keyword in decorator.keywords:
                        if keyword.arg == "frozen" and isinstance(
                            keyword.value, ast.Constant
                        ):
                            if keyword.value.value is True:
                                has_frozen = True

        # If it's a dataclass without frozen=True, it's mutable
        if has_dataclass and not has_frozen:
            violations.extend(
                violation(
                    node, ErrorCodes.EO008.format(name=f"@dataclass class {node.name}")
                )
            )

        # Check for mutable class attributes
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        if self._is_mutable_type(stmt.value):
                            violations.extend(
                                violation(
                                    stmt,
                                    ErrorCodes.EO008.format(
                                        name=f"class attribute '{target.id}'"
                                    ),
                                )
                            )

        return violations

    def _check_mutable_assignments(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        current_class: ast.ClassDef | None,
    ) -> Violations:
        """Check for mutable instance attribute assignments in methods."""
        violations = []

        # Skip if not in a class
        if not current_class:
            return violations

        # Check if this is __init__ method
        if node.name == "__init__":
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        # Check for self.attr = mutable_value
                        if (
                            isinstance(target, ast.Attribute)
                            and isinstance(target.value, ast.Name)
                            and target.value.id == "self"
                            and self._is_mutable_type(stmt.value)
                        ):
                            violations.extend(
                                violation(
                                    stmt,
                                    ErrorCodes.EO008.format(
                                        name=f"instance attribute 'self.{target.attr}'"
                                    ),
                                )
                            )

        return violations

    def _check_assignment_mutation(
        self, node: ast.Assign, current_class: ast.ClassDef | None
    ) -> Violations:
        """Check for mutations of instance attributes."""
        violations = []

        # Skip if not in a class
        if not current_class:
            return violations

        for target in node.targets:
            # Check for self.attr = value (mutation after __init__)
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                # Get the enclosing function
                parent = node
                while parent:
                    if isinstance(parent, ast.FunctionDef | ast.AsyncFunctionDef):
                        if parent.name != "__init__":
                            violations.extend(
                                violation(
                                    node,
                                    ErrorCodes.EO008.format(
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
        violations = []

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
                    ErrorCodes.EO008.format(
                        name=f"augmented assignment to 'self.{node.target.attr}'"
                    ),
                )
            )

        return violations

    def _check_mutating_method_call(
        self, node: ast.Call, current_class: ast.ClassDef | None
    ) -> Violations:
        """Check for calls to mutating methods on instance attributes."""
        violations = []

        if not current_class:
            return violations

        # Mutating methods for common mutable types
        mutating_methods = {
            "append",
            "extend",
            "insert",
            "remove",
            "pop",
            "clear",  # list
            "add",
            "discard",
            "update",  # set
            "popitem",
            "setdefault",  # dict
            "sort",
            "reverse",  # list in-place
        }

        # Check for self.attr.mutating_method()
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
                    ErrorCodes.EO008.format(
                        name=f"call to mutating method 'self.{node.func.value.attr}.{node.func.attr}()'"
                    ),
                )
            )

        return violations

    def _check_subscript_mutation(
        self, node: ast.Subscript, current_class: ast.ClassDef | None
    ) -> Violations:
        """Check for subscript mutations like self.data[0] = value."""
        violations = []

        if not current_class:
            return violations

        # Walk up to find if this subscript is being assigned to
        parent = getattr(node, "_parent", None)
        if parent and isinstance(parent, ast.Assign):
            # Check if assigning to self.attr[...]
            if (
                isinstance(node.value, ast.Attribute)
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "self"
            ):
                violations.extend(
                    violation(
                        parent,
                        ErrorCodes.EO008.format(
                            name=f"subscript assignment to 'self.{node.value.attr}[...]'"
                        ),
                    )
                )

        return violations

    def _is_mutable_type(self, node: ast.AST) -> bool:
        """Check if a node represents a mutable type."""
        # Direct mutable literals
        if isinstance(node, ast.List | ast.Dict | ast.Set):
            return True

        # Mutable type constructors
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            mutable_types = {"list", "dict", "set", "bytearray", "deque", "defaultdict"}
            return node.func.id in mutable_types

        # Check for mutable comprehensions
        return bool(isinstance(node, ast.ListComp | ast.DictComp | ast.SetComp))


class MutabilityVisitor(ast.NodeVisitor):
    """Helper visitor to track parent nodes for better mutation detection."""

    def __init__(self, checker: NoMutableObjects, current_class: ast.ClassDef | None):
        self.checker = checker
        self.current_class = current_class
        self.violations = []

    def visit(self, node: ast.AST) -> None:
        """Visit nodes and set parent references."""
        for child in ast.iter_child_nodes(node):
            child._parent = node
        self.generic_visit(node)


class MutableStateTracker:
    """Tracks mutable state across class definitions."""

    def __init__(self):
        self.instance_attrs: dict[str, set[str]] = {}  # class_name -> {attr_names}
        self.mutable_attrs: dict[
            str, set[str]
        ] = {}  # class_name -> {mutable_attr_names}

    def add_instance_attr(self, class_name: str, attr_name: str, is_mutable: bool):
        """Track instance attribute and whether it's mutable."""
        if class_name not in self.instance_attrs:
            self.instance_attrs[class_name] = set()
            self.mutable_attrs[class_name] = set()

        self.instance_attrs[class_name].add(attr_name)
        if is_mutable:
            self.mutable_attrs[class_name].add(attr_name)

    def is_mutable_attr(self, class_name: str, attr_name: str) -> bool:
        """Check if an attribute is known to be mutable."""
        return attr_name in self.mutable_attrs.get(class_name, set())


class DeepMutabilityChecker:
    """Enhanced checker for deep mutability patterns."""

    def __init__(self):
        self.state_tracker = MutableStateTracker()

    def check_deep_mutations(self, tree: ast.AST) -> Violations:
        """Check for deep mutation patterns across the entire tree."""
        violations = []

        # First pass: collect all class information
        class_visitor = ClassInfoCollector(self.state_tracker)
        class_visitor.visit(tree)

        # Second pass: check for violations
        mutation_visitor = MutationDetector(self.state_tracker)
        mutation_visitor.visit(tree)
        violations.extend(mutation_visitor.violations)

        return violations


class ClassInfoCollector(ast.NodeVisitor):
    """Collects information about class attributes and their mutability."""

    def __init__(self, state_tracker: MutableStateTracker):
        self.state_tracker = state_tracker
        self.current_class: str | None = None

    def visit_ClassDef(self, node: ast.ClassDef):
        """Visit class definition."""
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Visit function definition, particularly __init__."""
        if self.current_class and node.name == "__init__":
            # Analyze __init__ for instance attributes
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if (
                            isinstance(target, ast.Attribute)
                            and isinstance(target.value, ast.Name)
                            and target.value.id == "self"
                        ):
                            is_mutable = self._is_mutable_value(stmt.value)
                            self.state_tracker.add_instance_attr(
                                self.current_class, target.attr, is_mutable
                            )
        self.generic_visit(node)

    def _is_mutable_value(self, node: ast.AST) -> bool:
        """Determine if a value is mutable."""
        if isinstance(node, ast.List | ast.Dict | ast.Set):
            return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            return node.func.id in {"list", "dict", "set", "bytearray", "deque"}
        return False


class MutationDetector(ast.NodeVisitor):
    """Detects various mutation patterns."""

    def __init__(self, state_tracker: MutableStateTracker):
        self.state_tracker = state_tracker
        self.current_class: str | None = None
        self.current_function: str | None = None
        self.violations: Violations = []

    def visit_ClassDef(self, node: ast.ClassDef):
        """Visit class definition."""
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Visit function definition."""
        old_func = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = old_func

    def visit_Call(self, node: ast.Call):
        """Check for method calls that might mutate state."""
        if self.current_class and self.current_function != "__init__":
            # Check for chained mutations like self.data.get('key', []).append(value)
            if self._is_chained_mutation(node):
                self.violations.extend(
                    violation(node, ErrorCodes.EO008.format(name="chained mutation"))
                )
        self.generic_visit(node)

    def _is_chained_mutation(self, node: ast.Call) -> bool:
        """Detect chained mutations like self.dict.get('key', []).append()."""
        if isinstance(node.func, ast.Attribute):
            # Check if it's a mutating method
            if node.func.attr in {"append", "extend", "add", "update", "remove"}:
                # Check if it's called on a result of another method
                if isinstance(node.func.value, ast.Call):
                    # Check if the inner call is on self
                    inner_call = node.func.value
                    if (
                        isinstance(inner_call.func, ast.Attribute)
                        and isinstance(inner_call.func.value, ast.Attribute)
                        and isinstance(inner_call.func.value.value, ast.Name)
                        and inner_call.func.value.value.id == "self"
                    ):
                        return True
        return False


class FactoryMethodChecker:
    """Checks that objects are created immutably through factory methods."""

    def check_factory_pattern(self, node: ast.ClassDef) -> Violations:
        """Check if class follows immutable factory pattern."""
        violations = []
        has_mutable_init = False
        has_factory_methods = False

        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                if item.name == "__init__":
                    # Check if __init__ creates mutable state
                    for stmt in ast.walk(item):
                        if isinstance(stmt, ast.Assign):
                            for target in stmt.targets:
                                if (
                                    isinstance(target, ast.Attribute)
                                    and isinstance(target.value, ast.Name)
                                    and target.value.id == "self"
                                    and self._is_mutable_init(stmt.value)
                                ):
                                    has_mutable_init = True

                # Check for factory methods that return new instances
                elif self._returns_new_instance(item, node.name):
                    has_factory_methods = True

        # If class has mutable init but no factory methods, it's a violation
        if has_mutable_init and not has_factory_methods:
            violations.extend(
                violation(
                    node,
                    ErrorCodes.EO008.format(
                        name=f"class {node.name} with mutable state but no immutable factory methods"
                    ),
                )
            )

        return violations

    def _is_mutable_init(self, node: ast.AST) -> bool:
        """Check if initialization creates mutable state."""
        return isinstance(node, ast.List | ast.Dict | ast.Set)

    def _returns_new_instance(self, func: ast.FunctionDef, class_name: str) -> bool:
        """Check if function returns a new instance of the class."""
        for node in ast.walk(func):
            if isinstance(node, ast.Return) and node.value:
                if isinstance(node.value, ast.Call):
                    if (
                        isinstance(node.value.func, ast.Name)
                        and node.value.func.id == class_name
                    ):
                        return True
        return False


class SharedMutableStateChecker:
    """Detects shared mutable state violations."""

    def check_shared_state(self, node: ast.ClassDef) -> Violations:
        """Check for shared mutable state patterns."""
        violations = []

        # Check for class-level mutable defaults
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                for default in item.args.defaults:
                    if self._is_mutable_default(default):
                        violations.extend(
                            violation(
                                item,
                                ErrorCodes.EO008.format(
                                    name=f"mutable default argument in {item.name}"
                                ),
                            )
                        )

        return violations

    def _is_mutable_default(self, node: ast.AST) -> bool:
        """Check if a default argument is mutable."""
        return isinstance(node, ast.List | ast.Dict | ast.Set)


class ImmutabilityContractChecker:
    """Checks that classes properly declare and enforce immutability contracts."""

    def check_immutability_contract(self, node: ast.ClassDef) -> Violations:
        """Check if class has proper immutability declarations."""
        violations = []

        # Check for __slots__ to prevent dynamic attribute addition
        has_slots = any(
            isinstance(item, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "__slots__"
                for target in item.targets
            )
            for item in node.body
        )

        # Check for __setattr__ override to prevent mutation
        has_setattr_override = any(
            isinstance(item, ast.FunctionDef) and item.name == "__setattr__"
            for item in node.body
        )

        # Check if class has mutable attributes
        has_mutable_attrs = False
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                for stmt in ast.walk(item):
                    if isinstance(stmt, ast.Assign):
                        for target in stmt.targets:
                            if (
                                isinstance(target, ast.Attribute)
                                and isinstance(target.value, ast.Name)
                                and target.value.id == "self"
                                and self._is_mutable_value(stmt.value)
                            ):
                                has_mutable_attrs = True
                                break

        # If class has mutable attributes but no immutability enforcement
        if has_mutable_attrs and not (has_slots or has_setattr_override):
            violations.extend(
                violation(
                    node,
                    ErrorCodes.EO008.format(
                        name=f"class {node.name} with mutable attributes but no immutability enforcement"
                    ),
                )
            )

        return violations

    def _is_mutable_value(self, node: ast.AST) -> bool:
        """Check if a value is mutable."""
        return isinstance(node, ast.List | ast.Dict | ast.Set)


class CopyOnWriteChecker:
    """Checks for proper copy-on-write patterns for immutability."""

    def check_copy_on_write(self, node: ast.FunctionDef | ast.AsyncFunctionDef, class_name: str) -> Violations:
        """Check if mutations properly implement copy-on-write."""
        violations = []

        # Skip __init__ as it's allowed to set initial state
        if node.name == "__init__":
            return violations

        # Check for direct mutations without creating new instance
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                    ):
                        # Check if method returns a new instance
                        if not self._returns_new_instance(node, class_name):
                            violations.extend(
                                violation(
                                    stmt,
                                    ErrorCodes.EO008.format(
                                        name=f"mutation in {node.name} without returning new instance"
                                    ),
                                )
                            )

        return violations

    def _returns_new_instance(self, func: ast.FunctionDef | ast.AsyncFunctionDef, class_name: str) -> bool:
        """Check if function returns a new instance."""
        for node in ast.walk(func):
            if isinstance(node, ast.Return) and node.value:
                if isinstance(node.value, ast.Call):
                    if (
                        isinstance(node.value.func, ast.Name)
                        and node.value.func.id == class_name
                    ):
                        return True
        return False




# Additional specific violation patterns
class MutablePatternDetectors:
    """Collection of specific mutable pattern detectors."""

    @staticmethod
    def detect_aliasing_violations(node: ast.FunctionDef) -> Violations:
        """Detect aliasing that can lead to external mutation."""
        violations = []

        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Return) and stmt.value:
                # Check if returning internal mutable state directly
                if (
                    isinstance(stmt.value, ast.Attribute)
                    and isinstance(stmt.value.value, ast.Name)
                    and stmt.value.value.id == "self"
                ):
                    violations.extend(
                        violation(
                            stmt,
                            ErrorCodes.EO008.format(
                                name=f"returning internal mutable state 'self.{stmt.value.attr}'"
                            ),
                        )
                    )

        return violations

    @staticmethod
    def detect_defensive_copy_missing(node: ast.FunctionDef) -> Violations:
        """Detect missing defensive copies in constructors."""
        violations = []

        if node.name == "__init__":
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if (
                            isinstance(target, ast.Attribute)
                            and isinstance(target.value, ast.Name)
                            and target.value.id == "self"
                        ):
                            # Check if assigning parameter directly without copy
                            if isinstance(stmt.value, ast.Name) and stmt.value.id in [
                                arg.arg for arg in node.args.args[1:]
                            ]:
                                # This could be a mutable parameter assigned directly
                                violations.extend(
                                    violation(
                                        stmt,
                                        ErrorCodes.EO008.format(
                                            name=f"possible mutable parameter '{stmt.value.id}' assigned without defensive copy"
                                        ),
                                    )
                                )

        return violations


# Export the merged NoMutableObjects class
__all__ = ["NoMutableObjects"]
