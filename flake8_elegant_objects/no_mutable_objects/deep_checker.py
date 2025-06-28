"""Deep mutability checker for complex mutation patterns."""

import ast

from ..base import ErrorCodes, Violations, violation
from .base import MutableStateTracker


class DeepMutabilityChecker:
    """Enhanced checker for deep mutability patterns."""

    def __init__(self) -> None:
        self.state_tracker = MutableStateTracker()

    def check_deep_mutations(self, tree: ast.AST) -> Violations:
        """Check for deep mutation patterns across the entire tree."""
        violations = []

        class_visitor = ClassInfoCollector(self.state_tracker)
        class_visitor.visit(tree)

        mutation_visitor = MutationDetector(self.state_tracker)
        mutation_visitor.visit(tree)
        violations.extend(mutation_visitor.violations)

        return violations


class ClassInfoCollector(ast.NodeVisitor):
    """Collects information about class attributes and their mutability."""

    def __init__(self, state_tracker: MutableStateTracker):
        self.state_tracker = state_tracker
        self.current_class: str | None = None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definition."""
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definition, particularly __init__."""
        if self.current_class and node.name == "__init__":
            self._process_init_method(node)
        self.generic_visit(node)

    def _process_init_method(self, init_node: ast.FunctionDef) -> None:
        """Process __init__ method for instance attributes."""
        for stmt in ast.walk(init_node):
            if isinstance(stmt, ast.Assign):
                self._process_assignment(stmt)

    def _process_assignment(self, stmt: ast.Assign) -> None:
        """Process assignment statement for self attributes."""
        if not self.current_class:
            return

        for target in stmt.targets:
            if self._is_self_attribute(target):
                assert isinstance(target, ast.Attribute)  # Type narrowing for mypy
                is_mutable = self._is_mutable_value(stmt.value)
                self.state_tracker.add_instance_attr(
                    self.current_class, target.attr, is_mutable
                )

    def _is_self_attribute(self, target: ast.expr) -> bool:
        """Check if target is a self attribute."""
        return (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
        )

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

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definition."""
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definition."""
        old_func = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = old_func

    def visit_Call(self, node: ast.Call) -> None:
        """Check for method calls that might mutate state."""
        if self.current_class and self.current_function != "__init__":
            if self._is_chained_mutation(node):
                self.violations.extend(
                    violation(node, ErrorCodes.EO021.format(name="chained mutation"))
                )
        self.generic_visit(node)

    def _is_chained_mutation(self, node: ast.Call) -> bool:
        """Detect chained mutations like self.dict.get('key', []).append()."""
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in {"append", "extend", "add", "update", "remove"}:
                if isinstance(node.func.value, ast.Call):
                    inner_call = node.func.value
                    if (
                        isinstance(inner_call.func, ast.Attribute)
                        and isinstance(inner_call.func.value, ast.Attribute)
                        and isinstance(inner_call.func.value.value, ast.Name)
                        and inner_call.func.value.value.id == "self"
                    ):
                        return True
        return False
