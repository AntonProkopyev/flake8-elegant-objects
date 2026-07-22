"""Shared mutable state checker for detecting dangerous sharing patterns."""

import ast
from typing import final

from ..base import ErrorCodes, Violations, violation


@final
class SharedMutableState:
    """Detects shared mutable state violations."""

    def check_shared_state(self, node: ast.ClassDef) -> Violations:
        """Check for shared mutable state patterns."""
        violations = []

        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                for default in item.args.defaults:
                    if self._is_mutable_default(default):
                        violations.extend(
                            violation(
                                item,
                                ErrorCodes.EO023.format(
                                    name=f"mutable default argument in {item.name}"
                                ),
                            )
                        )

        return violations

    def _is_mutable_default(self, node: ast.AST) -> bool:
        """Check if a default argument is mutable."""
        return isinstance(node, ast.List | ast.Dict | ast.Set)
