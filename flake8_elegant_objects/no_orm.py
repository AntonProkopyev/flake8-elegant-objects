"""No ORM principle checker for Elegant Objects violations."""

import ast
from typing import final

from .base import ErrorCodes, Source, Violations, violation


@final
class NoOrm:
    """Checks for ORM/ActiveRecord pattern violations (EO013)."""

    def check(self, source: Source) -> Violations:
        """Check source for ORM pattern violations."""
        node = source.node

        if isinstance(node, ast.Call):
            return self._check_orm_patterns(node)

        return []

    def _check_orm_patterns(self, node: ast.Call) -> Violations:
        """Check for ORM/ActiveRecord patterns."""
        if not isinstance(node.func, ast.Attribute):
            return []

        orm_methods = {
            "save",
            "delete",
            "destroy",
            "update",
            "create",
            "reload",
            "find",
            "find_by",
            "where",
            "filter",
            "filter_by",
            "get",
            "get_or_create",
            "select",
            "insert",
            "update_all",
            "delete_all",
            "execute",
            "query",
            "order_by",
            "group_by",
            "having",
            "limit",
            "offset",
            "join",
            "includes",
            "eager_load",
            "preload",
            "create_table",
            "drop_table",
            "add_column",
            "remove_column",
        }
        if node.func.attr not in orm_methods:
            return []

        # Check if this is a valid non-ORM usage
        if self._is_allowed_method_usage(node.func.value):
            return []

        return violation(node, ErrorCodes.EO013.format(name=node.func.attr))

    def _is_allowed_method_usage(self, value: ast.AST) -> bool:
        """Check if the method usage is allowed (not ORM)."""
        # Built-in types
        if isinstance(value, ast.Name) and value.id in {
            "list",
            "dict",
            "set",
            "tuple",
            "str",
            "int",
            "float",
            "bool",
        }:
            return True

        # Allow methods on list/dict variables
        if isinstance(value, ast.Name) and value.id.endswith("_list"):
            return True

        # Literal values
        if isinstance(value, ast.Constant | ast.List | ast.Dict | ast.Tuple | ast.Set):
            return True

        # Constructor calls
        return (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id
            in {"open", "int", "str", "list", "dict", "set", "tuple", "bool", "float"}
        )
