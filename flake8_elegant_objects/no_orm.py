"""No ORM principle checker for Elegant Objects violations."""

import ast
from typing import final

from .base import ErrorCodes, Instance, Source, Violations, violation

ATTRIBUTE = Instance(ast.Attribute)
CALL = Instance(ast.Call)
LITERAL: Instance[ast.Constant | ast.List | ast.Dict | ast.Tuple | ast.Set] = Instance((
    ast.Constant,
    ast.List,
    ast.Dict,
    ast.Tuple,
    ast.Set,
))
NAME = Instance(ast.Name)


@final
class NoOrm:
    """Checks for ORM/ActiveRecord pattern violations (EO013)."""

    def check(self, source: Source) -> Violations:
        """Check source for ORM pattern violations."""
        node = source.node

        if CALL.covers(node):
            return self._check_orm_patterns(node)

        return []

    def _check_orm_patterns(self, node: ast.Call) -> Violations:
        """Check for ORM/ActiveRecord patterns."""
        if not ATTRIBUTE.covers(node.func):
            return []

        orm_methods = {
            "save",
            "delete",
            "destroy",
            "create",
            "reload",
            "find_by",
            "where",
            "filter",
            "filter_by",
            "get_or_create",
            "select",
            "update_all",
            "delete_all",
            "execute",
            "query",
            "order_by",
            "group_by",
            "having",
            "limit",
            "offset",
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
        if NAME.covers(value) and value.id in {
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
        if NAME.covers(value) and value.id.endswith("_list"):
            return True

        # Literal values
        if LITERAL.covers(value):
            return True

        # Constructor calls
        return (
            CALL.covers(value)
            and NAME.covers(value.func)
            and value.func.id
            in {"open", "int", "str", "list", "dict", "set", "tuple", "bool", "float"}
        )
