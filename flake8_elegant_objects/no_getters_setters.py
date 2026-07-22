"""No getters/setters principle checker for Elegant Objects violations."""

import ast
from typing import final

from .base import EO007, METHOD, REPORT, Instance, Principle, Source, Violations

ATTRIBUTE = Instance(ast.Attribute)
FUNCTION: Instance[ast.FunctionDef | ast.AsyncFunctionDef] = Instance((
    ast.FunctionDef,
    ast.AsyncFunctionDef,
))
NAME = Instance(ast.Name)
RETURN = Instance(ast.Return)


@final
class NoAccessMethods(Principle):
    """Checks for getter/setter methods (EO007)."""

    def check(self, source: Source) -> Violations:
        """Check source for getter/setter violations."""
        node = source.node
        if not FUNCTION.covers(node):
            return []
        return self._check_getters_setters(node)

    def _check_getters_setters(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> Violations:
        """Check for getter/setter methods."""
        if not METHOD.covers(node) or node.name.startswith("_"):
            return []

        # Property setters are setters, whatever the method is called
        for decorator in node.decorator_list:
            if ATTRIBUTE.covers(decorator) and decorator.attr == "setter":
                return REPORT.of(node, EO007.format(name=node.name))

        # Properties that only hand back an attribute are getters
        for decorator in node.decorator_list:
            if NAME.covers(decorator) and decorator.id == "property":
                if self._returns_attribute(node):
                    return REPORT.of(node, EO007.format(name=node.name))
                return []

        return self._check_names(node)

    def _returns_attribute(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        """Check if a method body is a single return of one's own attribute."""
        if len(node.body) != 1:
            return False
        statement = node.body[0]
        return (
            RETURN.covers(statement)
            and ATTRIBUTE.covers(statement.value)
            and NAME.covers(statement.value.value)
            and statement.value.value.id == "self"
        )

    def _check_names(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> Violations:
        """Check for Java-style getter and setter names."""
        name = node.name.lower()
        original_name = node.name

        # Check for getter patterns
        if (
            name.startswith("get_")
            or (
                name.startswith("get")
                and len(original_name) > 3
                and original_name[3].isupper()
            )
            or name == "get"
        ):
            return REPORT.of(node, EO007.format(name=node.name))

        # Check for setter patterns
        if (
            name.startswith("set_")
            or (
                name.startswith("set")
                and len(original_name) > 3
                and original_name[3].isupper()
            )
            or name == "set"
        ):
            return REPORT.of(node, EO007.format(name=node.name))

        return []
