"""No implementation inheritance principle checker for Elegant Objects violations."""

import ast
from typing import final

from .base import ErrorCodes, Source, Violations, violation


@final
class NoImplementationInheritance:
    """Checks for implementation inheritance violations (EO014)."""

    def check(self, source: Source) -> Violations:
        """Check source for implementation inheritance violations."""
        node = source.node

        if isinstance(node, ast.ClassDef):
            return self._check_implementation_inheritance(node)

        return []

    def _check_implementation_inheritance(self, node: ast.ClassDef) -> Violations:
        """Check for implementation inheritance violations."""
        for base in node.bases:
            is_abstract_base = False

            if isinstance(base, ast.Name):
                # Allow inheritance from abstract base classes and common patterns
                allowed_bases = {
                    # Abstract bases
                    "ABC",
                    "Protocol",
                    # Exception hierarchy (standard pattern)
                    "Exception",
                    "BaseException",
                    "ValueError",
                    "TypeError",
                    "RuntimeError",
                    "AttributeError",
                    "KeyError",
                    "IndexError",
                    "ImportError",
                    "OSError",
                    # Standard library abstract bases
                    "Enum",
                    "IntEnum",
                    "Flag",
                    "IntFlag",
                    # Generic object (unavoidable in Python)
                    "object",
                }
                is_abstract_base = base.id in allowed_bases

            elif isinstance(base, ast.Attribute):
                # Check for module.AbstractBase patterns
                if base.attr in {"Protocol", "ABC"}:
                    is_abstract_base = True
                elif isinstance(base.value, ast.Name) and base.value.id in {
                    "abc",
                    "typing",
                    "collections",
                    "enum",
                }:
                    is_abstract_base = True
                # Check for imported ABC/Protocol
                elif isinstance(base.value, ast.Name) and base.attr in {
                    "ABC",
                    "abstractmethod",
                    "Protocol",
                }:
                    is_abstract_base = True

            # If not an abstract base, it's implementation inheritance
            if not is_abstract_base:
                return violation(node, ErrorCodes.EO014.format(name=node.name))

        return []
