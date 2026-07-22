"""Suppression comments, as flake8 understands them.

flake8 filters noqa itself, so the plugin never had to. The standalone
command and anything else driving the core directly saw every violation,
suppressed or not, which made the two disagree about the same file.
"""

import re
from typing import final

COMMENT = re.compile(
    r"#\s*noqa(?::\s?(?P<codes>[A-Z]+[0-9]+(?:[,\s]+[A-Z]+[0-9]+)*))?",
    re.IGNORECASE,
)


@final
class Noqa:
    """The suppressions written into one source text."""

    def __init__(self, source: str) -> None:
        self._source = source

    def allows(self, line: int, code: str) -> bool:
        """Answer whether a violation of this code survives on this line."""
        lines = self._source.split("\n")
        if line < 1 or line > len(lines):
            return True
        found = COMMENT.search(lines[line - 1])
        if not found:
            return True
        codes = found.group("codes")
        if not codes:
            # A bare noqa suppresses everything on the line
            return False
        return code not in re.split(r"[,\s]+", codes.strip())
