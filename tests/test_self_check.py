"""The linter held to its own rules.

The plugin passes every rule it enforces, except where a line says
otherwise out loud. Each suppression is a place where obeying the rule
would mean disobeying something that cannot be argued with:

    EO005  one None, the absence of an enclosing class. A null object
           cannot stand in: checkers ask ast for a name, bases and body,
           and a hollow ClassDef would answer with lies, not with nothing
    EO009  entry points named in pyproject and in flake8's registry; the
           plugin does not get to choose whether they are functions
    EO010  the checker type flake8 wants in the tuple it is yielded, and
           one isinstance inside Instance, because a linter that reads
           syntax trees cannot ask a node what it is any other way
    EO011  a contract for every helper method would mean a Protocol per
           method and twice the classes, for nothing
    EO014  Generic, which is what keeps TypeGuard narrowing alive at
           every call site

The count is fixed. Adding a suppression is a decision, and this test
makes it one that has to be taken deliberately rather than in passing.
"""

import ast
import collections
import pathlib

from flake8_elegant_objects.base import ElegantObjectsCore
from flake8_elegant_objects.noqa import Noqa

SOURCES = pathlib.Path(__file__).parent.parent / "flake8_elegant_objects"

SUPPRESSED = {
    "EO005": 1,
    "EO009": 2,
    "EO010": 4,
    "EO011": 16,
    "EO014": 1,
}


class TestSelfCheck:
    """Test cases holding the plugin to the rules it enforces."""

    def _counted(self, marked: bool) -> dict[str, int]:
        """Count violations over the plugin's own source."""
        counts: collections.Counter[str] = collections.Counter()
        for path in sorted(SOURCES.rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            noqa = Noqa(source)
            for found in ElegantObjectsCore(ast.parse(source)).check_violations():
                code = found.message.split(" ")[0]
                if noqa.allows(found.line, code) is not marked:
                    counts[code] += 1
        return dict(counts)

    def test_the_plugin_obeys_itself(self) -> None:
        """Every rule the plugin enforces, it keeps."""
        assert self._counted(marked=False) == {}

    def test_no_suppression_is_added_unnoticed(self) -> None:
        """The suppressions are exactly the ones argued for above."""
        assert self._counted(marked=True) == SUPPRESSED
