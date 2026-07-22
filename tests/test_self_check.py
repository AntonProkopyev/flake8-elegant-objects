"""The linter checked against its own source.

The plugin cannot pass its own rules today: an AST visitor is built on
isinstance, and EO010 forbids exactly that. What this test guards is the
direction of travel. Every count below is a debt, and the test fails when a
debt grows or a new one appears. Lowering a number here is always welcome and
requires no discussion; raising one is a deliberate act that has to be argued
for in the commit that does it.
"""

import ast
import collections
import pathlib

from flake8_elegant_objects.base import ElegantObjectsCore

SOURCES = pathlib.Path(__file__).parent.parent / "flake8_elegant_objects"

BASELINE = {
    "EO005": 15,
    "EO006": 14,
    "EO007": 6,
    "EO009": 14,
    "EO010": 158,
    "EO011": 31,
    "EO013": 2,
    "EO014": 4,
    "EO017": 6,
    "EO019": 1,
    "EO020": 2,
    "EO025": 6,
    "EO029": 1,
}


class TestSelfCheck:
    """Test cases holding the plugin to its own rules."""

    def _counts(self) -> dict[str, int]:
        """Count violations the plugin reports over its own source."""
        counts: collections.Counter[str] = collections.Counter()
        for path in sorted(SOURCES.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for found in ElegantObjectsCore(tree).check_violations():
                counts[found.message.split(" ")[0]] += 1
        return dict(counts)

    def test_no_debt_grows(self) -> None:
        """Test that no existing violation count has increased."""
        counts = self._counts()
        grown = {
            code: (count, BASELINE[code])
            for code, count in counts.items()
            if code in BASELINE and count > BASELINE[code]
        }
        assert not grown, f"violation counts grew, code: (now, baseline) {grown}"

    def test_no_new_violation_appears(self) -> None:
        """Test that no rule starts firing on the plugin that did not before."""
        counts = self._counts()
        fresh = {code: count for code, count in counts.items() if code not in BASELINE}
        assert not fresh, f"new violations appeared: {fresh}"

    def test_baseline_is_not_stale(self) -> None:
        """Test that the baseline records no debt that has already been paid."""
        counts = self._counts()
        paid = {code: BASELINE[code] for code in BASELINE if code not in counts}
        assert not paid, f"baseline lists violations that no longer occur: {paid}"
