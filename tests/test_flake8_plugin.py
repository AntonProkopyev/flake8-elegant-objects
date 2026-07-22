"""Tests for plugin registration through flake8 itself."""

import pathlib
import subprocess
import sys

DIRTY = """class Parser:
    def parse(self, text):
        return None
"""


class TestFlake8Registration:
    """Test cases for the EO entry point advertised in pyproject.toml."""

    def _flake8(self, *args: str) -> subprocess.CompletedProcess[str]:
        """Run flake8 with the given arguments."""
        return subprocess.run(
            [sys.executable, "-m", "flake8", *args],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_plugin_is_registered(self) -> None:
        """flake8 --version lists the plugin under its distribution name."""
        result = self._flake8("--version")
        assert "flake8-elegant-objects" in result.stdout

    def test_violations_reach_flake8_output(self, tmp_path: pathlib.Path) -> None:
        """EO codes are reported through flake8, not only the standalone CLI."""
        target = tmp_path / "dirty.py"
        target.write_text(DIRTY)
        result = self._flake8("--select=EO", str(target))
        assert "EO001" in result.stdout
        assert result.returncode == 1

    def test_codes_are_individually_selectable(self, tmp_path: pathlib.Path) -> None:
        """A single EO code can be selected without pulling in the others."""
        target = tmp_path / "dirty.py"
        target.write_text(DIRTY)
        result = self._flake8("--select=EO001", str(target))
        assert "EO001" in result.stdout
        assert "EO005" not in result.stdout

    def test_noqa_suppresses_a_violation(self, tmp_path: pathlib.Path) -> None:
        """A noqa comment silences the code it names."""
        target = tmp_path / "suppressed.py"
        target.write_text(
            "class Parser:  # noqa: EO001\n    def name(self):\n        return 1\n"
        )
        result = self._flake8("--select=EO", str(target))
        assert "EO001" not in result.stdout
