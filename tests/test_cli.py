"""Unit tests for the standalone command-line interface."""

import pathlib
import subprocess
import sys

import pytest

from flake8_elegant_objects.__main__ import main

CLEAN = (
    "from typing import final\n\n"
    "@final\nclass Money:\n"
    "    def __init__(self, cents):\n        self.cents = cents\n"
)
DIRTY = "class Parser:\n    def parse(self, text):\n        return None\n"


class TestCommandLineInterface:
    """Test cases for the standalone CLI."""

    def _run(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        *args: str,
    ) -> tuple[int, str, str]:
        """Run main() with the given arguments and capture its output."""
        monkeypatch.setattr(sys, "argv", ["flake8-elegant-objects", *args])
        code = 0
        try:
            main()
        except SystemExit as exit_:
            code = int(exit_.code or 0)
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    def test_reports_violations_and_exits_nonzero(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Files with violations exit with status 1 and list every violation."""
        target = tmp_path / "dirty.py"
        target.write_text(DIRTY)
        code, out, _ = self._run(monkeypatch, capsys, str(target))
        assert code == 1
        assert "EO001" in out
        assert "Total violations found:" in out

    def test_clean_file_exits_zero(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A compliant file exits with status 0."""
        target = tmp_path / "clean.py"
        target.write_text(CLEAN)
        code, out, _ = self._run(monkeypatch, capsys, str(target))
        assert code == 0
        assert "No violations found" in out

    def test_show_source_prints_offending_line(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """--show-source echoes the source line of each violation."""
        target = tmp_path / "dirty.py"
        target.write_text(DIRTY)
        _, out, _ = self._run(monkeypatch, capsys, "--show-source", str(target))
        assert "class Parser:" in out

    def test_non_python_files_are_skipped(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Arguments without a .py suffix are ignored."""
        target = tmp_path / "notes.txt"
        target.write_text(DIRTY)
        code, out, _ = self._run(monkeypatch, capsys, str(target))
        assert code == 0
        assert "notes.txt" not in out

    def test_syntax_error_is_reported_on_stderr(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Unparseable files are reported without aborting the run."""
        target = tmp_path / "broken.py"
        target.write_text("class Broken(:\n")
        code, _, err = self._run(monkeypatch, capsys, str(target))
        assert code == 0
        assert "Error processing" in err

    def test_missing_file_is_reported_on_stderr(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Nonexistent paths are reported without aborting the run."""
        code, _, err = self._run(monkeypatch, capsys, str(tmp_path / "ghost.py"))
        assert code == 0
        assert "Error processing" in err

    def test_installed_console_script_runs(self, tmp_path: pathlib.Path) -> None:
        """The console script declared in pyproject.toml is importable."""
        target = tmp_path / "clean.py"
        target.write_text(CLEAN)
        result = subprocess.run(
            ["flake8-elegant-objects", str(target)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert "Traceback" not in result.stderr
        assert result.returncode == 0

    def test_module_entry_point_runs(self, tmp_path: pathlib.Path) -> None:
        """python -m flake8_elegant_objects works as documented in the README."""
        target = tmp_path / "clean.py"
        target.write_text(CLEAN)
        result = subprocess.run(
            [sys.executable, "-m", "flake8_elegant_objects", str(target)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
