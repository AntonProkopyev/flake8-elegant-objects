"""Tests for packaging metadata that only breaks after release."""

import pathlib
import re

from flake8_elegant_objects import ElegantObjectsPlugin

PYPROJECT = pathlib.Path(__file__).parent.parent / "pyproject.toml"


class TestPackaging:
    """Test cases for metadata declared in more than one place."""

    def _declared_version(self) -> str:
        """Read the version the distribution is built with."""
        text = PYPROJECT.read_text(encoding="utf-8")
        project = text.split("[project]", 1)[1]
        found = re.search(r'^version\s*=\s*"([^"]+)"', project, re.MULTILINE)
        assert found, "no version declared in the [project] section"
        return found.group(1)

    def test_plugin_version_matches_distribution(self) -> None:
        """The version flake8 reports is the version that was packaged."""
        assert ElegantObjectsPlugin.version == self._declared_version()

    def test_plugin_name_matches_distribution(self) -> None:
        """The name flake8 reports is the name of the distribution."""
        assert ElegantObjectsPlugin.name == "flake8-elegant-objects"
