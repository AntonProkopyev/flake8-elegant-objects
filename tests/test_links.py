"""Unit tests for the article behind each code."""

import ast

from flake8_elegant_objects.base import ElegantObjectsCore
from flake8_elegant_objects.links import LINKS, Links


class TestLinks:
    """Test cases for the reading attached to a violation."""

    def test_message_carries_its_article(self) -> None:
        """A message ends with the article behind its code."""
        assert LINKS.behind("EO005 Null usage").endswith(
            "https://www.yegor256.com/2014/05/13/why-null-is-bad.html"
        )

    def test_unknown_code_is_left_alone(self) -> None:
        """A message whose code has no article is returned untouched."""
        assert LINKS.behind("EO999 Nothing") == "EO999 Nothing"

    def test_every_link_points_at_a_source_of_the_rules(self) -> None:
        """No link points anywhere but where the rules come from."""
        hosts = ("https://www.yegor256.com/", "https://github.com/yegor256/")
        assert all(url.startswith(hosts) for url in Links.OF.values())

    def test_every_reported_code_has_an_article(self) -> None:
        """Every code the plugin raises on dirty code carries a link."""
        dirty = """
class Parser:
    items = []

    def __init__(self, data):
        self.data = list(data)

    def get_name(self):
        return self._name
"""
        raised = {
            found.message.split(" ")[0]
            for found in ElegantObjectsCore(ast.parse(dirty)).check_violations()
        }
        assert raised <= set(Links.OF)
