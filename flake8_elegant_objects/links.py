"""Where each rule comes from.

A code on its own tells you that you broke something, not why anyone
thinks it matters. Every link below is the one elegantobjects.org gives
for that principle; EO028 and EO029 come instead from the rules
yegor256 hands his own agents.
"""

from typing import ClassVar, final

ER = "https://www.yegor256.com/2015/03/09/objects-end-with-er.html"
NULL = "https://www.yegor256.com/2014/05/13/why-null-is-bad.html"
CONSTRUCTORS = "https://www.yegor256.com/2015/05/07/ctors-must-be-code-free.html"
ACCESSORS = "https://www.yegor256.com/2014/09/16/getters-and-setters-are-evil.html"
MUTABILITY = "https://www.yegor256.com/2014/06/09/objects-should-be-immutable.html"
STATIC = "https://www.yegor256.com/2017/02/07/private-method-is-new-class.html"
CASTING = "https://www.yegor256.com/2015/04/02/class-casting-is-anti-pattern.html"
CONTRACTS = (
    "https://www.yegor256.com/2014/11/20/seven-virtues-of-good-object.html"
    "#2-he-works-by-contracts"
)
TESTS = "https://www.yegor256.com/2017/05/17/single-statement-unit-tests.html"
ORM = "https://www.yegor256.com/2014/12/01/orm-offensive-anti-pattern.html"
INHERITANCE = "https://www.yegor256.com/2016/09/13/inheritance-is-procedural.html"
PROMPT = "https://github.com/yegor256/prompt"


@final
class Links:
    """The article behind each code."""

    OF: ClassVar[dict[str, str]] = {
        "EO001": ER,
        "EO002": ER,
        "EO003": ER,
        "EO004": ER,
        "EO005": NULL,
        "EO006": CONSTRUCTORS,
        "EO007": ACCESSORS,
        "EO008": MUTABILITY,
        "EO009": STATIC,
        "EO010": CASTING,
        "EO011": CONTRACTS,
        "EO012": TESTS,
        "EO013": ORM,
        "EO014": INHERITANCE,
        "EO015": MUTABILITY,
        "EO016": MUTABILITY,
        "EO017": MUTABILITY,
        "EO018": MUTABILITY,
        "EO019": MUTABILITY,
        "EO020": MUTABILITY,
        "EO021": MUTABILITY,
        "EO023": MUTABILITY,
        "EO025": MUTABILITY,
        "EO026": MUTABILITY,
        "EO027": MUTABILITY,
        "EO028": PROMPT,
        "EO029": PROMPT,
    }

    def behind(self, message: str) -> str:  # noqa: EO011
        """Append the article behind the code this message opens with."""
        link = self.OF.get(message.split(" ", maxsplit=1)[0], "")
        if not link:
            return message
        return f"{message}, see {link}"


LINKS = Links()
