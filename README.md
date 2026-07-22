# flake8-elegant-objects

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](https://github.com/AntonProkopyev/flake8-elegant-objects)
[![Coverage](https://img.shields.io/badge/coverage-94%25-brightgreen.svg)](https://github.com/AntonProkopyev/flake8-elegant-objects)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![EO principles respected here](https://www.elegantobjects.org/badge.svg)](https://www.elegantobjects.org)

A flake8 plugin that checks Python against the
[Elegant Objects](https://www.elegantobjects.org/) principles: no null, no code
in constructors, no getters and setters, no mutable objects, no -er names, no
static methods, no type discrimination, no public methods without a contract,
no statements in tests beyond the assertion, no ORM, no implementation
inheritance.

Every message ends with a link to the article the rule comes from, so a code
you have not met before explains itself.

## Install

```bash
pip install flake8-elegant-objects
```

## Run

Through flake8, which is how most projects will want it:

```bash
flake8 --select=EO your_package/
```

The plugin registers itself on install; `--select=EO` narrows the run to its
codes. Individual codes work too, and so does `# noqa: EO001`.

Standalone, when you want it without flake8:

```bash
python -m flake8_elegant_objects your_package/*.py
flake8-elegant-objects --show-source your_package/*.py
```

Both forms honour `noqa` comments and report the same violations.

## Codes

**Naming.** The principle names "readers, parsers, controllers, sorters, and so
on", so the -er and -or suffixes are matched rather than a fixed list of words.
Ordinary nouns are allowed by their final word: `ImmutableUser` and
`TaskCounter` pass, `Formatter` and `Recorder` do not.

| | |
|---|---|
| `EO001` | class name ends in -er |
| `EO002` | method name ends in -er |
| `EO003` | variable name ends in -er |
| `EO004` | function name ends in -er |

Verb names are not violations. A method is a noun that builds or a verb that
commands, and the principle objects to actors, not to verbs.

**Objects.**

| | |
|---|---|
| `EO005` | null used, including a bare `return` beside returns that carry a value |
| `EO006` | a constructor doing anything beyond binding its parameters |
| `EO007` | a getter or setter, whether spelled `get_name` or `@property` |
| `EO009` | a static method, a class method, or a module level function |
| `EO010` | `isinstance`, `type`, `cast`, reflection, or a `match` on classes |
| `EO011` | a public method with no Protocol or ABC behind it |
| `EO012` | a test holding anything but one assertion, closing with it |
| `EO013` | an ORM or ActiveRecord call |
| `EO014` | inheritance from something that is not a contract |
| `EO028` | a class left open, which invites implementation inheritance |
| `EO029` | a class holding more than four attributes |

**Mutability.** One principle, several shapes.

| | |
|---|---|
| `EO008` | a dataclass that is not frozen |
| `EO015` | a mutable class attribute |
| `EO016` | a mutable instance attribute |
| `EO017` | an instance attribute assigned outside the constructor |
| `EO018` | an augmented assignment to an attribute |
| `EO019` | a call that mutates in place, such as `append` |
| `EO020` | a subscript assignment to an attribute |
| `EO021` | a mutation reached through a chain |
| `EO023` | a mutable default argument |
| `EO025` | a method that mutates where it could return a new instance |
| `EO026` | internal mutable state handed out |
| `EO027` | a mutable argument stored without a copy |

## Where this is stricter or looser than the principles

`EO002`, `EO003` and `EO004` extend a rule about class names to methods,
variables and functions. `EO028` and `EO029` are not on elegantobjects.org at
all; they come from the
[rules yegor256 gives his own agents](https://github.com/yegor256/prompt).

`EO011` and `EO014` do not judge a base class they cannot see. A Protocol
declared in one module and implemented in another is ordinary, and a plugin
reading one file at a time has no evidence about an imported name. A base
defined in the file and not a contract is still reported.

`EO012` allows a bare `assert` and `self.assertEqual`, because Python has no
Hamcrest and `assertThat` would be a foreign requirement.

Test methods are exempt from `EO011`: they implement no contract.

## The plugin on itself

It passes. Run it over its own source and nothing is reported that is not
marked, and each mark says why:

- `EO009` and `EO010` where flake8 and `pyproject.toml` dictate a signature
- `EO010` once inside `Instance`, the object every type check goes through,
  because a linter reading syntax trees cannot ask a node what it is any other
  way
- `EO005` once, for the absence of an enclosing class, where a null object
  would answer with lies rather than with nothing
- `EO011` on helper methods whose contracts would cost a Protocol apiece
- `EO014` on `Generic`, which is what keeps type narrowing alive

`tests/test_self_check.py` fixes that count, so a new suppression has to be
argued for rather than added in passing.

## Configure

```ini
[flake8]
select = E,W,F,EO
per-file-ignores =
    tests/*:EO011,EO012
```

If you use ruff alongside, tell it these codes belong to someone else, or it
will strip the `noqa` comments that name them:

```toml
[tool.ruff.lint]
external = ["EO"]
```

## Develop

```bash
uv sync
uv run pytest tests/
uv run ruff check . && uv run ruff format --check .
uv run mypy .
```

### Layout

```
flake8_elegant_objects/
├── __init__.py                  # the flake8 plugin
├── __main__.py                  # the standalone command
├── base.py                      # Instance, Source, Violation, messages
├── links.py                     # the article behind each code
├── noqa.py                      # suppression comments outside flake8
├── no_constructor_code.py       # EO006
├── no_er_name.py                # EO001-EO004
├── no_getters_setters.py        # EO007
├── no_implementation_inheritance.py  # EO014
├── no_impure_tests.py           # EO012
├── no_null.py                   # EO005
├── no_open_classes.py           # EO028, EO029
├── no_orm.py                    # EO013
├── no_public_methods_without_contracts.py  # EO011
├── no_static.py                 # EO009
├── no_type_discrimination.py    # EO010
└── no_mutable_objects/          # EO008, EO015-EO027
    ├── base.py                  # what counts as mutable
    ├── core.py                  # the orchestrator
    ├── copy_on_write_checker.py # EO025
    ├── deep_checker.py          # mutations across a class
    ├── pattern_detectors.py     # EO026, EO027
    └── shared_state_checker.py  # shared mutable state
```

## Reading

- [Elegant Objects](https://www.elegantobjects.org/), the principles themselves
- [Object Thinking](http://amzn.to/2BVeiNl) by David West
- [Elegant Objects](http://www.yegor256.com/elegant-objects.html) vol. 1 and 2
  by Yegor Bugayenko

## Licence

MIT. See [LICENSE](LICENSE).
