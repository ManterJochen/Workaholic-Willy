"""Every name ``willy`` exports says what to pass it and what comes back, the way the owner's example does.

The owner, 2026-10-09: "ja es ist schön, dass man weiß es geht aber keine Ahnung was ich da alles einfügen muss". A
docstring that says why a function works and not what goes into it leaves its caller guessing. So every public
function and method behind the door is written in the Google style of the owner's example: what it does first, then
``Args:`` with every parameter as ``name (type): what it is``, then ``Returns:`` where something comes back, then
``Raises:``. A dataclass, a pydantic model and an enum list their fields or members under ``Attributes:``.

This test holds every one of them to it: a parameter the ``Args:`` block does not name fails, and so does an entry for
a parameter that is gone, so the block cannot drift from the signature. ``Raises:`` is not checked, since no reading of
a signature says what a function raises.
"""

from __future__ import annotations

import dataclasses
import enum
import inspect
import re
import unittest
from collections.abc import Iterator
from typing import Any

import willy

#: A Google-style section header on a line of its own.
_HEADER = re.compile(r"^(\s*)(Args|Arguments|Attributes|Returns|Yields|Raises|Note|Notes|Example|Examples|Warning|"
                     r"See Also):\s*$")
#: An entry of an Args or Attributes block: ``name (type): ...``, ``*args (...)``, or ``NAME: ...`` for an enum member.
_ENTRY = re.compile(r"^(\*{0,2}[A-Za-z_][A-Za-z0-9_]*)(?: \(|:)")


def _sections(doc: str | None) -> dict[str, list[str]]:
    """The sections of a docstring: each header and the lines at the first indentation under it, its entries."""
    found: dict[str, list[str]] = {}
    current: str | None = None
    header_indent = 0
    entry_indent: int | None = None
    for line in inspect.cleandoc(doc or "").splitlines():
        match = _HEADER.match(line)
        if match:
            current, header_indent, entry_indent = match.group(2), len(match.group(1)), None
            found.setdefault(current, [])
            continue
        if current is None or not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if indent <= header_indent:
            current = None
            continue
        if entry_indent is None:
            entry_indent = indent
        if indent == entry_indent:
            found[current].append(line.strip())
    return found


def _named(entries: list[str]) -> set[str]:
    return {match.group(1) for entry in entries if (match := _ENTRY.match(entry))}


def _parameters(function: Any) -> list[str]:
    """The parameters a caller passes, as an Args block names them: ``*args`` and ``**kwargs`` with their stars."""
    names = []
    for parameter in inspect.signature(function).parameters.values():
        if parameter.name in ("self", "cls"):
            continue
        prefix = {inspect.Parameter.VAR_POSITIONAL: "*", inspect.Parameter.VAR_KEYWORD: "**"}.get(parameter.kind, "")
        names.append(prefix + parameter.name)
    return names


def _returns_something(function: Any) -> bool:
    annotation = inspect.signature(function).return_annotation
    return annotation not in ("None", None, "NoReturn", "Never") and getattr(function, "__name__", "") != "__init__"


def _fields(cls: type) -> list[str]:
    """What an ``Attributes:`` block names: a dataclass's fields, a pydantic model's, an enum's members."""
    if isinstance(cls, enum.EnumMeta):
        return [member.name for member in cls]  # type: ignore[var-annotated]
    if dataclasses.is_dataclass(cls):
        return [field.name for field in dataclasses.fields(cls) if not field.name.startswith("_")]
    model_fields = getattr(cls, "model_fields", None)
    if isinstance(model_fields, dict):
        return [name for name in model_fields if not name.startswith("_")]
    return []


def _members(name: str, cls: type) -> Iterator[tuple[str, Any, str]]:
    """The public methods and properties ``cls`` defines itself, with ``__init__`` where it is written by hand."""
    generated_init = dataclasses.is_dataclass(cls) or isinstance(getattr(cls, "model_fields", None), dict)
    for attr, raw in vars(cls).items():
        if attr.startswith("_") and attr != "__init__":
            continue
        if attr == "__init__" and (generated_init or issubclass(cls, BaseException)):
            continue
        if isinstance(raw, property):
            yield f"{name}.{attr}", raw, "property"
            continue
        function = raw.__func__ if isinstance(raw, (staticmethod, classmethod)) else raw
        if attr == "__init__" and getattr(function, "__module__", None) != cls.__module__:
            continue  # one Python wrote, such as a Protocol's
        if inspect.isfunction(function):
            yield f"{name}.{attr}", function, "method"


def _targets() -> Iterator[tuple[str, Any, str]]:
    for name in willy.__all__:
        value = getattr(willy, name)
        if inspect.isclass(value):
            yield name, value, "class"
            yield from _members(name, value)
        elif inspect.isfunction(value):
            yield name, value, "function"


def _problems() -> list[str]:
    problems: list[str] = []
    for label, value, kind in _targets():
        doc = inspect.getdoc(value) if kind != "method" else value.__doc__
        if not (doc or "").strip() and not label.endswith(".__init__"):
            problems.append(f"{label}: no docstring")
            continue
        sections = _sections(doc)
        if kind == "property":
            continue
        if kind == "class":
            fields = _fields(value)
            if fields:
                named = _named(sections.get("Attributes", []))
                missing = [field for field in fields if field not in named]
                stale = sorted(named - set(fields))
                if missing:
                    problems.append(f"{label}: Attributes does not name {', '.join(missing)}")
                if stale:
                    problems.append(f"{label}: Attributes names {', '.join(stale)}, which it does not have")
            continue
        parameters = _parameters(value)
        if label.endswith(".__init__"):
            # Google style lets a constructor's arguments live in the class's docstring as well.
            owner = getattr(willy, label.split(".")[0])
            named = _named(sections.get("Args", [])) | _named(_sections(inspect.getdoc(owner)).get("Args", []))
        else:
            named = _named(sections.get("Args", []))
        missing = [parameter for parameter in parameters if parameter not in named]
        stale = sorted(named - set(parameters))
        if missing:
            problems.append(f"{label}: Args does not name {', '.join(missing)}")
        if stale:
            problems.append(f"{label}: Args names {', '.join(stale)}, which it does not take")
        if _returns_something(value) and not ({"Returns", "Yields"} & set(sections)):
            problems.append(f"{label}: no Returns")
    return problems


class EveryPublicNameSaysWhatToPassAndWhatComesBackTests(unittest.TestCase):

    def test_every_function_and_method_names_each_argument_and_what_it_returns(self) -> None:
        problems = _problems()
        self.assertEqual([], problems, f"{len(problems)} docstring(s) to write:\n" + "\n".join(problems))

    def test_the_reader_finds_a_google_block_and_its_entries(self) -> None:
        def example(points: Any, num_vectors: int = 8, *rest: Any, **options: Any) -> Any:
            """Finds the most prominent displacement vectors.

            Args:
                points (np.ndarray): A 2D array of lattice points of shape (N, 2).
                num_vectors (int): The number of basis vectors to find (default: 8).
                    A second line of the same entry.
                *rest (Any): More points.
                **options (Any): Anything else.

            Returns:
                Tuple[np.ndarray, dict]:
                    - basis_vectors (np.ndarray): The estimated basis vectors.
            """
            return None

        sections = _sections(example.__doc__)
        self.assertEqual({"points", "num_vectors", "*rest", "**options"}, _named(sections["Args"]))
        self.assertEqual(["points", "num_vectors", "*rest", "**options"], _parameters(example))
        self.assertIn("Returns", sections)
        self.assertTrue(_returns_something(example))


if __name__ == "__main__":
    unittest.main()
