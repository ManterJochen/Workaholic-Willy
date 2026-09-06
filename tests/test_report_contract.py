"""The calling convention, enforced: `render()` obeys its three rules and the sentinels agree.

⭐⭐ **THIS GUARD DISCOVERS, IT DOES NOT LIST.** An enumerated allow-list of conforming classes was
the first design and it was rejected for the same reason
`tests/test_grasping_wiring_guard.py:24-27` gives against one: *"deliberately no allow-list of
known-unwired flags, an empty exception list is only honest because it was measured."* A hand-kept
list of nouns would go stale silently, which is the exact shape the contract exists to prevent. So
the scan walks the tree and finds every `render` method itself.

⚠ **THE TREE-WIDE SCAN PARSES; THE POINTED CHECKS IMPORT.** Importing every module to find its
`render` methods would pull torch through several of them and cost minutes, and a guard nobody runs
is a guard that does not exist, so `ast` answers the three questions that matter (how many
arguments, is the printed text ASCII, does it end on a newline) without executing a line.

⛔ **BUT WHERE A NAMED THING CAN BE IMPORTED, IT IS IMPORTED, and this was the other way round for an
hour.** The duplicate-sentinel check parsed both files, as a workaround for `deep/train/plan.py`
pulling torch. The deep session removed that at the source once this test named it, and made the
better argument for dropping the workaround: **a test that reads source passes on the defect and
fails on the repair** whenever the fixing comment quotes the old text. Their branch hit that trap
four times in one day. Parse to FIND things; import to check what they DO.

⛔ **THE FLOOR IS PART OF THE GUARD.** A test that asserts over an empty set passes loudest of all.
`_MIN_RENDERERS` is measured, not guessed, and it fails if the scan stops finding things.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path
from typing import Any

from src.contracts import UNSET, Rendered, Structured, chosen, resolve

_REPO = Path(__file__).resolve().parent.parent

#: The trees this convention governs. `deep/` and `willy_sim/` are excluded by owner decision while
#: they are reorganised elsewhere; `datagen/` is a separate top-level tool with its own config tree.
_ROOTS = (
    _REPO / "src" / "config",
    _REPO / "src" / "robot",
    _REPO / "src" / "contracts",
)

_EXCLUDED = ("/deep/", "\\deep\\", "/willy_sim/", "\\willy_sim\\")

#: MEASURED on 2026-09-04, not guessed. Four `render()` methods on report-shaped classes existed
#: before this package (config/explain.py:132, ur/io_bench.py:62 and :96,
#: real_cell/preflight.py:65). The floor sits at four so a scan that silently stops matching fails
#: here instead of passing quietly.
_MIN_RENDERERS = 4


def _python_files() -> list[Path]:
    found: list[Path] = []
    for root in _ROOTS:
        for path in root.rglob("*.py"):
            text = str(path)
            if not any(marker in text for marker in _EXCLUDED):
                found.append(path)
    return found


def _render_methods() -> list[tuple[Path, str, ast.FunctionDef]]:
    """Every `def render(...)` defined inside a class, with the class it belongs to."""
    out: list[tuple[Path, str, ast.FunctionDef]] = []
    for path in _python_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover - a broken file is another test
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "render":
                    out.append((path, node.name, item))
    return out


def _required_arguments(func: ast.FunctionDef) -> list[str]:
    """The arguments a caller MUST supply, beyond `self`. Empty means `x.render()` works.

    ⛔ `kw_defaults` IS PARALLEL TO `kwonlyargs`, not a compact list of the defaults that exist: it
    holds `None` in the slot of every keyword-only argument without one. Reading it as a length is
    the bug this function was first written with, and the case list in
    `test_this_guard_rejects_the_shapes_it_is_meant_to` is what caught it.
    """
    args = func.args
    positional = [a.arg for a in (args.posonlyargs + args.args)][1:]  # drop self
    if args.defaults:
        positional = positional[:-len(args.defaults)]
    keyword = [a.arg for a, default in zip(args.kwonlyargs, args.kw_defaults) if default is None]
    return positional + keyword


def _body(func: ast.FunctionDef) -> list[ast.stmt]:
    """The statements, with a leading docstring dropped if there is one.

    ⚠ THE DOCSTRING IS DETECTED, NEVER ASSUMED. The first draft wrote `func.body[1:] or func.body`,
    which is correct only for a one-statement body: a two-statement function with no docstring would
    have had its first statement silently dropped and compared as if it were not there.
    """
    body = func.body
    first = body[0] if body else None
    has_doc = (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
               and isinstance(first.value.value, str))
    return body[1:] if has_doc else body


def _printed_strings(func: ast.FunctionDef) -> list[str]:
    """Every string literal in the body except the docstring.

    The docstring is exempt because it is read, never encoded to the terminal. That is the whole
    reason the decorated markers in this repository live in docstrings and not in `help=` strings.
    """
    return [n.value for n in ast.walk(ast.Module(body=_body(func), type_ignores=[]))
            if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def _trailing_newlines(func: ast.FunctionDef) -> list[str]:
    r"""Every literal in the body that could actually END the rendered text with a newline.

    ⛔⛔ **THE NEWLINE GUARD'S FIRST REAL FINDING WAS A FALSE ALARM, AND THE REPAIR HAD TWO BUGS OF
    ITS OWN.** MEASURED 2026-09-04: `backend/config/tree.py` returns ``f"config error:\n{self.error}"``.
    An f-string is a `JoinedStr` whose constant pieces are separate nodes, so the old check saw
    ``'config error:\n'``, flagged it, and was wrong: the value cannot end there, because the last
    component is a `FormattedValue`. The method's docstring said "no trailing newline" and it was right.

    Two things then went wrong in the fix, both caught by the case list below rather than by reading:

    1. `ast.walk` yields a `JoinedStr` AND each of its inner constants, so checking "the last piece"
       while also checking every constant put the interior pieces straight back in. The inner nodes
       are collected and skipped explicitly.
    2. A bare ``"\n"`` is excluded as a standalone literal, because ``"\n".join(lines)`` is the house
       idiom and flagging it would condemn every correct renderer. But the LAST piece of an f-string
       is a different thing: in ``f"a {x}\n"`` that same ``"\n"`` really is a trailing newline. So the
       exclusion applies to standalone constants only.

    ⚠ AND THE BLIND SPOT, NAMED RATHER THAN DISCOVERED LATER: a `JoinedStr` ending in a
    `FormattedValue` is UNKNOWN, so ``f"{x}"`` where `x` itself ends in a newline still passes. That
    is the honest limit of a parse, and the alternative cries wolf on correct code, which is worse.
    """
    body = ast.Module(body=_body(func), type_ignores=[])
    inner: set[int] = set()
    joined_tails: list[str] = []
    for node in ast.walk(body):
        if not isinstance(node, ast.JoinedStr):
            continue
        for piece in node.values:
            inner.add(id(piece))
        last = node.values[-1] if node.values else None
        if isinstance(last, ast.Constant) and isinstance(last.value, str):
            joined_tails.append(last.value)

    standalone = [
        n.value for n in ast.walk(body)
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in inner
    ]
    return (
        [t for t in joined_tails if t.endswith("\n")]
        + [t for t in standalone if t.endswith("\n") and t != "\n"]
    )


def _classes_defining(*methods: str) -> dict[str, tuple[Path, str]]:
    """PARSED, not imported: every class in scope that defines each of `methods` by name.

    ⚠ THIS FILE'S OWN RULE IS "parse to FIND things, import to check what they DO", and the reason is
    in its docstring: importing every module here would pull torch through several of them and cost
    minutes, and a guard nobody runs is a guard that does not exist. So discovery parses, and only
    the two chosen exemplars are imported.
    """
    out: dict[str, tuple[Path, str]] = {}
    for path in _python_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            defined = {n.name for n in node.body if isinstance(n, ast.FunctionDef)}
            if all(m in defined for m in methods):
                out[node.name] = (path, node.name)
    return out


def _import_class(path: Path, name: str) -> type:
    """Import one named class from one file. The pointed check, after the parse has found it."""
    import importlib

    module = importlib.import_module(".".join(path.relative_to(_REPO).with_suffix("").parts))
    obj = getattr(module, name)
    assert isinstance(obj, type)
    return obj


class RenderTakesNoArgumentsTests(unittest.TestCase):
    """⛔ A renderer with options has two outputs, and then the CLI prints one and the console the
    other. That is the drift the contract exists to stop, so the rule is structural rather than
    advisory."""

    def test_every_render_is_callable_with_nothing_but_self(self) -> None:
        offenders = []
        for path, cls, func in _render_methods():
            missing = _required_arguments(func)
            if missing:
                offenders.append(f"{path.relative_to(_REPO)}::{cls}.render{missing}")
        self.assertEqual(offenders, [], "render() must be callable with no arguments")

    def test_this_guard_rejects_the_shapes_it_is_meant_to(self) -> None:
        """⛔⛔ THE CONTROL THAT FOUND A HOLE IN ITS OWN GUARD. The first version of
        `_required_arguments` read `kwonlyargs[len(kw_defaults):]`, on the assumption that
        `kw_defaults` holds only the defaults that exist. It does not: it is PARALLEL to
        `kwonlyargs` and carries `None` for every argument without one. So the slice was always
        empty and `render(self, *, style)` was accepted by a guard written to reject exactly that.

        A conformance test without this case would have passed, and the repository would have had a
        fourth green-but-inert guard.
        """
        cases = {
            "def render(self): ...": [],
            "def render(self, verbose): ...": ["verbose"],
            "def render(self, verbose=False): ...": [],
            "def render(self, *, style): ...": ["style"],
            "def render(self, *, style='a'): ...": [],
            "def render(self, *, a, b='x', c): ...": ["a", "c"],
        }
        for source, expected in cases.items():
            with self.subTest(source):
                func = next(n for n in ast.walk(ast.parse(source))
                            if isinstance(n, ast.FunctionDef))
                self.assertEqual(_required_arguments(func), expected)

    def test_the_scan_actually_found_something(self) -> None:
        """A test that asserts over an empty set passes loudest of all."""
        self.assertGreaterEqual(len(_render_methods()), _MIN_RENDERERS)


class RenderedTextIsTerminalSafeTests(unittest.TestCase):
    """⛔ This terminal is cp1252. `tests/test_cli_help_is_ascii.py:3-9` records the same defect
    twice: one decorated glyph in printed text raises `UnicodeEncodeError` instead of printing."""

    def test_no_render_body_carries_a_non_ascii_literal(self) -> None:
        offenders = []
        for path, cls, func in _render_methods():
            for text in _printed_strings(func):
                if not text.isascii():
                    bad = "".join(sorted({c for c in text if not c.isascii()}))
                    offenders.append(f"{path.relative_to(_REPO)}::{cls}.render carries {bad!r}")
        self.assertEqual(offenders, [], "rendered text is printed to a cp1252 console")

    def test_no_render_body_ends_its_text_with_a_newline(self) -> None:
        """The caller owns the line break, because a caller nesting the text cannot remove one it
        did not ask for."""
        offenders = []
        for path, cls, func in _render_methods():
            if _trailing_newlines(func):
                offenders.append(f"{path.relative_to(_REPO)}::{cls}.render ends on a newline")
        self.assertEqual(offenders, [], "render() must not append a trailing newline")

    def test_the_newline_guard_still_catches_and_no_longer_cries_wolf(self) -> None:
        """⭐ THE SELF-FAILING CONTROL, ADDED AFTER THE GUARD'S FIRST REAL FINDING WAS A FALSE ALARM.

        Both directions are pinned: what it must still catch, and the exact shape that fooled it. A
        guard whose blind spot is written down is worth more than one whose reach is assumed.
        """
        cases: list[tuple[str, bool]] = [
            ('def render(self):\n    return "a\\n"', True),
            ('def render(self):\n    out = []\n    out.append("a\\n")\n    return "".join(out)', True),
            # An f-string whose LAST piece is a constant newline is still an offender.
            ('def render(self):\n    return f"a {self.x}\\n"', True),
            # The false alarm: a newline INSIDE an f-string whose last piece is the substitution.
            ('def render(self):\n    return f"config error:\\n{self.error}"', False),
            ('def render(self):\n    return "\\n".join(self.lines)', False),
            ('def render(self):\n    return f"{self.a}: {self.b}"', False),
        ]
        for source, should_flag in cases:
            with self.subTest(source=source):
                func = ast.parse(source).body[0]
                assert isinstance(func, ast.FunctionDef)
                flagged = bool(_trailing_newlines(func))
                self.assertEqual(flagged, should_flag, source)


class ProtocolsAreNotVacuousTests(unittest.TestCase):
    """⛔ THE MEASUREMENT THAT DECIDED THE DESIGN. A single protocol requiring `render()` AND
    `to_dict()` would have been satisfied by ZERO classes on the day it shipped. The two halves are
    disjoint in practice, so they are two protocols that compose."""

    def test_rendered_has_real_members(self) -> None:
        from src.robot.execution.real_cell.preflight import PreflightReport

        self.assertIsInstance(PreflightReport.__new__(PreflightReport), Rendered)

    def test_structured_has_real_members(self) -> None:
        from src.robot.grasping.decision import DecisionReport

        self.assertIsInstance(DecisionReport.__new__(DecisionReport), Structured)

    def test_the_two_halves_are_genuinely_disjoint(self) -> None:
        """The evidence, kept executable AND kept from going stale.

        ⛔⛔ **THIS TEST NAMED `PreflightReport` AS THE HUMAN-ONLY EXEMPLAR AND WENT RED WHEN THAT
        CLASS GOT BETTER.** `7a24dbe` gave it a `to_dict()`, because the only serialisation of a
        preflight lived in a FastAPI router and a library caller had to import fastapi to get a
        checklist as data. So it satisfies both halves now, exactly as intended, and a test that
        named it was asserting the absence of an improvement.

        ⚠ SAME SHAPE AS THE SENTINEL ASSERTION EARLIER THE SAME DAY, which demanded that two `UNSET`
        objects stay separate and failed the moment they were unified. Both were tests that forbade
        the outcome they were written to enable. The answer both times is to assert the PROPERTY
        rather than an instance of it.

        So the exemplars are DERIVED. As the convention lands, class after class gains both halves;
        that is progress, and this stays green while ANY class still sits on each side. It turns red
        only when one side empties, which is the day the two protocols really would be one, and that
        is a finding worth being told about rather than a breakage.
        """
        renders = _classes_defining("render")
        serialises = _classes_defining("to_dict")
        human_only = {k: v for k, v in renders.items() if k not in serialises}
        machine_only = {k: v for k, v in serialises.items() if k not in renders}

        self.assertTrue(
            human_only,
            "no class renders without serialising any more; the two protocols have collapsed into "
            "one and `contracts/reporting.py` should say so instead of pretending otherwise",
        )
        self.assertTrue(machine_only, "no class serialises without rendering any more")

        # ⭐ AND THE PROTOCOLS MUST ACTUALLY SEE THE SPLIT, not merely the source scan: a `to_dict`
        # inherited from a base or supplied by a decorator is real and a parse cannot see it. So one
        # exemplar per side is IMPORTED and checked against the runtime protocols.
        human = _import_class(*human_only[sorted(human_only)[0]])
        machine = _import_class(*machine_only[sorted(machine_only)[0]])
        self.assertIsInstance(human.__new__(human), Rendered)
        self.assertNotIsInstance(human.__new__(human), Structured)
        self.assertIsInstance(machine.__new__(machine), Structured)
        self.assertNotIsInstance(machine.__new__(machine), Rendered)

    def test_the_guard_can_fail(self) -> None:
        """⭐ THE SELF-FAILING CONTROL. A conformance check that cannot reject anything is decoration,
        and this repository has shipped three green-but-inert guards already."""

        class NotAReport:
            pass

        self.assertNotIsInstance(NotAReport(), Rendered)
        self.assertNotIsInstance(NotAReport(), Structured)


class UnsetSemanticsTests(unittest.TestCase):
    """⛔⛔ THE LANDMINE THIS GUARD EXISTS FOR. `UNSET` is defined twice while `deep/` is being
    reorganised elsewhere. Two sentinels that disagree about truthiness would make the same
    expression mean different things depending on which module the value came from."""

    def test_none_is_a_chosen_value(self) -> None:
        """The whole reason the sentinel is not `None`: `train_units=None` means every unit,
        `profile=None` means the base tree. Both are choices."""
        self.assertTrue(chosen(None))
        self.assertTrue(chosen(0))
        self.assertTrue(chosen(False))
        self.assertFalse(chosen(UNSET))

    def test_resolve_is_explicit_then_config_then_default(self) -> None:
        self.assertEqual(resolve("x", 7, 99, 5), 7)
        self.assertEqual(resolve("x", UNSET, 99, 5), 99)
        self.assertEqual(resolve("x", UNSET, UNSET, 5), 5)
        self.assertIsNone(resolve("x", UNSET, None, 5))

    def test_chosen_tests_the_TYPE_not_the_identity(self) -> None:
        """⛔⛔ THE PIN ON A FIX mypy FOUND, and the only runtime difference between the two forms.

        `chosen` was `value is not UNSET`, which is correct at run time and narrows NOTHING for a
        type checker, because `_Unset` is a plain class rather than a singleton mypy can reason
        about. The first real call site failed on it::

            backend/config/loader.py:172: error: Argument 2 to "_validated_chain" has incompatible
            type "str | _Unset | None"; expected "str | None"  [arg-type]

        It is a `TypeGuard` over `isinstance` now. A SECOND instance of `_Unset` is the one input
        that tells the two implementations apart: `is` would call it chosen, `isinstance` does not.
        Reverting to `is` would pass every other test in this file and fail this one.
        """
        from src.contracts.options import _Unset

        self.assertFalse(chosen(_Unset()), "chosen() is testing identity again")
        self.assertFalse(chosen(UNSET))

    def test_resolve_refuses_rather_than_returning_none(self) -> None:
        """Falling off the end is a programming error. A returned `None` would be carried into a
        plan and fail somewhere unrelated with no trace of which setting was missing."""
        with self.assertRaises(ValueError) as caught:
            resolve("epochs", UNSET, UNSET)
        self.assertIn("epochs", str(caught.exception))

    def test_there_is_exactly_one_sentinel_and_every_tree_uses_it(self) -> None:
        """⭐⭐ THE FOLLOW-UP LANDED, AND THIS TEST FAILING IS HOW I FOUND OUT.

        It used to assert ``assertIsNot(UNSET, DEEP_UNSET, "still two objects; unifying them is the
        follow-up")``, which was true while there were four private copies of the sentinel: this
        one, `execution/autonomous_grasp/service.py:188`, `deep/train/plan.py` and `datagen/api.py`.
        On 2026-09-04 the session that owns the last two folded them in, and the session that owns
        the first two folded in `service.py`, which was the ORIGINAL: the convention's rule was read
        off that line.

        So the test went red because the thing it guarded got FIXED, which is the good failure. It
        now asserts the opposite, and that is not a weaker claim. The copies were only ever kept
        identical so that unifying them WOULD be one import line, and a test that still demanded two
        objects would forbid the outcome it was written to enable.

        ⭐ THE SWAP PAID FOR ITSELF IN THE OTHER TREE WITHIN THE MINUTE. Giving the name a type made
        mypy find two sites a rename had missed (`chosen.get(name, fallback)` and
        `if "slots" in chosen:`), both green for as long as the sentinel was untyped. That is the
        whole argument for `chosen()` being a `TypeGuard` rather than an `is` comparison, arriving
        as evidence instead of as an opinion.

        ⚠ WHAT THIS GUARDS NOW is a fifth copy appearing. Identity is the only assertion that can
        see one: two sentinels that behave identically pass every `repr` and `bool` check ever
        written, which is exactly why this file spent a day asserting behaviour and still could not
        tell whether the trees had converged.
        """
        from datagen.api import UNSET as DATAGEN_UNSET
        from src.robot.grasping.deep.train.plan import UNSET as DEEP_UNSET

        self.assertIs(UNSET, DEEP_UNSET, "deep/ grew its own sentinel back")
        self.assertIs(UNSET, DATAGEN_UNSET, "datagen/ grew its own sentinel back")

    def test_no_module_defines_a_private_sentinel_of_its_own(self) -> None:
        """⛔ THE IDENTITY CHECK ABOVE CANNOT SEE A COPY NOBODY IMPORTS, and that is how all four
        started: each was written where it was needed, by someone who had not looked for an existing
        one. So this sweeps the source instead of the import graph.

        ⚠ A test over an empty set passes loudest, so it asserts the canonical definition IS found.
        Without that, deleting `contracts/options.py` would turn this green.
        """
        import re
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        canonical = root / "src" / "contracts" / "options.py"
        pattern = re.compile(r"^\s*(class\s+_?Unset\b|_?UNSET\s*:\s*Any\s*=\s*object\(\))", re.M)

        offenders: list[str] = []
        found_canonical = False
        for tree in ("src", "datagen", "api"):
            for path in (root / tree).rglob("*.py"):
                if pattern.search(path.read_text(encoding="utf-8", errors="ignore")) is None:
                    continue
                if path == canonical:
                    found_canonical = True
                else:
                    offenders.append(str(path.relative_to(root)))

        self.assertTrue(found_canonical, "the canonical sentinel is gone; this sweep proves nothing")
        self.assertEqual(
            offenders, [], "a private sentinel came back; import UNSET from backend.src.contracts"
        )


class NoWrapperDeclaresACalleesDefaultTests(unittest.TestCase):
    """⛔⛔ A DEFAULT IN A WRAPPER IS A CLAIM ABOUT WHAT THE WRAPPED THING DOES.

    MEASURED on 2026-09-04, in code written the same day:

        PolicyTraining declared `seed = 0`; every trainer declares `seed = 1`.
        The same nominal call produced a DIFFERENT ARTIFACT SHA256.

        Cell declared `prompt = "an object"`; `build_real_cell` declares `"object"`.
        The same nominal call produced a DIFFERENT VISION PROMPT.

    Neither raises. Neither fails a test. Neither is visible to ruff or mypy, because both sides are
    valid values of the right type, and the wrong one is exactly as plausible as the right one. The
    deep session found NINE more of these in its own fresh code within the hour, all of which AGREED
    with the callee that day.

    ⚠ AGREEMENT TODAY IS NOT THE PROPERTY WORTH TESTING, which is why this asserts the wrapper
    declares NOTHING rather than that the two values match. A test that compared them would pass on
    the day of the drift as readily as before it.
    """

    #: (wrapper callable, parameter, callee callable). Each pair is a place where a value passes
    #: straight through, so exactly one of the two may declare what it is when nobody says.
    def _pairs(self) -> list[tuple[str, Any, str, Any]]:
        from src.robot.execution.autonomous_grasp import build_real_cell
        from src.robot.execution.cell import Cell
        from src.robot.grasping.generation.support_footprint import (
            generate_support_footprint_grasps,
        )
        from src.robot.grasping.scene import Scene

        return [
            ("Cell.prompt", Cell, "prompt", build_real_cell),
            ("Scene.grasps.max_candidates", Scene.grasps, "max_candidates",
             generate_support_footprint_grasps),
            ("Scene.grasps.palm_aware", Scene.grasps, "palm_aware",
             generate_support_footprint_grasps),
        ]

    def test_the_wrapper_declares_unset_and_the_callee_declares_the_value(self) -> None:
        import inspect

        from src.contracts import UNSET

        for label, wrapper, name, callee in self._pairs():
            with self.subTest(label):
                mine = inspect.signature(wrapper).parameters[name].default
                theirs = inspect.signature(callee).parameters[name].default
                self.assertIs(mine, UNSET, f"{label} declares a default the callee already owns")
                self.assertIsNot(theirs, inspect.Parameter.empty,
                                 f"{label}: nobody declares this, so an omitted value has no answer")

    def test_the_check_can_actually_fail(self) -> None:
        """⭐ THE SELF-FAILING CONTROL, on a synthetic pair rather than on live code, so it keeps
        working after every real pair is fixed."""
        import inspect

        from src.contracts import UNSET

        def callee(*, size: int = 12) -> int:
            return size

        def bad_wrapper(*, size: int = 12) -> int:  # a copy of the callee's default
            return callee(size=size)

        self.assertIsNot(inspect.signature(bad_wrapper).parameters["size"].default, UNSET)
        self.assertEqual(inspect.signature(callee).parameters["size"].default, 12)


class DeliberateDuplicatesStayIdenticalTests(unittest.TestCase):
    """⛔ ONE NAME IS STILL DEFINED TWICE ON PURPOSE, and it was two this morning.

    A duplicate nobody is watching is how two halves of a repository start disagreeing quietly, so
    each one gets an assertion rather than a comment, and the assertion is deleted when the duplicate
    is. That has now happened once: `DEEP_GENERATOR_LOG_FILE` was resolved to a single definition
    within the hour (see the note below), and its guard went with it rather than lingering as a test
    comparing a name against itself.

    What remains is the `_Unset` pair. It stays duplicated for as long as `deep/` is reorganised in
    another session, and it is kept behaviourally identical so unifying it is one import line.
    """

    # ⛔ `test_the_deep_generator_log_file_name_agrees` WAS DELETED HERE ON 2026-09-04, and the
    # duplicate it guarded is gone rather than merely agreeing. `DEEP_GENERATOR_LOG_FILE` now has
    # ONE home, `grasping/constants.py`, and `deep/calculator.py` reads it there; the copy in
    # `deep/log_files.py` was removed and so was that module, whose only remaining job was to define
    # a symbol another module owned.
    #
    # ⚠ COLLAPSING TO A RE-EXPORT WOULD HAVE BEEN THE WRONG FIX. `deep/log_files.py` could have kept
    # its name and imported the constant, which is what the note in `grasping/constants.py`
    # suggested. That leaves two addresses for one symbol, and this package has just measured what
    # that costs: `deep/net/__init__.py` re-exported thirteen names and imported four modules at
    # package level, so importing anything beneath it dragged a retired model family into memory and
    # made those four files undeletable.
    #
    # The guard below is the one worth keeping, and it is unaffected: it asserts that the stable
    # tree never imports the moving one, which is the regression that motivated the whole exchange.



    def test_calculator_factory_does_not_import_the_deep_tree(self) -> None:
        """⚠ THE REGRESSION THIS EXISTS TO CATCH is a helpful re-import, added because the name is
        obviously 'about' the deep package. Read as text, so the check costs nothing and cannot
        itself pull the tree it is checking."""
        source = (_REPO / "src/robot/grasping/calculator_factory.py").read_text(
            encoding="utf-8")
        tree = ast.parse(source)
        top_level = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        offenders = [n.module for n in top_level
                     if isinstance(n, ast.ImportFrom) and n.module and ".deep" in n.module]
        self.assertEqual(offenders, [],
                         "calculator_factory imports the deep tree at module level again")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
