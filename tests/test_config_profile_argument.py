"""Selecting a config profile is an argument, not a mutation of the process.

⛔⛔ **THE DANCE THIS REPLACES.** Choosing an overlay chain used to be possible only by writing
``os.environ["WILLY_PROFILE"]``, so a caller who wanted one tree had to save the old value, set the
new one, load, and restore in a ``finally``. That was written out by hand at nine production sites,
no two quite alike, and one leak cost a whole afternoon: an example invoked with ``--profile ur3e``
left the variable set, and an unrelated safety-guard test three files later began failing while
passing in isolation. That example generation has since been retired, and the rule that replaced
it is asserted in `tests/test_examples_run.py`: nothing under `scripts/examples/` or
`scripts/checks/` may write to `os.environ` at all, because a flag that mutates the environment
is a second way to say what the environment already says.

⭐ **AND ONE OF THEM CLEARED THE WHOLE PROCESS CACHE, TWICE, TO READ ONE VALUE.**
`safety/planning/__main__.py` called ``reload_config()`` before and after its load, so a diagnostic
that only wanted to know which arm model is configured invalidated the cached tree of every other
holder in the process. ``load_config(profile=...)`` needs no reload at all: the cache is keyed per
chain, so two profiles are two entries rather than one entry fought over.

⚠ **``profile=None`` IS A VALUE, ``UNSET`` IS THE ABSENCE.** That distinction is the whole reason the
parameter defaults to :data:`~src.contracts.UNSET`, and it is the one an argparse ``None``
can silently destroy: a CLI that forwards its untyped ``--profile`` straight through would disable
``WILLY_PROFILE`` for every operator who exports it.
"""

from __future__ import annotations

import ast
import os
import pathlib
import unittest
from pathlib import Path

from src.config.loader import (
    ConfigError,
    join_profiles,
    load_config,
    load_robot_config,
    set_active_profile,
)
from src.contracts import UNSET

_REPO = Path(__file__).resolve().parent.parent


#: Where the profile overlays live, so an absent name can be proved absent.
_DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "config"


class _NoProfileInEnvironment:
    """Run a case with ``WILLY_PROFILE`` at a known value and put it back."""

    def __init__(self, value: str | None) -> None:
        self.value = value

    def __enter__(self) -> None:
        self.previous = os.environ.get("WILLY_PROFILE")
        set_active_profile(self.value)

    def __exit__(self, *exc: object) -> None:
        set_active_profile(self.previous)


class TheParameterTests(unittest.TestCase):

    def test_omitting_it_reads_the_environment_exactly_as_before(self) -> None:
        """Every existing call site passes nothing, so this is the byte-identical path."""
        with _NoProfileInEnvironment("sim"):
            self.assertEqual(str(load_config().robot.vendor), "sim")
        with _NoProfileInEnvironment(None):
            self.assertEqual(str(load_config().robot.vendor), "ur")

    def test_an_explicit_chain_does_not_touch_the_environment(self) -> None:
        with _NoProfileInEnvironment(None):
            self.assertEqual(str(load_config(profile="sim").robot.vendor), "sim")
            self.assertIsNone(os.environ.get("WILLY_PROFILE"))

    def test_none_means_the_base_tree_even_when_the_shell_says_otherwise(self) -> None:
        """⛔ THE CASE THE SENTINEL EXISTS FOR. A tool that must not inherit the operator's shell
        asks for `None`; a caller that says nothing gets the shell. Those are different questions
        and `None` can only answer one of them."""
        with _NoProfileInEnvironment("sim"):
            self.assertEqual(str(load_config().robot.vendor), "sim")
            self.assertEqual(str(load_config(profile=None).robot.vendor), "ur")

    def test_a_typo_is_refused_on_the_new_door_too(self) -> None:
        """⛔ FAIL-CLOSED PER LAYER, WHICHEVER DOOR. A layer with no file would otherwise merge as a
        silent no-op and bring the cell up with another robot's geometry. Both routes go through one
        validator rather than the new one going around it."""
        # ⚠ DERIVED, because a hand-picked example of "not a profile" only stays wrong until
        # somebody adds it. This test said `ur3` until 2026-09-09, when a `robot.ur3.yaml` landed
        # and the negative control quietly became a positive case. The name below is built from
        # the profile directory and asserted absent from it, so no future arm can repeat that.
        absent = "no_such_profile_" + "x" * 3
        available = {p.stem.split(".", 1)[1] for p in _DATA_DIR.glob("*/*.*.yaml")}
        self.assertNotIn(absent, available, "pick a name that is not a shipped profile")
        for bad in (absent, f"sim,{absent}", "nonsense"):
            with self.subTest(bad), self.assertRaises(ConfigError) as caught:
                load_config(profile=bad)
            self.assertIn(bad.split(",")[-1], str(caught.exception))

    def test_the_cache_is_keyed_per_chain(self) -> None:
        """Two profiles are two entries. This is what removes the need to `reload_config()` around
        a profile switch, which used to invalidate every other holder in the process."""
        with _NoProfileInEnvironment(None):
            base, sim = load_config(profile=None), load_config(profile="sim")
            self.assertIsNot(base, sim)
            self.assertIs(load_config(profile="sim"), sim)

    def test_join_profiles_builds_a_chain_the_parameter_accepts(self) -> None:
        with _NoProfileInEnvironment(None):
            self.assertEqual(str(load_config(profile=join_profiles("sim")).robot.vendor), "sim")

    def test_passing_unset_explicitly_is_the_same_as_omitting_it(self) -> None:
        """⚠ THE PROPERTY EVERY FORWARDING CALL SITE DEPENDS ON. A CLI translates its untyped
        ``--profile`` to ``UNSET`` and passes it on, so ``UNSET`` reaching this function has to mean
        exactly what an absent argument means. If those two ever diverge, every collapsed site
        quietly stops honouring ``WILLY_PROFILE``."""
        with _NoProfileInEnvironment("sim"):
            self.assertIs(load_config(profile=UNSET), load_config())
            self.assertEqual(str(load_config(profile=UNSET).robot.vendor), "sim")


class TheRobotHelperTests(unittest.TestCase):
    """⛔ FOUR SITES WROTE THIS OUT BY HAND, three of them character for character."""

    def test_it_returns_the_robot_block(self) -> None:
        with _NoProfileInEnvironment(None):
            self.assertEqual(str(load_robot_config(profile="sim").vendor), "sim")

    def test_it_raises_config_error_not_system_exit(self) -> None:
        """⛔ ``SystemExit`` FROM LIBRARY CODE READS AS AN EXIT REQUEST, NOT AS A FAULT, and it is
        invisible to a caller that catches exceptions properly. Two CLIs then caught `SystemExit`
        and nothing else, so a genuinely broken tree reached the terminal as a traceback rather than
        as the refusal they were written to print."""
        self.assertFalse(issubclass(ConfigError, SystemExit))
        with self.assertRaises(ConfigError):
            load_robot_config(profile="nonsense")


class TheDanceDoesNotComeBackTests(unittest.TestCase):
    """⚠ NAMED MODULES, NOT AN ALLOW-LIST. These are the four that were collapsed; the assertion is
    that they stay collapsed. The config package's own tooling and `api/cell.py` still set the
    variable deliberately, for a SCOPE rather than around a single load, and are not this test's
    business."""

    COLLAPSED = (
        "src/robot/execution/real_cell/__main__.py",
        "src/robot/execution/real_cell/calibrate.py",
        "src/robot/drivers/ur/__main__.py",
        "src/robot/safety/planning/__main__.py",
    )

    @staticmethod
    def _mutates_profile(path: Path) -> list[str]:
        """Assignments to ``os.environ[...WILLY_PROFILE...]`` and calls to ``set_active_profile``.

        Read from the AST, so the prose in a docstring explaining what USED to be here does not
        count as the thing itself. A substring search would fail on the repair and pass on the
        defect, which is the trap this branch has hit repeatedly.
        """
        tree = ast.parse(path.read_text(encoding="utf-8"))
        found: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) \
                    and node.value.attr == "environ" \
                    and "WILLY_PROFILE" in ast.dump(node.slice):
                found.append("os.environ[WILLY_PROFILE]")
            if isinstance(node, ast.Call):
                func = node.func
                name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
                if name in ("set_active_profile", "reload_config"):
                    found.append(f"{name}()")
        return found

    def test_none_of_them_mutates_process_state_any_more(self) -> None:
        offenders = {rel: found for rel in self.COLLAPSED
                     if (found := self._mutates_profile(_REPO / rel))}
        self.assertEqual(offenders, {}, "the profile dance is back")

    def test_the_scan_can_actually_see_the_dance(self) -> None:
        """⭐ THE SELF-FAILING CONTROL. A scan that cannot find the shape would pass on every file,
        including one that still has it. `api/cell.py` deliberately still calls
        `set_active_profile`, which makes it the honest positive fixture."""
        self.assertNotEqual(self._mutates_profile(_REPO / "api/cell.py"), [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
