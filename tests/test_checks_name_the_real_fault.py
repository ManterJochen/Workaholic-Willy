"""The four `scripts/checks/` files must name the fault that is actually there, and the fix that exists.

⭐ **A CHECK IS THE ONE FILE IN THIS REPOSITORY WHOSE OUTPUT IS AN INSTRUCTION.** An example prints
what happened and a test prints a verdict; a check prints "NOT READY" plus a `fix:` line, and an
operator on a bring-up does what that line says. So a wrong `fix:` is not cosmetic. It is the check
spending somebody's afternoon.

Three ways that can be wrong, measured on 2026-09-10, one test each:

* **A fix naming a profile that does not exist.** Ten sites across the checks and the examples
  advise a ``WILLY_PROFILE=`` value. The loader refuses an unknown layer rather than merging it as a
  no-op, which is right, and it means such a line does not merely fail to help: following it turns
  the command into a `ConfigError` traceback. All ten were checked against this tree's own overlays
  and every layer they name exists. The test below derives the answer from the tree rather than
  pinning today's names, so an overlay renamed or dropped tomorrow lands on the advice that quotes
  it.
* **A config refusal reported as a silent controller.** ``connect()`` on the UR driver refuses THREE
  configurations before it opens a socket, and `cell_bringup.py` reported all three as "the
  controller did not answer", sending the reader to a cable when the fault was in their YAML.
* **A sweep printing "no change" about a switch that changes the built cell.** `grasping_switches.py`
  diffs the effective-config snapshot, `grasping.deep_ranker` reaches the ORCHESTRATOR instead, and a
  block the instrument cannot see is indistinguishable from an inert one. That is the general shape
  of the thing: a mechanism that does not fire looks exactly like one with no work to do.
"""

from __future__ import annotations

import re
import subprocess
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_CHECKS = _ROOT / "scripts" / "checks"
_EXAMPLES = _ROOT / "scripts" / "examples"
_DATA = _ROOT / "config"

#: ``WILLY_PROFILE=sim,ur3e`` is a chain of layers, so the value is split on commas before each half
#: is looked for. Trailing punctuation is excluded from the class rather than stripped afterwards.
_PROFILE_ADVICE = re.compile(r"WILLY_PROFILE=([A-Za-z0-9_,]+)")


def _run(script: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run a check the way an operator does, with no profile inherited from this shell."""
    import os

    env = dict(os.environ)
    env.pop("WILLY_PROFILE", None)
    return subprocess.run([sys.executable, str(script), *args], cwd=_ROOT, capture_output=True,
                          text=True, timeout=300, env=env)


class TheAdvisedProfileExistsTests(unittest.TestCase):
    """⛔ **A `fix:` LINE NAMING A LAYER THE LOADER REFUSES COSTS MORE THAN NO FIX LINE AT ALL.**

    The loader does not merge an unknown layer as a no-op, it raises, so an operator who follows
    such a line gets a traceback where they expected help and has no reason to suspect the advice.
    The overlays a tree ships are a moving set: `robot.ur5e.yaml` is a real bench layer here and a
    tree where the base file already IS the UR5e cell has nothing for that layer to say and does not
    carry it.

    So the expectation is derived from `config/**/*.<layer>.yaml` rather than written down. Measured
    2026-09-10: sixteen overlays, ten advised sites across the checks and the examples naming four
    distinct layers (`ur5e`, `ur3e`, `sim`, `ursim`), all four present.
    """

    def _advised_files(self) -> list[Path]:
        return sorted(p for root in (_CHECKS, _EXAMPLES) for p in root.rglob("*")
                      if p.is_file() and p.suffix in {".py", ".sh", ".ps1", ".md"})

    def test_the_tree_has_the_overlays_this_test_reads(self) -> None:
        """Derived, not written down: the sweep must find a real overlay set to compare against.

        Without this the test goes green on an empty glob, which would make it agree with any advice
        at all, including the advice it exists to refuse.
        """
        self.assertGreater(len(self._layers()), 10, "found almost no profile overlays in the tree")

    def _layers(self) -> set[str]:
        return {p.name.split(".")[-2] for p in _DATA.rglob("*.*.yaml")}

    def test_no_check_or_example_advises_a_profile_that_does_not_exist(self) -> None:
        layers = self._layers()
        for path in self._advised_files():
            for advised in _PROFILE_ADVICE.findall(path.read_text(encoding="utf-8")):
                for layer in (part for part in advised.split(",") if part):
                    with self.subTest(file=str(path.relative_to(_ROOT)).replace("\\", "/"),
                                      layer=layer):
                        self.assertIn(
                            layer, layers,
                            f"advises WILLY_PROFILE={advised}, but no '*.{layer}.yaml' overlay "
                            f"exists under config/, so the loader refuses it",
                        )


class TheFaultIsNamedWhereItIsTests(unittest.TestCase):
    """⛔ **THREE PRE-SOCKET CONFIG REFUSALS WERE REPORTED AS A CONTROLLER THAT DID NOT ANSWER.**

    `URRobotArm.connect()` refuses before `self._conn.connect()` on three configurations, each of
    them a value in the operator's own YAML: `safety.payload.enforce` with `mass_kg: 0.0`, a declared
    mass still carrying the `[0, 0, 0]` CoG marker, and `gripper.tool_frame.source: undeclared`. All
    three raise `RobotConnectionError`, which `cell_bringup.py` caught in one handler and rendered as

        NOT READY: the controller did not answer (RobotConnectionError: safety.payload has ...)
          fix: is the controller reachable, is the robot in REMOTE control, ...

    Nothing was ever asked of the controller. The reader is sent to the cabinet, and the shipped tree
    reaches this state by default because `mass_kg: 0.0` is the deliberate "not weighed yet" marker.
    """

    def test_a_refusal_that_never_opened_a_socket_says_so(self) -> None:
        """Run the check on this tree, and only assert while this tree is one that refuses early.

        The condition is read from the config rather than assumed, so a cell that has since weighed
        its tool skips with a sentence instead of opening a socket to 192.168.1.100 and failing on a
        timeout, which would be a different fault reported under this test's name.
        """
        sys.path.insert(0, str(_ROOT))
        from src.config import load_robot_config  # noqa: PLC0415

        robot = load_robot_config(profile=None)
        payload, tool = robot.safety.payload, robot.gripper.tool_frame
        early = (
            (payload.enforce and payload.mass_kg == 0.0)
            or (payload.enforce and payload.mass_kg > 0.0 and all(v == 0.0 for v in payload.cog_mm))
            or tool.source == "undeclared"
        )
        if not early:
            self.skipTest("this tree's payload and tool frame are both declared, so connect() would "
                          "reach the socket and this test would be measuring the network")

        proc = _run(_CHECKS / "cell_bringup.py", "--live")
        output = proc.stdout + proc.stderr
        self.assertNotIn("Traceback", output, output[-1500:])
        self.assertEqual(proc.returncode, 2, f"expected NOT READY, got {proc.returncode}:\n{output}")
        self.assertNotIn("the controller did not answer", output,
                         f"a config refusal was reported as a silent controller:\n{output[-1200:]}")
        self.assertNotIn("is the controller reachable", output,
                         f"the fix line sends the reader to the cable:\n{output[-1200:]}")
        self.assertTrue(
            any(key in output for key in ("safety.payload", "gripper.tool_frame")),
            f"the refusal does not name the YAML key that caused it:\n{output[-1200:]}",
        )


class TheSweepMeasuresEverySwitchTests(unittest.TestCase):
    """⛔ **`deep_ranker` READ "no change" IN ALL FIVE MODES, AND IT CHANGES THE BUILT CELL.**

    `grasping_switches.py` switches each `grasping.*` block on and diffs `EffectiveGraspingConfig`.
    That snapshot is one of TWO places a block reaches: `apply_orchestrator_overlays` is the other,
    and `grasping.deep_ranker` reaches only the second one, as `orchestrator.deep_ranker_context`.
    Measured with the fitted ranker on disk, switching it on loads it (400 trees) and every pick then
    scores every candidate and stamps `deep_ranker_*` telemetry.

    So the sweep printed the same word for "measured, and nothing moved" as for "outside the only
    instrument I have", and closed with "13 block(s) swept". A row that cannot move is not evidence
    of an inert switch, and the two were indistinguishable in the output.

    ⭐ **THE THIRD WORD IS WHY THIS PASSES ON A FRESH CHECKOUT.** The ranker trees are git-ignored
    (`assets/models/grasp_ranker/**/*.json`; the cards beside them are committed), so on a clean
    clone the overlay fail-safes to `None` and `deep_ranker` moves nothing under either instrument.
    That is a fact about the box rather than an inert switch, and the check now prints `unloaded`
    plus the sentence the shipped code logged. What this test forbids is the silent word.
    """

    def test_no_block_reads_no_change_in_every_mode(self) -> None:
        proc = _run(_CHECKS / "grasping_switches.py")
        output = proc.stdout + proc.stderr
        self.assertNotIn("Traceback", output, output[-1500:])
        self.assertIn("deep_ranker", proc.stdout, "the sweep no longer reaches deep_ranker at all")
        inert = [line.split()[0] for line in proc.stdout.splitlines()
                 if line.startswith("  ") and " no change" in line
                 and "reads on" not in line and "runtime" not in line
                 and "unloaded" not in line]
        self.assertEqual(
            inert, [],
            f"{inert} moved nothing in any mode under either instrument and said nothing about "
            f"why, so the sweep measured nothing about them while reporting that it swept "
            f"them:\n{proc.stdout}",
        )
        self.assertEqual(proc.returncode, 0, f"the sweep failed:\n{output[-1500:]}")


if __name__ == "__main__":
    unittest.main()
