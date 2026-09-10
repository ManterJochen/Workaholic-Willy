"""`scripts/checks/safety_guards.py` must interrogate every guard the pipeline actually carries.

⛔ **MEASURED 2026-09-10 ON THIS TREE.** It printed

    6 guard(s), endpoints only
    workspace, joint_limit, ik_quality, self_collision, payload, motion_continuity
    ...
    OK: every wired guard refused its own violation (5 checked)

Six named, five asked, and the sentence says every. `self_collision` was absent from the hand-written
case table and nothing compared that table against the pipeline, so the omission could not surface:
the count beside the verdict was right and nobody reads a count against a sentence.

⭐ **A HAND-KEPT LIST BESIDE A DERIVATION IS THE DEFECT, NOT THE MISSING ENTRY.** Adding a fifth case
would fix today and leave the same hole for the next guard family somebody wires. So the check now
derives its coverage from `preflight.guards` and fails when the pipeline carries a family it has no
case for. What this file pins is that property, not the presence of any particular guard.

The self-collision case itself is derived rather than written down: four of five folded poses tried
against the shipped tree were rejected, so the check searches a small family built from the arm's own
degrees of freedom and reports which pose it used. An arm for which no fold collides is reported as
not exercised rather than counted as passed.
"""

from __future__ import annotations

import re
import subprocess
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_CHECK = _ROOT / "scripts" / "checks" / "safety_guards.py"


def _run(*args: str):
    return subprocess.run(
        [sys.executable, str(_CHECK), *args],
        cwd=_ROOT, capture_output=True, text=True, timeout=300,
    )


class TheCheckCoversWhatThePipelineCarriesTests(unittest.TestCase):
    def test_every_named_guard_is_accounted_for(self) -> None:
        """The property the old version violated, read out of the check's own output.

        The attestation line names the pipeline's families; every one of them has to appear in the
        rows below it, either exercised or explicitly reported as not exercised. Read from the output
        rather than from a list in this file, so a guard family added tomorrow is covered by this
        test the day it lands.
        """
        result = _run()
        if result.returncode == 2:
            self.skipTest(f"no guard pipeline on this tree: {result.stdout.strip()[:120]}")
        self.assertNotIn("Traceback", result.stdout + result.stderr)

        named = self._named_guards(result.stdout)
        self.assertGreater(len(named), 1, f"could not read the attestation line:\n{result.stdout}")
        rows = result.stdout
        missing = [g for g in named if not re.search(rf"^\s+{re.escape(g)}\s", rows, re.M)]
        self.assertEqual(
            missing, [],
            f"the pipeline carries {sorted(named)} and the check says nothing about {missing}",
        )

    def test_the_verdict_counts_what_it_names(self) -> None:
        """The count beside the verdict must equal the number of families it spoke about.

        This is the assertion that would have caught the original defect on its own: the sentence
        said every and the number said five, and the two disagreed in print for a day.
        """
        result = _run()
        if result.returncode == 2:
            self.skipTest("no guard pipeline on this tree")
        named = self._named_guards(result.stdout)
        match = re.search(r"\((\d+) of (\d+) checked\)", result.stdout)
        self.assertIsNotNone(match, f"no count in the verdict:\n{result.stdout}")
        checked, of_total = int(match.group(1)), int(match.group(2))
        self.assertEqual(
            checked, len(named),
            f"verdict counts {checked} of {len(named)} named guards",
        )
        self.assertEqual(of_total, len(named), "the verdict's own denominator is wrong")

    def test_self_collision_is_one_of_them(self) -> None:
        """The instance, kept beside the property.

        The property test above would pass on a tree whose pipeline happens to carry no
        self-collision guard, which is why the family that was actually missed is named once here.
        """
        result = _run()
        if result.returncode == 2:
            self.skipTest("no guard pipeline on this tree")
        if "self_collision" not in self._named_guards(result.stdout):
            self.skipTest("this cell wires no self-collision guard")
        # re.M: assertRegex anchors on the whole string, and every row here is indented.
        self.assertIsNotNone(
            re.search(r"^\s+self_collision\s+\S+", result.stdout, re.M), result.stdout)

    @staticmethod
    def _named_guards(text: str) -> list[str]:
        """The families the attestation printed, which is the pipeline's own account of itself."""
        for line in text.splitlines():
            stripped = line.strip()
            if "," in stripped and "workspace" in stripped and ":" not in stripped:
                return [p.strip() for p in stripped.split(",") if p.strip()]
        return []


if __name__ == "__main__":
    unittest.main()
