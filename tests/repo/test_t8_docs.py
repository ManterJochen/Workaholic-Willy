"""Phase T8 — RED tests for operator documentation.

The T8 deliverable adds the operator quickstart ``docs/Quickstart.md``
plus operator sections in ``src/robot/grasping/README.md``:

* A *mode behaviour table* with eight locked columns.
* A *troubleshooting matrix* (symptom → cause → mode/policy fix).
* One quickstart paragraph per shipped preset, until the presets left on 2026-10-16 (the owner's cleanup).

Robot-level safety triage subset lives in
``src/robot/README.md``.

This is purely a docs phase, so the tests are markdown / config
audits — no runtime behaviour is exercised.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_QUICKSTART = _REPO_ROOT / "docs" / "Quickstart.md"
_GRASPING_README = (
    _REPO_ROOT
    / "src"
    / "robot"
    / "grasping"
    / "README.md"
)
_ROBOT_README = (
    _REPO_ROOT / "src" / "robot" / "README.md"
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class QuickstartFileTests(unittest.TestCase):
    def test_quickstart_exists(self) -> None:
        self.assertTrue(
            _QUICKSTART.is_file(),
            f"missing the operator quickstart at {_QUICKSTART}",
        )

    def test_quickstart_drops_phase_narrative(self) -> None:
        body = _read(_QUICKSTART)
        offenders = re.findall(r"Phase [A-Z]\d?\b", body)
        self.assertEqual(
            offenders,
            [],
            f"{_QUICKSTART.name} must be phase-free; found {offenders[:5]!r}",
        )


class GraspingReadmeSectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.body = _read(_GRASPING_README)

    def test_mode_behaviour_table_present(self) -> None:
        # The mode table: `default_mode` (the single flip field), every mode, and what each samples and may
        # recover with. The presets it listed beside them left on 2026-10-16 (the owner's cleanup).
        self.assertIn("default_mode", self.body)
        for mode in ("easy", "auto", "dense_clutter"):
            self.assertIn(f"`{mode}`", self.body, f"mode table missing mode {mode!r}")
        for dimension in ("sampl", "recover"):
            self.assertIn(dimension, self.body, f"mode table missing behaviour dimension {dimension!r}")

    def test_troubleshooting_matrix_present(self) -> None:
        # The b26944c restyle points operators at docs/runbooks/ for the flagship KPI regressions
        # (the detailed matrix moved into the runbooks; the README carries the KPI triage entry point).
        self.assertIn("runbooks", self.body)
        # Must still call out the three flagship outcomes operators see.
        for token in (
            "false_positive_grasp_rate",
            "dead_loop_rate",
            "safety_rejection_rate",
        ):
            self.assertIn(
                token,
                self.body,
                f"README missing KPI-triage token {token!r}",
            )

    def test_operator_quickstart_section_present(self) -> None:
        # The b26944c restyle folds the per-preset quickstart into the Usage section; assert that operator
        # entry point survived the restyle (the presets it named left on 2026-10-16).
        #
        # Matched by REGEX, not by the exact string "## Usage": the 2026-08-20 doc-site restyle prefixes
        # every H2 with a decorative emoji, which broke an exact match while changing nothing an operator
        # cares about. The guard is "a top-level Usage section still exists" -- so pin THAT, and stay
        # blind to the decoration. Weakening it to a bare "Usage" substring would not: that appears in
        # prose.
        self.assertRegex(self.body, r"(?m)^##\s+\S*\s*Usage\s*$")


class RobotReadmeSafetyTriageTests(unittest.TestCase):
    def test_safety_triage_subset_present(self) -> None:
        body = _read(_ROBOT_README)
        # The heading was RENAMED, not removed: `## 🛑 Safety-rejection triage` in the
        # pre-migration `backend/src/robot/robot_README.md` is `## Triaging a safety rejection`
        # here -- the docs rewrite drops emoji and stops using a hyphen as a connector. Same
        # section, same guard table, same six guards below it. Pinned as an exact H2 line rather
        # than a loose substring, so a section that is merely mentioned in prose cannot satisfy it.
        self.assertRegex(body, r"(?m)^##\s+Triaging a safety rejection\s*$")
        for token in (
            "workspace",
            "joint_limit",
            "self_collision",
        ):
            self.assertIn(
                token,
                body,
                f"robot README safety triage missing {token!r}",
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
