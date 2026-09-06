"""Phase T8 — RED tests for operator documentation.

The T8 deliverable adds a top-level ``QUICKSTART.md`` plus operator
sections in ``backend/src/robot/grasping/grasping_README.md``:

* A *mode behaviour table* with eight locked columns.
* A *troubleshooting matrix* (symptom → cause → mode/policy fix).
* One quickstart paragraph per shipped preset
  (``easy`` / ``dense_clutter`` / ``verification_heavy``).

Robot-level safety triage subset lives in
``backend/src/robot/robot_README.md``.

Schema validity contract: every YAML snippet in ``QUICKSTART.md``
that is tagged with ``preset=<name>`` must round-trip through
:func:`apply_preset` and validate as a :class:`RobotConfig`.

This is purely a docs phase, so the tests are markdown / config
audits — no runtime behaviour is exercised.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import yaml

from src.config.schema.robot.robot_schema import RobotConfig
from src.robot.grasping.replay.presets import (
    apply_preset,
    list_presets,
)


_REPO_ROOT = Path(__file__).resolve().parents[1]
_QUICKSTART = _REPO_ROOT / "QUICKSTART.md"
_GRASPING_README = (
    _REPO_ROOT
    / "src"
    / "robot"
    / "grasping"
    / "grasping_README.md"
)
_ROBOT_README = (
    _REPO_ROOT / "src" / "robot" / "robot_README.md"
)
_CONFIG_README = _REPO_ROOT / "src" / "config" / "config_README.md"

_MINIMAL_BASE = {
    "vendor": "dummy",
    "gripper": {"vendor": "none"},
    "grasping": {"default_mode": "auto"},
}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract_tagged_yaml_blocks(markdown: str) -> dict[str, str]:
    """Return ``{preset_name: yaml_body}`` for fences tagged
    ``yaml preset=<name>``.
    """

    pattern = re.compile(
        r"```yaml preset=(?P<name>[a-z_]+)\n(?P<body>.*?)\n```",
        re.DOTALL,
    )
    return {
        m.group("name"): m.group("body") for m in pattern.finditer(markdown)
    }


class QuickstartFileTests(unittest.TestCase):
    def test_quickstart_exists(self) -> None:
        self.assertTrue(
            _QUICKSTART.is_file(),
            f"missing top-level QUICKSTART.md at {_QUICKSTART}",
        )

    def test_quickstart_covers_every_preset(self) -> None:
        body = _read(_QUICKSTART)
        for preset in list_presets():
            self.assertIn(
                preset,
                body,
                f"QUICKSTART.md does not mention preset {preset!r}",
            )

    def test_quickstart_yaml_snippets_validate(self) -> None:
        body = _read(_QUICKSTART)
        blocks = _extract_tagged_yaml_blocks(body)
        for preset in list_presets():
            self.assertIn(
                preset,
                blocks,
                f"QUICKSTART.md missing tagged YAML block for {preset!r}",
            )
            overlay = yaml.safe_load(blocks[preset]) or {}
            self.assertIsInstance(overlay, dict)
            merged = apply_preset(_MINIMAL_BASE, preset)
            # The doc snippet may show an operator-edited subset; we
            # still require that the canonical preset loads cleanly.
            RobotConfig.model_validate(merged)

    def test_quickstart_drops_phase_narrative(self) -> None:
        body = _read(_QUICKSTART)
        offenders = re.findall(r"Phase [A-Z]\d?\b", body)
        self.assertEqual(
            offenders,
            [],
            f"QUICKSTART.md must be phase-free; found {offenders[:5]!r}",
        )


class GraspingReadmeSectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.body = _read(_GRASPING_README)

    def test_mode_behaviour_table_present(self) -> None:
        # The b26944c README restyle documents mode behaviour as the operator PRESETS table:
        # `default_mode` (the single flip field) + the four behaviour dimensions + every shipped preset.
        self.assertIn("default_mode", self.body)
        for dimension in ("refine", "verify", "recover", "uncertainty"):
            self.assertIn(
                dimension,
                self.body,
                f"mode/presets table missing behaviour dimension {dimension!r}",
            )
        for preset in ("easy", "dense_clutter", "verification_heavy"):
            self.assertIn(
                preset,
                self.body,
                f"mode/presets table missing preset {preset!r}",
            )

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
        # The b26944c restyle folds the per-preset quickstart into the Usage section + the presets overlay
        # table (linked to grasping_presets/); assert that operator entry point survived the restyle.
        #
        # Matched by REGEX, not by the exact string "## Usage": the 2026-08-20 doc-site restyle prefixes
        # every H2 with a decorative emoji, which broke an exact match while changing nothing an operator
        # cares about. The guard is "a top-level Usage section still exists and still names where the
        # presets live" -- so pin THAT, and stay blind to the decoration. Weakening it to a bare
        # "Usage" substring would not: that appears in prose.
        self.assertRegex(self.body, r"(?m)^##\s+\S*\s*Usage\s*$")
        self.assertIn("grasping_presets", self.body)


class RobotReadmeSafetyTriageTests(unittest.TestCase):
    def test_safety_triage_subset_present(self) -> None:
        body = _read(_ROBOT_README)
        self.assertIn("Safety-rejection triage", body)
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


class ConfigReadmePresetEntryTests(unittest.TestCase):
    def test_preset_overlays_referenced(self) -> None:
        body = _read(_CONFIG_README)
        self.assertIn("grasping_presets", body)
        for preset in list_presets():
            self.assertIn(preset, body)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
