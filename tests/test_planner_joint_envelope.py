"""The planner plans inside the guard's joint envelope, and the elbow at UR's own limit (UM6, B4).

Two halves that only mean something together:

* **The elbow.** UR's own planning limit for ``elbow_joint`` is +-180 degrees (its ``joint_limits.yaml``), and six of
  Isaac's URDFs happen to ship it that way while ``ur10.urdf`` ships +-360. A descriptor built from the wrong one lets
  cuRobo plan an elbow turn no UR planner would, so the builder clamps it rather than inheriting whichever file it read.
* **Everything else.** The guard refuses outside the factory +-360 less ``margin_deg`` (5 degrees by default), and cuRobo
  narrows every joint by ``position_limit_clip`` (0.1 rad, 5.73 degrees) when it loads. 5.73 > 5.0 is the whole reason
  the planner cannot offer a configuration the guard then refuses, so it is asserted here rather than assumed.

The guard's table itself does not change: the factory envelope stays +-360, and this is about what the PLANNER may
propose inside it.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

import yaml

from src.config.schema.robot.safety_schema import JointLimitSafetyConfig
from src.robot.safety.joint_limits import UR_JOINT_LIMITS_DEG

_ROOT = Path(__file__).resolve().parents[1]
_CONTENT = _ROOT / "ext_deps" / "curobo" / "curobo" / "content" / "assets" / "robot" / "ur_description"
_ELBOW = 2

#: Shaped like the descriptions the builder writes, including the trap they carry: ``elbow_joint`` is declared TWICE,
#: first as a ros2_control block with no ``<limit>`` and only then as the revolute joint that has one (measured in
#: ext_deps/curobo/curobo/content/assets/robot/ur_description/ur5e.urdf, 2026-09-16). ur10 writes upper before lower.
_UR10_STYLE = """<?xml version="1.0"?>
<robot name="ur10">
  <ros2_control name="fake" type="system">
    <joint name="elbow_joint">
      <command_interface name="position">
        <param name="min">-3.141592653589793</param>
        <param name="max">3.141592653589793</param>
      </command_interface>
    </joint>
  </ros2_control>
  <joint name="shoulder_pan_joint" type="revolute">
    <limit effort="330.0" lower="-6.2831853" upper="6.2831853" velocity="2.16"/>
  </joint>
  <joint name="elbow_joint" type="revolute">
    <limit effort="150.0" upper="6.28318" lower="-6.28318" velocity="3.15"/>
    <safety_controller soft_lower_limit="-6.2" soft_upper_limit="6.2"/>
  </joint>
  <joint name="wrist_1_joint" type="revolute">
    <limit effort="54.0" lower="-6.2831853" upper="6.2831853" velocity="3.2"/>
  </joint>
</robot>
"""

_UR5E_STYLE = _UR10_STYLE.replace('upper="6.28318" lower="-6.28318"', 'lower="-3.14159265" upper="3.14159265"')
_NO_ELBOW = _UR10_STYLE.replace('name="elbow_joint"', 'name="forearm_joint"')


def _limits():
    path = _ROOT / "scripts" / "curobo" / "_planner_limits.py"
    spec = importlib.util.spec_from_file_location("_planner_limits_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _degrees(envelope: "tuple[tuple[float, ...], tuple[float, ...]]") -> "tuple[tuple[float, ...], tuple[float, ...]]":
    return tuple(tuple(math.degrees(v) for v in side) for side in envelope)  # type: ignore[return-value]


def _elbow_limit_blocks(text: str) -> "list[str]":
    """Every ``<joint>`` element named elbow_joint that declares a ``<limit>``, read here rather than by the module."""
    import re

    blocks = []
    for block in re.finditer(r"<joint\b[^>]*>.*?</joint>", text, re.S):
        name = re.search(r'name="([^"]+)"', block.group(0))
        if name is not None and name.group(1) == "elbow_joint" and "<limit" in block.group(0):
            blocks.append(block.group(0))
    return blocks


def _bounds(block: str) -> "list[float]":
    import re

    return [abs(float(v)) for v in re.findall(r'\b(?:lower|upper)="(-?[\d.eE+-]+)"', block)]


class ThePlannerStaysInsideTheGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.limits = _limits()
        self.margin = JointLimitSafetyConfig().margin_deg

    def test_every_bundled_arm_and_the_sim_cell(self) -> None:
        envelope = _degrees(self.limits.planner_envelope_rad(self.limits.POSITION_LIMIT_CLIP_RAD))
        cells = {model: (lo, hi) for model, (lo, hi) in UR_JOINT_LIMITS_DEG.items()}
        sim = yaml.safe_load((_ROOT / "config/robot/robot.sim.yaml").read_text(encoding="utf-8"))
        joint_limits = sim["robot"]["safety"]["joint_limits"]
        cells["robot.sim.yaml"] = (tuple(joint_limits["min_deg"]), tuple(joint_limits["max_deg"]))
        for cell, (lo, hi) in cells.items():
            with self.subTest(cell=cell):
                self.assertEqual(self.limits.violations(envelope, lo, hi, self.margin), ())

    def test_the_elbow_is_urs_own_limit_and_the_guard_keeps_the_factory_envelope(self) -> None:
        lo, hi = self.limits.planner_envelope_rad(0.0)
        self.assertEqual((lo[_ELBOW], hi[_ELBOW]), (-math.pi, math.pi))
        self.assertEqual((lo[0], hi[0]), (-2.0 * math.pi, 2.0 * math.pi))
        for model, (guard_lo, guard_hi) in UR_JOINT_LIMITS_DEG.items():
            with self.subTest(model=model):
                self.assertEqual((guard_lo[_ELBOW], guard_hi[_ELBOW]), (-360.0, 360.0))

    def test_the_comparison_can_fail(self) -> None:
        """⭐ THE CONTROL, twice. Without cuRobo's clip the planner is exactly at the factory envelope and the guard's
        margin refuses it; and a cell whose margin exceeds the clip is outside it too."""
        unclipped = _degrees(self.limits.planner_envelope_rad(0.0))
        lo, hi = UR_JOINT_LIMITS_DEG["ur5e"]
        first = self.limits.violations(unclipped, lo, hi, self.margin)
        self.assertTrue(first)
        self.assertIn("0", " ".join(first))
        clipped = _degrees(self.limits.planner_envelope_rad(self.limits.POSITION_LIMIT_CLIP_RAD))
        self.assertTrue(self.limits.violations(clipped, lo, hi, 6.0),
                        "a cell with a margin wider than the clip must be reported")


class TheElbowIsClampedInTheDescriptionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.limits = _limits()

    def test_a_full_turn_elbow_is_narrowed_and_nothing_else_moves(self) -> None:
        text, changed = self.limits.clamp_elbow_limit(_UR10_STYLE)
        self.assertTrue(changed)
        (elbow,) = _elbow_limit_blocks(text)
        self.assertIn(f'upper="{math.pi:.8f}"', elbow)
        self.assertIn(f'lower="{-math.pi:.8f}"', elbow)
        # The other joints, the elbow's soft limits and the ros2_control interface keep their own numbers.
        self.assertIn('<limit effort="330.0" lower="-6.2831853" upper="6.2831853" velocity="2.16"/>', text)
        self.assertIn('<limit effort="54.0" lower="-6.2831853" upper="6.2831853" velocity="3.2"/>', text)
        self.assertIn('<safety_controller soft_lower_limit="-6.2" soft_upper_limit="6.2"/>', text)
        self.assertIn('<param name="min">-3.141592653589793</param>', text)

    def test_the_ros2_control_block_is_not_the_joint(self) -> None:
        """⭐ THE CONTROL for a trap this box carries: a UR description declares elbow_joint twice, and the first
        element has no limit at all. An implementation that takes the first one changes nothing and reports the
        description already inside UR's limit, which is what a correct run reports on a correct file."""
        first = _UR10_STYLE.index('name="elbow_joint"')
        self.assertNotIn("<limit", _UR10_STYLE[first:first + 200], "the fixture lost the trap it exists for")
        _, changed = self.limits.clamp_elbow_limit(_UR10_STYLE)
        self.assertTrue(changed)

    def test_an_elbow_already_at_pi_comes_back_byte_for_byte(self) -> None:
        """⭐ THE CONTROL. Six of the seven URDFs already carry it, and an implementation that re-serialises the XML
        would rewrite every one of them."""
        text, changed = self.limits.clamp_elbow_limit(_UR5E_STYLE)
        self.assertFalse(changed)
        self.assertEqual(text, _UR5E_STYLE)

    def test_a_description_without_an_elbow_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            self.limits.clamp_elbow_limit(_NO_ELBOW)


class TheBuiltDescriptionsCarryItTests(unittest.TestCase):
    """This box only: what the builder actually wrote into the cuRobo content."""

    def test_every_built_urdf_keeps_the_elbow_inside_urs_limit(self) -> None:
        if not _CONTENT.is_dir():
            self.skipTest(f"no built UR descriptions at {_CONTENT}")
        built = sorted(_CONTENT.glob("ur*.urdf"))
        self.assertTrue(built, f"no ur*.urdf under {_CONTENT}")
        for path in built:
            with self.subTest(urdf=path.name):
                blocks = _elbow_limit_blocks(path.read_text(encoding="utf-8"))
                self.assertTrue(blocks, f"{path.name} declares no elbow joint with a limit")
                values = [value for block in blocks for value in _bounds(block)]
                self.assertTrue(values, f"{path.name} has no elbow bounds")
                self.assertLessEqual(max(values), math.pi + 1e-9,
                                     f"{path.name} lets the planner turn the elbow past UR's own limit")


if __name__ == "__main__":
    unittest.main()
