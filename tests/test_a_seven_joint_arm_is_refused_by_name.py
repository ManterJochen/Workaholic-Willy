"""The build's six joint rules meet a seven joint arm by name, never by a truncated vector (P0f F8).

Five places in the build held six: the gate's random draws, the joints the retract search moves, the envelope a
candidate is judged against, the planner check's joint loop and the fit plan's frames. Each now takes its count from
the arm description or refuses a length it was not given. Two poses were only ever measured on a six joint UR, the
gate's sweep base and the folded elbow every judge reads first as its control, and those refuse any other arm by name
until P1 measures that arm's own (owner decision, 2026-09-23).

The arm here is synthetic: ur5e's description with a seventh joint, written to a temporary registry and never admitted,
so nothing in the repository's registry changes.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

_ROOT = Path(__file__).resolve().parents[1]
_REGISTRY = _ROOT / "config" / "arms"
_MAPS = _ROOT / "src" / "robot" / "safety" / "planning" / "robot"
#: The hand maps this repository ships. A customer's cell fits its own beside them (the customer-hand trial adds two),
#: so these may not be missing and more may be there, and every map there is held to the plan.
_SHIPPED_HAND_MAPS = frozenset({"robotiq_2f85", "robotiq_hande", "schunk_egu50"})


def _by_path(name: str, relative: str):
    if name in sys.modules:
        return sys.modules[name]
    path = _ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _gate():
    return _by_path("_matrix_gate_for_seven_joints", "scripts/arms/_matrix_gate.py")


def _rule():
    return _by_path("_retract_rule_for_seven_joints", "scripts/arms/_retract_rule.py")


def _fit():
    return _by_path("_fit_for_seven_joints", "scripts/arms/fit.py")


def _described(model: str, *, vendor: str, extra_joint: bool) -> dict:
    """ur5e's ``arm:`` block under another model name, optionally with a seventh joint copied from its sixth."""
    arm = yaml.safe_load((_REGISTRY / "ur5e.yaml").read_text(encoding="utf-8"))["arm"]
    arm.update(model=model, vendor=vendor, admitted=False)
    if extra_joint:
        seventh = dict(arm["joints"][-1])
        seventh.update(name="wrist_4_joint", link="wrist_4_link", part="wrist_4")
        arm["joints"].append(seventh)
    return {"arm": arm}


class _SyntheticRegistry(unittest.TestCase):
    """A registry holding ur5e, a seven joint UR copy and a six joint copy under another vendor."""

    def setUp(self) -> None:
        self.gate = _gate()
        self.folder = Path(tempfile.mkdtemp(prefix="seven_joints_"))
        self.addCleanup(shutil.rmtree, self.folder, True)
        shutil.copy(_REGISTRY / "ur5e.yaml", self.folder / "ur5e.yaml")
        for model, vendor, extra in (("synthetic7", "ur", True), ("synthetic_other", "acme", False)):
            (self.folder / f"{model}.yaml").write_text(
                yaml.safe_dump(_described(model, vendor=vendor, extra_joint=extra), sort_keys=False), encoding="utf-8")
        self._point(self.gate)

    def _point(self, module) -> None:
        """``module``'s reader at the synthetic registry: each script binds the reader it imported."""
        arm_file = module._arm_file
        arm_file.read.cache_clear()
        self.addCleanup(arm_file.read.cache_clear)
        patch = mock.patch.object(arm_file, "ARMS", self.folder)
        patch.start()
        self.addCleanup(patch.stop)


class ThePoseSetRefusesByNameTests(_SyntheticRegistry):
    def test_a_seven_joint_arm_is_refused_by_name(self) -> None:
        with self.assertRaises(self.gate.PoseSetRefused) as caught:
            self.gate.pose_set([0.0] * 7, arm="synthetic7")
        said = str(caught.exception)
        self.assertIn("synthetic7", said)
        self.assertIn("7 joint", said)

    def test_a_six_joint_arm_of_another_vendor_is_refused_by_name(self) -> None:
        """Six joints are not the premise: the sweep base is a UR's pose, measured on URs."""
        with self.assertRaises(self.gate.PoseSetRefused) as caught:
            self.gate.pose_set([0.0] * 6, arm="synthetic_other")
        self.assertIn("synthetic_other", str(caught.exception))
        self.assertIn("acme", str(caught.exception))

    def test_a_retract_of_another_length_is_refused(self) -> None:
        with self.assertRaises(self.gate.PoseSetRefused) as caught:
            self.gate.pose_set([0.0] * 5, arm="ur5e")
        self.assertIn("retract of 5", str(caught.exception))

    def test_the_folded_control_is_refused_by_name(self) -> None:
        for model in ("synthetic7", "synthetic_other"):
            with self.subTest(model=model), self.assertRaises(self.gate.PoseSetRefused) as caught:
                self.gate.folded_control(model)
            self.assertIn(model, str(caught.exception))

    def test_a_six_joint_ur_keeps_both(self) -> None:
        """THE CONTROL: the refusals above are about the arm, not about the call."""
        poses, kinds = self.gate.pose_set([0.0, -1.57, 1.57, 0.0, 0.0, 0.0], arm="ur5e")
        self.assertEqual(1482, len(poses))
        self.assertEqual({6}, {len(pose) for pose in poses})
        self.assertEqual(len(poses), len(kinds))
        self.assertEqual([0.0, -1.57, 3.1, 0.0, 0.0, 0.0], self.gate.folded_control("ur5e"))


class TheRetractSearchTakesItsJointsFromTheAnchorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rule = _rule()

    def test_every_joint_but_the_pan_moves_on_seven_joints(self) -> None:
        """One step each way on six joints is 3 ** 6 candidates; the pan never moves and the seventh joint does."""
        seen = list(self.rule.candidates([0.0] * 7, step_rad=0.25, max_steps=1))
        self.assertEqual(3 ** 6, len(seen))
        self.assertEqual({7}, {len(pose) for pose, _steps in seen})
        self.assertEqual({0.0}, {pose[0] for pose, _steps in seen})
        self.assertEqual({-0.25, 0.0, 0.25}, {pose[6] for pose, _steps in seen})

    def test_six_joints_keep_the_five_joint_search(self) -> None:
        """THE CONTROL: at six joints the family is the committed table's, 13 ** 5 at the default six steps."""
        steps = [steps for _pose, steps in self.rule.candidates([0.0] * 6, max_steps=1)]
        self.assertEqual(3 ** 5, len(steps))
        self.assertEqual({5}, {len(vector) for vector in steps})

    def test_a_pose_against_an_envelope_of_another_length_is_refused(self) -> None:
        six = (tuple([-3.0] * 6), tuple([3.0] * 6))
        with self.assertRaises(ValueError) as caught:
            self.rule._inside([0.0] * 7, six)
        self.assertIn("pose of 7 joints", str(caught.exception))
        self.assertTrue(self.rule._inside([0.0] * 6, six))


class TheFitPlanReadsItsFramesTests(_SyntheticRegistry):
    def setUp(self) -> None:
        super().setUp()
        self.fit = _fit()
        self._point(self.fit)

    def test_a_part_without_a_budget_is_refused_by_name(self) -> None:
        frames = self.fit.arm_frames("synthetic7")
        self.assertEqual(7, frames["wrist_4"])
        with self.assertRaises(SystemExit) as caught:
            self.fit.planned(frames, self.fit.ARM_PLAN, target="--arm synthetic7")
        said = str(caught.exception.code)
        self.assertIn("wrist_4", said)
        self.assertIn("scripts/arms/fit.py --arm synthetic7 --curve", said)


class TheFitPlanIsWhatEveryCommittedMapWasFittedToTests(unittest.TestCase):
    """THE CONTROL: the plan built from a description or a bundle names the bodies and frames of every map."""

    def setUp(self) -> None:
        self.fit = _fit()

    def _recorded(self, path: Path) -> list[tuple[str, int]]:
        bodies = yaml.safe_load(path.read_text(encoding="utf-8"))["_provenance"]["bodies"]
        return [(body["body"], body["frame"]) for body in bodies]

    def test_every_arm_map(self) -> None:
        maps = sorted(_MAPS.glob("*_arm_spheres.yml"))
        self.assertEqual(7, len(maps))
        for path in maps:
            model = path.name.removesuffix("_arm_spheres.yml")
            plan = self.fit.planned(self.fit.arm_frames(model), self.fit.ARM_PLAN, target=f"--arm {model}")
            with self.subTest(model=model):
                self.assertEqual(self._recorded(path), [(part, ask.frame) for part, ask in plan.items()])

    def test_every_hand_map(self) -> None:
        maps = sorted(_MAPS.glob("*_gripper_spheres.yml"))
        found = {path.name.removesuffix("_gripper_spheres.yml") for path in maps}
        self.assertLessEqual(_SHIPPED_HAND_MAPS, found, "a shipped hand map is missing")
        for path in maps:
            hand = path.name.removesuffix("_gripper_spheres.yml")
            arrays = self.fit._arrays(self.fit.DATA / f"{hand}_hand_meshes.npz")
            plan = self.fit.planned(self.fit.hand_frames(arrays), self.fit.HAND_PLAN, target=f"--hand {hand}")
            with self.subTest(hand=hand):
                self.assertEqual(self._recorded(path), [(part, ask.frame) for part, ask in plan.items()])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
