"""Pure / mock-safe (no isaacsim) tests for the ``run_multiview_pick`` SCENES table.

``run_multiview_pick`` imports its Isaac touchpoints lazily inside the run function, so the scene
table and the prompt-selection helper import and run Isaac-free.

**Why a scene needs a test at all.** The multiview gate refuses every pick on the table-clearance
check, and the first candidate explanation was "the target is too short for a top-down jaw". Testing
that meant a scene with a TALLER TARGET -- and running ``--scene occlude`` for it produced numbers
bit-identical to ``clutter``, which reads exactly like a refutation and is not one: ``occlude``'s
optional size belongs to the OCCLUDER, so the target never changed and the grasp geometry never
moved. A null test that looks like a measurement is worse than no test, so the distinction is pinned
here rather than left to a comment: for each scene, the object the DEFAULT PROMPT selects is checked
against what that scene claims to vary.
"""

from __future__ import annotations

import unittest

import numpy as np

#: The runner's own default (``--prompt``), so the test selects the target the same way a run does.
DEFAULT_PROMPT = "a green cube"
#: ``SimObjectConfig.size_mm`` default -- the cube every spec without a 4th tuple element becomes.
DEFAULT_SIZE_MM = (30.0, 30.0, 50.0)


def _mod():  # noqa: ANN202 - lazy import keeps the test isaacsim-free
    from src.willy_sim import run_multiview_pick

    return run_multiview_pick


def _target_spec(scene: str):  # noqa: ANN202
    m = _mod()
    specs = m.SCENES[scene]
    idx = m._select_by_prompt([s[0] for s in specs], DEFAULT_PROMPT)
    return idx, specs[idx]


class SceneTableTests(unittest.TestCase):
    def test_tall_scene_target_is_the_tall_object(self) -> None:
        """The point of ``tall``: the object the prompt SELECTS is the one that got bigger."""
        _idx, spec = _target_spec("tall")
        self.assertGreater(len(spec), 3, "the tall scene's TARGET must carry an explicit size_mm")
        self.assertGreater(
            spec[3][2], DEFAULT_SIZE_MM[2],
            "the tall scene's target must be taller than the default cube, or it tests nothing")

    def test_occlude_scene_target_is_the_default_cube(self) -> None:
        """The null-test guard: ``occlude`` varies the OCCLUDER, so it cannot answer the height question."""
        _idx, spec = _target_spec("occlude")
        self.assertEqual(len(spec), 3, "occlude's target is the DEFAULT cube -- its size_mm is the occluder's")
        sized = [s for s in _mod().SCENES["occlude"] if len(s) > 3]
        self.assertTrue(sized, "occlude must still carry a sized occluder (that is the scene's point)")

    def test_tall_and_clutter_differ_only_in_the_target(self) -> None:
        """One variable. A second difference would make the A/B unreadable."""
        m = _mod()
        clutter, tall = m.SCENES["clutter"], m.SCENES["tall"]
        self.assertEqual(len(clutter), len(tall))
        idx, _spec = _target_spec("tall")
        for i, (c, t) in enumerate(zip(clutter, tall)):
            if i == idx:
                continue
            self.assertEqual(c, t, f"non-target object {i} differs between clutter and tall")
        # ... and the target keeps its footprint and its XY, so only the height moved.
        self.assertEqual(tall[idx][3][:2], DEFAULT_SIZE_MM[:2])
        self.assertEqual(tall[idx][2][:2], clutter[idx][2][:2])

    def test_every_object_rests_on_the_table(self) -> None:
        """``position_mm.z == size_mm.z / 2`` -- a sunk or floating object silently changes the support
        geometry these runs are measuring, which is the one confound a scene table can hide."""
        m = _mod()
        for scene, specs in m.SCENES.items():
            for spec in specs:
                size = spec[3] if len(spec) > 3 else DEFAULT_SIZE_MM
                self.assertAlmostEqual(
                    spec[2][2], size[2] / 2.0, places=6,
                    msg=f"{scene}: {spec[0]!r} at z={spec[2][2]} does not rest on the table (h={size[2]})")

    def test_scene_specs_carries_the_size_through(self) -> None:
        """``_scene_specs`` is what actually reaches Isaac; the tuple is only the declaration."""
        m = _mod()
        idx, spec = _target_spec("tall")
        built = m._scene_specs("tall")
        self.assertEqual(tuple(built[idx].size_mm), tuple(spec[3]))
        self.assertEqual(tuple(built[idx].position_mm), tuple(spec[2]))
        # An unsized neighbour must still land on the schema default rather than inherit anything.
        other = next(i for i in range(len(built)) if i != idx)
        self.assertEqual(tuple(built[other].size_mm), DEFAULT_SIZE_MM)


class TallTargetClearsTheTableCheckTests(unittest.TestCase):
    """The scene's height is a MEASUREMENT, so it is measured here rather than asserted in a comment.

    Runs the two stages the gate runs -- SFE plans the grasp, the ``ParallelJawGripperModel`` envelope
    re-checks its table clearance -- over a synthetic top-face cloud, and pins the claim the scene is
    built on: at the tall target's height the rank-0 grasp clears the required margin, and it clears
    by more than the default cube does. Thresholds, not magic numbers, so a re-measured gripper does
    not break the test for the wrong reason.
    """

    MIN_CLEARANCE_MM = 5.0

    def _rank0_clearance_mm(self, height_mm: float) -> float:
        from src.geometry import Frame
        from src.robot.grasping.collision import ParallelJawGripperModel, SupportPlane
        from src.robot.grasping.collision.table_collision import gripper_table_clearance_mm
        from src.robot.grasping.generation.support_footprint import (
            SupportFootprintJaw,
            generate_support_footprint_grasps,
        )
        from src.robot.grasping.planning import GraspPose

        model = ParallelJawGripperModel()
        xs = np.arange(435.0, 465.0 + 1e-9, 2.0)
        ys = np.arange(-15.0, 15.0 + 1e-9, 2.0)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        cloud = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, height_mm)])

        candidates = generate_support_footprint_grasps(
            cloud,
            support_height_mm=0.0,
            jaw=SupportFootprintJaw.from_model(model, table_clearance_mm=self.MIN_CLEARANCE_MM),
            obstacle_points_base_mm=None,
            max_candidates=12,
        )
        self.assertTrue(candidates, f"SFE produced no candidate for a {height_mm:.0f} mm target")
        top = candidates[0]  # score-ordered: the one the robot would take
        approach = np.asarray(top.approach, dtype=np.float64)
        closing = np.asarray(top.closing_axis, dtype=np.float64)
        binormal = np.cross(approach, closing)
        binormal /= float(np.linalg.norm(binormal))
        position = np.asarray(top.position_mm, dtype=np.float64)
        half = 0.5 * float(top.grip_width_mm)
        pose = GraspPose(
            position_mm=position,
            rotation_matrix=np.column_stack([closing, binormal, approach]),
            grip_width_mm=float(top.grip_width_mm),
            score=float(np.clip(top.score, 0.0, 1.0)),
            confidence=float(np.clip(top.score, 0.0, 1.0)),
            contacts=(position - half * closing, position + half * closing),
            frame=Frame.BASE,
        )
        return gripper_table_clearance_mm(
            pose,
            SupportPlane(normal=np.array([0.0, 0.0, 1.0]), offset_mm=0.0, frame=Frame.BASE),
            gripper_model=model,
        )

    def test_tall_target_height_clears_the_requirement(self) -> None:
        tall_h = _target_spec("tall")[1][3][2]
        self.assertGreaterEqual(self._rank0_clearance_mm(tall_h), self.MIN_CLEARANCE_MM)

    def test_tall_target_clears_by_more_than_the_default_cube(self) -> None:
        tall_h = _target_spec("tall")[1][3][2]
        self.assertGreater(
            self._rank0_clearance_mm(tall_h), self._rank0_clearance_mm(DEFAULT_SIZE_MM[2]),
            "a taller target must buy table clearance, otherwise the scene is not the experiment")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
