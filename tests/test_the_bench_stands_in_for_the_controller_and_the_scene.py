"""The grasp bench (``scripts/bench``): what stands in for the controller and the scene, checked without a GPU.

The bench runs the owner's own stack on an instant controller and a rendered scene (the owner, 2026-10-05). Pinned here:

* the instant controller's kinematics are the stack's own: a TCP pose read back through its IK lands on the same joints,
  a ``moveL`` lands where it was sent, a move a cell with ``brake_on_halt`` watches arrives as on a cell, and a tool
  output that changes is handed on with the TCP it changed at;
* a packed pile keeps every part apart by at least the gap it drew (touching allowed, never through each other), inside
  the bin's walls, and dense: the 30-part bin holds all 30;
* a bin's walls and floor are fixed: the jaws never grip them, a push never moves them;
* a rearranging push takes along what stands ahead of the part in its lane, just ahead of where the part stops,
  and nothing beside the lane (2026-10-16); a push that shoves nothing moves the part alone;
* the stand-in detector grounds every movable part where a scene asks for all of them, and the target alone otherwise.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_ROOT / "scripts" / "bench"), str(_ROOT / "scripts" / "ursim")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import _mat_scene as ms  # noqa: E402
import instant_ur  # noqa: E402
import scenes  # noqa: E402

LOOK0 = [math.radians(v) for v in (-87.50, -64.51, -85.79, -129.82, 90.27, -108.84)]
CROP = _ROOT / "tests" / "data" / "cell_2026_10_01" / "p1_pick-63e1b3dac29f.npz"


def _static() -> Any:
    static = ms.StaticScene.from_crop(CROP)
    static.whole_bench = True
    return static


class TheInstantControllerTests(unittest.TestCase):
    def _state(self) -> instant_ur.ControllerState:
        return instant_ur.ControllerState(model="ur10", joints=list(LOOK0), tcp_offset=[0.0, 0.0, 0.157, 0.0, 0.0, 0.0])

    def test_ik_of_the_tcp_lands_on_the_joints(self) -> None:
        state = self._state()
        tcp = state.tcp_mm()
        np.testing.assert_allclose(state.ik(instant_ur.ur_pose(tcp)), LOOK0, atol=1e-6)

    def test_a_move_l_lands_where_it_was_sent(self) -> None:
        state = self._state()
        before = instant_ur.install(state)
        try:
            from src.robot.drivers.ur import connection

            control = connection.rtde_control.RTDEControlInterface("127.0.0.1")
            goal = state.tcp_mm()
            goal[:3, 3] += [10.0, -20.0, -30.0]
            self.assertTrue(control.moveL(instant_ur.ur_pose(goal), 0.1, 0.1))
            np.testing.assert_allclose(state.tcp_mm()[:3, 3], goal[:3, 3], atol=1e-6)
        finally:
            instant_ur.remove(before)

    def test_a_watched_move_arrives_as_on_a_cell(self) -> None:
        """A cell with ``robot.ur.brake_on_halt`` sends each move asynchronously and waits to see its operation in the
        progress register (2026-10-16: the bank's generated views ended in an AttributeError there, the register
        carrying no operation id)."""
        state = self._state()
        before = instant_ur.install(state)
        try:
            from src.robot.drivers.ur import connection

            link = connection.URConnection("127.0.0.1", brake_on_halt=True)
            link.connect()
            try:
                target = [q + 0.05 for q in LOOK0]
                self.assertTrue(link.moveJ(target, 0.1, 0.1))
                np.testing.assert_allclose(state.joints, target, atol=1e-9)
                self.assertIs(connection.MoveEnd.ARRIVED, link.last_move_end)
                self.assertEqual(1, state.operations)
            finally:
                link.disconnect()
        finally:
            instant_ur.remove(before)

    def test_a_tool_output_that_changes_is_handed_on_with_the_tcp(self) -> None:
        state = self._state()
        seen: list[tuple[int, bool]] = []
        state.on_tool_output = lambda pin, level, tcp: seen.append((pin, level))
        state.set_output(16, True)
        state.set_output(16, True)   # no change, nothing handed on
        state.set_output(16, False)
        self.assertEqual([(0, True), (0, False)], seen)


class APackedPileTests(unittest.TestCase):
    def test_every_part_keeps_apart_and_inside_the_bin(self) -> None:
        static = _static()
        parts = scenes.SCENES["pile_bin_30_s6"].build(static)
        movable = [p for p in parts if not p.fixed]
        self.assertEqual(30, len(movable), "the 30-part bin does not hold its 30")
        cx, cy = scenes.CENTRE_XY
        lx, ly, _ = scenes.KLT_INNER
        polys = []
        for part in movable:
            kind = part.name.rsplit("_", 1)[0]
            yaw = math.atan2(part.frame[1, 2], part.frame[0, 2]) if scenes.PILE_KINDS[kind][0] == "lying" else \
                math.atan2(part.frame[1, 0], part.frame[0, 0])
            xy = part.frame[:2, 3] if scenes.PILE_KINDS[kind][0] != "lying" else \
                part.frame[:2, 3] + 0.5 * part.dims[2] * part.frame[:2, 2]
            poly = scenes._footprint(kind, xy, yaw)
            self.assertGreaterEqual(poly[:, 0].min(), cx - lx / 2 - 1e-6)
            self.assertLessEqual(poly[:, 0].max(), cx + lx / 2 + 1e-6)
            self.assertGreaterEqual(poly[:, 1].min(), cy - ly / 2 - 1e-6)
            self.assertLessEqual(poly[:, 1].max(), cy + ly / 2 + 1e-6)
            polys.append(poly)
        for i in range(len(polys)):
            for j in range(i):
                self.assertFalse(scenes._overlap(polys[i], polys[j]), f"{movable[i].name} and {movable[j].name}")

    def test_a_pile_packs(self) -> None:
        static = _static()
        parts = [p for p in scenes.SCENES["pile_bin_20_touching_s31"].build(static) if not p.fixed]
        centres = np.array([p.frame[:2, 3] for p in parts])
        spread = np.ptp(centres, axis=0)
        lx, ly, _ = scenes.KLT_INNER
        self.assertLess(float(spread[0] * spread[1]), 0.9 * lx * ly, "the pile is a scatter, not a pile")


class TheBinIsFixedTests(unittest.TestCase):
    def test_the_jaws_never_grip_a_wall_and_a_push_never_moves_one(self) -> None:
        static = _static()
        scene = ms.Scene(static)
        parts = scenes.SCENES["bin_wall_20"].build(static)
        scene.set_parts(parts, target="cylinder")
        wall = scene.parts["klt_wall_px"]
        tcp = np.eye(4)
        tcp[:3, 3] = wall.frame[:3, 3] + wall.frame[:3, 2] * 50.0
        scene.jaws_changed(True, tcp)
        self.assertIsNone(scene.held())
        before = wall.frame.copy()

        class _Plan:
            target_centre_mm = tuple(wall.frame[:3, 3])
            direction = (1.0, 0.0, 0.0)
            push_distance_mm = 30.0
            contact_start_tcp_mm = tuple(wall.frame[:3, 3])

        scene.keep_height_on_push = True
        scene.pushed(_Plan(), ("push",), None)
        np.testing.assert_allclose(wall.frame, before)


class ARearrangingPushShovesWhatStandsAheadTests(unittest.TestCase):
    """The scene push's U on URSim (``probe_push_on_the_mat.py``, scene): the cylinder pushed 50 mm along -y out of
    three 30 mm blocks 3 mm off its -x, +x and -y sides."""

    def _pushed(self, shoves: bool) -> "dict[str, np.ndarray]":
        static = _static()
        scene = ms.Scene(static)
        at = np.array([-150.0, -700.0])
        parts = [ms.cylinder("cylinder", *scene.foot_on_mat(at), radius_mm=20.0, height_mm=50.0)]
        for name, offset in (("ahead", (0.0, -38.0)), ("beside", (38.0, 0.0)), ("behind_x", (-38.0, 0.0))):
            parts.append(ms.box(name, *scene.foot_on_mat(at + np.array(offset)), size_mm=(30.0, 30.0, 30.0)))
        scene.set_parts(parts, target="cylinder")

        class _Plan:
            target_centre_mm = (float(at[0]), float(at[1]), 0.0)
            direction = (0.0, -1.0, 0.0)
            push_distance_mm = 50.0
            contact_start_tcp_mm = (float(at[0]), float(at[1]) + 30.0, 0.0)

        plan = _Plan()
        plan.shoves = shoves  # type: ignore[attr-defined]
        scene.pushed(plan, ("push",), None)
        return {name: part.frame[:2, 3].copy() for name, part in scene.parts.items()}

    def test_the_block_ahead_ends_just_ahead_of_the_part_and_the_blocks_beside_stay(self) -> None:
        where = self._pushed(True)
        np.testing.assert_allclose(where["cylinder"], [-150.0, -750.0], atol=0.01)
        # The cylinder's front is 20 mm past its centre, the block's centre 15 mm past its own face (the mat's 1 degree
        # tilts the block's frame, a micrometre in plan).
        np.testing.assert_allclose(where["ahead"], [-150.0, -785.0], atol=0.01)
        np.testing.assert_allclose(where["beside"], [-112.0, -700.0], atol=0.01)
        np.testing.assert_allclose(where["behind_x"], [-188.0, -700.0], atol=0.01)

    def test_a_push_that_shoves_nothing_moves_the_part_alone(self) -> None:
        where = self._pushed(False)
        np.testing.assert_allclose(where["cylinder"], [-150.0, -750.0], atol=0.01)
        np.testing.assert_allclose(where["ahead"], [-150.0, -738.0], atol=0.01)


class TheStandInDetectorTests(unittest.TestCase):
    def _grounded(self, all_parts: bool) -> list[str]:
        static = _static()
        scene = ms.Scene(static)
        parts = scenes.SCENES["pile_bin_15_s1"].build(static)
        scene.set_parts(parts, target="")
        from tests._cell_2026_10_01 import look

        recorded = look("P1")
        camera = ms.StandInD415(object(), scene, camera_to_tool=recorded.camera_to_tool,
                                tcp=lambda: recorded.tool_to_base)
        camera.open()
        camera.grab()
        backend = ms.StandInBackend(camera)
        backend.all_parts = all_parts
        return [o.segmentation.metadata["part"] for o in backend.perceive(None, "part")]

    def test_a_clearing_scene_grounds_every_movable_part_and_no_wall(self) -> None:
        grounded = self._grounded(True)
        self.assertGreaterEqual(len(grounded), 10)
        self.assertFalse([n for n in grounded if n.startswith("klt_")])

    def test_a_single_target_scene_grounds_nothing_but_its_target(self) -> None:
        self.assertEqual([], self._grounded(False), "an empty target grounded something")


if __name__ == "__main__":
    unittest.main()
