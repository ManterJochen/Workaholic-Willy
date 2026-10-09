"""A world of boxes goes into the planner's storage in one call where the cell asks, and through update_world otherwise.

Every world refresh re-registered all of the cell's boxes the way cuRobo's ``update_world`` does, one ``CuboidData.add``
per box with a device sync each: 217 ms a refresh at 129 boxes on the owner's cell (2026-10-08), about eight refreshes a
pick. cuRobo's own ``load_batch`` fills the same slots in one call (0.29 ms against 13.9 ms on the CPU, every slot the
same bit for bit). ``safety.planning_world.register_in_place`` (off by default) asks for it (``_curobo_world``).

This file holds it on the CPU:

* the sidecar's helper, against a storage of its own that keeps cuRobo's ``load_batch`` contract: every box written in
  one call, both scene references kept, the graph buffer reset; too many boxes raise before any slot is written; a
  storage that holds meshes or a grid is never written in place;
* when a scene is boxes alone (``boxes_only``) and when the variable asks for it (``in_place_asked``);
* the sidecar's ``set_world`` branch, read as source: in place only where asked and boxes alone, ``update_world``
  otherwise, the reply and the remembered world as they were;
* the client and the config: the variable written only where the cell asks, a shell's leftover never inherited.

The bit for bit comparison against cuRobo's own storage is ``scripts/curobo/probe_world_in_place.py`` (cuRobo on the CPU
or the GPU, and two sidecars judging 1,000 configurations in each world).
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from src.robot.safety.planning._curobo_world import (
    ENV_WORLD_IN_PLACE,
    InPlaceRefused,
    boxes_only,
    in_place_asked,
    register_boxes_in_place,
)

_SERVER = Path(__file__).resolve().parents[1] / "src" / "robot" / "safety" / "planning" / "curobo_planner_server.py"


class _Boxes:
    """A box storage with cuRobo's ``load_batch`` contract: more boxes than slots raise before anything is written."""

    def __init__(self, slots: int) -> None:
        self.max_n = slots
        self.names: list[list[Any]] = [[None] * slots]
        self.count = 0
        self.calls = 0

    def load_batch(self, cuboids: list, env_idx: int) -> None:
        if len(cuboids) > self.max_n:
            raise ValueError(f"Cannot load {len(cuboids)} cuboids, max cache size is {self.max_n}")
        self.calls += 1
        self.names[env_idx] = [box.name for box in cuboids] + [None] * (self.max_n - len(cuboids))
        self.count = len(cuboids)


class _Graph:
    def __init__(self) -> None:
        self.resets = 0

    def reset_buffer(self) -> None:
        self.resets += 1


def _checker(slots: int = 4, **other: Any) -> SimpleNamespace:
    data = SimpleNamespace(cuboids=_Boxes(slots), meshes=None, voxels=None, scene_model=None)
    for key, value in other.items():
        setattr(data, key, value)
    return SimpleNamespace(data=data, scene_model=None)


def _scene(*names: str) -> SimpleNamespace:
    return SimpleNamespace(cuboid=[SimpleNamespace(name=name) for name in names])


class TheBoxesAreWrittenInOneCallTests(unittest.TestCase):
    def test_every_box_goes_in_one_call_and_the_scene_is_kept_as_update_world_keeps_it(self) -> None:
        """⭐ What update_world does for boxes: the storage replaced, the scene kept twice, the graph buffer reset."""
        checker, graph, scene = _checker(), _Graph(), _scene("table", "seen_00", "seen_01")
        held = register_boxes_in_place(checker, scene, graph)
        self.assertEqual(3, held)
        self.assertEqual(1, checker.data.cuboids.calls, "one call, not one per box")
        self.assertEqual(["table", "seen_00", "seen_01", None], checker.data.cuboids.names[0])
        self.assertIs(scene, checker.data.scene_model, "load_from_scene_cfg keeps the scene on the storage")
        self.assertIs(scene, checker.scene_model, "load_collision_model keeps it on the checker")
        self.assertEqual(1, graph.resets, "update_world resets the graph planner's buffer")

    def test_a_smaller_world_after_a_larger_one_leaves_no_box_of_the_larger(self) -> None:
        checker = _checker()
        register_boxes_in_place(checker, _scene("a", "b", "c"), None)
        register_boxes_in_place(checker, _scene("a"), None)
        self.assertEqual(["a", None, None, None], checker.data.cuboids.names[0])
        self.assertEqual(1, checker.data.cuboids.count)

    def test_too_many_boxes_raise_before_any_slot_is_written(self) -> None:
        """⛔ The planner keeps the world it had: the registration is refused, and the client refuses the motion."""
        checker, graph = _checker(slots=2), _Graph()
        register_boxes_in_place(checker, _scene("a", "b"), graph)
        with self.assertRaises(ValueError):
            register_boxes_in_place(checker, _scene("x", "y", "z"), graph)
        self.assertEqual(["a", "b"], checker.data.cuboids.names[0])
        self.assertEqual(1, graph.resets, "nothing past the refusal ran")

    def test_a_storage_holding_meshes_or_a_grid_is_never_written_in_place(self) -> None:
        """⛔ update_world clears those too; written in place, a mesh or a field of the last world would stay."""
        for kind in ("meshes", "voxels"):
            with self.subTest(kind=kind):
                checker = _checker(**{kind: object()})
                with self.assertRaises(InPlaceRefused):
                    register_boxes_in_place(checker, _scene("a"), None)
                self.assertEqual(0, checker.data.cuboids.calls)
                self.assertIsNone(checker.scene_model)


class WhenTheWorldIsBoxesAloneTests(unittest.TestCase):
    def test_boxes_alone_on_a_planner_that_holds_nothing_else(self) -> None:
        self.assertTrue(boxes_only({"cuboid": {"a": {}}}, mesh_slots=0, voxel_reserved=False))
        self.assertTrue(boxes_only({"cuboid": {}}, mesh_slots=0, voxel_reserved=False))

    def test_a_mesh_a_field_mesh_slots_or_a_grid_go_through_update_world(self) -> None:
        """The control: each of the four alone is enough to keep today's path."""
        self.assertFalse(boxes_only({"cuboid": {}, "mesh": {"tote": {}}}, mesh_slots=0, voxel_reserved=False))
        self.assertFalse(boxes_only({"cuboid": {}, "voxel": {"scene": {}}}, mesh_slots=0, voxel_reserved=False))
        self.assertFalse(boxes_only({"cuboid": {}}, mesh_slots=1, voxel_reserved=False))
        self.assertFalse(boxes_only({"cuboid": {}}, mesh_slots=0, voxel_reserved=True))

    def test_only_a_one_asks_for_it(self) -> None:
        self.assertTrue(in_place_asked({ENV_WORLD_IN_PLACE: "1"}))
        for value in ("", "0", "true", "yes", " 2"):
            with self.subTest(value=value):
                self.assertFalse(in_place_asked({ENV_WORLD_IN_PLACE: value}))
        self.assertFalse(in_place_asked({}))


def _block(source: str, head: str) -> str:
    lines = source.splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip() == head), None)
    if start is None:
        return ""
    indent = len(lines[start]) - len(lines[start].lstrip())
    block = [lines[start]]
    for line in lines[start + 1:]:
        if line.strip() and len(line) - len(line.lstrip()) <= indent:
            break
        block.append(line)
    return "\n".join(block)


class TheSidecarRegistersInPlaceOnlyWhereAskedTests(unittest.TestCase):
    """The sidecar runs in the cuRobo environment, so its branch is read as source, scoped to the block."""

    def setUp(self) -> None:
        self.source = _SERVER.read_text(encoding="utf-8")
        self.set_world = _block(self.source, 'if cmd == "set_world":')

    def test_in_place_only_where_asked_and_boxes_alone(self) -> None:
        self.assertTrue(self.set_world, "the set_world branch is gone")
        guard = self.set_world.find("if _WORLD_IN_PLACE and boxes_only(")
        call = self.set_world.find("register_boxes_in_place(_planner.scene_collision_checker")
        self.assertTrue(0 <= guard < call, "registered in place without both conditions first")
        self.assertIn("_WORLD_IN_PLACE = in_place_asked(os.environ)", self.source)

    def test_switched_off_the_world_goes_through_update_world_as_it_always_did(self) -> None:
        """⭐ THE CONTROL of the default: today's call is still the path every other world takes."""
        self.assertIn("_planner.update_world(SceneCfg.create(world_scene))", self.set_world)
        self.assertIn('if _how != "in place":', self.set_world)
        self.assertIn("except InPlaceRefused", self.set_world)

    def test_the_reply_and_the_remembered_world_are_what_they_were(self) -> None:
        self.assertIn('reply: dict[str, Any] = {"world_set": len(world) + len(meshes)}', self.set_world)
        for kept in ("_LAST_CUBOIDS = world", "_LAST_MESHES = mesh_block", "_emit(reply)"):
            with self.subTest(kept=kept):
                self.assertIn(kept, self.set_world)

    def test_the_scans_can_fail(self) -> None:
        """The control: a synthetic branch that registers in place unconditionally is caught."""
        synthetic = ('if cmd == "set_world":\n    register_boxes_in_place(_planner.scene_collision_checker, s, g)\n'
                     '    if _WORLD_IN_PLACE and boxes_only(world_scene):\n        pass\n')
        block = _block(synthetic, 'if cmd == "set_world":')
        guard = block.find("if _WORLD_IN_PLACE and boxes_only(")
        call = block.find("register_boxes_in_place(_planner.scene_collision_checker")
        self.assertFalse(0 <= guard < call)


class TheClientAndTheConfigTests(unittest.TestCase):
    def test_the_variable_is_written_only_where_the_cell_asks_and_a_leftover_is_never_inherited(self) -> None:
        from src.robot.safety.planning.curobo_client import CuroboPlanClient

        with mock.patch.dict(os.environ, {ENV_WORLD_IN_PLACE: "1"}):
            self.assertNotIn(ENV_WORLD_IN_PLACE, CuroboPlanClient()._sidecar_env())  # noqa: SLF001
            self.assertEqual("1", CuroboPlanClient(world_in_place=True)._sidecar_env()[ENV_WORLD_IN_PLACE])  # noqa: SLF001

    def test_the_key_is_off_by_default_and_reaches_the_client_through_the_reservation(self) -> None:
        from src.config.schema.robot import RobotConfig
        from src.robot.safety.planning.curobo_client import CuroboPlanClient
        from src.robot.safety.planning.reservation import PlannerReservation

        today = RobotConfig.model_validate({"vendor": "ur"})
        self.assertFalse(today.safety.planning_world.register_in_place)
        self.assertFalse(PlannerReservation.from_config(robot_cfg=today).world_in_place)
        asked = RobotConfig.model_validate({"vendor": "ur", "safety": {"planning_world": {"register_in_place": True}}})
        reservation = PlannerReservation.from_config(robot_cfg=asked)
        self.assertTrue(reservation.world_in_place)
        client = CuroboPlanClient()
        client.reserve_world(reservation)
        self.assertEqual("1", client._sidecar_env()[ENV_WORLD_IN_PLACE])  # noqa: SLF001


if __name__ == "__main__":
    unittest.main()
