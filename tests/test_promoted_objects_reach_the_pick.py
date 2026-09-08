"""A camera other than the primary can introduce an object, and the default path does not change.

⛔⛔ WHAT THIS FEATURE FIXES. Detection and segmentation already run on EVERY camera of a fused cell:
each extra camera costs a full detect and segment pass per pick. But the object list came from the
primary camera alone, and every other camera's blobs were matched against it, so a part only the
second camera could see matched nothing, because there was nothing for it to match. It was found,
segmented, back-projected into BASE, and dropped. If it was the last part in a bin, the cell reported
the bin empty while holding a positive detection of it in memory.

Two tests here carry the whole change and neither can pass by accident:

* **The control that fails if the feature is inert.** The same scene with the flag off and on. Off:
  one object. On: two, the second one promoted. A flag that changes nothing when switched on is the
  defect this repository built a wiring guard for, and a selector is as capable of being inert as a
  switch.
* **The control that fails if it changes the default path.** With the flag off, the object list is
  the primary frame's segmentations, in order, same masks. Not "the pick still works": the list
  itself, because that is the thing every measured number in this repository was taken against.

⚠ NEVER RUN ON HARDWARE. No physical multi-camera cell exists; this is proved against a fake rig.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from src.config.schema.robot.grasping_schema import FusionGeometryConfig
from src.geometry import Frame, Transform
from src.robot.grasping.loop.pick_loop import BinPickingOrchestrator
from src.robot.grasping.types.perception import CameraObservation, PerceptionFrame

_IDENTITY = Transform.identity(from_frame=Frame.CAMERA, to_frame=Frame.BASE)
_K = np.array([[400.0, 0.0, 32.0], [0.0, 400.0, 32.0], [0.0, 0.0, 1.0]])


def _frame(*boxes: tuple[int, int], depth: float = 500.0) -> PerceptionFrame:
    """One segmentation per box, each a 10x10 patch whose top-left corner is the box."""
    segs = []
    for row, col in boxes:
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[row:row + 10, col:col + 10] = 1
        segs.append(SimpleNamespace(mask=mask, label=f"obj_{row}_{col}"))
    return PerceptionFrame(
        depth_map=np.full((64, 64), depth),
        intrinsics=_K,
        segmentations=tuple(segs),
    )


class _Resolver:
    def camera_to_base_for_frame(self, _frame, *, arm=None):  # noqa: ANN001, ANN202, ARG002
        return _IDENTITY


class _Rig:
    def __init__(self, *observations: CameraObservation) -> None:
        self._observations = observations
        self.last_failures: dict[str, str] = {}

    def acquire_all(self) -> tuple[CameraObservation, ...]:
        return self._observations


def _orchestrator(*, promote: bool, rig) -> BinPickingOrchestrator:  # noqa: ANN001
    return BinPickingOrchestrator(
        arm=SimpleNamespace(),  # type: ignore[arg-type]
        calculator=SimpleNamespace(),  # type: ignore[arg-type]
        perception=SimpleNamespace(),  # type: ignore[arg-type]
        multi_camera_perception=rig,
        camera_frame_resolvers={"right": _Resolver()},  # type: ignore[dict-item]
        primary_camera_id="left",
        fusion_geometry_config=FusionGeometryConfig(
            enabled=True, metric="centroid", promote_unmatched=promote),
    )


class TheFlagChangesTheSceneTests(unittest.TestCase):
    """The scene: the primary sees one part, the second camera sees that part and another."""

    def _scene(self, *, promote: bool):
        primary = _frame((10, 10))
        other = _frame((10, 10), (40, 40))
        orch = _orchestrator(promote=promote,
                             rig=_Rig(CameraObservation(camera_id="right", frame=other)))
        fused = orch._fused_scene(primary, _IDENTITY)
        return orch._scene_objects(primary, _IDENTITY, fused)

    def test_off_the_second_camera_s_extra_part_is_dropped(self) -> None:
        """The behaviour this repository measured everything against, kept exactly."""
        objects = self._scene(promote=False)

        self.assertEqual(1, len(objects))
        self.assertFalse(objects[0].promoted)

    def test_on_it_becomes_an_object(self) -> None:
        """⭐ THE CONTROL. If this passes with the flag off as well, the flag is inert."""
        objects = self._scene(promote=True)

        self.assertEqual(2, len(objects), "the part only the second camera saw must survive")
        promoted = [o for o in objects if o.promoted]
        self.assertEqual(1, len(promoted))
        self.assertEqual("right", promoted[0].camera_id)

    def test_the_part_BOTH_cameras_saw_is_still_one_object(self) -> None:
        """The other half of the same assertion, and the one that would fail if grouping were
        skipped and every camera's blobs were simply concatenated: three entries instead of two."""
        objects = self._scene(promote=True)

        self.assertEqual(("left", "right"), objects[0].views_used)
        self.assertFalse(objects[0].promoted)


class TheDefaultListIsTheOldListTests(unittest.TestCase):
    def test_the_objects_are_the_primary_frame_s_segmentations_in_order(self) -> None:
        """Asserted on the LIST rather than on the outcome. An equivalent pick is not the claim; the
        claim is that the loop iterates the same things in the same order."""
        primary = _frame((10, 10), (30, 30), (50, 5))
        orch = _orchestrator(promote=False, rig=_Rig())

        objects = orch._scene_objects(primary, _IDENTITY, None)

        self.assertEqual(len(primary.segmentations), len(objects))
        for seg, obj in zip(primary.segmentations, objects, strict=True):
            self.assertIs(seg, obj.segmentation)
            self.assertIs(primary.depth_map, obj.depth_map)
            self.assertEqual("left", obj.camera_id)
            self.assertFalse(obj.promoted)

    def test_a_cell_with_no_other_camera_at_all_is_untouched(self) -> None:
        primary = _frame((10, 10), (30, 30))
        orch = BinPickingOrchestrator(
            arm=SimpleNamespace(),  # type: ignore[arg-type]
            calculator=SimpleNamespace(),  # type: ignore[arg-type]
            perception=SimpleNamespace(),  # type: ignore[arg-type]
        )

        objects = orch._scene_objects(primary, _IDENTITY, None)

        self.assertEqual(2, len(objects))
        self.assertFalse(any(o.promoted for o in objects))


class ThePromotedObjectGetsItsOwnCalculatorTests(unittest.TestCase):
    """A calculator is bound to one camera's intrinsics at construction. The wrong one produces
    plausible numbers in the wrong place, which is the failure mode with no symptom."""

    def test_the_map_routes_by_camera_id(self) -> None:
        primary_calc, right_calc = SimpleNamespace(name="primary"), SimpleNamespace(name="right")
        orch = BinPickingOrchestrator(
            arm=SimpleNamespace(),  # type: ignore[arg-type]
            calculator=primary_calc,  # type: ignore[arg-type]
            perception=SimpleNamespace(),  # type: ignore[arg-type]
            primary_camera_id="left",
            camera_calculators={"left": primary_calc, "right": right_calc},
        )

        self.assertIs(right_calc, orch.calculator_for("right"))
        self.assertIs(primary_calc, orch.calculator_for("left"))

    def test_no_map_means_the_cell_s_own_calculator(self) -> None:
        """A one-camera cell has one calculator and no map, and every object is the primary's."""
        calc = SimpleNamespace(name="only")
        orch = BinPickingOrchestrator(
            arm=SimpleNamespace(),  # type: ignore[arg-type]
            calculator=calc,  # type: ignore[arg-type]
            perception=SimpleNamespace(),  # type: ignore[arg-type]
        )

        self.assertIs(calc, orch.calculator_for("anything"))

    def test_a_camera_with_no_calculator_is_SAID_rather_than_silently_wrong(self) -> None:
        """Promotion can produce an object from a camera nobody built a calculator for. Refusing the
        pick over it would be the larger failure; computing it in the primary lens without saying so
        would be the quieter one."""
        calc = SimpleNamespace(name="primary")
        orch = BinPickingOrchestrator(
            arm=SimpleNamespace(),  # type: ignore[arg-type]
            calculator=calc,  # type: ignore[arg-type]
            perception=SimpleNamespace(),  # type: ignore[arg-type]
            camera_calculators={"left": calc},
        )

        with self.assertLogs("src.robot.grasping.loop.pick_loop", level="WARNING") as logs:
            self.assertIs(calc, orch.calculator_for("right"))

        self.assertIn("right", "\n".join(logs.output))
        self.assertIn("wrong intrinsics", "\n".join(logs.output))


if __name__ == "__main__":
    unittest.main()
