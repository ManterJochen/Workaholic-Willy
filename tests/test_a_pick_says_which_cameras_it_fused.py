"""A pick on a fused cell says which cameras its grasp was planned from, and a tree that cannot fuse says so first.

One depth view sees one side of a part and a parallel jaw closes on two. A cell with two or more calibrated cameras
fuses each object's surface across them into one BASE cloud and plans the grasp on that (measured in simulation on the
datagen reference: top-1 43.5 % from one view, 55.9 % fused). The dangerous state is not "fusion is off" but "the
operator believes it is on": a camera that stopped delivering, an association that matched nothing, or a switch left
off all produce a cell that grasps from one view and looks healthy from outside. Two things close that here:

* **Asked before anything opens.** `CameraFusionPlan` reads the tree and names, in one sentence, every piece a cell is
  missing to fuse: both switches (`fusion.enabled` builds the other cameras' CAMERA to BASE, `fusion.geometry.enabled`
  hands the fused cloud to the generator), two or more cameras in `fusion.cameras` with the primary among them, and
  each an enabled RGB-D rig that declares its calibration. The owner's cell today has one wrist camera and is refused
  with what a fused cell needs and where the reference profile is.
* **Read off the attempt, not the configuration.** `fused_views` names the cameras behind the cloud the attempt's
  object was handed to the generator as, the camera it is grasped from first, and is empty whenever that cloud came
  from one camera. It is NOT the frame telemetry's `fused_views`, which lists every other camera that segmented
  anything, matched or not. `fused_objects` is how many objects of that frame gained a second camera's surface.

NEVER RUN ON HARDWARE: no physical multi-camera cell exists. Everything below is proved against fakes and real trees.
"""

from __future__ import annotations

import ast
import json
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np

from src.config.loader import ConfigError
from src.config.schema.robot.grasping_schema import FusionGeometryConfig
from src.config.tree import LoadedTree, load_tree
from src.geometry import Frame, Transform
from src.robot.execution.autonomous_grasp.config import GraspMode, _profile_for
from src.robot.execution.autonomous_grasp.report import AutonomousGraspOutcome, AutonomousGraspReport
from src.robot.execution.camera_fusion import CameraFusionPlan
from src.robot.execution.pick_run import PickAttempt, PickRun, Recording
from src.robot.execution.runtime_pick import RuntimePickService
from src.robot.grasping.loop import pick_loop
from src.robot.grasping.loop.pick_loop import BinPickingOrchestrator
from src.robot.grasping.motion.execution_policy import PolicyOutcome, PolicyReport
from src.robot.grasping.types.feedback import GraspResult
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint
from src.robot.grasping.types.perception import CameraObservation, PerceptionFrame

_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLE = _ROOT / "examples" / "real_robot" / "17_pick_with_fused_cameras.py"

# ---------------------------------------------------------------------------------------------------------------------
# The trees: the shipped two-camera reference, and a one-wrist-camera cell like the owner's
# ---------------------------------------------------------------------------------------------------------------------

#: A wrist camera's calibration block, as the eye-in-hand sweep writes it: the artifact is never opened by a plan.
_ON_THE_WRIST = {"mounting_mode": "eye_in_hand", "artifact_path": "../calibration/real/eih_wrist.json",
                 "shutter_motion_tolerance_mm": 1.0, "shutter_motion_tolerance_deg": 0.5}


def _eth2(**values: Any) -> LoadedTree:
    """The reference profile, `config/robot/robot.eth2.yaml` with `config/camera/cam.eth2.yaml`, on the bench arm."""
    tree = load_tree("ur5e,eth2")
    return tree.with_values(values) if values else tree


def _one_wrist_camera() -> LoadedTree:
    """The owner's cell today, in the base tree's words: one enabled RGB-D rig, on the wrist, and nothing fused."""
    return load_tree(None).with_values({
        "camera.cameras.primary_rig_id": "realsense_d435",
        "camera.cameras.rigs[0].enabled": False,
        "camera.cameras.rigs[3].enabled": True,
        "camera.cameras.rigs[3].extrinsics": _ON_THE_WRIST,
    })


def _plan(tree: LoadedTree) -> CameraFusionPlan:
    assert tree.ok, tree.error
    return CameraFusionPlan.from_tree(tree)


class TheReferenceCellFusesTests(unittest.TestCase):
    def test_the_eth2_profile_fuses_both_cameras_primary_first(self) -> None:
        plan = _plan(_eth2())

        self.assertIsNone(plan.refusal())
        self.assertEqual(("cam_left", "cam_right"), plan.cameras)
        self.assertEqual("cam_left", plan.primary_rig_id)
        self.assertEqual((), plan.missing)
        self.assertIn("'cam_left' (primary)", str(plan))
        self.assertIn("'cam_right'", str(plan))
        self.assertTrue(str(plan).isascii())
        json.dumps(plan.to_dict())

    def test_a_wrist_camera_is_one_of_the_views_a_cell_fuses(self) -> None:
        """The truth about wrist rigs: the fusion path places each frame of an eye-in-hand rig by the arm's pose at the
        shutter (`build_config_frame_resolvers` builds an `EyeInHandFrameResolver`, the cell stamps its frames, and
        `MappedCameraRig` passes a moved wrist frame through as a fault). So a plan does not refuse one; it says it."""
        plan = _plan(_eth2(**{"camera.cameras.rigs[1].extrinsics": _ON_THE_WRIST}))

        self.assertIsNone(plan.refusal())
        self.assertEqual(("cam_right",), plan.on_the_wrist)
        self.assertIn("'cam_right' (on the wrist)", str(plan))


class EachMissingPieceIsNamedTests(unittest.TestCase):
    """One case per piece, each on the reference tree with that one piece taken away."""

    def _refused(self, tree: LoadedTree, *said: str) -> str:
        plan = _plan(tree)
        refusal = plan.refusal()
        assert refusal is not None, "the tree cannot fuse and the plan did not refuse it"
        self.assertEqual((), plan.cameras, "a plan that refuses fuses no camera")
        self.assertTrue(refusal.startswith("this cell cannot fuse its cameras"), refusal)
        for words in said:
            self.assertIn(words, refusal)
        self.assertTrue(refusal.isascii())
        return refusal

    def test_geometry_fusion_switched_off(self) -> None:
        self._refused(_eth2(**{"robot.grasping.fusion.geometry.enabled": False}),
                      "robot.grasping.fusion.geometry.enabled is false")

    def test_the_switch_that_builds_the_other_cameras_transforms_switched_off(self) -> None:
        """`fusion.enabled` arms the shadow voxel grid, and the schema calls the two switches independent. They are not
        for a fused cell: `build_config_frame_resolvers` returns no resolver while it is off, so every other camera's
        view is dropped at the pick with a warning. A plan that ignored it would pass a cell that runs single view."""
        self._refused(_eth2(**{"robot.grasping.fusion.enabled": False}),
                      "robot.grasping.fusion.enabled is false", "CAMERA to BASE")

    def test_fewer_than_two_cameras_listed(self) -> None:
        refusal = self._refused(_eth2(**{"robot.grasping.fusion.cameras.cam_right.enabled": False}),
                                "robot.grasping.fusion.cameras lists one camera, 'cam_left'")
        self.assertIn("2 enabled RGB-D rigs, 'cam_left', 'cam_right'", refusal, "and what the tree could list")

    def test_the_primary_left_out_of_the_list(self) -> None:
        self._refused(_eth2(**{"robot.grasping.fusion.cameras.cam_left.enabled": False}),
                      "leaves out the primary rig 'cam_left'")

    def test_a_listed_rig_switched_off(self) -> None:
        self._refused(_eth2(**{"camera.cameras.rigs[1].enabled": False}), "rig 'cam_right' is switched off")

    def test_a_listed_rig_with_no_calibration(self) -> None:
        self._refused(_eth2(**{"camera.cameras.rigs[1].extrinsics": None}),
                      "rig 'cam_right' declares no calibration")

    def test_a_listed_rig_with_no_depth(self) -> None:
        tree = load_tree(None).with_values({
            "robot.grasping.fusion.enabled": True,
            "robot.grasping.fusion.geometry.enabled": True,
            "robot.grasping.fusion.cameras": {"webcam_main": {"enabled": True}, "realsense_d435": {"enabled": True}},
            "camera.cameras.rigs[3].enabled": True,
            "camera.cameras.rigs[3].extrinsics": {"mounting_mode": "eye_to_hand",
                                                  "artifact_path": "../calibration/real/eth_d435.json"},
        })
        self._refused(tree, "rig 'webcam_main' is a webcam_pair rig, with no depth of its own")

    def test_a_camera_that_names_no_rig_is_the_loader_s_refusal(self) -> None:
        """Refused where the tree loads, as the multiview README says; the plan passes the tree's own refusal on."""
        tree = _eth2(**{"robot.grasping.fusion.cameras": {"cam_x": {"enabled": True}}})

        self.assertFalse(tree.ok)
        with self.assertRaises(ConfigError):
            CameraFusionPlan.from_tree(tree)

    def test_the_refusal_is_one_sentence_of_what_is_missing_then_what_a_fused_cell_needs(self) -> None:
        plan = _plan(_eth2(**{"robot.grasping.fusion.geometry.enabled": False,
                              "camera.cameras.rigs[1].extrinsics": None}))
        refusal = plan.refusal()
        assert refusal is not None

        missing, needs = refusal.split(". ", 1)
        self.assertEqual(2, len(plan.missing))
        for clause in plan.missing:
            self.assertIn(clause, missing)
        self.assertIn("config/robot/robot.eth2.yaml", needs)
        self.assertIn("config/camera/cam.eth2.yaml", needs)


class TheOwnersCellTodayTests(unittest.TestCase):
    """One camera, on the wrist, and no fusion block: refused cleanly, with what a fused cell needs."""

    def test_it_is_refused_with_what_it_has_and_what_a_fused_cell_needs(self) -> None:
        plan = _plan(_one_wrist_camera())
        refusal = plan.refusal()

        assert refusal is not None
        self.assertEqual((), plan.cameras)
        self.assertIn("robot.grasping.fusion.enabled and robot.grasping.fusion.geometry.enabled are false", refusal)
        self.assertIn("robot.grasping.fusion.cameras lists no camera", refusal)
        self.assertIn("one enabled RGB-D rig, 'realsense_d435' (on the wrist)", refusal)
        self.assertIn("two or more", refusal)
        self.assertIn("config/robot/robot.eth2.yaml", refusal)
        self.assertNotIn("leaves out the primary", refusal, "nothing is listed, so the primary is not news")
        self.assertIn("no camera fusion", str(plan))

    def test_the_base_tree_is_refused_as_well(self) -> None:
        self.assertIsNotNone(_plan(load_tree(None)).refusal())


# ---------------------------------------------------------------------------------------------------------------------
# The pick loop: what each attempt records
# ---------------------------------------------------------------------------------------------------------------------

_IDENTITY = Transform.identity(from_frame=Frame.CAMERA, to_frame=Frame.BASE)
_K = np.array([[400.0, 0.0, 16.0], [0.0, 400.0, 16.0], [0.0, 0.0, 1.0]])
#: Two parts that never overlap: a 12 px square and a 6 px one.
_A = ("a", 10, 22)
_B = ("b", 24, 30)


def _frame(*boxes: tuple[str, int, int]) -> PerceptionFrame:
    """One labelled square mask per ``(label, lo, hi)``, on a flat depth of 500 mm."""
    segmentations = []
    for label, lo, hi in boxes:
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[lo:hi, lo:hi] = 1
        segmentations.append(SimpleNamespace(mask=mask, label=label))
    return PerceptionFrame(depth_map=np.full((32, 32), 500.0), intrinsics=_K, segmentations=tuple(segmentations))


class _Resolver:
    """A fixed camera whose CAMERA to BASE is the identity, so every camera measures one shared frame."""

    def camera_to_base_for_frame(self, _frame: Any, *, arm: Any = None) -> Transform:
        return _IDENTITY


class _Rig:
    def __init__(self, *observations: CameraObservation) -> None:
        self._observations = observations
        self.last_failures: dict[str, str] = {}

    def acquire_all(self) -> tuple[CameraObservation, ...]:
        return self._observations


class _Calculator:
    """One candidate for every object, scored by its label, and a record of the cloud each was handed."""

    render_debug_images = False

    def __init__(self, scores: dict[str, float] | None = None) -> None:
        self.scores = dict(scores or {})
        self.clouds: list[Any] = []

    def compute_result(self, seg: Any, *_args: Any, **kwargs: Any) -> GraspResult:
        self.clouds.append(kwargs.get("geometry_points_base_mm"))
        score = self.scores.get(str(getattr(seg, "label", "")), 0.9)
        grasp = GraspPoint(position=np.array([0.0, 0.0, 500.0]), approach=np.array([0.0, 0.0, 1.0]),
                           axis=np.array([1.0, 0.0, 0.0]), grip_width_mm=40.0, score=score, frame=GraspFrame.BASE)
        return GraspResult(candidates=(grasp,), top_score=score)


class _Policy:
    def execute(self, _grasp: Any) -> PolicyReport:
        return PolicyReport(outcome=PolicyOutcome.EXECUTED)


def _cell(primary: PerceptionFrame, *others: tuple[str, PerceptionFrame], fused: bool = True,
          calculator: _Calculator | None = None, **overrides: Any) -> BinPickingOrchestrator:
    """A fixed-camera cell whose primary is `left`, and whose other cameras deliver the frames given."""
    return BinPickingOrchestrator(
        arm=SimpleNamespace(),  # type: ignore[arg-type]
        calculator=calculator or _Calculator(),  # type: ignore[arg-type]
        perception=SimpleNamespace(acquire=lambda: primary),  # type: ignore[arg-type]
        frame_resolver=SimpleNamespace(camera_to_base_for_frame=lambda *_a, **_k: _IDENTITY),  # type: ignore[arg-type]
        policy=_Policy(),  # type: ignore[arg-type]
        max_attempts=1,
        primary_camera_id="left",
        multi_camera_perception=_Rig(*(CameraObservation(camera_id=name, frame=frame) for name, frame in others)),
        camera_frame_resolvers={name: _Resolver() for name, _frame_in in others},  # type: ignore[misc]
        fusion_geometry_config=FusionGeometryConfig(enabled=True, **overrides.pop("geometry", {})) if fused else None,
        **overrides,
    )


def _last_attempt(orch: BinPickingOrchestrator) -> pick_loop.PickAttempt:
    report = orch.run()
    return report.attempts[-1]


class TheLoopAttemptTests(unittest.TestCase):
    def test_two_cameras_that_agree_on_the_target_are_both_named_primary_first(self) -> None:
        calculator = _Calculator()
        attempt = _last_attempt(_cell(_frame(_A), ("right", _frame(_A)), calculator=calculator))

        self.assertEqual("executed", attempt.action)
        self.assertEqual(("left", "right"), attempt.fused_views)
        self.assertEqual(1, attempt.fused_objects)
        self.assertIsNotNone(calculator.clouds[0], "the fused cloud reached the generator, which is what is named")

    def test_fusion_switched_off_names_nothing(self) -> None:
        attempt = _last_attempt(_cell(_frame(_A), ("right", _frame(_A)), fused=False))

        self.assertEqual((), attempt.fused_views)
        self.assertEqual(0, attempt.fused_objects)

    def test_a_second_camera_that_saw_something_else_contributed_nothing(self) -> None:
        """⛔ THE HONESTY CASE. The frame's telemetry lists `right`, because it segmented something; nothing of it
        reached this grasp, so the attempt names no camera."""
        orch = _cell(_frame(_A), ("right", _frame(_B)))
        attempt = _last_attempt(orch)

        self.assertEqual(["right"], orch._fusion_geometry_telemetry["fused_views"])
        self.assertEqual((), attempt.fused_views)
        self.assertEqual(0, attempt.fused_objects)

    def test_a_second_camera_that_grounded_nothing_contributed_nothing(self) -> None:
        attempt = _last_attempt(_cell(_frame(_A), ("right", _frame())))

        self.assertEqual((), attempt.fused_views)
        self.assertEqual(0, attempt.fused_objects)

    def test_a_neighbour_fused_is_counted_and_not_claimed_for_the_target(self) -> None:
        """The target `a` scores best and only its neighbour `b` was seen twice: the count is the frame's, the views
        are the target's."""
        attempt = _last_attempt(_cell(_frame(_A, _B), ("right", _frame(_B)),
                                      calculator=_Calculator({"a": 0.9, "b": 0.5})))

        self.assertEqual(0, attempt.target_index)
        self.assertEqual((), attempt.fused_views)
        self.assertEqual(1, attempt.fused_objects)

    def test_a_camera_fused_with_its_own_copy_is_not_a_second_view(self) -> None:
        """A rig that lists the primary under its own name fuses the primary with itself; one camera twice is still
        one view."""
        attempt = _last_attempt(_cell(_frame(_A), ("left", _frame(_A))))

        self.assertEqual((), attempt.fused_views)
        self.assertEqual(0, attempt.fused_objects)

    def test_an_external_target_cloud_that_replaced_the_fused_one_is_not_claimed(self) -> None:
        """A simulator runner's ground-truth cloud wins over the fused one; the grasp was not planned on the fusion."""
        calculator = _Calculator()
        attempt = _last_attempt(_cell(_frame(_A), ("right", _frame(_A)), calculator=calculator, target_label="a",
                                      external_target_geometry_base_mm=np.zeros((4, 3))))

        self.assertEqual((), attempt.fused_views)
        self.assertEqual(1, attempt.fused_objects, "the frame still fused it; this grasp did not use it")
        self.assertEqual(4, len(calculator.clouds[0]))

    def test_an_object_only_other_cameras_saw_names_those_cameras(self) -> None:
        """`promote_unmatched`: `right` and `far` both see `b`, the primary does not, and the grasp is planned from
        `right` on both surfaces."""
        attempt = _last_attempt(_cell(_frame(_A), ("right", _frame(_B)), ("far", _frame(_B)),
                                      calculator=_Calculator({"a": 0.5, "b": 0.9}),
                                      geometry={"promote_unmatched": True}))

        self.assertEqual(1, attempt.target_index, "the promoted object follows the primary's own")
        self.assertEqual(("right", "far"), attempt.fused_views)
        self.assertEqual(1, attempt.fused_objects)

    def test_an_attempt_that_found_no_segmentation_says_nothing(self) -> None:
        attempt = _last_attempt(_cell(_frame(), ("right", _frame(_A))))

        self.assertIsNone(attempt.target_index)
        self.assertEqual((), attempt.fused_views)
        self.assertEqual(0, attempt.fused_objects)


# ---------------------------------------------------------------------------------------------------------------------
# The service report and the campaign
# ---------------------------------------------------------------------------------------------------------------------


def _report(**overrides: Any) -> AutonomousGraspReport:
    base = AutonomousGraspReport(outcome=AutonomousGraspOutcome.SUCCEEDED, mode=GraspMode.EASY,
                                 profile=_profile_for(GraspMode.EASY))
    return replace(base, **overrides)


def _loop_attempt(**fields: Any) -> pick_loop.PickAttempt:
    return pick_loop.PickAttempt(attempt_index=0, target_index=0, reasons=(), score=0.9, action="executed", **fields)


class TheServiceReportTests(unittest.TestCase):
    def test_it_reads_the_last_attempt_of_its_pick(self) -> None:
        report = _report(pick_report=SimpleNamespace(attempts=(
            _loop_attempt(), _loop_attempt(fused_views=("left", "right"), fused_objects=2))))

        self.assertEqual(("left", "right"), report.fused_views)
        self.assertEqual(2, report.fused_objects)
        self.assertIn("  fused      left + right; 2 object(s) fused in its frame", report.render())
        self.assertEqual(["left", "right"], report.to_dict()["fused_views"])
        self.assertEqual(2, report.to_dict()["fused_objects"])
        json.dumps(report.to_dict())

    def test_a_pick_with_nothing_fused_prints_no_fusion_line(self) -> None:
        for label, pick in (("no pick report", None), ("one view", SimpleNamespace(attempts=(_loop_attempt(),)))):
            with self.subTest(label):
                report = _report(pick_report=pick)
                self.assertEqual((), report.fused_views)
                self.assertEqual(0, report.fused_objects)
                self.assertNotIn("fused", report.render())

    def test_a_double_that_answers_every_attribute_names_nothing(self) -> None:
        report = _report(pick_report=MagicMock())

        self.assertEqual((), report.fused_views)
        self.assertEqual(0, report.fused_objects)

    def test_one_camera_alone_is_never_a_fusion(self) -> None:
        report = _report(pick_report=SimpleNamespace(attempts=(_loop_attempt(fused_views=("left",),
                                                                              fused_objects=True),)))

        self.assertEqual((), report.fused_views)
        self.assertEqual(0, report.fused_objects, "a bool is not a count")

    def test_a_neighbour_fused_says_the_grasp_had_one_view(self) -> None:
        report = _report(pick_report=SimpleNamespace(attempts=(_loop_attempt(fused_objects=1),)))

        self.assertIn("  fused      not this object; 1 other object(s) in its frame", report.render())


class _Service:
    """A service over a real orchestrator: its pick loop, its session report and the service report around it."""

    def __init__(self, orch: BinPickingOrchestrator) -> None:
        self.runtime = RuntimePickService(orchestrator=orch)

    def pick(self) -> AutonomousGraspReport:
        session = self.runtime.run_attempt()
        outcome = AutonomousGraspOutcome.SUCCEEDED if session.succeeded else AutonomousGraspOutcome.EXECUTION_FAILED
        return _report(outcome=outcome, pick_report=session)


def _campaign(orch: BinPickingOrchestrator) -> PickAttempt:
    report = PickRun.from_service(_Service(orch), runs=1, recording=Recording.off()).execute()
    return report.attempts[0]


class TheCampaignAttemptTests(unittest.TestCase):
    def test_a_fused_pick_reaches_the_attempt_a_program_reads(self) -> None:
        """End to end: the pick loop, the session report, the service report, the campaign."""
        attempt = _campaign(_cell(_frame(_A), ("right", _frame(_A))))

        self.assertTrue(attempt.passed)
        self.assertEqual(("left", "right"), attempt.fused_views)
        self.assertEqual(1, attempt.fused_objects)
        self.assertIn("    fused views: left + right (1 object(s) fused in that frame)", str(attempt))
        payload = attempt.to_dict()
        json.dumps(payload)
        self.assertEqual(["left", "right"], payload["fused_views"])
        self.assertEqual(1, payload["fused_objects"])

    def test_a_single_view_pick_prints_as_it_always_did(self) -> None:
        fused_off = _campaign(_cell(_frame(_A), ("right", _frame(_A)), fused=False))

        self.assertEqual((), fused_off.fused_views)
        self.assertEqual(0, fused_off.fused_objects)
        self.assertNotIn("fused", str(fused_off))
        self.assertEqual([], fused_off.to_dict()["fused_views"])

    def test_a_neighbour_fused_prints_that_the_grasp_had_one_view(self) -> None:
        attempt = _campaign(_cell(_frame(_A, _B), ("right", _frame(_B)), calculator=_Calculator({"a": 0.9, "b": 0.5})))

        self.assertEqual((), attempt.fused_views)
        self.assertIn("    planned from one view (1 other object(s) fused in that frame)", str(attempt))

    def test_a_service_that_says_it_in_another_type_leaves_the_attempt_as_it_was(self) -> None:
        """Only the type the report promises counts: a list of names, or a bool for a count, is read as nothing."""
        class _Loose:
            def pick(self) -> Any:
                return SimpleNamespace(outcome="succeeded", fused_views=["left", "right"], fused_objects=True,
                                       failure_summary=lambda: "")

        report = PickRun.from_service(_Loose(), runs=1, recording=Recording.off()).execute()

        self.assertEqual((), report.attempts[0].fused_views)
        self.assertEqual(0, report.attempts[0].fused_objects)


# ---------------------------------------------------------------------------------------------------------------------
# The example
# ---------------------------------------------------------------------------------------------------------------------


class TheExampleTests(unittest.TestCase):
    def test_it_asks_the_tree_before_it_builds_a_cell(self) -> None:
        """The refusal has to come before anything opens: a camera, a model, the arm. Checked on the source, because a
        real_robot example is never executed here."""
        source = _EXAMPLE.read_text(encoding="utf-8")

        asked = source.index("CameraFusionPlan.from_tree(")
        refused = source.index("raise SystemExit(plan.refusal())")
        built = source.index("Cell.from_tree(")
        self.assertLess(asked, refused)
        self.assertLess(refused, built)

    def test_it_reads_the_views_off_the_attempt(self) -> None:
        tree = ast.parse(_EXAMPLE.read_text(encoding="utf-8"))
        read = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}

        self.assertIn("fused_views", read)
        self.assertIn("fused_objects", read)


if __name__ == "__main__":
    unittest.main()
