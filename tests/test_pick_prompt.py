"""A campaign's prompt reaches the detector, and the cell's own phrase comes back after it (Q6, owner 2026-09-18).

A cell is built with one grounding phrase ("object" on the console), and until this step a per-run prompt
was only an exact-match label filter. A phrase grounder labels a box with the words of the phrase it
matched, so on a physical cell "the red cube" filtered on a label no detection carried, while the phrase
the detector read never changed. `PickRun(prompt=)` and `service.set_prompt` set the phrase, the labels
the detector's words map onto and the filter together, and put all three back.

Honesty bucket (2): the real service, pick loop and live camera source; a fake streamer and a phrase
grounder double in place of the device and the weights. No hardware, no model.

Why each test is red on the code before this step is said in its docstring.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

from src.camera.setup.image_taking.frames import RGBDFrame
from src.geometry import Frame, Pose
from src.robot.core import MotionCommand, MotionResult, RobotCapabilities
from src.robot.execution.autonomous_grasp import (
    AutonomousGraspOutcome,
    AutonomousGraspService,
    GraspMode,
)
from src.robot.execution.pick_run import PickRun, Recording
from src.robot.grasping.types.feedback import GraspResult
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint
from src.robot.grasping.types.perception import MappedCameraRig, PerceptionFrame
from src.robot.perception import RealSenseVisionPerceptionSource

_CAPS = RobotCapabilities(
    vendor="ur", model="ur5e", dof=6, supports_joint_move=True, supports_linear_move=True,
    supports_async_move=False, has_native_fk=True, has_native_ik=True, has_force_control=False,
    is_simulated=False,
)
_K = np.array([[400.0, 0.0, 32.0], [0.0, 400.0, 32.0], [0.0, 0.0, 1.0]], dtype=np.float64)


class _Arm:
    """Moves wherever it is told and says so."""

    def __init__(self) -> None:
        self._tcp = Pose(position_mm=np.array([0.0, 0.0, 500.0]),
                         quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]), frame=Frame.BASE, label="home")

    @property
    def capabilities(self) -> RobotCapabilities:
        return _CAPS

    def get_tcp_pose(self) -> Pose:
        return self._tcp

    def move(self, pose: Pose, **_: object) -> MotionResult:
        if pose.frame == Frame.BASE:
            self._tcp = pose
        return MotionResult.executed(MotionCommand.MOVE_TO, target_pose=pose)


class _Calculator:
    """One graspable candidate in BASE for every segmentation it is handed."""

    def compute_result(self, *_args: object, **_kwargs: object) -> GraspResult:
        return GraspResult(
            candidates=(GraspPoint(
                position=np.array([100.0, 50.0, 400.0]), approach=np.array([0.0, 0.0, 1.0]),
                axis=np.array([1.0, 0.0, 0.0]), grip_width_mm=40.0, score=0.9, frame=GraspFrame.BASE,
                label="test",
            ),),
            reasons=(), top_score=0.9,
        )


class _Streamer:
    """One fixed RGB-D frame: a plane at 500 mm with a box standing 40 mm toward the camera."""

    def __init__(self) -> None:
        depth = np.full((64, 64), 500, dtype=np.uint16)
        depth[24:40, 24:40] = 460
        self._frame = RGBDFrame(color=np.zeros((64, 64, 3), dtype=np.uint8), depth=depth)

    def grab(self) -> RGBDFrame:
        return self._frame

    def get_intrinsics(self) -> np.ndarray:
        return _K.copy()


@dataclass(frozen=True)
class _Seg:
    """Frozen with ``.mask`` and ``.label``, so the source's ``dataclasses.replace`` works as on the real type."""

    mask: np.ndarray
    label: str


class _PhraseGrounder:
    """Stands in for GroundingDINO behind the perception backend: one box per frame, and every caption kept.

    It labels the box with the words of the phrase it matched and drops the article, as a phrase grounder
    does ("the red cube" comes back as "red cube"). That is why a label filter set to the typed text alone
    matched nothing on a physical cell.
    """

    _ARTICLES = frozenset({"the", "a", "an"})

    def __init__(self) -> None:
        self.captions: list[str] = []

    def perceive(self, image_bgr: Any, prompt: str) -> tuple[Any, ...]:
        self.captions.append(prompt)
        label = " ".join(w for w in prompt.split() if w.lower() not in self._ARTICLES) or prompt
        mask = np.zeros(np.asarray(image_bgr).shape[:2], dtype=np.uint8)
        mask[24:40, 24:40] = 1
        detection = SimpleNamespace(box=[24.0, 24.0, 40.0, 40.0], label=label, score=0.9)
        return (SimpleNamespace(detection=detection, segmentation=_Seg(mask=mask, label=label)),)


def _source(grounder: _PhraseGrounder, phrase: str = "object") -> RealSenseVisionPerceptionSource:
    return RealSenseVisionPerceptionSource(streamer=_Streamer(), backend=grounder, prompt=phrase, warmup_grabs=0)


def _cell() -> tuple[AutonomousGraspService, RealSenseVisionPerceptionSource, _PhraseGrounder]:
    """A service over the live camera source, built with the console's phrase."""
    grounder = _PhraseGrounder()
    source = _source(grounder)
    service = AutonomousGraspService.from_components(
        arm=_Arm(),               # type: ignore[arg-type]
        calculator=_Calculator(),  # type: ignore[arg-type]
        perception=source,
        mode=GraspMode.EASY,
    )
    return service, source, grounder


class _FrameOnly:
    """A source that grounds no phrase, as the rehearsal scene: one unlabelled box, no ``set_prompt``."""

    def acquire(self) -> PerceptionFrame:
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[10:22, 10:22] = 1
        return PerceptionFrame(depth_map=np.full((32, 32), 500.0), intrinsics=_K.copy(),
                               segmentations=(SimpleNamespace(mask=mask),))


class ACampaignPromptReachesTheDetectorTests(unittest.TestCase):

    def test_a_campaign_grounds_its_prompt_and_the_cell_phrase_comes_back(self) -> None:
        """The owner's words: PickRun(prompt=) sets the grounding phrase and the label filter per campaign
        and restores both.

        Red before this step: `PickRun.from_service` takes no `prompt` (TypeError).
        """
        service, source, grounder = _cell()

        report = PickRun.from_service(
            service, runs=2, recording=Recording.off(), prompt="the red cube",
        ).execute()

        self.assertEqual(grounder.captions, ["the red cube", "the red cube"], "the detector read another phrase")
        self.assertTrue(report.passed, report.render())
        self.assertEqual(source.prompt, "object", "the cell's own phrase did not come back")
        self.assertEqual(source.object_labels, ())
        self.assertIsNone(service.runtime.orchestrator.target_label, "the label filter outlived the campaign")

    def test_a_blank_prompt_is_refused_at_the_factory(self) -> None:
        """A campaign that grounds nothing would build and connect a cell for picks that find nothing.

        Red before this step: the factory takes no `prompt` (TypeError, not ValueError).
        """
        service, _, _ = _cell()
        with self.subTest("from_service"), self.assertRaises(ValueError):
            PickRun.from_service(service, runs=1, recording=Recording.off(), prompt="   ")
        # The cell door refuses before it is ever asked to build, so a bare stand-in is enough.
        with self.subTest("from_cell"), self.assertRaises(ValueError):
            PickRun.from_cell(SimpleNamespace(), runs=1, recording=Recording.off(), prompt="")  # type: ignore[arg-type]


class TheServiceSetsAllThreeTests(unittest.TestCase):

    def test_the_label_filter_alone_found_nothing_and_the_prompt_finds_the_object(self) -> None:
        """The defect and its repair on one cell.

        The first half is today's console, green before and after: the label filter alone, while the
        detector still grounds "object", reports the prompted label as not found. The second half is red
        before this step: the service has no `set_prompt` (AttributeError).
        """
        service, _, grounder = _cell()
        service.set_target_label("the red cube")
        before = service.pick()
        self.assertEqual(set(grounder.captions), {"object"})
        self.assertIsNot(before.outcome, AutonomousGraspOutcome.SUCCEEDED)
        self.assertIn("target_label_not_found", before.failure_summary())
        service.set_target_label(None)

        service.set_prompt("the red cube")
        after = service.pick()

        self.assertEqual(grounder.captions[-1], "the red cube")
        self.assertIs(after.outcome, AutonomousGraspOutcome.SUCCEEDED, after.render())

    def test_set_prompt_returns_what_it_replaced_and_takes_it_back(self) -> None:
        """Red before this step: the service has no `set_prompt`."""
        service, source, _ = _cell()
        orchestrator = service.runtime.orchestrator

        previous = service.set_prompt("the red cube")

        self.assertEqual((previous.phrase, previous.target_label, previous.object_labels), ("object", None, ()))
        self.assertEqual(source.prompt, "the red cube")
        self.assertEqual(source.object_labels, ("the red cube",))
        self.assertEqual(orchestrator.target_label, "the red cube")

        replaced = service.set_prompt(previous)

        self.assertEqual(replaced.phrase, "the red cube")
        self.assertEqual((source.prompt, source.object_labels, orchestrator.target_label), ("object", (), None))

    def test_every_fused_camera_grounds_the_same_phrase(self) -> None:
        """A fused cell shares one set of weights and one phrase; a second camera left on the build phrase
        would ground another object than the one the primary targets.

        Red before this step: the service has no `set_prompt`.
        """
        service, _, _ = _cell()
        second = _source(_PhraseGrounder())
        service.runtime.orchestrator.multi_camera_perception = MappedCameraRig({"left": second})

        previous = service.set_prompt("the red cube")
        self.assertEqual((second.prompt, second.object_labels), ("the red cube", ("the red cube",)))

        service.set_prompt(previous)
        self.assertEqual((second.prompt, second.object_labels), ("object", ()))

    def test_a_cell_that_grounds_no_phrase_still_filters_on_the_prompt(self) -> None:
        """Fail closed on a rehearsal: no phrase to set, and the cell does not pick whatever it shows.

        Red before this step: the service has no `set_prompt`.
        """
        service = AutonomousGraspService.from_components(
            arm=_Arm(),               # type: ignore[arg-type]
            calculator=_Calculator(),  # type: ignore[arg-type]
            perception=_FrameOnly(),
            mode=GraspMode.EASY,
        )

        previous = service.set_prompt("the red cube")
        report = service.pick()

        self.assertEqual(previous.phrase, "", "a cell that grounds nothing has no phrase to give back")
        self.assertEqual(service.runtime.orchestrator.target_label, "the red cube")
        self.assertIsNot(report.outcome, AutonomousGraspOutcome.SUCCEEDED)
        self.assertIn("target_label_not_found", report.failure_summary())

    def test_a_blank_text_is_refused_before_anything_changes(self) -> None:
        """Red before this step: AttributeError rather than ValueError."""
        service, source, _ = _cell()
        with self.assertRaises(ValueError):
            service.set_prompt("  ")
        self.assertEqual(source.prompt, "object")
        self.assertIsNone(service.runtime.orchestrator.target_label)


class TheCameraSourceTests(unittest.TestCase):

    def test_the_source_grounds_the_new_phrase_from_the_next_frame(self) -> None:
        """No reopen and no reload: the same source object, the next frame, the new phrase, and the
        detector's words mapped onto it.

        Red before this step: the source has no `set_prompt`.
        """
        grounder = _PhraseGrounder()
        source = _source(grounder)
        source.acquire()

        source.set_prompt("a mug", object_labels=("a mug",))
        frame = source.acquire()

        self.assertEqual(grounder.captions, ["object", "a mug"])
        self.assertEqual(frame.segmentations[0].label, "a mug", "the detector's 'mug' was not mapped onto the prompt")
        self.assertEqual((source.prompt, source.object_labels), ("a mug", ("a mug",)))

    def test_an_empty_phrase_is_refused_before_a_frame_is_taken(self) -> None:
        """Red before this step: the source has no `set_prompt`."""
        source = _source(_PhraseGrounder())
        with self.assertRaises(ValueError):
            source.set_prompt("")
        self.assertEqual(source.prompt, "object")


class _Service:
    """A service double that answers a scripted list and records every prompt and label it was given."""

    def __init__(self, outcomes: list[str]) -> None:
        self.outcomes = outcomes
        self.calls = 0
        self.log: list[tuple[str, Any]] = []
        self.current: Any = "BUILD"

    def pick(self) -> Any:
        outcome = self.outcomes[self.calls]
        self.calls += 1
        if outcome == "BOOM":
            raise ValueError("a double that is wrong")
        return SimpleNamespace(outcome=outcome, fault=None, failure_summary=lambda: "reason=no_target")

    def set_prompt(self, prompt: Any) -> Any:
        self.log.append(("prompt", prompt))
        previous, self.current = self.current, prompt
        return previous

    def set_target_label(self, label: Any) -> None:
        self.log.append(("label", label))

    def enable_record_logging(self, path: Any, provenance: Any = None) -> None:
        pass


class TheCampaignPutsItBackTests(unittest.TestCase):

    def test_the_prompt_is_put_back_even_when_a_pick_raises(self) -> None:
        """Red before this step: `from_service` takes no `prompt`."""
        service = _Service(["BOOM"])
        report = PickRun.from_service(service, runs=1, recording=Recording.off(), prompt="the red cube").execute()

        self.assertTrue(report.raised)
        self.assertEqual(service.log, [("prompt", "the red cube"), ("prompt", "BUILD")])

    def test_a_prompt_and_a_label_end_on_what_the_prompt_replaced(self) -> None:
        """The label wins during the campaign and both are back after it.

        Red before this step: `from_service` takes no `prompt`.
        """
        service = _Service(["succeeded"])
        PickRun.from_service(
            service, runs=1, recording=Recording.off(), prompt="the red cube", target_label="red cube",
        ).execute()

        self.assertEqual(service.log, [
            ("prompt", "the red cube"), ("label", "red cube"), ("label", None), ("prompt", "BUILD"),
        ])

    def test_a_campaign_without_a_prompt_never_touches_the_phrase(self) -> None:
        """UNSET forwards nothing. A control, green before and after: a campaign that names no prompt must
        not reset a phrase somebody else set."""
        service = _Service(["succeeded"])
        PickRun.from_service(service, runs=1, recording=Recording.off()).execute()

        self.assertEqual([entry for entry in service.log if entry[0] == "prompt"], [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
