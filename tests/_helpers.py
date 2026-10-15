"""Shared test doubles for the bin-picking orchestrator suites (R10.2).

These are the byte-identical orchestrator test doubles that were duplicated verbatim across
``test_pick_loop`` / ``test_u2_probability_shadow`` / ``test_u4_probability_active_ranking`` /
``test_grasp_verification`` (that last one left on 2026-09-29 with the verifiers it tested). The
canonical (superset) forms live here so a contract change is made once; the call-sites are unchanged
(the no-arg / default forms reproduce every original call exactly).

INTENTIONALLY NOT shared (the anti-merge guard — these look alike but are different test contracts;
folding them would silently re-point assertions):

* ``test_u4``'s ``_ScriptedCalculator`` takes a ``points_factory`` and builds a fresh ``GraspResult``
  per call — a DIFFERENT contract from the ``results: list`` scripted calculator here. It stays local.
* The divergent ``_FakeArm`` shapes in other suites (MotionResult-arm, ik-arm, trivial-bool, no-tracking)
  are different contracts and are not hosted here.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from src.contracts import UNSET
from src.geometry import Frame, Pose
from src.robot.core.camera_world import CameraWorldStamp
from src.robot.core.motion_result import MotionCommand, MotionResult
from src.robot.grasping.types.feedback import GraspResult
from src.robot.grasping.types.perception import PerceptionFrame


class HoldsAWorld:
    """The two members of :class:`~src.robot.core.camera_world.HoldsAPlannerWorld`, for a double.

    A double that carries a live world for a keep-out test holds one and plans nothing, which is a real shape: holding
    a world and planning are separate questions with separate callers. Until 2026-10-13 both were read by name, so a
    double that set one attribute got the feature, and a driver missing one lost it in silence.
    """

    #: On the class, so a double whose own ``__init__`` does not chain up holds nothing instead of raising.
    _live_world: Any = None

    def __init__(self, world: object = None, *args: Any, **kwargs: Any) -> None:
        # Cooperative, because a double mixes this into a real driver: swallowing the chain here would skip that
        # driver's own __init__ and leave a half-built arm that answers plausibly and wrongly.
        self._live_world = world
        super().__init__(*args, **kwargs)

    @property
    def live_planner_world(self) -> Any:
        return self._live_world

    def set_live_planner_world(self, world: Any) -> None:
        self._live_world = world


class PlansItsMotions(HoldsAWorld):
    """A double that also answers what its motions stand on (``PlansMotions``).

    It plans nothing unless its test says so (``plans_paths``). One that plans reads a move as a planned move reads
    before a refresh vouched for it, UNSTATED, which is what the verbs read off these doubles while they answered
    ``plans_paths`` alone; one that plans nothing reads UNPLANNED.
    """

    plans_paths: bool = False
    camera_world_required: bool = False

    def camera_world_for_move(self, camera_world: Any = UNSET) -> CameraWorldStamp:
        if self.plans_paths:
            return CameraWorldStamp.unstated()
        return CameraWorldStamp.unplanned(f"{type(self).__name__} is a test double")

    def live_camera_world_wired(self) -> bool:
        return self._live_world is not None


class _FakeArm:
    """Minimal :class:`RobotArm` stand-in for the orchestrator's needs."""

    def __init__(self, tcp: Pose | None = None) -> None:
        self._tcp = tcp or Pose(
            position_mm=np.array([0.0, 0.0, 500.0]),
            quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            frame=Frame.BASE,
            label="home",
        )
        self.move_to_calls: list[Pose] = []

    def get_tcp_pose(self) -> Pose:
        return self._tcp

    def move(self, pose: Pose, **_: object) -> MotionResult:
        self.move_to_calls.append(pose)
        if pose.frame == Frame.BASE:
            self._tcp = pose
        return MotionResult.executed(MotionCommand.MOVE_TO, target_pose=pose)


class _FakePerception:
    def __init__(self, frames: list[PerceptionFrame]) -> None:
        self._frames = frames
        self.calls = 0

    def acquire(self) -> PerceptionFrame:
        idx = min(self.calls, len(self._frames) - 1)
        self.calls += 1
        return self._frames[idx]


class _ScriptedCalculator:
    """Returns a scripted sequence of :class:`GraspResult` per call."""

    def __init__(self, results: list[GraspResult]) -> None:
        self._results = results
        self.calls = 0

    def compute_result(self, *_args: object, **_kwargs: object) -> GraspResult:
        idx = min(self.calls, len(self._results) - 1)
        self.calls += 1
        return self._results[idx]


def _segmentation(shape: tuple[int, int] = (32, 32)) -> SimpleNamespace:
    mask = np.zeros(shape, dtype=np.uint8)
    mask[10:22, 10:22] = 1
    return SimpleNamespace(mask=mask)


def _perception_frame() -> PerceptionFrame:
    depth = np.full((32, 32), 500.0, dtype=np.float64)
    intrinsics = np.array(
        [[400.0, 0.0, 16.0], [0.0, 400.0, 16.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return PerceptionFrame(
        depth_map=depth,
        intrinsics=intrinsics,
        segmentations=(_segmentation(),),
    )
