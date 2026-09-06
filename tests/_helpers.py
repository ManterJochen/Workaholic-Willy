"""Shared test doubles for the bin-picking orchestrator suites (R10.2).

These are the byte-identical orchestrator test doubles that were duplicated verbatim across
``test_pick_loop`` / ``test_u2_probability_shadow`` / ``test_u4_probability_active_ranking`` /
``test_grasp_verification``. The canonical (superset) forms live here so a contract change is made
once; the call-sites are unchanged (the no-arg / default forms reproduce every original call exactly).

INTENTIONALLY NOT shared (the anti-merge guard — these look alike but are different test contracts;
folding them would silently re-point assertions):

* ``test_u4``'s ``_ScriptedCalculator`` takes a ``points_factory`` and builds a fresh ``GraspResult``
  per call — a DIFFERENT contract from the ``results: list`` scripted calculator here. It stays local.
* ``test_grasp_verification``'s ``_seg_at`` / ``_frame_with_seg`` (parametrised mask slices) and its
  ``_RefinementScriptedCalculator`` (initial/refined two-shot) are domain-specific and stay local.
* The divergent ``_FakeArm`` shapes in other suites (MotionResult-arm, ik-arm, trivial-bool, no-tracking)
  are different contracts and are not hosted here.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.geometry import Frame, Pose
from src.robot.grasping.types.feedback import GraspResult
from src.robot.grasping.types.perception import PerceptionFrame


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

    def move_to(self, pose: Pose, **_: object) -> None:
        self.move_to_calls.append(pose)
        if pose.frame == Frame.BASE:
            self._tcp = pose


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
