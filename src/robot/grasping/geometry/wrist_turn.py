"""Which of a grasp's two wrist turns keeps the wrist camera off what the camera saw, asked before the arm is.

A parallel jaw closes on the same two faces turned half a turn about its approach, so every grasp has two wrist turns
(``closing_axis``). A camera on the wrist stands on one side of the hand, and which of the two turns carries it into a
bin's wall is a question of the scene, not of the hand. The cell's natural orientation (``robot.natural_closing_axis``)
chose one turn for every grasp, the exact guard met the camera's housing only once the arm was asked, and a grasp whose
camera met a wall was refused although its twin cleared it: the grasp bench (2026-10-05) lost its deep-bin scenes so,
the camera 121 mm behind the jaws and 98 mm off their axis on the owner's cell.

So before the tries, each grasp's camera is placed both ways round:

* the housing is the guard's own (``body_link.WristBody``: the camera and its bracket, the body's margin in them),
  placed on the TCP by the calibration it was placed from;
* it is placed at the grasp and along the line in from the standoff, a step at most :data:`STEP_MM` apart;
* its least distance to the boxes the camera world builds of the neighbours (``scene_obstacles.seen_boxes``, the
  world's margin in them) is measured box to box, exactly (``scene_obstacles.box_distances_mm``);
* a grasp whose turn keeps the guard's distance (``perceived_min_distance_mm``) stays as it is; one whose twin alone
  keeps it is turned; one that keeps it neither way, or that nothing could measure, keeps its turn and is tried after
  every one that does, in its order.

Nothing is left out and nothing is admitted: the guard judges every grasp as it always did, and a box the world merges
past its slot budget only grows, so a turn this keeps clear is the turn the guard may still refuse, never the reverse
of a turn it would take. This chooses which of two equivalent turns is asked first, and in which order the grasps are
tried.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Final

import numpy as np

from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint

__all__ = ["STEP_MM", "WristHousing", "WristTurns", "turned_for_the_wrist"]

#: The farthest two placings of the housing along the line in lie apart, millimetres: a fifth of the D415's 20 mm depth
#: is not missed between them by a box the margin grew.
STEP_MM: Final[float] = 10.0

#: A distance this much under the guard's still keeps it: rounding, not a tolerance.
_ROUNDING_MM: Final[float] = 1e-6


@dataclass(frozen=True, slots=True, eq=False)
class WristHousing:
    """The boxes a wrist camera's body is, in the TCP frame: centres ``(K, 3)``, axes ``(K, 3, 3)`` (columns), half
    extents ``(K, 3)``, millimetres, the body's margin in them."""

    centres: np.ndarray
    turns: np.ndarray
    halves: np.ndarray

    @classmethod
    def of(cls, bodies: Sequence[Any]) -> "WristHousing | None":
        """The housing of every wrist body an arm carries (``body_link.WristBody``), or ``None`` where it carries none.

        Each body's boxes stand in its optical frame, which ``placement()`` places on tool0; the TCP stands on tool0 by
        the flange to TCP the body was placed from (``record_mm``), so the boxes stand on the TCP by the calibration's
        own CAMERA to TOOL.
        """
        centres: list[np.ndarray] = []
        turns: list[np.ndarray] = []
        halves: list[np.ndarray] = []
        for body in bodies:
            placement = np.asarray(body.placement(), dtype=np.float64).reshape(4, 4)
            record = np.asarray(body.record_mm, dtype=np.float64).reshape(4, 4)
            on_tcp = np.linalg.inv(record) @ placement
            for box in body.boxes:
                centre = np.asarray(box.centre_mm, dtype=np.float64).reshape(3)
                centres.append(on_tcp[:3, :3] @ centre + on_tcp[:3, 3])
                turns.append(on_tcp[:3, :3].copy())
                halves.append(np.asarray(box.half_extents_mm, dtype=np.float64).reshape(3))
        if not centres:
            return None
        return cls(centres=np.array(centres), turns=np.array(turns), halves=np.array(halves))

    def least_mm(self, poses: np.ndarray, centres_b: np.ndarray, turns_b: np.ndarray, halves_b: np.ndarray) -> float:
        """The housing's least distance to the boxes ``centres_b``/``turns_b``/``halves_b`` (BASE), placed on the TCP at
        every pose of ``poses`` ``(S, 4, 4)``, millimetres; 0 where one overlaps, ``inf`` where none is near."""
        poses = np.asarray(poses, dtype=np.float64).reshape(-1, 4, 4)
        if centres_b.shape[0] == 0 or poses.shape[0] == 0:
            return math.inf
        # Every housing box at every pose, in BASE.
        at = np.einsum("sij,kj->ski", poses[:, :3, :3], self.centres) + poses[:, None, :3, 3]
        axes = np.einsum("sij,kjl->skil", poses[:, :3, :3], self.turns)
        centres_a = at.reshape(-1, 3)
        turns_a = axes.reshape(-1, 3, 3)
        halves_a = np.tile(self.halves, (poses.shape[0], 1))
        # Only the pairs whose spheres about them come near enough to matter are measured.
        reach_a = np.linalg.norm(halves_a, axis=1)
        reach_b = np.linalg.norm(halves_b, axis=1)
        gaps = (np.linalg.norm(centres_a[:, None, :] - centres_b[None, :, :], axis=2)
                - reach_a[:, None] - reach_b[None, :])
        near_a, near_b = np.nonzero(gaps <= 50.0)
        if near_a.size == 0:
            return math.inf
        from src.robot.grasping.generation.scene_obstacles import box_distances_mm  # noqa: PLC0415

        distances = box_distances_mm(centres_a[near_a], turns_a[near_a], halves_a[near_a],
                                     centres_b[near_b], turns_b[near_b], halves_b[near_b])
        return float(distances.min())


@dataclass(frozen=True, slots=True)
class WristTurns:
    """What :func:`turned_for_the_wrist` did, counted, for one log line."""

    kept: int = 0
    turned: int = 0
    neither: int = 0
    unmeasured: int = 0

    def render(self) -> str:
        return (f"the wrist camera keeps the guard's distance at {self.kept} grasp(s) as turned, at {self.turned} only "
                f"turned half a turn (turned), at {self.neither} neither way (tried last)"
                + (f"; {self.unmeasured} not measured" if self.unmeasured else ""))


def _boxes_of(seen: Sequence[Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The seen boxes (``perceived.SeenBox``) as arrays: centres, axes and half extents, BASE millimetres."""
    if not seen:
        empty = np.zeros((0, 3), dtype=np.float64)
        return empty, np.zeros((0, 3, 3), dtype=np.float64), empty.copy()
    return (np.array([box.centre_mm for box in seen], dtype=np.float64).reshape(-1, 3),
            np.array([box.rotation for box in seen], dtype=np.float64).reshape(-1, 3, 3),
            np.array([box.half_extents_mm for box in seen], dtype=np.float64).reshape(-1, 3))


def _line_in(grasp: GraspPoint, standoff_mm: float) -> np.ndarray:
    """The TCP's poses from the standoff down to ``grasp``, ``(S, 4, 4)`` BASE, a step at most :data:`STEP_MM` apart."""
    pose = np.asarray(grasp.pose().to_matrix(), dtype=np.float64)
    approach = np.asarray(grasp.approach, dtype=np.float64)
    steps = max(1, int(math.ceil(max(0.0, float(standoff_mm)) / STEP_MM)))
    out = np.repeat(pose[None, :, :], steps + 1, axis=0)
    for index, back in enumerate(np.linspace(0.0, max(0.0, float(standoff_mm)), steps + 1)):
        out[index, :3, 3] = pose[:3, 3] - back * approach
    return out


def turned_for_the_wrist(
    candidates: Sequence[GraspPoint], housing: WristHousing, seen: Sequence[Any], *, standoff_mm: float,
    distance_mm: float,
) -> tuple[tuple[GraspPoint, ...], WristTurns]:
    """``candidates`` each the way round whose wrist camera keeps ``distance_mm`` from the ``seen`` boxes along its line
    in from ``standoff_mm``, those that keep it neither way after every one that does; and what was done, counted.

    A candidate that is not in BASE, where the boxes stand, is not measured and keeps its place among those that keep
    the distance: nothing is known against it.
    """
    centres_b, turns_b, halves_b = _boxes_of(seen)
    clear: list[GraspPoint] = []
    blocked: list[GraspPoint] = []
    kept = turned = neither = unmeasured = 0
    for grasp in candidates:
        if grasp.frame is not GraspFrame.BASE:
            clear.append(grasp)
            unmeasured += 1
            continue
        if housing.least_mm(_line_in(grasp, standoff_mm), centres_b, turns_b, halves_b) + _ROUNDING_MM >= distance_mm:
            clear.append(grasp)
            kept += 1
            continue
        twin = replace(grasp, axis=-np.asarray(grasp.axis, dtype=np.float64))
        if housing.least_mm(_line_in(twin, standoff_mm), centres_b, turns_b, halves_b) + _ROUNDING_MM >= distance_mm:
            clear.append(twin)
            turned += 1
            continue
        blocked.append(grasp)
        neither += 1
    return tuple(clear + blocked), WristTurns(kept=kept, turned=turned, neither=neither, unmeasured=unmeasured)
