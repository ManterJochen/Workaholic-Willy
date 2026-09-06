"""The ladder's floor: what luck and the dumbest sensible heuristic score on the same scenes.

A top-1 rate from `eval-grasps` read against zero has no scale. A scene whose target is a graspable
box seen from above hands a certain top-1 rate to a generator that proposes nothing except a random
pose on the object, because the reference verdict admits a wide band of approach directions. These
floors measure that rate. There are three of them, because they sit at very different heights and
answer different questions:

  * `random`: a seed drawn uniformly from the target's own surface, an approach drawn uniformly from
    the admissible cone, a random in-plane rotation, a random width. Luck within what the generator
    is even allowed to propose, and the weakest of the three: area-uniform over a 135 deg cap puts
    the mean approach at about 82 deg from vertical, nearly horizontal, and those rarely succeed.
    Quote it as "beats coin-flipping", never as "beats a baseline".
  * `topdown`: random seed, straight down, random in-plane rotation, mid-range width. What a
    practitioner writes first for a bin, and the floor a claim has to clear before the word
    "learned" earns its keep. It should run high: the analytic stack is largely top-down too, and
    the reference verdict admits a band around it.
  * `normal`: the same seed, approaching straight into the surface along its camera-oriented normal,
    mid-range width. The dumbest thing that uses the geometry it can see.

None of the three is a competitor and none is promotable. They exist to be beaten, and a rung that
reports them is reporting the scale of its own axis. They live in `datagen` rather than in `src`: an
evaluation instrument is not a runtime component, and putting them behind the calculator seam in the
library would make it possible to configure a cell to run one.

The admissible cone is the generator's, not the whole sphere. `random` draws inside the same 135 deg
cap the learned encoding uses, because a floor allowed to propose grasps the generator structurally
cannot propose is not a floor for that generator: it is a lower, flattering one.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.robot.grasping.deep.corpus.grasp_encoding import APPROACH_MAX_TILT_DEG
from src.robot.grasping.geometry.normals import estimate_surface_normals
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint

__all__ = ["NormalGraspCalculator", "RandomGraspCalculator", "TopDownGraspCalculator",
           "target_mask", "unproject_target"]

#: Local-PCA radius for the `normal` floor, in mm. The corpus builder's own default, so the two
#: agree about what a surface normal is on the same cloud.
_NORMAL_RADIUS_MM = 15.0


def target_mask(segmentation: Any) -> np.ndarray:
    """The boolean mask, whether it arrived as an array or as a `SegmentationLike` carrying `.mask`.

    This is the calling convention, not defensive programming. The ladder passes an object with a
    `.mask` attribute (`ladder._Segmentation`), exactly as the pick loop does, and a caller holding
    the array passes the array. Accepting only the array fails silently: `np.asarray(obj)` is a 0-d
    object array whose shape never matches the depth map, and the shape guard then refuses without
    a word, on every object.
    """
    # The same three attribute names the runtime's own `deep.calculator._mask_of` accepts. Kept in
    # step deliberately: a floor that understands a narrower protocol than the generator it is the
    # scale for would refuse on exactly the callers the generator serves.
    for attribute in ("mask", "binary_mask", "segmentation"):
        found = getattr(segmentation, attribute, None)
        if found is not None:
            return np.asarray(found).astype(bool)
    return np.asarray(segmentation).astype(bool)


def unproject_target(segmentation: Any, depth_map: np.ndarray,
                     camera_matrix: Any, camera_to_base: Any) -> np.ndarray:
    """``(N, 3)`` BASE-frame mm for the masked, finite pixels. Empty when the mask selects nothing."""
    mask = target_mask(segmentation)
    depth = np.asarray(depth_map, dtype=np.float64)
    if mask.shape != depth.shape:
        return np.zeros((0, 3))
    rows, columns = np.nonzero(mask & np.isfinite(depth) & (depth > 0.0))
    if not len(rows):
        return np.zeros((0, 3))

    matrix = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    z = depth[rows, columns]
    x = (columns - matrix[0, 2]) * z / matrix[0, 0]
    y = (rows - matrix[1, 2]) * z / matrix[1, 1]
    camera = np.column_stack([x, y, z])

    transform = np.asarray(getattr(camera_to_base, "matrix", camera_to_base), dtype=np.float64)
    transform = transform.reshape(4, 4)
    return camera @ transform[:3, :3].T + transform[:3, 3]


def _orthogonal_axis(approach: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """A unit axis perpendicular to each approach, rotated by ``angle`` within that plane."""
    # A reference that is never parallel to the approach: pick whichever world axis it aligns with
    # least. A fixed reference degenerates exactly where the approach is most common, straight down.
    reference = np.zeros_like(approach)
    reference[np.arange(len(approach)), np.argmin(np.abs(approach), axis=1)] = 1.0
    first = np.cross(approach, reference)
    first /= np.linalg.norm(first, axis=1, keepdims=True)
    second = np.cross(approach, first)
    return first * np.cos(angle)[:, None] + second * np.sin(angle)[:, None]


class _SurfaceFloor:
    """Shared plumbing: seeds on the target's own surface, and the calculator call signature."""

    def __init__(self, *, camera_matrix: Any = None, seed: int = 0, candidates: int = 12,
                 min_grip_width_mm: float = 5.0, max_grip_width_mm: float = 85.0,
                 **_ignored: Any) -> None:
        # `**_ignored` for the same reason the deep calculator has it: the ladder builds one kwargs
        # dict for every rung and a floor must not have to track the analytic stack's config surface.
        self.camera_matrix = camera_matrix
        self.seed = int(seed)
        self.candidates = int(candidates)
        self.min_grip_width_mm = float(min_grip_width_mm)
        self.max_grip_width_mm = float(max_grip_width_mm)
        self.render_debug_images = False
        self.last_debug_image_png: bytes | None = None

    # The ladder calls `compute`. Positional order matches `GraspCalculator.compute` exactly, so a
    # rung can swap a floor in without knowing which one it got.
    def compute(self, segmentation: Any, depth_map: np.ndarray,
                T_cam_to_base: Any = None, *args: Any,
                camera_to_base: Any = None, **kwargs: Any) -> list[GraspPoint]:
        transform = camera_to_base if camera_to_base is not None else T_cam_to_base
        if self.camera_matrix is None or transform is None:
            return []
        points = unproject_target(segmentation, depth_map, self.camera_matrix, transform)
        if not len(points):
            return []
        rng = np.random.default_rng(self.seed)
        count = min(self.candidates, len(points))
        chosen = rng.choice(len(points), size=count, replace=False)
        matrix = np.asarray(getattr(transform, "matrix", transform), dtype=np.float64).reshape(4, 4)
        return self._propose(points, chosen, rng, matrix[:3, 3])

    def compute_result(self, *args: Any, **kwargs: Any) -> Any:
        from src.robot.grasping.types.feedback import GraspResult  # noqa: PLC0415

        candidates = self.compute(*args, **kwargs)
        return GraspResult(candidates=tuple(candidates),
                           telemetry={"generator": self.label, "is_floor": True})

    label = "floor"

    def _propose(self, points: np.ndarray, chosen: np.ndarray, rng: np.random.Generator,
                 camera_mm: np.ndarray) -> list[GraspPoint]:
        raise NotImplementedError

    def _build(self, seeds: np.ndarray, approach: np.ndarray, axis: np.ndarray,
               width: np.ndarray) -> list[GraspPoint]:
        return [
            GraspPoint(
                position=seeds[index],
                approach=approach[index],
                axis=axis[index],
                grip_width_mm=float(width[index]),
                # Every floor scores the same. A floor that ranked its own proposals would be a
                # generator, and the ladder's top-1 would then be grading that ranking instead of
                # the scale it was added to establish.
                score=0.0,
                frame=GraspFrame.BASE,
                label=self.label,
                metadata={"generator": self.label, "is_floor": True,
                          "score_is_calibrated": False},
            )
            for index in range(len(seeds))
        ]


class RandomGraspCalculator(_SurfaceFloor):
    """Luck, given perfect segmentation. The floor for "did the generator learn anything"."""

    label = "random"

    def _propose(self, points: np.ndarray, chosen: np.ndarray, rng: np.random.Generator,
                 camera_mm: np.ndarray) -> list[GraspPoint]:
        count = len(chosen)
        # Uniform on the cap, not uniform in the two angles: sampling the polar angle uniformly
        # crowds the proposals around straight-down and would flatter a floor whose whole job is to
        # be unflattering. cos(theta) uniform over [cos(max), 1] is the area-uniform draw.
        top = float(np.cos(np.radians(APPROACH_MAX_TILT_DEG)))
        cosine = rng.uniform(top, 1.0, count)
        sine = np.sqrt(np.clip(1.0 - cosine**2, 0.0, 1.0))
        azimuth = rng.uniform(0.0, 2.0 * np.pi, count)
        # -z is straight down in BASE, and the cap is measured from it.
        approach = np.column_stack([sine * np.cos(azimuth), sine * np.sin(azimuth), -cosine])
        approach /= np.linalg.norm(approach, axis=1, keepdims=True)
        axis = _orthogonal_axis(approach, rng.uniform(0.0, np.pi, count))
        width = rng.uniform(self.min_grip_width_mm, self.max_grip_width_mm, count)
        return self._build(points[chosen], approach, axis, width)


class TopDownGraspCalculator(_SurfaceFloor):
    """Straight down, random seed and rotation. The floor a "learned" claim actually has to clear."""

    label = "topdown"

    def _propose(self, points: np.ndarray, chosen: np.ndarray, rng: np.random.Generator,
                 camera_mm: np.ndarray) -> list[GraspPoint]:
        count = len(chosen)
        approach = np.tile(np.array([0.0, 0.0, -1.0]), (count, 1))
        axis = _orthogonal_axis(approach, rng.uniform(0.0, np.pi, count))
        width = np.full(count, 0.5 * (self.min_grip_width_mm + self.max_grip_width_mm))
        return self._build(points[chosen], approach, axis, width)


class NormalGraspCalculator(_SurfaceFloor):
    """Straight into the surface, mid-range width. The floor for "was any of this worth building"."""

    label = "normal"

    def _propose(self, points: np.ndarray, chosen: np.ndarray, rng: np.random.Generator,
                 camera_mm: np.ndarray) -> list[GraspPoint]:
        estimated = estimate_surface_normals(points, radius_mm=_NORMAL_RADIUS_MM)
        normals = np.asarray(estimated.normals, dtype=np.float64)[chosen]
        # `valid_mask` is the name `SurfaceNormals` gives it. A finite normal is not a meaningful
        # one: local PCA on too few neighbours returns a finite eigenvector describing nothing, and
        # this floor would then approach along it.
        valid = np.asarray(estimated.valid_mask, dtype=bool)[chosen]
        length = np.linalg.norm(normals, axis=1)
        usable = valid & (length > 1e-9)
        if not usable.any():
            return []
        normals = normals[usable] / length[usable, None]
        seeds = points[chosen][usable]
        # Orient toward the camera first. Local PCA returns an eigenvector, and an eigenvector has
        # no sign: `estimate_surface_normals` does not orient, and the corpus builder orients
        # separately against the observing cameras. Without this, half the proposals approach
        # straight up from underneath (tilt 180 deg), from inside the object. The camera saw the
        # surface, so the half-space toward it is the free one.
        toward_camera = camera_mm[None, :] - seeds
        flip = (normals * toward_camera).sum(axis=1) < 0.0
        normals[flip] *= -1.0
        # Into the surface, i.e. away from the camera along the oriented normal.
        approach = -normals
        # A normal pointing up-ish becomes a top-down approach; one pointing sideways stays
        # sideways. Nothing here caps the tilt: this floor is a heuristic, not a draw from the
        # generator's admissible set, and clipping it would make it a third thing.
        axis = _orthogonal_axis(approach, rng.uniform(0.0, np.pi, int(usable.sum())))
        width = np.full(int(usable.sum()),
                        0.5 * (self.min_grip_width_mm + self.max_grip_width_mm))
        return self._build(seeds, approach, axis, width)
