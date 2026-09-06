"""Choosing the one configuration the arm holds for a scene, and refusing the ones it may not.

Separate from the Isaac authoring on purpose: which configuration to hold is decided from geometry,
and geometry is answerable without a GPU. Only the placing of prims needs one.

Three things disqualify a configuration, and all three give the caller the same answer, that the
viewpoint is not usable and it should try another or fall back to park:

* IK found nothing: the camera pose is outside the arm's reach or its joint limits.
* The arm hits itself, checked at the link-capsule level against the configured margin.
* The arm stands inside the scene, which a pile makes likely: a viewpoint at 260 mm, the floor of
  the sampled wrist radius, can be reachable and collision-free with itself while the forearm passes
  through the objects below it.

One configuration per scene, not per view. The wrist camera is the arm, so if it took the picture
from there, that is where the arm was while the oblique cameras were exposing too. Rendering the same
scene with the arm in two places would be two different scenes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from datagen.render.kinematics import (
    PARK_JOINTS_RAD,
    camera_pose_from_joints,
    solve_joints_for_camera,
    wrist_link_pose_mm,
)

__all__ = ["ArmPose", "arm_point_mask", "closest_link_clearance_mm", "resolve_arm_pose"]

#: Radius of the capsule standing in for a link when measuring clearance to scene objects. The UR
#: e-series upper arm and forearm are ~75 mm across at the joints; half of that, rounded up, is a
#: deliberately generous cylinder. The check exists to keep the arm out of the objects, so erring
#: thick costs a viewpoint and erring thin costs a scene that is physically impossible.
_LINK_RADIUS_MM = 45.0
#: The tool sticking out past the flange (2F-85 plus coupler). Included as one more capsule, because
#: the gripper is the part that actually reaches into a bin.
_TOOL_LENGTH_MM = 175.0


@dataclass(frozen=True, slots=True)
class ArmPose:
    """The configuration the arm holds for one scene, and how it came to hold it."""

    joints_rad: tuple[float, ...]
    #: ``"posed"``: placed by IK at the sampled wrist viewpoint.
    #: ``"park"``: no viewpoint was usable, so the arm stands in its defined park configuration and
    #: the wrist view is not rendered. Recorded so a consumer never reads park occlusion as a
    #: viewpoint.
    origin: str
    attempts: int
    #: Why the sampled viewpoints were rejected, in order. Empty when the first one worked.
    rejections: tuple[str, ...] = ()

    @property
    def is_park(self) -> bool:
        return self.origin == "park"


def _segments_mm(model: str, joints_rad: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """The arm as a chain of segments in base mm, tool included, for clearance measurement."""
    from src.robot.safety._ur_kinematics import ur_link_origins_mm  # noqa: PLC0415

    origins = ur_link_origins_mm(model, np.asarray(joints_rad, dtype=np.float64))
    if origins is None:
        raise ValueError(f"no DH table for {model!r}: the arm's geometry is unknown")
    segments = [(origins[i], origins[i + 1]) for i in range(len(origins) - 1)]
    flange = wrist_link_pose_mm(model, np.asarray(joints_rad, dtype=np.float64))
    # The tool points along the flange's own +Z, which is where a UR's tool flange faces.
    segments.append((flange[:3, 3], flange[:3, 3] + flange[:3, 2] * _TOOL_LENGTH_MM))
    return segments


def _point_to_segment_mm(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    span = end - start
    length_squared = float(span @ span)
    if length_squared < 1e-12:
        return float(np.linalg.norm(point - start))
    t = float(np.clip((point - start) @ span / length_squared, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + t * span)))


def closest_link_clearance_mm(
    model: str,
    joints_rad: np.ndarray,
    object_centres_mm: np.ndarray,
    object_radii_mm: np.ndarray,
) -> float:
    """Smallest gap between the arm and any scene object, in mm. Negative means it is inside one.

    Objects as bounding spheres and links as capsules: both conservative, both cheap, and the answer
    only ever errs towards refusing a viewpoint, which costs one resample.
    """
    centres = np.atleast_2d(np.asarray(object_centres_mm, dtype=np.float64))
    radii = np.atleast_1d(np.asarray(object_radii_mm, dtype=np.float64))
    if centres.size == 0:
        return float("inf")
    segments = _segments_mm(model, joints_rad)
    worst = float("inf")
    for centre, radius in zip(centres, radii):
        for start, end in segments:
            gap = _point_to_segment_mm(centre, start, end) - _LINK_RADIUS_MM - float(radius)
            worst = min(worst, gap)
    return worst


def arm_point_mask(model: str, joints_rad, points_mm, *, margin_mm: float = 0.0):
    """Which of ``points_mm`` belong to the posed arm. Vectorised over every point at once.

    The cloud corpus keeps table, walls and arm, because that is what `scene_points_mm` carries at
    runtime. An arm is not like the other two under augmentation: a scene rotated about gravity is
    still a valid scene, while a rotated arm stands where no arm on a fixed base could ever stand.
    Tagging the arm rather than dropping it keeps both facts, so the corpus has the arm and the
    augmentation can leave it behind.

    The same capsule chain `closest_link_clearance_mm` measures against, so the two cannot disagree
    about where the arm is. `_LINK_RADIUS_MM` is deliberately generous (45 mm against a ~37 mm real
    half-width): here that errs towards calling a table point "arm", which costs a few points of
    table and never mislabels arm as table.

    The DH chain and the Isaac USD base can differ by 180 deg about +Z, which is what
    `self_collision` carries `kinematics_base_yaw_deg` for. `_segments_mm` is what datagen poses the
    arm with, so the two are the same chain by construction; a caller that supplied its own origins
    would have to reconcile that itself.
    """
    points = np.atleast_2d(np.asarray(points_mm, dtype=np.float64))
    if points.size == 0:
        return np.zeros(0, dtype=bool)
    reach = _LINK_RADIUS_MM + float(margin_mm)
    inside = np.zeros(len(points), dtype=bool)
    for start, end in _segments_mm(model, np.asarray(joints_rad, dtype=np.float64)):
        span = end - start
        length_squared = float(span @ span)
        if length_squared < 1e-12:
            distance = np.linalg.norm(points - start, axis=1)
        else:
            t = np.clip((points - start) @ span / length_squared, 0.0, 1.0)
            distance = np.linalg.norm(points - (start + t[:, None] * span), axis=1)
        inside |= distance <= reach
    return inside


def resolve_arm_pose(  # noqa: PLR0913 (each argument is a separate source of truth)
    model: str,
    camera_to_link: np.ndarray,
    sample_camera_pose,  # noqa: ANN001 (Callable[[int], np.ndarray]; kept loose for the caller's rng)
    object_centres_mm: np.ndarray,
    object_radii_mm: np.ndarray,
    *,
    max_attempts: int = 12,
    self_collision_margin_mm: float = 8.0,
    self_clearance,  # noqa: ANN001 (Callable[[np.ndarray], float]; injected so this stays GPU-free)
    propose_joints=None,  # noqa: ANN001 (Callable[[int], np.ndarray] | None)
    accept_camera=None,  # noqa: ANN001 (Callable[[np.ndarray], str | None])
) -> tuple[ArmPose, np.ndarray | None]:
    """Find a configuration for a usable wrist viewpoint, or fall back to park.

    Returns ``(pose, camera_to_base)``. ``camera_to_base`` is ``None`` exactly when the arm parked:
    the wrist view is then not rendered, while the fixed cameras still are, which is the rule for an
    unreachable viewpoint and applies unchanged when the reason is collision instead of reach.

    The two configured viewpoint sources differ only in how a candidate is proposed, and they share
    every acceptance test below: two acceptance paths would drift, and "the arm may not stand inside
    the objects" must mean the same thing whichever way the pose was arrived at.

    * ``target_ik`` (default, ``propose_joints=None``): draw a camera pose and solve IK for it. It
      can fail at the IK step, which is the cost of keeping the designed viewpoint distribution.
    * ``joint_fk`` (``propose_joints`` given): draw a configuration directly, so there is no IK step
      to fail. It can still be refused by the collision tests and by ``accept_camera``, so this is
      "no unreachable viewpoints", not "never parks".

    ``accept_camera`` returns a rejection reason or ``None``. It exists for ``joint_fk``, where a
    drawn configuration can point the camera at nothing: a wrist image of the far wall is reachable,
    collision-free and worthless.
    """
    rejections: list[str] = []
    for attempt in range(max(1, max_attempts)):
        if propose_joints is None:
            target = np.asarray(sample_camera_pose(attempt), dtype=np.float64)
            solved = solve_joints_for_camera(model, target, camera_to_link)
            if solved is None:
                rejections.append(f"attempt {attempt}: no IK solution")
                continue
            joints = solved
        else:
            joints = np.asarray(propose_joints(attempt), dtype=np.float64)
        margin = float(self_clearance(joints))
        if margin < self_collision_margin_mm:
            rejections.append(f"attempt {attempt}: self-collision clearance {margin:.1f} mm")
            continue
        clearance = closest_link_clearance_mm(model, joints, object_centres_mm, object_radii_mm)
        if clearance < 0.0:
            rejections.append(f"attempt {attempt}: arm inside a scene object by {-clearance:.1f} mm")
            continue
        camera_to_base = camera_pose_from_joints(model, joints, camera_to_link)
        if accept_camera is not None:
            refused = accept_camera(camera_to_base)
            if refused:
                rejections.append(f"attempt {attempt}: {refused}")
                continue
        return (
            ArmPose(tuple(float(v) for v in joints), "posed", attempt + 1, tuple(rejections)),
            camera_to_base,
        )
    return ArmPose(PARK_JOINTS_RAD, "park", max(1, max_attempts), tuple(rejections)), None
