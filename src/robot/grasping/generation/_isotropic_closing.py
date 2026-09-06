"""Choosing a jaw-closing direction when the silhouette does not imply one.

A parallel-jaw grasp closes across the object's short dimension, which the candidate generator reads
off the mask's principal axis. That works while the object has a long and a short dimension. When the
silhouette is near-isotropic (a cube seen from above, a ball, a coin) the two extents are equal to
within segmentation noise and the principal axis is numerically degenerate: it is decided by which of
two nearly-equal eigenvalues comes out larger, and it flips between frames of the same static scene.

Following a degenerate axis is not neutral, because the arm does not treat every yaw equally. Measured
via the cuRobo planner in an empty collision world, so this is the arm alone:

    UR3e, top-down grasp at r=350 mm, z=150 mm, azimuth -90 deg .. +180 deg
        jaw closing radial (from base to target)       4/4 planned at every azimuth
        jaw closing tangential (perpendicular)         0/4 planned at every azimuth
    UR5e, same sweep at r=450 mm
        both                                           4/4 planned

A tangential close is unreachable for the shorter arm everywhere, not merely at one spot; the longer
arm absorbs it. On a symmetric object the yaw is free by definition, so spending it on the reachable
direction costs nothing.

Symptom when it is not done: a non-deterministic pick rate on a byte-identical scene (measured 3-7/10
on a UR3e M2 cube whose mask is 48.6 x 47.7 px, aspect 0.98), surfacing as ``MotionStatus.TIMEOUT``,
which reads as a bad grasp rather than an unreachable one.

Scope, deliberately narrow:
  * Only near-isotropic silhouettes. An elongated object's axis carries real information and rewriting
    it would close the jaw across the long dimension, the regression that ``_ELONGATED_ASPECT_MAX``
    exists to prevent.
  * Only near-vertical approaches. For a tilted approach the yaw is coupled to the approach direction
    and is not free to spend.
  * The sign is irrelevant (measured: +radial and -radial both plan 4/4 everywhere, a parallel jaw is
    symmetric), so the nearer one is chosen: the smallest rotation that reaches the goal.
"""

from __future__ import annotations

import math

import numpy as np

from src.robot.grasping.constants import (
    ISOTROPIC_CLOSING_LOG_FILE,
    create_grasping_logger,
)
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint

__all__ = [
    "ISOTROPIC_ASPECT_MIN",
    "MAX_APPROACH_TILT_DEG",
    "snap_isotropic_closing_to_radial",
]

#: The rewritten close is invisible in every other log, because the candidate keeps
#: its score, rank and label. This logger is the only record of whether the snap
#: fired and by how much.
logger = create_grasping_logger("IsotropicClosing", ISOTROPIC_CLOSING_LOG_FILE)

#: minor/major silhouette extent at or above which the principal axis is treated as undefined.
#: 0.9 means the two extents differ by <=10%; on a ~48 px mask a 1-2 px segmentation wobble is already
#: ~4%, so within this band the axis is noise-decided rather than shape-decided. (Contrast
#: ``_ELONGATED_ASPECT_MAX`` = 0.7, at or below which the shape genuinely dictates the close.)
ISOTROPIC_ASPECT_MIN: float = 0.9

#: How far the approach may tilt from straight down and still count as top-down. Beyond this the jaw
#: yaw is coupled to the approach and is no longer free to re-choose.
MAX_APPROACH_TILT_DEG: float = 25.0


def _yaw_about_z(vector: np.ndarray, cos_t: float, sin_t: float) -> np.ndarray:
    x, y, z = (float(v) for v in vector)
    return np.array([cos_t * x - sin_t * y, sin_t * x + cos_t * y, z], dtype=np.float64)


def snap_isotropic_closing_to_radial(
    grasps: list[GraspPoint],
    *,
    aspect_min: float = ISOTROPIC_ASPECT_MIN,
    max_approach_tilt_deg: float = MAX_APPROACH_TILT_DEG,
) -> list[GraspPoint]:
    """Re-aim the closing axis of near-isotropic BASE-frame grasps along their radial direction.

    ``grasps`` must already be in :attr:`GraspFrame.BASE`: "radial" means from the robot base to the
    grasp, so it is meaningless in camera frame. Candidates that do not qualify are returned
    unchanged (identity), so this is a no-op on any scene without a symmetric object.

    The rotation is a rigid yaw about base Z applied to both the closing axis and the approach, so the
    grasp frame stays orthonormal and the approach stays as vertical as it was: only the spin about the
    approach changes, which is exactly the degree of freedom a symmetric object leaves free.
    """
    min_cos = math.cos(math.radians(max_approach_tilt_deg))
    out: list[GraspPoint] = []
    corrections: list[float] = []
    for grasp in grasps:
        snapped = _snap_one(grasp, aspect_min, min_cos)
        if snapped is not None:
            corrections.append(
                float(snapped.metadata.get("closing_snapped_to_radial_deg", 0.0))
            )
        out.append(snapped if snapped is not None else grasp)
    # One line per call with a count, not one per candidate: the caller passes
    # the whole ranked list.
    if corrections:
        logger.info(
            "Snapped %d/%d near-isotropic grasp(s) to radial closing "
            "(max correction %.1f deg, aspect_min %.2f, tilt limit %.0f deg)",
            len(corrections),
            len(out),
            max(abs(c) for c in corrections),
            aspect_min,
            max_approach_tilt_deg,
        )
    else:
        logger.debug(
            "No isotropic snap: none of %d grasp(s) qualified", len(out)
        )
    return out


def _snap_one(grasp: GraspPoint, aspect_min: float, min_cos: float) -> GraspPoint | None:
    """The snapped grasp, or ``None`` when this candidate does not qualify."""
    if grasp.frame is not GraspFrame.BASE:
        return None  # "radial" is only defined about the robot base
    aspect = grasp.metadata.get("aspect_ratio")
    if not isinstance(aspect, (int, float)) or float(aspect) < aspect_min:
        return None  # the shape implies an axis, or cannot be read: leave it alone

    approach = np.asarray(grasp.approach, dtype=np.float64)
    n_appr = float(np.linalg.norm(approach))
    if n_appr < 1e-9 or abs(float(approach[2]) / n_appr) < min_cos:
        return None  # not a top-down grasp: the yaw is not free

    position = np.asarray(grasp.position, dtype=np.float64)
    radial = np.array([float(position[0]), float(position[1])], dtype=np.float64)
    n_radial = float(np.linalg.norm(radial))
    if n_radial < 1e-6:
        return None  # directly over the base: no radial direction exists

    axis = np.asarray(grasp.axis, dtype=np.float64)
    horizontal = np.array([float(axis[0]), float(axis[1])], dtype=np.float64)
    n_horizontal = float(np.linalg.norm(horizontal))
    if n_horizontal < 1e-6:
        return None  # closing axis is vertical: a yaw about Z cannot move it

    radial /= n_radial
    horizontal /= n_horizontal
    # Nearest of the two equivalent radial senses (a parallel jaw is symmetric and both senses are
    # equally reachable), so the correction is the smallest rotation that gets there.
    if float(np.dot(horizontal, radial)) < 0.0:
        radial = -radial
    cos_t = float(np.clip(np.dot(horizontal, radial), -1.0, 1.0))
    sin_t = float(horizontal[0] * radial[1] - horizontal[1] * radial[0])  # 2D cross: signed sine
    if abs(sin_t) < 1e-9 and cos_t > 0.0:
        return None  # already radial: unchanged, do not rebuild the object

    return GraspPoint(
        position=grasp.position,
        approach=_yaw_about_z(approach, cos_t, sin_t),
        axis=_yaw_about_z(axis, cos_t, sin_t),
        grip_width_mm=grasp.grip_width_mm,
        score=grasp.score,
        frame=grasp.frame,
        label=grasp.label,
        metadata={
            **grasp.metadata,
            # Telemetry, not decoration: a pick whose executed close differs from the silhouette's
            # principal axis should say so, and by how much.
            "closing_snapped_to_radial_deg": round(math.degrees(math.atan2(sin_t, cos_t)), 2),
        },
    )
