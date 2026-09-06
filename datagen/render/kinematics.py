"""Where the arm has to stand so its camera looks where the scene wants, and whether it may.

Three jobs, all pure numpy so they are answerable before a GPU is booted:

* forward: joints to the wrist-link pose, using the repository's own DH chain
  (:mod:`src.robot.safety._ur_kinematics`), the same one the self-collision guard measures against.
  Not a second model of the robot, because a second model is a second answer.
* inverse: the joints that put the wrist link at a wanted pose, by damped least squares on that same
  chain, inside the joint limits.
* failure: a viewpoint the arm cannot reach returns ``None`` rather than a nearby configuration.
  "Nearly there" is the answer that quietly writes a wrong extrinsic into a label.

The circularity is stated, not hidden. IK is solved against this FK, so checking it against that same
FK proves only self-consistency. The real check is on-box: set the joints on the Isaac articulation
and read the link's world pose back out of USD, which is computed from the asset rather than from the
DH table. That is what ``python -m datagen verify-robot`` does, and it is why the acceptance number
lives there and not here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.robot.safety._ur_kinematics import ur_link_transforms_mm

__all__ = [
    "JOINT_FK_BAND_RAD",
    "segment_gap_mm",
    "self_clearance_mm",
    "PARK_JOINTS_RAD",
    "camera_pose_from_joints",
    "sample_joint_configuration",
    "solve_joints_for_camera",
    "wrist_link_pose_mm",
]

#: The park configuration the sim runners reset to. The arm stands here when a wrist viewpoint could
#: not be reached: a defined, reproducible, physically valid place to be, rather than "wherever the
#: last failed attempt left it".
PARK_JOINTS_RAD: tuple[float, ...] = (3.14, -1.0, 0.9, 0.0, 0.0, 0.0)
#: The UR e-series' actual joint range, +-360 deg. A +-180 deg limit is wrong here: the park
#: configuration itself sits at 3.14 rad on joint 0, so every descent from park would be clamped
#: against the wall on its first step and reachable poses would come back "unreachable". A limit that
#: excludes the robot's own park pose is not a limit.
_JOINT_LIMIT_RAD = float(2.0 * np.pi)
#: Damped least squares. The damping is what keeps a near-singular configuration from producing a
#: joint step of thousands of radians instead of failing honestly.
_DLS_DAMPING = 0.05
_IK_MAX_ITERATIONS = 240
#: Largest joint move per iteration; the reason it is capped is at the use site in ``_descend``.
_MAX_JOINT_STEP_RAD = 0.25
#: Extra starting configurations beyond park. Damped least squares finds the branch nearest its seed,
#: so restarts are what separates "this arm cannot hold that pose" from "park was the wrong place to
#: start looking"; only the first of those is a reason to skip a viewpoint.
_IK_RESTARTS = 10
#: Fixed, so the restarts are the same in every run. A reachability answer that depends on a caller's
#: random state makes "same seed, same dataset" untrue.
_IK_SEED_STREAM = 20260812
_IK_POSITION_TOLERANCE_MM = 0.5
_IK_ROTATION_TOLERANCE_RAD = float(np.radians(0.25))
#: Finite-difference step for the Jacobian. Small enough to be a derivative, large enough to survive
#: the double-precision subtraction of two nearly equal transforms.
_JACOBIAN_STEP_RAD = 1e-6


#: How far a ``joint_fk`` draw may wander from its centre, per joint. Not the joint limit: the limit
#: is +-360 deg, and drawing across it points the camera away from the table in nearly every sample.
#: 25 deg keeps most draws looking at the workspace while still moving the viewpoint far enough to be
#: a different picture; a much wider band has most draws refused.
JOINT_FK_BAND_RAD = float(np.radians(25.0))
#: Its own stream, so changing how wrist viewpoints resample never shifts the joint draws (or back).
_JOINT_FK_STREAM = 20260812_2


def sample_joint_configuration(
    attempt: int, *, centre_rad: np.ndarray | None = None, band_rad: float = JOINT_FK_BAND_RAD,
) -> np.ndarray:
    """The ``joint_fk`` proposal: a configuration drawn around ``centre_rad``, deterministic in ``attempt``.

    Attempt 0 is the centre, so ``joint_fk`` and ``target_ik`` agree exactly on the easy case and
    diverge only once a draw has to be replaced. That makes the two modes comparable on the same
    scene rather than two unrelated experiments.

    The centre matters more than the band. Park is not an observation pose, because it faces away
    from the table, so a band around park points the camera at nothing and every scene falls back to
    park with no wrist view. The caller therefore passes the configuration that looks at the scene,
    and the spread from there is what makes the viewpoints arm-shaped instead of designed. Park
    remains the default only so this function has a defined answer with no caller context.

    Draws are clipped to the joint limits, which the band cannot exceed today but would the moment
    someone widens it.
    """
    centre = (np.asarray(PARK_JOINTS_RAD, dtype=np.float64) if centre_rad is None
              else np.asarray(centre_rad, dtype=np.float64))
    if attempt <= 0:
        return centre
    generator = np.random.default_rng([_JOINT_FK_STREAM, attempt])
    drawn = centre + generator.uniform(-band_rad, band_rad, size=centre.shape)
    return np.clip(drawn, -_JOINT_LIMIT_RAD, _JOINT_LIMIT_RAD)


def wrist_link_pose_mm(model: str, joints_rad: np.ndarray) -> np.ndarray:
    """4x4 ``wrist-link -> BASE`` (mm) for a joint configuration, from the repo's own DH chain."""
    transforms = ur_link_transforms_mm(model, np.asarray(joints_rad, dtype=np.float64))
    if transforms is None:
        raise ValueError(
            f"no bundled DH table for {model!r}, or the joint count does not match it: forward "
            f"kinematics is unavailable, so neither a posed arm nor a self-collision check is possible"
        )
    return transforms[-1]


def camera_pose_from_joints(
    model: str, joints_rad: np.ndarray, camera_to_link: np.ndarray,
) -> np.ndarray:
    """Where the eye-in-hand camera ends up: ``link_to_base @ camera_to_link``, both 4x4 in mm."""
    return wrist_link_pose_mm(model, joints_rad) @ np.asarray(camera_to_link, dtype=np.float64)


def _pose_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """6-vector (translation mm, rotation rad) taking ``current`` to ``target``.

    The rotation part is the axis-angle of ``R_target @ R_current.T``, the actual rotation between
    the two frames, rather than a difference of Euler angles, which is not a rotation at all near the
    wrap-around and sends an IK to the wrong branch.
    """
    translation = target[:3, 3] - current[:3, 3]
    relative = target[:3, :3] @ current[:3, :3].T
    angle = float(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)))
    if angle < 1e-9:
        rotation = np.zeros(3)
    else:
        axis = np.array([
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        ]) / (2.0 * np.sin(angle))
        rotation = axis * angle
    return np.concatenate([translation, rotation])


def _jacobian(model: str, joints: np.ndarray) -> np.ndarray:
    """6xN finite-difference Jacobian of the wrist-link pose. Same chain as everything else here."""
    base = wrist_link_pose_mm(model, joints)
    columns = []
    for index in range(joints.shape[0]):
        nudged = joints.copy()
        nudged[index] += _JACOBIAN_STEP_RAD
        columns.append(_pose_error(base, wrist_link_pose_mm(model, nudged)) / _JACOBIAN_STEP_RAD)
    return np.stack(columns, axis=1)


def solve_joints_for_camera(
    model: str,
    camera_to_base_target: np.ndarray,
    camera_to_link: np.ndarray,
    *,
    seed_joints_rad: np.ndarray | None = None,
) -> np.ndarray | None:
    """Joints putting the eye-in-hand camera at ``camera_to_base_target``, or ``None`` if unreachable.

    The camera is rigidly mounted, so the wanted link pose is
    ``camera_to_base_target @ inv(camera_to_link)`` and the whole problem reduces to ordinary IK for
    the wrist link. Damped least squares from a seed, joint-limited, returning ``None`` rather than
    its closest attempt: a configuration tens of millimetres off would render a convincing image
    labelled with a camera pose the camera was never at.
    """
    target_link = np.asarray(camera_to_base_target, dtype=np.float64) @ np.linalg.inv(
        np.asarray(camera_to_link, dtype=np.float64),
    )
    for seed in _seeds(seed_joints_rad):
        solved = _descend(model, seed, target_link)
        if solved is not None:
            return solved
    return None


def _seeds(seed_joints_rad: np.ndarray | None) -> list[np.ndarray]:
    """Starting configurations to descend from, most likely first, and always the same ones.

    Damped least squares converges to the branch nearest its seed, so a single start from park fails
    on poses the arm can plainly hold, including ones generated from a real configuration. A
    single-start solver reports those viewpoints "unreachable" and biases every wrist view towards
    whatever park happens to be near.

    The spread is drawn from a fixed generator, not a caller's rng: a viewpoint that is reachable
    must be reachable in every run of the same seed, or the dataset stops being reproducible.
    """
    park = np.asarray(PARK_JOINTS_RAD, dtype=np.float64)
    seeds = [park.copy()] if seed_joints_rad is None else [
        np.asarray(seed_joints_rad, dtype=np.float64).copy(), park.copy(),
    ]
    generator = np.random.default_rng(_IK_SEED_STREAM)
    for _ in range(_IK_RESTARTS):
        seeds.append(np.clip(park + generator.uniform(-2.0, 2.0, size=park.shape),
                             -_JOINT_LIMIT_RAD, _JOINT_LIMIT_RAD))
    return seeds


def _descend(model: str, start: np.ndarray, target_link: np.ndarray) -> np.ndarray | None:
    """One damped-least-squares descent. ``None`` when it did not reach the tolerance."""
    joints = start.copy()
    for _ in range(_IK_MAX_ITERATIONS):
        error = _pose_error(wrist_link_pose_mm(model, joints), target_link)
        if (float(np.linalg.norm(error[:3])) <= _IK_POSITION_TOLERANCE_MM
                and float(np.linalg.norm(error[3:])) <= _IK_ROTATION_TOLERANCE_RAD):
            return joints
        jacobian = _jacobian(model, joints)
        # Damped pseudo-inverse: (J^T J + lambda^2 I)^-1 J^T. The damping trades exactness for a
        # bounded step near a singularity, where an undamped solve explodes instead of failing.
        lhs = jacobian.T @ jacobian + (_DLS_DAMPING ** 2) * np.eye(joints.shape[0])
        step = np.linalg.solve(lhs, jacobian.T @ error)
        # Cap the step. A linearised step taken at full length overshoots and then oscillates, and
        # every oscillation is reported as "unreachable", which does not only cost a viewpoint: it
        # biases the whole wrist-view distribution towards the easy angles.
        largest = float(np.max(np.abs(step)))
        if largest > _MAX_JOINT_STEP_RAD:
            step = step * (_MAX_JOINT_STEP_RAD / largest)
        joints = np.clip(joints + step, -_JOINT_LIMIT_RAD, _JOINT_LIMIT_RAD)
    return None


def segment_gap_mm(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray) -> float:
    """Closest distance between two line segments, sampled. Cheap and conservative.

    Sampled rather than solved because this is a filter: a viewpoint it wrongly refuses costs one
    resample, and the authoritative check is the safety layer's own guard in ``verify-robot``.
    """
    ts = np.linspace(0.0, 1.0, 9)
    first = a0[None, :] + ts[:, None] * (a1 - a0)[None, :]
    second = b0[None, :] + ts[:, None] * (b1 - b0)[None, :]
    return float(np.min(np.linalg.norm(first[:, None, :] - second[None, :, :], axis=-1)))


def self_clearance_mm(model: str, joints_rad: Any) -> float:
    """Smallest non-adjacent link gap, from the safety layer's own geometry.

    Segment to segment between consecutive link origins, and one implementation for every engine.
    The trap is the near-miss: a point-to-point distance between two joint origins is not a distance
    between the links they span, and it reports a systematically larger gap, so a margin measured
    that way is systematically more permissive.

    At the `render.arm.self_collision_margin_mm` default of 8.0 the two forms rarely differ in
    verdict. The difference becomes load-bearing the moment anyone raises the margin, which is what a
    customer with a bulkier gripper or a cable dress does, and the same number would then mean two
    different things depending on which engine rendered the scene.
    """
    from src.robot.safety._ur_kinematics import ur_link_origins_mm  # noqa: PLC0415

    origins = ur_link_origins_mm(model, np.asarray(joints_rad, dtype=np.float64))
    if origins is None:
        return float("inf")
    points = np.asarray(origins, dtype=np.float64).reshape(-1, 3)
    worst = float("inf")
    for i in range(len(points) - 1):
        for j in range(i + 2, len(points) - 1):
            worst = min(worst, segment_gap_mm(points[i], points[i + 1],
                                              points[j], points[j + 1]))
    return worst
