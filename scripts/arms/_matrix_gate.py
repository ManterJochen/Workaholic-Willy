"""The pose set every gate judges, and the control each judge reads first. Standard library, numpy and PyYAML, beside
the scripts that ask, on purpose.

One rule for the poses, so the exact judge, the fidelity probe, the hand link equivalence probe and the matrix gate all
ask about the same configurations: the arm's own retract pose first, then seeded random draws, then the sweep over the
two joints between wrist_1 and the hand. A gate that generated its own poses would compare two measurements of
different things, and the difference would read as a finding.

The retract is passed in rather than read here, because where it comes from differs: Isaac's Lula ``default_q``, a
rule on the exact meshes, or a descriptor's own ``cspace.default_joint_position`` on the box. The arm is named, because
the draws take its joint count from its description (P0f F8), and because two poses here were only ever measured on a
six joint UR: the sweep's base and the folded elbow. Any other arm is refused by name until P1 measures its own (the
owner, in aurora, 2026-09-23).

The counts the matrix gate records over one pose set live here too, as pure functions over its rows.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import _arm_file  # noqa: E402 (beside this module, stdlib plus PyYAML)

__all__ = ["FOLDED", "RANDOM_POSES", "SEED", "SWEEP_BASE", "SWEEP_JOINT_5_DEG", "SWEEP_JOINT_6_DEG", "PoseSetRefused",
           "attribution_disagreements", "false_clears", "folded_control", "pose_set", "poses_sha256"]

#: The sweep: joint 5 every 10 degrees and joint 6 every 30, the only two joints between wrist_1 and the hand.
SWEEP_JOINT_5_DEG = (-180, 181, 10)
SWEEP_JOINT_6_DEG = (-180, 181, 30)
#: The four joints the sweep holds still, as a six joint UR stands for it.
SWEEP_BASE = (0.0, -1.57, 0.0, 0.0)
#: The elbow folded back onto the upper arm: a collision on every UR, and the control every judge reads first.
FOLDED = (0.0, -1.57, 3.1, 0.0, 0.0, 0.0)
#: What a pose set holds by default: 1 retract plus 1,000 random plus 481 sweep poses.
RANDOM_POSES = 1000
SEED = 20260915


class PoseSetRefused(ValueError):
    """A pose asked for an arm it was never measured on, or a retract that is not one of the arm's poses."""


def _measured_on_urs_only(arm: str, what: str) -> int:
    """``arm``'s joint count, once it is known to be a six joint UR, the only arms ``what`` was measured on."""
    described = _arm_file.read(arm)
    vendor, dof = described["vendor"], len(described["joints"])
    if vendor != "ur" or dof != 6:
        raise PoseSetRefused(
            f"{arm}: {what} was measured on six joint URs only, and {arm} is a {dof} joint {vendor} arm. Nobody has "
            f"measured this arm's own; P1 does, for each arm it admits.")
    return dof


def folded_control(arm: str) -> list[float]:
    """The pose every judge reads first, where a judge that sees no collision is not reading the meshes."""
    _measured_on_urs_only(arm, "the folded elbow control")
    return [float(v) for v in FOLDED]


def pose_set(
    retract: "list[float] | tuple[float, ...]", *, arm: str, random_n: int = RANDOM_POSES, seed: int = SEED,
) -> "tuple[list[list[float]], list[str]]":
    """``(poses, kinds)``: the retract pose, ``random_n`` seeded draws, then the sweep. One seed, one set."""
    dof = _measured_on_urs_only(arm, "the sweep over joints 5 and 6")
    if len(retract) != dof:
        raise PoseSetRefused(f"{arm}: a retract of {len(retract)} values for a {dof} joint arm is not one of its poses")
    poses: list[list[float]] = [[float(v) for v in retract]]
    kinds: list[str] = ["retract"]
    rng = np.random.default_rng(seed)
    for sample in rng.uniform(-np.pi, np.pi, size=(random_n, dof)):
        poses.append([float(v) for v in sample])
        kinds.append("random")
    for q5 in np.radians(np.arange(*SWEEP_JOINT_5_DEG)):
        for q6 in np.radians(np.arange(*SWEEP_JOINT_6_DEG)):
            poses.append([*SWEEP_BASE, float(q5), float(q6)])
            kinds.append("sweep")
    return poses, kinds


def poses_sha256(poses: "list[list[float]] | tuple") -> str:
    """One hash over the configurations judged, so a count cannot stand in for which ones they were.

    A file recording 1,482 poses says nothing about whether they were these 1,482, because another seed is another
    set. The hash runs over float64 bytes in the order sent, which is the order the gate reads its rows back in.
    """
    import hashlib

    asked = np.asarray(poses, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(f"{asked.shape}".encode("utf-8"))
    digest.update(np.ascontiguousarray(asked).tobytes())
    return digest.hexdigest()


def attribution_disagreements(
    collides: "list[bool] | tuple",
    pairs: "list | tuple",
    depths_mm: "list | tuple",
) -> int:
    """How often the planner's verdict and its own account of that verdict do not match.

    cuRobo answers with a self-collision term, and the pair of links is read separately out of sphere ownership. A
    pose the term calls a collision with no pair to name, or a pair named where the term says clear, is the model
    disagreeing with itself, and a verdict cannot be read through that. It is the one count the b1 rung refuses on.

    ``pairs`` and ``depths_mm`` are per pose and hold ``None`` where there was nothing to name. A caller that did not
    ask for names has nothing to compare and must not call this: not asked is ``UNSET`` on the wire, and passing a
    row of ``None`` here would count every pose as a disagreement.
    """
    if not (len(collides) == len(pairs) == len(depths_mm)):
        raise ValueError(
            f"one row per pose, and got {len(collides)} verdicts, {len(pairs)} pairs and {len(depths_mm)} depths. "
            f"A comparison over rows that do not line up is a number about nothing."
        )
    count = 0
    for hit, pair, depth in zip(collides, pairs, depths_mm):
        named = pair is not None and depth is not None
        if bool(hit) != named:
            count += 1
    return count


def false_clears(collides: "list[bool] | tuple", exact_clear: "list[bool] | tuple") -> int:
    """Poses the planner cleared and the exact meshes did not: the unsafe direction, which b1 does not refuse on.

    It is recorded rather than refused because the two models do not answer the same question. The fitted spheres
    reach past the body at a stated reach, so the planner is the conservative one almost everywhere; where it is not,
    the exact guard still has the last word before any motion. The number is the price of the fit, and it is recorded
    so that its movement can be watched.
    """
    if len(collides) != len(exact_clear):
        raise ValueError(f"one row per pose, and got {len(collides)} against {len(exact_clear)}")
    return sum(1 for hit, clear in zip(collides, exact_clear) if not bool(hit) and not bool(clear))
