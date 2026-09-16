"""The pose set every gate judges. Standard library plus numpy, beside the scripts that ask, on purpose.

One rule for the poses, so the exact judge, the fidelity probe, the hand link equivalence probe and the matrix gate all
ask about the same configurations: the arm's own retract pose first, then seeded random draws, then the sweep over the
two joints between wrist_1 and the hand. A gate that generated its own poses would compare two measurements of
different things, and the difference would read as a finding.

The retract is passed in rather than read here, because where it comes from differs: Isaac's Lula ``default_q``, a
rule on the exact meshes, or a descriptor's own ``cspace.default_joint_position`` on the box.
"""

from __future__ import annotations

import numpy as np

__all__ = ["RANDOM_POSES", "SEED", "SWEEP_JOINT_5_DEG", "SWEEP_JOINT_6_DEG", "pose_set"]

#: The sweep: joint 5 every 10 degrees and joint 6 every 30, the only two joints between wrist_1 and the hand.
SWEEP_JOINT_5_DEG = (-180, 181, 10)
SWEEP_JOINT_6_DEG = (-180, 181, 30)
#: What a pose set holds by default: 1 retract plus 1,000 random plus 481 sweep poses.
RANDOM_POSES = 1000
SEED = 20260915


def pose_set(
    retract: "list[float] | tuple[float, ...]", *, random_n: int = RANDOM_POSES, seed: int = SEED,
) -> "tuple[list[list[float]], list[str]]":
    """``(poses, kinds)``: the retract pose, ``random_n`` seeded draws, then the sweep. One seed, one set."""
    poses: list[list[float]] = [[float(v) for v in retract]]
    kinds: list[str] = ["retract"]
    rng = np.random.default_rng(seed)
    for sample in rng.uniform(-np.pi, np.pi, size=(random_n, 6)):
        poses.append([float(v) for v in sample])
        kinds.append("random")
    for q5 in np.radians(np.arange(*SWEEP_JOINT_5_DEG)):
        for q6 in np.radians(np.arange(*SWEEP_JOINT_6_DEG)):
            poses.append([0.0, -1.57, 0.0, 0.0, float(q5), float(q6)])
            kinds.append("sweep")
    return poses, kinds
