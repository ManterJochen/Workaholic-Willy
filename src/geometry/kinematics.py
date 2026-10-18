"""One serial kinematic chain, for every arm this stack drives.

A chain is a list of joints from the base to the flange, each with the geometry that carries the link after it.
Ask it for the frame of every link at a joint vector and you can place that link's collision mesh, measure the
arm's reach, or hand the pose to a planner. Nothing here knows about a vendor, a driver or a config tree: it is
numpy and the numbers a description gives it (aurora's new-vendors plan, section 4.2).

It is the same arithmetic the UR guard already runs, deliberately. ``safety/_ur_kinematics.py`` holds a
hand-typed DH table per UR model and the loop that walks it, and every committed UR artefact (the collision
bundles, the sphere maps, the retract rows, the evidence files) was measured through that loop. So
:meth:`SerialChain.link_transforms_mm` is that loop, expression for expression, rather than a tidier one that
agrees to a tolerance: ``tests/robot/safety/test_serial_chain_is_the_ur_chain.py`` holds the two to exact equality over every
model and a thousand random joint vectors, and a rewrite that changed the order of two multiplications would be
caught there before it reached a bundle.

No protocol, one class (the owner, 2026-09-19: no bloat). An arm that is not a DH chain gets a second
constructor here, not a parallel hierarchy; ``from_urdf`` is the one P1 needs for the Franka and lands with it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["DhRow", "SerialChain"]


@dataclass(frozen=True, slots=True)
class DhRow:
    """One joint's standard (Craig-style) Denavit-Hartenberg row, lengths in metres.

    ``theta_offset_rad`` is what the joint variable is measured from. Every UR is zero here, because the
    controller's ``q`` already corresponds to the canonical zero pose; an arm whose description states an offset
    carries it in this field rather than in the caller.
    """

    a_m: float
    d_m: float
    alpha_rad: float
    theta_offset_rad: float = 0.0


def _dh_transform(theta: float, row: DhRow) -> np.ndarray:
    """Standard 4x4 DH transform (metres).

    Expression for expression as ``safety/_ur_kinematics._dh_transform`` writes it. Floating point addition
    and multiplication are not associative, so a mathematically equal rearrangement is not a bit-equal one, and
    the committed UR bundles were cut through that spelling.
    """
    ct = float(np.cos(theta))
    st = float(np.sin(theta))
    ca = float(np.cos(row.alpha_rad))
    sa = float(np.sin(row.alpha_rad))
    a = row.a_m
    d = row.d_m
    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0.0, sa, ca, d],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class SerialChain:
    """A serial chain of revolute joints, base to flange.

    ``joint_names`` is the planner's order, which is also the order ``joints_rad`` arrives in. It is carried
    because a chain with no names cannot be matched against a controller's own ordering, and that mismatch is
    silent: a remapped joint vector is a valid joint vector.
    """

    rows: tuple[DhRow, ...]
    joint_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.rows:
            raise ValueError("a serial chain with no joints describes nothing")
        if len(self.joint_names) != len(self.rows):
            raise ValueError(
                f"the chain has {len(self.rows)} joint(s) and {len(self.joint_names)} name(s): a name per joint, "
                f"in the order a joint vector arrives in"
            )
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError(f"two joints share a name in {self.joint_names}: a name has to identify one joint")

    @classmethod
    def from_dh(cls, rows: "tuple[DhRow, ...]", *, joint_names: "tuple[str, ...]") -> "SerialChain":
        """A chain from DH rows, base to flange, one name per row in the order a joint vector arrives in."""
        return cls(rows=tuple(rows), joint_names=tuple(joint_names))

    @property
    def dof(self) -> int:
        """How many joints a vector for this chain carries."""
        return len(self.rows)

    def link_transforms_mm(self, joints_rad: np.ndarray) -> "list[np.ndarray] | None":
        """Full 4x4 frames for every link, translation in MM, or ``None`` on a wrong joint count.

        Index 0 is the base frame (identity) and the trailing entry is the flange. The rotation block is
        unitless and the translation column is in mm, so ``T @ [x_mm, y_mm, z_mm, 1]`` maps a mm-space mesh
        vertex straight into the mm base frame, which is what the exact-mesh guard wants.

        ``None`` rather than a raise, because the guard's contract is that an unanswerable geometry question is
        UNAVAILABLE and the caller falls back to a coarser check, not that a motion dies.
        """
        if joints_rad.shape[0] != len(self.rows):
            return None
        T = np.eye(4, dtype=np.float64)
        out: list[np.ndarray] = [T.copy()]
        for theta, row in zip(joints_rad, self.rows, strict=True):
            angle = float(theta) if not row.theta_offset_rad else float(theta) + row.theta_offset_rad
            T = T @ _dh_transform(angle, row)
            M = T.copy()
            M[:3, 3] = M[:3, 3] * 1000.0  # metres -> mm in the translation column only
            out.append(M)
        return out

    def link_origins_mm(self, joints_rad: np.ndarray) -> "list[np.ndarray] | None":
        """Just the origins of :meth:`link_transforms_mm`, in mm. ``None`` on a wrong joint count."""
        frames = self.link_transforms_mm(joints_rad)
        if frames is None:
            return None
        return [frame[:3, 3].copy() for frame in frames]

    def render(self) -> str:
        """Describe this chain to a person, as text, ASCII, no trailing newline."""
        lines = [f"{self.dof} joints, base to flange"]
        for name, row in zip(self.joint_names, self.rows, strict=True):
            lines.append(
                f"  {name:<16} a={row.a_m:+.6g} m  d={row.d_m:+.6g} m  alpha={row.alpha_rad:+.6g} rad"
                + (f"  theta0={row.theta_offset_rad:+.6g} rad" if row.theta_offset_rad else "")
            )
        return "\n".join(lines)
