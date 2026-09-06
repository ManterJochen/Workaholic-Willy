"""Typed joint-vector wrapper used across the vendor-neutral robot surface.

Robots differ in their DoF count (UR 6, Franka 7, KUKA iiwa 7, some humanoid arms
8). Passing a raw ``np.ndarray`` everywhere leaves that count implicit and easy to
mismatch, so ``JointPositions`` makes it an explicit, validated property of the
value.

Numerics
--------
* Values are radians in ``float64``.
* Storage is a frozen, write-protected ``np.ndarray`` of shape ``(dof,)``.
* Equality is exact at bit level, so an instance serves as a dict or cache key.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

__all__ = ["JointPositions"]


def _validate_joint_array(values: np.ndarray) -> np.ndarray:
    if values.ndim != 1:
        raise ValueError(
            f"JointPositions expects a 1-D array; got shape {values.shape}."
        )
    if values.size == 0:
        raise ValueError("JointPositions requires at least one joint value.")
    if not np.isfinite(values).all():
        raise ValueError("JointPositions values must all be finite (no NaN/inf).")
    return values


@dataclass(frozen=True, slots=True)
class JointPositions:
    """Immutable joint-space configuration.

    Construct from any 1-D iterable of floats. The array is copied, cast to
    ``float64``, validated and write-locked.

    Parameters
    ----------
    values : array-like
        Joint angles in radians, one per DoF.
    """

    values: np.ndarray
    # No frame is stored. Joint space is implicit per arm, and the arm's
    # RobotCapabilities.dof field gives the expected length.

    def __init__(self, values: Iterable[float] | np.ndarray) -> None:
        arr = np.array(values, dtype=np.float64, copy=True)
        _validate_joint_array(arr)
        arr.setflags(write=False)
        # A frozen dataclass assigns through object.__setattr__.
        object.__setattr__(self, "values", arr)

    # ---- structural -----------------------------------------------------

    @property
    def dof(self) -> int:
        """Degrees of freedom, the length of the joint vector."""
        return int(self.values.shape[0])

    def check_dof(self, expected: int) -> "JointPositions":
        """Return ``self`` if this vector has ``expected`` joints, else raise.

        This is the canonical DoF check, so a driver or a guard does not
        re-implement joint-count validation of its own. Joint space is implicit per
        arm, and the arm's ``RobotCapabilities.dof`` gives the expected length.
        """
        if self.dof != int(expected):
            raise ValueError(f"JointPositions has {self.dof} joints; expected {int(expected)}.")
        return self

    def __len__(self) -> int:
        return self.dof

    def __iter__(self):
        return iter(self.values.tolist())

    def __getitem__(self, idx: int) -> float:
        return float(self.values[idx])

    def __array__(self, dtype=None):
        # Lets ``np.asarray(jp)`` round-trip cheaply, still write-protected.
        return self.values if dtype is None else self.values.astype(dtype, copy=False)

    # ---- value semantics ------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, JointPositions):
            return NotImplemented
        if other.values.shape != self.values.shape:
            return False
        return bool(np.array_equal(self.values, other.values))

    def __hash__(self) -> int:
        return hash((self.values.shape, self.values.tobytes()))

    def __repr__(self) -> str:
        # Compact by design; full precision is available through .values.
        formatted = ", ".join(f"{v:.6f}" for v in self.values.tolist())
        return f"JointPositions(dof={self.dof}, values=[{formatted}])"

    # ---- conversions ----------------------------------------------------

    def tolist(self) -> list[float]:
        """Plain Python list of joint angles in radians."""
        return self.values.tolist()

    @classmethod
    def from_list(cls, values: Iterable[float]) -> JointPositions:
        """Alias for ``JointPositions(values)``."""
        return cls(values)
