"""URPose, the pose representation and conversions for Universal Robots.

A UR TCP pose is ``[x, y, z, rx, ry, rz]``, where the translations are metres and the
rotation is axis-angle in radians. This package uses millimetres throughout instead, to
match the OpenCV calibration pipeline of ChArUco detector, hand-eye calibration and
marker poses.

That is the central numerics contract:

* Inside this stack it is millimetres with axis-angle radians. That covers every
  ``URPose`` instance, every 4x4 matrix :meth:`to_T` produces, and every distance
  :meth:`distance_to` returns.
* At the UR boundary it is metres with axis-angle radians, and the conversion happens in
  :meth:`to_ur_list` and :meth:`from_ur_list`.

A multiplication by ``1000.0`` anywhere else in the package is the boundary being
crossed in the wrong place.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2 as cv
import numpy as np

from src.utility.io import dump_json, load_json


@dataclass(frozen=True, slots=True)
class URPose:
    """An immutable robot TCP pose.

    Every translation is in millimetres and the rotation is axis-angle in radians. The
    UR controller uses metres, and the conversion happens at the boundary, in
    ``to_ur_list`` and ``from_ur_list``.
    """

    x: float
    y: float
    z: float
    rx: float
    ry: float
    rz: float
    label: str = ""

    def to_ur_list(self) -> list[float]:
        """Return ``[x_m, y_m, z_m, rx, ry, rz]``, with the translation in metres."""
        return [
            self.x / 1000.0,
            self.y / 1000.0,
            self.z / 1000.0,
            self.rx,
            self.ry,
            self.rz,
        ]

    @classmethod
    def from_ur_list(cls, tcp: list[float], label: str = "") -> URPose:
        """Create one from the UR-format ``[x_m, y_m, z_m, rx, ry, rz]``.

        It converts the metres to millimetres on the way in.
        """
        if len(tcp) != 6:
            raise ValueError(
                f"UR TCP list must have 6 elements, got {len(tcp)}: {tcp!r}"
            )
        x, y, z, rx, ry, rz = tcp
        if not all(np.isfinite(v) for v in tcp):
            raise ValueError(f"UR TCP list must be finite, got {tcp!r}")
        return cls(
            x=x * 1000.0,
            y=y * 1000.0,
            z=z * 1000.0,
            rx=rx,
            ry=ry,
            rz=rz,
            label=label,
        )

    def to_T(self) -> np.ndarray:
        """Convert to a 4x4 homogeneous transform, in mm with a rotation matrix."""
        rvec = np.array([self.rx, self.ry, self.rz], dtype=np.float64)
        R, _ = cv.Rodrigues(rvec)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = [self.x, self.y, self.z]
        return T

    @classmethod
    def from_T(cls, T: np.ndarray, label: str = "") -> URPose:
        """Create one from a 4x4 homogeneous matrix in millimetres."""
        T = np.asarray(T)
        if T.shape != (4, 4):
            raise ValueError(f"T must have shape (4, 4), got {T.shape}")
        R = T[:3, :3].astype(np.float64)
        t = T[:3, 3]
        rvec, _ = cv.Rodrigues(R)
        rvec = rvec.flatten()
        return cls(
            x=float(t[0]),
            y=float(t[1]),
            z=float(t[2]),
            rx=float(rvec[0]),
            ry=float(rvec[1]),
            rz=float(rvec[2]),
            label=label,
        )

    @property
    def position_mm(self) -> np.ndarray:
        """The translation as a (3,) array in millimetres."""
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    def distance_to(self, other: URPose) -> float:
        """The Euclidean distance in millimetres between two poses."""
        return float(np.linalg.norm(self.position_mm - other.position_mm))

    def angle_to(self, other: URPose) -> float:
        """The rotation difference in degrees between two poses."""
        R1 = self.to_T()[:3, :3]
        R2 = other.to_T()[:3, :3]
        cos = np.clip((np.trace(R1.T @ R2) - 1.0) / 2.0, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos)))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> URPose:
        return cls(**d)

    @staticmethod
    def save_list(poses: list[URPose], path: str | Path) -> Path:
        """Write a list of poses to a JSON file."""
        path = Path(path)
        dump_json([p.to_dict() for p in poses], path)
        return path

    @staticmethod
    def load_list(path: str | Path) -> list[URPose]:
        """Load a list of poses from a JSON file."""
        data = load_json(path)
        return [URPose.from_dict(d) for d in data]
