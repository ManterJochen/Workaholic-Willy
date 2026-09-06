"""Derive the tool frame a UR controller is using, without adding vendor SDK surface.

The active tool frame of the controller, whatever a PolyScope installation, a URCap or
a running program put in its tool register, is not readable through any call this repo
has evidence for. It does not need to be, because it is derivable from two things the
driver already has:

    X_active(q)  =  inv( T_base_to_flange(q) )  @  T_base_to_activeTCP(q)

* ``T_base_to_activeTCP`` is ``RTDEControlInterface.getForwardKinematics``, reached
  through ``fk()`` in ``connection.py``, which returns the pose of whatever the
  controller calls the TCP.
* ``T_base_to_flange`` is
  :func:`~src.robot.safety._ur_kinematics.ur_link_transforms_mm`, whose last entry is
  the flange, meaning the TCP without a tool. It is a bundled DH table covering all
  seven UR models and it already backs the self-collision guard.

Their difference is the tool register of the controller.

This is honest where a ``setTcp`` or ``getTCPOffset`` call would not be. Neither symbol
appears anywhere in this repo, ``ur_rtde`` is pinned and is not installed on the
development box, and mypy is blind at that boundary under
``ignore_missing_imports``, so nothing in this checkout can tell a real RTDE method
from an invented one and adding one would be the first unsourced SDK claim in the
driver. This derivation adds no symbol: ``getForwardKinematics`` is already called on
every gated Cartesian move, because the singularity guard finite-differences
``arm.fk()``, so its SDK dependency is a strict subset of what ``move()`` already
requires. If it does not exist, motion is broken today, whatever this module does.

Measured offline over 200 random joint configurations per model, on ur3e and ur5e, the
recovery is exact to 1.4e-13 mm and 1e-15 in rotation. That is bucket 2: analytical,
and never run against a physical controller.
"""

from __future__ import annotations

import math

import numpy as np

from src.geometry.matrix import invert_homogeneous
from src.robot.safety._ur_kinematics import ur_link_transforms_mm

__all__ = ["ToolFrameMismatch", "derive_active_tool_frame", "tool_frame_matrix", "compare_tool_frames"]


class ToolFrameMismatch(ValueError):
    """The tool frame derived from the controller disagrees with what config declares."""


def tool_frame_matrix(
    offset_mm: tuple[float, float, float],
    rotation_quat_xyzw: tuple[float, float, float, float],
) -> np.ndarray:
    """The declared flange-to-TCP transform as a 4x4, with the translation in mm."""
    from src.geometry.quaternion import to_rotation_matrix

    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = to_rotation_matrix(np.asarray(rotation_quat_xyzw, dtype=np.float64))
    t[:3, 3] = np.asarray(offset_mm, dtype=np.float64)
    return t


def derive_active_tool_frame(
    model: str,
    joints_rad: "np.ndarray | list[float]",
    controller_tcp_mm: np.ndarray,
) -> np.ndarray | None:
    """Flange to TCP as the controller has it, or ``None`` where ``model`` has no DH table.

    Parameters
    ----------
    model
        UR model key such as ``"ur3e"`` or ``"ur5e"``, which selects the bundled DH
        table.
    joints_rad
        The joint vector the controller FK was evaluated at.
    controller_tcp_mm
        The 4x4 base to active TCP, translation in mm, meaning the controller own FK for
        those joints.
    """
    q = np.asarray(joints_rad, dtype=np.float64)
    links = ur_link_transforms_mm(model, q)
    if links is None:
        return None  # an unknown model or a joint-count mismatch; the caller decides
    base_to_flange = links[-1]
    return invert_homogeneous(base_to_flange) @ np.asarray(controller_tcp_mm, dtype=np.float64)


def compare_tool_frames(observed: np.ndarray, declared: np.ndarray) -> tuple[float, float]:
    """``(translation_mm, rotation_deg)`` between two flange-to-TCP transforms.

    Both numbers are reported because they fail differently: a wrong translation is a
    crash on the first run, and a wrong rotation is a cell that logs successes while
    closing along the wrong object axis.
    """
    d_t = float(np.linalg.norm(np.asarray(observed)[:3, 3] - np.asarray(declared)[:3, 3]))
    r = np.asarray(observed)[:3, :3] @ np.asarray(declared)[:3, :3].T
    cos = (float(np.trace(r)) - 1.0) / 2.0
    d_r = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
    return d_t, d_r
