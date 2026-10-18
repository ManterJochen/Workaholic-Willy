"""Derive the tool frame a controller is using, from the arm's own description, without adding vendor SDK surface.

The active tool frame of the controller, whatever a pendant installation, a URCap or a running program put in its tool
register, is not readable through any call this repo has evidence for. It does not need to be, because it is derivable
from two things a planned arm already has:

    X_active(q)  =  inv( T_base_to_flange(q) )  @  T_base_to_activeTCP(q)

* ``T_base_to_activeTCP`` is the transport's ``controller_tcp_now`` (``ControllerKinematics``), the pose of whatever
  the controller calls the TCP; on a UR that is ``getForwardKinematics``.
* ``T_base_to_flange`` is the last frame of the arm description's serial chain (:func:`described_chain`), the flange,
  meaning the TCP without a tool. Until P0g slice 15 this read the UR DH table in ``safety/_ur_kinematics.py``; the
  chain is that table expression for expression, held to exact equality over every described model, every link and a
  thousand random joint vectors by ``tests/robot/safety/test_serial_chain_is_the_ur_chain.py``, so the move changed no bit.

Their difference is the tool register of the controller.

This is honest where a ``setTcp`` or ``getTCPOffset`` call would not be. Neither symbol appears anywhere in this repo,
and adding one would be the first unsourced SDK claim in a driver. This derivation adds no symbol: the controller's FK is
already asked on every gated Cartesian move, so its SDK dependency is a strict subset of what ``move()`` already
requires.

Measured offline over 200 random joint configurations per model, on ur3e and ur5e, the recovery is exact to 1.4e-13 mm
and 1e-15 in rotation. That is bucket 2: analytical, and never run against a physical controller.
"""

from __future__ import annotations

import functools

import numpy as np

from src.geometry.kinematics import DhRow, SerialChain
from src.geometry.matrix import invert_homogeneous

__all__ = ["derive_active_tool_frame", "described_chain"]


@functools.lru_cache(maxsize=None)
def _chain_for(key: str) -> SerialChain | None:
    from src.config import ConfigError  # noqa: PLC0415 (the config layer is below this one; lazy for import cost)
    from src.config.arms import load_arm  # noqa: PLC0415

    try:
        description = load_arm(key)
    except ConfigError:
        return None
    # Standard DH only, and only where that is known. The chain reads a description's rows as standard (Craig)
    # Denavit-Hartenberg, which every UR description is. Another vendor's description states its own convention in P1
    # (the Franka's are modified DH), and until one does it gets no chain, exactly as it got none from the UR table this
    # replaced: a chain read in the wrong convention would refuse a correct cell with a distance nobody could act on.
    if description.vendor != "ur":
        return None
    rows = tuple(DhRow(a_m=joint.a_m, d_m=joint.d_m, alpha_rad=joint.alpha_rad) for joint in description.joints)
    return SerialChain.from_dh(rows, joint_names=tuple(joint.name for joint in description.joints))


def described_chain(model: str | None) -> SerialChain | None:
    """The serial chain the arm registry describes for ``model``, base to flange, or ``None`` where none is described.

    Cached per model: a connect asks it once, and the registry is read once per process.
    """
    if not model:
        return None
    return _chain_for(str(model).lower())


def derive_active_tool_frame(
    model: str | None,
    joints_rad: "np.ndarray | list[float]",
    controller_tcp_mm: np.ndarray,
) -> np.ndarray | None:
    """Flange to TCP as the controller has it, or ``None`` where ``model`` has no described chain.

    Parameters
    ----------
    model
        The arm registry's model key, such as ``"ur3e"`` or ``"ur5e"``, which selects the description's chain.
    joints_rad
        The joint vector the controller FK was evaluated at.
    controller_tcp_mm
        The 4x4 base to active TCP, translation in mm, meaning the controller's own FK for those joints.
    """
    chain = described_chain(model)
    if chain is None:
        return None
    links = chain.link_transforms_mm(np.asarray(joints_rad, dtype=np.float64))
    if links is None:
        return None  # a joint-count mismatch; the caller decides what that means
    return invert_homogeneous(links[-1]) @ np.asarray(controller_tcp_mm, dtype=np.float64)
