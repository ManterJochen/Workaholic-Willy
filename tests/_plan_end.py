"""The pose a stand-in planner's trajectory ends at: the only goal a real planner hands that trajectory back for.

The UR driver reads where a plan ends on the DH chain and refuses, before anything moves, a plan that does not put
the flange on its goal. A double that hands back one fixed trajectory whatever it is asked claims a planner that
does not exist, so a test commanding a move through one commands the pose that trajectory reaches.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.geometry import Frame, Pose


def pose_where_it_ends(arm: Any, joints: "list[float] | tuple[float, ...]") -> Pose:
    """The TCP pose ``joints`` put a UR arm at: its DH flange times the tool frame the arm declares, if any."""
    from src.robot.safety._ur_kinematics import ur_link_transforms_mm

    frames = ur_link_transforms_mm(str(arm.config.ur.model), np.asarray([float(v) for v in joints], dtype=np.float64))
    assert frames is not None, f"no DH chain for {arm.config.ur.model!r}"
    flange = np.asarray(frames[-1], dtype=np.float64)
    tool = arm._declared_tool_matrix()
    tcp = flange if tool is None else flange @ np.asarray(tool, dtype=np.float64)
    return Pose.from_matrix(tcp, frame=Frame.BASE, label="where the plan ends")


#: A workspace wide enough that the box never decides a test about something else: a planned configuration chosen for
#: its collision or its path can put the flange anywhere in reach.
OPEN_WORKSPACE: dict[str, float] = {
    "x_min": -1500.0, "x_max": 1500.0, "y_min": -1500.0, "y_max": 1500.0, "z_min": -500.0, "z_max": 1500.0,
}
