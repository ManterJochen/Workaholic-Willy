"""Isaac scene authoring for the willy_sim cell.

* :mod:`constants`: prim paths and geometry constants shared by the authoring modules.
* :mod:`build`: :func:`build_combined_scene`, the robot, table, object(s), lights and overhead camera.
* :mod:`cameras`: overhead and wrist camera authoring, and the CAMERA->BASE / CAMERA->TOOL transforms.
* :mod:`markers`: ArUco calibration-board authoring.

The public surface is re-exported here, so ``from src.willy_sim.scene import build_combined_scene``
resolves without naming a submodule. All ``isaacsim.*`` imports stay lazy inside the functions, so
importing this package needs no Isaac Sim.
"""

from __future__ import annotations

from .build import SceneAppearance, SceneHandles, build_combined_scene
from .cameras import (
    D435_RGB_HFOV_DEG,
    D435_RGB_VFOV_DEG,
    WRIST_CAM_PRIM,
    WRIST_LINK_PRIM,
    aperture_for_hfov,
    author_fixed_camera,
    camera_elevation_deg,
    camera_position_for_elevation,
    camera_to_base_ground_truth,
    camera_to_tcp_ground_truth,
    hfov_deg_from_intrinsics,
    look_at_quat_in_link_xyzw,
    mount_wrist_camera,
    set_horizontal_fov,
    top_down_camera_to_base_matrix,
)
from .constants import (
    ARM_PRIM,
    ARUCO_DICT_NAME,
    ARUCO_LENGTH_MM,
    ARUCO_MARKER_ID,
    BIN_WALL_PRIM,
    CAMERA_PRIM,
    MARKER_PRIM,
    OBJECT_PRIM,
    TABLE_PRIM,
)
from .markers import add_aruco_marker, mount_tool_aruco_marker

__all__ = [
    "ARM_PRIM",
    "OBJECT_PRIM",
    "CAMERA_PRIM",
    "TABLE_PRIM",
    "MARKER_PRIM",
    "BIN_WALL_PRIM",
    "WRIST_LINK_PRIM",
    "WRIST_CAM_PRIM",
    "SceneAppearance",
    "SceneHandles",
    "build_combined_scene",
    "add_aruco_marker",
    "ARUCO_LENGTH_MM",
    "ARUCO_DICT_NAME",
    "ARUCO_MARKER_ID",
    "top_down_camera_to_base_matrix",
    "author_fixed_camera",
    "mount_wrist_camera",
    "mount_tool_aruco_marker",
    "look_at_quat_in_link_xyzw",
    "camera_to_base_ground_truth",
    "camera_to_tcp_ground_truth",
    "D435_RGB_HFOV_DEG",
    "D435_RGB_VFOV_DEG",
    "aperture_for_hfov",
    "hfov_deg_from_intrinsics",
    "set_horizontal_fov",
    "camera_position_for_elevation",
    "camera_elevation_deg",
]
