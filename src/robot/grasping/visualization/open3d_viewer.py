"""3D Open3D viewer for grasp candidates.

Builds a scene from the point cloud, an origin frame, a support-plane rectangle
and one wireframe gripper per ranked candidate. Open3D is imported lazily, so the
import time of the grasping package is unaffected when the viewer is not used.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from src.robot.grasping.collision.gripper_model import (
    GripperGeometryStrategy,
    ParallelJawGripperModel,
)
from src.robot.grasping.collision.table_collision import SupportPlane
from src.robot.grasping.types.grasp_point import GraspPoint

__all__ = [
    "GraspViewerConfig",
    "build_grasp_scene",
    "show_grasp_scene",
]


_RGB_GRASP_COLORS = (
    (1.00, 0.78, 0.00),
    (0.00, 1.00, 0.40),
    (0.00, 0.78, 1.00),
    (0.78, 0.40, 1.00),
    (0.70, 0.70, 0.70),
)


@dataclass(frozen=True, slots=True)
class GraspViewerConfig:
    """Knobs for the Open3D scene builder."""

    max_grasps: int = 5
    coordinate_frame_size_mm: float = 60.0
    table_extent_mm: float = 600.0
    point_size_mm: float = 0.0  # use Open3D default when 0

    def __post_init__(self) -> None:
        if self.max_grasps < 1:
            raise ValueError("max_grasps must be >= 1")
        if self.coordinate_frame_size_mm <= 0.0:
            raise ValueError("coordinate_frame_size_mm must be > 0")
        if self.table_extent_mm <= 0.0:
            raise ValueError("table_extent_mm must be > 0")


def _import_open3d():
    """Import Open3D, or refuse with the remedy that matches the actual fault.

    There are two faults, not one, because Open3D is a native extension. When it is absent the
    import raises ``ImportError``; when it is present but unloadable it raises ``OSError``, for
    example ``OSError: [WinError 4551]`` from Windows Smart App Control, which re-evaluates
    unsigned native builds by reputation days after installation. Both are caught, since a helper
    whose whole job is an actionable message may not let either reach the caller raw.

    Telling somebody to ``pip install open3d`` when it is installed and refused is worse than the
    raw error, which at least names the policy.
    """
    try:
        import open3d as o3d  # type: ignore
    except ImportError as exc:  # pragma: no cover (depends on env)
        raise ImportError(
            "open3d is required for the 3D grasp viewer. Install with "
            "'pip install open3d' or include it in requirements.txt."
        ) from exc
    except OSError as exc:  # pragma: no cover (depends on the host's code-integrity policy)
        raise ImportError(
            f"open3d is installed, and its native library could not be loaded "
            f"({type(exc).__name__}: {exc}). This is not a missing package and reinstalling will not "
            f"help. On Windows it is usually Smart App Control or an application-control policy "
            f"refusing an unsigned extension."
        ) from exc
    return o3d


def _orthonormal_basis(approach: np.ndarray, axis: np.ndarray) -> np.ndarray:
    closing = np.array(axis, dtype=np.float64, copy=True)
    closing /= max(float(np.linalg.norm(closing)), 1e-9)
    z_axis = np.array(approach, dtype=np.float64, copy=True)
    z_axis /= max(float(np.linalg.norm(z_axis)), 1e-9)
    binormal = np.cross(z_axis, closing)
    binormal_norm = float(np.linalg.norm(binormal))
    if binormal_norm < 1e-9:
        # Degenerate: pick any perpendicular vector.
        helper = np.array([1.0, 0.0, 0.0]) if abs(closing[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        binormal = np.cross(z_axis, helper)
        binormal_norm = float(np.linalg.norm(binormal))
    binormal /= binormal_norm
    z_axis = np.cross(closing, binormal)
    z_axis /= max(float(np.linalg.norm(z_axis)), 1e-9)
    return np.column_stack((closing, binormal, z_axis))


def _build_gripper_wireframe(
    o3d,
    grasp: GraspPoint,
    gripper: GripperGeometryStrategy,
    color: tuple[float, float, float],
):
    """Return an open3d.geometry.LineSet wireframe for one gripper."""
    R = _orthonormal_basis(grasp.approach, grasp.axis)
    edges = (
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7),
    )
    all_points: list[np.ndarray] = []
    all_lines: list[tuple[int, int]] = []
    offset = 0
    for box in gripper.collision_boxes(grasp.grip_width_mm):
        local = box.corners_local_mm()
        world = local @ R.T + grasp.position
        all_points.append(world)
        all_lines.extend((a + offset, b + offset) for a, b in edges)
        offset += local.shape[0]
    points = np.vstack(all_points)
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(np.asarray(all_lines, dtype=np.int32)),
    )
    line_set.colors = o3d.utility.Vector3dVector(
        np.tile(np.asarray(color, dtype=np.float64), (len(all_lines), 1))
    )
    return line_set


def _build_table(o3d, plane: SupportPlane, extent_mm: float):
    """Build a thin square mesh aligned with the support plane."""
    n = np.asarray(plane.normal, dtype=np.float64)
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, helper)
    u /= max(float(np.linalg.norm(u)), 1e-9)
    v = np.cross(n, u)
    v /= max(float(np.linalg.norm(v)), 1e-9)
    half = 0.5 * float(extent_mm)
    centre = n * float(plane.offset_mm)
    corners = np.vstack(
        [
            centre + half * (-u - v),
            centre + half * (u - v),
            centre + half * (u + v),
            centre + half * (-u + v),
        ]
    )
    triangles = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(corners),
        triangles=o3d.utility.Vector3iVector(triangles),
    )
    mesh.paint_uniform_color((0.65, 0.65, 0.70))
    mesh.compute_vertex_normals()
    return mesh


def build_grasp_scene(
    grasps: Iterable[GraspPoint],
    *,
    points_mm: np.ndarray | None = None,
    support_plane: SupportPlane | None = None,
    gripper_model: GripperGeometryStrategy | None = None,
    config: GraspViewerConfig | None = None,
) -> Sequence:
    """Return the Open3D geometry list (feed to ``draw_geometries`` or a custom viewer)."""
    cfg = config or GraspViewerConfig()
    o3d = _import_open3d()
    geometries: list = []

    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=float(cfg.coordinate_frame_size_mm)
    )
    geometries.append(coord)

    if support_plane is not None:
        geometries.append(_build_table(o3d, support_plane, cfg.table_extent_mm))

    if points_mm is not None:
        points = np.asarray(points_mm, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points_mm must be (N, 3), got {points.shape}")
        cloud = o3d.geometry.PointCloud(
            points=o3d.utility.Vector3dVector(points)
        )
        cloud.paint_uniform_color((0.45, 0.55, 0.85))
        geometries.append(cloud)

    gripper = gripper_model or ParallelJawGripperModel()
    grasp_list = list(grasps)[: cfg.max_grasps]
    for rank, grasp in enumerate(grasp_list):
        color = _RGB_GRASP_COLORS[min(rank, len(_RGB_GRASP_COLORS) - 1)]
        geometries.append(_build_gripper_wireframe(o3d, grasp, gripper, color))
    return geometries


def show_grasp_scene(
    grasps: Iterable[GraspPoint],
    *,
    points_mm: np.ndarray | None = None,
    support_plane: SupportPlane | None = None,
    gripper_model: GripperGeometryStrategy | None = None,
    config: GraspViewerConfig | None = None,
    window_name: str = "Workaholic-Willy grasps",
) -> None:
    """Open a blocking Open3D viewer with the grasp scene (offline debugging only)."""
    o3d = _import_open3d()
    geometries = build_grasp_scene(
        grasps,
        points_mm=points_mm,
        support_plane=support_plane,
        gripper_model=gripper_model,
        config=config,
    )
    o3d.visualization.draw_geometries(geometries, window_name=window_name)
