"""The planted obstacle: a camera sees a wall nobody declared, and the arm stops planning through it.

Run from the repository root with the project venv. It spawns the real cuRobo sidecar, so it needs
the GPU and up to a minute to warm up::

    .venv/Scripts/python.exe scripts/curobo/probe_live_world.py [--voxel-mm 30] [--json PATH]

Exit codes: ``0`` the chain closes and every required step passes, ``1`` it does not, ``2`` the
planner could not be started.

Everything between the depth frame and the planner is the production path: the same converter, the
same world source, the same refresh call a driver makes before every plan, the same client, the same
protocol. What is synthetic is the camera, and deliberately so. A wall drawn into a depth map is a
wall that appears in no config file anywhere, which is exactly the obstacle a live world exists for
and exactly the one a declared world can never contain.

The chain, three questions:

  1. Does an empty cell plan?
  2. With a wall in front of the goal, does the plan stop?
  3. With the wall taken away, does the cell go back to planning?

Two and three together are necessary and not sufficient. A field that fills the whole grid stops the
plan in two and is gone in three exactly like a wall would be, which is why 4a.3 carries a decoy
control and 4a.8 asks the sign directly.

Then eight questions the installed planner answers and no unit test can, because the sidecar is never
imported by the suite. Each carries the id its result is keyed by in the report:

  4a.1  Did every refresh of the chain say ok, and did the wall reach the planner as a field? REQUIRED.
  4a.2  With the wall sent only as a field, does the plan stop, and does the batch check refuse the
        path the empty cell planned? REQUIRED. The same path against the wall as a box is the control.
  4a.3  Does a declared mesh survive a field? A mesh wall blocks on its own; a field somewhere else
        must not make it disappear, sent as a second request or in one scene request, and must not
        block on its own either. REQUIRED.
  4a.4  How does the planner read the field's values? A one voxel plane is swept toward a fixed arm in
        four encodings (millimetres and metres, each in both signs), and each first contact is
        compared with the same plane sent as a box. RECORDED.
  4a.5  Where does the planner put the grid? The production grid arithmetic for this cell is printed
        beside the sweep. RECORDED.
  4a.6  Does a refresh with nothing to show clear an earlier field? REQUIRED.
  4a.7  Does the exact mesh guard hold the hand where the planner's tool points? The bundle hand is
        placed the way the guard places it and compared with the approach of the planner's tool0.
        RECORDED; it measures whether the guard hand points along the tool approach.
  4a.8  Which sign does the planner read as inside? A uniform field over the arm, both signs and two
        magnitudes, with the same fields far away and no field at all as controls. RECORDED.
  4a.9  Is a field cut to a grid one voxel wider than the planner reserved refused, on the field request
        and on the one scene request, with both grids named, while the matching grid is taken? REQUIRED.

And three where a camera cannot vouch for the cell and the refresh raises instead of refusing:

  4e.1  A camera that stays silent raises after the first reading and its three fresh-frame attempts,
        and registers nothing: a wall the planner already holds still stops the plan. REQUIRED.
  4e.2  A camera that stays blind, every pixel zero, raises the same way and names blindness. REQUIRED.
  4e.3  A camera silent once and then sighted plans, after two readings, and the refresh names the
        camera and the capture time a PLANNED stamp carries. REQUIRED.

And one where the camera rides on the wrist:

  5e.1  A camera on the wrist sees the wall from two tool poses 60 mm apart, each frame placed by the
        tool pose it was stamped with, and both refreshes stop the plan. The same camera over the
        empty bench plans, and a wrist frame with no tool pose is refused before anything is
        registered. REQUIRED.

And one where a world has two cameras:

  5f.1  Two fixed cameras 100 mm apart both see the wall: the refresh names both and vouches PLANNED,
        and the plan stops. The second camera blind raises naming it after its fresh-frame attempts,
        and registers nothing. REQUIRED.

And two where a motion leaves space out of the world it plans against:

  7b.1  The wall both fixed cameras see stops the plan; held as a target box through ``keeping_out``
        the plan goes through; forgotten, it stops the plan again. REQUIRED.
  7e.1  A part drawn between the 2F-85's pads at the chain's goal stops the plan without the goal
        region and plans with it, the refresh counting the points it left out and the stamp carrying
        them; a post drawn where a finger closes still stops the plan with the region in force.
        REQUIRED.

A recorded step prints its numbers and the decision it feeds, and never changes the exit code: its
answer is a measurement, not a pass.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

# Run as a script, `python scripts/curobo/probe_live_world.py` puts only this directory on sys.path,
# and `import src...` then fails on a checkout that was never installed with `pip install -e . --no-deps`.
# Two parents up from here is the root.
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.config.schema.robot import RobotConfig  # noqa: E402
from src.robot.core.errors import CameraWorldUnavailable  # noqa: E402
from src.robot.core.keep_out import SegmentationOffer, keeping_out  # noqa: E402
from src.robot.safety._ur_kinematics import ur_link_transforms_mm  # noqa: E402
from src.robot.safety.planning import CuroboUnavailableError  # noqa: E402
from src.robot.safety.planning.curobo_client import CuroboPlanClient  # noqa: E402
from src.robot.safety.planning.environment import collision_mesh_bundle  # noqa: E402
from src.robot.safety.planning.live_world import (  # noqa: E402
    CameraView,
    DepthSnapshot,
    LivePlannerWorld,
    WorldRefresh,
    _write_voxel_field,
    refresh_planner_world,
)
from src.robot.safety.planning.perceived import (  # noqa: E402
    LinkCapsule,
    SelfEnvelope,
    VoxelField,
    WorldBuildLimits,
    WorldBuildTuning,
    build_voxel_field,
    voxel_grid_extent,
)
from src.robot.safety.planning.reservation import PlannerReservation  # noqa: E402
from src.robot.safety.planning.world import planner_cuboid  # noqa: E402

#: A camera on a mast over the bench, looking straight down from a metre up. The numbers are a
#: plausible cell rather than any particular one: what is being proved is the chain, not a layout.
_SHAPE = (240, 320)
_FX = _FY = 400.0
_CX, _CY = 160.0, 120.0
_CAMERA_HEIGHT_MM = 1000.0
_CAMERA_X_MM = 300.0
_INTRINSICS = np.array([[_FX, 0.0, _CX], [0.0, _FY, _CY], [0.0, 0.0, 1.0]], dtype=np.float64)
_CAMERA_TO_BASE = np.array(
    [
        [1.0, 0.0, 0.0, _CAMERA_X_MM],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, _CAMERA_HEIGHT_MM],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

#: The span in x is 800 mm, which is deliberately not a multiple of the default 30 mm voxel: the
#: production grid rounds it to 810 mm, and 4a.5 prints what that does to where the field lands.
_LIMITS = WorldBuildLimits(
    x_mm=(0.0, 800.0), y_mm=(-400.0, 400.0), z_mm=(-50.0, 700.0), support_plane_top_mm=0.0
)
_BENCH = planner_cuboid("support_plane", (400.0, 0.0, -25.0), (1200.0, 1000.0, 50.0))

#: Where the arm starts and where it is asked to go. The wall stands between them.
_START_Q = [0.0, -1.5, 1.5, -1.57, -1.57, 0.0]
_GOAL_POS_M = [0.55, 0.0, 0.15]
_GOAL_QUAT_WXYZ = [0.0, 1.0, 0.0, 0.0]
_WALL_X_MM = 420.0
_WALL_HEIGHT_MM = 400.0
_WALL_THICKNESS_MM = 40.0
_WALL_WIDTH_MM = 500.0

#: The arm, as one capsule standing clear of the wall so the self filter takes only itself. 135 mm plus the
#: world's 15 mm padding makes a 150 mm radius.
_SELF = SelfEnvelope(
    frames_mm=(np.eye(4),),
    capsules=(LinkCapsule(frame=0, start_mm=(0.0, 0.0, 0.0), end_mm=(0.0, 0.0, 300.0), radius_mm=135.0),),
)

#: A block the camera could see somewhere the path never goes, so a field exists without being in
#: the way. 4a.3 proves it is out of the way rather than assuming it.
_DECOY_MIN_MM = (100.0, 320.0, 0.0)
_DECOY_MAX_MM = (160.0, 380.0, 100.0)

#: Joint vectors for 4a.7. The first is the start pose; the other two turn every joint.
_HAND_JOINTS = (
    _START_Q,
    [0.5, -1.2, 1.3, -1.7, -1.57, 0.3],
    [-0.4, -1.8, 1.9, -1.2, -1.4, -0.6],
)
_HAND_PARTS = ("gripper", "lfinger", "rfinger")
_HAND_MODEL = "ur5e"

#: The field encodings 4a.4 sweeps, as a factor on the exact signed distance in millimetres, positive
#: inside. Production writes the last one.
_ENCODINGS = {"mm_inside_positive": 1.0, "m_inside_positive": 0.001,
              "mm_inside_negative": -1.0, "m_inside_negative": -0.001}

#: The uniform values 4a.8 fills a grid with, and where the far away control puts it.
_SIGN_VALUES = (-500.0, -0.5, 0.5, 500.0)
_FAR_X_MM = 5000.0

_FIELD_FILE = Path(tempfile.gettempdir()) / "willy_probe_live_world_field.npy"


def _probe_cell(voxel_mm: float) -> RobotConfig:
    """This probe's cell as a robot config: workspace, bench, one declared mesh, live scene, and its hand."""
    return RobotConfig.model_validate({
        "vendor": "ur",
        # The hand this cell carries, on the frame the Isaac cell declares, which places the hand model on the
        # identity. Its planner gets it as a body link on the arm's descriptor, the way every cell starts one,
        # so this probe measures the robot a cell actually plans with.
        "gripper": {"model": "robotiq_2f85", "tool_frame": {
            "source": "willy", "offset_mm": [0.0, 132.0, 0.0],
            "rotation_quat_xyzw": [-0.7071067811865476, 0.0, 0.0, 0.7071067811865476],
        }},
        "workspace_limits": {"x_min": _LIMITS.x_mm[0], "x_max": _LIMITS.x_mm[1],
                             "y_min": _LIMITS.y_mm[0], "y_max": _LIMITS.y_mm[1],
                             "z_min": _LIMITS.z_mm[0], "z_max": _LIMITS.z_mm[1]},
        "safety": {
            "payload": {"enforce": False},
            "planning_world": {
                "enabled": True,
                "support_plane": {"height_mm": 0.0, "extent_mm": [1200.0, 1000.0], "thickness_mm": 50.0,
                                  "center_mm": [400.0, 0.0]},
                "perceived": {"voxel_field_mm": voxel_mm, "max_boxes": 8, "pixel_stride": 1},
                "meshes": [{"name": "probe_mesh_wall", "path": "willy_probe_live_world_wall.obj"}],
            },
        },
    })


def _probe_hand_link(voxel_mm: float) -> dict[str, Any]:
    """The hand body this probe's planner starts with, derived from the probe cell's own gripper declaration."""
    from src.contracts import chosen
    from src.robot.safety.planning.body_link import HandLink
    from src.robot.safety.planning.hand import planner_hand

    hand = planner_hand(_probe_cell(voxel_mm))
    if not chosen(hand):
        raise ValueError("the probe cell names no hand, so its planner has no hand body to start with")
    return HandLink.from_hand(hand).to_dict()


class _Camera:
    """A depth camera this probe controls, so the scene can change between plans."""

    def __init__(self) -> None:
        self.depth = self._bench()

    @staticmethod
    def _bench() -> np.ndarray:
        return np.full(_SHAPE, _CAMERA_HEIGHT_MM, dtype=np.float64)

    def show_bench(self) -> None:
        """Nothing but the bench, which is already a declared box and becomes no obstacle."""
        self.depth = self._bench()

    def show_wall(self) -> None:
        """A wall across the path, of the kind that arrives in a cell without anyone editing YAML."""
        depth = self._bench()
        rows, cols = np.mgrid[0 : _SHAPE[0], 0 : _SHAPE[1]]
        top_depth = _CAMERA_HEIGHT_MM - _WALL_HEIGHT_MM
        scale = top_depth / _FX
        centre_col = _CX + (_WALL_X_MM - _CAMERA_X_MM) / scale
        half_thickness = _WALL_THICKNESS_MM / 2.0
        half_width = _WALL_WIDTH_MM / 2.0
        mask = (np.abs(cols - centre_col) <= half_thickness / scale) & (
            np.abs(rows - _CY) <= half_width / scale
        )
        depth[mask] = top_depth
        self.depth = depth

    def grab_surface_depth(self) -> DepthSnapshot:
        return DepthSnapshot(depth_mm=self.depth, intrinsics=_INTRINSICS, timestamp=time.time())


class _ScriptedCamera:
    """A camera that answers each grab with the next of `answers`, and with the last one from then on.

    `"silent"` answers no frame, `"blind"` a frame of zeros, `"bench"` the empty bench. Every frame is
    stamped when it is grabbed, so only the answer can make the world unusable, never its age.
    """

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.grabs = 0

    def grab_surface_depth(self) -> "DepthSnapshot | None":
        answer = self.answers[min(self.grabs, len(self.answers) - 1)]
        self.grabs += 1
        if answer == "silent":
            return None
        depth = np.zeros(_SHAPE, dtype=np.float64) if answer == "blind" else _Camera._bench()
        return DepthSnapshot(depth_mm=depth, intrinsics=_INTRINSICS, timestamp=time.time())


def _refresh(source: LivePlannerWorld, client: CuroboPlanClient) -> WorldRefresh:
    """One refresh, exactly as a driver does it, with the cached frame dropped first.

    The cache exists so two plans inside one perception cycle share a reading. This probe changes
    the scene between plans, which a cell cannot do, so the cache is cleared to make the camera
    speak again.
    """
    source.drop_cached_frames()
    return refresh_planner_world(source=source, client=client, self_envelope=_SELF)


def _plan(client: CuroboPlanClient) -> "list[list[float]] | None":
    return client.plan(_START_Q, _GOAL_POS_M, _GOAL_QUAT_WXYZ)


def _start_is_clear(client: CuroboPlanClient) -> bool:
    return bool(client.check_joints([_START_Q]).valid)


def _wall_cuboid() -> dict[str, Any]:
    return planner_cuboid(
        "probe_wall",
        (_WALL_X_MM, 0.0, _WALL_HEIGHT_MM / 2.0),
        (_WALL_THICKNESS_MM, _WALL_WIDTH_MM, _WALL_HEIGHT_MM),
    )


def _send_field(
    client: CuroboPlanClient,
    values: np.ndarray,
    *,
    dims_mm: Sequence[float],
    voxel_mm: float,
    centre_mm: Sequence[float],
) -> "int | None":
    """Hand the planner a field exactly as production does: a float16 file, dims and pose in metres."""
    np.save(_FIELD_FILE, np.asarray(values, dtype=np.float16))
    return client.set_voxels(
        str(_FIELD_FILE),
        dims_m=[float(d) / 1000.0 for d in dims_mm],
        voxel_size_m=float(voxel_mm) / 1000.0,
        pose=[float(c) / 1000.0 for c in centre_mm] + [1.0, 0.0, 0.0, 0.0],
    )


def _send_production_field(client: CuroboPlanClient, field: VoxelField) -> "int | None":
    """A field exactly as a refresh writes it: the production writer, its file, its sign and its unit."""
    payload = _write_voxel_field(client, field)
    return client.set_voxels(
        payload["path"], dims_m=payload["dims_m"], voxel_size_m=payload["voxel_size_m"],
        pose=payload["pose"],
    )


def _wall_field(source: LivePlannerWorld) -> "VoxelField | None":
    """The wall as the production converter turns the camera's frame into a field."""
    source.drop_cached_frames()
    snapshot = source.world_for(self_envelope=_SELF)
    if not snapshot.usable or snapshot.perceived is None:
        return None
    return snapshot.perceived.voxels


def _block_points(low: Sequence[float], high: Sequence[float], step_mm: float = 5.0) -> np.ndarray:
    axes = [np.arange(lo, hi + 1e-9, step_mm) for lo, hi in zip(low, high, strict=True)]
    grid = np.meshgrid(*axes, indexing="ij")
    return np.column_stack([g.reshape(-1) for g in grid]).astype(np.float64)


def _write_box_obj(path: Path, dims_m: Sequence[float]) -> None:
    """A closed box as OBJ, centred on its origin, in metres, faces wound outward."""
    hx, hy, hz = (float(d) / 2.0 for d in dims_m)
    vertices = [
        (sx * hx, sy * hy, sz * hz) for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)
    ]
    quads = ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3))
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices]
    for a, b, c, d in quads:
        lines.append(f"f {a + 1} {b + 1} {c + 1}")
        lines.append(f"f {a + 1} {c + 1} {d + 1}")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _first_contact(
    place: Callable[[float], None],
    clear: Callable[[], bool],
    *,
    start_mm: float,
    stop_mm: float,
    coarse_mm: float = 10.0,
    fine_mm: float = 1.0,
) -> "float | None":
    """Walk an obstacle from `start_mm` down toward `stop_mm` and return where the arm first touches it.

    `place(x)` puts the obstacle at x and `clear()` asks the planner whether the fixed arm is still
    free. A coarse walk finds the bracket and a fine walk inside it finds the contact. `None` means
    either the arm already touches at the start or nothing touches inside the window.
    """
    x = start_mm
    place(x)
    if not clear():
        return None
    previous = x
    while x - coarse_mm >= stop_mm:
        x -= coarse_mm
        place(x)
        if clear():
            previous = x
            continue
        y = previous
        while y - fine_mm > x + 1e-9:
            y -= fine_mm
            place(y)
            if not clear():
                return y
        return x
    return None


def _quat_matrix(wxyz: Sequence[float]) -> np.ndarray:
    w, x, y, z = (float(v) for v in wxyz)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _yaw(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _grid_shape(extent: "tuple[tuple[float, float, float], float]") -> tuple[int, int, int]:
    (dx, dy, dz), voxel = extent
    nx, ny, nz = (int(round(d / voxel)) for d in (dx, dy, dz))
    return nx, ny, nz


# ---------------------------------------------------------------------------------------------
# The steps
# ---------------------------------------------------------------------------------------------


def _step_field_only(
    client: CuroboPlanClient, camera: _Camera, source: LivePlannerWorld,
    empty_cell_path: "list[list[float]] | None",
) -> dict[str, Any]:
    """4a.2: the wall reaches the planner only as a field."""
    camera.show_wall()
    field = _wall_field(source)
    if field is None:
        return {"passed": False, "why": "the converter built no field from the wall frame"}
    out: dict[str, Any] = {}
    if empty_cell_path:
        client.set_world([_BENCH])
        client.set_voxels(None)
        out["bench_only_path_valid"] = client.check_joints(empty_cell_path).valid
        client.set_world([_BENCH, _wall_cuboid()])
        box_verdict = client.check_joints(empty_cell_path)
        out["wall_box_path_valid"] = box_verdict.valid
        out["wall_box_first_invalid"] = box_verdict.first_invalid
    client.set_world([_BENCH])
    out["voxels_set"] = _send_production_field(client, field)
    out["plan_found"] = bool(_plan(client))
    if empty_cell_path:
        verdict = client.check_joints(empty_cell_path)
        out["wall_field_path_valid"] = verdict.valid
        out["wall_field_first_invalid"] = verdict.first_invalid
    plan_stopped = out["voxels_set"] is not None and not out["plan_found"]
    if not empty_cell_path:
        out["passed"] = plan_stopped
        out["why"] = "the empty cell planned no path, so the batch check had nothing to judge"
    elif out["wall_box_path_valid"]:
        out["passed"] = plan_stopped
        out["why"] = "inconclusive for the batch check: the empty cell path does not cross the wall"
    else:
        out["passed"] = plan_stopped and not out["wall_field_path_valid"]
    return out


def _step_clearing(
    client: CuroboPlanClient, camera: _Camera, source: LivePlannerWorld
) -> dict[str, Any]:
    """4a.6: a field is registered, then a refresh with nothing to show, then a plan through the old wall.

    Runs straight after 4a.2, so the wall field is what the planner holds.
    """
    camera.show_bench()
    refresh = _refresh(source, client)
    found = bool(_plan(client))
    return {"refresh": refresh.to_dict(), "plan_found": found, "passed": refresh.ok and found}


def _step_mesh_survives_field(client: CuroboPlanClient, tuning: WorldBuildTuning) -> dict[str, Any]:
    """4a.3: a declared mesh wall, a field somewhere else, and whether the mesh is still there."""
    obj = Path(tempfile.gettempdir()) / "willy_probe_live_world_wall.obj"
    _write_box_obj(obj, [d / 1000.0 for d in (_WALL_THICKNESS_MM, _WALL_WIDTH_MM, _WALL_HEIGHT_MM)])
    mesh = {
        "name": "probe_mesh_wall",
        "file_path": str(obj),
        "pose": [_WALL_X_MM / 1000.0, 0.0, _WALL_HEIGHT_MM / 2000.0, 1.0, 0.0, 0.0, 0.0],
    }
    decoy = build_voxel_field(
        _block_points(_DECOY_MIN_MM, _DECOY_MAX_MM), limits=_LIMITS, tuning=tuning
    )
    if decoy is None:
        return {"passed": False, "why": "the decoy block built no field"}
    out: dict[str, Any] = {}

    client.set_voxels(None)
    out["mesh_alone_registered"] = client.set_world([_BENCH], [mesh])
    out["mesh_alone_plan_found"] = bool(_plan(client))

    client.set_world([_BENCH])
    out["decoy_alone_voxels_set"] = _send_production_field(client, decoy)
    out["decoy_alone_plan_found"] = bool(_plan(client))

    client.set_voxels(None)
    out["mesh_then_field_registered"] = client.set_world([_BENCH], [mesh])
    out["mesh_then_field_voxels_set"] = _send_production_field(client, decoy)
    out["mesh_then_field_plan_found"] = bool(_plan(client))

    client.set_voxels(None)
    one = client.set_scene([_BENCH], [mesh], _write_voxel_field(client, decoy))
    out["one_request_registered"] = one.world_set
    out["one_request_voxels_set"] = one.voxels_set
    out["one_request_plan_found"] = bool(_plan(client))

    controls_hold = (
        out["mesh_alone_registered"] == 2
        and not out["mesh_alone_plan_found"]
        and out["decoy_alone_voxels_set"] is not None
        and out["decoy_alone_plan_found"]
    )
    out["controls_hold"] = controls_hold
    out["passed"] = (
        controls_hold and not out["mesh_then_field_plan_found"] and not out["one_request_plan_found"]
    )
    if not controls_hold:
        out["why"] = "a control did not hold, so the mesh question has no answer in this cell"
    client.set_voxels(None)
    client.set_world([_BENCH])
    return out


def _step_grid_mismatch(client: CuroboPlanClient, tuning: WorldBuildTuning) -> dict[str, Any]:
    """4a.9: a field whose grid is one voxel wider than the reservation is refused, and the matching one is not.

    A field on the wrong grid registers without an error on a planner that does not check, and puts its
    geometry somewhere the cell is not, so the refusal is the whole point and its reason must name both.
    """
    decoy = build_voxel_field(
        _block_points(_DECOY_MIN_MM, _DECOY_MAX_MM), limits=_LIMITS, tuning=tuning
    )
    if decoy is None:
        return {"passed": False, "why": "the decoy block built no field"}
    payload = _write_voxel_field(client, decoy)
    dims = list(payload["dims_m"])
    wider = dict(payload, dims_m=[dims[0] + float(payload["voxel_size_m"]), *dims[1:]])
    client.set_world([_BENCH])
    out: dict[str, Any] = {
        "set_voxels_wider": client.set_voxels(
            wider["path"], dims_m=wider["dims_m"], voxel_size_m=wider["voxel_size_m"], pose=wider["pose"]
        ),
    }
    refused = client.set_scene([_BENCH], [], wider)
    out["set_scene_wider_voxels_set"] = refused.voxels_set
    out["set_scene_wider_reason"] = refused.reason
    out["set_scene_matching_voxels_set"] = client.set_scene([_BENCH], [], payload).voxels_set
    client.set_voxels(None)
    client.set_world([_BENCH])
    out["passed"] = (
        out["set_voxels_wider"] is None
        and out["set_scene_wider_voxels_set"] is None
        and "reserved" in refused.reason
        and out["set_scene_matching_voxels_set"] is not None
    )
    return out


def _step_camera_faults(
    client: CuroboPlanClient, tuning: WorldBuildTuning
) -> dict[str, dict[str, Any]]:
    """4e.1 to 4e.3: a silent and a blind camera raise after their attempts, and a recovering one plans.

    Against the installed planner, because a raise must leave the planner's world as it was. The wall is
    registered as a box first: while it is there the plan stops, so a raise that registered anything,
    the bench alone for instance, would show up as a plan.
    """
    client.set_voxels(None)
    client.set_world([_BENCH, _wall_cuboid()])
    wall_held_before = not _plan(client)
    steps: dict[str, dict[str, Any]] = {}
    for key, label, answers in (
        ("4e.1", "silent", ("silent",)),
        ("4e.2", "blind", ("blind",)),
        ("4e.3", "recovers", ("silent", "bench")),
    ):
        camera = _ScriptedCamera(*answers)
        source = LivePlannerWorld(
            cameras=(CameraView(name="overhead", depth_source=camera, camera_to_base=_CAMERA_TO_BASE),),
            declared=(_BENCH,),
            limits=_LIMITS,
            tuning=tuning,
            max_age_ms=5000.0,
        )
        entry: dict[str, Any] = {"camera": label, "fresh_frame_attempts": source.fresh_frame_attempts}
        try:
            refresh = refresh_planner_world(source=source, client=client, self_envelope=_SELF)
        except CameraWorldUnavailable as exc:
            entry.update(raised=True, verdict=str(exc.verdict), attempts=exc.attempts, message=str(exc))
            entry["wall_still_stops_the_plan"] = not _plan(client)
        else:
            entry.update(raised=False, refresh=refresh.to_dict(), plan_found=bool(_plan(client)))
        entry["grabs"] = camera.grabs
        steps[key] = entry

    expected_grabs = 1 + int(steps["4e.1"]["fresh_frame_attempts"])
    for key, verdict in (("4e.1", "no_frame"), ("4e.2", "blind")):
        entry = steps[key]
        entry["passed"] = (
            wall_held_before
            and entry["raised"]
            and entry.get("verdict") == verdict
            and entry["grabs"] == expected_grabs
            and bool(entry.get("wall_still_stops_the_plan"))
        )
    recovered = steps["4e.3"]
    refresh_dict = recovered.get("refresh") or {}
    recovered["passed"] = (
        not recovered["raised"]
        and bool(refresh_dict.get("ok"))
        and recovered["grabs"] == 2
        and bool(recovered.get("plan_found"))
        and refresh_dict.get("cameras") == ["overhead"]
        and refresh_dict.get("captured_at_s") is not None
    )
    client.set_voxels(None)
    client.set_world([_BENCH])
    return steps


#: A camera on the wrist, looking along the tool's approach: the mast camera's rotation, carried by
#: the tool (5e.1).
_CAMERA_TO_TOOL = np.diag([1.0, -1.0, -1.0, 1.0])
#: The two tool poses 5e.1 sees the wall from, along base X at the mast camera's height. 60 mm is 40
#: pixels at the wall's top, so the wall lands in two places in the two images and in one place in
#: the cell.
_WRIST_TOOL_X_MM = (_CAMERA_X_MM, _CAMERA_X_MM + 60.0)


class _WristCamera:
    """A depth camera on the tool, over the bench or the wall, stamping each frame with the tool pose."""

    def __init__(self, tool_x_mm: float, *, wall: bool, stamped: bool = True) -> None:
        self.tool_x_mm = tool_x_mm
        self.wall = wall
        self.stamped = stamped

    def _tool_to_base(self) -> np.ndarray:
        pose = np.eye(4)
        pose[0, 3] = self.tool_x_mm
        pose[2, 3] = _CAMERA_HEIGHT_MM
        return pose

    def grab_surface_depth(self) -> DepthSnapshot:
        depth = _Camera._bench()
        if self.wall:
            rows, cols = np.mgrid[0 : _SHAPE[0], 0 : _SHAPE[1]]
            top_depth = _CAMERA_HEIGHT_MM - _WALL_HEIGHT_MM
            scale = top_depth / _FX
            centre_col = _CX + (_WALL_X_MM - self.tool_x_mm) / scale
            mask = (np.abs(cols - centre_col) <= _WALL_THICKNESS_MM / 2.0 / scale) & (
                np.abs(rows - _CY) <= _WALL_WIDTH_MM / 2.0 / scale
            )
            depth[mask] = top_depth
        return DepthSnapshot(
            depth_mm=depth, intrinsics=_INTRINSICS, timestamp=time.time(),
            tool_to_base_mm=self._tool_to_base() if self.stamped else None,
        )


def _step_wrist_camera(client: CuroboPlanClient, tuning: WorldBuildTuning) -> dict[str, Any]:
    """5e.1: a camera on the wrist is placed by the tool pose its frame was stamped with.

    The wall is seen from two tool poses 60 mm apart, and both refreshes must stop the plan; a wall
    placed where the mast camera stands would land 60 mm off in one of them. The empty bench seen from
    the moved pose is the control that the wall stops the plan and the wrist view alone does not. A
    frame with no tool pose must be refused before anything is registered, so a wall box registered
    first still stops the plan.
    """

    def world(camera: _WristCamera) -> LivePlannerWorld:
        return LivePlannerWorld(
            cameras=(CameraView(name="wrist", depth_source=camera, camera_to_tool=_CAMERA_TO_TOOL),),
            declared=(_BENCH,),
            limits=_LIMITS,
            tuning=tuning,
            max_age_ms=5000.0,
        )

    client.set_voxels(None)
    client.set_world([_BENCH])
    entry: dict[str, Any] = {"tool_x_mm": list(_WRIST_TOOL_X_MM)}
    stopped: list[bool] = []
    for tool_x_mm in _WRIST_TOOL_X_MM:
        refresh = refresh_planner_world(
            source=world(_WristCamera(tool_x_mm, wall=True)), client=client, self_envelope=_SELF,
        )
        held = refresh.ok and not _plan(client)
        entry[f"wall_from_{tool_x_mm:.0f}_mm"] = {"refresh": refresh.render(), "plan_stopped": held}
        stopped.append(held)
    bench = refresh_planner_world(
        source=world(_WristCamera(_WRIST_TOOL_X_MM[1], wall=False)), client=client, self_envelope=_SELF,
    )
    entry["bench_plans"] = bool(bench.ok and _plan(client))

    client.set_voxels(None)
    client.set_world([_BENCH, _wall_cuboid()])
    unstamped = refresh_planner_world(
        source=world(_WristCamera(_WRIST_TOOL_X_MM[1], wall=False, stamped=False)), client=client,
        self_envelope=_SELF,
    )
    entry["unstamped"] = {"verdict": str(unstamped.verdict), "reason": unstamped.reason}
    entry["unstamped_registered_nothing"] = not unstamped.ok and not _plan(client)
    entry["passed"] = (
        all(stopped)
        and entry["bench_plans"]
        and str(unstamped.verdict) == "unusable"
        and entry["unstamped_registered_nothing"]
    )
    client.set_voxels(None)
    client.set_world([_BENCH])
    return entry


#: A second fixed camera beside the mast camera, 100 mm further along base X, looking down the same
#: way (5f.1).
_SIDE_CAMERA_X_MM = _CAMERA_X_MM + 100.0


def _camera_to_base_at(x_mm: float) -> np.ndarray:
    """The mast camera's CAMERA to BASE, carried to ``x_mm`` along base X."""
    placed = _CAMERA_TO_BASE.copy()
    placed[0, 3] = x_mm
    return placed


class _FixedCamera:
    """A fixed depth camera at ``x_mm`` along base X that sees the wall, or a blind one: every pixel zero."""

    def __init__(self, x_mm: float, *, blind: bool = False) -> None:
        self.x_mm = x_mm
        self.blind = blind

    def grab_surface_depth(self) -> DepthSnapshot:
        if self.blind:
            return DepthSnapshot(depth_mm=np.zeros(_SHAPE, dtype=np.float64), intrinsics=_INTRINSICS,
                                 timestamp=time.time())
        # The wrist camera's rendering from a camera at `x_mm`, with no tool pose: this camera does not move.
        return _WristCamera(self.x_mm, wall=True, stamped=False).grab_surface_depth()


def _step_two_cameras(client: CuroboPlanClient, tuning: WorldBuildTuning) -> dict[str, Any]:
    """5f.1: a world from two fixed cameras names both, and one of them blind stops the refresh naming it.

    Both cameras see the wall from 100 mm apart. The refresh must vouch for both, with a PLANNED stamp
    naming the two, and the plan must stop. With the second camera blind the refresh raises naming it
    after its fresh-frame attempts, and a wall box registered first still stops the plan, so the raise
    registered nothing.
    """

    def world(side: _FixedCamera) -> LivePlannerWorld:
        return LivePlannerWorld(
            cameras=(
                CameraView(name="overhead", depth_source=_FixedCamera(_CAMERA_X_MM), camera_to_base=_CAMERA_TO_BASE),
                CameraView(name="side", depth_source=side, camera_to_base=_camera_to_base_at(_SIDE_CAMERA_X_MM)),
            ),
            declared=(_BENCH,),
            limits=_LIMITS,
            tuning=tuning,
            max_age_ms=5000.0,
        )

    client.set_voxels(None)
    client.set_world([_BENCH])
    entry: dict[str, Any] = {}
    sighted = refresh_planner_world(source=world(_FixedCamera(_SIDE_CAMERA_X_MM)), client=client, self_envelope=_SELF)
    stamp = sighted.camera_world()
    entry["sighted"] = {"refresh": sighted.render(), "cameras": list(sighted.cameras),
                        "stamp": None if stamp is None else stamp.render()}
    entry["sighted_plan_stopped"] = sighted.ok and not _plan(client)

    client.set_voxels(None)
    client.set_world([_BENCH, _wall_cuboid()])
    source = world(_FixedCamera(_SIDE_CAMERA_X_MM, blind=True))
    try:
        refresh_planner_world(source=source, client=client, self_envelope=_SELF)
    except CameraWorldUnavailable as exc:
        entry["blind"] = {"raised": True, "camera": exc.camera, "attempts": exc.attempts, "verdict": str(exc.verdict)}
        entry["blind_registered_nothing"] = not _plan(client)
    else:
        entry["blind"] = {"raised": False}
        entry["blind_registered_nothing"] = False
    entry["passed"] = (
        bool(entry["sighted_plan_stopped"])
        and entry["sighted"]["cameras"] == ["overhead", "side"]
        and stamp is not None
        and stamp.use.name == "PLANNED"
        and entry["blind"].get("raised") is True
        and entry["blind"].get("camera") == "side"
        and entry["blind"].get("attempts") == 1 + source.fresh_frame_attempts
        and bool(entry["blind_registered_nothing"])
    )
    client.set_voxels(None)
    client.set_world([_BENCH])
    return entry


def _step_two_camera_target(client: CuroboPlanClient, tuning: WorldBuildTuning) -> dict[str, Any]:
    """7b.1: a target both fixed cameras see plans with the held offer, and stops again once forgotten.

    The wall stands for a part the pick is reaching into. Seen by both cameras it stops the plan.
    Offered as a target box alone, held through ``keeping_out`` on an arm that carries the world, it
    leaves both views and the plan goes through; after the block the world has the wall back and the
    plan stops again.
    """
    from types import SimpleNamespace

    world = LivePlannerWorld(
        cameras=(
            CameraView(name="overhead", depth_source=_FixedCamera(_CAMERA_X_MM), camera_to_base=_CAMERA_TO_BASE),
            CameraView(name="side", depth_source=_FixedCamera(_SIDE_CAMERA_X_MM),
                       camera_to_base=_camera_to_base_at(_SIDE_CAMERA_X_MM)),
        ),
        declared=(_BENCH,),
        limits=_LIMITS,
        tuning=tuning,
        max_age_ms=5000.0,
    )
    arm = SimpleNamespace(live_planner_world=world)
    half_t, half_w = _WALL_THICKNESS_MM / 2.0, _WALL_WIDTH_MM / 2.0
    top = _block_points((_WALL_X_MM - half_t, -half_w, _WALL_HEIGHT_MM), (_WALL_X_MM + half_t, half_w, _WALL_HEIGHT_MM))
    entry: dict[str, Any] = {}

    client.set_voxels(None)
    client.set_world([_BENCH])
    seen = refresh_planner_world(source=world, client=client, self_envelope=_SELF)
    entry["seen"] = seen.render()
    entry["seen_plan_stopped"] = seen.ok and not _plan(client)

    offer = SegmentationOffer(captured_at_s=time.time(), target_points_base_mm=top, target_label="wall")
    with keeping_out(arm, offer) as scope:
        world.drop_cached_frames()
        held = refresh_planner_world(source=world, client=client, self_envelope=_SELF)
        entry["held"] = held.render()
        entry["held_world_wired"] = scope.world_wired
        entry["held_plans"] = held.ok and bool(_plan(client))

    world.drop_cached_frames()
    after = refresh_planner_world(source=world, client=client, self_envelope=_SELF)
    entry["forgotten"] = after.render()
    entry["forgotten_plan_stopped"] = after.ok and not _plan(client)
    entry["passed"] = bool(entry["seen_plan_stopped"] and entry["held_plans"] and entry["forgotten_plan_stopped"])
    client.set_voxels(None)
    client.set_world([_BENCH])
    return entry


class _PointsCamera:
    """The mast camera over the bench, seeing ``points_base_mm`` as their nearest surface."""

    def __init__(self, points_base_mm: np.ndarray) -> None:
        self.depth = _Camera._bench()
        points = np.asarray(points_base_mm, dtype=np.float64)
        camera = (np.linalg.inv(_CAMERA_TO_BASE) @ np.column_stack([points, np.ones(len(points))]).T).T[:, :3]
        cols = np.rint(_FX * camera[:, 0] / camera[:, 2] + _CX).astype(int)
        rows = np.rint(_FY * camera[:, 1] / camera[:, 2] + _CY).astype(int)
        seen = (camera[:, 2] > 0.0) & (cols >= 0) & (cols < _SHAPE[1]) & (rows >= 0) & (rows < _SHAPE[0])
        np.minimum.at(self.depth, (rows[seen], cols[seen]), camera[seen, 2])

    def grab_surface_depth(self) -> DepthSnapshot:
        return DepthSnapshot(depth_mm=self.depth.copy(), intrinsics=_INTRINSICS, timestamp=time.time())


def _local_block(box_to_base_mm: np.ndarray, x: Sequence[float], y: Sequence[float], z: Sequence[float]) -> np.ndarray:
    """A solid block, 1 mm apart, spanning ``x``, ``y`` and ``z`` in a box's own axes, as BASE points."""
    xs, ys, zs = np.meshgrid(*(np.arange(lo, hi + 1e-9, 1.0) for lo, hi in (x, y, z)), indexing="ij")
    local = np.column_stack([xs.ravel(), ys.ravel(), zs.ravel(), np.ones(xs.size)])
    return (box_to_base_mm @ local.T).T[:, :3]


def _step_jaw_region(client: CuroboPlanClient, tuning: WorldBuildTuning, voxel_mm: float) -> dict[str, Any]:
    """7e.1: a part between the pads at the goal leaves the world with the goal region, a post at a finger stays.

    The goal is the chain's own flange goal, and its TCP is that goal on this cell's declared tool
    frame, which is how a driver places the region. The part is drawn inside the region, 60 mm across
    the closing axis, so the camera sees its top: without the region it is an obstacle the goal stands
    in and the plan stops; with ``goal_keep_out`` the refresh leaves it out, counts the points, and the
    plan goes through. The post is drawn 46 to 54 mm out along the closing axis, where a finger closes
    and outside the region: with the region in force it still stops the plan.
    """
    from types import SimpleNamespace

    from src.robot.drivers.ur.tool_frame import tool_frame_matrix
    from src.robot.safety.planning.hand import planner_hand
    from src.robot.safety.planning.self_envelope import goal_keep_out

    cell = _probe_cell(voxel_mm)
    frame = cell.gripper.tool_frame
    flange = np.eye(4)
    flange[:3, :3] = _quat_matrix(_GOAL_QUAT_WXYZ)
    flange[:3, 3] = np.asarray(_GOAL_POS_M, dtype=np.float64) * 1000.0
    tcp = flange @ tool_frame_matrix(frame.offset_mm, frame.rotation_quat_xyzw)
    hand = planner_hand(cell)
    goal = goal_keep_out(SimpleNamespace(planner_hand=lambda arm=None: hand), None, tcp)
    entry: dict[str, Any] = {"tcp_mm": [round(float(v), 2) for v in tcp[:3, 3]], "reason": goal.reason}
    if goal.region is None:
        entry["passed"] = False
        return entry
    entry["region"] = goal.region.render()
    part = _local_block(goal.region.matrix(), (-30.0, 30.0), (-10.0, 10.0), (-15.0, 15.0))
    post = _local_block(goal.region.matrix(), (46.0, 54.0), (-10.0, 10.0), (-15.0, 15.0))
    entry["part_inside_region"] = bool(goal.region.contains(part).all())
    entry["post_outside_region"] = not bool(goal.region.contains(post).any())

    def world(points: np.ndarray) -> LivePlannerWorld:
        return LivePlannerWorld(
            cameras=(CameraView(name="overhead", depth_source=_PointsCamera(points), camera_to_base=_CAMERA_TO_BASE),),
            declared=(_BENCH,),
            limits=_LIMITS,
            tuning=tuning,
            max_age_ms=5000.0,
        )

    client.set_voxels(None)
    client.set_world([_BENCH])
    bare = refresh_planner_world(source=world(part), client=client, self_envelope=_SELF)
    entry["without_region"] = bare.render()
    entry["without_region_plan_stopped"] = bare.ok and not _plan(client)

    kept = refresh_planner_world(source=world(part), client=client, self_envelope=_SELF, goal_keep_out=goal)
    stamp = kept.camera_world()
    entry["with_region"] = kept.render()
    entry["goal_points"] = None if kept.keep_out is None else kept.keep_out.goal_points
    entry["stamp"] = None if stamp is None else stamp.render()
    entry["with_region_plans"] = kept.ok and bool(_plan(client))

    finger = refresh_planner_world(source=world(np.vstack([part, post])), client=client, self_envelope=_SELF,
                                   goal_keep_out=goal)
    entry["post_at_a_finger"] = finger.render()
    entry["post_plan_stopped"] = finger.ok and not _plan(client)
    entry["passed"] = bool(
        entry["part_inside_region"] and entry["post_outside_region"]
        and entry["without_region_plan_stopped"] and entry["with_region_plans"] and entry["post_plan_stopped"]
        and (entry["goal_points"] or 0) > 0
        and stamp is not None and stamp.keep_out is not None
    )
    client.set_voxels(None)
    client.set_world([_BENCH])
    return entry


def _step_sign(
    client: CuroboPlanClient, extent: "tuple[tuple[float, float, float], float]"
) -> dict[str, Any]:
    """4a.8: fill the whole grid with one value, over the arm and far from it, and ask whether the arm is clear.

    No geometry is involved, so nothing about grid placement or voxel order can hide the answer: a value
    the planner reads as inside makes every sphere in the grid collide, and a value it reads as free
    makes none.
    """
    (dx, dy, dz), voxel = extent
    count = int(np.prod(_grid_shape(extent)))
    centre_y = (_LIMITS.y_mm[0] + _LIMITS.y_mm[1]) / 2.0
    centre_z = float(_LIMITS.support_plane_top_mm or 0.0) + dz / 2.0
    over_arm_x = _LIMITS.x_mm[0] + dx / 2.0

    client.set_world([_BENCH])
    client.set_voxels(None)
    out: dict[str, Any] = {"no_field_start_clear": _start_is_clear(client), "over_arm": {}, "far_away": {}}
    for label, x_mm in (("over_arm", over_arm_x), ("far_away", _FAR_X_MM)):
        for value in _SIGN_VALUES:
            registered = _send_field(
                client, np.full(count, value, dtype=np.float32), dims_mm=(dx, dy, dz), voxel_mm=voxel,
                centre_mm=(x_mm, centre_y, centre_z),
            )
            out[label][str(value)] = None if registered is None else _start_is_clear(client)
    client.set_voxels(None)

    over = out["over_arm"]
    negative_clear = bool(over["-0.5"]) and bool(over["-500.0"])
    positive_clear = bool(over["0.5"]) and bool(over["500.0"])
    if negative_clear and not positive_clear:
        out["inside_is"] = "positive"
    elif positive_clear and not negative_clear:
        out["inside_is"] = "negative"
    elif not negative_clear and not positive_clear:
        out["inside_is"] = "both, the grid blocks whatever it holds"
    else:
        out["inside_is"] = "neither, the field is ignored"
    out["controls_hold"] = bool(out["no_field_start_clear"]) and all(
        v is True for v in out["far_away"].values()
    )
    return out


def _step_sweep(
    client: CuroboPlanClient, extent: "tuple[tuple[float, float, float], float]"
) -> dict[str, Any]:
    """4a.4 and 4a.5: sweep a one voxel plane toward the fixed arm as a box and in four field encodings.

    The plane is moved by moving the pose of the grid, so it travels continuously and the field is
    the same array every time. Its values are the exact signed distance to a slab one voxel thick, so
    the encoding the planner reads correctly touches the arm where the box of the same size does. An
    encoding it misreads touches earlier or later, and one that turns the whole grid into an obstacle
    touches when the grid's low edge reaches the arm, half the grid ahead of the plane.
    """
    (dx, dy, dz), voxel = extent
    nx, ny, nz = _grid_shape(extent)
    index = nx // 2
    #: Where the plane centre sits relative to the pose centre, if a voxel centre is half a voxel
    #: inside the low corner.
    plane_offset = -dx / 2.0 + (index + 0.5) * voxel
    along = voxel / 2.0 - np.abs(np.arange(nx) - index) * voxel
    values_mm = np.broadcast_to(along[:, None, None], (nx, ny, nz)).astype(np.float32).reshape(-1)
    centre_z = float(_LIMITS.support_plane_top_mm or 0.0) + dz / 2.0

    def place_box(x: float) -> None:
        client.set_world([_BENCH, planner_cuboid("probe_plane", (x, 0.0, centre_z), (voxel, dy, dz))])

    def place_field(factor: float) -> Callable[[float], None]:
        np.save(_FIELD_FILE, (values_mm * factor).astype(np.float16))

        def place(x: float) -> None:
            client.set_voxels(
                str(_FIELD_FILE),
                dims_m=[dx / 1000.0, dy / 1000.0, dz / 1000.0],
                voxel_size_m=voxel / 1000.0,
                pose=[(x - plane_offset) / 1000.0, 0.0, centre_z / 1000.0, 1.0, 0.0, 0.0, 0.0],
            )

        return place

    window = {"start_mm": 1100.0, "stop_mm": 0.0}
    out: dict[str, Any] = {"voxel_mm": voxel, "window": window, "grid_half_width_x_mm": dx / 2.0}

    client.set_voxels(None)
    out["box_contact_mm"] = _first_contact(place_box, lambda: _start_is_clear(client), **window)
    client.set_world([_BENCH])
    box = out["box_contact_mm"]
    for name, factor in _ENCODINGS.items():
        contact = _first_contact(place_field(factor), lambda: _start_is_clear(client), **window)
        client.set_voxels(None)
        out[f"{name}_contact_mm"] = contact
        out[f"{name}_minus_box_mm"] = None if box is None or contact is None else contact - box
    known = {
        name: abs(out[f"{name}_minus_box_mm"])
        for name in _ENCODINGS if out[f"{name}_minus_box_mm"] is not None
    }
    out["closest_encoding"] = min(known, key=known.__getitem__) if known else None
    out["closest_minus_box_mm"] = (
        None if out["closest_encoding"] is None else out[f"{out['closest_encoding']}_minus_box_mm"]
    )
    client.set_world([_BENCH])

    # The production grid for this cell, as arithmetic: the field is indexed from the low corner of
    # the rounded grid while its pose centre comes from the unrounded span.
    span = _LIMITS.x_mm[1] - _LIMITS.x_mm[0]
    out["production_x_span_mm"] = span
    out["production_x_dims_mm"] = dx
    out["production_centre_offset_mm"] = (span - dx) / 2.0
    return out


def _step_hand(client: CuroboPlanClient) -> dict[str, Any]:
    """4a.7: the bundle hand where the guard places it, against the approach of the planner's tool0."""
    bundle = collision_mesh_bundle(_HAND_MODEL)
    with np.load(bundle) as data:
        parts = [p for p in _HAND_PARTS if f"{p}__v" in data.files]
        frames = sorted({int(data[f"{p}__frame"][0]) for p in parts})
        hand = np.vstack([np.asarray(data[f"{p}__v"], dtype=np.float64) for p in parts])
    out: dict[str, Any] = {"bundle": str(bundle), "hand_parts": parts, "hand_frames": frames, "poses": []}
    if len(frames) != 1:
        out["why"] = "the hand parts sit in more than one frame, so there is no single hand direction"
        return out
    frame = frames[0]
    centroid = hand.mean(axis=0)
    out["hand_centroid_in_frame_mm"] = np.round(centroid, 2).tolist()
    for joints in _HAND_JOINTS:
        fk = client.fk(list(joints))
        transforms = ur_link_transforms_mm(_HAND_MODEL, np.asarray(joints, dtype=np.float64))
        if fk is None or transforms is None:
            out["poses"].append({"joints": list(joints), "why": "no forward kinematics"})
            continue
        tool_pos_mm = np.asarray(fk[0], dtype=np.float64) * 1000.0
        tool_rot = _quat_matrix(fk[1])
        approach = tool_rot[:, 2]
        per_yaw = {}
        for yaw in (0.0, 180.0):
            rb = _yaw(yaw)
            flange = rb @ transforms[6][:3, 3]
            placed = rb @ transforms[frame][:3, :3]
            direction = placed @ centroid
            per_yaw[yaw] = {
                "flange_error_mm": float(np.linalg.norm(flange - tool_pos_mm)),
                "hand_vs_approach_deg": _angle_deg(direction, approach),
                "tool0_axes_in_frame": np.round(placed.T @ tool_rot, 4).tolist(),
            }
        yaw_best = min(per_yaw, key=lambda y: per_yaw[y]["flange_error_mm"])
        out["poses"].append(
            {"joints": list(joints), "base_yaw_deg": yaw_best, **per_yaw[yaw_best],
             "other_yaw_flange_error_mm": per_yaw[180.0 - yaw_best]["flange_error_mm"]}
        )
    angles = [p["hand_vs_approach_deg"] for p in out["poses"] if "hand_vs_approach_deg" in p]
    out["hand_vs_approach_deg_max"] = max(angles) if angles else None
    out["hand_vs_approach_deg_min"] = min(angles) if angles else None
    return out


def _print_decisions(steps: dict[str, dict[str, Any]]) -> None:
    print("\nWhat each answer decides")
    sign = steps.get("4a.8", {})
    if sign.get("inside_is"):
        held = "controls hold" if sign.get("controls_hold") else "a control did not hold"
        print(f"  4a.8: the planner reads {sign['inside_is']} as inside ({held})")
    sweep = steps.get("4a.4", {})
    if sweep.get("closest_encoding"):
        print(f"  4a.4: the encoding that touches where the box does is {sweep['closest_encoding']}, "
              f"{sweep['closest_minus_box_mm']:+.0f} mm from the box; production writes m_inside_negative, "
              f"{sweep.get('m_inside_negative_minus_box_mm')} mm from the box")
    mesh = steps.get("4a.3", {})
    if mesh.get("passed") is False and mesh.get("controls_hold"):
        print("  4a.3: a field removes the declared mesh, so 4b keeps meshes under a field")
    if steps.get("4a.6", {}).get("passed") is False:
        print("  4a.6: a refresh with nothing to show kept the old field, so 4b sends an explicit clear")
    hand = steps.get("4a.7", {})
    low = hand.get("hand_vs_approach_deg_min")
    if low is not None:
        if low > 45.0:
            print(f"  4a.7: the guard hand is at least {low:.1f} deg off the tool approach, so the "
                  "premise of Q8 holds and 4g re-bakes the hand into the flange frame")
        else:
            print(f"  4a.7: the guard hand lies within {hand['hand_vs_approach_deg_max']:.1f} deg "
                  "of the tool approach, so the premise of Q8 does not hold and 4g is not built")


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(
        prog="probe_live_world",
        description="Prove that a wall the cameras see, and nothing declares, stops a plan; then ask the "
        "installed planner what Step 4 builds on.",
    )
    parser.add_argument(
        "--voxel-mm", type=float, default=30.0,
        help="resolution of the distance field handed to the planner (0 sends boxes only)",
    )
    parser.add_argument(
        "--json", type=str, default=None,
        help="where to write the numbers (default logs/curobo/probe_live_world_<time>.json)",
    )
    args = parser.parse_args(argv)

    tuning = WorldBuildTuning(pixel_stride=1, voxel_field_mm=args.voxel_mm, max_boxes=8)
    extent = voxel_grid_extent(_LIMITS, tuning.voxel_field_mm)
    # The client is sized the way a cell sizes it: from a robot config describing this probe's cell, not
    # from a string built by hand, so the probe exercises the reservation the arms start their planner with.
    planned = PlannerReservation.from_config(robot_cfg=_probe_cell(args.voxel_mm))
    reservation = planned.voxel_grid
    print(f"planner reservation: {planned.render()}")

    # The planner a cell starts: this arm's own descriptor, with the cell's hand added as a body link when the
    # sidecar comes up. Started bare, this probe would measure a robot no cell plans with.
    from src.robot.drivers.sim.robot_models import curobo_arm_descriptor

    client = CuroboPlanClient(
        robot_config=curobo_arm_descriptor(_HAND_MODEL), body_links=[_probe_hand_link(args.voxel_mm)],
    )
    client.reserve_world(planned)
    started = time.perf_counter()
    try:
        client.start()
    except CuroboUnavailableError as exc:
        print(f"the planner could not be started: {exc}", file=sys.stderr)
        return 2
    print(f"sidecar ready in {time.perf_counter() - started:.1f} s")

    camera = _Camera()
    source = LivePlannerWorld(
        cameras=(CameraView(name="overhead", depth_source=camera, camera_to_base=_CAMERA_TO_BASE),),
        declared=(_BENCH,),
        limits=_LIMITS,
        tuning=tuning,
        max_age_ms=5000.0,
    )
    report: dict[str, Any] = {"voxel_mm": args.voxel_mm, "reservation": reservation, "chain": {}}

    results: dict[str, bool] = {}
    refreshes: dict[str, WorldRefresh] = {}
    empty_cell_path: "list[list[float]] | None" = None
    for step, (label, scene) in enumerate(
        (
            ("an empty cell", camera.show_bench),
            ("a wall nobody declared", camera.show_wall),
            ("the wall taken away", camera.show_bench),
        ),
        start=1,
    ):
        scene()
        print(f"\n{step}. {label}")
        refresh = _refresh(source, client)
        print(f"   {refresh.render()}")
        path = _plan(client)
        found = bool(path)
        if step == 1:
            empty_cell_path = path
        print(f"   plan: {'found' if found else 'NONE'}")
        results[label] = found
        refreshes[label] = refresh
        report["chain"][label] = {"refresh": refresh.to_dict(), "plan_found": found}

    closed = (
        results["an empty cell"]
        and not results["a wall nobody declared"]
        and results["the wall taken away"]
    )
    report["chain"]["closed"] = closed

    steps: dict[str, dict[str, Any]] = {}
    wall_refresh = refreshes["a wall nobody declared"]
    steps["4a.1"] = {
        "passed": all(r.ok for r in refreshes.values())
        and (extent is None or bool(wall_refresh.voxels_registered)),
        "voxels_registered_by_the_wall": wall_refresh.voxels_registered,
    }
    logging.getLogger("CuroboPlanClient").setLevel(logging.ERROR)
    try:
        if extent is None:
            skipped = {"passed": None, "why": "no voxel reservation, so there is no field to ask about"}
            for key in ("4a.2", "4a.3", "4a.4", "4a.5", "4a.6", "4a.8", "4a.9"):
                steps[key] = dict(skipped)
        else:
            steps["4a.2"] = _step_field_only(client, camera, source, empty_cell_path)
            steps["4a.6"] = _step_clearing(client, camera, source)
            steps["4a.3"] = _step_mesh_survives_field(client, tuning)
            steps["4a.9"] = _step_grid_mismatch(client, tuning)
            steps["4a.8"] = _step_sign(client, extent)
            sweep = _step_sweep(client, extent)
            production = ("production_x_span_mm", "production_x_dims_mm", "production_centre_offset_mm")
            steps["4a.4"] = {k: v for k, v in sweep.items() if k not in production}
            steps["4a.5"] = {k: sweep[k] for k in ("voxel_mm", "grid_half_width_x_mm", *production)}
        steps.update(_step_camera_faults(client, tuning))
        steps["5e.1"] = _step_wrist_camera(client, tuning)
        steps["5f.1"] = _step_two_cameras(client, tuning)
        steps["7b.1"] = _step_two_camera_target(client, tuning)
        steps["7e.1"] = _step_jaw_region(client, tuning, args.voxel_mm)
        steps["4a.7"] = _step_hand(client)
    finally:
        logging.getLogger("CuroboPlanClient").setLevel(logging.INFO)
        client.close()

    required = ("4a.1", "4a.2", "4a.3", "4a.6", "4a.9", "4e.1", "4e.2", "4e.3", "5e.1", "5f.1", "7b.1", "7e.1")
    report["steps"] = steps
    failing = [k for k in required if steps[k].get("passed") is False]
    report["required_failing"] = failing

    print("\nStep 4 questions")
    for key in sorted(steps):
        kind = "required" if key in required else "recorded"
        passed = steps[key].get("passed")
        word = {True: "PASS", False: "FAIL", None: "n/a"}[passed] if kind == "required" else "RECORD"
        shown = {k: v for k, v in steps[key].items() if k not in ("passed", "refresh", "poses")}
        print(f"  {key} [{kind}] {word}: {json.dumps(shown, default=str)}")
    for pose in steps["4a.7"].get("poses", []):
        print(f"  4a.7 pose {pose.get('joints')}: {json.dumps({k: v for k, v in pose.items() if k != 'joints'})}")

    _print_decisions(steps)

    json_path = Path(args.json) if args.json else (
        _ROOT / "logs" / "curobo" / f"probe_live_world_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nnumbers written to {json_path}")

    print()
    if closed and not failing:
        print("the chain closes and every required step passes")
        return 0
    if not closed:
        print("the chain does NOT close.", file=sys.stderr)
        for label, found in results.items():
            print(f"  {label}: {'plan found' if found else 'no plan'}", file=sys.stderr)
    if failing:
        print(f"required step(s) failing: {', '.join(failing)}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
