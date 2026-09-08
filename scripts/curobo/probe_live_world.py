"""The planted obstacle: a camera sees a wall nobody declared, and the arm stops planning through it.

Run from the repository root with the project venv. It spawns the real cuRobo sidecar, so it needs
the GPU and a few seconds to warm up::

    .venv/Scripts/python.exe scripts/curobo/probe_live_world.py

Exit codes: ``0`` the chain closes, ``1`` it does not, ``2`` the planner could not be started.

Everything between the depth frame and the planner is the production path: the same converter, the
same world source, the same refresh call a driver makes before every plan, the same client, the same
protocol. What is synthetic is the camera, and deliberately so. A wall drawn into a depth map is a
wall that appears in no config file anywhere, which is exactly the obstacle a live world exists for
and exactly the one a declared world can never contain.

Three questions, and the third is the one that makes it a proof:

  1. Does an empty cell plan?
  2. With a wall in front of the goal, does the plan stop?
  3. With the wall taken away, does the cell go back to planning?

Two alone proves nothing: a planner that refuses everything would look identical. Two and three
together say the planner saw the wall.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Run as a script, `python scripts/curobo/probe_live_world.py` puts only this directory on sys.path,
# and `import src...` then fails. The repository is not pip installable, so there is no import
# path without this. Two parents up from here is the root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.robot.safety.planning import CuroboUnavailableError  # noqa: E402
from src.robot.safety.planning.curobo_client import CuroboPlanClient
from src.robot.safety.planning.live_world import (  # noqa: E402
    CameraView,
    DepthSnapshot,
    LivePlannerWorld,
    refresh_planner_world,
)
from src.robot.safety.planning.perceived import (  # noqa: E402
    WorldBuildLimits,
    WorldBuildTuning,
    voxel_grid_extent,
)
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

#: The arm, as two capsule ends, standing clear of the wall so the self filter takes only itself.
_LINKS = [[0.0, 0.0, 0.0], [0.0, 0.0, 300.0]]


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
        mask = (np.abs(cols - centre_col) <= 20.0 / scale) & (np.abs(rows - _CY) <= 250.0 / scale)
        depth[mask] = top_depth
        self.depth = depth

    def grab_surface_depth(self) -> DepthSnapshot:
        return DepthSnapshot(depth_mm=self.depth, intrinsics=_INTRINSICS, timestamp=time.time())


def _refresh(source: LivePlannerWorld, client: CuroboPlanClient) -> str:
    """One refresh, exactly as a driver does it, with the cached frame dropped first.

    The cache exists so two plans inside one perception cycle share a reading. This probe changes
    the scene between plans, which a cell cannot do, so the cache is cleared to make the camera
    speak again.
    """
    source.drop_cached_frames()
    return refresh_planner_world(source=source, client=client, link_origins_mm=_LINKS).render()


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(
        prog="probe_live_world",
        description="Prove that a wall the cameras see, and nothing declares, stops a plan.",
    )
    parser.add_argument(
        "--voxel-mm", type=float, default=30.0,
        help="resolution of the distance field handed to the planner (0 sends boxes only)",
    )
    args = parser.parse_args(argv)

    tuning = WorldBuildTuning(pixel_stride=1, voxel_field_mm=args.voxel_mm, max_boxes=8)
    extent = voxel_grid_extent(_LIMITS, tuning.voxel_field_mm)
    reservation = (
        "" if extent is None else ",".join(f"{v / 1000.0:.4f}" for v in (*extent[0], extent[1]))
    )
    print(f"voxel reservation: {reservation or 'none, boxes only'}")

    client = CuroboPlanClient(mesh_cache=4, voxel_grid=reservation)
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

    results: dict[str, bool] = {}
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
        print(f"   {_refresh(source, client)}")
        found = bool(client.plan(_START_Q, _GOAL_POS_M, _GOAL_QUAT_WXYZ))
        print(f"   plan: {'found' if found else 'NONE'}")
        results[label] = found

    client.close()

    closed = (
        results["an empty cell"]
        and not results["a wall nobody declared"]
        and results["the wall taken away"]
    )
    print()
    if closed:
        print("the chain closes: the planner saw a wall that exists in no config, and went around "
              "nothing once it was gone")
        return 0
    print("the chain does NOT close.", file=sys.stderr)
    for label, found in results.items():
        print(f"  {label}: {'plan found' if found else 'no plan'}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
