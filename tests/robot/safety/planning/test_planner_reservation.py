"""One reservation decides what the planner allocates when it starts, for every client that starts one.

cuRobo allocates its collision storage once, when the sidecar starts: so many box slots, so many mesh
slots, and a voxel grid of a fixed size. Before this, the UR arm's default client and the sim client both
reserved 16 boxes, no mesh and no grid whatever the cell declared, and the one function that computed a
grid had no caller. A cell that declared a tote as a mesh, or turned the live scene on, was refused at its
first motion by a planner that had nowhere to put either.

The slot counts are derived (owner, Step 4 Q3): as many boxes as the cell declares plus the 128 camera
boxes every cell's world holds since 2026-10-17, never fewer than 16, one mesh slot per declared mesh, and
the distance field's grid at its 10 mm wherever a live scene is built. The config wins over a hand set
environment variable, and says so.
"""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest import mock


from src.config.schema.robot import RobotConfig
from src.robot.safety.planning.curobo_client import CuroboPlanClient
from src.robot.safety.planning.fine_world import FINE_REACH_MM, FineRegion
from src.robot.safety.planning.perceived import (
    CAMERA_FIELD_VOXEL_MM,
    CAMERA_MAX_BOXES,
    WorldBuildLimits,
    WorldBuildTuning,
    build_voxel_field,
)

_ENV = ("WILLY_CUROBO_CUBOID_CACHE", "WILLY_CUROBO_MESH_CACHE", "WILLY_CUROBO_VOXEL_GRID")
_LIMITS = WorldBuildLimits(
    x_mm=(-500.0, 500.0), y_mm=(-500.0, 500.0), z_mm=(0.0, 600.0), support_plane_top_mm=0.0
)
#: Every cell's 10 mm field: 1000 mm is 100 voxels on both horizontal axes, 600 mm is 60.
_GRID = "1.0000,1.0000,0.6000,0.0100"
#: The fine grid about a goal beside it (``perceived.near_grid_extent``): the 150 mm square and 120 mm past it on each
#: side, 540 mm at 4 mm, as high as the field at the most.
_NEAR = "0.5400,0.5400,0.6000,0.0040"


def _cell(
    *,
    fixtures: int = 0,
    meshes: int = 0,
    bench: bool = True,
    planner: str = "curobo",
    planner_margin_mm: "float | None" = 4.0,
) -> RobotConfig:
    """A UR cell. It declares a planner margin by default, because since B1 S17 a cuRobo cell that declares none
    refuses to start a planner, and every case here is about the reservation rather than about the margin. ``bench``
    declares the bench, which turns the planning world on; the guard's ``fixtures`` never reach the planner."""
    return RobotConfig.model_validate({
        "vendor": "ur",
        "motion": {"planner": planner,
                   "workspace": {"x_min": -500.0, "x_max": 500.0, "y_min": -500.0, "y_max": 500.0,
                                 "z_min": 0.0, "z_max": 600.0}},
        "gripper": {"model": "robotiq_2f85"},
        "safety": {
            "self_collision": {
                "planner_margin_mm": planner_margin_mm,
                # A cell that declares a fixture keeps 10 mm from it, as robot.ur5e.yaml does, and 10 from what the
                # camera saw: 4 over the 6 mm the camera world grows its boxes by.
                "min_distance_mm": 10.0 if fixtures else 3.0,
                "perceived_min_distance_mm": 4.0 if fixtures else 2.0,
                "fixtures": [
                    {"name": f"wall{i}", "center_mm": [400.0, -300.0 + 40.0 * i, 75.0],
                     "half_extents_mm": [10.0, 10.0, 75.0]}
                    for i in range(fixtures)
                ],
            },
            "planning_world": {
                "support_plane": ({"height_mm": 0.0, "extent_mm": [1600.0, 1600.0], "thickness_mm": 50.0}
                                  if bench else None),
                "meshes": [{"name": f"tote{i}", "path": f"assets/tote{i}.obj"} for i in range(meshes)],
            },
        },
    })


class _NoPlannerEnvironment(unittest.TestCase):
    """Every test starts with none of the three reservation variables set, whatever the shell has."""

    def setUp(self) -> None:
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        for name in _ENV:
            os.environ.pop(name, None)


class TheFieldCentreTests(unittest.TestCase):
    def test_the_field_centre_is_the_centre_of_the_grid_it_is_written_into(self) -> None:
        """The field is indexed from the low corner of the ROUNDED grid, so its pose is that grid's centre.

        It was taken from the unrounded span, 5 mm off on this cell (990 mm of grid over a 1000 mm
        span). Measured on the box on 2026-09-13: the planner puts a voxel centre half a voxel inside
        the low corner of the pose, so an off centre pose moves every obstacle by the difference.
        """
        field = build_voxel_field((), limits=_LIMITS, tuning=WorldBuildTuning(voxel_field_mm=30.0))
        assert field is not None

        self.assertEqual(field.dims_mm, (990.0, 990.0, 600.0))
        self.assertAlmostEqual(field.center_mm[0], -500.0 + 990.0 / 2.0)
        self.assertAlmostEqual(field.center_mm[1], -500.0 + 990.0 / 2.0)
        self.assertAlmostEqual(field.center_mm[2], 300.0)


class TheReservationTests(_NoPlannerEnvironment):
    def test_the_reservation_describes_the_field_a_refresh_builds(self) -> None:
        from src.robot.safety.planning.reservation import PlannerReservation

        reservation = PlannerReservation.from_config(robot_cfg=_cell())
        field = build_voxel_field((), limits=_LIMITS, tuning=WorldBuildTuning(voxel_field_mm=CAMERA_FIELD_VOXEL_MM),
                                  near=(FineRegion.about((0.0, 0.0, 0.0), FINE_REACH_MM), ()))
        assert field is not None and field.near is not None

        own, near = reservation.voxel_grid.split(";")
        self.assertEqual(own, ",".join(f"{v / 1000.0:.4f}" for v in (*field.dims_mm, field.voxel_size_mm)))
        self.assertEqual((own, near), (_GRID, _NEAR))
        # The fine grid about the goal stands as high as what it holds: as wide as the one reserved, never higher.
        x, y, z, voxel = (float(v) * 1000.0 for v in near.split(","))
        self.assertEqual((x, y, voxel), (*field.near.dims_mm[:2], field.near.voxel_size_mm))
        self.assertLessEqual(field.near.dims_mm[2], z)

    def test_the_slots_are_derived_from_what_the_cell_declares(self) -> None:
        from src.robot.safety.planning.reservation import PlannerReservation

        reservation = PlannerReservation.from_config(robot_cfg=_cell(fixtures=15, meshes=1))

        # The bench, plus the 128 boxes of what the cameras saw every cell's world holds; the fifteen walls are the
        # guard's, and no planner box since 2026-10-17.
        self.assertEqual(
            reservation,
            PlannerReservation(cuboid_slots=1 + CAMERA_MAX_BOXES, mesh_slots=1, voxel_grid=f"{_GRID};{_NEAR}"),
        )

    def test_every_live_scene_holds_the_camera_worlds_boxes_and_its_field(self) -> None:
        from src.robot.safety.planning.reservation import PlannerReservation

        reservation = PlannerReservation.from_config(robot_cfg=_cell(fixtures=2))

        self.assertEqual(reservation.cuboid_slots, 1 + CAMERA_MAX_BOXES)
        self.assertEqual(reservation.voxel_grid, f"{_GRID};{_NEAR}")

    def test_a_cell_with_no_bench_reserves_nothing_beyond_the_default(self) -> None:
        """The byte-identical path: the shipped base profile declares no bench, so its planning world is off."""
        from src.robot.safety.planning.reservation import PlannerReservation

        reservation = PlannerReservation.from_config(robot_cfg=_cell(fixtures=3, meshes=1, bench=False))

        self.assertEqual(reservation, PlannerReservation(cuboid_slots=16, mesh_slots=0, voxel_grid=""))


class TheClientsCarryTheReservationTests(_NoPlannerEnvironment):
    def test_the_planned_arms_default_factory_carries_the_reservation(self) -> None:
        from tests._arm_kit import planned_arm

        # Declares its planner margin, as every cuRobo cell must since B1 S17: the factory refuses without it.
        client = planned_arm(_cell(fixtures=15, meshes=1, planner_margin_mm=4.0))._default_curobo_client_factory()()

        self.assertEqual((client._scene, client._mesh_cache, client._voxel_grid),  # noqa: SLF001
                         (str(1 + CAMERA_MAX_BOXES), 1, f"{_GRID};{_NEAR}"))

    def test_a_shipped_base_profile_client_is_unchanged(self) -> None:
        """The control, green before and after: no planning world, the sidecar it always started."""
        from tests._arm_kit import planned_arm

        config = RobotConfig.model_validate({"vendor": "ur",
                                            "safety": {"self_collision": {"planner_margin_mm": 4.0}},
                                            "gripper": {"model": "robotiq_2f85"}})
        client = planned_arm(config)._default_curobo_client_factory()()

        self.assertEqual((client._scene, client._mesh_cache, client._voxel_grid), ("16", 0, ""))  # noqa: SLF001

    def test_the_sim_client_carries_the_reservation(self) -> None:
        """A non-UR double: the sim arm builds its own client, from its own driver config."""
        from src.robot.drivers.sim.arm import IsaacRobotArm
        from src.robot.drivers.sim.config import SimRobotConfig
        from src.robot.safety.planning.reservation import PlannerReservation

        reservation = PlannerReservation(cuboid_slots=24, mesh_slots=1, voxel_grid=_GRID)
        from src.willy_sim.config import load_sim_config, sim_safety_preflight

        # With the sim cell's preflight, which names its hand: a descriptor is named by the arm and the hand, and a
        # client whose descriptor says nothing of its hand is refused (Step 4i).
        arm = IsaacRobotArm(
            SimRobotConfig(
                enabled=True, scene="placeholder.usd", robot_prim_path="/World/Robot",
                planner_reservation=reservation,
            ),
            safety_preflight=sim_safety_preflight(load_sim_config()),
        )

        def started(client: CuroboPlanClient) -> None:
            from tests._sidecar_identity import arm_identity

            client.identity = arm_identity()

        with mock.patch.object(CuroboPlanClient, "start", autospec=True, side_effect=started):
            client = arm._get_curobo_client()  # noqa: SLF001

        self.assertEqual((client._scene, client._mesh_cache, client._voxel_grid), ("24", 1, _GRID))  # noqa: SLF001

    def test_sim_driver_config_fills_the_reservation_from_the_whole_robot(self) -> None:
        from src.robot.safety.planning.reservation import PlannerReservation
        from src.willy_sim.config import sim_driver_config

        cfg = _cell(fixtures=15, meshes=1)
        driver = sim_driver_config(SimpleNamespace(robot=cfg))  # type: ignore[arg-type]

        self.assertEqual(driver.planner_reservation, PlannerReservation.from_config(robot_cfg=cfg))


class TheConfigWinsTests(_NoPlannerEnvironment):
    def test_a_disagreeing_environment_variable_is_named_and_ignored(self) -> None:
        from src.robot.safety.planning.reservation import PlannerReservation

        os.environ["WILLY_CUROBO_CUBOID_CACHE"] = "4"
        client = CuroboPlanClient()

        with self.assertLogs("CuroboPlanClient", level="WARNING") as logs:
            client.reserve_world(PlannerReservation(cuboid_slots=24, mesh_slots=0, voxel_grid=""))

        self.assertEqual(client._scene, "24")  # noqa: SLF001
        self.assertTrue(
            any("WILLY_CUROBO_CUBOID_CACHE" in line and "ignored" in line for line in logs.output), logs.output
        )

    def test_the_sidecar_environment_says_what_the_reservation_says(self) -> None:
        """The sidecar reads the mesh and grid variables itself, so an inherited one would still win."""
        from src.robot.safety.planning.reservation import PlannerReservation

        os.environ["WILLY_CUROBO_MESH_CACHE"] = "4"
        os.environ["WILLY_CUROBO_VOXEL_GRID"] = "1,1,1,0.05"
        reserved = CuroboPlanClient()
        with self.assertLogs("CuroboPlanClient", level="WARNING"):
            reserved.reserve_world(PlannerReservation(cuboid_slots=16, mesh_slots=0, voxel_grid=""))

        env = reserved._sidecar_env()  # noqa: SLF001
        self.assertEqual(env["WILLY_CUROBO_MESH_CACHE"], "0")
        self.assertNotIn("WILLY_CUROBO_VOXEL_GRID", env)

        # The control: a client nobody reserved for still inherits the shell, as it always did.
        self.assertEqual(CuroboPlanClient()._sidecar_env()["WILLY_CUROBO_MESH_CACHE"], "4")  # noqa: SLF001

    def test_a_margin_the_caller_did_not_ask_for_is_not_inherited(self) -> None:
        """The same hole as the grid, in the variable that decides how much room the planner keeps.

        Nothing in this repo sets it: the client writes it and the sidecar reads it. So a value in the shell can only
        be a leftover, and inheriting one gives a cell a margin its config never asked for. The client's own number is
        the whole truth about it.
        """
        os.environ["WILLY_CUROBO_SELF_COLLISION_MARGIN_MM"] = "99.0"

        env = CuroboPlanClient()._sidecar_env()  # noqa: SLF001
        self.assertNotIn("WILLY_CUROBO_SELF_COLLISION_MARGIN_MM", env)

        # The control: what the caller DID ask for is written, whatever the shell holds.
        asked = CuroboPlanClient(self_collision_margin_mm=10.0)._sidecar_env()  # noqa: SLF001
        self.assertEqual(asked["WILLY_CUROBO_SELF_COLLISION_MARGIN_MM"], repr(10.0))


class ThePreflightPrintsTheReservationTests(_NoPlannerEnvironment):
    def test_a_curobo_cell_is_told_what_its_planner_will_allocate(self) -> None:
        """Owner, Step 4 Q4: no VRAM budget yet, but the grid is printed so the number is visible."""
        from src.robot.execution.real_cell.preflight import CheckStatus, run_config_preflight

        report = run_config_preflight(_cell(fixtures=1, meshes=1), curobo_available=True)

        rows = [c for c in report.checks if c.name == "planner reservation"]
        self.assertEqual(len(rows), 1, [c.name for c in report.checks])
        self.assertIs(rows[0].status, CheckStatus.OK)
        # Two layers, each as large as the larger grid: the fine one about the goal, 135 x 135 x 150.
        self.assertIn("5467500 cells", rows[0].detail)
        self.assertIn("1 mesh slot", rows[0].detail)

    def test_an_ik_cell_has_no_such_row(self) -> None:
        from src.robot.execution.real_cell.preflight import run_config_preflight

        report = run_config_preflight(_cell(planner="ik"), curobo_available=True)

        self.assertNotIn("planner reservation", [c.name for c in report.checks])


if __name__ == "__main__":
    unittest.main()
