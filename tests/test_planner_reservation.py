"""One reservation decides what the planner allocates when it starts, for every client that starts one.

cuRobo allocates its collision storage once, when the sidecar starts: so many box slots, so many mesh
slots, and a voxel grid of a fixed size. Before this, the UR arm's default client and the sim client both
reserved 16 boxes, no mesh and no grid whatever the cell declared, and the one function that computed a
grid had no caller. A cell that declared a tote as a mesh, or turned the live scene on, was refused at its
first motion by a planner that had nowhere to put either.

The slot counts are derived (owner, Step 4 Q3): as many boxes as the cell declares plus the perceived
boxes it allows, never fewer than 16, one mesh slot per declared mesh, and a grid only where a live scene
asks for one. The config wins over a hand set environment variable, and says so.
"""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from src.config.schema.robot import RobotConfig
from src.robot.safety.planning.curobo_client import CuroboPlanClient
from src.robot.safety.planning.perceived import (
    WorldBuildLimits,
    WorldBuildTuning,
    build_voxel_field,
)

_ENV = ("WILLY_CUROBO_CUBOID_CACHE", "WILLY_CUROBO_MESH_CACHE", "WILLY_CUROBO_VOXEL_GRID")
_LIMITS = WorldBuildLimits(
    x_mm=(-500.0, 500.0), y_mm=(-500.0, 500.0), z_mm=(0.0, 600.0), support_plane_top_mm=0.0
)
#: 1000 mm at 30 mm rounds to 33 voxels, 990 mm, on both horizontal axes; 600 mm is 20 voxels.
_GRID = "0.9900,0.9900,0.6000,0.0300"


def _cell(
    *,
    fixtures: int = 0,
    meshes: int = 0,
    voxel_mm: float = 0.0,
    max_boxes: int = 8,
    enabled: bool = True,
    payload: bool = False,
    planner: str = "curobo",
    planner_margin_mm: "float | None" = 4.0,
) -> RobotConfig:
    """A UR cell. It declares a planner margin by default, because since B1 S17 a cuRobo cell that declares none
    refuses to start a planner, and every case here is about the reservation rather than about the margin."""
    return RobotConfig.model_validate({
        "vendor": "ur",
        "ur": {"motion_planner": planner},
        "gripper": {"model": "robotiq_2f85"},
        "workspace_limits": {"x_min": -500.0, "x_max": 500.0, "y_min": -500.0, "y_max": 500.0,
                             "z_min": 0.0, "z_max": 600.0},
        "safety": {
            "payload": {"enforce": False},
            "self_collision": {
                "planner_margin_mm": planner_margin_mm,
                "fixtures": [
                    {"name": f"wall{i}", "center_mm": [400.0, -300.0 + 40.0 * i, 75.0],
                     "half_extents_mm": [10.0, 10.0, 75.0]}
                    for i in range(fixtures)
                ],
            },
            "planning_world": {
                "enabled": enabled,
                "support_plane": {"height_mm": 0.0, "extent_mm": [1600.0, 1600.0], "thickness_mm": 50.0},
                "perceived": {"voxel_field_mm": voxel_mm, "max_boxes": max_boxes},
                "meshes": [{"name": f"tote{i}", "path": f"assets/tote{i}.obj"} for i in range(meshes)],
                "payload": {"enabled": payload, "sphere_slots": 16},
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
        field = build_voxel_field(
            np.array([[0.0, 0.0, 50.0]]), limits=_LIMITS, tuning=WorldBuildTuning(voxel_field_mm=30.0)
        )
        assert field is not None

        self.assertEqual(field.dims_mm, (990.0, 990.0, 600.0))
        self.assertAlmostEqual(field.center_mm[0], -500.0 + 990.0 / 2.0)
        self.assertAlmostEqual(field.center_mm[1], -500.0 + 990.0 / 2.0)
        self.assertAlmostEqual(field.center_mm[2], 300.0)


class TheReservationTests(_NoPlannerEnvironment):
    def test_the_reservation_describes_the_field_a_refresh_builds(self) -> None:
        from src.robot.safety.planning.reservation import PlannerReservation

        reservation = PlannerReservation.from_config(robot_cfg=_cell(voxel_mm=30.0))
        field = build_voxel_field(
            np.array([[0.0, 0.0, 50.0]]), limits=_LIMITS, tuning=WorldBuildTuning(voxel_field_mm=30.0)
        )
        assert field is not None

        grid = ",".join(f"{v / 1000.0:.4f}" for v in (*field.dims_mm, field.voxel_size_mm))
        self.assertEqual(reservation.voxel_grid, grid)
        self.assertEqual(reservation.voxel_grid, _GRID)

    def test_the_slots_are_derived_from_what_the_cell_declares(self) -> None:
        from src.robot.safety.planning.reservation import PlannerReservation

        reservation = PlannerReservation.from_config(
            robot_cfg=_cell(fixtures=15, meshes=1, voxel_mm=30.0, payload=True)
        )

        # The bench and fifteen walls, plus the eight perceived boxes the cell allows.
        self.assertEqual(
            reservation, PlannerReservation(cuboid_slots=24, mesh_slots=1, voxel_grid=_GRID, sphere_slots=16)
        )

    def test_a_small_cell_keeps_the_sixteen_it_always_had(self) -> None:
        from src.robot.safety.planning.reservation import PlannerReservation

        reservation = PlannerReservation.from_config(robot_cfg=_cell(fixtures=2, max_boxes=4))

        self.assertEqual(reservation.cuboid_slots, 16)
        self.assertEqual(reservation.voxel_grid, "", "no field asked for, so no grid is reserved")

    def test_a_disabled_planning_world_reserves_nothing_beyond_the_default(self) -> None:
        """The byte-identical path: every shipped profile has this block disabled."""
        from src.robot.safety.planning.reservation import PlannerReservation

        reservation = PlannerReservation.from_config(
            robot_cfg=_cell(fixtures=3, meshes=1, voxel_mm=30.0, payload=True, enabled=False)
        )

        self.assertEqual(
            reservation, PlannerReservation(cuboid_slots=16, mesh_slots=0, voxel_grid="", sphere_slots=0)
        )


class TheClientsCarryTheReservationTests(_NoPlannerEnvironment):
    def test_the_ur_default_factory_carries_the_reservation(self) -> None:
        from src.robot.drivers.ur.arm import URRobotArm

        # Declares its planner margin, as every UR cuRobo cell must since B1 S17: the factory refuses without it.
        client = URRobotArm(_cell(fixtures=15, meshes=1, voxel_mm=30.0, planner_margin_mm=4.0))._default_curobo_client_factory()()

        self.assertEqual((client._scene, client._mesh_cache, client._voxel_grid), ("24", 1, _GRID))  # noqa: SLF001

    def test_a_shipped_base_profile_client_is_unchanged(self) -> None:
        """The control, green before and after: no planning world, the sidecar it always started."""
        from src.robot.drivers.ur.arm import URRobotArm

        config = RobotConfig.model_validate({"vendor": "ur",
                                            "safety": {"payload": {"enforce": False},
                                                       "self_collision": {"planner_margin_mm": 4.0}},
                                            "gripper": {"model": "robotiq_2f85"}})
        client = URRobotArm(config)._default_curobo_client_factory()()

        self.assertEqual((client._scene, client._mesh_cache, client._voxel_grid), ("16", 0, ""))  # noqa: SLF001

    def test_the_sim_client_carries_the_reservation(self) -> None:
        """A non-UR double: the sim arm builds its own client, from its own driver config."""
        from src.robot.drivers.sim.arm import IsaacRobotArm
        from src.robot.drivers.sim.config import SimRobotConfig
        from src.robot.safety.planning.reservation import PlannerReservation

        reservation = PlannerReservation(cuboid_slots=24, mesh_slots=1, voxel_grid=_GRID, sphere_slots=0)
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

        cfg = _cell(fixtures=15, meshes=1, voxel_mm=30.0)
        driver = sim_driver_config(SimpleNamespace(robot=cfg))  # type: ignore[arg-type]

        self.assertEqual(driver.planner_reservation, PlannerReservation.from_config(robot_cfg=cfg))


class TheConfigWinsTests(_NoPlannerEnvironment):
    def test_a_disagreeing_environment_variable_is_named_and_ignored(self) -> None:
        from src.robot.safety.planning.reservation import PlannerReservation

        os.environ["WILLY_CUROBO_CUBOID_CACHE"] = "4"
        client = CuroboPlanClient()

        with self.assertLogs("CuroboPlanClient", level="WARNING") as logs:
            client.reserve_world(PlannerReservation(cuboid_slots=24, mesh_slots=0, voxel_grid="", sphere_slots=0))

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
            reserved.reserve_world(PlannerReservation(cuboid_slots=16, mesh_slots=0, voxel_grid="", sphere_slots=0))

        env = reserved._sidecar_env()  # noqa: SLF001
        self.assertEqual(env["WILLY_CUROBO_MESH_CACHE"], "0")
        self.assertNotIn("WILLY_CUROBO_VOXEL_GRID", env)

        # The control: a client nobody reserved for still inherits the shell, as it always did.
        self.assertEqual(CuroboPlanClient()._sidecar_env()["WILLY_CUROBO_MESH_CACHE"], "4")  # noqa: SLF001

    def test_a_margin_and_a_payload_the_caller_did_not_ask_for_are_not_inherited(self) -> None:
        """The same hole as the grid, in the two variables that decide how much room the planner keeps.

        Nothing in this repo sets either variable: the client writes them and the sidecar reads them. So a value in
        the shell can only be a leftover, and inheriting one gives a cell a margin or a payload its config never
        asked for. The client's own numbers are the whole truth about them.
        """
        os.environ["WILLY_CUROBO_SELF_COLLISION_MARGIN_MM"] = "99.0"
        os.environ["WILLY_CUROBO_ATTACH_SPHERES"] = "7"

        env = CuroboPlanClient()._sidecar_env()  # noqa: SLF001
        self.assertNotIn("WILLY_CUROBO_SELF_COLLISION_MARGIN_MM", env)
        self.assertNotIn("WILLY_CUROBO_ATTACH_SPHERES", env)

        # The control: what the caller DID ask for is written, whatever the shell holds.
        asked = CuroboPlanClient(self_collision_margin_mm=10.0, attach_spheres=4)._sidecar_env()  # noqa: SLF001
        self.assertEqual(asked["WILLY_CUROBO_SELF_COLLISION_MARGIN_MM"], repr(10.0))
        self.assertEqual(asked["WILLY_CUROBO_ATTACH_SPHERES"], "4")


class ThePreflightPrintsTheReservationTests(_NoPlannerEnvironment):
    def test_a_curobo_cell_is_told_what_its_planner_will_allocate(self) -> None:
        """Owner, Step 4 Q4: no VRAM budget yet, but the grid is printed so the number is visible."""
        from src.robot.execution.real_cell.preflight import CheckStatus, run_config_preflight

        report = run_config_preflight(_cell(fixtures=1, meshes=1, voxel_mm=30.0), curobo_available=True)

        rows = [c for c in report.checks if c.name == "planner reservation"]
        self.assertEqual(len(rows), 1, [c.name for c in report.checks])
        self.assertIs(rows[0].status, CheckStatus.OK)
        self.assertIn("21780 cells", rows[0].detail)  # 33 x 33 x 20
        self.assertIn("1 mesh slot", rows[0].detail)

    def test_an_ik_cell_has_no_such_row(self) -> None:
        from src.robot.execution.real_cell.preflight import run_config_preflight

        report = run_config_preflight(_cell(voxel_mm=30.0, planner="ik"), curobo_available=True)

        self.assertNotIn("planner reservation", [c.name for c in report.checks])


if __name__ == "__main__":
    unittest.main()
