"""The cell's static geometry on its way to the trajectory planner.

What a cell declares to the planner is its bench and its meshes, wherever it declares the bench; the
guard's fixtures stay the guard's (2026-10-17). The silent failure this covers: a registration the
planner only partly confirms reads as success at every layer above, while the planner routes
straight through whatever it never received, and every cell refuses it.

The arms are ``PlannedArm`` over the fake transport since P0g slice 13 (``tests/_arm_kit.planned_arm``), and the planner
glue is built over the fake too: what reaches the planner is the planning half's, whichever vendor is below it.
"""

from __future__ import annotations

import pathlib
from typing import Any

import pytest

from src.config.schema.robot import (
    FixtureBoxConfig,
    PlanningWorldConfig,
    SupportPlaneConfig,
)
from src.robot.drivers.planned.curobo_motion import CuroboPlanner
from src.robot.safety.planning import CuroboUnavailableError
from src.robot.safety.planning.world import (
    build_planner_cuboids,
    describe_planner_world,
    planner_cuboid,
)

from src.config.arms import load_arm
from tests._arm_kit import planned_arm
from tests._fake_transport import fake_transport

#: The controller's joint order as the ur5e description states it; the planner is handed it since P0e.
_UR_JOINT_NAMES = tuple(joint.name for joint in load_arm("ur5e").joints)


def _bench(height_mm: float = 0.0, thickness_mm: float = 50.0) -> SupportPlaneConfig:
    return SupportPlaneConfig(
        height_mm=height_mm, extent_mm=(1600.0, 1600.0), thickness_mm=thickness_mm
    )


def _wall(name: str, y_mm: float) -> dict:
    """A wall a caller hands the planner itself, as the glue's ``set_world`` takes it."""
    return planner_cuboid(name, (400.0, y_mm, 75.0), (300.0, 16.0, 150.0))


def test_no_bench_registers_nothing() -> None:
    # Registering a world REPLACES the planner's own, boot table included. A cell that declares no
    # bench would therefore delete the only surface the planner knows about: it registers nothing.
    assert build_planner_cuboids(None) == []
    assert build_planner_cuboids(PlanningWorldConfig()) == []


def test_the_old_switch_is_refused_at_load_and_names_the_bench() -> None:
    """``enabled`` left on 2026-10-17: the world is on wherever the bench is declared."""
    from src.config.schema._removed import removed_key_sentence

    with pytest.raises(ValueError):
        PlanningWorldConfig.model_validate({"enabled": True})
    said = removed_key_sentence("robot.safety.planning_world.enabled")
    assert said is not None and "support_plane" in said


def test_the_support_height_is_the_TOP_surface() -> None:
    """The number an operator can measure is the surface, not the centre of a slab."""
    for thickness in (20.0, 50.0, 200.0):
        cuboids = build_planner_cuboids(PlanningWorldConfig(support_plane=_bench(-5.0, thickness)))
        (plane,) = cuboids
        centre_z_mm = 1000.0 * plane["pose"][2]
        top_mm = centre_z_mm + 1000.0 * plane["dims_m"][2] / 2.0
        assert top_mm == pytest.approx(-5.0), f"thickness {thickness} moved the surface"


def test_units_and_quaternion_order_flip_at_this_boundary() -> None:
    (plane,) = build_planner_cuboids(PlanningWorldConfig(support_plane=_bench(0.0, 50.0)))
    assert plane["dims_m"] == pytest.approx([1.6, 1.6, 0.05])
    # WXYZ identity, not the stack's XYZW.
    assert plane["pose"][3:] == [1.0, 0.0, 0.0, 0.0]


def test_the_bench_defaults_to_sitting_under_the_robot() -> None:
    """Every configuration written before `center_mm` existed meant this box."""
    (plane,) = build_planner_cuboids(PlanningWorldConfig(support_plane=_bench(0.0, 50.0)))
    assert plane["pose"][:2] == pytest.approx([0.0, 0.0])


def test_the_bench_can_sit_where_the_bench_is() -> None:
    """A robot at the head of its table has no floor under the half it works over."""
    plane_cfg = SupportPlaneConfig(
        height_mm=0.0, extent_mm=(600.0, 800.0), thickness_mm=50.0, center_mm=(525.0, 0.0)
    )
    (plane,) = build_planner_cuboids(PlanningWorldConfig(support_plane=plane_cfg))
    assert 1000.0 * plane["pose"][0] == pytest.approx(525.0)
    assert 1000.0 * plane["pose"][1] == pytest.approx(0.0)
    # The slab now spans x = 225 to 825, which is the bench, rather than -300 to 300, which is
    # mostly the robot's own column.
    half_x_mm = 1000.0 * plane["dims_m"][0] / 2.0
    assert 1000.0 * plane["pose"][0] - half_x_mm == pytest.approx(225.0)
    assert 1000.0 * plane["pose"][0] + half_x_mm == pytest.approx(825.0)


def test_the_bench_covers_every_fixture_standing_on_it() -> None:
    """The regression this exists for: a fixture with no floor under it.

    A cell that declares a bin at the far end of a bench and a slab centred on the robot gets a
    planner world where the bin floats: the planner will happily route a link through the space
    where the table is, because in its world there is no table there. The fixture is the guard's;
    the slab under it is the planner's.
    """
    bench = SupportPlaneConfig(
        height_mm=0.0, extent_mm=(600.0, 800.0), thickness_mm=50.0, center_mm=(525.0, 0.0)
    )
    bin_wall = FixtureBoxConfig(
        name="bin_far", center_mm=(700.0, 0.0, 75.0), half_extents_mm=(100.0, 200.0, 75.0)
    )
    (plane,) = build_planner_cuboids(PlanningWorldConfig(support_plane=bench))

    for axis in (0, 1):
        plane_lo = plane["pose"][axis] - plane["dims_m"][axis] / 2.0
        plane_hi = plane["pose"][axis] + plane["dims_m"][axis] / 2.0
        fixture_lo = (bin_wall.center_mm[axis] - bin_wall.half_extents_mm[axis]) / 1000.0
        fixture_hi = (bin_wall.center_mm[axis] + bin_wall.half_extents_mm[axis]) / 1000.0
        assert plane_lo <= fixture_lo and fixture_hi <= plane_hi, (
            f"axis {axis}: the fixture stands off the declared bench"
        )


def test_the_description_is_millimetres_for_a_human() -> None:
    cfg = PlanningWorldConfig(support_plane=_bench(-5.0))
    text = describe_planner_world([*build_planner_cuboids(cfg), _wall("left", -180.0)])
    assert "2 planner obstacle(s)" in text
    assert "support_plane" in text and "left" in text
    assert "1600.0" in text  # metres would read 1.6
    assert "no planning world" in describe_planner_world([])


# --------------------------------------------------------------------------------------------
# The planner side: registered once, and a partial registration stops the cell.
# --------------------------------------------------------------------------------------------


class _FakeClient:
    """A cuRobo client that confirms only ``confirm`` of the cuboids it is sent."""

    def __init__(self, traj, *, confirm: int | None = None) -> None:
        self.joint_names = list(_UR_JOINT_NAMES)
        self.dt = 0.0
        self._traj = traj
        self._confirm = confirm
        self.worlds: list[list[dict]] = []
        self.started = False
        # What a sidecar on this arm's descriptor with the hand added says of itself (Step 4i, UM lane S12): a planner
        # refuses one that says nothing.
        from tests._sidecar_identity import arm_identity

        self.identity = arm_identity()

    def start(self) -> None:
        self.started = True

    def set_world(self, cuboids):
        self.worlds.append(list(cuboids))
        return len(cuboids) if self._confirm is None else self._confirm

    def plan_joint(self, start, goal):
        return self._traj

    def close(self) -> None:
        pass


#: A joint goal a move asks the planner for: what starts the client and registers the world before the plan.
_GOAL = [0.1] * 6


def _zero() -> Any:
    """The fake transport standing at zero, as the hand-written controller of this file did."""
    return fake_transport(joints=[0.0] * 6)


def _cuboids(n: int = 2) -> list[dict]:
    """The declared bench and ``n - 1`` walls, the world a planner glue is handed."""
    cfg = PlanningWorldConfig(support_plane=_bench())
    return [*build_planner_cuboids(cfg), *(_wall(f"w{i}", 100.0 * i) for i in range(n - 1))]


def test_the_world_is_registered_once_not_per_move() -> None:
    client = _FakeClient([[0.0] * 6, [0.1] * 6])
    planner = CuroboPlanner(
        _zero(), client_factory=lambda: client, world_cuboids=_cuboids(), joint_names=_UR_JOINT_NAMES, controller_yaw_deg=180
    )
    planner.plan_joint(_GOAL)
    planner.plan_joint(_GOAL)
    assert len(client.worlds) == 1, "re-sending the world every move pays the cost for nothing"
    assert [c["name"] for c in client.worlds[0]] == ["support_plane", "w0"]


def test_no_declared_world_sends_nothing() -> None:
    """Byte-identical: the planner keeps the world it booted with."""
    client = _FakeClient([[0.0] * 6])
    CuroboPlanner(_zero(), joint_names=_UR_JOINT_NAMES, controller_yaw_deg=180,
                  client_factory=lambda: client).plan_joint(_GOAL)
    assert client.worlds == []


def test_a_partial_registration_refuses_to_plan() -> None:
    client = _FakeClient([[0.0] * 6], confirm=1)
    planner = CuroboPlanner(
        _zero(), client_factory=lambda: client, world_cuboids=_cuboids(3), joint_names=_UR_JOINT_NAMES, controller_yaw_deg=180
    )
    with pytest.raises(CuroboUnavailableError, match="confirmed 1 of 3"):
        planner.plan_joint(_GOAL)


def test_a_partial_registration_refuses_the_SECOND_move_too() -> None:
    """The defect this pins: a refusal that fired once and then waved everything through.

    The latch was set before the refusal was raised, so move one refused, and move two found the
    client already built, skipped registration entirely, and planned against exactly the partial
    world the refusal had been about. A guard that only ever fires once is worse than either answer.
    """
    client = _FakeClient([[0.0] * 6, [0.1] * 6], confirm=1)
    planner = CuroboPlanner(
        _zero(), client_factory=lambda: client, world_cuboids=_cuboids(3), joint_names=_UR_JOINT_NAMES, controller_yaw_deg=180
    )
    for attempt in (1, 2):
        with pytest.raises(CuroboUnavailableError, match="confirmed 1 of 3"):
            planner.plan_joint(_GOAL)
        assert len(client.worlds) == attempt, "each refused move must try to register again"


def test_the_driver_actually_hands_its_config_down_to_the_planner() -> None:
    """The wiring, not the pieces. A block nothing reads is the failure mode this guards.

    `robot.grasping.calculator` sat inert for weeks because every call site named the analytic class
    directly, and the flag guard could not see it. Same shape here: assert the declared bench arrives,
    and the guard's own fixture does not (2026-10-17: a cell's fixtures stay the guard's).
    """
    from src.config.schema.robot import RobotConfig

    config = RobotConfig.model_validate({
        "vendor": "ur",
        "gripper": {"model": "robotiq_2f85"},
        "safety": {
            "self_collision": {
                # The margin every cuRobo cell must declare (B1 S17). A stub client skips the factory that refuses an undeclared one,
                # so this used to pass without it; the evidence check reads the margin to find its file (S22), and meets it here.
                "planner_margin_mm": 4.0,
                # A cell that declares a fixture keeps 10 mm from it, as robot.ur5e.yaml does, and 10 from what the
                # camera saw: 4 over the 6 mm every camera world grows its boxes by.
                "min_distance_mm": 10.0,
                "perceived_min_distance_mm": 4.0,
                "fixtures": [{
                    "name": "bin_wall_left",
                    "center_mm": [400.0, -180.0, 75.0],
                    "half_extents_mm": [150.0, 8.0, 75.0],
                }],
            },
            "planning_world": {
                "support_plane": {"height_mm": -5.0, "extent_mm": [1600.0, 1600.0],
                                  "thickness_mm": 50.0},
            },
        },
    })
    client = _FakeClient([[0.0] * 6, [0.1] * 6])
    arm = planned_arm(config, transport=_zero())
    arm._curobo_client_factory = lambda: client  # type: ignore[assignment]
    arm._curobo_planner().plan_joint(_GOAL)

    assert [c["name"] for c in client.worlds[0]] == ["support_plane"]

    # And with no bench declared, nothing is registered: the byte-identical path.
    off = RobotConfig.model_validate({"vendor": "ur", "safety": {"self_collision": {"planner_margin_mm": 4.0}},
                                      "gripper": {"model": "robotiq_2f85"}})
    quiet = _FakeClient([[0.0] * 6])
    arm_off = planned_arm(off, transport=_zero())
    arm_off._curobo_client_factory = lambda: quiet  # type: ignore[assignment]
    arm_off._curobo_planner().plan_joint(_GOAL)
    assert quiet.worlds == []


class _ReservingClient(_FakeClient):
    """Records whether it was told its reservation before it started, and the meshes it is sent."""

    def __init__(self, traj) -> None:
        super().__init__(traj)
        self.events: list[str] = []
        self.meshes: list[list[dict]] = []
        self.reservation = None

    def reserve_world(self, reservation) -> None:
        self.events.append("reserve")
        self.reservation = reservation

    def start(self) -> None:
        self.events.append("start")
        super().start()

    def set_world(self, cuboids, meshes=None):
        self.meshes.append(list(meshes or ()))
        return super().set_world(cuboids) + len(meshes or ())


def test_an_injected_factory_still_gets_the_reservation() -> None:
    """Every test injects its own client factory, so the reservation cannot live only in the default one."""
    from src.robot.safety.planning.reservation import PlannerReservation

    reservation = PlannerReservation(cuboid_slots=24, mesh_slots=1, voxel_grid="")
    client = _ReservingClient([[0.0] * 6, [0.1] * 6])

    CuroboPlanner(_zero(), joint_names=_UR_JOINT_NAMES, controller_yaw_deg=180,
                  client_factory=lambda: client, reservation=reservation).plan_joint(_GOAL)

    assert client.events[:2] == ["reserve", "start"], client.events
    assert client.reservation == reservation


def test_a_declared_mesh_is_registered_without_a_live_world() -> None:
    """A tote declared as a mesh reached the planner only through the live world, never on its own."""
    from src.config.schema.robot import RobotConfig

    config = RobotConfig.model_validate({
        "vendor": "ur",
        "gripper": {"model": "robotiq_2f85"},
        "safety": {
            # The margin every cuRobo cell must declare (B1 S17). A stub client skips the factory that refuses an undeclared one,
            # so this used to pass without it; the evidence check reads the margin to find its file (S22), and meets it here.
            "self_collision": {"planner_margin_mm": 4.0},
            "planning_world": {
                "support_plane": {"height_mm": -5.0, "extent_mm": [1600.0, 1600.0],
                                  "thickness_mm": 50.0},
                "meshes": [{"name": "tote", "path": "assets/tote.obj", "center_mm": [450.0, 0.0, 0.0]}],
            },
        },
    })
    client = _ReservingClient([[0.0] * 6, [0.1] * 6])
    arm = planned_arm(config, transport=_zero())
    arm._curobo_client_factory = lambda: client  # type: ignore[assignment]

    arm._curobo_planner().plan_joint(_GOAL)

    assert [m["name"] for m in client.meshes[0]] == ["tote"], "the declared mesh never reached the planner"
    assert [c["name"] for c in client.worlds[0]] == ["support_plane"]


def test_the_preflight_reads_the_fixture_list_where_it_actually_lives() -> None:
    """It read `robot.fixtures`, which is not a schema key, so it reported "none declared" always.

    A warning that fires on every cell, correct or not, is a warning nobody reads.
    """
    from src.config.schema.robot import RobotConfig
    from src.robot.execution.real_cell.preflight import run_config_preflight

    declared = RobotConfig.model_validate({
        "vendor": "ur",
        "safety": {
            # A cell that declares a fixture keeps 10 mm from it, as robot.ur5e.yaml does.
            "self_collision": {"min_distance_mm": 10.0, "perceived_min_distance_mm": 4.0, "fixtures": [{
                "name": "bench_leg",
                "center_mm": [0.0, 0.0, 0.0],
                "half_extents_mm": [50.0, 50.0, 300.0],
            }]},
            "planning_world": {
                "support_plane": {"height_mm": 0.0, "extent_mm": [1600.0, 1600.0],
                                  "thickness_mm": 50.0},
            },
        },
    })
    rows = {c.name: c for c in run_config_preflight(declared).checks}
    assert "bench_leg" in rows["fixtures"].detail
    assert "support_plane" in rows["planning world"].detail
    assert "bench_leg" not in rows["planning world"].detail, "a fixture is the guard's, not the planner's"

    bare = RobotConfig.model_validate({"vendor": "ur"})
    bare_rows = {c.name: c for c in run_config_preflight(bare).checks}
    assert "none declared" in bare_rows["fixtures"].detail
    assert "no bench declared" in bare_rows["planning world"].detail


# --------------------------------------------------------------------------------------------
# The PATH, not only where it ends.
# --------------------------------------------------------------------------------------------


def _preflight(*, fixtures=()):
    from src.config.schema.robot import RobotSafetyConfig, WorkspaceLimitsConfig
    from src.robot.safety import SafetyPreflight

    safety = RobotSafetyConfig.model_validate({
        # A cell that declares a fixture keeps 10 mm from it, as robot.ur5e.yaml does.
        "self_collision": {"min_distance_mm": 10.0, "perceived_min_distance_mm": 4.0, "fixtures": [
            {"name": f.name, "center_mm": list(f.center_mm),
             "half_extents_mm": list(f.half_extents_mm)} for f in fixtures
        ]},
    })
    # Built for the ur5e the doubles here report, with its hand named: a pipeline judges only the arm it was built
    # for since P0e, and a guard that reads hand geometry on a UR refuses to build without the hand.
    from src.config.schema.robot import RobotConfig
    from src.robot.safety.planning.hand import planner_hand

    hand = planner_hand(RobotConfig.model_validate({"gripper": {"model": "robotiq_2f85"}}))
    return SafetyPreflight.from_safety_config(safety, WorkspaceLimitsConfig(), hand=hand, arm_model="ur5e")


def _samples(*configs):
    """The configurations as they are, so these tests keep asking about the guards, not the sampler.

    `gate_planned_path` derives its own samples from the geometry; `tests/robot/safety/test_path_samples.py` holds
    that half. Here a path IS its configurations, which is what these tests were always about.
    """
    from src.robot.safety import PathSamples

    return PathSamples(configs=tuple(tuple(c) for c in configs), step_bound_mm=10.0)


class _Arm:
    from src.config.arms import described_home
    from src.robot.core.capabilities import RobotCapabilities

    capabilities = RobotCapabilities(vendor="ur", model="ur5e", has_native_fk=True, dof=6, controller=None, owns_a_controller=True)
    #: The home it drives to, its description's, which the joint-limit guard centres every real arm's window on.
    home_joint_positions = described_home("ur5e")


#: A configuration that folds a finger into the forearm. The exact-mesh backend refuses it with
#: "forearm|lfinger: mesh distance 0.367 mm"; the capsule proxy cannot see the gripper at all here.
_FOLDED = [1.95, 0.38, -1.33, -0.55, 2.00, 0.79]
_CLEAR = [0.0, -1.5, 1.5, 0.0, 0.0, 0.0]


def test_there_is_no_longer_a_switch_that_waves_a_path_through() -> None:
    """The old gate had one, and every profile this repository shipped set it to off."""
    from src.robot.safety import SafetyPreflight

    assert not hasattr(SafetyPreflight, "gate_trajectory")
    assert not hasattr(SafetyPreflight, "checks_trajectories")


def _mesh_backend_available() -> bool:
    """The folded configuration is only refused by the EXACT-MESH backend; the capsule proxy cannot
    see the gripper on a joint-only context at all, which is why the shipped default is `fcl`."""
    from src.robot.safety._fcl_self_collision import mesh_backend_status

    return mesh_backend_status("ur5e") == "ok"


def test_a_colliding_waypoint_in_the_MIDDLE_is_caught() -> None:
    """The whole point. The endpoint gate cannot see this path: it starts and ends clear."""
    if not _mesh_backend_available():
        pytest.skip("no exact-mesh backend on this box")
    preflight = _preflight()
    rejected = preflight.gate_joint_path(_samples(_CLEAR, _FOLDED, _CLEAR), arm=_Arm())
    assert rejected is not None, "a path through a self-collision was accepted"
    assert "lfinger" in (rejected.message or "") or "collision" in (rejected.message or "").lower()


def test_a_clear_path_is_accepted() -> None:
    if not _mesh_backend_available():
        pytest.skip("no exact-mesh backend on this box")
    assert _preflight().gate_joint_path(_samples(_CLEAR, _CLEAR, _CLEAR), arm=_Arm()) is None


def test_an_empty_trajectory_is_not_an_error() -> None:
    from src.robot.safety import PathSamples

    assert _preflight().gate_joint_path(
        PathSamples(configs=(), step_bound_mm=0.0), arm=_Arm()
    ) is None


def test_every_configuration_is_visited_and_none_is_skipped() -> None:
    """The old gate could sample the path with a stride and leave the rest unexamined. This one
    cannot: the only configurations it does not look at are the ones nobody handed it."""
    if not _mesh_backend_available():
        pytest.skip("no exact-mesh backend on this box")
    for index in range(5):
        path = [_CLEAR] * 5
        path[index] = _FOLDED
        assert _preflight().gate_joint_path(_samples(*path), arm=_Arm()) is not None, index


def test_a_declared_fixture_is_checked_along_the_path_too() -> None:
    """The bench is a fixture, and the self-collision guard carries fixtures, so this closes the
    'do not drive into the table' half of the goal for every configuration and not just the last."""
    from src.config.schema.robot import FixtureBoxConfig

    # A box around the flange at _CLEAR (from ur_link_transforms_mm).
    post = FixtureBoxConfig(
        name="post", center_mm=(-422.3, -232.9, 486.7), half_extents_mm=(60.0, 60.0, 60.0)
    )
    preflight = _preflight(fixtures=(post,))
    assert preflight.gate_joint_path(_samples(_CLEAR), arm=_Arm()) is not None


# --------------------------------------------------------------------------------------------
# The sim path: a runner must not be able to delete the bench by declaring a wall.
# --------------------------------------------------------------------------------------------


class _WorldClient:
    def __init__(self, confirm: int | None = None) -> None:
        self.worlds: list[list[dict]] = []
        self._confirm = confirm

    def set_world(self, cuboids):
        self.worlds.append(list(cuboids))
        return len(cuboids) if self._confirm is None else self._confirm


def _sim_arm_with(client):
    from src.robot.drivers.sim.arm import IsaacRobotArm
    from src.robot.drivers.sim.config import SimRobotConfig

    arm = IsaacRobotArm(SimRobotConfig(enabled=True, mock_mode=True))
    arm._curobo_client = client  # type: ignore[assignment]
    return arm


def test_a_runner_registration_no_longer_deletes_the_declared_cell() -> None:
    """`set_world` REPLACES. Three sim runners register only a floor, which used to take the bench
    with it; the two that register bin walls took the boot table."""
    client = _WorldClient()
    arm = _sim_arm_with(client)
    arm.set_planner_base_world(_cuboids(2))  # bench + one wall from config
    arm.set_curobo_world([{"name": "bin_wall_far", "dims_m": [0.3, 0.02, 0.15],
                           "pose": [0.4, 0.2, 0.075, 1, 0, 0, 0]}])
    names = [c["name"] for c in client.worlds[0]]
    assert "support_plane" in names, "the declared bench survived the runner's registration"
    assert "bin_wall_far" in names, "and the runner's own obstacle arrived"


def test_the_base_world_wins_a_name_collision() -> None:
    """The cell's declaration is the one an operator wrote down; a runner cannot quietly move it."""
    client = _WorldClient()
    arm = _sim_arm_with(client)
    arm.set_planner_base_world(_cuboids(1))
    arm.set_curobo_world([{"name": "support_plane", "dims_m": [0.01, 0.01, 0.01],
                           "pose": [0, 0, -9.0, 1, 0, 0, 0]}])
    (plane,) = [c for c in client.worlds[0] if c["name"] == "support_plane"]
    assert plane["pose"][2] != -9.0, "the runner overwrote the declared bench"


def test_a_partial_registration_is_reported_not_swallowed(caplog) -> None:
    """The client returns 0 and logs 'planning continues against the PREVIOUS world'; three of the
    four sim runners never looked at the count."""
    import logging

    arm = _sim_arm_with(_WorldClient(confirm=1))
    arm.set_planner_base_world(_cuboids(2))
    with caplog.at_level(logging.ERROR):
        count = arm.set_curobo_world([])
    assert count == 1
    assert any("confirmed 1 of 2" in r.getMessage() for r in caplog.records)


def test_a_hand_is_composed_onto_the_arm_it_is_placed_on() -> None:
    """The EGU-50 on a ur3e is judged on the ur3e's arm, which is what the admission list used to stand in for.

    MEASURED before UM lane S05: the EGU-50's file held the ur5e arm links byte for byte, and the status token was
    `ok` for a ur3e cell, which would have checked UR5e arm geometry on UR3e DH frames with the guard reporting
    itself healthy. S05 cut the hands out of the arm files, and `hand__admitted_arms` refused every arm but the
    one a hand was cut from, as a second guard. S22 retired that list: which arms a hand may be composed onto is
    MEASURED now, one file per combination, and the matrix measured the EGU-50 on the ur3e at b1. So the property
    is held directly: the arm links the guard composes are the ur3e's own, and the hand parts are the EGU-50's.
    """
    import numpy as np

    from src.robot.safety._fcl_self_collision import composed_parts, mesh_backend_status

    root = pathlib.Path("src/robot/safety/data")
    if not (root / "ur3e_collision_meshes.npz").is_file():
        pytest.skip("no ur3e bundle on this box")
    assert mesh_backend_status("ur3e", None, "schunk_egu50") in {"ok", "no_engine"}
    parts = composed_parts("ur3e", None, "schunk_egu50")
    with np.load(root / "ur3e_collision_meshes.npz") as arm, np.load(root / "schunk_egu50_hand_meshes.npz") as hand:
        for link in ("shoulder", "upper_arm", "forearm", "wrist_1", "wrist_2", "wrist_3"):
            assert np.array_equal(parts[link][0], arm[f"{link}__v"]), f"{link} is not the ur3e's own"
        for part in ("gripper", "lfinger", "rfinger"):
            assert np.array_equal(parts[part][0], hand[f"{part}__v"]), f"{part} is not the EGU-50's"


def test_the_arms_that_test_tells_apart_really_do_differ() -> None:
    """If this ever fails, the test above is testing nothing: a ur5e forearm would pass for a ur3e one."""
    import numpy as np

    root = pathlib.Path("src/robot/safety/data")
    if not (root / "ur3e_collision_meshes.npz").is_file():
        pytest.skip("no ur3e bundle on this box")
    with np.load(root / "ur5e_collision_meshes.npz") as ur5e, np.load(root / "ur3e_collision_meshes.npz") as ur3e:
        assert not np.array_equal(ur5e["forearm__v"], ur3e["forearm__v"])
