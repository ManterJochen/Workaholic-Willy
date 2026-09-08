"""The cell's static geometry on its way to the trajectory planner.

Two silent failures are what this covers. A duplicate NAME collapses two obstacles into one, because
the planner keys its world by name. And a registration the planner only partly confirms reads as
success at every layer above, while the planner routes straight through whatever it never received.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from src.config.schema.robot import (
    FixtureBoxConfig,
    PlanningWorldConfig,
    SupportPlaneConfig,
)
from src.geometry import Frame, Pose
from src.robot.drivers.ur.curobo_motion import UR_ARM_JOINT_NAMES, CuroboUrPlanner
from src.robot.safety.planning import CuroboUnavailableError
from src.robot.safety.planning.world import (
    DEFAULT_MAX_CUBOIDS,
    PlanningWorldError,
    build_planner_cuboids,
    describe_planner_world,
)


def _bench(height_mm: float = 0.0, thickness_mm: float = 50.0) -> SupportPlaneConfig:
    return SupportPlaneConfig(
        height_mm=height_mm, extent_mm=(1600.0, 1600.0), thickness_mm=thickness_mm
    )


def _wall(name: str, y_mm: float) -> FixtureBoxConfig:
    return FixtureBoxConfig(
        name=name, center_mm=(400.0, y_mm, 75.0), half_extents_mm=(150.0, 8.0, 75.0)
    )


def test_absent_or_disabled_registers_nothing() -> None:
    assert build_planner_cuboids(None) == []
    assert build_planner_cuboids(PlanningWorldConfig()) == []
    assert build_planner_cuboids(PlanningWorldConfig(), [_wall("w", 0.0)]) == []


def test_enabling_without_a_bench_is_refused_at_load() -> None:
    # Registering a world REPLACES the planner's own, boot table included. Enabling this block
    # without declaring a bench would therefore delete the only surface the planner knows about.
    with pytest.raises(ValueError, match="support_plane"):
        PlanningWorldConfig(enabled=True)


def test_the_support_height_is_the_TOP_surface() -> None:
    """The number an operator can measure is the surface, not the centre of a slab."""
    for thickness in (20.0, 50.0, 200.0):
        cuboids = build_planner_cuboids(
            PlanningWorldConfig(enabled=True, support_plane=_bench(-5.0, thickness))
        )
        (plane,) = cuboids
        centre_z_mm = 1000.0 * plane["pose"][2]
        top_mm = centre_z_mm + 1000.0 * plane["dims_m"][2] / 2.0
        assert top_mm == pytest.approx(-5.0), f"thickness {thickness} moved the surface"


def test_units_and_quaternion_order_flip_at_this_boundary() -> None:
    (plane,) = build_planner_cuboids(
        PlanningWorldConfig(enabled=True, support_plane=_bench(0.0, 50.0))
    )
    assert plane["dims_m"] == pytest.approx([1.6, 1.6, 0.05])
    # WXYZ identity, not the stack's XYZW.
    assert plane["pose"][3:] == [1.0, 0.0, 0.0, 0.0]


def test_the_bench_defaults_to_sitting_under_the_robot() -> None:
    """Every configuration written before `center_mm` existed meant this box."""
    (plane,) = build_planner_cuboids(
        PlanningWorldConfig(enabled=True, support_plane=_bench(0.0, 50.0))
    )
    assert plane["pose"][:2] == pytest.approx([0.0, 0.0])


def test_the_bench_can_sit_where_the_bench_is() -> None:
    """A robot at the head of its table has no floor under the half it works over."""
    plane_cfg = SupportPlaneConfig(
        height_mm=0.0, extent_mm=(600.0, 800.0), thickness_mm=50.0, center_mm=(525.0, 0.0)
    )
    (plane,) = build_planner_cuboids(PlanningWorldConfig(enabled=True, support_plane=plane_cfg))
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
    where the table is, because in its world there is no table there.
    """
    bench = SupportPlaneConfig(
        height_mm=0.0, extent_mm=(600.0, 800.0), thickness_mm=50.0, center_mm=(525.0, 0.0)
    )
    bin_wall = FixtureBoxConfig(
        name="bin_far", center_mm=(700.0, 0.0, 75.0), half_extents_mm=(100.0, 200.0, 75.0)
    )
    plane, fixture = build_planner_cuboids(
        PlanningWorldConfig(enabled=True, support_plane=bench), [bin_wall]
    )

    for axis in (0, 1):
        plane_lo = plane["pose"][axis] - plane["dims_m"][axis] / 2.0
        plane_hi = plane["pose"][axis] + plane["dims_m"][axis] / 2.0
        fixture_lo = fixture["pose"][axis] - fixture["dims_m"][axis] / 2.0
        fixture_hi = fixture["pose"][axis] + fixture["dims_m"][axis] / 2.0
        assert plane_lo <= fixture_lo and fixture_hi <= plane_hi, (
            f"axis {axis}: the fixture stands off the declared bench"
        )


def test_a_fixture_arrives_as_full_extents_at_its_centre() -> None:
    cfg = PlanningWorldConfig(enabled=True, support_plane=_bench())
    _, wall = build_planner_cuboids(cfg, [_wall("left", -180.0)])
    assert wall["name"] == "left"
    assert wall["dims_m"] == pytest.approx([0.300, 0.016, 0.150])
    assert wall["pose"][:3] == pytest.approx([0.400, -0.180, 0.075])


def test_include_fixtures_false_keeps_only_the_bench() -> None:
    cfg = PlanningWorldConfig(enabled=True, support_plane=_bench(), include_fixtures=False)
    assert [c["name"] for c in build_planner_cuboids(cfg, [_wall("left", -180.0)])] == [
        "support_plane"
    ]


def test_a_zero_extent_fixture_is_skipped() -> None:
    """The schema documents an all-zero fixture as the way to disable one without deleting it."""
    cfg = PlanningWorldConfig(enabled=True, support_plane=_bench())
    disabled = FixtureBoxConfig(name="off", half_extents_mm=(0.0, 0.0, 0.0))
    assert [c["name"] for c in build_planner_cuboids(cfg, [disabled])] == ["support_plane"]


def test_duplicate_names_are_refused_because_they_would_collapse() -> None:
    cfg = PlanningWorldConfig(enabled=True, support_plane=_bench())
    with pytest.raises(PlanningWorldError, match="more than once"):
        build_planner_cuboids(cfg, [_wall("wall", -180.0), _wall("wall", 180.0)])


def test_a_world_larger_than_the_reserved_slots_is_refused() -> None:
    cfg = PlanningWorldConfig(enabled=True, support_plane=_bench())
    many = [_wall(f"w{i}", float(i)) for i in range(DEFAULT_MAX_CUBOIDS)]
    with pytest.raises(PlanningWorldError, match="collision slots"):
        build_planner_cuboids(cfg, many)
    assert len(build_planner_cuboids(cfg, many[:-1])) == DEFAULT_MAX_CUBOIDS


def test_the_description_is_millimetres_for_a_human() -> None:
    cfg = PlanningWorldConfig(enabled=True, support_plane=_bench(-5.0))
    text = describe_planner_world(build_planner_cuboids(cfg, [_wall("left", -180.0)]))
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
        self.joint_names = list(UR_ARM_JOINT_NAMES)
        self.dt = 0.0
        self._traj = traj
        self._confirm = confirm
        self.worlds: list[list[dict]] = []
        self.started = False

    def start(self) -> None:
        self.started = True

    def set_world(self, cuboids):
        self.worlds.append(list(cuboids))
        return len(cuboids) if self._confirm is None else self._confirm

    def plan(self, start, pos_m, quat_wxyz):
        return self._traj

    def close(self) -> None:
        pass


class _FakeConn:
    def __init__(self) -> None:
        self.moves: list[list[float]] = []

    @property
    def is_connected(self) -> bool:
        return True

    def get_joint_positions(self):
        return [0.0] * 6

    def moveJ(self, joints, vel=None, acc=None):
        self.moves.append([float(v) for v in joints])
        return True


def _pose() -> Pose:
    return Pose(
        position_mm=np.array([400.0, 0.0, 300.0], dtype=np.float64),
        quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        frame=Frame.BASE,
        label="tcp",
    )


def _cuboids(n: int = 2) -> list[dict]:
    cfg = PlanningWorldConfig(enabled=True, support_plane=_bench())
    return build_planner_cuboids(cfg, [_wall(f"w{i}", 100.0 * i) for i in range(n - 1)])


def test_the_world_is_registered_once_not_per_move() -> None:
    client = _FakeClient([[0.0] * 6, [0.1] * 6])
    planner = CuroboUrPlanner(
        _FakeConn(), client_factory=lambda: client, world_cuboids=_cuboids()
    )
    planner.plan(_pose())
    planner.plan(_pose())
    assert len(client.worlds) == 1, "re-sending the world every move pays the cost for nothing"
    assert [c["name"] for c in client.worlds[0]] == ["support_plane", "w0"]


def test_no_declared_world_sends_nothing() -> None:
    """Byte-identical: the planner keeps the world it booted with."""
    client = _FakeClient([[0.0] * 6])
    CuroboUrPlanner(_FakeConn(), client_factory=lambda: client).plan(_pose())
    assert client.worlds == []


def test_a_partial_registration_refuses_to_plan() -> None:
    client = _FakeClient([[0.0] * 6], confirm=1)
    planner = CuroboUrPlanner(
        _FakeConn(), client_factory=lambda: client, world_cuboids=_cuboids(3)
    )
    with pytest.raises(CuroboUnavailableError, match="confirmed 1 of 3"):
        planner.plan(_pose())


def test_a_partial_registration_refuses_the_SECOND_move_too() -> None:
    """The defect this pins: a refusal that fired once and then waved everything through.

    The latch was set before the refusal was raised, so move one refused, and move two found the
    client already built, skipped registration entirely, and planned against exactly the partial
    world the refusal had been about. A guard that only ever fires once is worse than either answer.
    """
    client = _FakeClient([[0.0] * 6, [0.1] * 6], confirm=1)
    planner = CuroboUrPlanner(
        _FakeConn(), client_factory=lambda: client, world_cuboids=_cuboids(3)
    )
    for attempt in (1, 2):
        with pytest.raises(CuroboUnavailableError, match="confirmed 1 of 3"):
            planner.plan(_pose())
        assert len(client.worlds) == attempt, "each refused move must try to register again"


def test_a_partial_registration_can_be_accepted_deliberately() -> None:
    client = _FakeClient([[0.0] * 6, [0.1] * 6], confirm=1)
    planner = CuroboUrPlanner(
        _FakeConn(),
        client_factory=lambda: client,
        world_cuboids=_cuboids(3),
        require_registration=False,
    )
    assert planner.plan(_pose()) is not None


def test_the_driver_actually_hands_its_config_down_to_the_planner() -> None:
    """The wiring, not the pieces. A block nothing reads is the failure mode this guards.

    `robot.grasping.calculator` sat inert for weeks because every call site named the analytic class
    directly, and the flag guard could not see it. Same shape here: assert the declared boxes arrive.
    """
    from src.config.schema.robot import RobotConfig
    from src.robot.drivers.ur.arm import URRobotArm

    config = RobotConfig.model_validate({
        "vendor": "ur",
        "safety": {
            "payload": {"enforce": False},
            "self_collision": {
                "fixtures": [{
                    "name": "bin_wall_left",
                    "center_mm": [400.0, -180.0, 75.0],
                    "half_extents_mm": [150.0, 8.0, 75.0],
                }],
            },
            "planning_world": {
                "enabled": True,
                "support_plane": {"height_mm": -5.0, "extent_mm": [1600.0, 1600.0],
                                  "thickness_mm": 50.0},
            },
        },
    })
    client = _FakeClient([[0.0] * 6, [0.1] * 6])
    arm = URRobotArm(config)
    arm._conn = _FakeConn()  # type: ignore[assignment]
    arm._curobo_client_factory = lambda: client  # type: ignore[assignment]
    arm._curobo_ur_planner().plan(_pose())

    assert [c["name"] for c in client.worlds[0]] == ["support_plane", "bin_wall_left"]

    # And with the block off, nothing is registered: the byte-identical path.
    off = RobotConfig.model_validate({"vendor": "ur", "safety": {"payload": {"enforce": False}}})
    quiet = _FakeClient([[0.0] * 6])
    arm_off = URRobotArm(off)
    arm_off._conn = _FakeConn()  # type: ignore[assignment]
    arm_off._curobo_client_factory = lambda: quiet  # type: ignore[assignment]
    arm_off._curobo_ur_planner().plan(_pose())
    assert quiet.worlds == []


def test_the_preflight_reads_the_fixture_list_where_it_actually_lives() -> None:
    """It read `robot.fixtures`, which is not a schema key, so it reported "none declared" always.

    A warning that fires on every cell, correct or not, is a warning nobody reads.
    """
    from src.config.schema.robot import RobotConfig
    from src.robot.execution.real_cell.preflight import run_config_preflight

    declared = RobotConfig.model_validate({
        "vendor": "ur",
        "safety": {
            "self_collision": {"fixtures": [{
                "name": "bench_leg",
                "center_mm": [0.0, 0.0, 0.0],
                "half_extents_mm": [50.0, 50.0, 300.0],
            }]},
            "planning_world": {
                "enabled": True,
                "support_plane": {"height_mm": 0.0, "extent_mm": [1600.0, 1600.0],
                                  "thickness_mm": 50.0},
            },
        },
    })
    rows = {c.name: c for c in run_config_preflight(declared).checks}
    assert "bench_leg" in rows["fixtures"].detail
    assert "support_plane" in rows["planning world"].detail
    assert "bench_leg" in rows["planning world"].detail

    bare = RobotConfig.model_validate({"vendor": "ur"})
    bare_rows = {c.name: c for c in run_config_preflight(bare).checks}
    assert "none declared" in bare_rows["fixtures"].detail
    assert "not declared" in bare_rows["planning world"].detail


# --------------------------------------------------------------------------------------------
# The PATH, not only where it ends.
# --------------------------------------------------------------------------------------------


def _preflight(*, enabled: bool, stride: int = 1, fixtures=()):
    from src.config.schema.robot import RobotSafetyConfig, WorkspaceLimitsConfig
    from src.robot.safety import SafetyPreflight

    safety = RobotSafetyConfig.model_validate({
        "payload": {"enforce": False},
        "self_collision": {"fixtures": [
            {"name": f.name, "center_mm": list(f.center_mm),
             "half_extents_mm": list(f.half_extents_mm)} for f in fixtures
        ]},
        "trajectory_check": {"enabled": enabled, "stride": stride},
    })
    return SafetyPreflight.from_safety_config(safety, WorkspaceLimitsConfig())


class _Arm:
    from src.robot.core.capabilities import RobotCapabilities

    capabilities = RobotCapabilities(vendor="ur", model="ur5e", has_native_fk=True)


#: A configuration that folds a finger into the forearm. The exact-mesh backend refuses it with
#: "forearm|lfinger: mesh distance 0.367 mm"; the capsule proxy cannot see the gripper at all here.
_FOLDED = [1.95, 0.38, -1.33, -0.55, 2.00, 0.79]
_CLEAR = [0.0, -1.5, 1.5, 0.0, 0.0, 0.0]


def test_the_check_is_off_by_default_and_waves_any_path_through() -> None:
    preflight = _preflight(enabled=False)
    assert preflight.checks_trajectories is False
    assert preflight.gate_trajectory([_CLEAR, _FOLDED, _CLEAR], arm=_Arm()) is None


def _mesh_backend_available() -> bool:
    """The folded configuration is only refused by the EXACT-MESH backend; the capsule proxy cannot
    see the gripper on a joint-only context at all, which is why the shipped default is `fcl`."""
    from src.robot.safety._fcl_self_collision import mesh_backend_status

    return mesh_backend_status("ur5e") == "ok"


def test_a_colliding_waypoint_in_the_MIDDLE_is_caught() -> None:
    """The whole point. The endpoint gate cannot see this path: it starts and ends clear."""
    if not _mesh_backend_available():
        pytest.skip("no exact-mesh backend on this box")
    preflight = _preflight(enabled=True)
    assert preflight.checks_trajectories is True
    rejected = preflight.gate_trajectory([_CLEAR, _FOLDED, _CLEAR], arm=_Arm())
    assert rejected is not None, "a path through a self-collision was accepted"
    assert "lfinger" in (rejected.message or "") or "collision" in (rejected.message or "").lower()


def test_a_clear_path_is_accepted() -> None:
    preflight = _preflight(enabled=True)
    assert preflight.gate_trajectory([_CLEAR, _CLEAR, _CLEAR], arm=_Arm()) is None


def test_an_empty_trajectory_is_not_an_error() -> None:
    assert _preflight(enabled=True).gate_trajectory([], arm=_Arm()) is None


def test_the_endpoint_is_checked_whatever_the_stride() -> None:
    """A stride samples the path, but the arm certainly stops in the last configuration."""
    if not _mesh_backend_available():
        pytest.skip("no exact-mesh backend on this box")
    path = [_CLEAR] * 8 + [_FOLDED]
    assert _preflight(enabled=True, stride=64).gate_trajectory(path, arm=_Arm()) is not None


def test_a_stride_really_does_skip_configurations() -> None:
    """Honest about what the knob costs: index 1 is not visited with stride 4."""
    if not _mesh_backend_available():
        pytest.skip("no exact-mesh backend on this box")
    path = [_CLEAR, _FOLDED, _CLEAR, _CLEAR, _CLEAR]
    assert _preflight(enabled=True, stride=4).gate_trajectory(path, arm=_Arm()) is None
    assert _preflight(enabled=True, stride=1).gate_trajectory(path, arm=_Arm()) is not None


def test_a_declared_fixture_is_checked_along_the_path_too() -> None:
    """The bench is a fixture, and the self-collision guard carries fixtures, so this closes the
    'do not drive into the table' half of the goal for every configuration and not just the last."""
    from src.config.schema.robot import FixtureBoxConfig

    # A box around the flange at _CLEAR (from ur_link_transforms_mm).
    post = FixtureBoxConfig(
        name="post", center_mm=(-422.3, -232.9, 486.7), half_extents_mm=(60.0, 60.0, 60.0)
    )
    preflight = _preflight(enabled=True, fixtures=(post,))
    assert preflight.gate_trajectory([_CLEAR], arm=_Arm()) is not None


# --------------------------------------------------------------------------------------------
# The carried part. The planner's model used to end at the gripper.
# --------------------------------------------------------------------------------------------


def _ur5e_config_path() -> str:
    return "ext_deps/curobo/curobo/content/configs/robot/ur5e.yml"


def test_the_derived_config_declares_a_payload_link_on_the_tool_frame() -> None:
    import yaml

    from src.robot.safety.planning._curobo_attach import (
        ATTACHED_LINK_NAME, apply_attached_object_link,
    )

    src = pathlib.Path(_ur5e_config_path())
    if not src.is_file():
        pytest.skip("cuRobo content not installed on this box")
    config = yaml.safe_load(src.read_text(encoding="utf-8"))
    derived, added = apply_attached_object_link(config, spheres=16)
    assert added
    kin = derived["robot_cfg"]["kinematics"]
    assert kin["collision_link_names"][-1] == ATTACHED_LINK_NAME
    assert kin["extra_collision_spheres"][ATTACHED_LINK_NAME] == 16
    assert kin["extra_links"][ATTACHED_LINK_NAME]["parent_link_name"] == "tool0"
    # The source is untouched: the transform copies.
    assert ATTACHED_LINK_NAME not in config["robot_cfg"]["kinematics"]["collision_link_names"]


def test_the_payload_is_ignored_against_the_links_that_HOLD_it() -> None:
    """Without this the planner refuses everything.

    MEASURED on ur5e.yml against the real planner: with the link derived but no ignore entries,
    carrying a 10 mm cube turned a 61-waypoint plan into no plan. That is not the part meeting the
    world, it is the part permanently inside the gripper holding it.
    """
    from src.robot.safety.planning._curobo_attach import (
        ATTACHED_LINK_NAME, apply_attached_object_link,
    )

    config = {"robot_cfg": {"kinematics": {
        "tool_frames": ["tool0"],
        "collision_link_names": ["wrist_3_link", "tool0"],
        "self_collision_ignore": {"wrist_3_link": ["tool0"], "wrist_2_link": ["tool0"]},
    }}}
    derived, added = apply_attached_object_link(config, spheres=8)
    assert added
    ignore = derived["robot_cfg"]["kinematics"]["self_collision_ignore"]
    # The parent, and everything the config itself already calls adjacent to the parent.
    assert ATTACHED_LINK_NAME in ignore["tool0"]
    assert ATTACHED_LINK_NAME in ignore["wrist_3_link"]
    assert ATTACHED_LINK_NAME in ignore["wrist_2_link"]


def test_a_config_that_already_declares_the_link_is_left_alone() -> None:
    """franka.yml ships one. A robot whose author declared it keeps exactly what they wrote."""
    from src.robot.safety.planning._curobo_attach import apply_attached_object_link

    config = {"robot_cfg": {"kinematics": {
        "tool_frames": ["panda_hand"],
        "collision_link_names": ["panda_hand", "attached_object"],
    }}}
    derived, added = apply_attached_object_link(config, spheres=16)
    assert added is False
    assert derived is config


def test_zero_slots_is_the_byte_identical_path() -> None:
    from src.robot.safety.planning._curobo_attach import apply_attached_object_link

    config = {"robot_cfg": {"kinematics": {"tool_frames": ["tool0"], "collision_link_names": []}}}
    derived, added = apply_attached_object_link(config, spheres=0)
    assert added is False
    assert derived is config


def test_a_config_with_no_tool_frame_is_refused_rather_than_guessed() -> None:
    from src.robot.safety.planning._curobo_attach import apply_attached_object_link

    config = {"robot_cfg": {"kinematics": {"collision_link_names": ["link"]}}}
    assert apply_attached_object_link(config, spheres=16)[1] is False


def test_the_arm_turns_a_jaw_WIDTH_into_a_box_the_planner_can_use() -> None:
    """The jaw opening measures the part at the grasp LINE. The length is a declared worst case."""
    from src.config.schema.robot import RobotConfig
    from src.robot.drivers.ur.arm import URRobotArm

    config = RobotConfig.model_validate({
        "vendor": "ur",
        "ur": {"motion_planner": "curobo"},
        "safety": {
            "payload": {"enforce": False},
            "planning_world": {
                "enabled": True,
                "support_plane": {"height_mm": 0.0, "extent_mm": [1600.0, 1600.0],
                                  "thickness_mm": 50.0},
                "payload": {"enabled": True, "length_mm": 120.0, "lateral_margin_mm": 10.0},
            },
        },
    })
    arm = URRobotArm(config)
    arm._conn = _FakeConn()  # type: ignore[assignment]
    seen: dict = {}

    class _Planner:
        def attach_payload(self, joints, dims_mm, offset_mm):
            seen.update(joints=list(joints), dims=tuple(dims_mm), offset=offset_mm)
            return True

    arm._curobo_ur = _Planner()  # type: ignore[assignment]
    assert arm.attach_payload(60.0) is True
    assert seen["dims"] == (70.0, 70.0, 120.0), "jaw width plus the lateral margin, then the length"
    assert seen["offset"] == 60.0, "the box centre sits half its length beyond the flange"


def test_the_payload_is_off_by_default_and_the_arm_says_so() -> None:
    from src.config.schema.robot import RobotConfig
    from src.robot.drivers.ur.arm import URRobotArm

    arm = URRobotArm(RobotConfig.model_validate({"vendor": "ur"}))
    arm._conn = _FakeConn()  # type: ignore[assignment]
    assert arm.attach_payload(60.0) is False
    assert arm.detach_payload() is True  # nothing was ever attached


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


def test_a_gripper_variant_may_not_be_paired_with_another_robot() -> None:
    """The variant bundle carries the ARM meshes of the robot it was baked from.

    MEASURED: `schunk_egu50_collision_meshes.npz` holds the ur5e arm links byte-for-byte
    (`forearm__v` is array-equal to ur5e's and not to ur3e's), and the status token used to be `ok`
    for a ur3e cell. That cell would have checked UR5e arm geometry on UR3e DH frames with the guard
    reporting itself healthy. The module docstring already stated the rule; nothing enforced it.
    """
    from src.robot.safety._fcl_self_collision import make_backend, mesh_backend_status

    if not pathlib.Path("src/robot/safety/data/ur3e_collision_meshes.npz").is_file():
        pytest.skip("no ur3e bundle on this box")
    assert mesh_backend_status("ur5e", None, "schunk_egu50") in {"ok", "no_engine"}
    assert mesh_backend_status("ur3e", None, "schunk_egu50") == "variant_model_mismatch"
    assert make_backend("ur3e", None, "schunk_egu50") is None, "it must refuse, not run on wrong meshes"


def test_the_bundles_the_mismatch_check_relies_on_really_do_differ() -> None:
    """If this ever fails, the check above is testing nothing."""
    import numpy as np

    root = pathlib.Path("src/robot/safety/data")
    if not (root / "ur3e_collision_meshes.npz").is_file():
        pytest.skip("no ur3e bundle on this box")
    with np.load(root / "schunk_egu50_collision_meshes.npz") as variant, \
            np.load(root / "ur5e_collision_meshes.npz") as ur5e, \
            np.load(root / "ur3e_collision_meshes.npz") as ur3e:
        assert np.array_equal(variant["forearm__v"], ur5e["forearm__v"])
        assert not np.array_equal(variant["forearm__v"], ur3e["forearm__v"])
