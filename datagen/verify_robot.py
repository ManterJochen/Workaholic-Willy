"""Prove on-box that a posed arm really lands where the generator says it does.

Every other check in this package is self-consistent: the IK is solved against this package's DH
chain, so checking it with that same chain proves arithmetic, not truth. This one asks a different
oracle. It sets the joints on the Isaac articulation and reads the link's world pose back out of USD,
a number computed from the robot asset rather than from the DH table.

That distinction is not academic. A base frame turned 180 deg about Z stands the arm in every scene,
reaches away from the workspace and appears in none of the images, while every self-consistent check
reports success.

A robot passes when, over a spread of configurations:

* the joints read back as they were set, so the pose was applied rather than merely requested;
* USD's link pose matches the forward kinematics here within 1 mm / 0.5 deg, which is under a pixel
  at every camera in the default rig, where the closest view resolves 0.44 mm;
* the arm clears itself by the configured margin;
* cuRobo can reach each configuration from park without hitting the table or itself.

That last check is a different question from the three above. The capsule clearance in
:mod:`datagen.render.arm` answers "does this pose overlap anything", using a deliberately crude model
that is fast enough to run per viewpoint. cuRobo answers "is this pose somewhere the robot could
actually go", against the same mesh-level world the pick path plans in, and a configuration no planner
will move to is a configuration a real cell will never photograph from, however clear its capsules
are.

It is asked about the model under test, not the repo default: the descriptor name is derived from the
model key, so verifying a UR3e cannot quietly validate a UR5e. When cuRobo is not installed the check
reports ``UNAVAILABLE`` and the robot does not pass on its account; ``--require-curobo`` turns that
into a failure for a box where it is meant to be present.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.utility.log_cfg import create_logger

from datagen.constants import DATAGEN_LOG_DIR, VERIFY_ROBOT_LOG_FILE

__all__ = ["RobotCheck", "verify_robot"]

logger = create_logger("datagen.verify_robot", VERIFY_ROBOT_LOG_FILE, log_dir=DATAGEN_LOG_DIR)

#: One millimetre is at or under a pixel at the fixed cameras of the default rig, and two to three
#: pixels at the wrist, which sits much closer to the objects.
_POSITION_TOLERANCE_MM = 1.0
_ROTATION_TOLERANCE_DEG = 0.5
#: Configurations to test: park plus a deterministic spread around it. The seed is fixed because a
#: robot that passes must pass again, or "verified" means nothing.
_PROBE_COUNT = 12
_PROBE_SEED = 20260812
# The table cuRobo plans through. Its footprint comes from `workspace.table_size_mm`, so the table
# is declared once and only the thickness lives here; a metre copy and a millimetre copy kept in step
# by a comment would let this check go on planning through the old size and passing.
#
# The slab hangs below z=0 because objects sit on z=0. Without a world cuRobo checks only
# self-collision, and "the arm may not be inside the table" is most of what makes a posed
# configuration physically possible.
_TABLE_THICKNESS_M = 0.02


@dataclass(frozen=True, slots=True)
class RobotCheck:
    """What the on-box check found. ``ok`` is the verdict; the rest is the evidence."""

    model: str
    probes: int
    worst_position_mm: float
    worst_rotation_deg: float
    worst_self_clearance_mm: float
    problems: tuple[str, ...]
    curobo: str = "not run"

    @property
    def ok(self) -> bool:
        return not self.problems

    def summary(self) -> str:
        head = (
            f"{self.model}: {self.probes} configuration(s)\n"
            f"  USD vs our FK   position {self.worst_position_mm:6.2f} mm  "
            f"rotation {self.worst_rotation_deg:5.2f} deg   "
            f"(limit {_POSITION_TOLERANCE_MM} mm / {_ROTATION_TOLERANCE_DEG} deg)\n"
            f"  self-clearance  {self.worst_self_clearance_mm:6.1f} mm (worst)\n"
            f"  cuRobo world    {self.curobo}"
        )
        if self.ok:
            return f"PASS  {head}"
        return f"FAIL  {head}\n  " + "\n  ".join(self.problems[:12])


def _rotation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = first[:3, :3] @ second[:3, :3].T
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _probe_configurations() -> list[np.ndarray]:
    """Park plus a deterministic spread around it. Shared by both checks, so they test one thing."""
    from datagen.render.kinematics import PARK_JOINTS_RAD

    generator = np.random.default_rng(_PROBE_SEED)
    park = np.asarray(PARK_JOINTS_RAD, dtype=np.float64)
    return [park if probe == 0 else park + generator.uniform(-0.5, 0.5, size=park.shape)
            for probe in range(_PROBE_COUNT)]


def _curobo_reachability(model: str, configurations: list[np.ndarray],
                         table_size_m: float) -> tuple[str, list[str]]:
    """Ask cuRobo to plan from park to each configuration through the table. ``(status, problems)``.

    Plans rather than queries because planning is the question that matters and the only one the
    client exposes: a configuration cuRobo will not move to is one a real cell never reaches, whatever
    a static overlap test says. Park is both the start and the first probe, so probe 0 also proves the
    trivial case is not being failed by the setup itself.

    Runs before Isaac boots. It needs no stage, and Isaac's ``close()`` terminates the process, so a
    check that ran after the renderer could not report what it found.
    """
    from src.robot.safety.planning.curobo_client import (
        CuroboPlanClient,
        CuroboUnavailableError,
    )
    from src.robot.safety.planning.environment import curobo_env_available

    if not curobo_env_available():
        return ("UNAVAILABLE; no cuRobo env on this box, so world collision was NOT checked", [])

    table = [{
        "name": "table",
        "dims_m": [table_size_m, table_size_m, _TABLE_THICKNESS_M],
        "pose": [0.0, 0.0, -_TABLE_THICKNESS_M / 2.0, 1.0, 0.0, 0.0, 0.0],
    }]
    problems: list[str] = []
    reached = 0
    park = [float(v) for v in configurations[0]]
    try:
        with CuroboPlanClient(robot_config=f"{model}.yml") as client:
            registered = client.set_world(table)
            if registered < 1:
                return (f"UNAVAILABLE; cuRobo accepted {registered} of 1 world cuboid, so the "
                        f"table was not in the world it planned against", [])
            for probe, joints in enumerate(configurations):
                goal = [float(v) for v in joints]
                if client.plan_joint(park, goal) is not None:
                    reached += 1
                    continue
                problems.append(
                    f"probe {probe}: cuRobo found no collision-free path from park to "
                    f"{np.round(joints, 3).tolist()}; the arm cannot actually get to a "
                    f"configuration the generator is willing to pose it at"
                )
    except CuroboUnavailableError as exc:
        return (f"UNAVAILABLE; {exc}", [])
    except Exception as exc:  # noqa: BLE001 (an unusable planner must not look like a clean pass)
        return (f"UNAVAILABLE; {type(exc).__name__}: {exc}", [])
    return (f"{reached}/{len(configurations)} reachable from park through the table", problems)


def verify_robot(model: str, *, headless: bool = True, require_curobo: bool = False) -> RobotCheck:
    """Boot Isaac, pose the arm through a spread of configurations, and compare against USD."""
    from datagen.config import ArmConfig, DatagenConfig, RenderConfig, WorkspaceConfig
    from datagen.render.kinematics import wrist_link_pose_mm
    from datagen.robots import resolve_robot

    definition = resolve_robot(model)  # refuses here, before Isaac, if anything is undeclared
    config = DatagenConfig(
        scenes=1,
        workspace=WorkspaceConfig(center_mm=definition.workspace_center_mm,
                                  half_extents_mm=definition.workspace_half_extents_mm),
        render=RenderConfig(arm=ArmConfig(mode="posed", robot_model=definition.key)),
    )

    configurations = _probe_configurations()
    logger.info("verifying %s over %d configuration(s), tolerance %.1f mm / %.2f deg, headless=%s",
                definition.key, len(configurations), _POSITION_TOLERANCE_MM,
                _ROTATION_TOLERANCE_DEG, headless)
    curobo_status, curobo_problems = _curobo_reachability(
        definition.key, configurations, config.workspace.table_size_mm / 1000.0)
    # An `UNAVAILABLE` planner is the degraded path this check is most often read wrong on: the run
    # still finishes and still reports `PASS`, having never asked whether the arm could get there.
    if curobo_status.startswith("UNAVAILABLE"):
        logger.warning("cuRobo world collision NOT checked: %s", curobo_status)
    else:
        logger.info("cuRobo world collision: %s", curobo_status)

    from datagen.render.isaac import ROBOT_ROOT, IsaacRenderer

    problems: list[str] = list(curobo_problems)
    if require_curobo and curobo_status.startswith("UNAVAILABLE"):
        problems.append(
            f"--require-curobo was given and the world-collision check did not run: {curobo_status}"
        )
    worst_position = worst_rotation = 0.0
    worst_clearance = float("inf")

    with IsaacRenderer(config, headless=headless) as renderer:
        renderer._author_arm()  # noqa: SLF001
        renderer._world.reset()  # noqa: SLF001
        renderer._step(10, render=False)  # noqa: SLF001
        renderer._bind_articulation()  # noqa: SLF001

        from pxr import UsdGeom  # type: ignore[import-not-found]

        import omni.usd  # type: ignore[import-not-found]

        stage = omni.usd.get_context().get_stage()
        link_path = f"{ROBOT_ROOT}/{definition.wrist_link_name}"
        if not stage.GetPrimAtPath(link_path).IsValid():
            problems.append(f"no prim at {link_path}; wrist_link_name does not match the asset")
            logger.error("%s: no prim at %s; wrist_link_name does not match the asset; "
                         "nothing was compared", definition.key, link_path)
            return RobotCheck(definition.key, 0, 0.0, 0.0, 0.0, tuple(problems), curobo_status)

        for probe, joints in enumerate(configurations):
            renderer._hold_joints(joints)  # noqa: SLF001

            read_back = np.asarray(renderer._articulation.get_joint_positions(), dtype=np.float64)  # noqa: SLF001
            drift = float(np.max(np.abs(read_back[:6] - joints)))
            if drift > 1e-3:
                problems.append(
                    f"probe {probe}: joints drifted {drift:.4f} rad from what was set; the arm is "
                    f"not holding the configuration it was posed at"
                )

            matrix = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(link_path))
            from_usd = np.eye(4)
            from_usd[:3, :3] = np.array([[matrix[r][c] for c in range(3)] for r in range(3)]).T
            from_usd[:3, 3] = np.array([matrix[3][0], matrix[3][1], matrix[3][2]]) * 1000.0

            ours = wrist_link_pose_mm(definition.key, joints)
            position_error = float(np.linalg.norm(from_usd[:3, 3] - ours[:3, 3]))
            rotation_error = _rotation_error_deg(from_usd, ours)
            worst_position = max(worst_position, position_error)
            worst_rotation = max(worst_rotation, rotation_error)
            if position_error > _POSITION_TOLERANCE_MM or rotation_error > _ROTATION_TOLERANCE_DEG:
                problems.append(
                    f"probe {probe}: USD says {np.round(from_usd[:3, 3], 1)} mm, our forward "
                    f"kinematics says {np.round(ours[:3, 3], 1)} mm; {position_error:.2f} mm / "
                    f"{rotation_error:.2f} deg apart"
                )

            clearance = renderer._self_clearance_mm(joints)  # noqa: SLF001
            worst_clearance = min(worst_clearance, clearance)
            if clearance < config.render.arm.self_collision_margin_mm:
                problems.append(
                    f"probe {probe}: self-clearance {clearance:.1f} mm is below the "
                    f"{config.render.arm.self_collision_margin_mm} mm margin"
                )

        report = RobotCheck(
            definition.key, _PROBE_COUNT, worst_position, worst_rotation,
            0.0 if worst_clearance == float("inf") else worst_clearance, tuple(problems),
            curobo_status,
        )
        # Printed here, inside the block, and that is not a style choice. Isaac's close() ends in
        # shutdown_and_release_framework(), which terminates the process, so anything printed after
        # the `with` never reaches the console. The log line is inside for the same reason, and it is
        # the half of the verdict that survives the teardown at all.

        if report.ok:
            logger.info("%s PASS: %d probe(s), worst %.2f mm / %.2f deg, self-clearance %.1f mm, "
                        "cuRobo %s", report.model, report.probes, report.worst_position_mm,
                        report.worst_rotation_deg, report.worst_self_clearance_mm, report.curobo)
        else:
            logger.error("%s FAIL: %d problem(s) over %d probe(s), worst %.2f mm / %.2f deg. "
                         "First: %s", report.model, len(report.problems), report.probes,
                         report.worst_position_mm, report.worst_rotation_deg, report.problems[0])
        print(report.summary(), flush=True)
    return report
