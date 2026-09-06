"""32: the path between two checked endpoints, and the three keys that make it visible.

    python scripts/examples/safety/32_declare_your_bench.py
    python scripts/examples/safety/32_declare_your_bench.py --profile ur5e

Every guard in 30 judges where a move ends. A planner does not hand over an endpoint, it hands over
a path, and with `safety.trajectory_check.enabled` false nothing looks at the middle of it: the sim
applies each waypoint straight to the articulation and a real UR runs them in turn, so a plan that
grazes a bench halfway and lands clear passes every check there is.

That is not a bug, it is a default, and the code says so where it becomes true. `gate_trajectory`
logs one warning per preflight naming the waypoint count and the three keys to declare, and then
returns None and lets the path run. It warns rather than refuses because every profile in this
repository ships with the check off, no fixtures and no planning world, so a refusal there would
refuse the shipped tree. What is missing is not a rule, it is the sentence, and nobody reads a YAML
comment with an arm in front of them.

Three keys, and they are not interchangeable:

  safety.trajectory_check.enabled   whether the middle of a path is judged at all
  safety.self_collision.fixtures    what the guard knows is standing in the cell
  safety.planning_world             what the planner is told, so it stops proposing those paths

Turning the check on with no fixtures declared judges 81 configurations against an empty room, and
this file runs exactly that case to show it comes back clear. `config/robot/robot.ur5e.yaml`
declares the fixtures and the planning world for one worked bench, and leaves the check off, so a
cell that merges that layer still needs the first key.

The path here is synthesised rather than planned: it interpolates from your declared home into a
configuration that reaches into your declared fixture and back out, so both endpoints are clear and
the middle is not. cuRobo, which is what would produce a real one, runs in a separate environment
this process does not spawn, and the probe below reports whether it is even installed. A real plan
is smoother and no more examined.

Nothing here connects to anything and there is no `--live`. It builds the driver your config names
and asks its guard pipeline about a list of joint vectors.
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import EXIT_FAILED, EXIT_OK, Example, not_ready  # noqa: E402

EPILOG = """
exit codes
  0  the unexamined path was shown, and then shown refused with the keys declared
  1  the keys were declared and the path still came back clear
  2  no cell to ask: no `robot` block, an arm with no guard pipeline, or no
     configuration in the ladder that reaches the declared fixture
"""

#: How many waypoints the synthesised path carries. A cuRobo plan for a reach of this size is the
#: same order of magnitude, and the count is what the warning quotes back.
_WAYPOINTS = 81

#: Configurations tried, in order, until one reaches into the declared fixture while the same guard
#: with no fixtures accepts it. That pair of answers is the whole demonstration, so the ladder is
#: searched rather than assumed: which joint vector dips into a bench depends on the arm and on
#: where the bench is.
_DIP_LADDER: tuple[tuple[float, ...], ...] = (
    (3.14, -0.8, 2.0, -2.15, -1.57, 0.0),
    (3.14, -0.9, 2.2, -2.3, -1.57, 0.0),
    (3.14, -0.7, 1.8, -2.0, -1.57, 0.0),
    (3.14, -1.0, 2.3, -2.3, -1.57, 0.0),
    (3.14, -0.6, 1.6, -1.9, -1.57, 0.0),
)

#: Strides tried after the full check, to show what sampling a path costs. The endpoint is always
#: checked whatever the stride; every configuration between two samples is not.
_STRIDES: tuple[int, ...] = (4, 8)

#: The logger the preflight writes its unexamined-path warning to. `create_robot_logger` builds it
#: with `propagate = False` and its own handlers, so a handler has to be attached to this name
#: rather than to a parent.
_PREFLIGHT_LOGGER = "SafetyPreflight"


class _Collect(logging.Handler):
    """Keeps the warnings a block of code logs, so the run can print them where they belong.

    The whole subject of this file announces itself on the logging channel and nowhere else, so an
    example that told the reader to go and look in a log would be demonstrating the problem instead
    of the warning. Collected here and printed inline, the warning is part of the output.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@contextmanager
def _warnings_from(logger_name: str) -> Iterator[list[str]]:
    """Collect WARNING records from ``logger_name`` for the duration."""
    handler = _Collect()
    logger = logging.getLogger(logger_name)
    logger.addHandler(handler)
    try:
        yield handler.messages
    finally:
        logger.removeHandler(handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__ or "", epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--profile", default=None,
        help="WILLY_PROFILE to load (e.g. `ur5e`, `ur3e`, `ursim`). Default: the environment's.")
    args = parser.parse_args(argv)

    from src.config import ConfigError, load_robot_config
    from src.config.schema.robot import TrajectoryCheckConfig
    from src.robot.constants import home_joints_default
    from src.robot.core import JointPositions, MotionCommand, RobotVendor
    from src.robot.drivers import create_arm
    from src.robot.safety import SafetyContext, SafetyPreflight, SelfCollisionGuard
    from src.robot.safety.planning.stack import MotionStack
    from src.robot.safety.planning.world import build_planner_cuboids, describe_planner_world

    with Example("32 declare_your_bench", "the waypoints between the two that are checked",
                 hardware=False, profile=args.profile) as run:
        with run.step("load the config tree") as report:
            try:
                config = load_robot_config()
            except ConfigError as error:
                return not_ready(f"the config tree has no cell to check ({error})",
                                 "select a profile that has a `robot` block: WILLY_PROFILE=ur5e")
            block = config.safety.self_collision
            report(f"trajectory_check {config.safety.trajectory_check.enabled}, "
                   f"{len(block.fixtures)} fixture(s), "
                   f"planning_world {config.safety.planning_world.enabled}")

        with run.step("build the arm, and do not connect it") as report:
            try:
                arm = create_arm(RobotVendor.from_string(config.vendor), config=config)
            except Exception as error:                       # noqa: BLE001  (report, not raise)
                # The typed refusal of the factory, turned into an answer. The `sim` vendor is the
                # case that gets here: its driver is built by the Isaac runners from a
                # `SimRobotConfig`, not from the `robot` block, so there is no arm to ask.
                return not_ready(
                    f"the `{config.vendor}` arm could not be built "
                    f"({type(error).__name__}: {error})",
                    "this example reads the guard pipeline of a real cell's driver; try a profile "
                    "whose vendor builds one from the robot block: WILLY_PROFILE=ur5e")
            report(f"{type(arm).__name__}, {arm.capabilities.vendor} {arm.capabilities.model}")

        with run.step("who would plan the path") as report:
            probe = MotionStack.from_robot_config(config).probe()
            report(f"cuRobo {'available' if probe.environment.curobo.available else 'MISSING'}, "
                   f"so the path below is interpolated here")

        fixtures = list(block.fixtures)
        if not fixtures:
            # Borrowed from the shipped bench layer rather than invented here. A fixture is three
            # measurements of one cell, and a plausible one written into an example is how the next
            # cell inherits a wrong value in silence.
            run.note("")
            run.note("This tree declares no fixtures, so there is no bench in the collision world")
            run.note("at all. The worked one from `config/robot/robot.ur5e.yaml` is loaded instead")
            run.note("and used below; on your own cell it is three measurements of your bench.")
            try:
                fixtures = list(load_robot_config(profile="ur5e").safety.self_collision.fixtures)
            except ConfigError:
                fixtures = []
        if not fixtures:
            return not_ready("no fixture to reach into, declared or shipped",
                             "declare safety.self_collision.fixtures for your bench, bin or "
                             "enclosure; robot.ur5e.yaml has a worked example")
        declared = block.model_copy(update={"fixtures": fixtures})
        run.note("")
        for fixture in fixtures:
            centre = tuple(round(float(v)) for v in fixture.center_mm)
            half = tuple(round(float(v)) for v in fixture.half_extents_mm)
            run.note(f"  fixture '{fixture.name}': centre {centre} mm, half extents {half} mm")

        if block.kinematics_model is None and arm.capabilities.vendor != "ur":
            # Before the ladder, because the ladder would come back empty for a reason that has
            # nothing to do with the joint vectors in it. Without a DH chain the guard has no arm
            # to place anywhere, so no configuration can reach the fixture.
            return not_ready(
                f"this arm reports vendor '{arm.capabilities.vendor}' and the tree sets no "
                "safety.self_collision.kinematics_model, so the guard has no arm geometry to "
                "check a path against",
                "set safety.self_collision.kinematics_model to the arm this cell physically is, "
                "or run this against a UR profile: WILLY_PROFILE=ur5e")

        with_bench = SelfCollisionGuard(declared)
        empty_room = SelfCollisionGuard(declared.model_copy(update={"fixtures": []}))

        def joint_ctx(values: "tuple[float, ...] | np.ndarray") -> SafetyContext:
            return SafetyContext(command=MotionCommand.MOVE_JOINTS, arm=arm,
                                 target_joints=JointPositions([float(v) for v in values]))

        home = np.asarray(config.home_joint_positions or home_joints_default(), dtype=np.float64)
        with run.step("a configuration that reaches into the fixture") as report:
            dip: np.ndarray | None = None
            for candidate in _DIP_LADDER:
                if (empty_room.evaluate(joint_ctx(candidate)).accepted
                        and not with_bench.evaluate(joint_ctx(candidate)).accepted):
                    dip = np.asarray(candidate, dtype=np.float64)
                    break
            report("none of the ladder reaches it" if dip is None
                   else f"joints {[round(float(v), 2) for v in dip]}")
        if dip is None:
            return not_ready(
                "no configuration in the ladder reaches this cell's fixture",
                "the ladder is written for an arm reaching forward over a bench in front of the "
                "base; edit _DIP_LADDER for your own cell, or run with WILLY_PROFILE=ur5e")
        if not with_bench.evaluate(joint_ctx(home)).accepted:
            return not_ready(
                "this cell's home configuration already touches the fixture",
                "the demonstration needs two clear endpoints; check home_joint_positions and the "
                "fixture against each other before anything is planned")

        # Out and back, so both endpoints are the same clear configuration and everything the guard
        # would refuse is strictly between them.
        out = _WAYPOINTS // 2 + 1
        waypoints = [tuple(home + (dip - home) * (i / (out - 1))) for i in range(out)]
        waypoints += [tuple(dip + (home - dip) * (i / (_WAYPOINTS - out)))
                      for i in range(1, _WAYPOINTS - out + 1)]

        as_shipped = SafetyPreflight.from_safety_config(
            config.safety.model_copy(update={
                "self_collision": declared,
                "trajectory_check": TrajectoryCheckConfig(enabled=False)}),
            config.workspace_limits)

        run.note("")
        with run.step("gate the endpoint, the way a move is gated") as report:
            endpoint = as_shipped.gate_joint_target(
                JointPositions([float(v) for v in waypoints[-1]]), arm=arm)
            report("clear" if endpoint is None else f"refused: {endpoint.message}")

        with run.step("gate the path, with the check off") as report:
            with _warnings_from(_PREFLIGHT_LOGGER) as logged:
                refusal = as_shipped.gate_trajectory(waypoints, arm=arm)
            report(f"{len(waypoints)} waypoints, "
                   f"{'nothing refused' if refusal is None else refusal.message}, "
                   f"{len(logged)} warning(s)")
        for message in logged:
            run.note("")
            for line in textwrap.wrap(message, width=74):
                run.note(f"    | {line}")
        run.note("")
        run.note("That line is logged once per preflight, not once per move, so a pick loop says")
        run.note("it at the first plan and never again. It is the only thing standing between a")
        run.note("plan through your bench and a plan around it.")

        checked = SafetyPreflight.from_safety_config(
            config.safety.model_copy(update={
                "self_collision": declared,
                "trajectory_check": TrajectoryCheckConfig(enabled=True, stride=1)}),
            config.workspace_limits)
        with run.step("the same path, with the check on") as report:
            started = time.perf_counter()
            refused = checked.gate_trajectory(waypoints, arm=arm)
            elapsed_ms = 1000.0 * (time.perf_counter() - started)
            report(f"{'clear' if refused is None else 'REFUSED'}, "
                   f"{elapsed_ms / len(waypoints):.1f} ms per configuration on this box")
        if refused is not None:
            run.note(f"       {refused.status.value}: {refused.message}")
            run.note("       That configuration is in the middle of the path. Both endpoints are")
            run.note("       clear, and the endpoint gate above said so.")
            if "fixture:fixture" in (refused.message or ""):
                # Not a mistake in your YAML. The exact-mesh backend is handed boxes without their
                # names, so it can say which link and not which fixture; the capsule path names it.
                run.note("       The pair reads `fixture:fixture` rather than the name declared")
                run.note("       above: the exact-mesh backend receives the boxes without names.")

        blind = SafetyPreflight.from_safety_config(
            config.safety.model_copy(update={
                "self_collision": declared.model_copy(update={"fixtures": []}),
                "trajectory_check": TrajectoryCheckConfig(enabled=True, stride=1)}),
            config.workspace_limits)
        with run.step("check on, fixtures not declared") as report:
            missed = blind.gate_trajectory(waypoints, arm=arm)
            report("clear" if missed is None else f"refused: {missed.message}")
        run.note("       Every configuration was judged, against a cell with nothing in it. The")
        run.note("       switch is not the declaration: the guard refuses what it was told about.")

        run.note("")
        run.note("Sampling instead of checking. `stride` trades coverage for time, and the")
        run.note("endpoint is checked whatever it is set to:")
        for stride in _STRIDES:
            sampled = SafetyPreflight.from_safety_config(
                config.safety.model_copy(update={
                    "self_collision": declared,
                    "trajectory_check": TrajectoryCheckConfig(enabled=True, stride=stride)}),
                config.workspace_limits)
            verdict = sampled.gate_trajectory(waypoints, arm=arm)
            run.note(f"  stride {stride}: {len(range(0, len(waypoints), stride))} of "
                     f"{len(waypoints)} configurations judged, "
                     f"{'still refused' if verdict is not None else 'came back CLEAR'}")

        run.note("")
        run.note("The third key is a different audience. `fixtures` is what refuses a path;")
        run.note("`planning_world` is what the planner is given, so it stops proposing one.")
        run.note("Registering a world replaces the planner's own rather than adding to it, which")
        run.note("is why declaring it requires a support plane. As this tree stands:")
        for line in describe_planner_world(
                build_planner_cuboids(config.safety.planning_world, block.fixtures)).splitlines():
            run.note(f"  {line.rstrip()}")
        # Always print the other side, so the contrast is visible whichever way the tree is set.
        flipped = not config.safety.planning_world.enabled
        run.note(f"and with `planning_world.enabled: {str(flipped).lower()}`:")
        for line in describe_planner_world(build_planner_cuboids(
                config.safety.planning_world.model_copy(update={"enabled": flipped}),
                fixtures)).splitlines():
            run.note(f"  {line.rstrip()}")

        run.note("")
        run.note("What to put in your robot layer. The fixture numbers are yours to measure; the")
        run.note("two switches are not:")
        run.note("")
        run.note("  safety:")
        run.note("    trajectory_check:")
        run.note("      enabled: true          # judge the middle of a plan, not only its end")
        run.note("      stride: 1              # every configuration; raise it to trade coverage")
        run.note("    self_collision:")
        run.note("      fixtures:")
        run.note("        - name: \"bench\"     # centre and half extents in the base frame, mm")
        run.note("          center_mm: [.., .., ..]")
        run.note("          half_extents_mm: [.., .., ..]")
        run.note("    planning_world:")
        run.note("      enabled: true          # hand the same boxes to the planner")
        run.note("      support_plane: {height_mm: 0.0, extent_mm: [800.0, 800.0], "
                 "thickness_mm: 50.0}")
        run.note("")
        run.note("The cost is the time printed above, once per planned move, before the arm")
        run.note("starts. It is not interleaved with control and it does not slow a motion down.")

        if refused is None:
            # The keys were declared and the path still came back clear. Nothing was demonstrated,
            # so this must not exit 0: an example whose point did not happen is the thing this
            # file exists to argue against.
            return EXIT_FAILED
        return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
