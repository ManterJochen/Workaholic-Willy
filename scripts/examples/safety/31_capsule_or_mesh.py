"""31: `safety.self_collision.backend`, and the failure that does not refuse.

    python scripts/examples/safety/31_capsule_or_mesh.py
    python scripts/examples/safety/31_capsule_or_mesh.py --profile ur3e

The decision is one key, `robot.safety.self_collision.backend`, with two answers.

`capsule`, always available, approximates the arm as a chain of capsules, one per link, plus a
base column and one capsule for the tool, and measures closed-form distances. It costs a fraction
of a millisecond and needs nothing installed.

`fcl`, the shipped default, measures exact distance between the real collision meshes, which are
committed per model as `src/robot/safety/data/{model}_collision_meshes.npz`. It needs a collision
engine, Coal where present and python-fcl otherwise, and it needs a bundle for your model. It
removes two errors of the proxy that are measured in this repository: the wrist pairs the proxy has
to skip, and the false rejections its 60 mm default link radius invents.

The part worth reading twice. With the engine or the bundle missing, `fcl` does not fail closed. It
logs one warning naming the reason and runs the capsule proxy for the rest of that cell's life, and
the guard then answers ACCEPTED with reason `ok` on configurations the exact path refuses. Nothing
in the return value says the cell was degraded. This file makes that happen on purpose and prints
both the warning and the verdict, because a cell that swapped one backend for the other in silence
is the kind of regression an operator finds with a broken gripper.

The schema disagrees with the code here, and the code wins. The `SelfCollisionSafetyConfig`
docstring in `src/config/schema/robot/safety_schema.py` still says that `fcl` without a populated
`mesh_dir` reports UNAVAILABLE and rejects the motion while `enforce` is true. It does not:
`src/robot/safety/README.md` already names that docstring as stale and the code as authoritative.
The run below is which of the two is true on your box.

Nothing here connects to anything, and there is no `--live`. It builds the driver your config
names, asks it for its own guard pipeline, and evaluates contexts.
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
  0  both backends answered and the difference between them is printed above
  1  the configured backend could not be exercised on this cell
  2  no cell to ask: no `robot` block, or an arm with no guard pipeline
"""

#: The configuration the two backends disagree about, and why it is a constant rather than
#: something derived from the config: a self-collision is a pose of an arm, not a number in a file.
#: It drives a gripper finger into the forearm. The exact-mesh path measures it; the capsule path
#: cannot see it at all on a joint-only context, because it builds its tool capsule from
#: `ctx.target_pose` and there is none.
_FOLDED = (1.95, 0.38, -1.33, -0.55, 2.00, 0.79)


class _Collect(logging.Handler):
    """Keeps the warnings a block of code logs, so the run can print them where they belong.

    The fallback this file demonstrates announces itself on the logging channel and nowhere else,
    so an example that told the reader to go and look in a log would be demonstrating the problem
    rather than the warning. Collected here and printed inline, the warning is part of the output.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@contextmanager
def _warnings_from(logger_name: str) -> Iterator[list[str]]:
    """Collect WARNING records from ``logger_name`` and its children for the duration."""
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
    from src.geometry import Frame, Pose
    from src.geometry.quaternion import from_axis_angle
    from src.robot.core import JointPositions, MotionCommand, RobotVendor
    from src.robot.drivers import create_arm
    from src.robot.safety import (
        UR_JOINT_LIMITS_DEG,
        SafetyContext,
        SafetyDecision,
        SelfCollisionGuard,
    )
    from src.robot.safety.planning import probe_collision_engine
    from src.robot.safety.planning.stack import MotionStack

    with Example("31 capsule_or_mesh", "the exact meshes, the proxy, and the silent swap",
                 hardware=False, profile=args.profile) as run:
        with run.step("load the config tree") as report:
            try:
                config = load_robot_config()
            except ConfigError as error:
                return not_ready(f"the config tree has no cell to check ({error})",
                                 "select a profile that has a `robot` block: WILLY_PROFILE=ur5e")
            block = config.safety.self_collision
            report(f"backend {block.backend}, enforce {block.enforce}, "
                   f"min_distance {block.min_distance_mm:.1f} mm, "
                   f"{len(block.fixtures)} fixture(s)")

        with run.step("which robot the bundle is keyed on") as report:
            # Not `ur.model`. The guard keys its mesh bundle on `self_collision.kinematics_model`
            # when that is set and falls back to the vendor block, and `MotionStack` resolves the
            # same ladder and records which key answered. A reading about ur5e on a UR3e cell is
            # a green light for a robot nobody configured.
            stack = MotionStack.from_robot_config(config)
            report(f"{stack.model}, from {stack.model_source}")

        with run.step("what resolves on this box") as report:
            probe = stack.probe()
            report("engine "
                   f"{probe.environment.collision.engine or 'none (capsule only)'}, "
                   f"{stack.model} bundle "
                   f"{'present' if probe.environment.collision.mesh_bundle_present else 'MISSING'}")
        for line in probe.render().splitlines():
            run.note(line)

        run.note("")
        run.note("Which models have a committed bundle. A bundle is per robot, so a present ur5e")
        run.note("bundle says nothing about a UR3e cell, and a model starts using exact meshes as")
        run.note("soon as its own file lands, with no code change:")
        for model in sorted(UR_JOINT_LIMITS_DEG):
            status = probe_collision_engine(model)
            run.note(f"  {model:<7} {'bundle present' if status.mesh_bundle_present else '-'}")

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

        folded = SafetyContext(
            command=MotionCommand.MOVE_JOINTS, target_joints=JointPositions(list(_FOLDED)), arm=arm)

        def timed(guard: SelfCollisionGuard, ctx: SafetyContext) -> tuple[SafetyDecision, float]:
            """One evaluation, and what it cost in milliseconds on this box."""
            started = time.perf_counter()
            decision = guard.evaluate(ctx)
            return decision, 1000.0 * (time.perf_counter() - started)

        run.note("")
        run.note("The same configuration, both backends. This is the context `gate_joint_target`")
        run.note("builds: every commanded joint move, and the final configuration of a plan.")
        verdicts: dict[str, SafetyDecision] = {}
        for backend in ("capsule", "fcl"):
            guard = SelfCollisionGuard(block.model_copy(update={"backend": backend}))
            with run.step(f"backend '{backend}' on a folded arm") as report:
                first, first_ms = timed(guard, folded)
                # The second call is the steady-state cost. The first pays for the BVH build,
                # once per guard instance, and a driver builds one guard per cell.
                again, again_ms = timed(guard, folded)
                verdicts[backend] = again
                report(f"{'ACCEPTED' if again.accepted else again.reason.value}, "
                       f"first {first_ms:.1f} ms, then {again_ms:.2f} ms")
            if not first.accepted:
                run.note(f"       {first.message}")

        if verdicts["capsule"].accepted and not verdicts["fcl"].accepted:
            run.note("")
            run.note("That is the fork, in one line each. The proxy accepted a gripper finger")
            run.note("inside the forearm, because a joint-only context carries no pose to root a")
            run.note("tool capsule at, and because it skips the link pairs a capsule of realistic")
            run.note("radius cannot separate. The exact path measured the meshes and refused.")
        elif verdicts["fcl"].accepted:
            # Which of the three gates stopped it matters, and they are not interchangeable: a
            # missing bundle is a bake, a missing engine is an install, and an arm with no DH
            # chain is a cell that can never run this backend however it is configured.
            engine = probe_collision_engine(stack.model)
            if block.kinematics_model is None and arm.capabilities.vendor != "ur":
                why = (f"this arm reports vendor '{arm.capabilities.vendor}' and "
                       "self_collision.kinematics_model is unset, so the guard has no DH chain to "
                       "place link meshes on")
            elif engine.engine is None:
                why = "no collision engine imports on this box, neither Coal nor python-fcl"
            elif not engine.mesh_bundle_present:
                why = f"no {stack.model} mesh bundle ships in src/robot/safety/data"
            else:
                why = "the exact path ran and found this configuration clear"
            run.note("")
            run.note("Both accepted. The reason the two agree:")
            for line in textwrap.wrap(why, width=74):
                run.note(f"  {line}")

        run.note("")
        run.note("Now the failure that does not refuse. Ask for `fcl` on a model with no bundle,")
        run.note("which is what a cell running an arm nobody baked meshes for looks like:")
        missing = next((m for m in sorted(UR_JOINT_LIMITS_DEG)
                        if not probe_collision_engine(m).mesh_bundle_present), "")
        if not missing:
            run.finding("no model without a bundle",
                        "every UR model in the table ships one on this checkout, so the fallback "
                        "cannot be triggered honestly here")
        else:
            with run.step(f"backend 'fcl' for model '{missing}'") as report:
                degraded = SelfCollisionGuard(block.model_copy(
                    update={"backend": "fcl", "kinematics_model": missing}))
                with _warnings_from("src.robot.safety") as logged:
                    decision, _ms = timed(degraded, folded)
                report(f"{'ACCEPTED' if decision.accepted else decision.reason.value}, "
                       f"reason {decision.reason.value}, {len(logged)} warning(s) logged")
            for message in logged:
                run.note("")
                for line in textwrap.wrap(message, width=74):
                    run.note(f"    | {line}")
            run.note("")
            if logged:
                run.note("Those lines are the whole difference. The return value says `ok`, so")
                run.note("nothing a caller can branch on tells this cell apart from one with an")
                run.note("exact-mesh guard, and the reading to trust at boot is the probe rather")
                run.note("than the config key.")
            else:
                run.note("No warning was logged, so the guard did not even reach its backend: on")
                run.note("this arm the exact path returns before it looks for a bundle.")

        # The other half of the fork, and the one that surprises people the other way round: with
        # no joints in the context there is no exact path to run at all, whatever the key says.
        if block.fixtures:
            fixture = block.fixtures[0]
            top_mm = float(fixture.center_mm[2]) + float(fixture.half_extents_mm[2])
            # Half the tool radius above the fixture's top face: a plausible grasp centre on a part
            # standing on it, and close enough that the bounding cylinder still reaches through.
            height_mm = top_mm + 0.5 * float(block.tool_radius_mm)
            over = Pose(
                position_mm=np.array([float(fixture.center_mm[0]), float(fixture.center_mm[1]),
                                      height_mm], dtype=np.float64),
                # Top-down: the tool z axis points at the fixture, so the tool capsule runs from
                # the grasp centre back up toward the flange, which is where the tool material is.
                quaternion_xyzw=from_axis_angle(np.array([np.pi, 0.0, 0.0])),
                frame=Frame.BASE, label="a top-down grasp")
            pose_only = SafetyContext(command=MotionCommand.MOVE_TO, target_pose=over, arm=arm)
            run.note("")
            run.note(f"A Cartesian command with no IK solution yet, over fixture "
                     f"'{fixture.name}' at z {height_mm:.0f} mm:")
            for backend in ("capsule", "fcl"):
                guard = SelfCollisionGuard(block.model_copy(update={"backend": backend}))
                answer = guard.evaluate(pose_only)
                run.note(f"  {backend:<8} {'ACCEPTED' if answer.accepted else answer.message}")
            run.note("The two answers are identical because only one path ran. `_evaluate_fcl`")
            run.note("returns None without `target_joints`, so `backend: fcl` buys nothing on a")
            run.note("pose whose IK the driver has not resolved yet, and what refuses there is the")
            run.note(f"{block.tool_radius_mm:.0f} mm bounding cylinder of the proxy.")
        else:
            run.note("")
            run.note("This tree declares no fixtures, so the second half of the fork, what the")
            run.note("proxy over-rejects against a bench, cannot be shown. 32 is that decision.")

        run.note("")
        run.note("What to choose. Keep `fcl`, which is the shipped default, and read the probe")
        run.note("above at boot rather than the config key: the key states an intention and the")
        run.note("probe states what will run. Choose `capsule` only where no engine can be")
        run.note("installed, and then know that the arm-against-arm coverage you have is the one")
        run.note("printed in this run, not the one in the key.")
        run.note("")
        run.note("Next: 32_declare_your_bench.py, the same guard on a path instead of a pose.")

        if verdicts["fcl"].accepted and block.backend == "fcl":
            # The configured backend could not be exercised on this cell: worth a non-zero status,
            # because a run that proves nothing about the guard should not read as a pass.
            return EXIT_FAILED
        return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
