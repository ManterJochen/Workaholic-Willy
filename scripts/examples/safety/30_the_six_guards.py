"""30: the six guards, each one made to refuse on purpose.

    python scripts/examples/safety/30_the_six_guards.py
    python scripts/examples/safety/30_the_six_guards.py --profile ur5e

The decision this covers is which safety families your cell enforces. `robot.safety` carries six
blocks and each one has an `enforce` flag. `enforce: false` does not quieten a guard, it leaves the
guard out of the pipeline at construction, so that family is never evaluated again and the
difference is invisible in every log: a cell with self-collision off and a cell with it on both
report a working pipeline. Only `omitted_guards` names the difference, and this file prints it.

Nothing here connects to anything. It builds the driver your config names, asks that driver for its
own `SafetyPreflight`, and hands the pipeline the contexts a real move would produce. The guards
are asked; the controller is not. There is no `--live`, because there is nothing to drive: a
guard's answer is a return value, not a motion.

The six run in a fixed order, workspace, joint_limit, ik_quality, self_collision, payload,
motion_continuity, and the pipeline stops at the first rejection. A command that is wrong in three
ways is reported as wrong in one, and the guard named is the earliest, not the worst. That is why
each guard below is evaluated alone: it is the only way to see what the other five would have said.

What a guard reads matters as much as what it decides. `joint_limit`, `ik_quality` and the
arm-against-arm half of `self_collision` read `ctx.target_joints`, and a Cartesian command carries
none until the driver has resolved IK. Handed a context without them those guards return
UNAVAILABLE, which is a refusal and not a pass: with `enforce: true` the preflight fails closed and
the caller is told CONTROLLER_REJECTED. `URRobotArm._gate_pose` pre-resolves IK for exactly this
reason and can only do it against a connected controller, so at a desk two of the six answer
UNAVAILABLE on every Cartesian pose. That is the correct answer, and this file prints it rather
than arranging for it not to happen.

The violations are built from your own configuration rather than from constants written here: the
workspace one is your box plus its margin, the joint one is your resolved limit table plus ten
degrees, the step is your `max_tcp_step_mm` doubled, and the payload is your ceiling plus 2.5 kg.
Point this at another cell and the numbers move with it. Self-collision is the exception and says
why below: a fold is a configuration of an arm, not a number in a file.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import EXIT_FAILED, EXIT_OK, Example, not_ready  # noqa: E402

EPILOG = """
exit codes
  0  every wired guard refused the violation it was handed
  1  a guard was handed a violation of its own family and accepted it
  2  the arm this config names carries no guard pipeline to interrogate
"""

#: Joint configurations tried, in order, until one folds the arm into itself. A self-collision is a
#: configuration of an arm rather than a number, so it cannot be derived from the config the way
#: the other five violations are. Both entries fold on the bundled UR collision meshes: the first
#: drives a finger into the forearm, the second is the flipped IK branch that puts the whole
#: gripper inside it. On the capsule backend both are accepted, which is the decision 31 covers.
_FOLDED_CANDIDATES: tuple[tuple[str, tuple[float, ...]], ...] = (
    ("a finger driven into the forearm", (1.95, 0.38, -1.33, -0.55, 2.00, 0.79)),
    ("the flipped IK branch", (3.011, -2.227, -1.915, 0.994, -3.010, -0.005)),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__ or "", epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--profile", default=None,
        help="WILLY_PROFILE to load (e.g. `ur5e`, `ur3e`, `ursim`). Default: the environment's.")
    args = parser.parse_args(argv)

    # Imported after the parse, so `--help` answers in a checkout whose dependencies are not
    # installed yet.
    from src.config import ConfigError, load_robot_config
    from src.geometry import Frame, Pose
    from src.geometry.quaternion import IDENTITY_QUAT_XYZW
    from src.robot.core import JointPositions, MotionCommand, RobotVendor
    from src.robot.drivers import create_arm
    from src.robot.safety import (
        PayloadGuard,
        SafetyAttestation,
        SafetyContext,
        SafetyDecision,
        SafetyGated,
        SafetyPreflight,
        SafetyReason,
        resolve_joint_limits_deg,
    )

    with Example("30 the_six_guards", "make every guard refuse, on purpose",
                 hardware=False, profile=args.profile) as run:
        run.note("It builds the driver your config names and never connects it. What answers")
        run.note("below is the guard pipeline, not the controller.")
        run.note("")

        with run.step("load the config tree") as report:
            try:
                config = load_robot_config()
            except ConfigError as error:
                return not_ready(f"the config tree has no cell to check ({error})",
                                 "select a profile that has a `robot` block: WILLY_PROFILE=ur5e")
            report(f"vendor {config.vendor}, workspace x {config.workspace_limits.x_min:.0f} to "
                   f"{config.workspace_limits.x_max:.0f} mm")

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
            report(f"{type(arm).__name__}, {arm.capabilities.vendor} {arm.capabilities.model}, "
                   f"{arm.capabilities.dof} axes, connected={arm.is_connected}")

        with run.step("what this arm says it will refuse") as report:
            attestation = SafetyAttestation.of(arm)
            report(f"{attestation.posture.value.upper()}, {len(attestation.guards)} guard(s)")
        for line in attestation.render().splitlines():
            run.note(line)

        # `SafetyGated` is a capability, not part of `RobotArm`: a driver opts in by exposing
        # `safety_preflight`, and one that does not is UNSTATED rather than safe. Asking through
        # the Protocol is how a caller reads the pipeline without knowing which vendor it has.
        preflight = arm.safety_preflight if isinstance(arm, SafetyGated) else None
        if preflight is None or not attestation.enforced:
            # The honest stop. A dummy or sim arm states that nothing gates its motion, so there is
            # no pipeline here to interrogate, and inventing one would demonstrate a cell nobody
            # has.
            return not_ready(
                f"{attestation.arm} reports posture {attestation.posture.value}, so this cell has "
                "no guard pipeline to show",
                "point this at a cell whose driver carries one: WILLY_PROFILE=ur5e, or ur3e")

        guards = {guard.name: guard for guard in preflight.guards}
        run.note("")

        box = config.workspace_limits
        margin_mm = float(config.safety.limits.workspace_margin_mm)
        centre = ((box.x_min + box.x_max) / 2.0, (box.y_min + box.y_max) / 2.0,
                  (box.z_min + box.z_max) / 2.0)

        def pose_at(x: float, y: float, z: float, label: str) -> Pose:
            return Pose(position_mm=np.array([x, y, z], dtype=np.float64),
                        quaternion_xyzw=IDENTITY_QUAT_XYZW, frame=Frame.BASE, label=label)

        def joint_ctx(joints: JointPositions,
                      current: JointPositions | None = None) -> SafetyContext:
            return SafetyContext(command=MotionCommand.MOVE_JOINTS, target_joints=joints,
                                 current_joints=current, arm=arm)

        def pose_ctx(pose: Pose, previous: Pose | None = None) -> SafetyContext:
            return SafetyContext(command=MotionCommand.MOVE_TO, target_pose=pose,
                                 last_target_pose=previous, arm=arm)

        # One axis past the box, and past the margin the preflight shrinks it by. The guard owns a
        # shrunk copy, so the number in the YAML is not the number that refuses; the guard's own
        # log line prints the shrunk bounds it used.
        outside = pose_at(box.x_max + margin_mm + 100.0, centre[1], centre[2], "past the box")

        # The same table `JointLimitGuard` will consult: the operator's static lists first, then
        # the built-in vendor table. `None` means neither answered, and the guard is UNAVAILABLE.
        limits = resolve_joint_limits_deg(
            config.safety.joint_limits,
            vendor=arm.capabilities.vendor, model=arm.capabilities.model)
        beyond_limit: JointPositions | None = None
        if limits is not None:
            values = [0.0] * arm.capabilities.dof
            values[0] = math.radians(limits[1][0] + 10.0)
            beyond_limit = JointPositions(values)

        # A jump larger than `ik_quality.max_jump_rad` on one axis: the wrap-around signature,
        # where the same TCP pose is reached by slamming one axis most of a turn around.
        dof = arm.capabilities.dof
        here = JointPositions([0.0] * dof)
        there = JointPositions(
            [float(config.safety.ik_quality.max_jump_rad) + 1.0] + [0.0] * (dof - 1))

        step_mm = float(config.safety.motion_continuity.max_tcp_step_mm)
        was = pose_at(centre[0], centre[1], centre[2], "the last accepted target")
        now = pose_at(centre[0] + 2.0 * step_mm, centre[1], centre[2], "the next one")

        # The pipeline's own payload guard holds the operator's block, and the schema already
        # refused an over-ceiling mass at load, so it can only accept. `model_copy` skips
        # validation, which is the case this guard exists for: its evaluate() states that the three
        # runtime checks are a second line for a config assembled by a path that bypassed the
        # schema, such as a grasping preset overlay. This is that path, deliberately.
        overweight = PayloadGuard(config.safety.payload.model_copy(
            update={"mass_kg": float(config.safety.payload.max_mass_kg) + 2.5}))

        cases: list[tuple[str, str, SafetyReason, Callable[[], SafetyDecision]]] = [
            ("workspace", "reads target_pose", SafetyReason.WORKSPACE,
             lambda: guards["workspace"].evaluate(pose_ctx(outside))),
            ("ik_quality", "reads target vs current joints", SafetyReason.IK_QUALITY,
             lambda: guards["ik_quality"].evaluate(joint_ctx(there, here))),
            ("payload", "reads the payload block", SafetyReason.PAYLOAD,
             lambda: overweight.evaluate(joint_ctx(here))),
            ("motion_continuity", "reads target_pose vs the memo", SafetyReason.CONTINUITY,
             lambda: guards["motion_continuity"].evaluate(pose_ctx(now, previous=was))),
        ]
        if beyond_limit is not None:
            cases.insert(1, ("joint_limit", "reads target_joints", SafetyReason.JOINT_LIMIT,
                             lambda: guards["joint_limit"].evaluate(joint_ctx(beyond_limit))))

        refused = 0
        accepted: list[str] = []
        for name, reads, expected, evaluate in cases:
            if name not in guards:
                # The family is out of the pipeline, which is precisely what `enforce: false` does.
                run.finding(f"{name} is not wired",
                            f"safety.{name}.enforce is false, so this family never runs")
                continue
            with run.step(f"{name}: {reads}") as report:
                decision = evaluate()
                report("ACCEPTED" if decision.accepted
                       else f"{decision.reason.value} -> {decision.motion_status}")
            if decision.accepted:
                accepted.append(name)
            else:
                refused += 1
                run.note(f"       {decision.message}")
                if decision.reason is not expected:
                    run.finding(f"{name} reason",
                                f"expected {expected.value}, got {decision.reason.value}")

        folded_by = ""
        if "self_collision" not in guards:
            run.finding("self_collision is not wired",
                        "safety.self_collision.enforce is false, so this family never runs")
        else:
            with run.step("self_collision: reads joints and arm") as report:
                for label, candidate in _FOLDED_CANDIDATES:
                    decision = guards["self_collision"].evaluate(
                        joint_ctx(JointPositions(list(candidate))))
                    if not decision.accepted:
                        folded_by = label
                        break
                report(f"{decision.reason.value} -> {decision.motion_status}" if folded_by
                       else "ACCEPTED both candidates")
            if folded_by:
                refused += 1
                run.note(f"       {folded_by}")
                run.note(f"       {decision.message}")
            else:
                # Not counted as a guard that failed. On the capsule backend this is the documented
                # answer rather than a broken cell: the proxy has no tool on a joint-only context
                # and skips the wrist pairs. 31 is that decision, with both backends measured.
                run.finding("self_collision accepted both folds",
                            f"expected on backend '{config.safety.self_collision.backend}' without "
                            "a mesh bundle; 31_capsule_or_mesh.py measures both")

        run.note("")
        run.note("What a guard reads, when the context does not carry it. A Cartesian command has")
        run.note("no joint solution until the driver resolves IK against a connected controller,")
        run.note("and two guards read nothing else:")
        inside = pose_at(centre[0], centre[1], centre[2], "well inside the box")
        for name in ("joint_limit", "ik_quality"):
            if name in guards:
                blind = guards[name].evaluate(pose_ctx(inside))
                run.note(f"  {name:<18} {blind.reason.value} -> {blind.motion_status}")
                run.note(f"  {'':<18} {blind.message}")
        run.note("UNAVAILABLE is a refusal and not a pass. That is the fail-closed shape: a guard")
        run.note("that cannot decide does not wave the motion through.")

        run.note("")
        run.note("The pipeline stops at the first rejection, so a command wrong in two ways is")
        run.note("reported as wrong in one:")
        first = preflight.evaluate(preflight.context_for_pose(
            outside, target_joints=beyond_limit, current_joints=here, arm=arm))
        run.note(f"  a pose past the box and an axis past its limit -> {first.guard}/"
                 f"{first.reason.value}")
        run.note("  The joint violation is still in that command. Nothing looked at it.")

        run.note("")
        run.note("What switching one off costs. `enforce: false` removes the family at")
        run.note("construction, so there is no longer a guard of that family to ask:")
        without = config.safety.model_copy(update={
            "self_collision": config.safety.self_collision.model_copy(update={"enforce": False})})
        thinner = SafetyPreflight.from_safety_config(without, config.workspace_limits)
        run.note(f"  guards now:   {', '.join(thinner.guard_names)}")
        run.note(f"  not enforced: {', '.join(thinner.omitted_guards) or '(none)'}")
        folded_ctx = thinner.context_for_joints(
            JointPositions(list(_FOLDED_CANDIDATES[0][1])), arm=arm)
        answers = [g.evaluate(folded_ctx) for g in thinner.guards]
        collision = [d for d in answers if d.reason is SafetyReason.SELF_COLLISION]
        run.note(f"  the fold above now draws {len(collision)} self-collision verdict(s) from "
                 f"{len(thinner.guards)} guard(s)")
        instead = thinner.evaluate(folded_ctx)
        run.note(f"  what this thinner pipeline answers instead: {instead.guard}/"
                 f"{instead.reason.value}")
        run.note("`omitted_guards` is the line worth printing in a boot banner. A guard count")
        run.note("reads as complete, and a pipeline that still refuses things reads as working;")
        run.note("only the absence names the family that was switched off.")

        run.note("")
        run.note(f"{refused} guard(s) refused the violation they were handed.")
        if accepted:
            run.note(f"{', '.join(accepted)} accepted a violation of its own family, which is why")
            run.note("this run exits 1. Read the block that guard belongs to in your robot.yaml.")
            return EXIT_FAILED
        run.note("Next: 31_capsule_or_mesh.py, which self-collision backend answered above, and")
        run.note("32_declare_your_bench.py, the path between two checked endpoints.")
        return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
