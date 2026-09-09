"""Does every safety guard in THIS cell refuse a violation of its own family, for the right reason?

    python scripts/checks/safety_guards.py

⛔ **THE PYTEST SUITE DOES NOT ANSWER THIS.** `test_safety_workspace`, `_joint_limits`, `_payload`,
`_ik_quality` and `_continuity` all exist and all pass, and every one of them runs against a fixture.
None of them loads the tree this box loads, so none of them can tell you that YOUR workspace box,
YOUR joint-limit table and YOUR payload ceiling produce a refusal. A guard family whose `enforce` is
false is silently absent from the pipeline, and that is a config decision no fixture can see.

Each violation below is derived from the operator's own YAML rather than hardcoded: the workspace
box plus `safety.limits.workspace_margin_mm`, the resolved joint-limit table plus 10 degrees,
`ik_quality.max_jump_rad` plus 1.0, `motion_continuity.max_tcp_step_mm` doubled, and
`payload.max_mass_kg` plus 2.5 kg. So a cell that widens a limit is checked against its own new
limit, and the check cannot go stale against a config it has never seen.

Exit codes: 0 every wired guard refused for its own reason, 1 a guard accepted a violation of its
own family or refused with the wrong reason, 2 this cell has no guard pipeline to interrogate.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from src.config import ConfigError, load_robot_config  # noqa: E402
from src.geometry import Frame, Pose  # noqa: E402
from src.geometry.quaternion import IDENTITY_QUAT_XYZW  # noqa: E402
from src.robot.core import JointPositions, MotionCommand, RobotVendor  # noqa: E402
from src.robot.drivers import create_arm  # noqa: E402
from src.robot.safety import (  # noqa: E402
    PayloadGuard,
    SafetyGuard,
    SafetyAttestation,
    SafetyContext,
    SafetyGated,
    SafetyReason,
    resolve_joint_limits_deg,
)

EXIT_OK, EXIT_FAILED, EXIT_NOT_READY = 0, 1, 2


def _not_ready(what: str, fix: str) -> int:
    print(f"NOT READY: {what}\n  fix: {fix}")
    return EXIT_NOT_READY


def main() -> int:
    try:
        config = load_robot_config()
    except ConfigError as error:
        return _not_ready(f"the config tree has no cell to check ({error})",
                          "select a profile that has a `robot` block: WILLY_PROFILE=ur5e")

    try:
        arm = create_arm(RobotVendor.from_string(config.vendor), config=config)
    except Exception as error:  # noqa: BLE001  (report, not raise)
        return _not_ready(
            f"the `{config.vendor}` arm could not be built ({type(error).__name__}: {error})",
            "this reads the guard pipeline of a real cell's driver: WILLY_PROFILE=ur5e")

    attestation = SafetyAttestation.of(arm)
    print(attestation.render())

    # `SafetyGated` is a capability, not part of `RobotArm`: a driver opts in by exposing
    # `safety_preflight`, and one that does not is UNSTATED rather than safe.
    preflight = arm.safety_preflight if isinstance(arm, SafetyGated) else None
    if preflight is None or not attestation.enforced:
        return _not_ready(
            f"{attestation.arm} reports posture {attestation.posture.value}, so there is no guard "
            "pipeline here to interrogate",
            "point this at a cell whose driver carries one: WILLY_PROFILE=ur5e, or ur3e")

    guards = {guard.name: guard for guard in preflight.guards}
    box = config.workspace_limits
    centre = ((box.x_min + box.x_max) / 2.0, (box.y_min + box.y_max) / 2.0,
              (box.z_min + box.z_max) / 2.0)
    dof = arm.capabilities.dof

    def pose(x: float, y: float, z: float, label: str) -> Pose:
        return Pose(position_mm=np.array([x, y, z], dtype=np.float64),
                    quaternion_xyzw=IDENTITY_QUAT_XYZW, frame=Frame.BASE, label=label)

    def joints(target: JointPositions, current: JointPositions | None = None) -> SafetyContext:
        return SafetyContext(command=MotionCommand.MOVE_JOINTS, target_joints=target,
                             current_joints=current, arm=arm)

    def moving(target: Pose, previous: Pose | None = None) -> SafetyContext:
        return SafetyContext(command=MotionCommand.MOVE_TO, target_pose=target,
                             last_target_pose=previous, arm=arm)

    # One axis past the box AND past the margin the preflight shrinks it by. The guard owns a
    # shrunk copy, so the number in the YAML is not the number that refuses.
    margin = float(config.safety.limits.workspace_margin_mm)
    here = JointPositions([0.0] * dof)
    step = float(config.safety.motion_continuity.max_tcp_step_mm)

    # The payload guard is built here rather than taken from the pipeline, and that is the point of
    # it: the schema already refuses an over-ceiling mass at load, so the guard in the pipeline can
    # only ever accept. `model_copy` skips validation, which is exactly the path this guard exists
    # to catch, a config assembled by something that bypassed the schema such as a preset overlay.
    overweight = PayloadGuard(config.safety.payload.model_copy(
        update={"mass_kg": float(config.safety.payload.max_mass_kg) + 2.5}))

    #: (name, expected reason, context, guard to ask or None for "the one in the pipeline").
    cases: list[tuple[str, SafetyReason, SafetyContext, SafetyGuard | None]] = [
        ("workspace", SafetyReason.WORKSPACE,
         moving(pose(box.x_max + margin + 100.0, centre[1], centre[2], "past the box")), None),
        ("ik_quality", SafetyReason.IK_QUALITY,
         joints(JointPositions([float(config.safety.ik_quality.max_jump_rad) + 1.0]
                               + [0.0] * (dof - 1)), here), None),
        ("payload", SafetyReason.PAYLOAD, joints(here), overweight),
        ("motion_continuity", SafetyReason.CONTINUITY,
         moving(pose(centre[0] + 2.0 * step, centre[1], centre[2], "the next one"),
                previous=pose(*centre, "the last accepted target")), None),
    ]

    # The same table `JointLimitGuard` consults: the operator's static lists first, then the
    # built-in vendor table. `None` means neither answered and the guard is UNAVAILABLE, which is
    # a legitimate state rather than a failure.
    limits = resolve_joint_limits_deg(
        config.safety.joint_limits, vendor=arm.capabilities.vendor, model=arm.capabilities.model)
    if limits is not None:
        values = [0.0] * dof
        values[0] = math.radians(limits[1][0] + 10.0)
        cases.insert(1, ("joint_limit", SafetyReason.JOINT_LIMIT,
                         joints(JointPositions(values)), None))

    wrong: list[str] = []
    for name, expected, context, own in cases:
        guard = own if own is not None else guards.get(name)
        if guard is None:
            # `enforce: false` removes the family from the pipeline entirely. Reported, not failed:
            # it is a decision the operator made and this check's job is to make it visible.
            print(f"  {name:20s} NOT WIRED   safety.{name}.enforce is false")
            continue
        decision = guard.evaluate(context)
        if decision.accepted:
            wrong.append(f"{name} ACCEPTED a violation of its own family")
            print(f"  {name:20s} ACCEPTED    <- this guard is not doing its job")
        elif decision.reason is not expected:
            wrong.append(f"{name} refused as {decision.reason.value}, expected {expected.value}")
            print(f"  {name:20s} {decision.reason.value:11s} <- expected {expected.value}")
        else:
            print(f"  {name:20s} {decision.reason.value:11s} -> {decision.motion_status}")

    if wrong:
        print(f"\nFAILED: {len(wrong)} guard(s) did not behave as their own family requires")
        for line in wrong:
            print(f"  {line}")
        return EXIT_FAILED
    print(f"\nOK: every wired guard refused its own violation ({len(cases)} checked)")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
