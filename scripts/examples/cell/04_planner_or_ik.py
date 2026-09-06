"""04: how the arm gets from where it is to the grasp. `robot.ur.motion_planner`: curobo or ik.

    python scripts/examples/cell/04_planner_or_ik.py               # read the paths, spawn nothing
    python scripts/examples/cell/04_planner_or_ik.py --doctor       # load every engine, slower
    python scripts/examples/cell/04_planner_or_ik.py --profile ursim

Two answers, and they are not two qualities of the same thing.

`ik` is the controller's straight line. It resolves one inverse-kinematics solution for the target,
runs the static guards over that one configuration, and drives there. It needs nothing installed
and it knows nothing about the bench, the bin, the fixture or the camera arm: those are not in its
model, so it will drive through all of them.

`curobo` plans a collision-free trajectory in an isolated planner process and executes it waypoint
by waypoint. It needs a second Python environment holding cuRobo and torch, plus a robot descriptor
built inside that environment, and what it knows about your cell is exactly what
`safety.planning_world` declares. With that block off it plans against its own robot model and a
generic table.

The cost of the second answer is the one to read twice: it is fail-closed. Where the cuRobo
environment is not there, the UR driver refuses every move with `CONTROLLER_REJECTED` and never
degrades to blind IK, so a cell configured for a planner it does not have does not run badly, it
does not run. This example triggers that refusal from its real source. The simulator driver makes
the opposite choice, degrades to blind IK and reports `curobo_degraded`, which is why a pick rate
measured in the simulator has to be read next to that flag.

Nothing here moves an arm, at any flag. `--doctor` spawns the planner's own interpreter to see what
resolves inside it, which costs a subprocess and, on a box that has one, tens of seconds.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import EXIT_FAILED, EXIT_OK, Example, not_ready  # noqa: E402

EPILOG = """
exit codes
  0  the planner this config names is anchored on this box (or is `ik`, which needs nothing)
  1  the config asks for `curobo` and this box cannot provide it: every move would be refused
  2  nothing to read: the config tree has no `robot` block
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "", epilog=EPILOG,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", default=None,
                        help="WILLY_PROFILE to read the cell from (e.g. `ursim`, `ur3e`)")
    parser.add_argument("--doctor", action="store_true",
                        help="load every engine instead of reading paths: it imports the collision "
                             "engine, runs one distance query, and spawns the planner's own "
                             "interpreter to ask what resolves there")
    args = parser.parse_args(argv)

    from src.config.loader import load_config
    from src.robot.safety.planning.curobo_client import (
        CuroboPlanClient,
        CuroboUnavailableError,
        curobo_env_available,
    )
    from src.robot.safety.planning.stack import MotionStack

    with Example("04 planner_or_ik", "which planner this cell would use, and whether it is here",
                 hardware=False, profile=args.profile) as run:
        with run.step("load the config tree") as report:
            robot = load_config().robot
            report("no `robot` block in this tree" if robot is None else
                   f"planner {robot.ur.motion_planner}, model {robot.ur.model}")
        if robot is None:
            return not_ready("this config tree has no `robot` block",
                             "point the loader at a tree that has one; `config/robot/robot.yaml` "
                             "in this repository is one")
        planner = str(robot.ur.motion_planner)

        # The model is not decoration on this reading. The mesh bundle ships as
        # `{model}_collision_meshes.npz` and the descriptor as `{model}.yml`, so a present ur5e
        # bundle says nothing about a UR3e cell, and `MotionStack` carries which key decided.
        stack = MotionStack.from_robot_config(robot)
        with run.step("which robot this reading is about") as report:
            report(f"{stack.model}, from {stack.model_source}")

        with run.step("are both engines anchored here") as report:
            reading = stack.probe()
            report("fully anchored" if reading.fully_anchored
                   else "partially anchored, a degraded fallback is in force")
        for line in reading.render().splitlines():
            run.note(line)
        run.note("Two engines, and only one of them is this decision. The exact-mesh collision")
        run.note("engine backs the self-collision guard whichever planner runs; the cuRobo row")
        run.note("matters only where the planner is `curobo`.")
        run.note("")

        if args.doctor:
            # `probe()` reads paths; this loads. A present interpreter can still fail to import,
            # and on Windows an application-control policy can refuse a correct, unmodified binary
            # outright, which presents as a pick rate of zero with no error naming a planner.
            from src.robot.safety.planning.doctor import run_doctor

            with run.step("load every engine") as report:
                report_doctor = run_doctor(model=stack.model,
                                           robot_config=f"{stack.model}.yml")
                report(f"{sum(1 for p in report_doctor.probes if p.ok)} of "
                       f"{len(report_doctor.probes)} probe(s) ok")
            for line in report_doctor.render().splitlines():
                run.note(line)
            if any("sidecar" in p.name and not p.ok for p in report_doctor.probes):
                # The remedy printed above is written for the other driver, and a reader who takes
                # it at face value expects a cell that keeps running.
                run.note("That remedy says the driver plans with blind IK. It is describing the")
                run.note("simulator arm, which does degrade. A UR refuses instead, as below.")
            if report_doctor.blocked:
                run.finding("code integrity", "an OS policy refused a binary that is present and "
                                              "intact; re-downloading does not fix that")
            run.note("")

        # What the planner would be given. Both answers are read off the same config the driver
        # reads, and the second one is what an operator forgets: a declared world is only consulted
        # by a planner, so `ik` plus a fully declared bench is a bench nothing knows about.
        world = robot.safety.planning_world
        fixtures = tuple(robot.safety.self_collision.fixtures or ())
        with run.step("what the planner would know about this cell") as report:
            report(f"planning_world enabled {world.enabled}, "
                   f"{len(fixtures)} declared fixture(s), "
                   f"include_fixtures {world.include_fixtures}")
        if planner == "curobo" and not world.enabled and not fixtures:
            run.note("Nothing declares your bench, bin or fixtures, so the planner routes around")
            run.note("its own robot model and a generic table only. Declare them under")
            run.note("`safety.planning_world` and `safety.self_collision.fixtures`.")
        if planner == "ik" and (world.enabled or fixtures):
            run.finding("planning world", "declared, and `ik` reads none of it; only the planner "
                                          "consults a world")

        trajectory = robot.safety.trajectory_check
        with run.step("what gates the path between the endpoints") as report:
            report(f"trajectory_check enabled {trajectory.enabled}, stride {trajectory.stride}")
        if not trajectory.enabled:
            # True for both answers, and worth saying under this decision because a planned path
            # invites the assumption that something judged all of it. The guard judges the
            # configuration the move ends in.
            run.note("Off. A multi-waypoint path executes unexamined between its endpoints: the")
            run.note("static guards judge the final configuration, so a path that grazes a bin")
            run.note("wall halfway and lands clear passes every check there is.")
        run.note("")

        if planner == "ik":
            with run.step("what `ik` needs") as report:
                report("nothing beyond this repository; the controller resolves the line")
            run.note("It never refuses a move for want of a planner. What it refuses is what the")
            run.note("static guards catch on the one configuration it computed: the workspace box,")
            run.note("joint limits, IK quality, self-collision and payload.")
            run.note("")
            run.note("Next: 03_first_pick.py runs one grasp with whatever this decided.")
            return run.exit_code

        # The `curobo` half. Where the environment is absent, the refusal is triggered here from
        # the same client the driver calls, rather than described: `CuroboPlanClient.start()` is
        # what `_drive_curobo` reaches through, and the `CuroboUnavailableError` it raises is what
        # the UR driver turns into `MotionStatus.CONTROLLER_REJECTED` for every commanded move.
        if curobo_env_available():
            with run.step("the planner environment") as report:
                report("present; --doctor is what boots it and checks the descriptor inside it")
            run.note("A present interpreter is not a working planner: the robot descriptor lives")
            run.note("inside that environment and only a process running there can confirm it.")
            run.note("")
            run.note("Next: 03_first_pick.py runs one grasp with whatever this decided.")
            return run.exit_code

        with run.step("what a move does without the planner") as report:
            client = CuroboPlanClient()
            try:
                client.start()
            except CuroboUnavailableError as error:
                report(f"CONTROLLER_REJECTED: {error}")
            else:
                # Reached where the interpreter appeared between the check above and this call.
                client.close()
                report("the planner started after all; nothing would be refused")
        run.note("That refusal is the whole motion path for this cell. It is not a warning and it")
        run.note("is not a fallback: `ik` is a different value of a config key, and the driver")
        run.note("will not substitute it for you.")
        run.note("")
        run.note("Two ways forward. Install the environment (ext_deps/README.md) and confirm it")
        run.note("with --doctor, or set `robot.ur.motion_planner: ik` and accept that nothing")
        run.note("plans around the cell. The shipped `ursim` profile takes the second one on")
        run.note("purpose: it exercises the driver against a controller, not the planner.")
        # Not `run.exit_code`. Every step above completed, so it would answer 0 for a cell whose
        # every move is refused, and whether this box can drive the planner it names is the
        # question this example is asked.
        return EXIT_FAILED if planner == "curobo" else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
