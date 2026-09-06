"""12: a second camera, and the two places a two-camera cell is wired wrong.

    python scripts/examples/calibration/12_two_cameras.py                  # the shipped eth2 cell
    python scripts/examples/calibration/12_two_cameras.py --profile ur5e,eth2
    python scripts/examples/calibration/12_two_cameras.py --profile my_cell

The decision is how many cameras look into the bin. One camera sees one side of a part, and the
finger placement on the side it never measured is a guess; a second view measured in the same frame
turns that guess into geometry. Fusing a second view is the largest measured lever in this stack,
and the size of the gain belongs to the corpus it was measured on rather than to your bin, so no
number for it appears here.

What the second camera costs is concrete: another device on the USB controller, another calibration
sweep with `10_eye_to_hand.py`, one more detect and segment pass per pick (the weights are shared,
the inference is not), and two config keys that are easy to get wrong in opposite directions. Those
two are what this file is about, because both were found by building the cell rather than by reading
it, and both are silent.

The cell it runs on is the one that ships: `WILLY_PROFILE=ur5e,eth2`, which is `robot.eth2.yaml`
plus `camera/cam.eth2.yaml`, two fixed RGB-D cameras over one bin. It has never run on hardware. The
config path is complete and the refusals below are armed; the bring-up is somebody's first day.

The first trap: the primary camera's artifact does not live in `fusion.cameras`. It is the separate
`grasping.fusion.extrinsics_artifact_path`, and that key alone satisfies the CAMERA to BASE refusal
a real cell raises at build. The calibration routine prints a `fusion.cameras` block to paste after
every camera, and pasting only those leaves this key null, so a cell can be fully calibrated, load
every artifact it names, and still be refused. Worse, the config preflight passes: it checks that
the key is set and never opens the file.

The second trap points the other way: the primary camera must not appear in `fusion.cameras`. Two
readers of that map disagree about what belongs in it. The rig builder filters the primary out,
because the primary already streams through the main perception source and fusing a view with a
second copy of itself costs a full inference pass for nothing. The orchestrator's expectation keeps
every enabled entry. List the primary and it is permanently expected and never delivered: a warning
on every pick under `on_camera_unavailable: degrade`, a raised `RuntimeError` on every pick under
`refuse`. `robot.sim.yaml` does list all of its cameras, correctly, because the simulator builds its
own rig that includes the primary. Copying that shape into a real cell is the mistake.

Calibration has no library twin, so `10_eye_to_hand.py` drives `real_cell.calibrate` and this file
reads what it wrote. Everything below runs on a copy of the loaded config: no YAML is edited, no
artifact is written, no device is opened and no arm is built.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import Example, not_ready  # noqa: E402

EPILOG = """
exit codes
  0  the cell is wired coherently and every camera's artifact loaded
  1  a trap is live in this config: read the FAILED and finding lines
  2  not calibrated yet: an artifact this cell names is not on disk. Run 10_eye_to_hand.py per rig
"""


def _why(error: BaseException) -> str:
    """One line of a refusal: the reason, without the paragraph that follows it.

    These messages are written for a terminal, so they run several sentences and often quote an
    absolute path first. The first sentence after the path is the part that says what was wrong.
    Both trims degrade to the whole message rather than to nothing.

    Takes a `BaseException`, not an `Exception`: two of the refusals shown here are `SystemExit`,
    which the build path raises for a configuration a person has to go and fix.
    """
    text = " ".join(str(error).split())
    if "failed to load: " in text:
        text = text.split("failed to load: ", 1)[1]
    sentence = text.split(". ", 1)[0]
    return sentence if len(sentence) <= 150 else sentence[:150] + " ..."


def _rgbd_rigs(camera_cfg: Any) -> list[str]:
    """Every `source: rgbd` rig id, in declaration order.

    Order is load bearing: the first one is the primary, the camera the grasp is synthesised from,
    which is how `build_real_components` picks it. Moving it moves what every other camera confirms.
    """
    rigs = getattr(getattr(camera_cfg, "cameras", None), "rigs", None) or []
    return [str(rig.rig_id) for rig in rigs if getattr(rig, "source", None) == "rgbd"]


def _with_fusion(robot: Any, **fields: Any) -> Any:
    """A copy of the robot config with `grasping.fusion` fields replaced. Nothing is written."""
    fusion = robot.grasping.fusion.model_copy(update=fields)
    grasping = robot.grasping.model_copy(update={"fusion": fusion})
    return robot.model_copy(update={"grasping": grasping})


def _the_primary_key_is_the_one_that_counts(run: Example, robot: Any) -> None:
    """Null the single key, keep every other camera, and show the whole cell refused."""
    from src.robot.execution.autonomous_grasp.cells import build_rehearsal_components
    from src.robot.execution.autonomous_grasp.service import AutonomousGraspService

    probe = _with_fusion(robot, extrinsics_artifact_path=None)
    try:
        calculator, perception, _resolver, _cameras = build_rehearsal_components(probe)
    except Exception as error:                                  # noqa: BLE001  (report, not raise)
        # `build_calculator` is fail-closed under `grasping.calculator: deep` with no readable
        # artifact, and that refusal arrives before the one under test. Saying so beats reporting a
        # missing generator as a frame problem.
        run.finding("primary artifact key",
                    f"not reached: the calculator refused first ({type(error).__name__})")
        return
    with run.step("with the key nulled, the cell is refused") as report:
        try:
            AutonomousGraspService.from_robot_config(
                probe, calculator=calculator, perception=perception)
        except Exception as error:                              # noqa: BLE001  (the refusal is it)
            report(f"{type(error).__name__}: {_why(error)}")
            return
        run.finding("primary artifact key",
                    "the cell BUILT with no primary artifact; this vendor is exempt from the "
                    "refusal, as sim and dummy are")


def _the_primary_is_filtered_out_of_the_rig(run: Example, robot: Any, primary: str) -> None:
    """Show that the extra-camera rig never opens the primary, and that it does open the others.

    Reaches for `_build_multi_camera_rig` by name. It is private and there is no public seam that
    builds the rig without opening devices, and this is the function the config file names when it
    warns about this trap, so a check that skipped it would be checking something else.

    The provider handed in is `None`, which cannot open anything. That is the instrument: a call
    that returns without touching it opened no camera, and a call that reaches for `open_rig` says
    so by name in an `AttributeError`. The two probes differ only in whether a camera other than the
    primary is in the map, so the filter is the only thing that can explain the difference.
    """
    from src.config.schema.robot.grasping_schema import CameraExtrinsicsConfig
    from src.robot.execution.autonomous_grasp.cells import _build_multi_camera_rig
    from src.config.loader import load_config

    app = load_config()
    entry = CameraExtrinsicsConfig(
        enabled=True, mounting_mode="eye_to_hand", extrinsics_artifact_path="unread.json")
    only_primary = _with_fusion(robot, cameras={primary: entry})

    with run.step("a map naming only the primary builds no rig") as report:
        rig = _build_multi_camera_rig(only_primary, app, provider=None, primary_rig_id=primary,
                                      backend=None, prompt="an object")
        report(f"rig is {rig}; the provider was never asked to open anything")
    if rig is not None:
        run.finding("multi-camera rig", "the primary was NOT filtered out; it will be fused with a "
                                        "second copy of itself, at the cost of an inference pass")

    others = sorted(cam for cam in (robot.grasping.fusion.cameras or {}) if cam != primary)
    if not others:
        run.note("This cell names no camera besides the primary, so the second probe is skipped.")
        return
    both = _with_fusion(robot, cameras={primary: entry, **dict(robot.grasping.fusion.cameras)})
    with run.step("and it does try to open the others") as report:
        try:
            _build_multi_camera_rig(both, app, provider=None, primary_rig_id=primary,
                                    backend=None, prompt="an object")
        except AttributeError as error:
            report(f"reached open_rig for {others}: {_why(error)}")
        except SystemExit as error:
            # A camera in the map with no matching RGB-D rig is refused before any device is
            # touched, which is a different trap and worth naming rather than swallowing.
            report(f"refused at build: {_why(error)}")
        else:
            run.finding("multi-camera rig",
                        f"no camera was opened although {others} are configured")


def _what_the_pick_loop_will_expect(run: Example, robot: Any, primary: str) -> None:
    """The arithmetic that turns the second trap into a failure on every pick.

    `expected` is the enabled entries of `fusion.cameras`, which is how the orchestrator's
    `configured_camera_ids` is built; `delivered` is what the rig above actually opens, which never
    includes the primary. The pick loop subtracts one from the other.
    """
    cameras = dict(robot.grasping.fusion.cameras or {})
    policy = str(getattr(robot.grasping.fusion.geometry, "on_camera_unavailable", "degrade"))
    consequence = "a raised RuntimeError" if policy == "refuse" else "a warning"

    def _sets(ids: list[str]) -> tuple[list[str], list[str], list[str]]:
        expected = sorted(ids)
        delivered = [cam for cam in expected if cam != primary]
        return expected, delivered, sorted(set(expected) - set(delivered))

    configured = [cam for cam, cfg in cameras.items() if bool(getattr(cfg, "enabled", True))]
    expected, delivered, missing = _sets(configured)
    with run.step("what every pick compares, as configured") as report:
        report(f"expected {expected}, delivered {delivered}, missing {missing}")
    if missing:
        run.finding(
            "fusion.cameras",
            f"the primary {primary!r} is listed, so every pick reports it missing: "
            f"{consequence} under on_camera_unavailable: {policy}")
        return
    # Nothing is wrong with this cell, so the failure is shown as the counterfactual it is: the same
    # arithmetic on the same map with one entry added, and that entry is the one the calibration
    # routine's own paste-ready block invites.
    expected, delivered, missing = _sets([*configured, primary])
    with run.step("and with the primary added to the map") as report:
        report(f"expected {expected}, delivered {delivered}, missing {missing}")
    run.note(f"That is {consequence} on every pick, forever, under on_camera_unavailable: "
             f"{policy}.")
    run.note("The cost of leaving the primary out is one false line at construction: the counter")
    run.note("that warns about a single-view cell reads the same map, so it reports one calibrated")
    run.note("camera while the cell fuses two.")


def _a_camera_with_no_rig_is_refused(run: Example, robot: Any, primary: str) -> None:
    """A camera named in the map that no rig declares is refused while an operator is watching."""
    from src.config.loader import load_config
    from src.config.schema.robot.grasping_schema import CameraExtrinsicsConfig
    from src.robot.execution.autonomous_grasp.cells import _build_multi_camera_rig

    app = load_config()
    cameras = dict(robot.grasping.fusion.cameras or {})
    cameras["a_camera_that_is_not_there"] = CameraExtrinsicsConfig(
        enabled=True, mounting_mode="eye_to_hand", extrinsics_artifact_path="unread.json")
    probe = _with_fusion(robot, cameras=cameras)
    with run.step("a camera with no rig is refused at build") as report:
        try:
            _build_multi_camera_rig(probe, app, provider=None, primary_rig_id=primary,
                                    backend=None, prompt="an object")
        except SystemExit as error:
            report(_why(error))
        except AttributeError:
            run.finding("fusion.cameras", "a camera with no matching rig was NOT refused; it "
                                          "would be expected and never deliver, on every pick")


def _load_every_artifact(run: Example, robot: Any) -> int | None:
    """Build the per camera resolver map for real. Returns an exit code only when it refuses."""
    from src.robot.execution.autonomous_grasp.builders import build_config_frame_resolvers

    # Outside a step, because this refusal is not the outcome under test: it is this cell being
    # uncalibrated. A step that reported it would print a tick beside a failure.
    try:
        resolvers = build_config_frame_resolvers(robot.grasping)
    except Exception as error:                                  # noqa: BLE001  (report, not raise)
        run.finding("fusion.cameras", f"{type(error).__name__}: {_why(error)}")
        return not_ready(
            "a camera named in grasping.fusion.cameras has no readable artifact",
            "calibrate each camera, one run per rig: "
            "python scripts/examples/calibration/10_eye_to_hand.py --rig <rig> --live")
    with run.step("load every camera's calibration") as report:
        report(", ".join(f"{cam} is a {type(res).__name__}"
                         for cam, res in sorted(resolvers.items()))
               or "no camera besides the primary")
    return None


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0911
    parser = argparse.ArgumentParser(description=__doc__ or "", epilog=EPILOG,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", default="ur5e,eth2",
                        help="WILLY_PROFILE for the two-camera cell (default: the shipped one)")
    args = parser.parse_args(argv)

    from src.config.loader import load_config
    from src.robot.execution.real_cell.preflight import run_config_preflight

    # No robot is built, no camera is opened and nothing is commanded, so claiming a rehearsal here
    # would imply something was held back that was not. The work is config and artifacts on disk.
    with Example("12 two_cameras", f"the two-camera cell of profile '{args.profile}'",
                 hardware=False, profile=args.profile) as run:
        with run.step("load the config tree") as report:
            try:
                app = load_config()
            except Exception as error:                          # noqa: BLE001  (report, not raise)
                return not_ready(
                    f"the config tree did not load ({type(error).__name__}: {error})",
                    f"run `WILLY_PROFILE={args.profile} python -m src.config --print` and read the "
                    f"first refusal")
            robot = app.robot
            rgbd = _rgbd_rigs(app.camera)
            report(f"{len(rgbd)} RGB-D rig(s): {rgbd or '(none)'}")
        if robot is None:
            return not_ready("this config tree has no `robot` block",
                             "select a profile that has one, such as WILLY_PROFILE=ur5e,eth2")
        if len(rgbd) < 2:
            return not_ready(
                f"this profile declares {len(rgbd)} RGB-D rig(s), so there is no second view",
                "a second camera is a hardware decision before it is a config one. The shipped "
                "two-camera cell is WILLY_PROFILE=ur5e,eth2; --profile selects your own")

        primary, *rest = rgbd
        fusion = robot.grasping.fusion
        with run.step("primary camera and the map beside it") as report:
            report(f"primary {primary!r} (first rgbd rig), second view(s) {rest}; "
                   f"fusion.cameras names {sorted(fusion.cameras or {}) or '(nothing)'}")

        with run.step("the two switches, which are not one switch") as report:
            report(f"fusion.enabled {fusion.enabled} (calibration and the shadow voxel grid); "
                   f"fusion.geometry.enabled {fusion.geometry.enabled} (the grasp itself)")
        with run.step("how a second view is admitted") as report:
            report(f"metric {fusion.geometry.metric}, min_score {fusion.geometry.min_score}, "
                   f"neighbour {fusion.geometry.neighbour_mm} mm, "
                   f"on_camera_unavailable {fusion.geometry.on_camera_unavailable}")
        run.note("`overlap` abstains when a camera shares no surface with the primary, which is")
        run.note("what an occluded camera should do. `centroid` never abstains, so an")
        run.note("absent target is answered with a neighbour's surface welded into the")
        run.note("cloud you plan the grasp on.")

        run.note("")
        run.note("Trap one: the primary artifact key. The preflight below reads it, and only it.")
        with run.step("config preflight over the whole cell") as report:
            outcome = run_config_preflight(robot)
            camera_check = next((c for c in outcome.checks if c.name == "camera -> base"), None)
            report(f"{len(outcome.checks)} check(s); "
                   f"camera -> base is {getattr(camera_check, 'status', 'absent')}")
        run.note("That check tests that the key is set. It does not open the file, so it passes on")
        run.note("a cell whose artifact was never written, and the build then fails.")
        _the_primary_key_is_the_one_that_counts(run, robot)

        run.note("")
        run.note("Trap two: who belongs in fusion.cameras.")
        _the_primary_is_filtered_out_of_the_rig(run, robot, primary)
        _what_the_pick_loop_will_expect(run, robot, primary)
        _a_camera_with_no_rig_is_refused(run, robot, primary)

        run.note("")
        run.note("Now the artifacts themselves, loaded exactly as a cell loads them.")
        refused = _load_every_artifact(run, robot)
        if refused is not None:
            return refused
        from src.robot.execution.autonomous_grasp.builders import build_config_frame_resolver

        try:
            resolver = build_config_frame_resolver(robot.grasping)
        except Exception as error:                              # noqa: BLE001  (report, not raise)
            run.finding("fusion.extrinsics_artifact_path", f"{type(error).__name__}: {_why(error)}")
            return not_ready(
                f"grasping.fusion.extrinsics_artifact_path names "
                f"{fusion.extrinsics_artifact_path}, which does not load",
                f"calibrate the primary camera: python "
                f"scripts/examples/calibration/10_eye_to_hand.py --rig {primary} --live")
        with run.step("the primary's own artifact") as report:
            report(f"a {type(resolver).__name__}" if resolver is not None else
                   "no resolver: the key is unset or fusion.enabled is false")
        if resolver is None:
            run.finding("fusion.extrinsics_artifact_path",
                        "no primary resolver, so a real cell is refused at build")
        return run.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
