"""40: a jaw or a cup, and the two different questions they ask.

    python scripts/examples/grasping/40_jaw_or_suction.py
    python scripts/examples/grasping/40_jaw_or_suction.py --payload-g 2000
    python scripts/examples/grasping/40_jaw_or_suction.py --profile ur3e

A parallel jaw asks where two opposing surfaces are, how far apart they sit, and along which axis
the fingers travel. Its candidate therefore carries an aperture, an antipodal pair and a closing
direction. A cup asks a different question: where is one patch flat enough for the rim to seal, and
along which direction is it pressed. Its candidate carries a contact, a press direction and a seal
score, and none of the three jaw quantities, because a cup has no opening to size. The split
reaches the driver too. The `Gripper` Protocol is width based because jaws are, so the vacuum
driver reinterprets width as a threshold: `set_width_mm` below `vacuum_on_below_mm` means engage,
not close to that gap.

Both answers are analytic and neither is learned. `GraspCalculator` ranks jaw candidates,
`synthesize_suction_grasps` scores seal times wrench resistance from the published statics. There
is no learned suction scorer here and there will not be one wrapping a network trained on
non-commercially licensed data. The seam is the `SuctionScorer` Protocol, and what a replacement
needs is licence-clean data. See `src/robot/grasping/suction/README.md`.

The decision lives in three places, and they have to agree:

    robot.gripper.vendor                   which driver actuates it
    robot.grasping.gripper_geometry.kind   which envelope candidates are filtered against
    which synthesizer proposes the candidates

Only the first two are config keys, and that is the trap. The config-driven pick path has no
suction branch: it runs the jaw calculator whatever is bolted on, and `synthesize_suction_grasps`
is reached from the simulation runners rather than from the pick loop. So `kind: suction` changes
which envelope the jaw generator filters against; it does not switch modality. This file prints
that rather than implying it.

No robot is touched. The half of this decision only hardware can answer, the pin numbers, the port
block and how long an ejector takes to build vacuum, is config in `VacuumGripperConfig` and is
named here rather than exercised.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import EXIT_FAILED, Example, not_ready  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.geometry import Transform

EPILOG = """
exit codes
  0  both modalities ran and reported what they propose
  1  one of them proposed nothing on the scene it is meant to be good at
  2  the config tree has no `robot` block, so there is no gripper to describe
"""

#: Box heights swept in the reach step, in millimetres, tall to flat. On the shipped envelope the
#: jaw stops somewhere in this range and the cup does not. Where exactly is printed rather than
#: written down here, because it moves with `gripper_geometry.parallel_jaw` and with the gripper.
_HEIGHTS_MM: tuple[float, ...] = (80.0, 60.0, 50.0, 45.0, 40.0, 30.0, 20.0)

#: Where the box stands in BASE. Only x and y are chosen here; the height comes from the scene, so
#: the support surface is z = 0 and a candidate's BASE z is directly comparable with the box height
#: that produced it.
_BOX_XY_MM: tuple[float, float] = (400.0, 0.0)

#: Standard gravity, for turning the cup's pull-off force into the mass it balances.
_G_MPS2 = 9.81


def _nadir_camera_to_base(scene: Any) -> "Transform":
    """The CAMERA to BASE transform of a camera looking straight down at ``scene``.

    Built from the scene's own `plane_mm` rather than from a constant, so the support surface lands
    at z = 0 and the box top at exactly `object_height_mm`. A real cell reads this transform from
    the calibration artifact instead, which the calibration examples write and which
    `from_robot_config` refuses to start without.
    """
    import numpy as np

    from src.geometry import Frame, Transform

    return Transform(
        translation_mm=np.array([_BOX_XY_MM[0], _BOX_XY_MM[1], float(scene.plane_mm)]),
        # A half turn about X: the camera looks along its own +Z, and the base's down is -Z.
        quaternion_xyzw=np.array([1.0, 0.0, 0.0, 0.0]),
        from_frame=Frame.CAMERA,
        to_frame=Frame.BASE,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__ or "", epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--payload-g", type=float, default=250.0,
        help="part mass for the wrench term, in grams. Without a mass a cup candidate's quality is "
             "the seal score alone and nothing asks whether it can hold the part.")
    parser.add_argument(
        "--profile", default=None,
        help="WILLY_PROFILE chain to load (e.g. `ur3e`, `sim`). Default: the environment's.")
    args = parser.parse_args(argv)

    # Imported after the parse, so `--help` pays for none of it and still answers in a checkout
    # whose dependencies are not installed yet.
    import numpy as np

    from src.config import load_config
    from src.geometry import Frame
    from src.robot.execution.autonomous_grasp.builders import build_gripper_geometry
    from src.robot.execution.autonomous_grasp.rehearsal import RehearsalPerceptionSource
    from src.robot.grasping.collision import SupportPlane
    from src.robot.grasping.generation.calculator import GraspCalculator
    from src.robot.grasping.suction import (
        SuctionConfig,
        WrenchConfig,
        synthesize_suction_grasps,
    )
    from src.robot.grippers.registry import available_gripper_vendors

    with Example("40 jaw or suction", "two end effectors, two different questions",
                 hardware=False) as run:
        with run.step("what this cell declares it has") as report:
            tree = load_config(profile=args.profile) if args.profile else load_config()
            robot = tree.robot
            report("no `robot` block in this tree" if robot is None else
                   f"gripper.vendor {robot.gripper.vendor}, "
                   f"{robot.gripper.min_width_mm:.0f} to {robot.gripper.max_width_mm:.0f} mm, "
                   f"gripper_geometry.kind {robot.grasping.gripper_geometry.kind}")
        if robot is None:
            return not_ready("this config tree has no `robot` block",
                             "point the loader at a tree that has one; `config/robot/robot.yaml` "
                             "in this repository is one")

        run.note(f"drivers registered: {', '.join(available_gripper_vendors())}")
        run.note("`vacuum` is the suction one, and it is vendor neutral on purpose: an ejector on")
        run.note("an output pin, with an optional vacuum switch on an input, is the whole")
        run.note("interface, so a cell can be configured for suction before a cup is chosen.")
        run.note("")

        # One scene for both modalities, so the two answers differ by the modality and not by the
        # object. It is a synthetic box on a plane at a known place, the same scene the real-cell
        # rehearsal picks against. Grasp quality is not measured here and cannot be.
        scene = RehearsalPerceptionSource()
        frame = scene.acquire()
        camera_to_base = _nadir_camera_to_base(scene)
        support = SupportPlane(normal=np.array([0.0, 0.0, 1.0]), offset_mm=0.0, frame=Frame.BASE)

        with run.step("the scene, once, for both") as report:
            mask = frame.segmentations[0].mask
            report(f"{int(mask.sum())} px box, top {scene.object_height_mm:.0f} mm above the "
                   f"support, camera {scene.plane_mm:.0f} mm above it")

        jaw_model = build_gripper_geometry(robot.grasping.gripper_geometry)
        calculator = GraspCalculator(
            camera_matrix=frame.intrinsics,
            min_grip_width_mm=robot.gripper.min_width_mm,
            max_grip_width_mm=robot.gripper.max_width_mm,
            max_candidates=8,
        )
        with run.step("jaw: an aperture, a pair, a closing axis") as report:
            jaw = calculator.compute(
                frame.segmentations[0], frame.depth_map, camera_to_base=camera_to_base,
                gripper_model=jaw_model, support_plane=support, unit="mm")
            report(f"{len(jaw)} candidate(s), best score {jaw[0].score:.3f}, "
                   f"aperture {jaw[0].grip_width_mm:.1f} mm" if jaw else "no candidate")
        if jaw:
            best = jaw[0]
            run.note(f"position {np.round(best.position, 1).tolist()} mm in {best.frame.value}, "
                     f"approach {np.round(best.approach, 2).tolist()}")
            run.note(f"closing axis {np.round(best.axis, 2).tolist()}: the direction of travel.")
            run.note("A cup candidate has no answer to this line, and no field for one.")
            run.note("")

        with run.step("suction: one sealable contact, a press direction") as report:
            cup = synthesize_suction_grasps(
                frame.segmentations[0], frame.depth_map, frame.intrinsics,
                camera_to_base=camera_to_base, payload_mass_g=args.payload_g,
                config=SuctionConfig(max_results=5))
            report(f"{len(cup)} candidate(s), best quality {cup[0].quality:.3f}, "
                   f"seal {cup[0].seal_score:.3f}" if cup else "no candidate")
        if cup:
            top = cup[0]
            run.note(f"position {np.round(top.position_mm, 1).tolist()} mm in {top.frame.value}, "
                     f"approach {np.round(top.approach, 2).tolist()}")
            run.note("`quality` is the seal score times the wrench resistance; with no payload it")
            run.note("would be the seal alone. The seal terms that produced it:")
            run.note(f"  {top.metadata}")
            run.note("")

        with run.step("the envelope each candidate is checked against") as report:
            cup_model = build_gripper_geometry(
                robot.grasping.gripper_geometry.model_copy(update={"kind": "suction"}))
            width = jaw[0].grip_width_mm if jaw else robot.gripper.max_width_mm / 2.0
            spans: dict[str, tuple[float, float]] = {}
            for name, model in (("jaw", jaw_model), ("cup", cup_model)):
                corners = model.local_corners_mm(width)
                low, high = corners.min(axis=0), corners.max(axis=0)
                spans[name] = (float(high[0] - low[0]), float(high[2]))
            report(f"jaw {spans['jaw'][0]:.1f} mm across, {spans['jaw'][1]:.1f} mm past the "
                   f"contact; cup {spans['cup'][0]:.1f} mm across, {spans['cup'][1]:.1f} mm past "
                   f"it")
        run.note("The second number decides table clearance: it is how far the envelope reaches")
        run.note("beyond the grasp point, toward whatever the part is standing on.")
        run.note("`gripper_geometry.kind` selects which of these two the jaw generator filters")
        run.note("against. It does not move the pick path to suction, because there is no suction")
        run.note("branch on that path to move to.")
        run.note("")

        with run.step("where each modality stops") as report:
            # The same scene at falling heights, nothing else changed, so the boundary belongs to
            # the envelope and the table check rather than to the object.
            reach: list[tuple[float, int, int]] = []
            for height in _HEIGHTS_MM:
                shorter = RehearsalPerceptionSource(object_height_mm=height)
                shot = shorter.acquire()
                pose = _nadir_camera_to_base(shorter)
                jaws = GraspCalculator(
                    camera_matrix=shot.intrinsics,
                    min_grip_width_mm=robot.gripper.min_width_mm,
                    max_grip_width_mm=robot.gripper.max_width_mm,
                    max_candidates=8,
                ).compute(shot.segmentations[0], shot.depth_map, camera_to_base=pose,
                          gripper_model=jaw_model, support_plane=support, unit="mm")
                cups = synthesize_suction_grasps(
                    shot.segmentations[0], shot.depth_map, shot.intrinsics, camera_to_base=pose,
                    payload_mass_g=args.payload_g, config=SuctionConfig(max_results=1))
                reach.append((height, len(jaws), len(cups)))
            standing = [height for height, n_jaw, _ in reach if n_jaw > 0]
            every_cup = all(n_cup > 0 for _, _, n_cup in reach)
            cup_says = ("the cup proposes at every height swept" if every_cup
                        else "the cup stops somewhere too")
            report(f"jaw proposes nothing below {min(standing):.0f} mm; {cup_says}"
                   if standing else f"the jaw proposed nothing at any height; {cup_says}")
        for height, n_jaw, n_cup in reach:
            run.note(f"box {height:5.0f} mm tall    jaw {n_jaw:2d} candidate(s)    "
                     f"cup {n_cup:2d} candidate(s)")
        run.note("")
        run.note("That is the decision, not a preference. A flat part lying on a table is the case")
        run.note("the jaw cannot reach around, because the fingers would have to go under it, and")
        run.note("the cup does not need to. Tall parts, and anything with no sealable face, are")
        run.note("the other way round.")
        run.note("")

        # What the cup holds, from the model's own constants rather than a number typed here:
        # F_vac = P * A over the cup area, and the mass that force balances under gravity.
        wrench = WrenchConfig()
        hold_g = wrench.vacuum_force_n / _G_MPS2 * 1000.0
        with run.step("how heavy a part the cup holds") as report:
            light = synthesize_suction_grasps(
                frame.segmentations[0], frame.depth_map, frame.intrinsics,
                camera_to_base=camera_to_base, payload_mass_g=hold_g * 0.5,
                config=SuctionConfig(max_results=1, min_quality=0.0))
            heavy = synthesize_suction_grasps(
                frame.segmentations[0], frame.depth_map, frame.intrinsics,
                camera_to_base=camera_to_base, payload_mass_g=hold_g * 1.5,
                config=SuctionConfig(max_results=1, min_quality=0.0))
            report(f"{wrench.vacuum_kpa:.0f} kPa on a {wrench.cup_radius_mm:.0f} mm cup is "
                   f"{wrench.vacuum_force_n:.1f} N, about {hold_g / 1000.0:.2f} kg. Quality "
                   f"{light[0].quality:.3f} at half that mass, {heavy[0].quality:.3f} at one and "
                   f"a half times it")
        # The same call without the transform, which is the mistake worth meeting once here rather
        # than on a cell. Gravity is taken to be -Z of whatever frame the cloud is in, and a
        # downward-looking camera's -Z points up, so the load comes out compressive and the wrench
        # term never bites however heavy the part is.
        unposed = synthesize_suction_grasps(
            frame.segmentations[0], frame.depth_map, frame.intrinsics,
            payload_mass_g=hold_g * 1.5, config=SuctionConfig(max_results=1, min_quality=0.0))
        run.note("The next line is produced on purpose, by dropping one argument:")
        if unposed and heavy and unposed[0].quality > heavy[0].quality:
            run.finding("the same call, with no camera_to_base",
                        f"the too-heavy part scores {unposed[0].quality:.3f} rather than "
                        f"{heavy[0].quality:.3f}: gravity is taken to be -Z of the cloud's own "
                        f"frame, and a camera's -Z is not down")
        run.note("")
        run.note("A cup can also be asked whether it is holding something. With a vacuum switch on")
        run.note("an input the driver implements `ObjectDetectingGripper`, which opts the cell")
        run.note("into post-close verification. A jaw mostly has to be trusted.")
        run.note("")
        run.note("Next: 41_geometric_or_deep.py, which generator proposes the jaw candidates.")

        if not jaw or not cup:
            # A modality that proposed nothing on this scene is the answer, and it is a wrong one:
            # the box is graspable both ways at the shipped height. `Example.finding` does not move
            # `exit_code`, so the status is returned rather than inherited.
            run.finding("one modality proposed nothing",
                        f"jaw {len(jaw)}, cup {len(cup)} on a "
                        f"{scene.object_height_mm:.0f} mm box both should reach")
            return EXIT_FAILED
        return run.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
