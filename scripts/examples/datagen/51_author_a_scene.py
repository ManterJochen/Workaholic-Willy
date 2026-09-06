"""51: what a scene is, and what the layout has already decided before anything renders.

    python scripts/examples/datagen/51_author_a_scene.py
    python scripts/examples/datagen/51_author_a_scene.py --scenes 40 --index 7
    python scripts/examples/datagen/51_author_a_scene.py --pile-only --seed 3

Nothing here renders, imports an engine or needs a GPU. `datagen.scenes.layout_scene` takes a config
and a scene index and returns a `SceneSpec`: which assets, where each one starts, which cameras look
at it, how the lighting was drawn. That seam is the reason a layout bug costs seconds instead of a
night of path tracing, and it is where every decision below is made.

Two of those decisions change what a corpus teaches, and both are one line of config.

The family weights say how the objects are arranged. `sparse` separates them, `packed` has them
touching in one layer, which is where segmentation merges two masks into one object, `bin` puts them
inside walls, and `pile` drops them from a height. They are four physics regimes rather than four
densities, and the mix is planned up front so 40 scenes at equal weights is exactly 10 of each
rather than roughly 10.

`families.flat_orientation` says how an object rests when it is placed flat, `upright` or `random`,
and it is the quieter of the two. An object standing on its own base sits in a stable equilibrium,
so the settle leaves it there; a tipped object is far more likely to admit a jaw grasp, and a bin a
cell actually picks from holds parts that were dumped rather than stood up. The default is
`upright`, because that is what an existing config already generates.

The one number the layout does not decide is where anything ends up. These are spawn poses. The
settled poses are an output of whichever engine 52 selects, which is why a scene is reproducible
from the seed plus the engine and the physics version, and not from the seed alone.
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path
from typing import Any, Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import Example, not_ready  # noqa: E402

EPILOG = """
exit codes
  0  the plan, one scene and both rest-pose answers were produced
  1  the layout produced something contradictory: read the WARN line
  2  the configuration could not be built at all, and the message names the key

nothing is written. This example only authors specs in memory.
"""


def _validator_message(error: Any) -> str:
    """The complaint a config validator raised, without pydantic's wrapper.

    `str(ValidationError)` is a multi-line report ending in an error-code URL, so any fixed-width
    slice of it quotes the documentation link rather than the reason.
    """
    messages = [str(entry.get("msg", "")) for entry in error.errors()]
    text = " ".join(" ".join(messages).split())
    return text.removeprefix("Value error, ")


def _is_upright(orientation_xyzw: tuple[float, float, float, float]) -> bool:
    """Is this a yaw about the vertical, rather than a full orientation?

    A rotation about z alone has x and y exactly zero in the quaternion, and `_yaw_quat_xyzw` builds
    exactly that. So this is a test of what the layout did, not an approximation of it, which is why
    there is no tolerance.
    """
    return orientation_xyzw[0] == 0.0 and orientation_xyzw[1] == 0.0


def _families(pile_only: bool, orientation: Literal["upright", "random"]) -> Any:
    """The family block both halves of the fork share, differing only in the rest pose.

    A `FamilyMixConfig` rather than a nested dict. Both are accepted at runtime, and the model is
    what a type checker can see: the config tree is frozen and strict, so a mistyped key is an error
    here rather than a field that silently keeps its default.
    """
    from datagen.config import FamilyMixConfig  # noqa: PLC0415

    if pile_only:
        return FamilyMixConfig(flat_orientation=orientation, pile_weight=1.0, sparse_weight=0.0,
                               packed_weight=0.0, bin_weight=0.0)
    return FamilyMixConfig(flat_orientation=orientation)


def _rest_poses(config: Any, scenes: int) -> tuple[int, int, int]:
    """`(upright, flat_total, dropped_total)` over the first `scenes` scenes of a config.

    The pile is counted apart rather than pooled in. `flat_orientation` is read by the flat
    placement only; the pile draws a full orientation whatever it says, because an object about to
    be dropped has no rest pose yet. Pooled, the two would move the same number and the fork would
    look weaker than it is.
    """
    from datagen.scenes import SceneFamily, layout_scene

    upright = flat = dropped = 0
    for index in range(scenes):
        spec = layout_scene(config, index)
        if spec.family is SceneFamily.PILE:
            dropped += len(spec.objects)
            continue
        for placed in spec.objects:
            flat += 1
            upright += int(_is_upright(placed.orientation_xyzw))
    return upright, flat, dropped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "", epilog=EPILOG,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenes", type=int, default=40, help="how many scenes to plan")
    parser.add_argument("--index", type=int, default=0, help="which scene to describe in full")
    parser.add_argument("--seed", type=int, default=0, help="dataset seed")
    parser.add_argument("--pile-only", action="store_true",
                        help="weight the pile family alone, which is the one family a solver has "
                             "to settle")
    args = parser.parse_args(argv)

    from pydantic import ValidationError

    from datagen.config import ArmConfig, CameraRigConfig, DatagenConfig, RenderConfig
    from datagen.scenes import layout_scene, plan_families

    if not 0 <= args.index < args.scenes:
        return not_ready(f"--index {args.index} is outside the {args.scenes} scene(s) planned",
                         "raise --scenes, or pick an index inside the plan")

    try:
        config = DatagenConfig(scenes=args.scenes, seed=args.seed,
                               families=_families(args.pile_only, "upright"))
    except ValidationError as refusal:
        # Reached by setting every family weight to zero, which is a config that describes no
        # scene at all. The schema says so rather than generating an empty dataset.
        return not_ready(_validator_message(refusal),
                         "set at least one family weight above zero")

    with Example("51 scenes", f"{args.scenes} scene(s) planned from seed {args.seed}",
                 hardware=False) as run:

        # Every listing below is collected inside its step and printed after it. A step announces
        # itself on an open line, so a body that prints its own lines writes over the verdict.
        mix: list[str] = []
        with run.step("the family plan, decided for every scene up front") as report:
            plan = plan_families(config)
            counts = collections.Counter(family.value for family in plan)
            for family, number in sorted(counts.items()):
                mix.append(f"  {family:8} {number:5d} scene(s)")
            mix.append(f"  order    {' '.join(f.value[:2] for f in plan[:16])} ...")
            report(f"{len(plan)} scene(s), {len(counts)} family/families, exact counts")
        run.note("")
        for line in mix:
            run.note(line)
        run.note("")
        run.note("The order is interleaved on purpose. Drawn per scene the mix would only be right")
        run.note("on average, and rendered in blocks an interrupted run would hold every sparse")
        run.note("scene and no bin scene at all.")
        run.note("")

        described: list[str] = []
        with run.step(f"scene {args.index}, in full") as report:
            spec = layout_scene(config, args.index)
            described.append(f"  id         {spec.scene_id}")
            described.append(f"  family     {spec.family.value} in the {spec.domain.value} domain")
            described.append(f"  objects    {len(spec.objects)}")
            described.append(f"  drop       {spec.drop_height_mm:.0f} mm above the table")
            described.append(f"  bin walls  {len(spec.bin_walls)}")
            for camera in spec.cameras:
                described.append(f"  camera     {camera.name:14} {camera.mount.value:6} at "
                                 f"{tuple(round(v) for v in camera.position_mm)} mm, "
                                 f"{camera.horizontal_fov_deg:.0f} deg")
            for placed in spec.objects[:4]:
                described.append(f"  object     {placed.asset_id:22} at "
                                 f"{tuple(round(v, 1) for v in placed.position_mm)} mm, "
                                 f"{placed.mass_kg * 1000:.0f} g")
            if len(spec.objects) > 4:
                described.append(f"  object     ... and {len(spec.objects) - 4} more")
            report(f"{len(spec.objects)} object(s), {len(spec.cameras)} view(s), "
                   f"{len(spec.bin_walls)} wall(s)")
        run.note("")
        for line in described:
            run.note(line)
        run.note("")
        run.note("Millimetres and XYZW quaternions, the repository's units rather than a")
        run.note("renderer's. The metres and WXYZ conversion happens inside the render module, at")
        run.note("the boundary, exactly the way a vendor driver keeps its own conventions inside")
        run.note("itself. A wall is a centre and half extents, in the form the safety layer's")
        run.note("fixture boxes take, so the wall the camera sees and the wall the planner avoids")
        run.note("are one description.")
        run.note("")

        with run.step("the same seed, the same scene, byte for byte") as report:
            again = layout_scene(config, args.index)
            other = layout_scene(config, (args.index + 1) % args.scenes)
            report(f"index {args.index} reproduces: {again == spec}; "
                   f"differs from index {(args.index + 1) % args.scenes}: {other != spec}")
        run.note("")
        run.note("The randomisation is resolved here rather than in the renderer, which is what")
        run.note("makes that hold. A renderer rolling its own dice would make the seed a")
        run.note("half-truth, and two runs of the same dataset would differ in ways nothing could")
        run.note("diff.")
        run.note("")

        upright_config = config
        random_config = DatagenConfig(scenes=args.scenes, seed=args.seed,
                                      families=_families(args.pile_only, "random"))
        sampled = min(args.scenes, 12)
        with run.step("the fork: upright, the default") as report:
            standing, flat, dropped = _rest_poses(upright_config, sampled)
            report(f"{standing} of {flat} flat placement(s) stand on their base")
        with run.step("the fork: random, the other answer") as report:
            tumbled, flat_random, _ = _rest_poses(random_config, sampled)
            report(f"{tumbled} of {flat_random} flat placement(s) stand on their base")
        run.note("")
        run.note(f"Counted over the first {sampled} scene(s). The {dropped} placement(s) in the")
        run.note("pile family are left out of both numbers: an object about to be dropped has no")
        run.note("rest pose yet, so the pile draws a full orientation whichever answer this is.")
        run.note("")
        run.note("What that costs. An upright object rests where it was put, so the corpus is")
        run.note("stable and its objects are standing, which is not what a dumped bin looks like")
        run.note("and which loses jaw labels: a tipped object presents a graspable width far more")
        run.note("often than a standing one. A random orientation buys those poses and spends")
        run.note("settling: on the no-engine backend an object that will not stand is refused")
        run.note("rather than seated, so the yield falls. Neither is free, and the default is the")
        run.note("one that reproduces an existing corpus.")
        run.note("")
        run.note("The orientation is in the spec rather than in an engine deliberately. Reaching")
        run.note("for a drop height instead would make the variety a property of the physics, and")
        run.note("`none` and `mujoco` would then produce different distributions from one config.")
        run.note("")

        with run.step("the rig is a default, and a contradiction in it is refused here") as report:
            try:
                DatagenConfig(scenes=1, seed=0,
                              camera_rig=CameraRigConfig(views=("wrist", "oblique_left")),
                              render=RenderConfig(arm=ArmConfig(mode="absent")))
            except ValidationError as refusal:
                # `errors()[0]["msg"]`, not `str(refusal)`. The string form ends in pydantic's own
                # error-code URL, so a verdict cut from its tail quotes the documentation link and
                # not the complaint.
                report(_validator_message(refusal)[:92])
            else:
                run.finding("camera rig", "an eye-in-hand view was planned with no arm to mount it")
        run.note("")
        run.note("An eye-in-hand dataset whose images contain no hand is a different dataset")
        run.note("wearing the same name, and the arm is the cell's dominant occluder, so its")
        run.note("absence changes every image rather than decorating it. The refusal fires at")
        run.note("config time because the alternative is a finished dataset that records the")
        run.note("disagreement in a field somebody has to think to read.")
        run.note("")
        run.note("Describe your own cell instead of taking the default rig:")
        run.note("  from datagen.config import CameraSpec")
        run.note("  camera_rig={'cameras': (CameraSpec(name='left', position_mm=(100, -400, 700)),")
        run.note("                          CameraSpec(name='eih', mount='wrist'))}")
        run.note("A fixed camera with no position is refused rather than defaulted, and")
        run.note("`workspace.table_size_mm`, `workspace.bin` and `families.placement` are yours in")
        run.note("the same way.")
        run.note("")
        run.note("Which to pick. Start on the default mix: four families are four different ways")
        run.note("for a pick to fail, and a corpus of one of them teaches one of them. Turn")
        run.note("`flat_orientation` to random when your cell picks out of a bin somebody tips")
        run.note("parts into. And note which family carries a drop: `pile` is the only one whose")
        run.note(f"objects leave the table, {counts.get('pile', 0)} scene(s) of this plan, and it is")
        run.note("exactly the family the cheapest engine in the next example refuses to render.")
        run.note("")
        run.note("Next: 52_which_engine.py, which settles these spawn poses into real ones.")
        return run.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
