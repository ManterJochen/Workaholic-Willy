"""54: extracting the corpus a generator trains on, and what `--kinds both` changes.

    python scripts/examples/datagen/54_build_a_corpus.py
    python scripts/examples/datagen/54_build_a_corpus.py --kinds jaw
    python scripts/examples/datagen/54_build_a_corpus.py --dataset <dir> --out <dir>

Extraction is the third verb of `DatasetBuild`. It fuses every rendered view of a scene into one
point cloud in base millimetres, keeps the environment in it because a generator that never sees a
bin wall proposes grasps into one, orients the normals towards the cameras that saw each point, and
writes the labels beside the geometry as one `.npz` per scene.

The decision here is `kinds`, and it is the one place a corpus decides which grasps exist.

Jaw alone is the default and reproduces every corpus this repository has built. `kinds=("jaw",
"suction")` adds the suction labels as their own `suction_*` arrays. Far more objects earn a suction
label than a jaw one, so a jaw-only corpus records objects that a cell with a cup can pick as
objects with nothing on them. The first step below counts that on your own labels rather than
quoting somebody else's, because the ratio is a property of your parts.

What it changes is one stage, and the boundary is worth stating before you run it. The pose heads
are jaw-only by construction and a mixed corpus is not a fix for them: a jaw head cannot learn a
pose from a suction label, because a cup has no closing axis and no opening, and the corpus stores
exactly that, `closing_axis [0, 0, 0]` and `width_mm 0.0`. The sample contract refuses those two
shapes by name, which is why suction lives in separate arrays rather than in a `kind` column inside
the jaw ones: mixed in, they would produce samples that fail the validator, and any consumer that
forgot to filter would train a jaw head on a zero axis.

The stage it does change is the one that asks "is this bit of surface worth attempting at all". That
stage learns from jaw labels alone today, so it is taught to call a suction-only object empty, which
is false for any cell carrying both end effectors. The training step reads the extra array as
`graspability_suction`; where it is absent the run is byte-identical, which is why every corpus
built before this flag is unaffected.

One trap on the way here: `--kinds` on the labelling step decides nothing. The labeller always
writes both kinds, and this is where the choice is made.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import EXIT_FAILED, Example, not_ready  # noqa: E402

EPILOG = """
exit codes
  0  both corpora were extracted and compared
  1  an extraction produced nothing, or the jaw half was not identical between the two
  2  nothing to extract: no dataset, or it carries no grasp labels
"""

#: Arrays that describe the jaw half of a scene. The comparison below requires every one of them to
#: be identical between the two extractions: `--kinds both` may add, and may not alter.
_JAW_ARRAYS: tuple[str, ...] = (
    "grasp_position_mm", "grasp_approach", "grasp_axis", "grasp_width_mm", "grasp_instance",
    "grasp_held", "grasp_approach_admissible", "grasp_part_role", "grasp_asset_id",
    "contact_points_mm", "contact_grasp_index", "points_mm", "normals", "instance_id",
    "view_count", "object_instance", "object_asset_id",
)


def _coverage(scene_file: Path) -> tuple[int, int, int]:
    """`(objects, objects with a jaw label, objects with a suction label)` in one scene file."""
    import numpy as np

    with np.load(scene_file, allow_pickle=False) as handle:
        objects = {int(value) for value in handle["object_instance"]}
        jaw = {int(value) for value in handle["grasp_instance"]}
        suction = {int(value) for value in handle["suction_instance"]}
    return len(objects), len(objects & jaw), len(objects & suction)


def _identical(left: Path, right: Path) -> list[str]:
    """Which of the jaw arrays differ between two extractions of the same scene."""
    import numpy as np

    with np.load(left, allow_pickle=False) as one, np.load(right, allow_pickle=False) as two:
        return [name for name in _JAW_ARRAYS
                if name in one.files and not np.array_equal(one[name], two[name])]


def _extract(build: Any, out: Path, kinds: tuple[str, ...], physics: str | None) -> Any:
    """One extraction, with `physics` forwarded only when there is a verdict file to forward."""
    if physics is None:
        return build.clouds(out, kinds=kinds)
    return build.clouds(out, kinds=kinds, physics=physics)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "", epilog=EPILOG,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="logs/examples/example_corpus",
                        help="the labelled dataset from 53_label_and_screen.py")
    parser.add_argument("--out", default="logs/examples/example_corpus_clouds",
                        help="where the corpus you keep is written")
    parser.add_argument("--kinds", default="both", choices=("jaw", "both"),
                        help="which grasp kinds land in --out. The other one is extracted beside "
                             "it so the two can be compared")
    parser.add_argument("--physics", default=None,
                        help="a grasp_physics.jsonl to fold in as `grasp_held`. Default: the one "
                             "in the dataset, when 53 has written it")
    args = parser.parse_args(argv)

    dataset = Path(args.dataset)
    if not (dataset / "grasps.jsonl").is_file():
        return not_ready(f"{dataset} carries no grasps.jsonl",
                         "run 53_label_and_screen.py first; a corpus extracted from an unlabelled "
                         "dataset trains successfully and teaches nothing")

    from datagen.api import DatasetBuild
    from datagen.config import DatagenConfig

    physics = args.physics
    if physics is None:
        found = dataset / "grasp_physics.jsonl"
        physics = str(found) if found.is_file() else None

    build = DatasetBuild.from_config(DatagenConfig(scenes=1, seed=0), name=dataset.name,
                                     out_root=dataset.parent)
    chosen = ("jaw", "suction") if args.kinds == "both" else ("jaw",)
    other = ("jaw",) if args.kinds == "both" else ("jaw", "suction")
    kept = Path(args.out)
    beside = kept.with_name(f"{kept.name}_{'jaw_only' if args.kinds == 'both' else 'both'}")

    with Example("54 corpus", f"{dataset.name} extracted as `{args.kinds}`, and as the other one",
                 hardware=False) as run:

        # Each table is collected inside its step and printed after it. A step announces itself on
        # an open line, so a body that prints puts its own output where the verdict goes.
        by_family: list[str] = [
            f"  {'family':8} {'objects':>8} {'with jaw':>9} {'with suction':>13}"]
        with run.step("the decision, counted on your own labels") as report:
            summary = json.loads((dataset / "grasp_label_report.json").read_text(encoding="utf-8"))
            families = dict(summary.get("by_family", {}))
            objects = with_jaw = with_suction = 0
            for family, bucket in sorted(families.items()):
                by_family.append(
                    f"  {family:8} {bucket['objects']:8d} {bucket['objects_with_jaw']:9d} "
                    f"{bucket['objects_with_suction']:13d}")
                objects += int(bucket["objects"])
                with_jaw += int(bucket["objects_with_jaw"])
                with_suction += int(bucket["objects_with_suction"])
            report(f"{with_jaw} of {objects} object(s) earn a jaw label, {with_suction} a suction "
                   f"one")
        run.note("")
        for line in by_family:
            run.note(line)
        run.note("")
        run.note("Those two columns are the decision. Every object in the third column and not in")
        run.note("the second is an object a jaw-only corpus records as carrying nothing, and a")
        run.note("cell with a cup can pick it. Read the bin row in particular: a jaw needs room")
        run.note("beside the object and a wall takes that room away, while a cup only needs a flat")
        run.note("patch on top.")
        run.note("")

        with run.step(f"extract `{'+'.join(chosen)}` into {kept.name}") as report:
            first = _extract(build, kept, chosen, physics)
            report(f"{first.summary.get('scenes_written', 0)} scene(s), "
                   f"{first.summary.get('grasps', 0)} jaw grasp(s), "
                   f"{first.summary.get('points', 0):,} point(s)")
        if not first.ok:
            run.finding("extract", first.reason)
            return EXIT_FAILED

        with run.step(f"extract `{'+'.join(other)}` into {beside.name}") as report:
            second = _extract(build, beside, other, physics)
            report(f"{second.summary.get('scenes_written', 0)} scene(s), "
                   f"{second.summary.get('grasps', 0)} jaw grasp(s)")
        if not second.ok:
            run.finding("extract", second.reason)
            return EXIT_FAILED

        both_dir, jaw_dir = (kept, beside) if args.kinds == "both" else (beside, kept)
        per_scene: list[str] = [
            f"  {'scene':22} {'objects':>8} {'jaw':>6} {'suction':>8} {'rows':>7}"]
        with run.step("what the two corpora hold, scene by scene") as report:
            import numpy as np

            drifted: list[str] = []
            rows = blind = 0
            for scene_file in sorted(both_dir.glob("*.npz")):
                twin = jaw_dir / scene_file.name
                if not twin.is_file():
                    continue
                objects_here, jaw_here, suction_here = _coverage(scene_file)
                with np.load(scene_file, allow_pickle=False) as handle:
                    suction_rows = int(handle["suction_position_mm"].shape[0])
                per_scene.append(f"  {scene_file.stem:22} {objects_here:8d} {jaw_here:6d} "
                                 f"{suction_here:8d} {suction_rows:7d}")
                drifted.extend(f"{scene_file.stem}:{name}" for name in _identical(scene_file, twin))
                rows += suction_rows
                blind += int(jaw_here == 0 and suction_here > 0)
            report(f"{rows:,} suction row(s) added, {blind} scene(s) that jaw alone calls empty, "
                   f"{len(drifted)} jaw array(s) changed")
        run.note("")
        for line in per_scene:
            run.note(line)
        if drifted:
            # The claim this example makes about `--kinds both` is that it adds and does not alter.
            # Checked rather than asserted, because a corpus whose jaw half moved would invalidate
            # every comparison against one built before the flag existed.
            run.finding("jaw half", f"changed between the two extractions: {drifted[:3]}")
            return EXIT_FAILED
        run.note("")
        run.note("")
        if blind:
            run.note(f"In {blind} of those scene(s) not one object earns a jaw label, while every")
            run.note("object in them earns a suction one. Jaw alone, such a scene is a point cloud")
            run.note("with an empty grasp table: not one object misread, a whole scene taught as")
            run.note("nothing worth attempting.")
            run.note("")
        run.note("The jaw half is identical array for array, which is what makes this safe to")
        run.note("turn on: a corpus built with both kinds is byte-identical to a jaw-only one")
        run.note("everywhere a jaw consumer looks. Note also that the `suction_*` arrays exist in")
        run.note("both corpora and are empty in the jaw-only one. The difference is the rows, not")
        run.note("the keys, so a reader that looks for the key alone learns nothing.")
        run.note("")

        if physics:
            run.note(f"Physics verdicts from {Path(physics).name} were folded in:")
            run.note(f"{first.summary.get('grasps_with_physics', 0)} of "
                     f"{first.summary.get('grasps', 0)} jaw grasp(s) carry one. A grasp the physics")
            run.note("never tried is stored as unmeasured rather than as zero, because an")
            run.note("unmeasured grasp and a measured failure are different facts.")
        else:
            run.note("No physics verdict file was folded in, so every `grasp_held` is unmeasured.")
            run.note("Run 53_label_and_screen.py to produce one, then pass --physics.")
        run.note("")
        run.note("What the extra rows reach, and what they do not. The pose heads are jaw-only by")
        run.note("construction, and a suction row carries closing_axis [0, 0, 0] and width_mm 0.0,")
        run.note("which the sample contract refuses under the names `zero-length direction` and")
        run.note("`non-positive opening`. What the rows do reach is the stage that decides whether")
        run.note("a patch of surface is worth attempting: `graspability_suction` in the sample,")
        run.note("folded into that stage's target in the training step. A corpus without the array")
        run.note("makes that a no-op and the run byte-identical.")
        run.note("")
        run.note("Which to pick. Build with both kinds when the cell you are building for carries")
        run.note("a cup as well as a jaw, which is the case this repository ships for; the extra")
        run.note("rows cost a fraction of the file size and nothing that reads jaw labels can see")
        run.note("them. Build jaw alone when the cell has one jaw and nothing else, and then the")
        run.note("empty objects in the corpus are honest: your cell really cannot pick them.")
        run.note("")
        run.note("Two more settings on the same call, because they decide the corpus as much as")
        run.note("the kinds do. `mask_suffix` chooses between the renderer's own instance masks")
        run.note("and the ones a real segmenter produced, which is the difference between")
        run.note("measuring a generator and measuring a pipeline. `voxel_mm` decides how much")
        run.note("geometry survives at all. Neither raises when you leave it out; you simply get")
        run.note("the default and a corpus that is not the one you meant.")
        run.note("")
        run.note(f"The corpus to train on: {kept}")
        return run.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
