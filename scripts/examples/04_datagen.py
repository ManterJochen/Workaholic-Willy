"""04: build a training corpus from your own objects, and know the bill before you start.

    python scripts/examples/04_datagen.py                     # price it, build 8 scenes, no GPU
    python scripts/examples/04_datagen.py --scenes 200 --engine mujoco
    python scripts/examples/04_datagen.py --scenes 2000 --engine isaac --price-only

No GPU and no Isaac install are needed for this. `--engine none` seats objects analytically and
rasterises depth and masks in numpy; `--engine mujoco` settles them with a solver. Only
`--engine isaac` needs a multi-gigabyte install and an RTX-class card, and only it produces
path-traced RGB.

A request is not a yield, and that is the number people get wrong. "How many scenes" is not the
question; "how many usable scenes" is. `--engine none` refuses the `pile` family by name, because a
pile is the physics, so at its 75 % yield a request for 2,000 leaves about 1,500.
`datagen.cost.estimate` answers in usable scenes and says what to request instead, and this example
prints that before it builds anything.

The trap that costs a held-out dataset: `assets.mesh_asset_ids` restricts which meshes are drawn,
not which scenes are built. Procedural objects carry their own weight, so a corpus that names your
parts and leaves `assets.procedural_weight` at its default comes out part full of objects nobody
asked for, and nothing says so. Zero both weights when the point of the dataset is which objects are
in it; `assets.refuse_procedural_fallback: true` turns that surprise into a refusal.

The three stages are the three verbs of one `datagen.api.DatasetBuild`, so every count printed below
is the one the builder computed rather than a directory listing taken afterwards.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import EXIT_FAILED, Example, not_ready  # noqa: E402

EPILOG = """
exit codes
  0  the corpus was priced and (unless --price-only) built
  1  a stage ran and produced nothing: read the WARN line for which one
  2  nothing to build with: no engine available, or the request was contradictory
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "", epilog=EPILOG,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenes", type=int, default=8, help="usable scenes to end up with")
    parser.add_argument("--engine", default="none", choices=("none", "mujoco", "isaac"),
                        help="none (default) needs nothing beyond this repo")
    parser.add_argument("--name", default="example_corpus", help="dataset name")
    parser.add_argument("--out", default="logs/examples", help="dataset root")
    parser.add_argument("--price-only", action="store_true", help="print the bill and stop")
    args = parser.parse_args(argv)

    from datagen.api import DatasetBuild
    from datagen.cost import ENGINE_COSTS, estimate, format_estimate

    with Example("04 datagen", f"{args.scenes} usable scene(s) on `{args.engine}`",
                 hardware=False) as run:
        with run.step("price it, before spending anything") as report:
            # The calculator, not a guess in this file. Its coefficients came off timed runs and it
            # carries the evidence string for each engine into its own output, which
            # `format_estimate` prints below.
            plan = estimate(args.scenes, engine=args.engine, jobs=1)
            report(f"{plan.hours:.2f} h, {plan.gigabytes:.2f} GB, "
                   f"request {plan.requested_scenes} to get {plan.usable_scenes}")
        run.note("")
        for line in format_estimate(plan).splitlines():
            run.note(line)
        run.note("")

        if plan.requested_scenes != plan.usable_scenes:
            run.note(f"WARN note the two numbers: `{args.engine}` yields "
                     f"{ENGINE_COSTS[args.engine].yield_fraction:.0%}, so the build is asked for "
                     f"{plan.requested_scenes} to leave you {plan.usable_scenes}.")
            run.note("")

        if args.price_only:
            run.note("--price-only: nothing was built.")
            return run.exit_code

        # One object for all three stages, built from the shipped defaults with the settings this
        # example exposes layered on top. A customer who has described their own cell passes
        # that JSON file as the first argument instead of `None`, and everything below is unchanged.
        # `--engine` names a key nested under `render`, and `from_file` is the one place that knows
        # it, which is why the engine is passed as a keyword rather than through `overrides`.
        build = DatasetBuild.from_file(None, name=args.name, scenes=plan.requested_scenes,
                                       engine=args.engine, out_root=args.out)
        for line in build.describe().splitlines():
            run.note(line)
        run.note("")

        with run.step("build the scenes") as report:
            rendered = build.render()
            report(f"{rendered.summary.get('by_status', {}).get('ok', 0)} scene(s) "
                   f"under {build.root}")
        if not rendered.ok:
            # Exit 2, not 1. Nothing rendered means there was nothing to build with, and the
            # commonest cause is an engine this interpreter cannot reach.
            return not_ready(
                rendered.reason,
                "for `isaac` you must run under Isaac's own python.bat; `none` and `mujoco` "
                "run under this repo's interpreter")

        with run.step("label the grasps") as report:
            # Closed-form geometry from the settled poses. No GPU and no model: this is the step
            # that turns a pile of renders into supervision.
            labelled = build.label()
            report(f"{labelled.summary.get('jaw', 0)} jaw and "
                   f"{labelled.summary.get('suction', 0)} suction label(s)")
        if not labelled.ok:
            # A finding does not move `run.exit_code`, so the status is returned explicitly. An
            # unlabelled dataset extracts into a structurally valid corpus that trains successfully
            # and teaches nothing, so stopping here is the whole point of checking the count.
            run.finding("label the grasps", labelled.reason)
            return EXIT_FAILED

        corpus = f"{args.out}/{args.name}_clouds"
        with run.step("extract the corpus the trainer reads") as report:
            extracted = build.clouds(corpus)
            report(f"{extracted.summary.get('scenes_written', 0)} scene(s) in {corpus}")
        if not extracted.ok:
            run.finding("extract the corpus the trainer reads", extracted.reason)
            return EXIT_FAILED

        run.note("")
        run.note(f"Next: 05_train.py --clouds {corpus}")
        run.note("")
        run.note("WARN before you scale this up, two things worth knowing:")
        run.note("  `datagen.heldout.held_out_assets(dataset_root)` lists the assets a trained")
        run.note("  model has not seen, and `datagen.heldout.format_report` prints that list. The")
        run.note("  argument is the dataset directory named above, not the cloud corpus. Without")
        run.note("  it, a model measured on its own training objects reads better than it is.")
        run.note("  To build from your own parts, copy them into the mesh library with")
        run.note("  `datagen.assets.library.import_from_directory('custom', <dir>, license=...)`,")
        run.note("  then set assets.custom_weight and zero the procedural weights. The licence has")
        run.note("  no default: it is yours to declare, and the licence audit reads that string.")
        run.note("  `datagen.assets.service.MeshPreparation.fetch` downloads the research")
        run.note("  collections; it has no parameter for a directory of your own meshes.")
        return run.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
