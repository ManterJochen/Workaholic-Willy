"""53: the labels are geometry, and the screen is physics. Which of the two you ship.

    python scripts/examples/datagen/53_label_and_screen.py
    python scripts/examples/datagen/53_label_and_screen.py --physics-engine mujoco --per-class 10
    python scripts/examples/datagen/53_label_and_screen.py --no-screen

Labelling is the step that turns a pile of renders into supervision, and it is closed form. For
every object in every settled scene it enumerates the grasps the geometry admits, from the antipodal
structure rather than by searching, and writes them to `grasps.jsonl`. No GPU, no model, minutes
rather than hours, and re-runnable on its own, which matters because re-labelling a dataset is
minutes and re-rendering it is a night.

What comes out is a reference rather than a second opinion: it is computed from the solid the
renderer authored and the pose the physics left it in, it never opens a depth image, and it never
imports the grasp stack it exists to grade. What it does not know is a short and specific list.
There is no inverse kinematics in it, so a grasp it calls real may be unreachable by the arm you
own. There is no dynamics, so a valid grasp is one that closes inside the friction cone rather than
one that provably survives being lifted.

That last gap is this file's decision. `datagen.grasps.service.PhysicsSampling` teleports a gripper
to the labelled pose, closes it, takes the table away, and reports whether the object stayed. The
verdict joins back to the label by file and line, and `DatasetBuild.clouds(physics=...)` folds it
into the corpus as `grasp_held`.

Three properties of that screen decide whether you should run it.

It needs an engine, `mujoco` or `isaac`, and four controls run before any trial: the shut jaw must
be solid, a textbook grasp through a lone block must hold, the same grasp made to grip air must not,
and the first control repeated after an unrelated scene must still hold. A harness that silently
reports that nothing holds is indistinguishable from one that is wired wrong.

It refuses rather than guessing. In a full bin the teleport drives the gripper into an overlap the
solver cannot resolve, and that trial is refused, not scored. A refused trial is not a failed grasp,
and `physics_verdicts` drops refusals rather than folding them into `held=False`.

And it is jaw-only. `sample_trials` skips every row whose kind is not `jaw`, because a teleported
parallel jaw is not an instrument for a suction cup.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import EXIT_FAILED, Example, not_ready  # noqa: E402

EPILOG = """
exit codes
  0  the dataset was labelled, and the screen ran or was declined
  1  a stage ran and produced nothing: a dataset with no labels is the case worth stopping for
  2  nothing to label: no dataset on disk, or no physics engine for --physics-engine
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "", epilog=EPILOG,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="logs/examples/example_corpus",
                        help="the dataset 52_which_engine.py wrote")
    parser.add_argument("--density", default=None, choices=("default", "dense", "grid"),
                        help="how finely the labeller samples each object. Not passed at all when "
                             "you leave it off, which is different from passing the default")
    parser.add_argument("--physics-engine", default="mujoco", choices=("mujoco", "isaac"),
                        help="the referee. `mujoco` is a pip wheel and needs no GPU")
    parser.add_argument("--per-class", type=int, default=4,
                        help="trials per stratum. The draw is stratified by source and family, so "
                             "this is not a total")
    parser.add_argument("--no-screen", action="store_true",
                        help="label only, and print what the screen would have added")
    args = parser.parse_args(argv)

    dataset = Path(args.dataset)
    if not (dataset / "scenes").is_dir():
        return not_ready(f"no rendered dataset at {dataset}",
                         "run 52_which_engine.py first; it prints the dataset path it wrote")

    from datagen.api import DatasetBuild
    from datagen.config import DatagenConfig
    from datagen.corpus.clouds import physics_verdicts
    from datagen.grasps.service import PhysicsSampling
    from datagen.render.engine import engine_is_available

    # `from_config` rather than `from_file`: the label and the screen work on a dataset that is
    # already on disk, so nothing here needs the config that rendered it. The name and the root are
    # what address the dataset.
    build = DatasetBuild.from_config(DatagenConfig(scenes=1, seed=0), name=dataset.name,
                                     out_root=dataset.parent)

    with Example("53 labels", f"the grasps in {dataset.name}, and whether they hold",
                 hardware=False) as run:

        run.note("Labelling every scene. On a large dataset this is the stage worth watching.")
        with run.step("label the grasps, closed form") as report:
            labelled = build.label(density=args.density)
            report(f"{labelled.summary.get('jaw', 0)} jaw and "
                   f"{labelled.summary.get('suction', 0)} suction label(s) over "
                   f"{labelled.summary.get('objects', 0)} object(s)")
        if not labelled.ok:
            # Not a finding, because the failure is silent downstream. An unlabelled dataset
            # extracts into a structurally valid corpus that trains successfully and teaches
            # nothing, so the exit code has to carry it.
            run.finding("label the grasps", labelled.reason)
            return EXIT_FAILED

        report_path = dataset / "grasp_label_report.json"
        summary = json.loads(report_path.read_text(encoding="utf-8"))
        density = summary.get("density", {})
        run.note("")
        run.note(f"  jaw model    {summary.get('jaw_model')}")
        run.note(f"  density      {density.get('approach_azimuths')} approach azimuth(s), "
                 f"{density.get('radial_axes')} radial axis/axes per anchor")
        run.note(f"  scenes       {summary.get('scenes')} of {summary.get('scenes_available')} "
                 f"available, partial={summary.get('partial')}")
        run.note("")
        run.note("The density is in the report because without it the label count is")
        run.note("uninterpretable: two corpora sampled at different densities differ by a factor")
        run.note("for no reason a reader can see, and that difference is a fact about the sampler")
        run.note("rather than about the objects. Same for the jaw model: a label count from a")
        run.note("140 mm jaw and one from an 85 mm jaw are not comparable.")
        run.note("")

        # Collected inside the step, printed after it: a step's announcement is an open line on a
        # terminal, and anything the body prints lands on it.
        histogram: list[str] = []
        with run.step("why the geometry refused the candidates it refused") as report:
            rejected = dict(summary.get("rejected", {}))
            total = sum(rejected.values()) or 1
            for reason, count in sorted(rejected.items(), key=lambda row: -row[1])[:6]:
                histogram.append(f"  {reason:26} {count:>9,}  {count / total * 100:5.1f} %")
            report(f"{total:,} rejection(s) over {len(rejected)} reason(s)")
        run.note("")
        for line in histogram:
            run.note(line)
        run.note("")
        run.note("This histogram is worth reading rather than hiding. Approach directions are not")
        run.note("pre-filtered, so an approach from underneath is measured as `below_table` and")
        run.note("counted, which is why that reason tends to lead. `finger_collision` is the one")
        run.note("about your scene: it is the neighbour, or the bin wall, in the way of a jaw that")
        run.note("would otherwise close.")
        run.note("")
        run.note("What these labels do not know, in full: reachability, because there is no")
        run.note("inverse kinematics here and no verdict depends on which arm is bolted to the")
        run.note("table; dynamics, which is the screen below; and the sensor, because the geometry")
        run.note("is the settled truth rather than what a camera could see of it. On a scanned")
        run.note("mesh the label set is also sound but not complete: a box admits three closing")
        run.note("axes and they can all be listed, a scanned surface admits a continuum and the")
        run.note("labeller samples it, so a missing label there proves nothing.")
        run.note("")

        if args.no_screen:
            run.note("--no-screen: the labels above are what this dataset ships with, and every")
            run.note("one of them is a geometric claim. The screen is")
            run.note("  PhysicsSampling.from_dataset(<dataset>, engine='mujoco').sample()")
            run.note("and its verdicts reach a corpus through")
            run.note("  DatasetBuild.clouds(out_dir, physics=<dataset>/grasp_physics.jsonl)")
            return run.exit_code

        available, why = engine_is_available(args.physics_engine)
        if not available:
            return not_ready(f"the {args.physics_engine} referee cannot run here: {why}",
                             "pass --physics-engine mujoco after `pip install mujoco`, or "
                             "--no-screen to stop at the analytic labels")

        screen = PhysicsSampling.from_dataset(dataset, engine=args.physics_engine)
        for line in screen.describe().splitlines():
            run.note(line)
        run.note("")
        run.note("Four controls run before the first trial; the pass refuses to continue without")
        run.note("them.")
        with run.step(f"shake them, refereed by {args.physics_engine}") as report:
            verdict = screen.sample(per_class=args.per_class)
            report(f"{verdict.held} of {verdict.trials} scored trial(s) held, "
                   f"{verdict.refused} refused")
        run.note("")
        for line in verdict.render().splitlines():
            run.note(line)
        run.note("")
        run.note("Read the strata, not the pooled rate. The draw is stratified by source and")
        run.note("family, so a pooled number describes the sampler as much as the data. A dataset")
        run.note("that has only been labelled has one stratum, `label`; the accepted and rejected")
        run.note("candidate strata come from `eval-grasps`, and they are the half that shows")
        run.note("whether the analytic verdict discriminates at all. If the grasps it rejected")
        run.note("fail no more often than the ones it accepted, the predicate is decoration.")
        run.note("")

        with run.step("the join back to the labels") as report:
            verdicts = physics_verdicts(dataset / verdict.out_name)
            held = sum(1 for value in verdicts.values() if value)
            report(f"{len(verdicts)} label row(s) now carry a verdict, {held} of them `held`")
        run.note("")
        run.note("Keyed by file and line, never by pose: a label row and the candidate row")
        run.note("describing the same grasp carry identical poses, so a pose join is ambiguous by")
        run.note("construction. Refusals are absent from that map on purpose. Re-running a refused")
        run.note("trial in a fresh session leaves most of them refused, so a refusal is a property")
        run.note("of the grasp rather than of the moment, and folding it into `held=False` would")
        run.note("record an unmeasured grasp as a measured failure.")
        run.note("")
        run.note("Which to ship. The labels alone are enough to train on and they are what the")
        run.note("corpus carries by default; every one is a real grasp under the model, and the")
        run.note("model has no dynamics. Run the screen when you are about to make a claim that")
        run.note("rests on grasps holding rather than on grasps existing, and read it with its two")
        run.note("limits in hand: it is jaw-only, and a teleported gripper is the wrong instrument")
        run.note("in a full bin, which is what the refusals are.")
        run.note("")
        run.note("Next: 54_build_a_corpus.py, which reads the verdict file above with")
        run.note(f"  --physics {dataset / verdict.out_name}")
        return run.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
