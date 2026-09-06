"""52: which engine settles and renders your scenes, and what each one costs you.

    python scripts/examples/datagen/52_which_engine.py                  # price all three, build on `none`
    python scripts/examples/datagen/52_which_engine.py --engine mujoco --scenes 20
    python scripts/examples/datagen/52_which_engine.py --scenes 2000 --engine isaac --price-only

The decision is `render.engine` in the datagen config, one of `isaac`, `mujoco` or `none`, and it is
a fork with an installation attached to two of its three answers.

  isaac   NVIDIA Isaac Sim, a separate multi-gigabyte install with its own bundled Python and an
          RTX-class card. The only backend that produces path-traced RGB.
  mujoco  `pip install mujoco`, a wheel of a few tens of megabytes on Windows, macOS, Linux and ARM.
          Settles the scene with a solver and rasterises depth and masks. No RGB.
  none    nothing beyond this repository's own requirements. Seats objects analytically, tests them
          against their support polygon, and rasterises depth and masks in numpy. No RGB, and it
          refuses the `pile` family by name rather than faking a settle.

This is not a fidelity preference. The contract between an engine and everything downstream is a
file format, not an API: depth in millimetres, per-object silhouettes, the settled poses, and the
camera pose and intrinsics each view used. `label-grasps` never opens an image at all. So a corpus
built for geometry is a corpus `none` can build, and an install nobody in your company will approve
is not the wall the plan hits.

Two things this file makes visible because both are easy to get wrong. A request is not a yield: the
engine-free backend refuses a whole family, so asking for 2,000 scenes leaves you fewer, and
`datagen.cost.estimate` answers in usable scenes and prints the request that reaches them. And the
factory fails closed: an engine this interpreter cannot reach is a refusal that names the fix, never
a quiet substitution, because a corpus filed under the wrong renderer looks fine forever.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import Example, not_ready  # noqa: E402

EPILOG = """
exit codes
  0  the three engines were priced, and (unless --price-only) the chosen one built its scenes
  1  the build ran and produced nothing usable
  2  the chosen engine cannot be used from this interpreter; the message names what is missing
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "", epilog=EPILOG,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", default="none", choices=("none", "mujoco", "isaac"),
                        help="which backend to build with. `none` needs nothing beyond this repo")
    parser.add_argument("--scenes", type=int, default=8, help="usable scenes to end up with")
    parser.add_argument("--jobs", type=int, default=1,
                        help="shards you intend to run in parallel. Priced, not started: there is "
                             "no parallel mode inside one run")
    parser.add_argument("--meshes", type=int, default=0,
                        help="how many scanned meshes the config draws, which is what the render "
                             "actually costs. 0 prices the procedural case")
    parser.add_argument("--name", default="example_corpus", help="dataset name")
    parser.add_argument("--out", default="logs/examples", help="dataset root")
    parser.add_argument("--price-only", action="store_true", help="print the bill and stop")
    args = parser.parse_args(argv)

    from datagen.api import DatasetBuild
    from datagen.config import DatagenConfig
    from datagen.cost import ENGINE_COSTS, estimate, format_estimate
    from datagen.render.engine import ENGINES, build_engine, engine_is_available

    with Example("52 engines", f"the same {args.scenes} scene(s), priced on three backends",
                 hardware=False) as run:

        # Both tables are collected inside their step and printed after it. On a terminal a step's
        # announcement is an open line, so anything the body prints lands on it.
        probed: list[str] = []
        with run.step("which of the three this interpreter can reach") as report:
            usable = []
            for name, what in ENGINES.items():
                available, why = engine_is_available(name)
                probed.append(f"  {name:8} {'available' if available else 'not available'}")
                probed.append(f"           {what}")
                if not available:
                    probed.append(f"           {why}")
                else:
                    usable.append(name)
            report(f"{len(usable)} of {len(ENGINES)} available here: {', '.join(usable)}")
        run.note("")
        for line in probed:
            run.note(line)
        run.note("")

        priced: list[str] = [
            f"  {'engine':8} {'request':>8} {'usable':>7} {'hours':>8} {'GB':>7}  yield"]
        with run.step("the same request, priced on each") as report:
            plans = {name: estimate(args.scenes, engine=name, jobs=args.jobs, meshes=args.meshes)
                     for name in ENGINES}
            for name, plan in plans.items():
                priced.append(f"  {name:8} {plan.requested_scenes:8d} {plan.usable_scenes:7d} "
                              f"{plan.hours:8.2f} {plan.gigabytes:7.2f}  "
                              f"{ENGINE_COSTS[name].yield_fraction:.0%}")
            report(f"{args.scenes} usable scene(s) costs {plans[args.engine].hours:.2f} h on "
                   f"`{args.engine}`")
        run.note("")
        for line in priced:
            run.note(line)
        run.note("")
        run.note("Read the first two columns together. They differ because a request is not a")
        run.note("yield: `none` refuses `pile` by name, and with the default equal family weights")
        run.note("that is a quarter of every request. A build asks for the first number so that")
        run.note("you end up with the second.")
        run.note("")
        for line in format_estimate(plans[args.engine]).splitlines():
            run.note(line)
        run.note("")

        # The refusal, on whichever engine this machine cannot reach. Run rather than described: it
        # is the message a reader will meet, and it names the fix rather than raising ImportError
        # eight frames down.
        missing = [name for name in ENGINES if not engine_is_available(name)[0]]
        if missing:
            with run.step(f"what a missing engine does: {missing[0]}") as report:
                try:
                    # `model_validate`, because the engine name is a string here rather than one of
                    # the three literals the schema declares. It is validated all the same: an
                    # engine this build does not know is refused by the config before the factory
                    # is reached at all.
                    build_engine(DatagenConfig.model_validate(
                        {"scenes": 1, "seed": 0, "render": {"engine": missing[0]}}))
                except RuntimeError as refusal:
                    report(" ".join(str(refusal).split())[:88])
                else:
                    run.finding("engine factory", f"{missing[0]} was built although its probe says "
                                                  f"it is not available here")
            run.note("")
            run.note("It refuses and never falls back. An engine that quietly substituted another")
            run.note("would file one backend's geometry under the other's name, and")
            run.note("`provenance.json` would then say so wrongly, which is worse than not")
            run.note("starting: the corpus would look fine.")
            run.note("")
        else:
            run.note("Every engine is available here, so the fail-closed refusal has nothing to")
            run.note("fire on. On a machine without Isaac it is what `--engine isaac` produces.")
            run.note("")

        if args.price_only:
            run.note("--price-only: nothing was built.")
            return run.exit_code

        available, why = engine_is_available(args.engine)
        if not available:
            return not_ready(f"--engine {args.engine} cannot be used here: {why}",
                             "run with `--engine none`, which needs nothing beyond this "
                             "repository, or `--engine mujoco` after `pip install mujoco`")

        plan = plans[args.engine]
        build = DatasetBuild.from_file(None, name=args.name, scenes=plan.requested_scenes,
                                       engine=args.engine, out_root=args.out)
        for line in build.describe().splitlines():
            run.note(line)
        run.note("")
        run.note(f"Rendering {plan.requested_scenes} scene(s). The backend prints one line each.")
        with run.step(f"build on `{args.engine}`") as report:
            rendered = build.render()
            status = dict(rendered.summary.get("by_status", {}))
            report(", ".join(f"{count} {name}" for name, count in sorted(status.items()))
                   or "nothing rendered")
        if not rendered.ok:
            return not_ready(rendered.reason,
                             "for `isaac` you must run under Isaac's own python.bat; `none` and "
                             "`mujoco` run under this repository's interpreter")
        refused = int(status.get("refused_family", 0))
        if refused:
            run.note("")
            run.note(f"{refused} scene(s) came back as `refused_family`, which is the yield above")
            run.note("arriving in the output rather than an error. A refusal is recorded per scene")
            run.note("with its reason, so a corpus can say what it does not contain.")

        run.note("")
        with run.step("what the dataset records about the choice") as report:
            stamp = json.loads((build.root / "provenance.json").read_text(encoding="utf-8"))
            report(f"renderer {stamp.get('renderer')}, {len(stamp)} provenance field(s)")
        run.note("")
        run.note("The engine is stamped into the dataset and into every point-cloud file the")
        run.note("corpus writes, because two engines never settle a scene identically and a corpus")
        run.note("that cannot say which one made it is a corpus nobody can compare. Nothing")
        run.note("refuses a mixed corpus: the field is recorded and recoverable, and no consumer")
        run.note("reads it back. Saying otherwise would describe a guard this package does not")
        run.note("have.")
        run.note("")
        run.note("Which to pick. Take `none` for a geometry corpus on a machine with nothing")
        run.note("installed, and accept that it has no pile family and no images. Take `mujoco`")
        run.note("when you want the pile, which is the family where objects occlude and stack, or")
        run.note("when a real solver settling the contact matters; it is one wheel and no GPU.")
        run.note("Take `isaac` when you need path-traced RGB, which is to say when something")
        run.note("downstream of the corpus looks at an image rather than at geometry.")
        run.note("")
        run.note("Two costs that are about your assets rather than about the engine. The cheap")
        run.note("backends rasterise triangles in Python, so a scanned mesh is orders of magnitude")
        run.note("slower than a procedural box: pass --meshes N to price that case. And MuJoCo")
        run.note("collides meshes as convex hulls, so a concave mesh is split first; warm the")
        run.note("cache with `MeshPreparation.decompose(config, jobs=8)` before a build, or the")
        run.note("render pays for it one mesh at a time inside its own loop.")
        run.note("")
        run.note("Next: 53_label_and_screen.py, on the dataset this just wrote.")
        return run.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
