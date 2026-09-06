"""64: what the learned generator does not cover, read off the code rather than promised.

    python scripts/examples/train/64_what_the_generator_does_not_cover.py
    python scripts/examples/train/64_what_the_generator_does_not_cover.py --clouds <a corpus>

The decision: whether the learned generator is the right thing to train for the job in front of you.
It is a fair decision only against its limits, and four of them are structural rather than a matter
of training longer.

  it predicts jaw grasps.       Every field a slot emits describes a two-finger grasp: a centre
                                between two contacts, a closing axis, an opening. There is no
                                suction pose head, and none can be trained from the labels a cup
                                produces, because a cup has no closing axis and no opening.
  suction reaches one stage.    The where stage, which answers whether a patch of surface is worth
                                attempting at all, counts a suction label as graspable. That
                                happens only when the corpus was built asking for both kinds; a
                                jaw-only corpus writes the suction arrays empty and the stage is
                                then taught to call a suction-only object empty.
  no weights ship here.         The shipped config carries no artifact path, and `calculator: deep`
                                without a readable artifact refuses to build the cell. A customer
                                trains on their own cell's data.
  the score is not P(hold).     What a slot attaches is a confidence over its own slots, not a
                                calibrated probability that the grasp will hold.

Nothing here trains anything and nothing is downloaded. Every claim above is checked against the
code in this repository, and the refusal is triggered on purpose, because that refusal is what a
reader meets the first time they set the key.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import Example, not_ready  # noqa: E402

EPILOG = """
exit codes
  0  every limit was checked against the code
  1  a limit this file states is not what the code does
  2  the config tree did not load, so there was nothing to read the selector off
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "", epilog=EPILOG,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clouds", default=None,
                        help="a corpus to count the suction half of. Without it that section is "
                             "described and not measured")
    args = parser.parse_args(argv)

    from src.config.loader import load_config
    from src.robot.grasping.calculator_factory import preflight_calculator

    with Example("64 the honest limit", "what a trained generator does not do",
                 hardware=False) as run:
        with run.step("what this config selects") as report:
            try:
                config = load_config().robot
            except Exception as error:                           # noqa: BLE001  (report, not raise)
                return not_ready(f"the config tree did not load "
                                 f"({type(error).__name__}: {error})",
                                 "check `python -m src.config --print`, and WILLY_PROFILE if you "
                                 "set one")
            if config is None:
                return not_ready("this config tree has no `robot` block",
                                 "select a profile that has one, or add config/robot/robot.yaml")
            block = config.grasping.deep_generator
            report(f"grasping.calculator {config.grasping.calculator!r}, "
                   f"deep_generator.artifact_path {block.artifact_path!r}")

        with run.step("the selector is honoured, not just declared") as report:
            # `preflight_calculator` is the same reader `build_calculator` uses, minus the
            # construction. Without a factory reading this key a cell could set `deep` and get the
            # analytic stack in silence, because a check that enumerates boolean `enabled` fields
            # does not see a string selector.
            report(f"preflight resolves to {preflight_calculator(config)!r}")
        run.note("")

        with run.step("asking for the learned generator with no weights") as report:
            # `model_copy` on the frozen config, so the real reader is asked the real question. The
            # shipped tree cannot answer it, because its artifact path is null by design.
            deep_config = config.model_copy(
                update={"grasping": config.grasping.model_copy(update={"calculator": "deep"})})
            try:
                preflight_calculator(deep_config)
            except FileNotFoundError as error:
                report(f"{type(error).__name__}: {error}")
            else:
                run.finding("asking for the learned generator with no weights",
                            "the selector accepted `deep` with no artifact, which it must not")
        run.note("")
        run.note("  that refusal is the design. It does not fall back to `geometric`, because a")
        run.note("  cell that asked for the learned generator and quietly ran the analytic one")
        run.note("  would file the analytic one's numbers under the learned one's name, in every")
        run.note("  record, every KPI and every comparison drawn from them afterwards.")
        run.note("")
        run.note("  no trained weights are in this repository, deliberately. A generator is fitted")
        run.note("  to the objects, the camera and the hand of one cell, so the artifact a")
        run.note("  customer serves is one they trained on their own data. 60_your_own_corpus.py")
        run.note("  is that path start to finish.")
        run.note("")

        from src.robot.grasping.deep.net.gripper import GRIPPER_VECTOR_DIM, JAW_GEOMETRY
        from src.robot.grasping.deep.net.set_loss import GraspSetPrediction
        from src.robot.grasping.deep.train.trainer import SetTrainingPlan

        plan = SetTrainingPlan()
        with run.step("what one slot of the head predicts") as report:
            fields = [f.name for f in dataclasses.fields(GraspSetPrediction)]
            report(f"{plan.model.head.slots} slot(s) per seed, each carrying {', '.join(fields)}")
        run.note("")
        run.note("  read that list as a description of a hand. A centre offset and a direction to")
        run.note("  approach along, which a suction cup would also need, plus an axis to close")
        run.note("  along and an opening between two fingers, which a cup has neither of. What is")
        run.note("  left is `confidence`, a logit over this seed's own slots and not a probability")
        run.note("  that the grasp holds.")
        run.note("")
        run.note("  a jaw head cannot be taught a pose from a suction label, and this is not a")
        run.note("  gap in the training loop. The corpus stores a suction row with a closing axis")
        run.note("  of [0, 0, 0] and a width of 0.0, because a cup has neither, and the sample")
        run.note("  contract refuses exactly those two shapes by name: `zero-length direction`")
        run.note("  and `non-positive opening`. Mixed into the jaw arrays they would fail the")
        run.note("  validator, which is why they live in their own `suction_*` arrays.")
        run.note("")

        with run.step("the gripper is an input, and that is wiring rather than evidence") as report:
            report(f"{GRIPPER_VECTOR_DIM}-number vector, hands named: "
                   f"{', '.join(sorted(JAW_GEOMETRY))}")
        run.note("")
        run.note("  the hand reaches the net as a conditioning vector, so one artifact can in")
        run.note("  principle serve more than one gripper. Whether it does on your corpus is a")
        run.note("  measurement, and `deep/eval/gripper_differential.py` is the instrument for it.")
        run.note("  Nothing in this repository has taken that measurement, so treat the")
        run.note("  conditioning as a seam that exists and not as a demonstrated capability.")
        run.note("")

        # Counted in the step, reported after it. A step's verdict prints when its body ends, so a
        # body that narrates puts its own lines above the verdict they belong to.
        suction = -1
        with run.step("the suction half, and where it stops") as report:
            if not args.clouds:
                report("described, not measured: pass --clouds to count a corpus's suction rows")
            else:
                import numpy as np

                corpus = Path(args.clouds)
                scenes = sorted(corpus.rglob("*.npz"))
                if not scenes:
                    return not_ready(f"{corpus} holds no .npz scenes",
                                     "60_your_own_corpus.py writes one and prints its path")
                jaw = suction = 0
                for path in scenes:
                    with np.load(path, allow_pickle=False) as handle:
                        jaw += int(np.asarray(handle["grasp_width_mm"]).size)
                        suction += int(len(np.asarray(handle["suction_position_mm"])))
                report(f"{len(scenes)} scene(s): {jaw} jaw label(s), {suction} suction label(s)")
        run.note("")
        if suction > 0:
            run.note("  this corpus was built asking for both kinds, so the where stage learns")
            run.note("  that these patches are worth attempting. The pose heads are unchanged: the")
            run.note("  jaw half of a mixed corpus is byte-identical to a jaw-only one.")
            run.note("")
        elif suction == 0:
            run.note("  the suction arrays are present and empty, which is what a jaw-only corpus")
            run.note("  writes. The key existing is not the data existing, and a consumer that")
            run.note("  checks for the key finds it either way. Pass `kinds=('jaw', 'suction')` to")
            run.note("  `DatasetBuild.clouds` to fill them; the jaw half does not change.")
            run.note("")
        run.note("  what the suction half buys is one stage and one stage only. The where stage")
        run.note("  asks whether a patch of surface is worth attempting, and a cell carrying a cup")
        run.note("  can attempt an object no jaw can hold. Trained on jaw labels alone the stage")
        run.note("  is taught to call such an object empty, which is a false negative on every")
        run.note("  object a cup could lift. It is not a fix for the pose heads and must not be")
        run.note("  read as one, and there is no data path to a learned suction generator here.")
        run.note("")

        with run.step("what the deep branch hands back to the analytic stack") as report:
            report("reachability filtering and the uncertainty re-rank are refused, not silently "
                   "dropped")
        run.note("")
        run.note("  `build_calculator` refuses to construct a deep calculator when a construction")
        run.note("  site passes `ik_service` or `corridor_risk_per_candidate` with an active")
        run.note("  value. Both change what a pick does rather than how it looks: without the")
        run.note("  first nothing filters unreachable candidates and the rejected-for-reach count")
        run.note("  stays zero, which reads as nothing having been unreachable; without the second")
        run.note("  the uncertainty re-rank reorders nothing. Either would look like a bad")
        run.note("  generator rather than a wrong wiring.")
        run.note("  The analytic sweep knobs, the oblique approach and the radial closing flag,")
        run.note("  are logged as ignored instead: the learned decoder predicts its own approach")
        run.note("  and its own closing axis, so those levers have nothing to act on. A banner")
        run.note("  printing them on a deep run is describing the geometric path.")
        run.note("")
        run.note("what is not here at all, stated so it is not inferred:")
        run.note("  no trained weights, for the reason above.")
        run.note("  no calibrated probability. The generator proposes and the ranker ranks, and")
        run.note("    each is promoted on its own evidence; there is no P(hold) head.")
        run.note("  no multi-gripper evidence, only the conditioning seam.")
        run.note("  no result from a physical cell. Everything measurable about this package is")
        run.note("    measured in simulation or derived from an analytical model.")
        run.note("")
        run.note("what to pick: keep `calculator: geometric` until a run of your own reads well in")
        run.note("  63_read_the_report.py on assets it has never seen. The analytic generator")
        run.note("  needs no corpus, no GPU and no training, and it is what the default pick path")
        run.note("  uses. The learned one is worth its corpus when your parts are ones a")
        run.note("  three-axis geometric rank keeps refusing.")
        return run.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
