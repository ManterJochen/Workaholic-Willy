"""21: the prompts a phrase grounder answers confidently and wrongly, and where they should go.

    python scripts/examples/perception/21_when_the_detector_is_wrong.py
    python scripts/examples/perception/21_when_the_detector_is_wrong.py --prompt "not the red cube"
    python scripts/examples/perception/21_when_the_detector_is_wrong.py --profile sim --run

Two decisions live here, and they are not the same decision.

`models.pipeline.zero_shot.backend` chooses which model grounds. `grounded_sam` is GroundingDINO,
which grounds a phrase. `vlm` puts Qwen3-VL in the same detector slot, and it is the only route in
this stack that reaches negation, comparatives, relative clauses, quantifiers and a prompt that is
not in English. `models.pipeline.router.enabled` then decides per prompt, from the text alone,
which of the two answers it.

`models.pipeline.zero_shot.vlm.on_unavailable` chooses what a cell does when that model cannot be
loaded. `refuse` rejects the prompt with a typed error carrying the cause. `degrade` grounds it
with the phrase detector instead and warns on every use.

The reason the second decision is not obvious is the reason the first one exists. The phrase
grounder does not fail on a prompt it cannot represent. It returns a high-scoring box for the
wrong object, and nothing downstream, not the gate, not the record, not the operator, can tell that
apart from a correct answer. So a quiet fallback is not slightly worse perception; it is the cell
grasping something nobody asked for. That is also why there is no cheap-first cascade here: a
cascade needs its cheap stage to fail loudly, and this one does not.

Two warnings belong on this page and they point in opposite directions.

Nothing in `src/models/vlm/` has ever run against real weights in this repository. The parser, the
coordinate space and the unavailability contract are exercised without a GPU; the grounding
quality, the VRAM cost and the choice between checkpoints are unmeasured, and any figure attributed
to this route is an estimate. Measure it on your own cell before you rely on it.

And `refuse` against `degrade` is a live decision even on a cell that never configures the VLM,
because the answer decides whether a missing model stops the cell or quietly hands back a worse
answer that looks exactly like a good one.

The router is free. It is deterministic, reads the prompt text and nothing else, loads no model and
sees no image, so every routing verdict below is computed on this machine whatever is installed.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import EXIT_FAILED, Example, not_ready  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover (typing only)
    import numpy as np

    from src.config.schema.models.models_schema import PipelineConfig
    from src.models.perception_spec import PerceptionSpec

EPILOG = """
exit codes
  0  the routes and the four answers were reported, and with --run the stack answered
  1  this cell cannot build the stack its config selects; the refusal is printed above
  2  this tree has no models block, or the weights that stack needs are not on this box

A prompt that this cell would ground with the wrong model is a finding, not a failure: it is a
statement about the configuration, and the run still exits 0.
"""

#: One prompt per routing reason the rules can report, so the panel covers the vocabulary rather
#: than sampling it. Each is written the way an operator would type it at a bench. The first is
#: the control: nothing in it is beyond a phrase grounder.
_PANEL: tuple[str, ...] = (
    "a red cube",
    "the small red plastic cube",
    "not the red cube",
    "the largest cube",
    "the cube that is on the box",
    "the cube to the left of the tray",
    "every cube",
    "the cube and the cup",
    "the broken part",
    "greif den kaputten Wuerfel",
)


def _zero_shot_pipeline(
    pipeline: "PipelineConfig | None", *, backend: str, router: bool, on_unavailable: str,
) -> "PipelineConfig":
    """This tree's own zero-shot block with the three keys under discussion set explicitly.

    Rebuilt field by field rather than copied, so the schema's cross-checks run again: routing
    without a VLM to route to is refused by the validator, and a copy that skipped validation would
    hand back a block the loader would have rejected.

    The tree's own `vlm` sub-block is carried through rather than defaulted. The schema default
    names the FP8 checkpoint and `config/models/object.yaml` names the bf16 one, so a fresh default
    here would print a model id this cell does not use.
    """
    from src.config.schema.models.models_schema import (
        PipelineConfig,
        PromptRouterConfig,
        VlmConfig,
        ZeroShotPipelineConfig,
    )

    zero_shot = pipeline.zero_shot if pipeline is not None else ZeroShotPipelineConfig()
    vlm = zero_shot.vlm if zero_shot.vlm is not None else VlmConfig()
    return PipelineConfig(
        kind="zero_shot",
        zero_shot=ZeroShotPipelineConfig(
            backend=backend,                                    # type: ignore[arg-type]
            segmenter=zero_shot.segmenter,
            vlm=VlmConfig(
                model_id=vlm.model_id, model_path=vlm.model_path, local=vlm.local,
                preload=vlm.preload,
                on_unavailable=on_unavailable,                  # type: ignore[arg-type]
            ),
        ),
        # Named explicitly on every branch. The schema default is true and the shipped YAML says
        # false, so a block built without this key routes when the tree does not.
        router=PromptRouterConfig(enabled=router),
    )


def _answer(spec: "PerceptionSpec", label: str, pipeline: "PipelineConfig") -> tuple[str, bool]:
    """`(rendered answer, buildable)` for one configuration of the three keys, loading nothing."""
    resolution = replace(spec, pipeline=pipeline).resolve()
    return f"{label}\n{resolution.render()}", resolution.buildable


def _weights_on_this_box(model_id: str) -> bool | None:
    """Is `model_id` already in the local Hugging Face cache? Never downloads and never raises.

    `None` means the question could not be asked at all, which is not the same answer as absent.

    There is no library function for this, and that is a gap rather than an oversight. The console's
    diagnostics endpoint and a test guard each run this same `snapshot_download(local_files_only=
    True)` probe, `datagen.prompts.model.weights_present` runs a third against its own
    in-repository cache, and `scripts/model_weights/fetch.py` records the gap in its own docstring.
    The twin it wants is one function in `src/models/`, beside `PerceptionSpec`, taking a model id
    and answering present or absent, called by all of them.
    """
    if not model_id:
        return None
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=model_id, local_files_only=True)
    except ImportError:
        return None
    except Exception:                               # noqa: BLE001  (absent is itself the answer)
        return False
    return True


def _two_square_scene() -> "np.ndarray":
    """A grey field with a red square on the left and a blue one on the right, 80 px each.

    A drawn image, not a camera and not a claim about a scene. Two squares rather than one because
    the question this file asks needs a wrong answer to be available: a prompt that excludes the
    red one has somewhere else to land, and a grounder that ignores the exclusion has something to
    be caught doing. Red is centred at (200, 240), blue at (440, 240). 22 is where a camera
    appears.
    """
    import numpy as np

    image = np.full((480, 640, 3), 200, dtype=np.uint8)
    image[200:280, 160:240] = (40, 40, 220)                     # BGR, so this is red
    image[200:280, 400:480] = (220, 60, 40)                     # BGR, so this is blue
    return image


def _which_square(x_center: float) -> str:
    """Which drawn square a returned box centre landed on. `_two_square_scene` sets the numbers."""
    if 160.0 <= x_center <= 240.0:
        return "the RED square"
    if 400.0 <= x_center <= 480.0:
        return "the BLUE square"
    return "neither square"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "", epilog=EPILOG,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt", default=None,
                        help="one more prompt to route, added to the panel. Type the wording you "
                             "would use at your own bench")
    parser.add_argument("--run", action="store_true",
                        help="load the weights this cell configures and ground the routed prompts "
                             "on one drawn frame. Without it nothing reaches a GPU")
    parser.add_argument("--profile", default=None,
                        help="WILLY_PROFILE to load. `sim` is the one that names Hub ids instead "
                             "of the local weight directories a fresh checkout does not have")
    args = parser.parse_args(argv)

    # After the parse, so --help answers in a checkout whose dependencies are not installed.
    from src.config import load_config
    from src.models.perception_spec import PerceptionSpec
    from src.models.routing import Route, route

    prompts = _PANEL + ((args.prompt,) if args.prompt else ())
    with Example("21 when_the_detector_is_wrong", f"{len(prompts)} prompts, routed and priced",
                 hardware=False, profile=args.profile) as run:
        with run.step("load the config tree") as report:
            models = load_config().models
            spec = PerceptionSpec.from_config(models)
            here = spec.resolve()
            report(f"{here.kind} / {here.detector} + {here.segmenter}, "
                   f"router {'on' if here.router_enabled else 'off'}")

        with run.step("route every prompt, with no model and no image") as report:
            decisions = {prompt: route(prompt) for prompt in prompts}
            hard = [p for p, d in decisions.items() if d.route is Route.VLM]
            report(f"{len(hard)} of {len(prompts)} need the VLM route")
        for prompt, decision in decisions.items():
            run.note(f"  {prompt!r:38.38} {decision.describe()}")
        run.note("")
        run.note("The reason names the linguistic property the phrase grounder cannot represent,")
        run.note("and the first matching rule wins, so a prompt that trips several reports the")
        run.note("most specific one. This is a vocabulary rather than a parser, and the edges are")
        run.note("sharp: 'to the left of' is in it and a bare 'left of' is not, so dropping two")
        run.note("words from the spatial prompt above sends it to the phrase grounder. Language is")
        run.note("read off accented letters alone, so an ASCII-only non-English prompt routes")
        run.note("simple and grounds badly.")

        run.note("")
        with run.step("what this cell would actually do with them") as report:
            # The route a prompt wants and the route a cell has are different facts, and the gap
            # between them is the whole subject of this file.
            reachable = here.detector in ("vlm", "routed")
            report("the VLM route is configured" if reachable else
                   f"no VLM route: all {len(hard)} hard prompt(s) go to the phrase grounder")
        if not reachable and hard:
            run.finding("the hard prompts have nowhere better to go",
                        f"{len(hard)} of {len(prompts)} would be ground by a model that does not "
                        f"fail on them")
            run.note("  A confident box for the wrong object is what those prompts get. The pick")
            run.note("  that follows is a normal-looking pick on the wrong object.")

        run.note("")
        run.note("The four answers this fork offers, resolved and loading nothing:")
        run.note("")
        with run.step("resolve the four answers") as report:
            answers = [
                _answer(spec, "backend: grounded_sam            (this stack's default)",
                        _zero_shot_pipeline(models.pipeline, backend="grounded_sam",
                                            router=False, on_unavailable="refuse")),
                _answer(spec, "backend: vlm, router: false      (every prompt to the VLM)",
                        _zero_shot_pipeline(models.pipeline, backend="vlm",
                                            router=False, on_unavailable="refuse")),
                _answer(spec, "backend: vlm, router: true       (each prompt to its own route)",
                        _zero_shot_pipeline(models.pipeline, backend="vlm",
                                            router=True, on_unavailable="refuse")),
                _answer(spec, "the same, with on_unavailable: degrade",
                        _zero_shot_pipeline(models.pipeline, backend="vlm",
                                            router=True, on_unavailable="degrade")),
            ]
            report(f"{sum(1 for _, ok in answers if ok)} of {len(answers)} buildable on this tree")
        for block, _ in answers:
            for line in block.splitlines():
                run.note(line)
            run.note("")

        model_id = here.vlm_model_id or _vlm_model_id(models)
        with run.step("is that checkpoint on this box") as report:
            present = _weights_on_this_box(model_id)
            if present is None:
                report("cannot ask: this tree names no VLM checkpoint, or huggingface_hub is "
                       "not installed")
            elif present:
                report(f"{model_id}: in the local Hugging Face cache")
            else:
                report(f"{model_id}: NOT on this box, so a cell switched to the VLM route would "
                       f"meet on_unavailable on its first complex prompt")
        run.note("")
        run.note("What on_unavailable decides, for the moment that checkpoint cannot be loaded:")
        run.note("  refuse : the prompt is rejected, with the cause named, and nothing is grasped.")
        run.note("  degrade: the phrase grounder answers instead and every such answer is warned")
        run.note("           about, so a run that has fallen back never looks normal again.")
        run.note("  Only three shapes of a missing model degrade at all: no dependency, no weights")
        run.note("  on disk, no VRAM. Any other exception is a bug and is left to surface.")
        run.note("  `degrade` with no `models.objectdetector` block is refused at build time")
        run.note("  rather than mid-pick, because there would be nothing to fall back to.")
        run.note("")
        run.note("The sentence a refusing cell prints, with the cause filled in at the time:")
        for line in textwrap.wrap(_refusal_sentence(model_id), width=84):
            run.note(f"  {line}")

        if not args.run:
            run.note("")
            run.note("Nothing was loaded. `--run` builds the stack this config selects and grounds")
            run.note("the routed prompts on one drawn frame, which is where a confident wrong")
            run.note("answer stops being a paragraph and becomes a box you can read.")
            run.note("")
            run.note("Next: 22_depth_source.py, for where the depth behind those boxes comes from.")
            return run.exit_code

        if not here.buildable:
            # Exit 1, not 2. This configuration cannot be built at all, and no fetch changes that.
            run.finding("build the configured stack", here.refusal)
            run.note("This cell cannot build the stack its own config selects. Fix that first.")
            return EXIT_FAILED

        run.note("")
        backend: Any = None
        with run.step(f"build {here.kind} / {here.detector}") as report:
            try:
                backend = spec.build()
            except Exception as error:                          # noqa: BLE001  (report, not raise)
                return not_ready(
                    f"the stack did not build ({type(error).__name__}: {error})",
                    "fetch the weights (`python scripts/model_weights/fetch.py dino-tiny sam2`) "
                    "and run with --profile sim, which names Hub ids instead of local directories")
            report(type(backend).__name__)

        image = _two_square_scene()
        run.note("")
        run.note("One drawn frame: a red square centred at (200, 240), a blue one at (440, 240).")
        run.note("Read the last column. A prompt that excludes the red square and comes back with")
        run.note("the red square is the confident wrong answer this whole file is about.")
        with run.step("ground the routed prompts") as report:
            landed: list[tuple[str, str]] = []
            grounded = 0
            for prompt in prompts:
                objects = backend.perceive(image, prompt)
                if not objects:
                    landed.append((prompt, "nothing grounded"))
                    continue
                grounded += 1
                best = objects[0]
                # The score is printed because "confident" is the claim this file makes about a
                # wrong answer, and a score is the only evidence for it available out here.
                landed.append((prompt, f"{best.detection.label!r} at {best.detection.score:.2f} "
                                       f"on {_which_square(best.detection.x_center)}"))
            report(f"{grounded} of {len(prompts)} came back with a box")
        for prompt, where in landed:
            run.note(f"  {prompt!r:38.38} {where}")
        run.note("")
        run.note("Nothing above is asserted. The square each box landed on is computed from the")
        run.note("geometry this file drew, and the label and the score are printed as they came")
        run.note("out. Every box shown cleared the configured threshold, and none of them carries")
        run.note("any mark saying the prompt was beyond the model: that is the failure this file")
        run.note("exists for, and the one no downstream stage can detect.")
        return run.exit_code


def _vlm_model_id(models: Any) -> str:
    """The checkpoint this tree names, even where the resolution had no reason to report one.

    `PerceptionResolution.vlm_model_id` is filled only on a stack that actually reaches the VLM, so
    on the shipped `grounded_sam` tree it is `None` while the config still names a checkpoint. The
    question "is that checkpoint here" is worth answering before the backend is switched, not after.
    """
    pipeline = getattr(models, "pipeline", None)
    if pipeline is None:
        return ""
    vlm = getattr(pipeline.zero_shot, "vlm", None)
    return str(getattr(vlm, "model_id", "") or "")


def _refusal_sentence(model_id: str) -> str:
    """`VlmUnavailableError`'s own text, read rather than paraphrased.

    Built with no cause, because there is no failure here to describe: this is the shape of the
    sentence, and a real refusal fills in whether the weights are missing, a dependency is absent
    or the card is out of memory.
    """
    from src.models.vlm import VlmUnavailableError

    return str(VlmUnavailableError(model_id or "<the configured checkpoint>"))


if __name__ == "__main__":
    raise SystemExit(main())
