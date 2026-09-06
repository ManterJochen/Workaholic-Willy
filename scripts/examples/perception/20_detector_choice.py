"""20: an open-vocabulary detector against a closed-set one, resolved before a weight loads.

    python scripts/examples/perception/20_detector_choice.py
    python scripts/examples/perception/20_detector_choice.py --kind closed_set
    python scripts/examples/perception/20_detector_choice.py --profile sim --run

The decision is `models.pipeline.kind` in `config/models/object.yaml`. `zero_shot` puts
GroundingDINO in the detector slot: the prompt is text that reaches a text encoder, the phrase is
grounded, and a noun phrase nobody trained on still gets an answer. `closed_set` puts RT-DETR
there: it has no text encoder, the prompt never reaches the model, and it answers with its own
trained class list. The prompt is applied afterwards as a case-insensitive class-name filter.

That single fact is what each side costs. The open-vocabulary detector takes any phrase, including
the ones it should refuse, which is what 21 is about. The closed-set detector is one forward pass
with no text tower and is blind outside its vocabulary: a prompt longer than two words matches no
class name, and rather than hand back an empty list that reads exactly like an empty bin, it raises
and prints the class names it does know.

Its refusal does not survive the seam, and that is worth knowing before you configure it.
`TwoStageBackend.perceive` swallows a detector failure and returns no objects, deliberately, so
that a model error reaches the pick loop as `no_valid_grasp` rather than as a traceback in the
middle of a pick. Out at the caller, "this detector cannot answer your prompt" and "the bin is
empty" therefore arrive as the same empty tuple, and only the log tells them apart. The run below
asks the detector on its own whenever the stack comes back empty, and prints the reason it finds.

Two keys decide this and they decide for different callers, so one tree can answer twice.
`models.pipeline` is read by `build_perception`, which is how a cell is assembled.
`models.detector` and `models.segmenter_backend` are read by `build_object_detector` and
`build_segmenter`, which is what `python -m src.robot.perception` calls. The run below prints both
answers off the same tree, because a bench check that grounds with a different detector from the
cell is not evidence about the cell.

Nothing here loads a weight unless you pass `--run`. `PerceptionSpec.resolve()` reports what
`build()` would construct and quotes, verbatim, every refusal it would raise, because it reads the
builder's own constants rather than restating them. That is the half of this decision worth having
before fetching gigabytes.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import EXIT_FAILED, Example, not_ready  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover (typing only)
    import numpy as np

    from src.config.schema.models.models_schema import PipelineConfig
    from src.models.perception_spec import PerceptionSpec

EPILOG = """
exit codes
  0  both sides of the fork resolved, and with --run the selected stack built and answered
  1  the selected stack cannot be built from this config; the refusal is printed above
  2  this tree has no models block, or the weights that stack needs are not on this box

Resolving costs nothing and loads nothing. --run loads real weights onto whatever device
`WILLY_DEVICE` and torch agree on. A detector that refuses the prompt is reported as a finding
and still exits 0: refusing a question it cannot answer is the correct answer.
"""

#: The two sides of the fork, spelled as the schema spells them.
_FORK: tuple[Literal["zero_shot", "closed_set"], ...] = ("zero_shot", "closed_set")

#: What `--kind` accepts. `config` is this file's own word for "do not swap anything".
_KINDS = ("config", *_FORK)


def _with_kind(
    pipeline: "PipelineConfig | None", kind: Literal["zero_shot", "closed_set"],
) -> "PipelineConfig":
    """This tree's own pipeline block with `kind` swapped and every other setting carried over.

    Rebuilt field by field rather than copied, so the schema's cross-checks run again on the
    result. That matters here: a tree with the prompt router on cannot cross this fork at all, and
    a copy that skipped validation would hand back an object the loader would have rejected.
    """
    from src.config.schema.models.models_schema import PipelineConfig, PromptRouterConfig

    if pipeline is None:
        return PipelineConfig(kind=kind)
    return PipelineConfig(
        kind=kind,
        zero_shot=pipeline.zero_shot,
        closed_set=pipeline.closed_set,
        # Restated rather than passed through, so `enabled` is a value this tree chose rather than
        # the schema default, which is true while the shipped YAML says false.
        router=PromptRouterConfig(
            enabled=pipeline.router.enabled,
            route_non_english_to_vlm=pipeline.router.route_non_english_to_vlm,
        ),
    )


def _schema_sentence(error: ValueError) -> str:
    """A pydantic validation message as the sentence its author wrote.

    A `ValidationError` renders as a header line, then the message wrapped in `Value error, ...
    [type=value_error, input_value=...]`, then a documentation URL. The sentence in the middle is
    the one the schema author wrote for an operator, and printing the wrapper around it buries it.
    Anything that is not shaped like that comes back unchanged rather than truncated.
    """
    for line in str(error).splitlines():
        stripped = line.strip()
        if stripped.startswith("Value error, "):
            return stripped[len("Value error, "):].split(" [type=")[0]
    return str(error)


def _side(spec: "PerceptionSpec", kind: Literal["zero_shot", "closed_set"]) -> tuple[str, bool]:
    """`(rendered answer, buildable)` for one side of the fork, without loading anything.

    A schema refusal is an answer too, and it arrives as a `ValueError` from the model validator
    rather than inside the resolution, so both shapes are caught and both are returned as text.
    """
    try:
        swapped = replace(spec, pipeline=_with_kind(spec.pipeline, kind))
    except ValueError as error:
        return f"  REFUSED BY THE SCHEMA: {_schema_sentence(error)}", False
    resolution = swapped.resolve()
    return resolution.render(), resolution.buildable


def _refusal_from_detector(backend: Any, image: "np.ndarray", prompt: str) -> str:
    """Why the stack came back empty, asked of the detector alone. Empty string means "no reason".

    Reaching past `perceive` on purpose, and only after it has already answered nothing.
    `TwoStageBackend` exposes `detector` and swallows its failures by design, so this is the one
    question the seam cannot be asked: it has already turned the refusal into an empty tuple.
    A backend with no `detector` attribute, the guarded and routed ones, answers nothing here.
    """
    detector = getattr(backend, "detector", None)
    if detector is None:
        return ""
    try:
        detector.detect_all(image, prompt)
    except Exception as error:                                  # noqa: BLE001  (report, not raise)
        return f"{type(error).__name__}: {error}"
    return ""


def _synthetic_scene() -> "np.ndarray":
    """A grey field with one red square, 80 px on a side, centred at (320, 240).

    A drawn image, not a camera and not a claim about a scene: it exists so both detectors can be
    asked the same question on a machine with no rig attached. 22 is where a camera appears. The
    geometry is written here because the boxes printed below are read against it.
    """
    import numpy as np

    image = np.full((480, 640, 3), 200, dtype=np.uint8)
    image[200:280, 280:360] = (40, 40, 220)          # BGR, so this is red
    return image


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "", epilog=EPILOG,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kind", choices=_KINDS, default="config",
                        help="which side --run builds. Default: whatever the config selects")
    parser.add_argument("--prompt", default="a red cube",
                        help="what to ask for. A class name for closed_set, any phrase for "
                             "zero_shot")
    parser.add_argument("--run", action="store_true",
                        help="load the weights and ground one drawn frame. Without it nothing is "
                             "downloaded and nothing reaches a GPU")
    parser.add_argument("--profile", default=None,
                        help="WILLY_PROFILE to load. `sim` is the one that names Hub ids instead "
                             "of the local weight directories a fresh checkout does not have")
    args = parser.parse_args(argv)

    # After the parse, so --help answers in a checkout whose dependencies are not installed.
    from src.config import load_config
    from src.models.perception_spec import PerceptionSpec

    with Example("20 detector_choice", f"zero_shot against closed_set, asking for '{args.prompt}'",
                 hardware=False, profile=args.profile) as run:
        with run.step("load the config tree") as report:
            models = load_config().models
            spec = PerceptionSpec.from_config(models)
            report(f"detector {models.detector}, segmenter_backend {models.segmenter_backend}, "
                   f"pipeline {'set' if models.pipeline is not None else 'absent'}")

        with run.step("what this tree would build") as report:
            here = spec.resolve()
            report(f"{here.kind} / {here.detector} + {here.segmenter}, "
                   f"decided by {here.source}")

        run.note("")
        run.note("The same tree, both answers. Neither of these loads a weight: every line below")
        run.note("is read off the config and off the builder's own refusal constants.")
        run.note("")
        with run.step("resolve both sides of the fork") as report:
            sides = {kind: _side(spec, kind) for kind in _FORK}
            report(", ".join(f"{kind} {'buildable' if ok else 'REFUSED'}"
                             for kind, (_, ok) in sides.items()))
        for kind, (block, _) in sides.items():
            run.note(f"models.pipeline.kind: {kind}"
                     + ("   (the side this tree selects)" if kind == here.kind else ""))
            for line in block.splitlines():
                run.note(line)
            run.note("")

        with run.step("the other fork, which nobody writes down") as report:
            # `models.pipeline` absent is not a hypothetical: every cell that predates the block
            # runs this way, and `build_perception` falls back to the two legacy keys
            # byte-identically. Dropping the block from the spec is exactly what that tree is.
            legacy = replace(spec, pipeline=None).resolve()
            report(f"without models.pipeline: {legacy.kind} / {legacy.detector} + "
                   f"{legacy.segmenter}, decided by {legacy.source}")
        run.note("`python -m src.robot.perception` reads only those two legacy keys, so on a tree")
        run.note("where the pipeline block and the legacy pair disagree, the bench exerciser and")
        run.note("the cell ground with different detectors and the bench run stops being evidence.")
        if (legacy.kind, legacy.detector) != (here.kind, here.detector):
            run.finding("the two halves disagree",
                        f"a cell builds {here.detector}, `python -m src.robot.perception` builds "
                        f"{legacy.detector}")

        run.note("")
        with run.step("what the schema will not let you write") as report:
            # Triggered deliberately. The combination is legal to type and impossible to honour: a
            # closed-set detector answers only from its class list, so there is no second route for
            # a router to send a hard prompt to. The message names the two ways out.
            from src.config.schema.models.models_schema import (
                PipelineConfig,
                PromptRouterConfig,
            )
            try:
                PipelineConfig(kind="closed_set", router=PromptRouterConfig(enabled=True))
                refusal = "nothing refused, which this example did not expect"
            except ValueError as error:
                refusal = _schema_sentence(error)
            report("the loader answers with the sentence below")
        for line in textwrap.wrap(refusal, width=84):
            run.note(f"  {line}")

        if not args.run:
            run.note("")
            run.note("Nothing was loaded. `--run` builds the stack --kind selects and grounds one")
            run.note("drawn frame with it, which is where the difference stops being a config key.")
            run.note("On the shipped tree that needs the weight directories `local: True` names;")
            run.note("`--profile sim` names Hub ids instead, and fetches on first use.")
            run.note("")
            run.note("Next: 21_when_the_detector_is_wrong.py, for the prompts a phrase grounder")
            run.note("answers confidently and wrongly.")
            return run.exit_code

        if args.kind == "config":
            chosen = spec
        else:
            side: Literal["zero_shot", "closed_set"] = (
                "closed_set" if args.kind == "closed_set" else "zero_shot")
            chosen = replace(spec, pipeline=_with_kind(spec.pipeline, side))
        resolution = chosen.resolve()
        if not resolution.buildable:
            # Exit 1, not 2. The weights are not the problem: this configuration cannot be built at
            # all, and fetching anything would not change that.
            run.finding("build the selected stack", resolution.refusal)
            return EXIT_FAILED

        run.note("")
        run.note(f"Loading weights for {resolution.detector} plus {resolution.segmenter}.")
        run.note("A block whose `local` is false pulls its checkpoint from the Hugging Face cache,")
        run.note("and downloads it once if the cache does not have it yet.")
        backend: Any = None
        with run.step(f"build {resolution.kind} / {resolution.detector}") as report:
            try:
                backend = chosen.build()
            except Exception as error:                          # noqa: BLE001  (report, not raise)
                return not_ready(
                    f"the stack did not build ({type(error).__name__}: {error})",
                    "fetch the weights (`python scripts/model_weights/fetch.py dino-tiny sam2`) "
                    "and run with --profile sim, which names Hub ids instead of local directories")
            report(type(backend).__name__)

        image = _synthetic_scene()
        with run.step(f"ground '{args.prompt}' in one drawn frame") as report:
            objects = backend.perceive(image, args.prompt)
            report(f"{len(objects)} object(s)")
        for obj in objects:
            det, seg = obj.detection, obj.segmentation
            run.note(f"  {det.label!r} score {det.score:.3f} "
                     f"box {[round(v, 1) for v in det.box]} "
                     f"mask {seg.mask_area_px} px "
                     f"centroid {tuple(round(float(v), 1) for v in seg.centroid_xy)}")
        if not objects:
            # An empty result is two different answers wearing one face, and this is where the
            # closed-set side costs most. `TwoStageBackend.perceive` swallows a detector failure
            # and returns no objects on purpose, so that a model error reaches the pick loop as
            # `no_valid_grasp` rather than as a traceback mid-pick. The consequence here is that
            # "this detector refused your prompt" and "the bin is empty" arrive identically.
            # Asking the detector on its own is the only way to tell them apart from out here.
            refused = _refusal_from_detector(backend, image, args.prompt)
            if refused:
                run.finding("the detector refused, and the seam swallowed it",
                            "an empty result, not an error; the sentence follows")
                for line in textwrap.wrap(refused, width=84):
                    run.note(f"  {line}")
            else:
                # Each side keeps its confidence floor in its own block, so the key named here is
                # the one that would move this answer rather than the one that reads first.
                floor = ("models.rtdetr" if resolution.detector == "rtdetr"
                         else "models.objectdetector")
                run.note("  Nothing grounded, and the detector raised nothing either, so this is")
                run.note("  an honest empty scene. On a drawn frame that is a checkpoint question")
                run.note("  before it is a code one: try another checkpoint before lowering")
                run.note(f"  `{floor}.threshold`.")
        run.note("")
        run.note("The square is 6400 px at (280, 200) to (360, 280) with its centre at (320, 240),")
        run.note("so a box near those corners and a mask near that area mean the front end works.")
        if resolution.detector == "groundingdino":
            run.note("A label that is a fragment of the prompt is normal: GroundingDINO grounds")
            run.note("sub-phrases, which is why the simulator's source maps a label onto the")
            run.note("nearest canonical object name before anything matches on it.")
        return run.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
