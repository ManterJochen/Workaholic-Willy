"""Which grasp generator a cell runs: the reader of `robot.grasping.calculator`.

Without this factory the key is inert: a cell can set `grasping.calculator: deep` and get the
analytic stack, silently. A check that enumerates boolean `enabled` fields does not see a string
selector, so a selector is as capable of being inert as a switch.

Fails closed. `deep` without a readable artifact raises here rather than falling back to
`geometric`. A cell that asked for the learned generator and quietly got the analytic one would
report the analytic one's numbers under the learned one's name, which is worse than not starting.

The same holds for the cell's hand. `deep` on a cell whose `robot.gripper.model` is unset, is not in
the cell's gripper registry, or is not among the hands the artifact was trained across refuses at
build, as a `ValueError`, after the artifact checks.

And for the artifact's proof (the owner's decision of 2026-10-09): no finished models ship, every
customer trains their own, so a cell never grasps with a trained generator whose proof has not
passed. `deep` for a cell refuses an artifact without a passed promotion at phase `active` beside it
(`deep/promotion.py`), as a `ValueError` with one sentence that says why and what to do. Nothing
writes a promotion yet (`deep judge` and `deep promote` come next), so today that is every artifact.
`purpose="evaluate"` skips this check and no other: the ladder, the simulation runners and the offline
sweeps build a deep calculator to measure an artifact, which is how its proof is made.

Importing this costs no torch. `deep.calculator` keeps torch inside `_ensure_model`, so the factory
can name both implementations without putting a 2 GB import on the path of a cell that runs neither.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

from src.utility.log_cfg import create_logger

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.config.schema.robot.robot_schema import RobotConfig

__all__ = ["build_calculator", "preflight_calculator"]

# From `grasping.constants`, not from `grasping.deep`. This module is on the boot path of every
# cell, including one that runs the analytic generator and will never touch the learned stack, so a
# module-level import into `deep/` makes the stable half of the package depend on the half that
# moves. The hazard is direction, not torch cost: `deep/` moves modules, and this import would
# follow them.
from src.robot.grasping.constants import DEEP_GENERATOR_LOG_FILE

logger = create_logger("CalculatorFactory", DEEP_GENERATOR_LOG_FILE)

#: Who a calculator is built for (2026-10-09). `cell` is a cell that will grasp with it, and the default,
#: so a construction site that says nothing gets the gate. `evaluate` is a run that measures it: the
#: ladder, the simulation runners and the offline sweeps.
Purpose = Literal["cell", "evaluate"]
_PURPOSES: Final[tuple[str, ...]] = ("cell", "evaluate")

#: How every promotion refusal ends: what the gate is for and the two ways on.
_UNTIL_ITS_PROOF_PASSED: Final[str] = (
    "so this cell will not grasp with it: a trained generator drives a cell only once its proof has passed "
    "(deep judge, coming); until then set robot.grasping.calculator: geometric, or evaluate the artifact with "
    "the ladder.")


def _checked_purpose(purpose: str) -> str:
    """The purpose, or a refusal. An unknown one reads as neither, so the gate cannot be skipped by a typo."""
    if purpose not in _PURPOSES:
        raise ValueError(
            f"unknown calculator purpose {purpose!r}: expected 'cell', for a cell that will grasp with it, or "
            f"'evaluate', for a run that measures it (the ladder, the simulation runners).")
    return purpose


def _refuse_unless_promoted(artifact: str, purpose: str) -> None:
    """Raise unless ``artifact`` may drive a cell, where a cell is what it is built for.

    The owner's decision of 2026-10-09: no finished models ship and every customer trains their own, so a
    cell never grasps with a trained generator that has not passed its proof. The proof leaves a promotion
    record beside the artifact (`deep/promotion.py`), and nothing writes one yet, so today every artifact
    refuses a cell, in one sentence that says why and what to do.

    ``purpose="evaluate"`` skips this check and nothing else. The ladder, the simulation runners and the
    offline sweeps build a deep calculator to measure an artifact, which is how its proof will be made, so
    a gate on them would make the proof impossible. The kind, version and hand refusals hold for both.

    Fails closed. A record that cannot be read refuses as no record does, and a check that cannot run
    refuses too: it never waves an artifact through.
    """
    name = Path(artifact).name
    if purpose == "evaluate":
        logger.info("%s is taken for evaluation, so whether it may drive a cell is not asked", name)
        return
    from src.robot.grasping.deep.promotion import why_not_deployable  # noqa: PLC0415

    try:
        why = why_not_deployable(artifact)
    except Exception as exc:  # noqa: BLE001 (a check that cannot run refuses; it never waves through)
        why = f"{name}'s promotion could not be checked ({type(exc).__name__}: {exc})"
    if why:
        logger.warning("refused the DEEP grasp generator for a cell: %s", why)
        raise ValueError(f"{why}, {_UNTIL_ITS_PROOF_PASSED}")


def _refuse_unless_artifact(path: str) -> "tuple[str, tuple[str, ...]]":
    """Raise unless ``path`` is a runtime generator artifact, naming what it actually is.

    Returns the hand the artifact was stamped for and the hands it was trained across.

    One family only. A file from the retired binned generator is refused here by name rather than
    loaded: its numbers were produced by a different model and are not comparable with anything
    this build reports.

    The retired names below are literals on purpose, not imports from the module that wrote them.
    An import here would sit in a function body, where nothing runs it until the function does, and
    `ignore_missing_imports = true` in pyproject makes a vanished module a silent pass for mypy. A
    literal cannot rot that way.
    """
    import torch

    from src.robot.grasping.deep.set_artifact import (
        SET_ARTIFACT_KIND,
        SET_ARTIFACT_VERSION,
    )

    #: What the retired binned family stamped into its files. Kept only to give its owner a straight
    #: answer instead of a puzzled one about an unexpected `kind`.
    retired_artifact = "grasp_generator"
    retired_checkpoint = "grasp_generator_checkpoint"

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 (re-raised as a typed config error)
        raise ValueError(
            f"grasping.deep_generator.artifact_path {path!r} is not a readable torch file: {exc}"
        ) from exc
    kind = payload.get("kind") if isinstance(payload, dict) else None
    if kind in (retired_artifact, retired_checkpoint):
        raise ValueError(
            f"{path!r} was written by the BINNED grasp generator, an architecture this project "
            "retired on 2026-09-04. Nothing loads it any more, deliberately, because its numbers "
            "are not comparable with the current model. Train a replacement with "
            "`python -m src.robot.grasping.deep train-set`."
        )
    if kind != SET_ARTIFACT_KIND:
        raise ValueError(
            f"{path!r} is not a grasp-generator artifact (kind={kind!r}, expected "
            f"{SET_ARTIFACT_KIND!r}). A fold checkpoint from a run in progress carries no kind at "
            "all; point at the artifact a finished run writes instead."
        )
    version = int(payload.get("artifact_version", -1))
    if version != SET_ARTIFACT_VERSION:
        raise ValueError(
            f"{path!r} is a {kind} at artifact_version {version}, this build reads "
            f"{SET_ARTIFACT_VERSION}."
        )
    stamp = str(payload.get("gripper", ""))
    # An artifact written before the list existed reads back as its stamp alone, as
    # `load_set_generator` reads it.
    trained = tuple(str(name) for name in payload.get("trained_grippers") or (stamp,))
    return stamp, trained


#: What the deep branch reads off `kwargs`. Everything else is dropped, and the two sets below
#: decide whether that drop is a refusal or a log line.
_CARRIED: Final[frozenset[str]] = frozenset({
    "camera_matrix", "max_candidates", "min_grip_width_mm", "max_grip_width_mm",
})

#: Analytic-only construction knobs the learned generator has no equivalent for, and does not need.
#:
#: Listed so the drop is stated rather than silent. `scene_obstacles` and `side_approaches` (the cell fixes,
#: 2026-10-01) are the analytic stage's view of the parts beside the one it grasps and its tilted approaches. Every cell
#: passes both from its tree, so without them here every deep cell would refuse to build; the deep decoder plans its own
#: approach, and the guard judges its grasps either way.
#:
#: `support_footprint_rim_mm` (2026-10-09) is the rim the support-footprint stage's input loses, which this factory
#: hands every analytic calculator: a deep cell runs no such stage, so a rim a site passes has nothing to cut there,
#: and says so.
_IGNORED_BY_DEEP: Final[frozenset[str]] = frozenset({
    "support_footprint_rim_mm", "scene_obstacles", "side_approaches",
})


def _is_active(value: Any) -> bool:
    """Would dropping this kwarg change anything?

    The answer is keyed on the value, not on the key. A construction site passes its switches
    unconditionally, at whatever position they sit in: `ik_service=None` is the off position of an optional
    feature, and refusing on the key alone would refuse a deep run that loses nothing.

    `0` and `0.0` count as off. Every kwarg reaching this path is a switch, a live handle or a
    millimetre threshold, and a zero threshold means "do not apply it" in all of them. A future
    kwarg where zero is meaningful has to be named here.
    """
    if value is None or value is False:
        return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    if isinstance(value, (str, bytes, list, tuple, dict, set, frozenset)):
        return len(value) > 0
    return True


def _the_cells_hand(robot_cfg: "RobotConfig", artifact: str, trained: "tuple[str, ...]",
                    data_dir: "str | Path | None") -> str:
    """The registry model of the hand this cell names, checked against the hands ``artifact`` saw.

    The generator conditions on a hand, and a conditioning vector the model never saw produces
    grasps that read as a bad model rather than as a wrong hand. So a deep cell names its hand, the
    registry of its own tree holds that hand, and the artifact was trained across it, or the build
    refuses here, while a caller can still refuse. The calculator's lazy loader checks the same
    thing, but a cell's compute path swallows that refusal. Both sides are compared by canonical
    name: an artifact written before the registry is stamped ``2f85``, the alias of
    ``robotiq_2f85``. A registry refusal is re-raised as ``ValueError``, the type this factory's
    callers catch (``ConfigError`` is a ``RuntimeError``).
    """
    from src.config.grippers import load_gripper  # noqa: PLC0415
    from src.config.loader import ConfigError  # noqa: PLC0415
    from src.robot.grasping.deep.hands import canonical_hand  # noqa: PLC0415

    model = getattr(getattr(robot_cfg, "gripper", None), "model", None)
    if not model:
        raise ValueError(
            "grasping.calculator is 'deep' and robot.gripper.model is unset. The learned generator conditions on the "
            "hand, so a deep cell names it: set robot.gripper.model in the cell profile to a registry name, for "
            "example robotiq_2f85.")
    try:
        spec = load_gripper(str(model), data_dir=data_dir, aliases=False)
    except ConfigError as exc:
        raise ValueError(f"robot.gripper.model {model!r} is not a hand this cell's registry holds: {exc}") from exc
    seen = sorted({canonical_hand(name, data_dir=data_dir) for name in trained})
    if spec.model not in seen:
        raise ValueError(
            f"robot.gripper.model is {spec.model!r}, and the generator artifact {Path(artifact).name} was trained "
            f"across {', '.join(seen)}. Refusing to build a deep cell for a hand the model never saw: its grasps "
            f"would read as a bad model rather than as a wrong hand. Train an artifact whose corpus carries "
            f"{spec.model!r}, or name the hand this cell actually has.")
    return spec.model


def preflight_calculator(robot_cfg: "RobotConfig", *, data_dir: "str | Path | None" = None,
                         purpose: Purpose = "cell") -> str:
    """Check the selector without building anything, and return what it chose.

    For callers that build inside a loop and a `try`. `build_calculator` fails closed, so a sweep
    that constructs one calculator per scene inside `except Exception: continue` turns a single
    configuration error into N warnings and a report full of tidy zeros, which reads as a dead
    feature pipeline rather than as a wrong artifact path.

    A loop calls this once, before the first scene, and the refusal arrives as itself.

    ``data_dir`` is the config tree the cell came from, whose gripper registry answers for a deep
    cell's hand; ``None`` is the repository's.

    ``purpose`` is the one `build_calculator` will be given (2026-10-09): ``"cell"`` refuses a deep
    artifact whose proof has not passed, ``"evaluate"`` does not ask (:func:`_refuse_unless_promoted`).
    """
    _checked_purpose(purpose)
    choice = str(getattr(robot_cfg.grasping, "calculator", "geometric"))
    if choice == "geometric":
        return choice
    if choice != "deep":
        raise ValueError(f"unknown grasping.calculator {choice!r}: expected 'geometric' or 'deep'")
    block = getattr(robot_cfg.grasping, "deep_generator", None)
    artifact = str(getattr(block, "artifact_path", "") or "")
    if not artifact:
        raise FileNotFoundError(
            "grasping.calculator is 'deep' but robot.grasping.deep_generator.artifact_path is unset.")
    if not Path(artifact).is_file():
        raise FileNotFoundError(
            f"grasping.calculator is 'deep' but no generator artifact is readable at {artifact!r}.")
    _stamp, trained = _refuse_unless_artifact(artifact)
    _refuse_unless_promoted(artifact, purpose)
    _the_cells_hand(robot_cfg, artifact, trained, data_dir)
    return choice


def _stage_kwargs(supplied: "dict[str, Any]") -> "dict[str, Any]":
    """The rim a cell's calculator cuts off the support-footprint stage's input (``footprint_rim.FOOTPRINT_RIM_MM``,
    2026-10-09), where the construction site passed none: every site that goes through this factory, the rehearsal,
    the real cell and each camera's calculator, cuts the cell's rim. An argument the construction site passed wins."""
    from src.robot.grasping.generation.footprint_rim import FOOTPRINT_RIM_MM  # noqa: PLC0415

    return {} if "support_footprint_rim_mm" in supplied else {"support_footprint_rim_mm": FOOTPRINT_RIM_MM}


def opens_as_needed(robot_cfg: "RobotConfig") -> bool:
    """Whether the cell's hand opens only as wide as a grasp needs before it closes: a Robotiq hand, which is told a
    width (lever G, the owner's "nur wenn Robotiq genutzt wird", 2026-10-13), as its driver's spec says. A toggle, an
    I/O jaw and every other hand open fully."""
    from src.robot.grippers import gripper_driver  # noqa: PLC0415

    gripper = getattr(robot_cfg, "gripper", None)
    driver = gripper_driver(str(getattr(gripper, "vendor", "") or "").strip().lower())
    return bool(driver is not None and driver.opens_as_needed)


def inner_grasps(robot_cfg: "RobotConfig") -> bool:
    """Whether the calculator searches the part's bores for grasps from inside: where the cell's hand is a Robotiq, which
    is told a width to come down at, opens onto the bore's wall at its force and says it stopped there
    (``core.gripper.GripsFromInside``). Every such cell's since 2026-10-17, as the owner's cell ran it
    (``robot.grasping.inner_grasps`` until then). A toggle would need two changes of its output more a pick, and no other
    hand says how it holds.

    Args:
        robot_cfg: The cell's robot tree.

    Returns:
        bool: ``True`` where the cell's hand can grip so.
    """
    return opens_as_needed(robot_cfg)


def build_calculator(robot_cfg: "RobotConfig", *, data_dir: "str | Path | None" = None,
                     purpose: Purpose = "cell", **kwargs: Any) -> Any:
    """The generator this cell's config asks for, built from ``kwargs`` common to both.

    ``kwargs`` are whatever the construction site already passes to `GraspCalculator`: camera
    matrix, grip limits, the scene it reads. The deep implementation ignores what it does not
    understand, which is the protocol's rule and the reason a new analytic config block cannot
    break it.

    ``data_dir`` is the config tree the cell came from: a deep cell's hand is checked against that
    tree's gripper registry at build, and the calculator resolves its conditioning vector from it.
    ``None`` is the repository's.

    ``purpose`` is who the calculator is for (2026-10-09). ``"cell"``, the default and what every
    cell builder passes by passing nothing, refuses a deep artifact whose proof has not passed.
    ``"evaluate"`` is for a run that measures one, the ladder and the simulation runners, and skips
    that check alone (:func:`_refuse_unless_promoted`). The analytic generator needs no proof.
    """
    _checked_purpose(purpose)
    choice = str(getattr(robot_cfg.grasping, "calculator", "geometric"))
    if choice == "geometric":
        from src.robot.grasping.generation.calculator import GraspCalculator  # noqa: PLC0415

        calculator = GraspCalculator(**{**_stage_kwargs(kwargs), **kwargs})
        # A Robotiq hand is told a width: it opens only as wide as a grasp needs (lever G, the owner, 2026-10-13: "aber
        # nur wenn Robotiq genutzt wird"). Every other hand opens fully, as before.
        calculator.opens_as_needed = opens_as_needed(robot_cfg)
        # The part's bores are searched for grasps from inside where the cell's hand grips so.
        calculator.inner_grasps = inner_grasps(robot_cfg)
        return calculator
    if choice != "deep":
        raise ValueError(f"unknown grasping.calculator {choice!r}: expected 'geometric' or 'deep'")

    from src.robot.grasping.collision.support_resolver import bench_height_mm  # noqa: PLC0415
    from src.robot.grasping.deep.calculator import (  # noqa: PLC0415
        DeepCalculatorConfig,
        DeepGraspCalculator,
    )

    block = getattr(robot_cfg.grasping, "deep_generator", None)
    artifact = str(getattr(block, "artifact_path", "") or "")
    if not artifact or not Path(artifact).is_file():
        # Fail closed rather than fall back to `geometric`: the cell would run the analytic stack
        # and every record, every KPI and every ladder row would be filed under the learned
        # generator's name.
        where = ("robot.grasping.deep_generator.artifact_path is unset" if not artifact
                 else f"no generator artifact is readable at {artifact!r}")
        raise FileNotFoundError(
            f"grasping.calculator is 'deep' but {where}. "
            f"Train one with `python -m src.robot.grasping.deep train-set --clouds DIR --out DIR`, "
            f"or set calculator: geometric.")
    # And it has to be an artifact, checked here rather than on first use. The kind check also lives
    # in the calculator's lazy loader, but the protocol forbids that method from raising, so a wrong
    # file surfaces there as NO_CANDIDATES_GENERATED on every unit instead of as an error, and a run
    # graded that way reads as a bad generator rather than as a wrong path. A training checkpoint is
    # the wrong file that is easiest to reach for.
    _stamp, trained = _refuse_unless_artifact(artifact)
    # And for a cell, its proof: the promotion beside it, read before the hand because it is a fact about
    # the file, and a file no proof has passed is no cell's whatever hand it was trained for.
    _refuse_unless_promoted(artifact, purpose)
    # The cell's hand, checked here and after the artifact checks, so a tree with no weights still
    # refuses as a missing file. The calculator's loader checks it again; this is the check that
    # fires while a caller can refuse.
    hand = _the_cells_hand(robot_cfg, artifact, trained, data_dir)
    # A dropped kwarg is either a refusal or a stated loss, never silence. The geometric branch is
    # `GraspCalculator(**kwargs)` verbatim; this one maps an explicit list, so anything a
    # construction site passes and this list omits vanishes without a word.
    #
    # One of the droppable ones changes outcomes rather than style, so it is refused by name:
    #   ik_service   the reachability filter that discards unreachable candidates. Without it nothing filters and
    #                `rejected_ik` stays 0, which reads as "nothing was unreachable".
    # The refusal reads the value, never the key alone: see `_is_active`.
    unknown = sorted(k for k, v in kwargs.items()
                     if k not in _CARRIED and k not in _IGNORED_BY_DEEP and _is_active(v))
    if unknown:
        raise ValueError(
            f"grasping.calculator is 'deep' but this construction site passes {unknown} with an "
            f"ACTIVE value, and the learned generator cannot honour them. Those change what a pick "
            f"DOES, not how it looks, so running without them would report a different experiment "
            f"under the same name. Switch them off, use calculator: geometric here, or teach the "
            f"deep path to carry them.")
    ignored = sorted(k for k, v in kwargs.items() if k in _IGNORED_BY_DEEP and _is_active(v))
    if ignored:
        logger.warning(
            "the deep generator ignores %s: it decodes its own approach, closing axis and seeds, so "
            "these analytic knobs have nothing to act on. Any banner printing them is describing the "
            "geometric path.", ", ".join(ignored))
    logger.info("built the DEEP grasp generator from %s for the hand %s, for %s", artifact, hand,
                "a cell" if purpose == "cell" else "evaluation")
    return DeepGraspCalculator(DeepCalculatorConfig(
        artifact_path=artifact,
        camera_matrix=kwargs.get("camera_matrix"),
        max_candidates=int(kwargs.get("max_candidates", 12)),
        device=getattr(block, "device", None) or None,
        minimum_score=float(getattr(block, "minimum_score", 0.5)),
        support_height_mm=bench_height_mm(robot_cfg),
        # The jaw, carried across. `DeepGraspCalculator._fits_the_jaw` reads both bounds and treats
        # 0.0 as "not told", so without them a deep cell can command a width its gripper cannot
        # open to; the analytic generator filters that at generation instead.
        min_grip_width_mm=float(kwargs.get("min_grip_width_mm", 0.0)),
        max_grip_width_mm=float(kwargs.get("max_grip_width_mm", 0.0)),
        # The cell's hand, and the tree its registry was read from, so the conditioning vector is
        # that hand's.
        gripper=hand,
        data_dir=None if data_dir is None else str(data_dir),
    ))
