"""Which grasp generator a cell runs: the reader of `robot.grasping.calculator`.

Without this factory the key is inert: a cell can set `grasping.calculator: deep` and get the
analytic stack, silently. A check that enumerates boolean `enabled` fields does not see a string
selector, so a selector is as capable of being inert as a switch.

Fails closed. `deep` without a readable artifact raises here rather than falling back to
`geometric`. A cell that asked for the learned generator and quietly got the analytic one would
report the analytic one's numbers under the learned one's name, which is worse than not starting.

Importing this costs no torch. `deep.calculator` keeps torch inside `_ensure_model`, so the factory
can name both implementations without putting a 2 GB import on the path of a cell that runs neither.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

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


def _refuse_unless_artifact(path: str) -> None:
    """Raise unless ``path`` is a runtime generator artifact, naming what it actually is.

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


#: What the deep branch reads off `kwargs`. Everything else is dropped, and the two sets below
#: decide whether that drop is a refusal or a log line.
_CARRIED: Final[frozenset[str]] = frozenset({
    "camera_matrix", "max_candidates", "min_grip_width_mm", "max_grip_width_mm",
})

#: Analytic-only construction knobs the learned generator has no equivalent for, and does not need.
#:
#: Listed so the drop is stated rather than silent. The deep decoder predicts its own approach and
#: its own closing axis, so an oblique-approach sweep or a radial-closing flag has nothing to act
#: on. `robot.ur3e.yaml` sets `isotropic_radial_closing: true` on a cell measured at 5/10 -> 10/10,
#: so a deep run there says the lever is not in play instead of printing `True` and changing
#: nothing.
_IGNORED_BY_DEEP: Final[frozenset[str]] = frozenset({
    "isotropic_radial_closing",
    "oblique_approach", "oblique_tilt_deg", "oblique_azimuths",
    "support_footprint_geometry", "support_footprint_inflate_mm",
})


def _is_active(value: Any) -> bool:
    """Would dropping this kwarg change anything?

    The answer is keyed on the value, not on the key. A construction site passes its switches
    unconditionally, at whatever position they sit in: `ik_service=None` and
    `corridor_risk_per_candidate=False` are the off positions of two optional features, and refusing
    on the key alone would refuse a deep run that loses nothing.

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


def preflight_calculator(robot_cfg: "RobotConfig") -> str:
    """Check the selector without building anything, and return what it chose.

    For callers that build inside a loop and a `try`. `build_calculator` fails closed, so a sweep
    that constructs one calculator per scene inside `except Exception: continue` turns a single
    configuration error into N warnings and a report full of tidy zeros, which reads as a dead
    feature pipeline rather than as a wrong artifact path.

    A loop calls this once, before the first scene, and the refusal arrives as itself.
    """
    choice = str(getattr(robot_cfg.grasping, "calculator", "geometric"))
    if choice == "geometric":
        return choice
    if choice != "deep":
        raise ValueError(f"unknown grasping.calculator {choice!r}: expected 'geometric' or 'deep'")
    block = getattr(robot_cfg.grasping, "deep_generator", None)
    artifact = str(getattr(block, "artifact_path", "") or "")
    if not artifact or not Path(artifact).is_file():
        raise FileNotFoundError(
            f"grasping.calculator is 'deep' but no generator artifact is readable at {artifact!r}.")
    _refuse_unless_artifact(artifact)
    return choice


#: The depth levers on `grasping.geometry`, with the value that means "do nothing". Only a value
#: that differs from these is forwarded, for two reasons. A cell that configures none of them builds
#: the calculator with exactly the arguments it built before, so the default path is unchanged. And
#: the deep branch below refuses any active kwarg it cannot honour: forwarding the do-nothing value
#: unconditionally would refuse every deep cell over settings that ask for nothing.
_DEPTH_LEVERS: Final[dict[str, Any]] = {
    "grasp_depth_reference": "centre",
    "grasp_top_penetration_mm": 0.0,
    "grasp_top_quantile": 10.0,
    "depth_band_mm": None,
    "depth_band_near_pct": 10.0,
}


def _depth_kwargs(robot_cfg: "RobotConfig", supplied: "dict[str, Any]") -> "dict[str, Any]":
    """What `grasping.geometry` asks the calculator to do with the depth under a mask.

    An argument the construction site passed already wins: an explicit value at the call site is a
    deliberate choice by a runner, and config is the default for cells that do not make one.
    """
    geometry = getattr(getattr(robot_cfg, "grasping", None), "geometry", None)
    if geometry is None:
        return {}
    out: dict[str, Any] = {}
    for name, inert in _DEPTH_LEVERS.items():
        if name in supplied:
            continue
        value = getattr(geometry, name, inert)
        if value != inert:
            out[name] = value
    return out


def build_calculator(robot_cfg: "RobotConfig", **kwargs: Any) -> Any:
    """The generator this cell's config asks for, built from ``kwargs`` common to both.

    ``kwargs`` are whatever the construction site already passes to `GraspCalculator`: camera
    matrix, grip limits, geometry stage. The deep implementation ignores what it does not
    understand, which is the protocol's rule and the reason a new analytic config block cannot
    break it.
    """
    kwargs.update(_depth_kwargs(robot_cfg, kwargs))
    choice = str(getattr(robot_cfg.grasping, "calculator", "geometric"))
    if choice == "geometric":
        from src.robot.grasping.generation.calculator import GraspCalculator  # noqa: PLC0415

        return GraspCalculator(**kwargs)
    if choice != "deep":
        raise ValueError(f"unknown grasping.calculator {choice!r}: expected 'geometric' or 'deep'")

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
        raise FileNotFoundError(
            f"grasping.calculator is 'deep' but no generator artifact is readable at {artifact!r}. "
            f"Train one with `python -m src.robot.grasping.deep train-set --clouds DIR --out DIR`, "
            f"or set calculator: geometric.")
    # And it has to be an artifact, checked here rather than on first use. The kind check also lives
    # in the calculator's lazy loader, but the protocol forbids that method from raising, so a wrong
    # file surfaces there as NO_CANDIDATES_GENERATED on every unit instead of as an error, and a run
    # graded that way reads as a bad generator rather than as a wrong path. A training checkpoint is
    # the wrong file that is easiest to reach for.
    _refuse_unless_artifact(artifact)
    # A dropped kwarg is either a refusal or a stated loss, never silence. The geometric branch is
    # `GraspCalculator(**kwargs)` verbatim; this one maps an explicit list, so anything a
    # construction site passes and this list omits vanishes without a word.
    #
    # Two of the droppable ones change outcomes rather than style, so they are refused by name:
    #   ik_service                    the reachability filter that discards unreachable candidates,
    #                                 and the source of `joint_margin_deg` for telemetry and for the
    #                                 success model. Without it nothing filters and `rejected_ik`
    #                                 stays 0, which reads as "nothing was unreachable".
    #   corridor_risk_per_candidate   the producer for the per-candidate uncertainty re-rank.
    #                                 Without it `grasping.uncertainty.rerank_enabled` reorders
    #                                 nothing and reports `missing_uncertainty`, which the schema
    #                                 already warns about at that key.
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
    logger.info("cell runs the DEEP grasp generator from %s", artifact)
    return DeepGraspCalculator(DeepCalculatorConfig(
        artifact_path=artifact,
        camera_matrix=kwargs.get("camera_matrix"),
        max_candidates=int(kwargs.get("max_candidates", 12)),
        device=getattr(block, "device", None) or None,
        minimum_score=float(getattr(block, "minimum_score", 0.5)),
        support_height_mm=float(robot_cfg.grasping.support.height_mm),
        # The jaw, carried across. `DeepGraspCalculator._fits_the_jaw` reads both bounds and treats
        # 0.0 as "not told", so without them a deep cell can command a width its gripper cannot
        # open to; the analytic generator filters that at generation instead.
        min_grip_width_mm=float(kwargs.get("min_grip_width_mm", 0.0)),
        max_grip_width_mm=float(kwargs.get("max_grip_width_mm", 0.0)),
    ))
