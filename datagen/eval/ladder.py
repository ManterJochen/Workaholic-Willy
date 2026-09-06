"""Grade ``GraspCalculator`` against the scene geometry, on data it has never seen.

Two measurements, deliberately independent, because they fail in different ways:

* Precision needs no tolerance and no labels. Every candidate the calculator returns is checked
  against the true scene by :mod:`datagen.grasps.verdict`: does the jaw open wide enough, do the
  pads land inside the friction cone, is the path clear. A candidate is wrong or it is not; there
  is nothing to tune.
* Recall needs both, so it is reported as a curve. How many graspable objects the calculator found
  depends on how close a candidate has to be before it counts as "the same grasp", and choosing one
  threshold would hide that. The sweep over (position mm x angle deg) is the result; a single
  working point is marked on it for later comparisons, not used to compute it.

Every rung in ``CONFIGURATIONS`` runs on the same scenes and adds one thing to the rung above it.
``topdown_dense`` forces the approach straight down because the shipped default takes its approach
direction from the camera optical axis, which on the default oblique pair stands 36.9 deg off
vertical and so proposes no vertical grasp at all; the rung measures what that costs.

The expensive pass writes one row per candidate to ``grasp_eval.jsonl``; every rate, curve and
breakdown is a pure function of that file. Re-asking with a different tolerance costs no GPU and no
re-run.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src.utility.log_cfg import create_logger

from datagen.constants import DATAGEN_LOG_DIR, GRASP_EVAL_LOG_FILE
from datagen.corpus.clouds import fuse_instance_clouds
from datagen.eval.floors import (
    NormalGraspCalculator,
    RandomGraspCalculator,
    TopDownGraspCalculator,
)
from datagen.grasps.labels import GraspLabel, SceneAssets, SceneGeometry, scene_assets
from datagen.grasps.shapes import Solid
from datagen.grasps.verdict import (
    STANDARD_CUP,
    JawGrasp,
    JawModel,
    SuctionGrasp,
    check_jaw_grasp,
    check_suction_grasp,
)

if TYPE_CHECKING:  # the schema import is deferred at runtime (heavy pydantic tree)
    from src.config.schema.robot.grasping_schema import GraspingSupportConfig

__all__ = ["CONFIGURATIONS", "DEEP_CONFIGURATIONS", "evaluate_dataset", "select_configurations",
           "summarise"]

#: Module scope, the house idiom. The worker processes below import this module too but never log
#: (every call site here is on the parent's side of the pool), so the shared rotating file has
#: exactly one writer, which is the condition ``create_logger``'s path-keyed handlers rely on.
logger = create_logger("datagen.eval.ladder", GRASP_EVAL_LOG_FILE, log_dir=DATAGEN_LOG_DIR)

#: Where the tolerance curve is sampled. Position in mm, angle in degrees, applied to both the approach
#: direction and the closing axis.
POSITION_TOLERANCES_MM = (2.0, 5.0, 10.0, 15.0, 20.0, 30.0)
ANGLE_TOLERANCES_DEG = (5.0, 10.0, 20.0, 30.0, 45.0)
#: The point marked on the curve for later comparisons. Chosen, not derived: 10 mm is a quarter of the
#: median object's grip span and 20 deg is inside the friction cone the verdict already uses.
WORKING_POINT = (10.0, 20.0)
#: Below this many visible pixels no segmentation-driven method can be expected to do anything; such
#: objects are counted separately rather than folded into a miss.
MIN_VISIBLE_PX = 100

#: How close a candidate must be to a reference label before that label's part role means anything,
#: in working points: 1.0 is 10 mm of position and 20 deg of approach and 20 deg of closing axis, the
#: worst of the three.
#:
#: `_best_match` returns the argmin however far it is, so without this gate every candidate on a
#: labelled object is stamped with the nearest label's role whatever the distance, and a per-role
#: precision then describes which label happened to be nearest rather than which part was grasped.
#:
#: The gate is applied in `summarise`, not in the row writer, and that is the whole design. A writer
#: that stripped the role would make `by_part_role_unattributed` structurally 0 on every fresh file:
#: the counter could only ever count rows an older writer produced. The row stores the role and the
#: distances; the report decides.
MATCH_ACCEPT_COST: float = 1.0

#: What a row written by this code guarantees. Bumped whenever `summarise` starts reading a field the
#: writer did not always write, because that is the moment an old row and a new row that means zero
#: become indistinguishable.
#:
#:   1  `match_part_role` only: no per-role denominator, no verdict part, and the attribution was
#:      the nearest label at any distance.
#:   2  `jaw_labels_by_role` / `suction_labels_by_role` on every row, so a role with labels and no
#:      candidate is readable as a result rather than as an absence.
#:   3  `verdict_part_role` / `verdict_part_index`: the candidate is judged against every part with
#:      its siblings as obstacles, not against the primary solid alone.
#:
#: A number rather than a feature sniff, because `row.get("jaw_labels_by_role") or {}` reads a row
#: written before the field as a row whose every role count is zero. An absent field and a field
#: that means zero are then the same thing, which is the defect the field was added to fix, inverted.
EVAL_ROW_SCHEMA: int = 3




@dataclass(frozen=True, slots=True)
class CalculatorConfig:
    """One way of running the calculator: constructor kwargs plus the per-call inputs.

    ``build`` may depend on the view: the approach vector is in CAMERA frame, so "straight down in
    BASE" is a different vector for every camera. ``neighbours`` and ``dense_sampling`` are
    ``compute()`` arguments rather than constructor ones, which is why they live here separately.
    """

    name: str
    build: Callable[[np.ndarray, np.ndarray], dict]
    note: str = ""
    #: What to construct, when it should not be `GraspCalculator`. `None` constructs
    #: `GraspCalculator`, so a rung that names no class is graded as the analytic one. Grading a
    #: second generator against the same scenes, the same reference and the same denominators is
    #: what makes a claim like "the learned generator beats `sfe_fused`" comparable at all.
    #:
    #: The ladder calls `compute()`, the runtime calls `compute_result()`. They are different
    #: methods, so satisfying `deep.protocol.GraspCandidateGenerator` does not make a generator
    #: measurable here, and passing here does not make one runtime-ready. Both are required and each
    #: has to be checked on its own.
    make: Callable[..., Any] | None = None
    neighbours: bool = False
    dense_sampling: bool | None = None
    #: Pass the target's cloud fused from every rendered view, in BASE. A single depth view sees one
    #: side of an object and an antipodal grasp needs two, so this rung changes what the calculator
    #: can see rather than how it searches. The calculator already has the seam
    #: (``geometry_points_base_mm``); nothing in the default path fills it.
    fused_geometry: bool = False
    #: Pass the other objects' fused clouds as ``scene_points_mm``: the obstacle half of the same
    #: observation. Implies ``fused_geometry``, because both come from the same cross-camera
    #: association. A calculator cannot reject a collider it never saw, and this is what lets it see
    #: the neighbours. It does not cover the container: a bin wall is not an instance any camera
    #: segments, so walls arrive only through ``container_walls``.
    fused_neighbours: bool = False
    #: Sample the scene's own bin walls as obstacles through the same production helper the pick
    #: loop uses (``container_wall_points_base_mm``), with the interior box derived from the scene's
    #: ``bin_walls`` spec. The reference verdict judges against the walls (``obstacles_for()`` is
    #: other solids plus walls) whether or not the calculator was given them, so without this rung a
    #: wall collision is a rejection the calculator had no way to predict.
    container_walls: bool = False


#: The silhouette rungs pin that stage explicitly. The calculator's own default is
#: ``support_footprint_geometry=True``, so leaving this to the default would turn every rung of the
#: ladder into the same measurement, and a ladder whose rungs mean the same thing measures nothing.
_SILHOUETTE = {"support_footprint_geometry": False}

#: Applied to every rung, after its own kwargs, so no rung can forget it.
#:
#: The dense sampler carries a wall-clock budget (150 ms in production, where a cell has a cycle
#: time). On overrun it falls back to silhouette geometry and then cools down, which changes the
#: calls after it too, so which candidates exist depends on how busy the machine is. That is correct
#: in a cell and wrong in a reference: a baseline that moves with load is not a baseline. Only the
#: dense-dependent rungs move between runs, but they are the ones the ladder compares.
#:
#: So the reference runs with the budget effectively off. The ladder then measures something
#: production does not do, which is the right trade here: it answers what this geometry can yield,
#: not what this box managed today.
#:
#: One hour rather than infinity, because ``validate_calculator_args`` refuses a non-finite budget.
#: An unbounded budget in a cell is a defect, and relaxing a production guard to suit an offline
#: tool is how guards die. If a single sampling call ever did reach an hour, that is a real failure
#: and the fallback firing is the correct answer.
_REFERENCE_ONLY_KWARGS = {"dense_runtime_budget_ms": 3_600_000.0}


def _plain(intrinsics: np.ndarray, camera_to_base: np.ndarray) -> dict:
    return {"camera_matrix": intrinsics, **_SILHOUETTE}


def _floor(intrinsics: np.ndarray, camera_to_base: np.ndarray) -> dict:
    """A floor takes the camera and the jaw limits and nothing else; it has no config surface."""
    return {"camera_matrix": intrinsics, "seed": 0, "candidates": 12,
            "min_grip_width_mm": 5.0, "max_grip_width_mm": 85.0}


def _oblique(intrinsics: np.ndarray, camera_to_base: np.ndarray) -> dict:
    return {"camera_matrix": intrinsics, "oblique_approach": True, **_SILHOUETTE}


def _sfe_voxel8(intrinsics: np.ndarray, camera_to_base: np.ndarray) -> dict:
    """The 8 mm-voxelised contact cloud instead of the full-resolution one, kept as the control.

    Full resolution is the default, so this rung measures the difference rather than remembering it.
    ``reconstruct_support_prism`` refuses below 20 points, and how many points an object has is
    decided by ``geometry_voxel_size_mm``: two numbers with nothing to do with each other. A 32 mm
    cube is 4x4 = 16 cells at 8 mm and falls off that cliff, losing every candidate while all four
    rejection counters stay at zero. SFE also re-voxelises at 3 mm itself, so the pre-downsample
    throws away resolution it was about to thin more finely.
    """
    return {"camera_matrix": intrinsics, "support_footprint_geometry": True,
            "support_footprint_full_resolution": False}


def _sfe_fallback(intrinsics: np.ndarray, camera_to_base: np.ndarray) -> dict:
    """SFE, but an abstention falls back to the silhouette instead of vetoing it.

    ``reconstruct_support_prism`` gives up below 20 surviving cloud points, and the calculator then
    replaces the silhouette candidates with SFE's empty list, so a small or grazing-angle object
    produces no candidate at all with every rejection counter still at zero.

    The fallback buys almost nothing and costs precision: where SFE abstains, the silhouette cannot
    grasp the object either, so the extra candidates it restores are overwhelmingly invalid. The
    abstention is the right answer and the default stays off.
    """
    return {"camera_matrix": intrinsics, "support_footprint_geometry": True,
            "support_footprint_fallback": True}


def _sfe(intrinsics: np.ndarray, camera_to_base: np.ndarray) -> dict:
    return {"camera_matrix": intrinsics, "support_footprint_geometry": True}


#: Where the learned generator's artifact lives. An env var and not a rung field, because it is a
#: machine-local path: baking one into a committed configuration would make the ladder unrunnable on
#: every other box and would quietly grade whatever happened to sit at that path.
ENV_DEEP_ARTIFACT = "WILLY_DEEP_ARTIFACT"

#: One built generator per artifact path, per process. `--jobs N` gives each worker its own; a
#: shared model across processes would need a lock the ladder has no reason to carry.
#:
#: The key is the artifact path alone, so pointing the env var at another artifact rebuilds. A rung
#: that varies the generator by anything other than its path has to put that something in the key
#: too, or the first rung's calculator is served to every later one under a different name.
_DEEP_CACHE: dict[str, Any] = {}


def _deep(intrinsics: np.ndarray, camera_to_base: np.ndarray) -> dict:
    """The learned generator takes the camera and the jaw, like a floor; it has no config surface.

    The jaw limits are not decoration: the deep decoder refuses a span this gripper cannot open to
    (`deep/calculator.py::_fits_the_jaw`), and a rung that withheld them would grade a generator
    that proposes widths no cell would ever execute.
    """
    return {"camera_matrix": intrinsics, "min_grip_width_mm": 5.0, "max_grip_width_mm": 85.0}


def _make_deep(**kwargs: Any) -> Any:
    """Construct the learned generator through the production factory.

    An evaluation that builds its subject differently from the cell measures a different system.
    `build_calculator` is what `cells.py` calls, so what the ladder grades here is what a cell would
    run, including the fail-closed refusal, the jaw limits and the `deep_generator` block's device
    and score threshold.

    A missing artifact refuses loudly rather than skipping the rung: a ladder with no deep row reads
    exactly like a ladder that ran.
    """
    import os  # noqa: PLC0415

    from src.config.schema.robot import RobotConfig  # noqa: PLC0415
    from src.robot.grasping.calculator_factory import build_calculator  # noqa: PLC0415

    artifact = os.environ.get(ENV_DEEP_ARTIFACT, "")
    # One calculator per process, fidelity before speed. The ladder constructs a calculator per
    # view; for the analytic one that is free, but a learned one would load its artifact and open a
    # CUDA context per view, which is not what a cell does. A cell loads once and picks many times,
    # so an evaluation that reloads per view measures a warm-up this stack never pays.
    #
    # A rung that substitutes one head of the decoder for the geometry's own answer is a real
    # instrument, but it has to be rebuilt against the decoder that ships: a switch the decoder no
    # longer reads still accepts its value and reports plain `deep` numbers under an ablation name.
    key = artifact
    cached = _DEEP_CACHE.get(key)
    if cached is not None:
        return cached
    if not artifact:
        raise FileNotFoundError(
            f"the `deep` rung needs {ENV_DEEP_ARTIFACT} set to a generator artifact "
            f"(the .pt written beside the card by `python -m src.robot.grasping.deep train`). "
            f"not the .ckpt.pt: a checkpoint is refused by kind, on purpose.")
    robot_cfg = RobotConfig.model_validate({
        "grasping": {"calculator": "deep", "deep_generator": {"artifact_path": artifact}}})
    built = build_calculator(robot_cfg, **kwargs)
    # Load it now, so a bad artifact refuses the run instead of every object. The runtime catches
    # everything inside `compute()`, correctly, because a cell must not go down over one frame: a
    # checkpoint handed to the `deep` rung would raise, be caught, and become "no candidates" for
    # every object, which reads as a model that finds nothing rather than one that never loaded.
    # The refusal above already warns against the .ckpt.pt; this is what makes it arrive in time.
    preload = getattr(built, "preload", None)
    if callable(preload):
        preload()
    _DEEP_CACHE[key] = built
    return built


def _sfe_cone(intrinsics: np.ndarray, camera_to_base: np.ndarray) -> dict:
    """SFE ranked by the contact angle alone: the first of its five terms, at full weight.

    `cone_slack = 1 - contact_angle/cone` carries 0.35 of the shipped blend and is the term that
    separates valid from invalid most sharply within a candidate list, so this rung asks what the
    other four terms contribute.

    The candidate set is unchanged: every legality filter runs first and this only orders what
    survived. A grasp that grazes the table is still refused by the filter, not by the rank.
    """
    return {"camera_matrix": intrinsics, "support_footprint_geometry": True,
            "support_footprint_score_weights": (1.0, 0.0, 0.0, 0.0, 0.0)}


def _sfe_cone70(intrinsics: np.ndarray, camera_to_base: np.ndarray) -> dict:
    """The milder half of the same question: the angle at 0.70 instead of 0.35, the rest shrunk.

    Two rungs rather than one because "the first term is the only one that works" and "the first term
    is under-weighted" are different claims, and a single all-or-nothing arm cannot separate them.
    """
    return {"camera_matrix": intrinsics, "support_footprint_geometry": True,
            "support_footprint_score_weights": (0.70, 0.10, 0.10, 0.05, 0.05)}


def _sfe_palm(intrinsics: np.ndarray, camera_to_base: np.ndarray) -> dict:
    """SFE that reasons about the palm, which its own downstream collision filter always has.

    The planner sizes its table clearance from finger points (27 mm wide); the envelope re-checks it
    with the palm (70 mm wide, 35 mm deep) included. Straight down the palm sits behind the fingers
    and is irrelevant; as the approach tilts, the binormal swings toward vertical and the palm
    becomes the lowest part of the gripper. The two height requirements differ by
    35.0 - 27.0 = 8 mm at 90 degrees.

    This rung changes both halves: the check refuses a candidate whose palm digs in, and the height
    solve accounts for the palm so SFE relocates the anchor instead of merely losing it. Only the
    second half can buy anything; the first can only take away. It buys nothing here, and precision
    does not rise to pay for the grasps it costs, so the default stays off.

    The losses are the open question: this reference calls a grasp valid by matching an analytic
    label, not by checking a table, so a grasp the palm term rejects still matches ground truth.
    Either the labels do not model the palm (they are generated for a floating-base 2F-85, which has
    no collision in Isaac) or the palm term is stricter than the hardware.
    """
    return {"camera_matrix": intrinsics, "support_footprint_geometry": True,
            "support_footprint_palm_aware": True}


def _topdown(intrinsics: np.ndarray, camera_to_base: np.ndarray) -> dict:
    # approach_vector_cam is expressed in the CAMERA frame; BASE -Z rotated back into it.
    rotation = np.asarray(camera_to_base, dtype=np.float64)[:3, :3]
    return {"camera_matrix": intrinsics, **_SILHOUETTE,
            "approach_vector_cam": rotation.T @ np.array([0.0, 0.0, -1.0])}


#: A ladder: each rung adds exactly one thing to the one above it, so every difference between two
#: adjacent rows means one thing and not three. ``neighbours`` matters more than it looks: the pick
#: loop always passes the other masks of the view, and the auto gate turns the dense geometry
#: sampler on when it sees them, so an evaluation that withholds them is not evaluating the
#: production path at all.
#:
#: The learned generator is kept out of `CONFIGURATIONS` on purpose. Every rung in the default tuple
#: runs on every ladder invocation, and this one needs a machine-local artifact, which would make
#: the ordinary ladder refuse on a box that has no trained model. It is selected by name instead
#: (`eval-grasps --rungs sfe_fused,deep`), which is also the shape the comparison wants: the full
#: ladder is hours, and the question is a pair.
#:
#: It is paired with `sfe_fused`, not with `default`. Both rungs get the target's fused three-view
#: cloud and the neighbours' masks, and the deep decoder reads both (`deep/calculator.py` consumes
#: `geometry_points_base_mm`, `scene_points_mm` and `other_object_masks`), so this is a
#: like-for-like observation rather than a rung that hands a generator inputs it ignores.
#:
#: The population is not neutral: most of the ladder's objects are assets the shipped generator
#: trained on. Read the deep row against the held-out remainder, or read it knowing that.
# Named, not lambdas. `--jobs > 1` pickles every CalculatorConfig to its workers, and a lambda
# cannot be pickled: the child dies unpacking it and the parent reports `WinError 87` from
# `spawn_main`, which names neither the rung nor the reason.


























DEEP_CONFIGURATIONS: tuple[CalculatorConfig, ...] = (
    CalculatorConfig("deep", _deep,
                     "the LEARNED generator, same fused observation as `sfe_fused`",
                     make=_make_deep, neighbours=True, fused_geometry=True),
    CalculatorConfig("deep_scene", _deep,
                     "+ the NEIGHBOURS' fused clouds as obstacles, as `sfe_fused_scene` gets them",
                     make=_make_deep, neighbours=True, fused_geometry=True, fused_neighbours=True),
    # `make=_make_deep` is not optional, and leaving it off fails late rather than loudly:
    # `_evaluate_view` reads `config.make or GraspCalculator`, so a deep rung without it is built as
    # the analytic calculator and then raises on the first kwarg that calculator has never heard of.
)


CONFIGURATIONS: tuple[CalculatorConfig, ...] = (
    # The floors come first, and they are not part of the ladder's one-change-per-rung progression.
    # They are its scale. Zero is the wrong reference for a top-1 rate: the analytic verdict admits
    # a band around vertical, so a proposal that knows nothing but "down" already collects some of
    # it. Read `floor_topdown` before reading any generator, learned or otherwise. See
    # `datagen/eval/floors.py`.
    CalculatorConfig("floor_random", _floor, "FLOOR: luck inside the generator's admissible cone",
                     make=RandomGraspCalculator),
    CalculatorConfig("floor_topdown", _floor, "FLOOR: straight down, random seed and rotation",
                     make=TopDownGraspCalculator),
    CalculatorConfig("floor_normal", _floor, "FLOOR: into the surface along its oriented normal",
                     make=NormalGraspCalculator),
    CalculatorConfig("default", _plain, "as it ships, no neighbour masks"),
    CalculatorConfig("neighbours", _plain, "+ other_object_masks, as the pick loop passes them",
                     neighbours=True),
    CalculatorConfig("dense", _plain, "+ the geometry sampler forced on, not left to the AUTO gate",
                     neighbours=True, dense_sampling=True),
    CalculatorConfig("topdown_dense", _topdown, "+ the approach forced to BASE -Z",
                     neighbours=True, dense_sampling=True),
    CalculatorConfig("oblique_dense", _oblique, "+ multi-angle approach generation instead",
                     neighbours=True, dense_sampling=True),
    CalculatorConfig("fused", _plain,
                     "+ the target's FUSED three-view cloud, so both contact faces are visible",
                     neighbours=True, fused_geometry=True),
    # The support-footprint geometry stage replacing the silhouette one. Not a knob on the ladder
    # above but a different answer to the same question, so it gets its own rungs rather than being
    # folded into one of theirs.
    CalculatorConfig("sfe", _sfe, "support-footprint enumeration instead of the silhouette",
                     neighbours=True),
    CalculatorConfig("sfe_fused", _sfe, "+ the target's FUSED three-view cloud",
                     neighbours=True, fused_geometry=True),
    CalculatorConfig("sfe_fused_scene", _sfe, "+ the NEIGHBOURS' fused clouds as obstacles",
                     neighbours=True, fused_geometry=True, fused_neighbours=True),
    # The ranking pair. Same candidates as `sfe_fused`, ordered by a different blend of the same
    # measured margins, so any difference is ranking and nothing else.
    CalculatorConfig("sfe_fused_cone", _sfe_cone,
                     "sfe_fused ranked by the CONTACT ANGLE alone (its first term, full weight)",
                     neighbours=True, fused_geometry=True),
    CalculatorConfig("sfe_fused_cone70", _sfe_cone70,
                     "sfe_fused with the contact angle at 0.70 instead of 0.35",
                     neighbours=True, fused_geometry=True),
    CalculatorConfig("sfe_voxel8", _sfe_voxel8,
                     "the pre-2026-08-15 SFE: fed the 8 mm contact cloud (the control)",
                     neighbours=True),
    CalculatorConfig("sfe_fallback", _sfe_fallback,
                     "SFE, but an abstention keeps the silhouette instead of vetoing it",
                     neighbours=True),
    # The palm pair. Each is the rung directly above it plus one thing: the planner reasoning about
    # the same gripper part its own collision filter already checks. Two rungs because the fused cloud
    # changes which grasps exist at all, and a fix measured only on the weaker geometry says nothing
    # about the one that ships.
    CalculatorConfig("sfe_palm", _sfe_palm, "+ the planner reasons about the PALM, as the filter does",
                     neighbours=True),
    CalculatorConfig("sfe_fused_palm", _sfe_palm, "sfe_fused + the planner reasons about the PALM",
                     neighbours=True, fused_geometry=True),
    CalculatorConfig("sfe_fused_walls", _sfe, "+ the container WALLS as obstacles",
                     neighbours=True, fused_geometry=True, container_walls=True),
    CalculatorConfig("sfe_fused_scene_walls", _sfe, "+ both neighbours AND walls",
                     neighbours=True, fused_geometry=True, fused_neighbours=True,
                     container_walls=True),
)


class _Segmentation:
    """The whole segmentation contract the calculator needs."""

    def __init__(self, mask: np.ndarray) -> None:
        self.mask = mask


def _acute_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between two lines; a closing axis and its negation are the same grasp."""
    return float(np.degrees(np.arccos(np.clip(abs(float(a @ b)), 0.0, 1.0))))


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(float(a @ b), -1.0, 1.0))))


def _labels_by_role(labels: "Sequence[GraspLabel]") -> dict[str, int]:
    """How many labels of each named part role are on this object, the denominator half of the pair.

    Without it, absence is unreadable. `by_part_role` reports only roles that a candidate matched,
    so a corpus with no handles and a stack that never grasps a handle would produce the same table:
    no handle row. The pair is what separates "the matcher rejects handles" from "handle objects get
    no candidate".
    """
    out: dict[str, int] = {}
    for label in labels:
        role = str(getattr(label, "part_role", "") or "")
        if role:
            out[role] = out.get(role, 0) + 1
    return out


def _verdict_over_parts(grasp: "JawGrasp", geometry: "SceneGeometry", instance_id: int,
                        scene_obstacles: "list[Solid]", *, model: "JawModel | None"):
    """Judge a candidate against every part of the object, each paired with its own siblings.

    `obstacles_for` excludes an object's own parts on purpose, so judging `geometry.objects[i]`
    alone leaves the other parts neither target nor obstacle, while the labeller loops every part
    and pairs each with its siblings (`label_jaw_grasps` in `datagen/grasps/labels.py`). Two rules
    for the same object, and only one of them can agree with the labels.

    A grasp on a handle never enters the body and returns `LINE_MISSES_OBJECT`, so judging the
    primary part alone scores correct handle grasps zero. `COMPOSITE_KINDS` puts the body first for
    mug, jug, pan and bucket and the grip first for hammer, so per-role numbers taken that way
    measure which slot the family builder used, not where the grasp landed. In the other direction
    body precision is inflated, because a body grasp whose fingers pass through the object's own
    handle is never rejected.

    Returns `(verdict, part_index, role)`. A part that accepts wins, because an object is grasped if
    the grasp holds anywhere on it; with no acceptance the primary part's verdict is returned, so
    the rejection histogram stays comparable across runs.
    """
    def judge(target: "Solid", obstacles: "list[Solid]"):
        return check_jaw_grasp(grasp, target, obstacles, model=model)

    return _first_part_that_accepts(geometry, instance_id, judge, scene_obstacles)


def _first_part_that_accepts(geometry: "SceneGeometry", instance_id: int, judge, 
                             scene_obstacles: "list[Solid]"):
    """Apply ``judge(target, obstacles)`` to every part, each paired with its own siblings.

    One shape for both modalities, deliberately: the jaw path and the suction path must not carry
    two different rules for what counts as grasping a composite object.

    A part that accepts wins, because an object is grasped if the grasp holds anywhere on it. With
    no acceptance the primary part's verdict is returned, so the rejection histogram stays
    comparable across runs.
    """
    solids = geometry.parts_of(instance_id)
    if len(solids) <= 1:
        # `parts_of` returns [] only when the instance is absent from `objects`, and the callers all
        # check that first. Typed rather than assumed: passing None into a checker that documents a
        # Solid would be a crash one frame later.
        target = solids[0] if solids else geometry.objects.get(instance_id)
        if target is None:
            raise KeyError(f"instance {instance_id} has no solid in this scene's geometry")
        return judge(target, scene_obstacles), 0, ""
    first = None
    for part_index, target in enumerate(solids):
        siblings = [s for i, s in enumerate(solids) if i != part_index]
        verdict = judge(target, scene_obstacles + siblings)
        if first is None:
            first = (verdict, part_index, geometry.role_of(instance_id, part_index))
        if verdict.ok:
            return verdict, part_index, geometry.role_of(instance_id, part_index)
    return first


def _best_match(
    position: np.ndarray, approach: np.ndarray, axis: np.ndarray | None,
    labels: Sequence[GraspLabel],
) -> tuple[float, float, float, GraspLabel | None]:
    """Closest label as ``(position mm, approach deg, axis deg, the label itself)``.

    Minimised on the worst of the three, each normalised by the working point, so a candidate cannot
    look like a match by being close in one coordinate while being wrong in another. ``axis`` is
    ``None`` for suction, which has no closing direction to compare.

    The label itself comes back because the corpus carries a `part_role` on every label (`body`,
    `handle`, `grip`, `neck`, `head`), and the distances alone cannot say whether a proposal that
    matched landed on the handle or on the body.

    There is no distance threshold here: the argmin is returned however far it is.
    `MATCH_ACCEPT_COST` is applied in `summarise`.
    """
    if not labels:
        return (float("inf"), float("inf"), float("inf"), None)
    best: tuple[float, float, float, GraspLabel | None] = (
        float("inf"), float("inf"), float("inf"), None)
    best_cost = float("inf")
    for label in labels:
        dp = float(np.linalg.norm(position - np.asarray(label.position_mm)))
        da = _angle_deg(approach, np.asarray(label.approach))
        dc = 0.0 if axis is None else _acute_angle_deg(axis, np.asarray(label.closing_axis))
        cost = max(dp / WORKING_POINT[0], da / WORKING_POINT[1], dc / WORKING_POINT[1])
        if cost < best_cost:
            best_cost, best = cost, (dp, da, dc, label)
    return best


def _labels_by_object(rows: Sequence[dict], scene_id: str) -> dict[int, list[GraspLabel]]:
    out: dict[int, list[GraspLabel]] = {}
    for row in rows:
        if row["scene_id"] != scene_id:
            continue
        out.setdefault(row["instance_id"], []).append(GraspLabel(
            scene_id=row["scene_id"], instance_id=row["instance_id"], kind=row["kind"],
            position_mm=tuple(row["position_mm"]), approach=tuple(row["approach"]),
            closing_axis=tuple(row["closing_axis"]), width_mm=row["width_mm"],
            approach_tilt_deg=row["approach_tilt_deg"], contact_angle_deg=row["contact_angle_deg"],
            cup=row.get("cup", ""), payload_ok=row.get("payload_ok", True),
            # The labeller writes `part_role` on every composite label and `GraspLabel` defaults
            # it to "", so a reconstruction that forgets the field fails silently: the labels look
            # fine, the matcher works, and every attribution is empty.
            part_role=row.get("part_role", ""),
        ))
    return out


@dataclass(frozen=True, slots=True)
class _NeighbourScene:
    """Every object's fused cloud, sliceable as "everything except object i", in BASE mm.

    Built once per scene rather than per (object, view): thinning every cloud to an 8 mm voxel and
    then masking is one pass, whereas vstacking the others for each object in turn is one pass per
    object.

    This is the ceiling, not the production path. The clouds come from ground-truth instance masks
    fused across every rendered view, so it answers "is there a lever here at all" with perfect
    segmentation and perfect association. The runtime seam
    (``grasping.fusion.geometry.neighbour_scene_enabled``) has to earn each of those, and the
    predicted-mask run is where that shows up.
    """

    points_base_mm: np.ndarray
    owner: np.ndarray

    @classmethod
    def build(cls, clouds: Mapping[int, np.ndarray], voxel_mm: float = 8.0) -> "_NeighbourScene":
        from src.robot.grasping.geometry.sampling import (  # noqa: PLC0415
            voxel_downsample_indices,
        )

        parts: list[np.ndarray] = []
        owners: list[np.ndarray] = []
        for instance_id, cloud in sorted(clouds.items()):
            points = np.asarray(cloud, dtype=np.float64).reshape(-1, 3)
            if not points.size:
                continue
            if voxel_mm > 0.0:
                points = points[voxel_downsample_indices(points, voxel_mm)]
            parts.append(points)
            owners.append(np.full(len(points), int(instance_id), dtype=np.int64))
        if not parts:
            return cls(np.zeros((0, 3), dtype=np.float64), np.zeros(0, dtype=np.int64))
        return cls(np.vstack(parts), np.concatenate(owners))

    def for_object(self, instance_id: int) -> np.ndarray | None:
        if not self.points_base_mm.size:
            return None
        points = self.points_base_mm[self.owner != int(instance_id)]
        return points if points.size else None


def _wall_points_base_mm(payload: dict) -> np.ndarray | None:
    """The scene's own bin walls, sampled by the same helper the pick loop uses.

    The interior box is derived from the wall solids rather than hardcoded: for each horizontal
    axis, the two walls that are thin on that axis bound the interior at their inner faces, and the
    real wall thickness is used so the sampled shell matches the solid the verdict judges against.
    A scene with no ``bin_walls`` (every family except ``bin``) returns ``None``: absent, which is
    the honest answer, not an empty array that would read as "checked and found nothing".
    """
    from src.robot.grasping.collision import (  # noqa: PLC0415
        container_wall_points_base_mm,
    )

    walls = payload["spec"].get("bin_walls") or ()
    if not walls:
        return None
    low = [0.0, 0.0, 0.0]
    high = [0.0, 0.0, 0.0]
    thickness = 0.0
    for axis in (0, 1):
        facing = [w for w in walls
                  if int(np.argmin(np.asarray(w["half_extents_mm"][:2], dtype=np.float64))) == axis]
        if len(facing) != 2:
            return None
        near, far = sorted(facing, key=lambda w: float(w["center_mm"][axis]))
        half_near = float(near["half_extents_mm"][axis])
        low[axis] = float(near["center_mm"][axis]) + half_near
        high[axis] = float(far["center_mm"][axis]) - float(far["half_extents_mm"][axis])
        thickness = max(thickness, 2.0 * half_near)
    low[2] = min(float(w["center_mm"][2]) - float(w["half_extents_mm"][2]) for w in walls)
    high[2] = max(float(w["center_mm"][2]) + float(w["half_extents_mm"][2]) for w in walls)
    return container_wall_points_base_mm(low, high, thickness_mm=thickness, sample_mm=8.0)


def _evaluate_view(
    geometry: SceneGeometry, view: dict, depth: np.ndarray, instances: np.ndarray,
    labels: dict[int, list[GraspLabel]], *, model: JawModel, configs: Sequence[CalculatorConfig],
    with_suction: bool = True, with_jaw: bool = True,
    fused_clouds: Mapping[int, np.ndarray] | None = None,
    neighbour_scene: "_NeighbourScene | None" = None,
    wall_points_base_mm: np.ndarray | None = None,
    support_cfg: "GraspingSupportConfig | None" = None,
    mask_completion: str = "none",
) -> list[dict]:
    from src.robot.grasping.generation.calculator import GraspCalculator  # noqa: PLC0415
    from src.robot.grasping.suction.synthesis import (  # noqa: PLC0415
        synthesize_suction_grasps,
    )

    from datagen.render.camera import unproject_to_base  # noqa: PLC0415

    # The mask table. Every per-object mask in this view, built once, so the three places that need
    # one cannot disagree, and so the mask-completion policy is applied in exactly one place.
    #
    # `none` reproduces `instances == (id + 1)` exactly, which is the ladder's default measurement.
    # The other policies exist because the two perception sources apply one and this evaluation does
    # not: a top-1 measured with `none` is measured on a silhouette a real cell would have replaced.
    # The detector box here is the mask's own extent, which is exact for ground-truth masks (they
    # are complete by construction) and is why the honest run of this is `--masks gt`.
    from src.robot.perception.mask_completion import (  # noqa: PLC0415
        MaskCompletion,
        complete_mask,
    )

    _policy = MaskCompletion(mask_completion)

    class _BoxOf:
        __slots__ = ("box",)

        def __init__(self, mask: np.ndarray) -> None:
            ys, xs = np.nonzero(mask)
            self.box = (None if xs.size == 0 else
                        (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)))

    _masks: dict[int, np.ndarray] = {}
    for _obj in view["objects"]:
        _iid = int(_obj["instance_id"])
        _raw = instances == (_iid + 1)
        _masks[_iid] = (_raw if _policy is MaskCompletion.NONE
                        else complete_mask(_raw, _BoxOf(_raw), policy=_policy))

    def _mask_of(instance_id: int) -> np.ndarray:
        return _masks.get(int(instance_id), instances == (int(instance_id) + 1))

    from src.config.schema.robot.grasping_schema import (  # noqa: PLC0415
        GraspingGripperGeometryConfig,
        GraspingSupportConfig,
    )
    from src.robot.execution.autonomous_grasp.builders import (  # noqa: PLC0415
        build_gripper_geometry,
    )
    from src.robot.grasping.collision import resolve_support_plane  # noqa: PLC0415

    intrinsics = np.asarray(view["intrinsics"], dtype=np.float64)
    camera_to_base = np.asarray(view["camera_to_base_mm"], dtype=np.float64)
    rows: list[dict] = []
    # What the pick loop passes. An evaluation that withholds either the gripper model or the
    # support config stops being an evaluation of the production path.
    gripper_model = build_gripper_geometry(GraspingGripperGeometryConfig())
    support_cfg = support_cfg or GraspingSupportConfig()

    for config in configs:
        factory = config.make or GraspCalculator
        calculator = factory(**{**config.build(intrinsics, camera_to_base),
                                **_REFERENCE_ONLY_KWARGS})
        for obj in view["objects"] if with_jaw else ():
            instance_id = int(obj["instance_id"])
            target: Solid | None = geometry.objects.get(instance_id)
            if target is None:
                continue
            mask = _mask_of(instance_id)
            visible_px = int(mask.sum())
            # Jaw labels only. This is the coverage denominator, and `reverdict` recomputes it
            # the same way, so a fresh file and a re-judged one have to agree: counting suction
            # labels here too would inflate the denominator and make the two incomparable.
            jaw_labels = [label for label in labels.get(instance_id, ()) if label.kind == "jaw"]
            base = {
                "scene_id": geometry.scene_id, "family": geometry.family, "view": view["name"],
                "instance_id": instance_id, "asset_id": target.asset_id, "config": config.name,
                "visibility": float(obj.get("visibility", 0.0)), "visible_px": visible_px,
                "jaw_labels": len(jaw_labels),
                "jaw_labels_by_role": _labels_by_role(jaw_labels),
                # The generation this row was written by, so a summariser can tell it from a row
                # that means zero.
                "row_schema": EVAL_ROW_SCHEMA,
            }
            if visible_px < MIN_VISIBLE_PX:
                rows.append({**base, "kind": "jaw", "outcome": "not_visible", "candidates": 0})
                continue
            obstacles = geometry.obstacles_for(instance_id)
            extra: dict = {}
            if config.neighbours:
                # Exactly what pick_loop.py builds: the other segmentations of this view.
                extra["other_object_masks"] = [
                    _mask_of(other["instance_id"]) for other in view["objects"]
                    if int(other["instance_id"]) != instance_id
                    and int(_mask_of(other["instance_id"]).sum()) > 0
                ]
            if config.dense_sampling is not None:
                extra["dense_sampling"] = config.dense_sampling
            extra["gripper_model"] = gripper_model
            # The support plane, resolved exactly as pick_loop._resolve_support does it: the declared
            # height raised by the target's own lowest point, fused across cameras when a fused cloud
            # is available for this rung and from this view alone otherwise.
            fused_here = (fused_clouds or {}).get(instance_id) if config.fused_geometry else None
            own = unproject_to_base(depth, mask, camera_to_base, intrinsics)
            resolution = resolve_support_plane(
                declared_height_mm=support_cfg.height_mm,
                container_floor_mm=support_cfg.container.floor_height_mm,
                normal=support_cfg.normal,
                target_clouds_base_mm=[c for c in (fused_here, own) if c is not None and c.size],
                refine_from_target=support_cfg.refine_from_target,
            )
            extra["support_plane"] = resolution.plane
            # The same clearance the pick loop passes, read from the same config field rather than
            # left to compute()'s default: an evaluation that hardcodes a threshold the production
            # path can change is measuring a system nobody runs.
            extra["min_table_clearance_mm"] = support_cfg.min_clearance_mm
            base["support_height_mm"] = round(resolution.height_mm, 2)
            if config.fused_geometry and fused_clouds is not None:
                cloud = fused_clouds.get(instance_id)
                if cloud is not None and cloud.shape[0] > 0:
                    extra["geometry_points_base_mm"] = cloud
            # The obstacle half. ``scene_points_mm`` is CAMERA-frame (the frame the collision filter
            # plans in), exactly as pick_loop maps it before the call. The same-view neighbours
            # ``other_object_masks`` already supplies stay: the calculator vstacks the two, so the
            # overlap is duplicated points, which a proximity query does not care about.
            if config.fused_neighbours and neighbour_scene is not None:
                # Not ``obstacles``: that name already belongs to the ground-truth solids the
                # verdict judges against, and shadowing it would hand the reference a point cloud
                # instead of the meshes.
                obstacle_cloud = neighbour_scene.for_object(instance_id)
                if obstacle_cloud is not None:
                    inverse = np.linalg.inv(camera_to_base)  # BASE -> CAMERA
                    extra["scene_points_mm"] = (
                        (inverse[:3, :3] @ obstacle_cloud.T).T + inverse[:3, 3]
                    )
            # Walls travel on their own kwarg, exactly as pick_loop passes them: alongside the
            # neighbour cloud rather than instead of it, and undilated because they are declared
            # geometry rather than a sparse observation.
            if config.container_walls and wall_points_base_mm is not None:
                inverse = np.linalg.inv(camera_to_base)  # BASE -> CAMERA
                extra["rigid_obstacle_points_mm"] = (
                    (inverse[:3, :3] @ wall_points_base_mm.T).T + inverse[:3, 3]
                )
            started = time.perf_counter()
            candidates = calculator.compute(
                _Segmentation(mask), depth, camera_to_base=camera_to_base, unit="mm", **extra)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if not candidates:
                rows.append({**base, "kind": "jaw", "outcome": "no_candidate", "candidates": 0,
                             "elapsed_ms": round(elapsed_ms, 2)})
                continue
            for rank, point in enumerate(candidates):
                grasp = JawGrasp(np.asarray(point.position, dtype=np.float64),
                                 np.asarray(point.approach, dtype=np.float64),
                                 np.asarray(point.axis, dtype=np.float64),
                                 float(point.grip_width_mm))
                verdict, verdict_part, verdict_role = _verdict_over_parts(
                    grasp, geometry, instance_id, obstacles, model=model)
                dp, da, dc, matched = _best_match(grasp.position_mm, grasp.approach, grasp.closing_axis,
                                         jaw_labels)
                rows.append({
                    **base, "kind": "jaw", "outcome": "candidate", "rank": rank,
                    "candidates": len(candidates), "score": round(float(point.score), 4),
                    # The candidate's own geometry, stored rather than re-derived. The physics sample
                    # has to replay the same grasp, and re-running the generator to recover it would
                    # make the replay depend on the generator reproducing itself exactly.
                    "position_mm": [round(float(v), 4) for v in point.position],
                    "approach": [round(float(v), 6) for v in point.approach],
                    "closing_axis": [round(float(v), 6) for v in point.axis],
                    "valid": bool(verdict.ok), "reason": str(verdict.reason),
                    "object_span_mm": round(verdict.object_span_mm, 2),
                    "commanded_width_mm": round(float(point.grip_width_mm), 2),
                    "contact_angle_deg": round(verdict.contact_angle_deg, 2),
                    "approach_tilt_deg": round(verdict.approach_tilt_deg, 2),
                    "match_position_mm": None if np.isinf(dp) else round(dp, 2),
                    "match_approach_deg": None if np.isinf(da) else round(da, 2),
                    "match_axis_deg": None if np.isinf(dc) else round(dc, 2),
                    # Which part the matched label sits on. Empty for a single-solid object; one of
                    # `body` / `handle` / `grip` / `neck` / `head` for a composite. This is what makes
                    # "found a grasp" separable from "found the grasp on the part that was asked for".
                    "match_part_role": ("" if matched is None else str(matched.part_role)),
                    # Which part the grasp actually holds, from the verdict rather than from the
                    # nearest label. `match_part_role` is an ungated nearest-label attribution with
                    # no distance threshold; this one is the part whose solid the closing line
                    # entered. They answer different questions, and the second is the one "did it
                    # grasp the handle" needs.
                    "verdict_part_role": str(verdict_role),
                    "verdict_part_index": int(verdict_part),
                    "elapsed_ms": round(elapsed_ms, 2),
                })

        # Suction runs once per view, not once per configuration: the configurations are jaw-side
        # knobs and re-running it for each would only inflate the file. The guard is positional, so
        # it means "the first rung, whichever it is", not "the default rung". The rows are written
        # with `config: "suction"` and never read the rung's name.
        if config.name != configs[0].name or not with_suction:
            continue
        for obj in view["objects"]:
            instance_id = int(obj["instance_id"])
            target = geometry.objects.get(instance_id)
            if target is None:
                continue
            mask = _mask_of(instance_id)
            suction_labels = [label for label in labels.get(instance_id, ())
                              if label.kind == "suction" and label.cup == STANDARD_CUP.name]
            base = {
                "scene_id": geometry.scene_id, "family": geometry.family, "view": view["name"],
                "instance_id": instance_id, "asset_id": target.asset_id, "config": "suction",
                "visibility": float(obj.get("visibility", 0.0)), "visible_px": int(mask.sum()),
                "suction_labels": len(suction_labels),
                "row_schema": EVAL_ROW_SCHEMA,
                # The same question, the other modality. Without this the suction rung's part table
                # is empty by construction rather than empirically, which reads identically to "this
                # corpus has no parts". Suction is not a footnote here: an object too short to
                # jaw-grasp top-down can only be reached by the cup, and without the denominator
                # whether it was reached is unanswerable.
                "suction_labels_by_role": _labels_by_role(suction_labels),
            }
            if int(mask.sum()) < MIN_VISIBLE_PX:
                # A row, not a `continue`. An object too small to attempt has to stay in the second
                # denominator, exactly as the jaw pass keeps it. Dropping it would make suction's
                # two denominators identical by construction and compute its coverage only over
                # objects something had already found, which is the selection effect the second
                # denominator exists to expose.
                rows.append({**base, "kind": "suction", "outcome": "not_visible", "candidates": 0})
                continue
            produced = synthesize_suction_grasps(_Segmentation(mask), depth, intrinsics)
            if not produced:
                rows.append({**base, "kind": "suction", "outcome": "no_candidate", "candidates": 0})
                continue
            obstacles = geometry.obstacles_for(instance_id)
            for rank, cup_grasp in enumerate(produced):
                # synthesize_suction_grasps returns CAMERA frame without a transform; ask in BASE.
                position = np.asarray(cup_grasp.position_mm, dtype=np.float64)
                approach = np.asarray(cup_grasp.approach, dtype=np.float64)
                position = camera_to_base[:3, :3] @ position + camera_to_base[:3, 3]
                approach = camera_to_base[:3, :3] @ approach
                def judge_cup(solid: "Solid", against: "list[Solid]",
                              _p=position, _a=approach):
                    return check_suction_grasp(SuctionGrasp(_p, _a), solid, against,
                                               cup=STANDARD_CUP)

                verdict, suction_part, suction_role = _first_part_that_accepts(
                    geometry, instance_id, judge_cup, obstacles)
                dp, da, _, matched = _best_match(position, approach, None, suction_labels)
                rows.append({
                    **base, "kind": "suction", "outcome": "candidate", "rank": rank,
                    "candidates": len(produced), "quality": round(float(cup_grasp.quality), 4),
                    "seal_score": round(float(cup_grasp.seal_score), 4),
                    "valid": bool(verdict.ok), "reason": str(verdict.reason),
                    "contact_angle_deg": round(verdict.contact_angle_deg, 2),
                    "approach_tilt_deg": round(verdict.approach_tilt_deg, 2),
                    "match_position_mm": None if np.isinf(dp) else round(dp, 2),
                    "match_approach_deg": None if np.isinf(da) else round(da, 2),
                    # Written, not omitted, and 0.0 by convention: a cup has no closing axis, so
                    # this is not a measurement here. An absent column would be defaulted to a large
                    # angle by the recall curve and pin the whole suction curve to zero.
                    "match_axis_deg": 0.0,
                    "match_part_role": ("" if matched is None else str(matched.part_role)),
                    "verdict_part_role": str(suction_role),
                    "verdict_part_index": int(suction_part),
                })
    return rows




def _scene_rows(
    scene_dir: Path, *, ordinal0: int, assets: SceneAssets, label_rows, model: JawModel,
    configs: Sequence[CalculatorConfig], mask_source: str, with_jaw: bool, with_suction: bool,
    suction_every: int, support_cfg: "GraspingSupportConfig | None",
    mask_completion: str = "none",
) -> "tuple[int, list[dict]]":
    """Every row one scene produces, for every configuration. Returns ``(views, rows)``.

    Separate from :func:`evaluate_dataset` so a worker process can call it: a scene reads only its
    own files and the dataset-wide label and extent tables, and writes nothing, so scenes are
    independent and the loop over them parallelises exactly. ``ordinal0`` is the scene's 0-based
    position among the accepted scenes, which is what the suction subsample keys on; it is passed in
    rather than counted, because a worker cannot know how many scenes came before it.
    """
    from PIL import Image  # noqa: PLC0415

    from datagen.grasps.masks import PRED_SUFFIX  # noqa: PLC0415

    suffix = "instances" if mask_source == "gt" else PRED_SUFFIX
    payload = json.loads((scene_dir / "scene.json").read_text(encoding="utf-8"))
    geometry = assets.geometry(payload, scene_dir.name)
    labels = _labels_by_object(label_rows, scene_dir.name)
    fused_clouds = (fuse_instance_clouds(scene_dir, payload, geometry, mask_suffix=suffix)
                    if any(c.fused_geometry for c in configs) else None)
    neighbour_scene = (
        _NeighbourScene.build(fused_clouds)
        if fused_clouds is not None and any(c.fused_neighbours for c in configs)
        else None
    )
    wall_points = (_wall_points_base_mm(payload)
                   if any(c.container_walls for c in configs) else None)
    out: list[dict] = []
    views = 0
    for view in payload["views"]:
        if str(view.get("outcome")) not in ("rendered", "rendered_after_resample"):
            continue
        depth = np.asarray(Image.open(scene_dir / f"{view['name']}_depth.png"), dtype=np.float64)
        mask_path = scene_dir / f"{view['name']}_{suffix}.png"
        if not mask_path.exists():
            continue        # a view the mask pass has not covered; absent, not silently empty
        instances = np.asarray(Image.open(mask_path))
        rows = _evaluate_view(geometry, view, depth, instances, labels,
                              model=model, configs=configs, with_jaw=with_jaw,
                              with_suction=with_suction and ordinal0 % max(1, suction_every) == 0,
                              fused_clouds=fused_clouds, neighbour_scene=neighbour_scene,
                              wall_points_base_mm=wall_points, support_cfg=support_cfg,
                              mask_completion=mask_completion)
        for row in rows:
            row["masks"] = mask_source
            out.append(row)
        views += 1
    return views, out


#: Per-worker state, built once by :func:`_pool_init` instead of being pickled with every scene. The
#: label table and asset facts are dataset-wide and identical for every task; shipping them per task
#: would dominate the transfer.
_POOL: dict = {}


def _pool_init(root_str: str, mask_source: str, model: JawModel,
               configs: Sequence[CalculatorConfig], support_cfg, with_jaw: bool,
               with_suction: bool, suction_every: int, mask_completion: str = "none") -> None:
    root = Path(root_str)
    _POOL.update(
        root=root, assets=scene_assets(root),
        label_rows=[json.loads(line) for line
                    in (root / "grasps.jsonl").read_text(encoding="utf-8").splitlines()],
        mask_source=mask_source, model=model, configs=configs, support_cfg=support_cfg,
        with_jaw=with_jaw, with_suction=with_suction, suction_every=suction_every,
        mask_completion=mask_completion,
    )


def _pool_scene(task: "tuple[str, int]") -> "tuple[int, list[dict]]":
    name, ordinal0 = task
    return _scene_rows(
        _POOL["root"] / "scenes" / name, ordinal0=ordinal0, assets=_POOL["assets"],
        label_rows=_POOL["label_rows"], model=_POOL["model"], configs=_POOL["configs"],
        mask_source=_POOL["mask_source"], with_jaw=_POOL["with_jaw"],
        with_suction=_POOL["with_suction"], suction_every=_POOL["suction_every"],
        support_cfg=_POOL["support_cfg"], mask_completion=_POOL["mask_completion"],
    )



def select_configurations(names: str | None) -> tuple[CalculatorConfig, ...]:
    """The rungs a run should measure: all of the default ladder, or the named subset.

    ``None`` returns `CONFIGURATIONS` unchanged.

    An unknown name refuses and lists what there is. Quietly running only the rungs it recognised
    would make a run with rungs that no longer exist read exactly like a run that worked.
    """
    if names is None:
        return CONFIGURATIONS
    known = {config.name: config for config in (*CONFIGURATIONS, *DEEP_CONFIGURATIONS)}
    wanted = [part.strip() for part in str(names).split(",") if part.strip()]
    missing = [name for name in wanted if name not in known]
    if missing:
        raise ValueError(f"unknown rung(s) {missing}; available: {sorted(known)}")
    if not wanted:
        raise ValueError("--rungs was given but named nothing")
    return tuple(known[name] for name in wanted)


def _refuse_a_drifted_manifest(root: Path) -> None:
    """Refuse to grade a dataset whose assets no longer rebuild to what the build recorded.

    Fail-closed here, while `prompts/build.py` only warns. An evaluation reconstructs each object's
    solid from the manifest and judges every candidate against it, so a drifted manifest grades
    proposals against geometry the images do not show. The prompts describe sizes in words; this
    decides numbers, and there is nothing useful it can produce from a drifted manifest.

    The usual cause is a schema field added since the build: a stamp written before the field has no
    such key and takes the new default, and a default that changes which family a composite slot
    draws changes the manifest, because `rng.choice` selects by index. The rebuilt hash and the
    recorded one then disagree, and the ids they share come back with different extents.
    """
    stamp_path = root / "provenance.json"
    if not stamp_path.is_file():
        return                                # a hand-assembled directory has nothing to check against
    try:
        stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
        recorded = str(stamp["asset_manifest_sha256"])
        from datagen.build import build_manifest  # noqa: PLC0415 (only this path needs it)
        from datagen.config import DatagenConfig  # noqa: PLC0415
        from datagen.provenance import sha256_of  # noqa: PLC0415, SLF001

        rows = [record.as_row() for record in build_manifest(DatagenConfig.from_stamp(stamp["config"]))]
    except KeyError:
        return                                # an older stamp without the field; nothing to compare
    rebuilt = sha256_of(rows)
    if rebuilt != recorded:
        raise ValueError(
            f"{root}: the asset manifest rebuilt from this dataset's own provenance hashes to "
            f"{rebuilt[:16]} over {len(rows)} row(s), but the build recorded {recorded[:16]}. "
            "Every object's solid, and therefore every `valid` verdict and every rejection "
            "reason, would be judged against geometry this dataset does not contain. Refusing. "
            "Either a schema field added since the build changes what its default fills in "
            "(pass the value the build used explicitly in the stamped config, or re-stamp the "
            "dataset), or the asset files on disk have changed under it, in which case re-render."
        )


def evaluate_dataset(
    root: Path, *, limit: int | None = None, configs: Sequence[CalculatorConfig] = CONFIGURATIONS,
    model: JawModel | None = None, progress_every: int = 25, suction_every: int = 1,
    out_name: str = "grasp_eval.jsonl", mask_source: str = "gt", scene_step: int = 1,
    with_jaw: bool = True, with_suction: bool = True, files: Sequence[str] | None = None,
    support_cfg: "GraspingSupportConfig | None" = None, jobs: int = 1,
    mask_completion: str = "none",
) -> dict:
    """Run every rung over every rendered view, one row per candidate. Returns a summary.

    The rows go to ``out_name`` in ``root``, which is ``grasp_eval.jsonl`` by default.

    ``suction_every`` subsamples the suction pass, because ``synthesize_suction_grasps`` dominates
    this evaluation's runtime: nearly all of its cost is ``estimate_surface_normals``, which
    computes a normal for every cloud point before ``max_eval_candidates=80`` throws all but 80
    away. Every Nth scene keeps the suction sample honest and the jaw pass complete; the sample size
    lands in the report so no rate is quoted without its n.
    """
    root = Path(root)
    model = model or JawModel()
    if mask_source not in ("gt", "pred"):
        raise ValueError(f"mask_source must be 'gt' or 'pred', got {mask_source!r}")

    # The accepted scenes, with the ordinal the suction subsample keys on, decided up front. Both
    # paths read the same list, so the parallel run cannot drift from the sequential one on which
    # scenes it runs or which of them get a suction pass.
    _refuse_a_drifted_manifest(root)

    tasks: list[tuple[str, int]] = []
    index = 0
    for scene_dir in sorted((root / "scenes").iterdir()):
        if not (scene_dir / "scene.json").exists():
            continue
        if limit is not None and len(tasks) >= limit:
            break
        index += 1
        # Every Nth scene, in the renderer's own order, which is alphabetical by family. That keeps
        # the family mix of a subset close to the whole; a contiguous slice would be one family.
        if (index - 1) % max(1, scene_step):
            continue
        tasks.append((scene_dir.name, len(tasks)))

    # A limit selects a family, and that has to be said out loud rather than found later. Scene ids
    # are `<family>_<index>`, so sorted order is family order and a contiguous slice is one family:
    # every rate a limited run reports then describes that family, not the corpus.
    if limit is not None:
        chosen = {name.rsplit("_", 1)[0] for name, _ in tasks}
        available = {d.name.rsplit("_", 1)[0] for d in (root / "scenes").iterdir()
                     if (d / "scene.json").exists()}
        if len(chosen) < len(available):
            logger.warning(
                "--limit %d selected %d scene(s) covering only %s, while this dataset holds %s. "
                "Scenes are walked in sorted order and ids sort by family, so a small limit returns "
                "one family and every rate below describes that family, not the corpus. Drop the "
                "limit, or raise it past the first family.",
                limit, len(tasks), sorted(chosen), sorted(available))

    handle = (root / out_name).open("w", encoding="utf-8")
    scenes = views = written = 0
    started = time.perf_counter()
    # The four things that decide what this run means, in one line. `--summary-only` returns in
    # seconds from an old file and reads exactly like a fast run, so a run that actually drove the
    # generator has to be distinguishable from one that did not, long afterwards.
    logger.info("evaluating %s: %d scene(s), masks=%s, mask_completion=%s, %d config(s) [%s], "
                "jobs=%d -> %s",
                root, len(tasks), mask_source, mask_completion, len(configs),
                ", ".join(c.name for c in configs), jobs, out_name)

    def _record(result: "tuple[int, list[dict]]") -> None:
        nonlocal scenes, views, written
        view_count, rows = result
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        scenes += 1
        views += view_count
        written += len(rows)
        if progress_every and scenes % progress_every == 0:
            rate = (time.perf_counter() - started) / scenes
            print(f"  {scenes} Szenen, {views} Ansichten, {written} Zeilen "
                  f"({rate:.2f} s/Szene)", flush=True)

    try:
        if jobs <= 1:
            assets = scene_assets(root)
            label_rows = [json.loads(line) for line
                          in (root / "grasps.jsonl").read_text(encoding="utf-8").splitlines()]
            for name, ordinal0 in tasks:
                _record(_scene_rows(
                    root / "scenes" / name, ordinal0=ordinal0, assets=assets,
                    label_rows=label_rows, model=model, configs=configs, mask_source=mask_source,
                    with_jaw=with_jaw, with_suction=with_suction, suction_every=suction_every,
                    support_cfg=support_cfg, mask_completion=mask_completion))
        else:
            # One scene per task, results consumed in submission order. ``Executor.map`` yields in
            # input order regardless of completion order, so the rows land in the same order as the
            # sequential path and the two files agree row for row, apart from ``elapsed_ms``, a
            # wall-clock measurement of the call that differs between any two runs, sequential ones
            # included. BLAS threading is pinned to 1 in the children: at this width the work already
            # saturates the cores, and letting each worker spawn its own thread pool turns a 32-core
            # box into 32 workers fighting over 32 threads each.
            import concurrent.futures as _futures  # noqa: PLC0415
            import os  # noqa: PLC0415

            for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS"):
                os.environ.setdefault(var, "1")
            print(f"  parallel: {jobs} workers over {len(tasks)} scenes", flush=True)
            with _futures.ProcessPoolExecutor(
                max_workers=jobs, initializer=_pool_init,
                initargs=(str(root), mask_source, model, configs, support_cfg,
                          with_jaw, with_suction, suction_every, mask_completion),
            ) as pool:
                for result in pool.map(_pool_scene, tasks, chunksize=1):
                    _record(result)
    finally:
        handle.close()
    out_path = root / out_name
    logger.info("wrote %d row(s) from %d scene(s)/%d view(s) in %.1f s -> %s (%d bytes)",
                written, scenes, views, time.perf_counter() - started, out_path,
                out_path.stat().st_size if out_path.exists() else 0)
    # A run reports on what it wrote, not on whatever else is in the directory. `summarise` globs
    # every `grasp_eval*.jsonl` when it is given no file list, so a run into a directory holding
    # older ones would return a mixture nobody chose: files from different runs, different rungs and
    # different scoring code, pooled into one rate. Rows from different schema generations do not
    # pool at all: `summarise` refuses them.
    #
    # Passing `files=` explicitly still overrides, and the whole-directory report is still one call
    # away; it is just no longer what a run silently returns about itself.
    return summarise(root, files=files if files is not None else [out_name])


def reverdict(root: Path, *, model: JawModel | None = None) -> dict:
    """Re-judge every stored candidate against the geometry, without re-running the generator.

    This is what storing each candidate's own position, approach and closing axis buys. When the
    gripper model or the verdict changes, the candidates are unchanged and only the verdict moves,
    so re-deciding costs minutes of CPU instead of hours of re-generation. It cannot quietly become
    a different experiment: the rows it reads are the ones the calculator produced.
    """
    root = Path(root)
    # The same fail-closed check the fresh pass runs. Re-judging is exactly where a drifted manifest
    # does the most damage: the candidates are real and only the verdict moves, so a wrong solid
    # rewrites `valid` for every row in place with nothing left to compare against.
    _refuse_a_drifted_manifest(root)
    model = model or JawModel()
    assets = scene_assets(root)
    label_rows = [json.loads(line)
                  for line in (root / "grasps.jsonl").read_text(encoding="utf-8").splitlines()]

    changed = rejudged = 0
    for path in sorted(root.glob("grasp_eval*.jsonl")):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        by_scene: dict[str, SceneGeometry] = {}
        labels_cache: dict[str, dict[int, list[GraspLabel]]] = {}
        for row in rows:
            # Every row is stamped, not only the ones re-judged below. `reverdict` lifts a whole
            # file to the current rule, and stamping only the candidates would leave `no_candidate`,
            # `not_visible` and suction rows on the old generation, so `summarise` would meet two
            # generations under one config key and refuse after the file had already been rewritten.
            # The stamp goes first for the same reason: a row this pass touches is a row this pass
            # guarantees.
            row["row_schema"] = EVAL_ROW_SCHEMA
            if row.get("kind") != "jaw":
                continue
            scene_id = row["scene_id"]
            if scene_id not in by_scene:
                payload = json.loads(
                    (root / "scenes" / scene_id / "scene.json").read_text(encoding="utf-8"))
                by_scene[scene_id] = assets.geometry(payload, scene_id)
                labels_cache[scene_id] = _labels_by_object(label_rows, scene_id)
            geometry = by_scene[scene_id]
            jaw_labels = [label for label in labels_cache[scene_id].get(row["instance_id"], ())
                          if label.kind == "jaw"]
            # Every jaw row gets the fresh count, not only the ones carrying a candidate. A
            # no_candidate row is still an object in the coverage denominator, and leaving its stale
            # count behind would give each configuration a different denominator, which turns a
            # comparison between rungs into a comparison between bookkeeping.
            row["jaw_labels"] = len(jaw_labels)
            row["jaw_labels_by_role"] = _labels_by_role(jaw_labels)
            target = geometry.objects.get(row["instance_id"])
            if target is None or row.get("outcome") != "candidate" or "position_mm" not in row:
                continue
            grasp = JawGrasp(np.asarray(row["position_mm"], dtype=np.float64),
                             np.asarray(row["approach"], dtype=np.float64),
                             np.asarray(row["closing_axis"], dtype=np.float64),
                             float(row.get("commanded_width_mm", 0.0)))
            # The same rule as the fresh pass, deliberately. If this site judged against
            # `geometry.objects[i]` alone while `_evaluate_view` judged every part, a re-judged run
            # and a fresh run would report different numbers from the same file.
            verdict, verdict_part, verdict_role = _verdict_over_parts(
                grasp, geometry, int(row["instance_id"]),
                geometry.obstacles_for(row["instance_id"]), model=model)
            dp, da, dc, matched = _best_match(
                grasp.position_mm, grasp.approach, grasp.closing_axis, jaw_labels)
            rejudged += 1
            changed += int(bool(verdict.ok) != bool(row.get("valid")))
            row.update({
                "valid": bool(verdict.ok), "reason": str(verdict.reason),
                "object_span_mm": round(verdict.object_span_mm, 2),
                "contact_angle_deg": round(verdict.contact_angle_deg, 2),
                "approach_tilt_deg": round(verdict.approach_tilt_deg, 2),
                "match_position_mm": None if np.isinf(dp) else round(dp, 2),
                "match_approach_deg": None if np.isinf(da) else round(da, 2),
                "match_axis_deg": None if np.isinf(dc) else round(dc, 2),
                "match_part_role": ("" if matched is None else str(matched.part_role)),
                "verdict_part_role": str(verdict_role),
                "verdict_part_index": int(verdict_part),
            })
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    report = summarise(root)
    report["rejudged"] = rejudged
    report["verdict_flipped"] = changed
    return report


def _match_cost(row: "Mapping[str, Any]") -> float:
    """The stored match, in working points, re-derived from the row's own distance columns.

    Deliberately not read from a stored `match_cost`. Every row this file writes carries the three
    distances, so re-deriving here means a file written before the gate existed reports the same
    table a fresh run does, and no role attribution can survive a re-summarise ungated.

    A missing axis is 0.0, not a large angle. Suction has no closing axis, so there is nothing to
    compare, and defaulting it to a large angle would pin the whole suction recall curve to zero.

    A row with no distance at all returns `inf`, which the caller must treat as "never measured"
    rather than as "too far": they are different absences, and conflating them is the defect this
    whole gate is about.
    """
    position = row.get("match_position_mm")
    if position is None:
        return float("inf")
    approach = row.get("match_approach_deg") or 0.0
    axis = row.get("match_axis_deg") or 0.0
    return max(float(position) / WORKING_POINT[0],
               float(approach) / WORKING_POINT[1],
               float(axis) / WORKING_POINT[1])


def _rows_of(path: Path) -> list[dict]:
    """Parse one row file, naming the file and the line when a row will not parse.

    A bare ``json.loads(line)`` raises ``Expecting value: line 1 column 1`` and names neither the
    file nor the row, which is a decoding error about a string the caller cannot see. A torn row is
    rare, so the fragment travels with the exception rather than being reconstructed by hand.
    """
    out: list[dict] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{path}:{number} is not a JSON row ({exc.msg}). "
                f"{len(line)} chars, starts {line[:60]!r}, ends {line[-40:]!r}. "
                f"The file holds {number - 1} good row(s) before it."
            ) from exc
    return out


def summarise(root: Path, *, files: Sequence[str] | None = None,
              match_cost: float = MATCH_ACCEPT_COST) -> dict:
    """Every rate and curve, derived from the candidate rows alone: no re-run, no GPU.

    ``files`` names exactly which row files to read. Without it, every ``grasp_eval*.jsonl`` in the
    directory is used, so a rung added later can be run on its own and still land in the same
    comparison instead of costing a full re-run. The gate passes its own file, because a gate that
    quietly folded in the last full run would compare a subset against a whole.
    """
    root = Path(root)
    # The glob pools every run that ever wrote into this directory, including verdicts from before
    # a scoring change, so its report is a mixture nobody chose. Pass `files=` to summarise one run.
    # The glob stays the default because callers depend on it, but it says what it did.
    paths = ([root / name for name in files] if files is not None
             else sorted(root.glob("grasp_eval*.jsonl")))
    if files is None and len(paths) > 1:
        logger.warning(
            "summarise(%s) is pooling %d eval files (%s). These may come from different runs, "
            "different rungs and different scoring code. Pass files=[...] to report one run.",
            root, len(paths), ", ".join(p.name for p in paths[:6]) + ("..." if len(paths) > 6 else ""),
        )
    rows = [row for path in paths if path.exists() for row in _rows_of(path)]
    report: dict = {"rows": len(rows), "working_point": list(WORKING_POINT),
                    "match_accept_cost": float(match_cost), "by_config": {}}
    # The mask source is part of the key, not a footnote. Both passes write the same configuration
    # names, and folding them together would average a ground-truth silhouette with a predicted one.
    for row in rows:
        row["key"] = (row["config"] if row.get("masks", "gt") == "gt"
                      else f"{row['config']}/{row['masks']}")

    for config in sorted({row["key"] for row in rows}):
        subset = [row for row in rows if row["key"] == config]
        candidates = [row for row in subset if row["outcome"] == "candidate"]
        objects = {(row["scene_id"], row["view"], row["instance_id"]): row for row in subset}
        valid = [row for row in candidates if row.get("valid")]
        entry: dict = {
            "candidates": len(candidates),
            "valid": len(valid),
            "precision": round(len(valid) / len(candidates), 4) if candidates else 0.0,
            "objects_seen": sum(1 for row in objects.values() if row["outcome"] != "not_visible"),
            "objects_no_candidate": sum(1 for row in objects.values()
                                        if row["outcome"] == "no_candidate"),
            "rejections": {},
            "by_family": {},
            "by_visibility": {},
            "recall_curve": {},
            "by_part_role": {},
        }
        for row in candidates:
            if not row.get("valid"):
                entry["rejections"][row["reason"]] = entry["rejections"].get(row["reason"], 0) + 1
        entry["rejections"] = dict(sorted(entry["rejections"].items(), key=lambda kv: -kv[1]))

        # Which part the accepted grasps land on. A single top1 cannot tell a mug grasped by its
        # handle from the same mug grasped round the body: both are grasps, and only one is the
        # affordance a person asked for.
        #
        # Reported as a pair per role, because either half alone misleads. `valid` counts accepted
        # proposals that matched a label of that role; `labels` counts how many of that role were
        # there to find. A calculator that only ever grasps the body scores well on `body` and
        # reveals itself only against the handle's denominator.
        parts: dict[str, dict[str, int]] = {}

        def _bucket(role: str) -> dict[str, int]:
            return parts.setdefault(role, {"valid": 0, "candidates": 0, "labels": 0,
                                           "objects": 0, "objects_not_visible": 0})

        # The denominator first, over every object and not only the ones that produced a candidate.
        # An object the stack refused is exactly the object whose handle went ungrasped, so dropping
        # it here would hide what this block exists to surface. `objects` is already one row per
        # (scene, view, instance), so it is the deduplicated denominator the rest of this entry
        # uses; building a second one by hand would be a third denominator.
        #
        # `not_visible` is counted separately rather than folded in: a handle nobody could see is
        # not a handle the stack declined to grasp, and `objects_seen` above draws the same line.
        for row in objects.values():
            not_visible = row.get("outcome") == "not_visible"
            per_role = row.get("jaw_labels_by_role") or row.get("suction_labels_by_role") or {}
            for role, count in per_role.items():
                bucket = _bucket(str(role))
                if not_visible:
                    bucket["objects_not_visible"] += 1
                else:
                    bucket["labels"] += int(count)
                    bucket["objects"] += 1
        # The gate lives here and nowhere else. `_best_match` returns the nearest label at any
        # distance, so an attribution without a threshold names whichever label was closest.
        #
        # Three different absences, counted apart, because collapsing them is the defect:
        #   * no role at all: a single-solid object, nothing to attribute;
        #   * a role but no stored distance: never measured, a file older than the distance columns;
        #   * a role and a distance beyond the tolerance: measured and too far, which is a result.
        unattributed = {"too_far": 0, "never_measured": 0}
        for row in candidates:
            role = str(row.get("match_part_role", "") or "")
            if not role:
                continue                      # a single-solid object has no part to attribute
            cost = _match_cost(row)
            if cost == float("inf"):
                unattributed["never_measured"] += 1
                continue
            if cost > match_cost:
                unattributed["too_far"] += 1
                continue
            bucket = _bucket(role)
            bucket["candidates"] += 1
            bucket["valid"] += int(bool(row.get("valid")))
        # What the gate cost, beside the table rather than inside it. An empty role row with no count
        # next to it reads as "this corpus has no handles"; with the count it reads as "the stack
        # proposed nothing within reach of one", which is the finding.
        entry["by_part_role_unattributed"] = dict(unattributed)
        entry["match_accept_cost"] = float(match_cost)
        # Whether this table can be read at all, stated rather than left to the reader. A row older
        # than a field is not a row that means zero, and reporting the two the same way is the
        # defect the per-role denominator was added to fix.
        #
        # A mixture refuses rather than averages. Rows judged against the primary solid alone and
        # rows judged against every part are two different experiments, and pooling them reports a
        # rate that belongs to neither.
        schemas = {int(row.get("row_schema", 1)) for row in subset}
        if len(schemas) > 1:
            raise ValueError(
                f"{root}: config {config!r} pools rows from schema generations {sorted(schemas)}. "
                f"Generation 3 judges a candidate against every part of a composite with its siblings "
                f"as obstacles; earlier ones judged it against the primary solid alone, so a handle "
                f"grasp could not be valid. These are two experiments, not one. Re-judge the old rows "
                f"with `reverdict`, or summarise them separately with files=[...]."
            )
        schema = min(schemas) if schemas else 1
        entry["row_schema"] = schema
        entry["by_part_role_status"] = (
            "reported" if schema >= 3 else
            "rows predate the part-aware verdict, so no role precision is trustworthy" if schema == 2
            else "rows predate the per-role denominator, so an empty role is unreadable")
        entry["by_part_role"] = {
            role: {**counts,
                   "precision": round(counts["valid"] / counts["candidates"], 4)
                   if counts["candidates"] else 0.0}
            for role, counts in sorted(parts.items(), key=lambda kv: -kv[1]["labels"])
        }

        # Object-level: did any candidate for this object survive the verdict?
        per_object: dict[tuple, dict] = {}
        for row in candidates:
            key = (row["scene_id"], row["view"], row["instance_id"])
            record = per_object.setdefault(key, {
                "family": row["family"], "visibility": row["visibility"],
                "labels": row.get("jaw_labels", row.get("suction_labels", 0)),
                "any_valid": False, "top_valid": False, "matches": [],
            })
            record["any_valid"] |= bool(row.get("valid"))
            # The number a pick actually gets. The robot takes the first candidate, and that is the
            # highest-scored one (`_filter_and_build_candidates` in the calculator sorts by score
            # descending), so precision over the whole pool and coverage over "any valid candidate"
            # both miss what matters: whether the one it executes would close. Without this key the
            # gate cannot see a change to the ranker at all, because neither precision nor coverage
            # moves when only the order does.
            if row.get("rank") == 0:
                record["top_valid"] = bool(row.get("valid"))
            if row.get("match_position_mm") is not None:
                # An absent axis is 0.0, not a large angle. A suction row has no closing axis to
                # compare, so defaulting the missing column to a large angle would put the suction
                # recall curve at a hard zero in every cell of the sweep, beside the jaw curve and
                # reading like a measurement.
                record["matches"].append((row["match_position_mm"],
                                          row.get("match_approach_deg") or 0.0,
                                          row.get("match_axis_deg") or 0.0))
        # Objects the calculator produced nothing for still count: silence is a miss, not an absence.
        # So do objects with too few visible pixels to attempt: they are in the scene, they carry
        # labels, and no single-view method can reach them. Keeping them in a second denominator is
        # what makes the occlusion loss readable instead of averaged away.
        for key, row in objects.items():
            if row["outcome"] in ("no_candidate", "not_visible"):
                per_object.setdefault(key, {
                    "family": row["family"], "visibility": row["visibility"],
                    "labels": row.get("jaw_labels", row.get("suction_labels", 0)),
                    "visible": row["outcome"] != "not_visible",
                    "any_valid": False, "top_valid": False, "matches": [],
                })
        for bucket in per_object.values():
            bucket.setdefault("visible", True)

        graspable = [b for b in per_object.values() if b["labels"] > 0]
        visible = [b for b in graspable if b["visible"]]
        entry["objects_with_labels"] = len(visible)
        entry["objects_with_labels_incl_unseen"] = len(graspable)
        # Every key above counts an object-view, not an object, and the name does not say so. A
        # scene is rendered from several cameras, so one physical object contributes one entry per
        # view it appears in, and a per-view count and a physical count are not comparable.
        #
        # A view is the right unit for coverage: a pick happens from one camera, and an object the
        # stack can take from one angle and not another is genuinely two different situations. So
        # the per-view keys keep their values and their meaning, and the physical count is reported
        # beside them rather than replacing them.
        distinct = {(key[0], key[2]) for key in per_object}                       # (scene, instance)
        distinct_labelled = {(key[0], key[2]) for key, b in per_object.items() if b["labels"] > 0}
        entry["unit_note"] = ("objects_* count OBJECT-VIEWS; distinct_objects_* count physical "
                              "objects. A scene is seen from several cameras.")
        entry["distinct_objects_seen"] = len(distinct)
        entry["distinct_objects_with_labels"] = len(distinct_labelled)
        entry["views_per_object"] = (round(len(per_object) / len(distinct), 3) if distinct else 0.0)
        # Two denominators, side by side. The gap between them is the occlusion loss.
        entry["coverage"] = (round(sum(1 for b in visible if b["any_valid"]) / len(visible), 4)
                             if visible else 0.0)
        # Reported beside coverage, never instead of it: coverage says what the generator can find,
        # top1 says what the robot is handed. The distance between them is the ranker.
        entry["top1"] = (round(sum(1 for b in visible if b.get("top_valid")) / len(visible), 4)
                         if visible else 0.0)
        entry["coverage_incl_unseen"] = (
            round(sum(1 for b in graspable if b["any_valid"]) / len(graspable), 4)
            if graspable else 0.0)

        for family in sorted({b["family"] for b in per_object.values()}):
            fam = [b for b in visible if b["family"] == family]
            fam_all = [b for b in graspable if b["family"] == family]
            entry["by_family"][family] = {
                "objects_with_labels": len(fam),
                "coverage": round(sum(1 for b in fam if b["any_valid"]) / len(fam), 4) if fam else 0.0,
                "coverage_incl_unseen": (round(sum(1 for b in fam_all if b["any_valid"]) / len(fam_all), 4)
                                         if fam_all else 0.0),
            }
        for low, high in ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)):
            seen = [b for b in graspable if low <= b["visibility"] < high]  # every object, seen or not
            entry["by_visibility"][f"{low:.2f}-{high:.2f}"] = {
                "objects": len(seen),
                "coverage": (round(sum(1 for b in seen if b["any_valid"]) / len(seen), 4)
                             if seen else None),
            }
        for dp in POSITION_TOLERANCES_MM:
            for da in ANGLE_TOLERANCES_DEG:
                hit = sum(1 for b in graspable
                          if any(m[0] <= dp and m[1] <= da and m[2] <= da for m in b["matches"]))
                entry["recall_curve"][f"{dp:.0f}mm/{da:.0f}deg"] = (
                    round(hit / len(graspable), 4) if graspable else 0.0)
        report["by_config"][config] = entry
    if files is None:
        (root / "grasp_eval_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


# --------------------------------------------------------------------- the gate

#: The gate's fixed subset. Every 4th scene keeps the family mix of the whole (the renderer's order
#: is alphabetical by family, so a contiguous slice would be one family) and costs minutes instead
#: of an hour. The step and the rungs are part of the contract: change either and the baseline is
#: void.
GATE_SCENE_STEP = 4
#: Two rungs, not the whole ladder. ``sfe`` is the production path, the one a regression would
#: actually reach a customer through. ``sfe_fused`` is the structural result this project builds on,
#: so a change that quietly undoes it has to fail something.
GATE_CONFIGS = ("sfe", "sfe_fused")
#: How far a rate may move before it is a regression.
#:
#: The evaluation is not bit-reproducible. The generator's dense sampler is gated on a wall-clock
#: budget (`compute` in the calculator times the sample step and compares the elapsed milliseconds
#: against a budget), so which candidates exist depends on how busy the machine was: two runs of
#: this gate with no code change differ on a handful of objects. The band is therefore a noise band
#: derived from that spread rather than chosen, and it has to stay wider than it.
#: ``gate_baseline.json`` records the step, the rungs and the tolerance its rates were taken with.
GATE_TOLERANCE_PP = 0.6
GATE_BASELINE = Path(__file__).with_name("gate_baseline.json")
GATE_OUT_NAME = "grasp_eval_gate.jsonl"


def run_gate(root: Path, *, write: bool = False, model: JawModel | None = None) -> dict:
    """Re-measure the gate rungs on a fixed subset and compare against the committed baseline.

    This is what makes the ladder a gate rather than a study: it answers whether a change made
    grasping better or worse, on data the calculator has not seen, and it costs minutes.

    Returns the comparison. ``write`` replaces the baseline instead of checking against it, which is
    a deliberate second step: a gate that regenerates its own reference on failure is not a gate.
    """
    configs = tuple(c for c in CONFIGURATIONS if c.name in GATE_CONFIGS)
    if len(configs) != len(GATE_CONFIGS):
        raise RuntimeError(f"gate rungs {GATE_CONFIGS} are not all in CONFIGURATIONS")
    measured = evaluate_dataset(
        # with_suction=False, not a huge suction_every: the subsample test is true on the first
        # scene whatever N is, so a "disabled" suction pass still runs once and puts a rung with a
        # handful of objects into the baseline, and a rate over so few objects moves on its own.
        #
        # `mask_completion` is deliberately not exposed here. The gate compares against a committed
        # baseline; a regression gate that can be re-pointed at a different mask policy stops
        # comparing like with like, and would report the policy change as a regression in the grasp
        # path. Measure policies with `eval-grasps --mask-completion`, which writes its own file;
        # the gate stays pinned to the default.
        root, configs=configs, model=model, scene_step=GATE_SCENE_STEP, with_suction=False,
        out_name=GATE_OUT_NAME, files=(GATE_OUT_NAME,), progress_every=0,
    )
    current = {
        name: {
            "precision": entry["precision"],
            "top1": entry["top1"],
            "coverage": entry["coverage"],
            "coverage_incl_unseen": entry["coverage_incl_unseen"],
            "candidates": entry["candidates"],
            "objects_with_labels": entry["objects_with_labels"],
        }
        for name, entry in measured["by_config"].items()
    }
    if write:
        GATE_BASELINE.write_text(json.dumps(
            {"scene_step": GATE_SCENE_STEP, "rungs": list(GATE_CONFIGS),
             "tolerance_pp": GATE_TOLERANCE_PP, "rates": current},
            indent=2, sort_keys=True), encoding="utf-8")
        return {"wrote": str(GATE_BASELINE), "rates": current, "ok": True, "deltas": {}}

    if not GATE_BASELINE.exists():
        raise FileNotFoundError(
            f"no baseline at {GATE_BASELINE}; run the gate once with --write-baseline, and commit it"
        )
    baseline = json.loads(GATE_BASELINE.read_text(encoding="utf-8"))
    if baseline.get("scene_step") != GATE_SCENE_STEP or baseline.get("rungs") != list(GATE_CONFIGS):
        raise RuntimeError(
            f"the baseline was taken with {baseline.get('rungs')} every {baseline.get('scene_step')} "
            f"scenes; this gate runs {list(GATE_CONFIGS)} every {GATE_SCENE_STEP}. Comparing them would "
            "compare two different measurements"
        )
    deltas: dict[str, dict] = {}
    ok = True
    for name, want in sorted(baseline["rates"].items()):
        got = current.get(name)
        if got is None:
            deltas[name] = {"missing": True}
            ok = False
            continue
        entry = {}
        for key in ("precision", "top1", "coverage", "coverage_incl_unseen"):
            delta = (got[key] - want[key]) * 100.0
            entry[key] = round(delta, 3)
            # Only a drop fails. An improvement is reported just as loudly and then the baseline is
            # meant to be rewritten deliberately, so that a gain cannot be lost again unnoticed.
            if delta < -GATE_TOLERANCE_PP:
                ok = False
        entry["n"] = got["objects_with_labels"]
        deltas[name] = entry
    return {"ok": ok, "rates": current, "deltas": deltas,
            "tolerance_pp": GATE_TOLERANCE_PP, "baseline": str(GATE_BASELINE)}
