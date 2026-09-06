"""Does a grasp the geometry calls valid actually hold? The pass that answers with a number.

Runs on box: the Isaac referee needs Isaac and a GPU, and every Isaac import is lazy and lives
below this line. `run_physics_sample` also has a MuJoCo referee, which needs neither.

The second half of the sample is the point. Testing only grasps the verdict accepted would measure
its false-positive rate and nothing else: a predicate that says yes to everything would score
perfectly. So the trials are drawn from both sides:

* labels the verdict produced (accepted by construction),
* candidates the verdict accepted,
* candidates the verdict rejected, stratified by the reason it gave.

If the rejected ones fail at the same rate as the accepted ones hold, the predicate does not
discriminate and every number the analytic verdict produces is decoration. That is the result this
module exists to obtain, and it is equally publishable either way.

Four controls run first and the pass refuses to continue without them, because a physics harness
that silently reports "nothing holds" is indistinguishable from one that is wired wrong. Each
referee asks the same four questions its own way:

* the shut jaw must be solid, so the gripper has collision geometry at all. Isaac releases a block
  onto the shut jaw and requires it to rest there; MuJoCo compares that rest height against the
  same drop onto an open jaw, because an absolute height passes even when nothing is simulated,
* a textbook top-down grasp through the centre of a lone block must hold,
* the same grasp made to grip air must not. Isaac translates it 150 mm sideways; MuJoCo commands
  the jaw 60 mm wider than the block,
* and the first control repeated after an unrelated scene has been built and retired in between
  must still hold (the scene lifecycle does not leak).

"Held" is measured by taking the table away, not by lifting the gripper. The gripper is world-fixed
and teleported, and a teleported body transmits no tangential force, so a lift measures the
teleport rather than the grip. Removing the support asks the same question, whether the jaw can
carry this object's weight, with no gripper motion at all, so nothing about how the gripper is
moved can enter the answer.

The gripper is teleported to the grasp pose rather than carried there by an arm, so no verdict here
depends on whether a UR5e could have reached it. That is deliberate and it is also a limit:
reachability is a separate question, asked separately.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from src.utility.log_cfg import create_logger

from datagen.constants import DATAGEN_LOG_DIR, PHYSICS_SAMPLE_LOG_FILE
from datagen.grasps.identity import grasp_identity
from datagen.grasps.labels import scene_assets
from datagen.grasps.shapes import Solid

__all__ = [
    "paired_trials","PhysicsTrial", "TrialOutcome", "sample_trials", "run_physics_sample"]

logger = create_logger("datagen.grasps.physics", PHYSICS_SAMPLE_LOG_FILE, log_dir=DATAGEN_LOG_DIR)

#: Rungs whose rows are a scale for the ladder, not proposals worth spending physics on.
#: `datagen.eval.ladder` registers `floor_random` / `floor_topdown` / `floor_normal`, built by
#: `datagen.eval.floors`.
_FLOOR_PREFIX = "floor_"

_MM_TO_M = 0.001
#: How far back along -approach the gripper starts. Far enough to be clear of the object, short enough
#: that the straight-line approach does not have to cross the whole bin.
_STANDOFF_MM = 90.0
#: Physics steps per phase. 60 Hz substeps; the grip needs time to build force before the table goes.
_STEPS_APPROACH = 40
_STEPS_CLOSE = 60


class PhysicsCell(Protocol):
    """What a physics referee has to be, whichever engine it is.

    `physics_isaac` and `physics_mujoco` each define a class called `PhysicsCell`, and the engine
    switch in `run_physics_sample` binds one of them to this name. The Protocol is what makes them
    one shape: without it an engine that grew a method or dropped one is caught by neither the type
    checker nor a reader.

    It declares a shape, not an agreement. Both engines pass the same four controls, which makes
    each a working harness. It does not make their verdicts interchangeable, and nobody has yet
    measured the two on the same grasps. `engine` is written into every row so a corpus can always
    say which referee shook it.
    """

    def __enter__(self) -> "PhysicsCell": ...

    def __exit__(self, *exc: Any) -> Any: ...

    def build_scene(self, geometry: Any, *, kinematic_objects: bool = False) -> None: ...

    def jaw_fault(self) -> str: ...

    def gripper_offset_mm(self) -> float: ...

    def restore(self, geometry: Any) -> None: ...

    def check_ready(self, geometry: Any, instance_id: int) -> str: ...

    def target_drift_mm(self, geometry: Any, instance_id: int) -> float: ...

    def run_trial(self, trial: Any, geometry: Any) -> Any: ...

    def run_controls(self) -> dict: ...


@dataclass(frozen=True, slots=True)
class PhysicsTrial:
    """One grasp to try, and where it came from."""

    scene_id: str
    instance_id: int
    source: str            # "label" | "valid" | "rejected:<reason>"
    position_mm: tuple[float, ...]
    approach: tuple[float, ...]
    closing_axis: tuple[float, ...]
    width_mm: float
    family: str = ""
    config: str = ""
    #: Which file this candidate was read from, and which line of it. Together they are the trial's
    #: identity, and the only reason a physics result can be joined back to the candidate's features.
    #: The alternative, matching on (scene, instance, position, approach, closing, width), is a join
    #: on float equality: it does not fail loudly, it silently drops the rows that do not match and
    #: reports a smaller dataset. Line index is stable under append, which matters because
    #: `grasp_eval.jsonl` is appended across runs.
    origin: str = ""
    row_index: int = -1

    def as_row(self) -> dict:
        return {
            "scene_id": self.scene_id, "instance_id": self.instance_id, "source": self.source,
            "position_mm": list(self.position_mm), "approach": list(self.approach),
            "closing_axis": list(self.closing_axis), "width_mm": self.width_mm,
            "family": self.family, "config": self.config,
            "origin": self.origin, "row_index": self.row_index,
        }


@dataclass(frozen=True, slots=True)
class TrialOutcome:
    """What physics said. ``held`` is the measurement; the rest is why it should be believed.

    ``rise_mm`` is signed and, under the drop test, negative: it is how far the object moved vertically
    once its support was removed. A held object stays put (~0); a dropped one falls thousands.
    """

    trial: PhysicsTrial
    held: bool
    rise_mm: float
    note: str = ""

    def as_row(self) -> dict:
        return {**self.trial.as_row(), "held": self.held, "rise_mm": round(self.rise_mm, 2),
                "note": self.note}


def sample_trials(
    root: Path, *, per_class: int = 40, seed: int = 20260813,
    reasons: Sequence[str] = ("not_antipodal", "too_wide", "below_table", "finger_collision"),
    configs: Sequence[str] | None = None,
) -> list[PhysicsTrial]:
    """Draw a stratified sample across sources, families and rejection reasons.

    Deterministic given ``seed``: the sample is part of the measurement, so it has to be reproducible
    and it has to be stated. Stratifying by family as well as by source stops the draw from becoming
    a statement about whichever family happens to have the most rows.

    ``configs`` restricts and stratifies by rung, and it exists for one job: comparing two generators
    on the same objects under the same physics. `grasp_eval.jsonl` mixes every rung, so an
    unrestricted draw is dominated by whichever rung produced the most rows, and a generator that
    proposes four times as many candidates as another quietly buys four times the sample, which is
    not a comparison but a weighting. Passing `("sfe_fused", "deep")` keeps only those two and gives
    each its own strata, so each gets ``per_class`` per (source, family).

    ``None`` (the default) is the unrestricted behaviour, filter and stratum key both, so a sample
    drawn without ``configs`` stays byte-identical.

    The `label` pool comes from `grasps.jsonl`, which has no rung at all: those are the analytic
    reference, not a generator's output. Restricting by config therefore drops the label stratum
    entirely, and that is correct, because a rung comparison has no use for it. Do not read the
    absence of `label` rows in a restricted sample as a corpus defect.

    This is stratified, not paired, and the difference decides what the result may claim. Each rung
    gets its own strata, so the two arms come out unequal and land on different objects. That is a
    valid between-groups comparison with object difficulty as a variance source inside it, which in
    this corpus is large, because `bin` objects fail for reasons that have nothing to do with the
    generator. Two arms drawn here can end up sharing a handful of objects, which is two samples of
    almost disjoint populations rather than a paired comparison. Use `paired_trials` when the
    question is "which of these two is better": it draws the same objects into every arm and does
    not pre-filter by the analytic verdict. This function remains the right one for its own
    question, what the rejection categories look like under physics, which is a statement about the
    corpus rather than about two generators.
    """
    wanted = frozenset(configs) if configs is not None else None
    root = Path(root)
    rng = np.random.default_rng(seed)
    pools: dict[str, list[PhysicsTrial]] = {}
    # One grasp, one trial, per file. `grasp_eval.jsonl` is appended across runs, so the same
    # candidate is in it many times. A uniform draw over lines would weight a grasp by how often
    # someone re-evaluated it, and it costs twice: the sample is biased, and only the first copy of a
    # grasp has a feature row in the corpus, so the later copies join to nothing.
    #
    # Per file rather than globally. The pools come out identical either way, because `grasp_identity`
    # carries the view and the analytic labels have none, so a label and a calculator candidate never
    # collide. It is written per file anyway so the invariant is stated rather than inherited from a
    # field that happens to be absent: the join key is per file, so the identity that governs it
    # should be too.
    seen: set[tuple[str, tuple]] = set()
    repeats = 0

    for index, line in enumerate((root / "grasps.jsonl").read_text(encoding="utf-8").splitlines()):
        row = json.loads(line)
        if row["kind"] != "jaw":
            continue
        if wanted is not None:
            # Restricted to rungs means no label pool. These rows are the analytic reference and
            # carry no rung at all, so they would arrive with `config=""` and sit in their own
            # stratum, adding a third arm nobody asked for to a two-arm comparison.
            break
        identity = ("grasps", grasp_identity(row))
        if identity in seen:
            repeats += 1
            continue
        seen.add(identity)
        pools.setdefault("label", []).append(PhysicsTrial(
            row["scene_id"], row["instance_id"], "label", tuple(row["position_mm"]),
            tuple(row["approach"]), tuple(row["closing_axis"]), float(row["width_mm"]),
            origin="grasps", row_index=index))

    eval_path = root / "grasp_eval.jsonl"
    if eval_path.exists():
        for index, line in enumerate(eval_path.read_text(encoding="utf-8").splitlines()):
            row = json.loads(line)
            if row.get("kind") != "jaw" or row.get("outcome") != "candidate":
                continue
            if "position_mm" not in row:
                continue      # written before the candidate vectors were stored; nothing to replay
            # The identity is claimed before the reason filter, and that ordering matters. The
            # corpus dedupes without knowing about `reasons`, so its survivor is the first candidate
            # row, full stop. Filtering first would skip a first occurrence and draw a later copy of
            # the same grasp, which then has no feature row: that happens wherever one eval config
            # calls a grasp `approach_blocked` and another calls the same grasp `not_antipodal`.
            # Claiming first makes "first occurrence here" and "the row the corpus kept" one
            # statement.
            identity = ("grasp_eval", grasp_identity(row))
            if identity in seen:
                repeats += 1
                continue
            seen.add(identity)
            # After the identity claim, like the reason filter and for the same reason: the corpus
            # dedupes without knowing about either, so its survivor is the first candidate row.
            config = str(row.get("config", ""))
            if wanted is not None:
                if config not in wanted:
                    continue
            elif config.startswith(_FLOOR_PREFIX):
                # The floors are not candidates for physics. They exist to give the ladder a scale,
                # they propose 12 poses per object-view where the analytic rungs propose a handful,
                # and an unrestricted draw would therefore spend most of an Isaac night shaking
                # deliberate nonsense. Ask for them by name to include them.
                continue
            source = "valid" if row.get("valid") else f"rejected:{row.get('reason')}"
            if source != "valid" and row.get("reason") not in reasons:
                continue
            pools.setdefault(source, []).append(PhysicsTrial(
                row["scene_id"], row["instance_id"], source, tuple(row["position_mm"]),
                tuple(row["approach"]), tuple(row["closing_axis"]),
                float(row.get("commanded_width_mm", 0.0)),
                family=row.get("family", ""), config=row.get("config", ""),
                origin="grasp_eval", row_index=index))

    # Stratified by (source, family). Keyed by `source` alone, the draw becomes exactly what the
    # docstring warns about: a statement about whichever family happens to have the most rows.
    #
    # Families come from the scene id when a row did not carry one: `grasps.jsonl` has no family
    # column, and the generator names every scene `<family>_<index>`.
    strata: dict[tuple[str, ...], list[PhysicsTrial]] = {}
    for source, pool in pools.items():
        for trial in pool:
            family = trial.family or str(trial.scene_id).rsplit("_", 1)[0]
            # The rung joins the key only when the caller restricted to rungs. Adding it
            # unconditionally would re-cut every stratum in every existing sample.
            key = (source, family) if wanted is None else (source, family, trial.config)
            strata.setdefault(key, []).append(trial)

    logger.info("sample_trials: %d distinct grasp(s) after dropping %d repeat(s)",
                len(seen), repeats)
    drawn: list[PhysicsTrial] = []
    for _stratum, pool in sorted(strata.items()):
        if not pool:
            continue
        take = min(per_class, len(pool))
        picked = rng.choice(len(pool), size=take, replace=False)
        drawn.extend(pool[int(i)] for i in picked)
    return drawn


#: How a paired draw chooses which of an arm's candidates to shake.
#:
#: ``"rank"`` takes the generator's own first choice, which is the question promotion actually asks:
#: "when this thing picks, does its pick hold?" ``"random"`` takes a uniform one, which asks the
#: broader "are its candidates good on average" and is the right mode for estimating a rate rather
#: than for comparing two deciders.
_PAIRED_PICKS = ("rank", "random")


def paired_trials(
    root: Path, *, configs: Sequence[str], per_class: int = 20, seed: int = 20260825,
    pick: str = "rank", require_all: bool = True,
) -> list[PhysicsTrial]:
    """One candidate per arm, on the same objects: the comparison `sample_trials` cannot make.

    `sample_trials(configs=...)` gives each rung its own strata, so the two arms come out unequal and
    land on different objects. Object difficulty then sits inside the comparison as a variance
    source, and in this corpus that is large: `bin` objects fail for reasons that have nothing to do
    with which generator proposed the grasp. Pairing removes it, because every drawn object
    contributes exactly one trial to each arm, so the arms are the only thing left that varies.

    No reason filter, and that is the entire point of the anchor. `sample_trials` selects by the
    analytic verdict (`valid` / `rejected:<reason>`), which is right when the question is "what do
    the rejection categories look like under physics". It is exactly wrong here: a generator trained
    on labels derived from that verdict is partly being graded on mimicry, and pre-filtering its
    candidates by the same verdict would grade the mimicry twice. The referee is the independent
    judge, so it gets the candidate the generator actually chose, valid or rejected.

    An arm that proposed nothing is not a tie, it is a refusal, and refusals are a real difference
    between generators. With ``require_all`` (the default) an object enters the draw only when every
    arm proposed something there, so what gets shaken is like-for-like; the objects dropped for that
    reason are counted and logged, because a hold rate over what an arm proposed says nothing
    without how many objects it proposed on.

    Identity dedupe is per arm here, not global. Two arms picking the same pose is a legitimate tie
    and both get shaken: PhysX is not deterministic here, so those are two samples of one grasp
    rather than one wasted twice.

    Returns a flat list, one trial per arm per object, tagged with `config`. The pairing is recovered
    from `(scene_id, instance_id)`, which is also the join the analysis wants.
    """
    if pick not in _PAIRED_PICKS:
        raise ValueError(f"pick must be one of {_PAIRED_PICKS}, not {pick!r}")
    arms = tuple(dict.fromkeys(configs))
    if len(arms) < 2:
        raise ValueError(f"a paired draw needs at least two arms, got {arms}")

    root = Path(root)
    eval_path = root / "grasp_eval.jsonl"
    if not eval_path.exists():
        raise FileNotFoundError(f"no grasp_eval.jsonl under {root}")

    by_object: dict[tuple[str, int], dict[str, list[dict]]] = {}
    seen: set[tuple[str, tuple]] = set()
    for index, line in enumerate(eval_path.read_text(encoding="utf-8").splitlines()):
        if not line:
            continue
        row = json.loads(line)
        if row.get("kind") != "jaw" or row.get("outcome") != "candidate":
            continue
        if "position_mm" not in row:
            continue      # written before the candidate vectors were stored; nothing to replay
        config = str(row.get("config", ""))
        if config not in arms:
            continue
        identity = (config, grasp_identity(row))
        if identity in seen:
            continue
        seen.add(identity)
        row["_row_index"] = index
        by_object.setdefault((str(row["scene_id"]), int(row["instance_id"])), {}) \
                 .setdefault(config, []).append(row)

    complete = {key: found for key, found in by_object.items()
                if not require_all or all(arm in found for arm in arms)}
    logger.info("paired_trials: %d object(s) carry every arm (%s), %d dropped for a missing arm",
                len(complete), ", ".join(arms), len(by_object) - len(complete))
    if not complete:
        return []

    # Stratified by family over objects, not over candidates: the unit of this draw is the object.
    rng = np.random.default_rng(seed)
    strata: dict[str, list[tuple[str, int]]] = {}
    for key in sorted(complete):
        # From whichever arm is present. Reading `arms[0]` fixed is a KeyError the moment
        # `require_all=False` lets through an object only one arm reached, which is exactly what
        # that option is for.
        present = next(rows for arm in arms if (rows := complete[key].get(arm)))
        family = str(present[0].get("family") or key[0].rsplit("_", 1)[0])
        strata.setdefault(family, []).append(key)

    drawn: list[PhysicsTrial] = []
    identical = 0
    for _family, keys in sorted(strata.items()):
        take = min(per_class, len(keys))
        for position in rng.choice(len(keys), size=take, replace=False):
            key = keys[int(position)]
            chosen: dict[str, dict] = {}
            for arm in arms:
                rows = complete[key].get(arm)
                if not rows:
                    continue
                if pick == "rank":
                    # The generator's own first choice. A row that carries no `rank` sorts last
                    # rather than silently becoming rank 0.
                    row = min(rows, key=lambda r: int(r.get("rank", 10**6)))
                else:
                    row = rows[int(rng.choice(len(rows)))]
                chosen[arm] = row
            if len(chosen) > 1 and len({grasp_identity(row) for row in chosen.values()}) == 1:
                identical += 1
            for arm, row in chosen.items():
                drawn.append(PhysicsTrial(
                    row["scene_id"], row["instance_id"],
                    "valid" if row.get("valid") else f"rejected:{row.get('reason')}",
                    tuple(row["position_mm"]), tuple(row["approach"]), tuple(row["closing_axis"]),
                    float(row.get("commanded_width_mm", 0.0)),
                    family=row.get("family", ""), config=arm,
                    origin="grasp_eval", row_index=int(row["_row_index"])))

    if identical:
        # Visible, because a large share means the two arms are not as different as the comparison
        # assumes, and because it is referee time spent on a near-foregone conclusion.
        logger.info("paired_trials: %d object(s) where every arm picked the SAME pose", identical)
    logger.info("paired_trials: %d trial(s) over %d object(s)", len(drawn),
                len(drawn) // max(len(arms), 1))
    return drawn


# ------------------------------------------------------------------ on-box


def _grasp_frame(approach: np.ndarray, closing_axis: np.ndarray) -> np.ndarray:
    """Rotation whose columns are (closing, binormal, approach), the 2F-85's own convention.

    The mounted gripper's grasp frame is measured in `willy_sim` as approach = +Z of the base after
    the mount rotation, closing = +X. Reproducing that here is what makes a pose computed in BASE
    land as the same physical grasp.
    """
    z = approach / float(np.linalg.norm(approach))
    x = closing_axis - float(closing_axis @ z) * z
    x = x / float(np.linalg.norm(x))
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def trials_from_proposals(path: Path | str) -> list[PhysicsTrial]:
    """Read a generator's own proposals as trials, in the order it ranked them.

    This is what the primary metric is made of: whether a grasp the model proposes actually holds.
    A sample drawn from the labels answers a different question.

    `source` stays `generator` and is not rewritten. Every rate the referee reports is grouped by
    it, so folding model proposals into the label pool would silently change what every hold rate
    already recorded means. Two numbers that answer different questions must not share a name.

    The order is the product and it is preserved. A cell takes the first candidate it can reach, so
    a shuffled list would measure the model's average grasp rather than its first one, and those are
    different claims.
    """
    rows: list[PhysicsTrial] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append(PhysicsTrial(
            scene_id=str(row["scene_id"]), instance_id=int(row["instance_id"]),
            source="generator", position_mm=tuple(row["position_mm"]),
            approach=tuple(row["approach"]), closing_axis=tuple(row["closing_axis"]),
            width_mm=float(row["width_mm"]), family=str(row.get("family", "")),
            config=str(row.get("config", "generator")), origin="proposals",
            row_index=int(row.get("rank", -1))))
    if not rows:
        raise ValueError(f"{path} holds no proposal; the model produced nothing to judge")
    return rows


def run_physics_sample(
    root: Path, *, trials: Sequence[PhysicsTrial] | None = None, per_class: int = 40,
    headless: bool = True, out_name: str = "grasp_physics.jsonl", engine: str = "isaac",
) -> dict:
    """Run the sample on-box. Writes one row per trial as it lands, so a teardown cannot swallow it.

    `engine` chooses the referee, and it is a product decision rather than a preference. Isaac is
    Windows or Linux, NVIDIA only, and about 47 GB, and it is the one step in this pipeline that
    forces a workstation: everything else, rendering included, runs under MuJoCo. `mujoco` therefore
    makes "generate your own data" a claim a customer can act on, and it runs beside other work on a
    machine Isaac would occupy alone.

    A second referee is not automatically the same referee. Both pass the same four controls, which
    makes each a working harness; it does not make their verdicts interchangeable. That measurement
    on the same grasps does not exist yet, so the engine is written into every row and a corpus can
    say which one shook it.
    """
    cell_type: type[PhysicsCell]
    if engine == "mujoco":
        from datagen.grasps.physics_mujoco import PhysicsCell as MujocoCell  # noqa: PLC0415
        cell_type = MujocoCell
    elif engine == "isaac":
        # noqa: PLC0415 (the Isaac half, imported only when asked for: it pulls in a simulator.)
        from datagen.grasps.physics_isaac import PhysicsCell as IsaacCell  # noqa: PLC0415
        cell_type = IsaacCell
    else:
        raise ValueError(f"unknown physics engine {engine!r}; choose from isaac, mujoco")

    root = Path(root)
    trials = list(trials) if trials is not None else sample_trials(root, per_class=per_class)
    assets = scene_assets(root)
    by_scene: dict[str, list[PhysicsTrial]] = {}
    for trial in trials:
        by_scene.setdefault(trial.scene_id, []).append(trial)

    out_path = root / out_name
    counts: dict[str, dict[str, int]] = {}
    logger.info("physics sample over %s: %d trial(s) across %d scene(s), headless=%s -> %s",
                root, len(trials), len(by_scene), headless, out_path)
    with (cell_type(headless=headless, mesh_collision=assets.mesh_collision) as cell,
          out_path.open("w", encoding="utf-8") as handle):
        controls = cell.run_controls()
        controls.setdefault("engine", engine)
        handle.write(json.dumps({"controls": controls}) + "\n")
        handle.flush()
        # Before the two refusals below, and unconditionally: the controls are what makes every hold
        # rate in this file a statement about the grasp rather than about the harness, so they are
        # read first and recorded rather than assumed.
        logger.info("harness controls: %s", controls)
        # The repeat is the control that matters. A positive and a negative can both pass while the
        # harness holds nothing, because a fault can appear only on the second scene build.
        # Requiring the positive to still hold after an unrelated scene has come and gone is what
        # makes the three together a statement about the harness rather than about one lucky grasp.
        if not controls.get("jaw_is_solid"):
            raise RuntimeError(
                f"a block released above the SHUT jaw passed through it ({controls}); the gripper has "
                "no contact with dynamic bodies, so every grasp would report 'did not hold' for a "
                "reason that has nothing to do with the grasp"
            )
        if not (controls.get("positive_held") and not controls.get("negative_held")
                and controls.get("repeat_held")):
            raise RuntimeError(
                f"the physics harness failed its own controls ({controls}); a sample taken with a "
                "harness that cannot grip a block, that grips thin air, or that stops gripping once a "
                "second scene has been built measures nothing"
            )
        for scene_id, scene_trials in sorted(by_scene.items()):
            payload = json.loads(
                (root / "scenes" / scene_id / "scene.json").read_text(encoding="utf-8"))
            geometry = assets.geometry(payload, scene_id)
            # Built once per scene and restored between trials, because rebuilding per trial costs
            # a full authoring pass each time.
            cell.build_scene(geometry)
            for index, trial in enumerate(scene_trials):
                if index:
                    cell.restore(geometry)
                outcome = cell.run_trial(trial, geometry)
                handle.write(json.dumps(outcome.as_row()) + "\n")
                handle.flush()
                bucket = counts.setdefault(trial.source, {"trials": 0, "held": 0, "refused": 0})
                # A refused trial is not a trial that failed. Counting a harness fault as "did not
                # hold" reports a broken harness as a result.
                if outcome.note.startswith("refused"):
                    bucket["refused"] += 1
                    # Per refusal, not per trial: a trial is seconds of physics, and a refusal is the
                    # harness declining to grade one. Silent, they look like a smaller sample.
                    logger.warning("%s/%d %s: %s", scene_id, trial.instance_id, trial.source,
                                   outcome.note)
                    continue
                bucket["trials"] += 1
                bucket["held"] += int(outcome.held)
    # Bound to a typed local rather than read back out of `report`: the report is a heterogeneous
    # dict, so mypy widens its values to `object` and the log line below could not index them.
    by_source: dict[str, dict[str, float]] = {
        k: {**v, "hold_rate": round(v["held"] / v["trials"], 4) if v["trials"] else 0.0}
        for k, v in sorted(counts.items())
    }
    n_trials = sum(v["trials"] for v in counts.values())
    n_refused = sum(v["refused"] for v in counts.values())
    report = {
        "trials": n_trials,
        "drawn": len(trials),
        "refused": n_refused,
        "by_source": by_source,
    }
    # The report follows the output file. A fixed name would let a proposal run writing
    # `grasp_physics_proposals.jsonl` and a label run writing `grasp_physics.jsonl` share one
    # `grasp_physics_report.json`, where the second silently replaces the first. Two runs answering
    # different questions must not share a report, which is the rule that also keeps `source`
    # unrewritten in `trials_from_proposals`.
    (root / f"{Path(out_name).stem}_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("%d trial(s) scored, %d refused, %d drawn; by source %s",
                n_trials, n_refused, len(trials),
                {k: f"{v['held']}/{v['trials']}" for k, v in by_source.items()})
    return report


def solid_top_z(solid: Solid) -> float:
    """Convenience for the harness: the object's highest point, from `Solid.top_z_mm`."""
    return solid.top_z_mm()
