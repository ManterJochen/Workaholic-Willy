"""What a trained generator would actually do, written down so a referee can judge it.

The architecture plan names the referee success rate as its primary metric: does a grasp the model
proposes actually hold. Computing it needs a trained model turned into a list of grasps a physics
cell can try, and so does section 3.6, which asks for a scorer trained on the generator's own
proposals with a physics target. This module produces that list. An artifact goes in and ranked
grasps come out; the physics referee judges them, and the judgment yields both the referee success
rate (metric 1) and the (proposal, held) pairs stage 3.6 needs.

The proposals must be the model's own, unfiltered. A scorer trained on an offline label set sees
only collision-free grasps, while a generator produces slightly colliding ones, so the distribution
shift sits entirely in the negatives and the scorer never learns to refuse the things its own
generator emits. Filtering these proposals through the analytic verdict first is worse still: the
generator is trained on labels derived from that verdict, so pre-filtering grades the mimicry twice.
The referee gets what the model actually chose, valid or not.

The confidence order is the product. A cell takes the first candidate it can reach, so a proposal
list in emission order would measure slot zero of seed zero rather than what the head believes.
`decode_set_prediction` ranks by confidence and this keeps that order, which is also why `--top`
takes a prefix: that prefix is exactly what a cell would try.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

__all__ = ["Proposal", "propose_for_scenes", "serving_shares", "suppress", "distinct_share",
           "write_proposals"]

#: The radius the diagnostic uses, always, whatever the caller suppresses at. Reported rather than
#: applied, so the duplicate rate is a number in the output instead of an assumption in a reader.
_DIAGNOSTIC_MM: float = 20.0
#: Past this, a proposal is not on an object at all. Half the jaw's aperture: a grasp centre further
#: than that from every object point cannot have the object between its fingers.
_OFF_OBJECT_MM: float = 42.5
#: Two proposals count as the same grasp only if they also agree on the approach. Two poses 5 mm
#: apart approaching from opposite sides are two grasps, and a radius alone would merge them.
_DIAGNOSTIC_DEG: float = 20.0


@dataclass(frozen=True, slots=True)
class Proposal:
    """One grasp a model proposed, with everything a referee and a scorer both need.

    It carries its own provenance. A proposal that reached a physics verdict without saying which
    scene, which object and which slot it came from cannot be joined back to the features that
    produced it: a join on pose equality is structurally ambiguous, because a label and the row
    describing it carry identical poses.
    """

    scene_id: str
    instance_id: int
    position_mm: tuple[float, ...]
    approach: tuple[float, ...]
    closing_axis: tuple[float, ...]
    width_mm: float
    confidence: float
    seed_index: int
    slot: int
    rank: int
    #: How far the proposed centre is from the nearest point of any object, and it separates two
    #: failures a bare hold rate merges. A grasp that closed on an object and slipped is a grasp that
    #: failed. A grasp 200 mm from every object never had a chance to fail: the model aimed at
    #: nothing. A proposal can name a non-object point as its target, and the referee refuses each
    #: such trial as "instance -1 is not in this scene".
    object_distance_mm: float = 0.0


    # There is no `as_trial` here, and a layering guard is why. Building a trial would import
    # `datagen.grasps.physics`, and `src/**` may never import `datagen`: the dependency runs one
    # way.
    #
    # `src` writes proposals as JSON and datagen reads them (`trials_from_proposals`), so neither
    # package imports the other and the file on disk is the whole contract. A shared object would
    # couple a training run to a simulator's dataclass.


def serving_shares(trained: Any) -> Any:
    """The seed mixture to serve with, derived from the one the artifact was trained with.

    The labelled share cannot exist at serve time, and where it goes decides everything. Sending it
    to `predicted` collapses the draw: the graspability field is nearly flat, so its top-k is one
    small blob whose points carry identical features, and a head handed one input returns one
    answer.

    Redistributing it over the shares that do exist, in the proportion they were trained at, keeps
    serve-time seeding as close to train-time seeding as the missing labels allow. A head fitted at
    predicted:random of 1:1 is served at 1:1.

    A model trained with neither falls back to an even split rather than to nothing. That
    combination means the head only ever saw labelled seeds, so no mixture is faithful, and refusing
    would strand an artifact over a choice that has no right answer.
    """
    from src.robot.grasping.deep.net.set_targets import SeedShares  # noqa: PLC0415

    predicted = float(getattr(trained, "predicted", 0.0))
    random = float(getattr(trained, "random", 0.0))
    total = predicted + random
    if total <= 0.0:
        return SeedShares(labelled=0.0, predicted=0.5, random=0.5)
    return SeedShares(labelled=0.0, predicted=predicted / total, random=random / total)


def suppress(proposals: Sequence["Proposal"], *, radius_mm: float,
             angle_deg: float = _DIAGNOSTIC_DEG) -> list["Proposal"]:
    """Keep a proposal only if no higher-ranked one is within `radius_mm` and `angle_deg` of it.

    A referee needs this. A run can produce eight grasps per scene whose approach vectors agree to
    six decimal places, at positions a few millimetres apart; a referee handed that list judges one
    grasp eight times and reports it as eight trials, so the success rate it prints is a statement
    about one pose wearing the sample size of eight. The underlying cause is in the corpus, where
    candidates concentrate on a single approach vector.

    Greedy, in confidence order. The head's ranking is the product; suppression may only remove
    what the head ranked lower, never reorder what it kept.

    The angle is not optional. Two poses 5 mm apart approaching from opposite sides are two grasps
    and a radius alone would merge them, which flatters the diversity number by throwing away the
    one kind of variety the head is asked for.
    """
    kept: list[Proposal] = []
    limit = np.cos(np.radians(angle_deg))
    for proposal in proposals:
        position = np.asarray(proposal.position_mm, dtype=np.float64)
        approach = np.asarray(proposal.approach, dtype=np.float64)
        duplicate = False
        for other in kept:
            if other.scene_id != proposal.scene_id:
                continue
            near = float(np.linalg.norm(position - np.asarray(other.position_mm))) <= radius_mm
            aligned = float(approach @ np.asarray(other.approach)) >= limit
            if near and aligned:
                duplicate = True
                break
        if not duplicate:
            kept.append(proposal)
    return kept


def distinct_share(proposals: Sequence["Proposal"]) -> float:
    """What fraction of a proposal list survives the diagnostic radius. 1.0 means all distinct.

    Reported, never applied. A number a run prints about itself costs nothing and cannot be
    forgotten; a filter that ran by default would change every rate taken before it.
    """
    if not proposals:
        return 0.0
    return len(suppress(proposals, radius_mm=_DIAGNOSTIC_MM)) / len(proposals)


def propose_for_scenes(artifact: str | Path, files: Sequence[str | Path], *,
                       top: int = 8, seeds: int = 64, points: int = 8192,
                       device: str | None = None, seed: int = 0, spread_mm: float = 0.0,
                       report: Any = None) -> list[Proposal]:
    """Run a trained generator over scenes and return its ranked grasps.

    `top` is a prefix of the confidence order, per scene, because that prefix is what a cell would
    actually try. Everything below it is what the head itself ranked lower.
    """
    import torch  # noqa: PLC0415 (heavy, and only this path needs it)

    from src.robot.grasping.deep.corpus.sample import build_sample, load_scene  # noqa: PLC0415
    from src.robot.grasping.deep.net.gripper import (  # noqa: PLC0415
        JAW_GEOMETRY, gripper_vector)
    from src.robot.grasping.deep.net.set_targets import (  # noqa: PLC0415
        sample_seeds)
    from src.robot.grasping.deep.set_artifact import load_set_generator  # noqa: PLC0415
    from src.robot.grasping.deep.set_decode import decode_set_prediction  # noqa: PLC0415

    say = report if report is not None else (lambda _line: None)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    loaded = load_set_generator(artifact, device=device)
    net = loaded.net.eval()
    shares = serving_shares(loaded.step.shares)
    # `load_set_generator` strips `shares` out of the payload on the way in
    # (`if k not in ("shares", "weights")`), so `loaded.step.shares` is whatever `SetStepConfig`
    # defaults to, never what the run trained under. The two coincide only while nothing can set a
    # mixture from the CLI; once a mixture flag exists, this line must not claim the artifact's own
    # mixture.
    say(f"  seeds drawn at predicted {shares.predicted:.2f} / random {shares.random:.2f}, "
        f"from the SERVING defaults (the artifact's own mixture is not carried through the reader). "
        f"Serving all-predicted collapses the draw onto one blob of identical features")
    gripper = gripper_vector(loaded.gripper).to(device)
    # The augmentation is off here. `SampleSpec.rotate_z` defaults to true because it is a training
    # device: `build_sample` in `corpus/sample.py` draws `rng.uniform(0, 2*pi)` and rotates the
    # cloud, the normals and the grasp targets about the cloud centroid. At training time that is
    # free, because the labels rotate with the scene. At propose time nothing rotates back, so a
    # pose would be emitted in a frame the scene never had, while `_nearest_instance` below and the
    # physics referee both judge it against the unrotated cloud.
    #
    # Left on, each scene carries its own angle, so the proposals land off their objects and a
    # referee figure taken from such a run measures that defect and not the model.
    #
    # `calculator.py` overrides `rotate_z` the same way, so the judgment stage and the deployment
    # stage measure one system.
    spec = dataclasses.replace(loaded.sample, rotate_z=False)
    rng = np.random.default_rng(seed)
    generator = torch.Generator().manual_seed(seed)

    out: list[Proposal] = []
    for index, path in enumerate(files, start=1):
        scene = load_scene(path)
        sample = build_sample(scene, rng, spec, target_instance=None)
        cloud = torch.as_tensor(np.asarray(sample["points_m"])[None], dtype=torch.float32,
                                device=device)
        features = torch.as_tensor(np.asarray(sample["features"])[None], dtype=torch.float32,
                                   device=device)
        features = features[..., :net.backbone.config.in_features]
        with torch.no_grad():
            encoded = net.encode(cloud, features)
            field = net.graspability_logit(encoded)[0]
            # The seed mixture is the artifact's own. Pointing it all at the predicted field
            # collapses the draw.
            #
            # `labelled` stays empty: seeding from the answer key would grade the head on poses it
            # was handed, which is the mimicry problem one level up from filtering the proposals.
            # The freed share goes to `predicted` and `random` instead. Same model, same cloud,
            # same encoder: with the whole share on `predicted` the seeds land in one tight blob of
            # near-identical features, where a random draw spreads them across the scene. The
            # graspability field is nearly flat, so "the best 64 points" is an arbitrary small blob,
            # every one of its points carries the same feature direction, and a head handed one
            # input returns one answer.
            #
            # A head fitted under `labelled=0.5, predicted=0.25, random=0.25` and served
            # `predicted=1.0` is train/serve skew in the seeding, the same rule this file states for
            # the sample spec: serving a net a different sample than it was fitted to reads exactly
            # like a bad model. The labelled half cannot exist at serve time, so it is redistributed
            # over the two that can, in the proportion they were trained at.
            candidate = torch.as_tensor(np.asarray(sample["supervise"]), dtype=torch.bool)
            if not bool(candidate.any()):
                candidate = torch.ones(cloud.shape[1], dtype=torch.bool)
            draw = sample_seeds(count=seeds,
                                labelled=torch.zeros(cloud.shape[1], dtype=torch.bool),
                                score=field.detach().cpu(), candidate=candidate,
                                shares=serving_shares(loaded.step.shares),
                                generator=generator)
            picks = draw.point_index.to(device)
            # The cloud goes through, because a stage-3 net crops it. `propose` refuses
            # rather than silently skipping the crop, so this is what keeps the flag live
            # here as well as in training.
            prediction = net.propose(encoded, torch.zeros_like(picks), picks,
                                     gripper.expand(len(picks), gripper.shape[-1]),
                                     cloud)
        # The decode is a CPU function and every other argument here is already on the CPU. It
        # mixes the prediction with the cloud, so one tensor left on the GPU is not a slow path, it
        # is a RuntimeError on the first scene. Moved here rather than inside the decoder, which has
        # no business knowing that a caller ran on a GPU.
        prediction = replace(prediction, **{
            field.name: getattr(prediction, field.name).cpu()
            for field in fields(prediction)
            if isinstance(getattr(prediction, field.name), torch.Tensor)})
        grasps = decode_set_prediction(
            prediction, picks.cpu(), cloud[0].cpu(),
            # The frame the sample was built in, not a zero. `points_m` has its xy mean removed
            # and the support height taken off z, so decoding without those puts every proposal in
            # a frame the scene never had: an offset wrong by a per-scene constant, which is the
            # shape a head can almost learn around.
            centre_xy_mm=np.asarray(sample["centre_xy_mm"], dtype=np.float64),
            support_height_mm=float(np.asarray(sample["support_height_mm"]).reshape(-1)[0]),
            min_width_mm=1.0,
            max_width_mm=float(JAW_GEOMETRY[loaded.gripper]["aperture_mm"]),
            # From the artifact's own head, so a role is named by the net that was trained.
            part_roles=tuple(net.config.head.part_roles))
        scene_id = str(np.asarray(scene.get("scene_id", Path(path).stem)).reshape(-1)[0])
        # Suppressed before the prefix is taken, not after. Taking the top 8 and then removing
        # duplicates leaves however many happen to survive, which is a different number per scene and
        # therefore not a prefix of anything. Suppress first and the prefix is the head's best `top`
        # distinct grasps, which is what a cell would actually try.
        if spread_mm > 0.0:
            grasps = _spread(grasps, radius_mm=spread_mm, scene_id=scene_id)
        for rank, grasp in enumerate(grasps[:top]):
            instance, distance = _nearest_instance(scene, grasp.position_mm)
            out.append(Proposal(
                scene_id=scene_id, instance_id=instance, object_distance_mm=distance,
                position_mm=tuple(float(v) for v in grasp.position_mm),
                approach=tuple(float(v) for v in grasp.approach),
                closing_axis=tuple(float(v) for v in grasp.axis),
                width_mm=float(grasp.width_mm), confidence=float(grasp.confidence),
                seed_index=int(grasp.seed_index), slot=int(grasp.slot), rank=len(out)))
        if index % 25 == 0:
            say(f"  {index}/{len(files)} scene(s), {len(out)} proposal(s)")
    off = [p for p in out if not np.isfinite(p.object_distance_mm)
           or p.object_distance_mm > _OFF_OBJECT_MM]
    if off:
        say(f"  {len(off)} of {len(out)} proposal(s) sit more than {_OFF_OBJECT_MM:.0f} mm from any "
            f"object ({100.0 * len(off) / max(1, len(out)):.1f} %). Those did not fail, they aimed "
            f"at nothing, and a hold rate that mixes the two answers neither question")
    share = distinct_share(out)
    say(f"  distinct at {_DIAGNOSTIC_MM:.0f} mm / {_DIAGNOSTIC_DEG:.0f} deg: {share * 100:.1f} % "
        f"({len(out)} proposal(s)). A low number means the referee below would judge one grasp "
        f"several times and report it as several trials")
    return out


def _spread(grasps: list[Any], *, radius_mm: float, scene_id: str) -> list[Any]:
    """`suppress` over decoded grasps, before they become proposals. Same rule, same order."""
    kept: list[Any] = []
    limit = np.cos(np.radians(_DIAGNOSTIC_DEG))
    del scene_id                      # one scene at a time here; the field exists on Proposal only
    for grasp in grasps:
        position = np.asarray(grasp.position_mm, dtype=np.float64)
        approach = np.asarray(grasp.approach, dtype=np.float64)
        if any(float(np.linalg.norm(position - np.asarray(other.position_mm))) <= radius_mm
               and float(approach @ np.asarray(other.approach)) >= limit for other in kept):
            continue
        kept.append(grasp)
    return kept


def _nearest_instance(scene: dict, position_mm: np.ndarray) -> tuple[int, float]:
    """Which object a proposal is aimed at, and how far its centre is from that object.

    Object points only. A cloud carries the table and the bin walls as instance ids below zero, and
    the nearest point to a proposal is very often one of them: such a proposal comes back as
    instance -1 or -2 and is refused downstream as "instance -1 is not in this scene". An unjudged
    trial leaves a metric answering a different question, on a denominator chosen by a defect.

    The target is inferred, not known. The head proposes a pose in the scene, not a pose on an
    object, so this returns the nearest object and the distance to it. The distance is the honest
    part: a grasp 200 mm from every object did not fail, it aimed at nothing, and the two must not
    share a number.
    """
    # The absence is checked before the cast, not after. `np.asarray(None, dtype=np.int64)` raises
    # a TypeError about `int()` rather than saying that a cloud has no instance channel, and a
    # foreign corpus is exactly the case that arrives without one.
    if scene.get("points_mm") is None or scene.get("instance_id") is None:
        return 0, float("nan")
    points = np.asarray(scene.get("points_mm"), dtype=np.float64)
    instances = np.asarray(scene.get("instance_id"), dtype=np.int64)
    if not len(points) or len(instances) != len(points):
        return 0, float("nan")
    on_object = instances >= 0
    if not bool(on_object.any()):
        return 0, float("nan")
    distances = np.linalg.norm(points[on_object] - position_mm, axis=1)
    best = int(np.argmin(distances))
    return int(instances[on_object][best]), float(distances[best])


def write_proposals(path: str | Path, proposals: Sequence[Proposal]) -> dict[str, Any]:
    """One JSON line per proposal, in the confidence order they were ranked in."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for proposal in proposals:
            handle.write(json.dumps({
                "scene_id": proposal.scene_id, "instance_id": proposal.instance_id,
                "position_mm": list(proposal.position_mm), "approach": list(proposal.approach),
                "closing_axis": list(proposal.closing_axis), "width_mm": proposal.width_mm,
                "confidence": proposal.confidence, "seed_index": proposal.seed_index,
                "slot": proposal.slot, "rank": proposal.rank,
                "object_distance_mm": proposal.object_distance_mm}) + "\n")
    scenes = len({p.scene_id for p in proposals})
    on_object = sum(1 for p in proposals
                    if np.isfinite(p.object_distance_mm)
                    and p.object_distance_mm <= _OFF_OBJECT_MM)
    return {"proposals": len(proposals), "scenes": scenes, "path": str(target),
            "on_object": on_object, "distinct_share": round(distinct_share(proposals), 4)}
