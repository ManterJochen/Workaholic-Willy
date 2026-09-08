"""Which detection in the other cameras is the same object?

The measurement that motivates this module: feeding the grasp generator a target cloud fused from three
views instead of one is worth 8.4 percentage points of coverage on the datagen reference, end to end
with real GroundingDINO and SAM2 masks. That is about three times the best single knob, with fewer
candidates per object and half the position error. The reason is structural rather than incidental. A
candidate is accepted or rejected on its closing axis, the closing axis comes from the silhouette, and
one depth view only ever sees one side of an object while an antipodal grasp needs two.

Everything else for that fusion exists already. ``multiview/localize.py`` fuses per-camera centroids,
``multiview/synthesis.py`` turns a cloud into a grasp, and ``pick_loop`` forwards
``external_target_geometry_base_mm`` into the ``geometry_points_base_mm`` seam of the calculator. The
one place that fills the seam otherwise is a sim runner using ground-truth perception in every camera,
``run_pile_baseline.py`` through ``MultiObjectGroundTruthPerceptionSource``. Ground truth knows which
blob in camera B is the blob camera A is looking at. A real cell does not.

So this module answers exactly that and nothing else: given the target's points from one camera and the
candidate objects another camera segmented, decide which candidate is the same physical object, or
decide that none of them is.

There are two shapes of that question, because a real cell asks both. :func:`associate_target` answers
it for one labelled target and scores each view independently. :func:`fuse_scene_clouds` answers it for
every object in the scene at once, under the extra constraint that no candidate may be handed to two
primaries; :func:`assign_view` sets out why that constraint is not optional.

The module is fail-closed rather than best-effort. A view whose best candidate does not clear the
threshold contributes nothing, exactly as an occluded camera does in
:mod:`~src.robot.grasping.multiview.localize`. Fusing the wrong object is worse than fusing one fewer
view: it invents a contact face on the far side of a neighbour, and the generator has no way to tell
that cloud from a real one.

Pure and deterministic: numpy plus the deterministic ``linear_sum_assignment`` of scipy for the
whole-scene one-to-one constraint, input order preserved, no set iteration, no vendor SDKs. Frames are
BASE millimetres throughout, per the repo convention, and every input is a point cloud rather than a
mask, so the module is independent of how any particular camera produced its segmentation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from scipy.optimize import linear_sum_assignment

__all__ = [
    "AssociationMetric",
    "AssociationResult",
    "SceneAssociation",
    "SceneCluster",
    "ViewCandidates",
    "assign_view",
    "associate_target",
    "cluster_views",
    "fuse_scene_clouds",
    "fuse_target_cloud",
]

#: Default gate on the match score, measured over 68 scenes with 1240 answerable decisions and 488
#: where the target was not in the other view at all, which is 28 % of all pairs and the normal case
#: for a real cell rather than an edge case. At 0.30 the OVERLAP metric gets 92.0 % of the answerable
#: decisions right with 0.08 % wrong, and correctly refuses 98.2 % of the impossible ones.
#:
#: That sweep is why the default is OVERLAP and not the cheaper CENTROID. Scored on the answerable half
#: alone, CENTROID looks like the winner at 99.5 % correct, but it never abstains anywhere in the
#: useful range, so when the target is absent it takes a neighbour instead in up to 73 % of cases, and
#: in 50 % even at this threshold. That fabricates a neighbour's far surface into roughly a fifth of
#: all fused clouds, and the generator cannot tell that cloud from a real one. A metric measured on the
#: answerable half alone looks best exactly where it is most dangerous.
DEFAULT_MIN_SCORE = 0.30
#: How near two points must be to count as the same surface, in mm. Larger than the depth noise of a
#: stereo camera at working distance and smaller than the gap between two touching objects.
DEFAULT_NEIGHBOUR_MM = 12.0


class AssociationMetric(StrEnum):
    """How "the same object" is scored. Which one to use is a measurement, not a preference."""

    #: 1 - (centroid distance / max_centroid_mm), clipped. The cheapest metric, and it degrades in a
    #: dense pile where neighbouring centroids sit closer together than the localisation error.
    CENTROID = "centroid"
    #: Intersection over union of the two axis-aligned BASE bounding boxes. It uses extent as well as
    #: position, so a small object nested against a large one stays separable.
    BOX_IOU = "box_iou"
    #: Fraction of the target's points that have a candidate point within ``neighbour_mm``. It is the
    #: only metric that uses the surfaces themselves, and it costs a nearest-neighbour query per
    #: candidate.
    OVERLAP = "overlap"


@dataclass(frozen=True, slots=True)
class ViewCandidates:
    """One other camera's segmented objects, already back-projected into BASE millimetres.

    ``clouds_base_mm`` is one ``(N, 3)`` array per detected object, in that camera's own detection
    order. A camera that segmented nothing contributes an empty sequence and is skipped.
    """

    name: str
    clouds_base_mm: tuple[np.ndarray, ...]


@dataclass(frozen=True, slots=True)
class AssociationResult:
    """Which candidate was chosen in one view, and the evidence for it.

    ``index`` is ``None`` when nothing cleared the threshold. ``scores`` is kept whatever the outcome,
    because a refusal with its runner-up score is diagnosable where a bare ``None`` is not.
    """

    view: str
    index: int | None
    score: float
    runner_up: float
    scores: tuple[float, ...]

    @property
    def matched(self) -> bool:
        return self.index is not None

    @property
    def margin(self) -> float:
        """How much better the winner was than the next candidate. Small means ambiguous, not wrong."""
        return self.score - self.runner_up


def _centroid(cloud: np.ndarray) -> np.ndarray:
    return np.asarray(cloud, dtype=np.float64).reshape(-1, 3).mean(axis=0)


def _bounds(cloud: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(cloud, dtype=np.float64).reshape(-1, 3)
    return points.min(axis=0), points.max(axis=0)


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    lo_a, hi_a = _bounds(a)
    lo_b, hi_b = _bounds(b)
    overlap = np.maximum(0.0, np.minimum(hi_a, hi_b) - np.maximum(lo_a, lo_b))
    inter = float(np.prod(overlap))
    if inter <= 0.0:
        return 0.0
    vol_a = float(np.prod(np.maximum(hi_a - lo_a, 1e-6)))
    vol_b = float(np.prod(np.maximum(hi_b - lo_b, 1e-6)))
    return inter / (vol_a + vol_b - inter)


def _voxel_decimate(points: np.ndarray, voxel_mm: float) -> np.ndarray:
    """One representative point per ``voxel_mm`` cube. ``voxel_mm <= 0`` returns the input unchanged.

    For scoring only, and that distinction is the whole point. The association decides which object in
    another view is the same object, and the fused cloud handed to the grasp generator is then rebuilt
    from the original, full-resolution clouds. This makes the decision cheaper without making the
    geometry coarser; a downsample applied to the output instead would quietly degrade every grasp.

    It exists because ``_overlap_fraction`` below states a precondition it cannot always hold to, that
    the clouds are a few thousand points at most. Against the path-traced renders of datagen one
    object's cloud is 37,119 points, an order of magnitude past that, at which the brute-force pass
    becomes about 1.4e9 distance evaluations and one scene's fusion takes 182.7 s.

    Off by default everywhere, at ``0.0``, so no configured cell changes behaviour by a single bit.
    """
    if voxel_mm <= 0.0:
        return points
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        return pts
    keys = np.floor(pts / float(voxel_mm)).astype(np.int64)
    # `return_index` on the unique rows keeps one real point per cell rather than a cell centroid.
    # A centroid is a point that was never observed, and the overlap metric is a statement about
    # observed surface.
    _unique, first = np.unique(keys, axis=0, return_index=True)
    return pts[np.sort(first)]


def _overlap_fraction(target: np.ndarray, candidate: np.ndarray, *, neighbour_mm: float) -> float:
    """Share of target points with a candidate point within ``neighbour_mm``.

    Chunked brute force rather than a KD-tree: the clouds are a few thousand points at most, this layer
    stays numpy-only, and a tree would add a dependency for no measured gain. Chunking bounds the
    temporary at about 8 MB whatever the cloud size, which a plain outer difference does not.
    """
    a = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    b = np.asarray(candidate, dtype=np.float64).reshape(-1, 3)
    if a.size == 0 or b.size == 0:
        return 0.0
    limit = float(neighbour_mm) ** 2
    hits = 0
    chunk = max(1, int(1e6 // max(1, b.shape[0])))
    for start in range(0, a.shape[0], chunk):
        block = a[start:start + chunk]
        d2 = ((block[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
        hits += int((d2.min(axis=1) <= limit).sum())
    return hits / a.shape[0]


def _score(
    target: np.ndarray, candidate: np.ndarray, *, metric: AssociationMetric,
    max_centroid_mm: float, neighbour_mm: float,
) -> float:
    if metric is AssociationMetric.CENTROID:
        distance = float(np.linalg.norm(_centroid(target) - _centroid(candidate)))
        return float(np.clip(1.0 - distance / max(1e-6, max_centroid_mm), 0.0, 1.0))
    if metric is AssociationMetric.BOX_IOU:
        return _box_iou(target, candidate)
    return _overlap_fraction(target, candidate, neighbour_mm=neighbour_mm)


def associate_target(
    target_cloud_base_mm: np.ndarray,
    views: Sequence[ViewCandidates],
    *,
    metric: AssociationMetric = AssociationMetric.OVERLAP,
    min_score: float = DEFAULT_MIN_SCORE,
    max_centroid_mm: float = 150.0,
    neighbour_mm: float = DEFAULT_NEIGHBOUR_MM,
) -> tuple[AssociationResult, ...]:
    """For each view, which of its segmented objects is the target, or none of them.

    One result per view, in input order, whether or not it matched. A caller that wants only the
    matches filters on :attr:`AssociationResult.matched`, and a caller that wants to log why a view was
    dropped has the score and the runner-up.
    """
    target = np.asarray(target_cloud_base_mm, dtype=np.float64).reshape(-1, 3)
    results: list[AssociationResult] = []
    for view in views:
        scores = tuple(
            _score(target, cloud, metric=metric, max_centroid_mm=max_centroid_mm,
                   neighbour_mm=neighbour_mm)
            if np.asarray(cloud).size else 0.0
            for cloud in view.clouds_base_mm
        )
        if not scores or target.size == 0:
            results.append(AssociationResult(view.name, None, 0.0, 0.0, scores))
            continue
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))   # a tie goes to the first
        best = order[0]
        runner_up = scores[order[1]] if len(order) > 1 else 0.0
        matched = best if scores[best] >= min_score else None
        results.append(AssociationResult(view.name, matched, scores[best], runner_up, scores))
    return tuple(results)


def fuse_target_cloud(
    target_cloud_base_mm: np.ndarray,
    views: Sequence[ViewCandidates],
    *,
    metric: AssociationMetric = AssociationMetric.OVERLAP,
    min_score: float = DEFAULT_MIN_SCORE,
    max_centroid_mm: float = 150.0,
    neighbour_mm: float = DEFAULT_NEIGHBOUR_MM,
) -> tuple[np.ndarray, tuple[AssociationResult, ...]]:
    """The target's surface as every camera that can identify it sees it, in BASE millimetres.

    Returns the fused cloud and the per-view evidence. The primary cloud is always first, so a caller
    that ends up with no matches gets exactly the single-view cloud it started with, unchanged when no
    other camera can confirm the target.
    """
    results = associate_target(
        target_cloud_base_mm, views, metric=metric, min_score=min_score,
        max_centroid_mm=max_centroid_mm, neighbour_mm=neighbour_mm)
    parts = [np.asarray(target_cloud_base_mm, dtype=np.float64).reshape(-1, 3)]
    for view, result in zip(views, results, strict=True):
        if result.index is not None:
            cloud = np.asarray(view.clouds_base_mm[result.index], dtype=np.float64).reshape(-1, 3)
            if cloud.size:
                parts.append(cloud)
    return np.vstack(parts), results


# ---------------------------------------------------------------------------
# Whole-scene association: every candidate object, not one labelled target
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SceneAssociation:
    """One other camera's answer for every primary object at once.

    ``assignment[i]`` is the index into ``ViewCandidates.clouds_base_mm`` that belongs to primary
    object ``i``, or :data:`None` when this camera cannot identify it. ``scores[i][j]`` is kept in full,
    because a refusal is only diagnosable next to what it refused.
    """

    view: str
    assignment: tuple[int | None, ...]
    scores: tuple[tuple[float, ...], ...]

    @property
    def matched_count(self) -> int:
        return sum(1 for index in self.assignment if index is not None)


def assign_view(
    primary_clouds_base_mm: Sequence[np.ndarray],
    view: ViewCandidates,
    *,
    metric: AssociationMetric = AssociationMetric.OVERLAP,
    min_score: float = DEFAULT_MIN_SCORE,
    max_centroid_mm: float = 150.0,
    neighbour_mm: float = DEFAULT_NEIGHBOUR_MM,
    score_voxel_mm: float = 0.0,
) -> SceneAssociation:
    """Match every primary object to at most one object in this view, and vice versa.

    One-to-one rather than each object taking its own best match. Scoring each primary independently is
    what :func:`associate_target` does, and it is correct when there is exactly one target. Run per
    object over a whole scene, it hands the same candidate to two primaries, and the loser of that pair
    gets a neighbour's far surface welded onto its cloud. That is the failure the metric sweep singled
    out as the dangerous one, in :data:`DEFAULT_MIN_SCORE`: the generator cannot tell a fabricated
    contact face from a real one, so the result is not a slightly worse cloud, it is a confidently
    wrong grasp.

    The assignment therefore maximises the total score subject to each candidate being used at most
    once. Sub-threshold pairs are zeroed before the optimisation rather than filtered after it, so they
    can never be selected merely to fill a slot: any positive alternative beats them, and a primary
    with no admissible partner is left unmatched instead of being given the least bad one.

    Measured against the per-object alternative on 68 datagen scenes with real GroundingDINO and SAM2
    masks, over 1240 answerable decisions and 461 where the object was not in the other view at all::

                   correct    wrong  abstain     held  fabricated  double-assigned
        per-object  91.7 %    0.0 %    8.3 %   98.3 %       1.7 %                6
        one-to-one  91.5 %    0.0 %    8.5 %   99.1 %       0.9 %                0

    The double-assignments go to zero by construction, and the rate at which a camera invents a match
    for an object it cannot see nearly halves. It is not free: 0.2 percentage points of correct matches
    are given up. They become abstentions rather than errors, so the object gets one fewer view, which
    is the same outcome as an occluded camera and the trade this module is built to make.

    Deterministic: the score matrix is built in input order and
    :func:`scipy.optimize.linear_sum_assignment` is a deterministic solver.
    """

    primaries = [np.asarray(cloud, dtype=np.float64).reshape(-1, 3) for cloud in primary_clouds_base_mm]
    candidates = [np.asarray(cloud, dtype=np.float64).reshape(-1, 3) for cloud in view.clouds_base_mm]
    # Scoring-only decimation, once per cloud rather than once per pair, because the loop below is
    # |primaries| x |candidates| and decimating inside it would repeat the same work N times over.
    #
    # Restricted to OVERLAP because that is the only O(N*M) metric. Centroid and box-IoU are cheap and
    # must not shift by a voxel just because this is enabled. `assignment` below indexes the original
    # candidate list and `fuse_scene_clouds` rebuilds the fused cloud from the originals, so this
    # changes which objects are matched to each other and nothing about the geometry that reaches the
    # grasp generator.
    if score_voxel_mm > 0.0 and metric is AssociationMetric.OVERLAP:
        primaries = [_voxel_decimate(cloud, score_voxel_mm) for cloud in primaries]
        candidates = [_voxel_decimate(cloud, score_voxel_mm) for cloud in candidates]
    scores = tuple(
        tuple(
            _score(primary, candidate, metric=metric, max_centroid_mm=max_centroid_mm,
                   neighbour_mm=neighbour_mm)
            if primary.size and candidate.size else 0.0
            for candidate in candidates
        )
        for primary in primaries
    )
    assignment: list[int | None] = [None] * len(primaries)
    if primaries and candidates:
        matrix = np.asarray(scores, dtype=np.float64)
        # Zeroed, not filtered afterwards: an inadmissible pair must never be worth taking.
        admissible = np.where(matrix >= min_score, matrix, 0.0)
        rows, columns = linear_sum_assignment(admissible, maximize=True)
        for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
            if admissible[row, column] > 0.0:
                assignment[row] = int(column)
    return SceneAssociation(view.name, tuple(assignment), scores)


def fuse_scene_clouds(
    primary_clouds_base_mm: Sequence[np.ndarray],
    views: Sequence[ViewCandidates],
    *,
    metric: AssociationMetric = AssociationMetric.OVERLAP,
    min_score: float = DEFAULT_MIN_SCORE,
    max_centroid_mm: float = 150.0,
    neighbour_mm: float = DEFAULT_NEIGHBOUR_MM,
    score_voxel_mm: float = 0.0,
) -> tuple[tuple[np.ndarray, ...], tuple[SceneAssociation, ...]]:
    """Every primary object's surface as every camera that can identify it sees it.

    Returns one fused cloud per primary object, in input order, plus the per-view evidence. Each
    object's own cloud is always first, so an object no other camera confirms comes back exactly as it
    went in: the single-view path is unchanged rather than degraded.
    """

    associations = tuple(
        assign_view(primary_clouds_base_mm, view, metric=metric, min_score=min_score,
                    score_voxel_mm=score_voxel_mm,
                    max_centroid_mm=max_centroid_mm, neighbour_mm=neighbour_mm)
        for view in views
    )
    fused: list[np.ndarray] = []
    for index, primary in enumerate(primary_clouds_base_mm):
        parts = [np.asarray(primary, dtype=np.float64).reshape(-1, 3)]
        for view, association in zip(views, associations, strict=True):
            matched = association.assignment[index]
            if matched is None:
                continue
            cloud = np.asarray(view.clouds_base_mm[matched], dtype=np.float64).reshape(-1, 3)
            if cloud.size:
                parts.append(cloud)
        fused.append(np.vstack(parts))
    return tuple(fused), associations


# --------------------------------------------------------------------------------------------------
# Every camera equal: grouping without a privileged view.
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SceneCluster:
    """One physical object, and every blob any camera saw of it.

    ``members`` are ``(view_index, blob_index)`` pairs into the sequence of views this was built
    from, in view order and then blob order, so the result is deterministic and a caller can name
    which camera contributed what.

    A cluster of one member is an object exactly one camera saw. That is not a failure: it is what a
    part behind another part looks like from every angle but one, and it is the case the whole
    symmetric pass exists to keep.
    """

    members: tuple[tuple[int, int], ...]
    #: The mean of the scores between every pair inside the cluster, or 1.0 for a cluster of one.
    #: Not the minimum: the minimum is already known to clear the threshold, and the mean is what
    #: says whether the cameras merely agreed or agreed strongly.
    agreement: float

    @property
    def views(self) -> tuple[int, ...]:
        return tuple(dict.fromkeys(view for view, _ in self.members))


def cluster_views(
    views: Sequence[ViewCandidates],
    *,
    metric: AssociationMetric = AssociationMetric.OVERLAP,
    min_score: float = DEFAULT_MIN_SCORE,
    neighbour_mm: float = DEFAULT_NEIGHBOUR_MM,
    max_centroid_mm: float = 150.0,
    score_voxel_mm: float = 0.0,
) -> tuple[SceneCluster, ...]:
    """Group every camera's blobs into objects, with no camera privileged over the others.

    Complete linkage, and the choice is the whole point of this function. A blob joins a group only
    if it clears ``min_score`` against every blob already in it, never merely against one of them.

    The case that separates the two rules is three blobs where A matches B and B matches C and A does
    not match C, which a threshold produces routinely on objects that touch. Single linkage, which is
    what a union-find over the same edges computes, puts all three in one group: one false edge welds
    three objects and the grasp is then planned on a surface that does not exist, with the closing
    axis running through the gap between two parts. Complete linkage refuses, B goes with whichever
    of A or C it scores higher against, and the third stays its own object.

    That leaves a duplicate rather than a phantom, and the two failures are not equally priced: a
    duplicate costs one attempt that finds nothing, a phantom drives a jaw into the space between two
    objects. The owner chose splitting over welding for that reason and this implements it.

    Blobs are never matched within one camera. Two blobs in one view are two detections that camera
    made, and the segmenter, not this function, is what decides whether they are one object.

    ``score_voxel_mm`` decimates the clouds for scoring only, exactly as :func:`assign_view` does,
    and only for the overlap metric, which is the only one whose cost is quadratic in the points.
    """

    clouds: list[list[np.ndarray]] = [
        [np.asarray(cloud, dtype=np.float64).reshape(-1, 3) for cloud in view.clouds_base_mm]
        for view in views
    ]
    if score_voxel_mm > 0.0 and metric is AssociationMetric.OVERLAP:
        scored = [[_voxel_decimate(cloud, score_voxel_mm) for cloud in view] for view in clouds]
    else:
        scored = clouds

    #: Every blob in the whole rig, as (view, blob). One flat list so a cluster can hold blobs from
    #: any camera without the structure implying an order between cameras.
    blobs: list[tuple[int, int]] = [
        (view_index, blob_index)
        for view_index, view in enumerate(scored)
        for blob_index, cloud in enumerate(view)
        if cloud.size
    ]

    # Pairwise scores, across cameras only. Symmetric by construction: the pair is stored once, under
    # the lower blob first, and read through `_pair_score` from either side.
    pair: dict[tuple[int, int], float] = {}
    for left in range(len(blobs)):
        for right in range(left + 1, len(blobs)):
            if blobs[left][0] == blobs[right][0]:
                continue  # one camera's own two detections are two objects, by that camera's word
            a = scored[blobs[left][0]][blobs[left][1]]
            b = scored[blobs[right][0]][blobs[right][1]]
            score = _score(a, b, metric=metric, max_centroid_mm=max_centroid_mm,
                           neighbour_mm=neighbour_mm)
            if score >= min_score:
                pair[(left, right)] = score

    # Greedy from the strongest edge outward. Deterministic: ties break on the pair's own indices,
    # which are the input order, so the same rig produces the same clusters every time.
    order = sorted(pair.items(), key=lambda item: (-item[1], item[0]))
    group_of: dict[int, int] = {}
    groups: list[list[int]] = []
    for (left, right), _score_value in order:
        left_group, right_group = group_of.get(left), group_of.get(right)
        if left_group is not None and left_group == right_group:
            continue
        if left_group is None and right_group is None:
            group_of[left] = group_of[right] = len(groups)
            groups.append([left, right])
            continue
        if left_group is None or right_group is None:
            joiner, group_index = (left, right_group) if left_group is None else (right, left_group)
            assert group_index is not None
            if _clears_every_member(joiner, groups[group_index], pair):
                group_of[joiner] = group_index
                groups[group_index].append(joiner)
            continue
        # Both already grouped. Merging two groups is admissible only when every cross pair clears,
        # which is the same rule applied to a whole group rather than to one blob.
        if all(_pair_score(a, b, pair) is not None
               for a in groups[left_group] for b in groups[right_group]):
            groups[left_group].extend(groups[right_group])
            for member in groups[right_group]:
                group_of[member] = left_group
            groups[right_group] = []

    clustered = {index for group in groups for index in group}
    singles = [[index] for index in range(len(blobs)) if index not in clustered]

    result: list[SceneCluster] = []
    for group in [g for g in groups if g] + singles:
        members = tuple(sorted(blobs[index] for index in group))
        result.append(SceneCluster(members=members, agreement=_mean_pair_score(group, pair)))
    result.sort(key=lambda cluster: cluster.members)
    return tuple(result)


def _pair_score(a: int, b: int, pair: "dict[tuple[int, int], float]") -> "float | None":
    return pair.get((a, b), pair.get((b, a)))


def _clears_every_member(
    joiner: int, group: "Sequence[int]", pair: "dict[tuple[int, int], float]"
) -> bool:
    """Complete linkage, stated as the one predicate the whole rule reduces to."""
    return all(_pair_score(joiner, member, pair) is not None for member in group)


def _mean_pair_score(group: "Sequence[int]", pair: "dict[tuple[int, int], float]") -> float:
    scores = [
        score
        for i, a in enumerate(group)
        for b in group[i + 1:]
        if (score := _pair_score(a, b, pair)) is not None
    ]
    return float(sum(scores) / len(scores)) if scores else 1.0
