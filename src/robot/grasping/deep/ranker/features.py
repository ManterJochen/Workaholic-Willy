"""What the learned scorer eats, declared once and read by the trainer and the dataset gate.

A gate that infers its columns from a file can only say "this column is constant". It cannot say
"this feature is constant", and that is what a declared list buys: aletheia's ``quality`` column
holds one value on every row, and this repo's ``contact_angle_deg`` does the same; both look
exactly like features and neither carries a bit of information.

The frame is part of the contract. Two corpora can hold different quantities under identical column
names, and nothing in the numbers says so: a closing axis zeroed in the CAMERA frame instead of the
support plane, a table check reading a camera depth as a height, and a 180 deg base error that only
a USD-against-FK cross-check caught. So a spec carries its frame and the gate refuses to compare
two corpora that do not share it.

## The frame this scorer uses, and the two it does not

``grasp`` is the object's centre expressed in the grasp's own frame. Four numbers: the jaw width,
and the offset from the grasp point to the object centre projected onto the closing axis, the
approach, and their binormal. Exact at runtime by construction: the grasp frame is the grasp scored.

``object`` (aletheia's frame) is not recoverable at runtime. The corpus stores every vector rotated
into the target's own frame (`scene_grasps.py` writes ``axis = r_t.T @ rot[:, 0]``) and the scene
pose is not recorded. At runtime there is no object pose, only a segmented cloud from perception, so
the frame would have to come from the cloud's principal axes. Those axes disagree with the frame
the corpus was authored in: the longest one points elsewhere, objects permute their axes, and many
are near-degenerate, meaning no unique frame exists (a cube has no distinguishable axes; a cylinder
none in its barrel plane). Some sign flips are undecidable by any rule reading only the cloud.

The frame is also not where the signal lives. The object frame's contribution is almost entirely
"how big is the object along this direction", which ``width_mm`` already measures, so a perfect
object frame buys little over rotation-invariant scalars, while the grasp frame does better than
either and needs no frame estimation at all. It also holds up under a realistic centre estimate,
because dot products degrade gracefully where an angle of a normalised offset blows up as the
offset shrinks.

## The one runtime dependency: the object centre

A fused cloud never sees the underside, so its centroid is pulled up systematically rather than
noisily. The support plane is known, so taking z half way between table and measured top uses a fact
instead of averaging over a hole. Through the features the correction shows at the top of the
ranked list and barely at all in AUROC, which is why AUROC alone would hide it.

Units are the repo's: millimetres and degrees. A source that ships metres converts at its loader.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Literal

import numpy as np

__all__ = [
    "BOOTSTRAP_JAW_V1",
    "BOOTSTRAP_OBJECT_FRAME_V0",
    "ENVIRONMENT_FEATURES",
    "FULL_FEATURES",
    "FeatureSpec",
    "GRASP_FRAME_FEATURES",
    "HELD_JAW_V1",
    "SPECS",
    "VALID_JAW_V1",
    "WILLY_JAW_WITH_CONTACT_ANGLE",
    "contact_normal_alignment",
    "derive_grasp_frame_columns",
    "environment_features",
    "grasp_frame_offsets",
    "jaw_clearance_mm",
    "object_centre_mm",
    "spec_named",
]


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """The columns one model reads, the column it predicts, and the column a split may not cross."""

    #: Stable identifier. Appears in the gate's report and in a trained model's provenance, so a
    #: model can always be traced back to the exact input contract it was fitted under.
    name: str
    #: Which frame every direction and position in `features` is expressed in. Two corpora in
    #: different frames hold different quantities under the same column names.
    frame: Literal["object", "base", "grasp"]
    #: The columns the model eats, in canonical names. A source file's own column names are mapped
    #: onto these by its loader; anything derived from them is built by `derive_grasp_frame_columns`.
    features: tuple[str, ...]
    #: The column being predicted.
    label: str
    #: The column a train/validation split must not cross. Rows of one object are not independent
    #: samples of each other: one object contributes many rows to a corpus.
    group: str
    #: True when `features` are computed rather than read. The gate derives them before judging, and a
    #: spec that lied about this would be judged on columns the model never sees.
    derived: bool = False

    def __post_init__(self) -> None:
        if not self.features:
            raise ValueError(f"spec {self.name!r} declares no features")
        clashes = {self.label, self.group} & set(self.features)
        if clashes:
            # A label inside the feature vector is the oldest leak there is, and it produces a model
            # that scores perfectly and predicts nothing.
            raise ValueError(
                f"spec {self.name!r} lists {sorted(clashes)} as both a feature and the label/group"
            )
        if len(set(self.features)) != len(self.features):
            raise ValueError(f"spec {self.name!r} repeats a feature name")


#: The four grasp-frame columns. Order is part of the contract: a trained artifact indexes into it.
GRASP_FRAME_FEATURES: Final[tuple[str, ...]] = (
    "width_mm",
    "centre_along_axis_mm",
    "centre_along_approach_mm",
    "centre_along_binormal_mm",
)

#: The channel the grasp-frame four cannot see: they know only the jaw width and where the object's
#: centre is. On this repo's candidate population `finger_collision` dominates the rejections,
#: ahead of `not_antipodal`, `too_wide` and `below_table`.
#:
#: On aletheia's corpus that is invisible by construction: its candidates carry `collision_free`, so
#: a ranker fitted there is never asked the question this candidate population is dominated by.
#:
#: Every one of these is estimated from the point cloud, never read from the reference. `verdict.py`
#: computes the contact geometry exactly, so a feature copied from it trains a model to reproduce
#: the referee, which looks perfect on every scene and has learnt an estimator, not the world.
#: `contact_normal_alignment` is the one most exposed to that and says so at its own definition.
ENVIRONMENT_FEATURES: Final[tuple[str, ...]] = (
    "jaw_clearance_mm",
    "grasp_height_mm",
    "approach_tilt_deg",
    "contact_normal_alignment",
)

#: `target_observedness` is named here because it is not computable at runtime and must stay out of
#: every spec.
#:
#: In the corpus it comes from the renderer's `visibility`, the ratio of visible to unoccluded
#: pixels, which needs ground truth. A cell has no unoccluded count, so at serving time the feature
#: would be constant while it varied in training: the train/serve skew the dataset gate refuses.
#:
#: Leaving it out costs nothing: an ablation measures it as the weakest column, and removing it
#: does not move the score. A runtime-computable proxy exists (what fraction of the fused target
#: cloud this view contributed) but it is a different quantity and may not carry this name.
_REMOVED_NOT_COMPUTABLE_AT_RUNTIME: Final[tuple[str, ...]] = ("target_observedness",)

#: What the second-stage scorer reads: the grasp, and what is around it.
FULL_FEATURES: Final[tuple[str, ...]] = (*GRASP_FRAME_FEATURES, *ENVIRONMENT_FEATURES)


#: The grasp-frame contract. See the module docstring for the frame it uses and the two it does not.
#:
#: Deliberately absent from the bootstrap corpus and named so the gate keeps saying it:
#:   * ``quality``: one value on every row. A dropped return value, not missing data;
#:     `_closing_axis` computes a score, ranks on it, and returns three of its four tuple elements.
#:   * ``split``: 'train' on every row. The corpus ships no held-out set; one is made here,
#:     grouped by `object_id`.
BOOTSTRAP_JAW_V1: Final[FeatureSpec] = FeatureSpec(
    name="bootstrap_jaw_v1",
    frame="grasp",
    features=GRASP_FRAME_FEATURES,
    label="held",
    group="object_id",
    derived=True,
)


#: The object-frame design. Not what this scorer uses; kept so the alternative stays on the record
#: and so the gate's frame check has a genuine mismatch to refuse.
BOOTSTRAP_OBJECT_FRAME_V0: Final[FeatureSpec] = FeatureSpec(
    name="bootstrap_object_frame_v0",
    frame="object",
    features=(
        "width_mm",
        "approach_x", "approach_y", "approach_z",
        "axis_x", "axis_y", "axis_z",
        "centre_x_mm", "centre_y_mm", "centre_z_mm",
    ),
    label="held",
    group="object_id",
)


#: Stage one of the two-stage design: predict whether a candidate closes, fitted on this repo's
#: judged candidates rather than aletheia's shaken ones.
#:
#: A ranker fitted under `bootstrap_jaw_v1` puts a valid grasp first more often than the
#: calculator does, and still less often than picking at random. It loses to a coin toss because
#: the failure it has to predict is collision and its features cannot see the surroundings. This
#: spec adds them.
#:
#: Fitted on `valid`, the analytic verdict `verdict.py` computes, so the eval-grasps ladder is
#: self-referential for this model: a perfect reproduction of `verdict.py` would look perfect on
#: every scene and have learnt an estimator rather than the world. The ladder stays a development
#: signal and promotion is measured against the Isaac shake only.
VALID_JAW_V1: Final[FeatureSpec] = FeatureSpec(
    name="valid_jaw_v1",
    frame="grasp",
    features=FULL_FEATURES,
    label="valid",
    group="object_key",
    derived=True,
)


#: Stage two of the same design: predict whether a closed grasp survives being shaken. Same features
#: as stage one, different target; `held` comes from Isaac, not from `verdict.py`.
#:
#: That difference is the point of the two stages. Nothing measured on `valid` can tell a model that
#: learnt the world from one that learnt `verdict.py`; physics can, so `valid` trains and the shake
#: alone promotes.
#:
#: The shake draw is stratified, so a pooled hold rate over it is an artefact of the draw and not a
#: property of the corpus. The strata sit far apart end to end, `valid` holding far more often than
#: `rejected:too_wide`, so pooling them reports a number describing the sampler. Slice by
#: `physics_source`, or re-weight by the stratum sizes the shake writes beside its results.
HELD_JAW_V1: Final[FeatureSpec] = FeatureSpec(
    name="held_jaw_v1",
    frame="grasp",
    features=FULL_FEATURES,
    label="held",
    group="object_key",
    derived=True,
)


#: A spec that deliberately declares `contact_angle_deg` a feature, which it is not fit to be. It
#: exists to keep the train/serve check honest: the analytic reference reports one distinct value
#: on every label (the labeller enumerates exact antipodal axes, so the worst deviation is 0 by
#: construction) while the same field over the calculator's own candidates spans a wide range of
#: values.
WILLY_JAW_WITH_CONTACT_ANGLE: Final[FeatureSpec] = FeatureSpec(
    name="willy_jaw_with_contact_angle",
    frame="base",
    features=(*BOOTSTRAP_OBJECT_FRAME_V0.features, "approach_tilt_deg", "contact_angle_deg"),
    label="held",
    group="object_id",
)


SPECS: Final[dict[str, FeatureSpec]] = {
    BOOTSTRAP_JAW_V1.name: BOOTSTRAP_JAW_V1,
    BOOTSTRAP_OBJECT_FRAME_V0.name: BOOTSTRAP_OBJECT_FRAME_V0,
    HELD_JAW_V1.name: HELD_JAW_V1,
    VALID_JAW_V1.name: VALID_JAW_V1,
    WILLY_JAW_WITH_CONTACT_ANGLE.name: WILLY_JAW_WITH_CONTACT_ANGLE,
}


def spec_named(name: str) -> FeatureSpec:
    """Look one up, or fail with the list; a typo must not silently become a different contract."""
    try:
        return SPECS[name]
    except KeyError:
        known = ", ".join(sorted(SPECS)) or "(none)"
        raise KeyError(f"no feature spec named {name!r}; known specs: {known}") from None


# ------------------------------------------------------------------ the runtime side


def object_centre_mm(points_base_mm: np.ndarray, *, support_z_mm: float = 0.0) -> np.ndarray:
    """Estimate an object's centre from its segmented cloud, in BASE mm.

    xy from the cloud's centroid, z half way between the support plane and the highest measured
    point, not the cloud's own z centroid. A fused multi-view cloud never sees the underside, so its
    centroid is pulled upward systematically rather than noisily. The support plane is known (the
    pick loop resolves it; z = 0 in this generator) and the top is measured, so the midpoint uses a
    fact instead of averaging over a hole.

    Through the features the correction shows at the top of the ranked list and barely at all in
    AUROC, which is why AUROC alone would not show it.
    """
    points = np.atleast_2d(np.asarray(points_base_mm, dtype=np.float64))
    if points.size == 0:
        raise ValueError("cannot estimate an object centre from an empty cloud")
    centre = points.mean(axis=0)
    centre[2] = (float(support_z_mm) + float(points[:, 2].max())) / 2.0
    return centre


def grasp_frame_offsets(
    centre_mm: np.ndarray, position_mm: np.ndarray,
    approach: np.ndarray, closing_axis: np.ndarray,
) -> tuple[float, float, float]:
    """The object centre relative to the grasp, in the grasp's frame: (axis, approach, binormal).

    Signed. The magnitude of the binormal component adds nothing while the signed value lifts the
    whole set: which side of the finger plane the mass sits on is a torque, its distance is not.

    The frame is orthonormalised from the grasp's own two vectors, so it exists exactly whenever the
    grasp does; there is no estimation and nothing to be degenerate.
    """
    axis = np.asarray(closing_axis, dtype=np.float64)
    approach_v = np.asarray(approach, dtype=np.float64)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    approach_v = approach_v / max(float(np.linalg.norm(approach_v)), 1e-12)
    binormal = np.cross(axis, approach_v)
    norm = float(np.linalg.norm(binormal))
    if norm < 1e-9:
        # Parallel axis and approach: the GraspPoint contract promises perpendicular, so this is a
        # broken grasp rather than an edge case. Refused, because a zero binormal would silently make
        # the most informative feature 0 for that row.
        raise ValueError(
            "closing axis and approach are parallel, so the grasp frame has no binormal; the "
            "GraspPoint contract requires them perpendicular"
        )
    binormal = binormal / norm
    offset = np.asarray(centre_mm, dtype=np.float64) - np.asarray(position_mm, dtype=np.float64)
    return (float(offset @ axis), float(offset @ approach_v), float(offset @ binormal))


def jaw_clearance_mm(
    position_mm: np.ndarray, closing_axis: np.ndarray, width_mm: float,
    obstacle_points_mm: np.ndarray | None,
) -> float:
    """How much free space the two jaws have, from the nearest non-target point to either of them.

    The direct counterpart to `finger_collision`, which is the dominant rejection on the
    calculator's candidate population and the one the grasp-frame features are blind to.

    The two jaw positions rather than a sampled finger envelope: the envelope is the gripper model's
    business and lives behind a driver, while this has to be computable from a point cloud at
    scoring time and offline from the same cloud. Coarser, and it needs no collision stack.

    ``inf`` when no obstacle cloud is available: a large number meaning "nothing known to be in the
    way", which is what the grasp looks like when nothing was observed near it. A zero would claim a
    collision that was never measured.
    """
    if obstacle_points_mm is None:
        return float("inf")
    points = np.atleast_2d(np.asarray(obstacle_points_mm, dtype=np.float64))
    if points.size == 0:
        return float("inf")
    axis = np.asarray(closing_axis, dtype=np.float64)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    centre = np.asarray(position_mm, dtype=np.float64)
    half = float(width_mm) / 2.0
    jaws = np.vstack([centre + half * axis, centre - half * axis])
    return float(np.linalg.norm(points[None, :, :] - jaws[:, None, :], axis=2).min())


def contact_normal_alignment(
    position_mm: np.ndarray, closing_axis: np.ndarray, width_mm: float,
    target_points_mm: np.ndarray, *, neighbourhood_mm: float = 8.0, minimum_points: int = 6,
) -> float:
    """How antipodal the two contacts look to the cloud: |cos| between local normals and the axis,
    worst of the two.

    The most circularity-exposed feature in this file. `verdict.py` computes exactly this quantity
    exactly, from the true solid, and `not_antipodal` is the second most common rejection, so a
    model fed the reference's own answer would score beautifully and have learnt `verdict.py`.

    This never touches it. The normal is estimated the way a runtime has to: take the cloud points
    within `neighbourhood_mm` of where each jaw lands, and use the smallest principal direction of
    that patch. It is a worse answer than the reference's on purpose; it is the answer available
    from a depth image.

    Returns 0.0 when neither contact has enough neighbours to fit a plane. Not ``nan``: the scorer
    refuses non-finite input, and "the cloud is too sparse here to say" is a real state that should
    score badly rather than abort a pick.
    """
    points = np.atleast_2d(np.asarray(target_points_mm, dtype=np.float64))
    if points.size == 0:
        return 0.0
    axis = np.asarray(closing_axis, dtype=np.float64)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    centre = np.asarray(position_mm, dtype=np.float64)
    half = float(width_mm) / 2.0

    alignments: list[float] = []
    for jaw in (centre + half * axis, centre - half * axis):
        near = points[np.linalg.norm(points - jaw, axis=1) <= neighbourhood_mm]
        if len(near) < minimum_points:
            continue
        centred = near - near.mean(axis=0)
        # The smallest principal direction of a surface patch is its normal; `svd` returns them
        # largest-first, so the last row is the one wanted.
        normal = np.linalg.svd(centred, full_matrices=False)[2][-1]
        alignments.append(abs(float(normal @ axis)))
    if not alignments:
        return 0.0
    # The worst of the two, because a grasp is as antipodal as its poorer contact; the same rule
    # `check_jaw_grasp` applies when it takes the max deviation rather than the mean.
    return float(min(alignments))


def environment_features(
    position_mm: np.ndarray, approach: np.ndarray, closing_axis: np.ndarray, width_mm: float,
    target_points_mm: np.ndarray, *, obstacle_points_mm: np.ndarray | None = None,
    support_z_mm: float = 0.0, observedness: float = 1.0,
) -> dict[str, float]:
    """The four `ENVIRONMENT_FEATURES`, all from the cloud and the known support plane.

    ``grasp_height_mm`` and ``approach_tilt_deg`` go together on purpose: whether a fingertip ends
    up under the table depends on both how high the grasp is and how tilted the approach is.
    Mistaking one for the other puts every object under about 33 mm of grasp height out of reach;
    neither alone carries it.
    """
    approach_v = np.asarray(approach, dtype=np.float64)
    approach_v = approach_v / max(float(np.linalg.norm(approach_v)), 1e-12)
    down = np.array([0.0, 0.0, -1.0])
    return {
        "jaw_clearance_mm": jaw_clearance_mm(
            position_mm, closing_axis, width_mm, obstacle_points_mm),
        "grasp_height_mm": float(np.asarray(position_mm, dtype=np.float64)[2]) - float(support_z_mm),
        "approach_tilt_deg": float(
            np.degrees(np.arccos(np.clip(float(approach_v @ down), -1.0, 1.0)))),
        "contact_normal_alignment": contact_normal_alignment(
            position_mm, closing_axis, width_mm, target_points_mm),
    }


def derive_grasp_frame_columns(table: dict[str, Any]) -> dict[str, Any]:
    """Add `GRASP_FRAME_FEATURES` to a corpus table, from its canonical raw columns.

    Vectorised rather than looping `grasp_frame_offsets`, because a corpus runs to many thousands
    of rows and a gate should answer in under a second. The two paths must agree row for row; a
    fast path that disagrees with the reference path is worse than no fast path.

    The centre columns are the object centre relative to the grasp position already, in whatever
    frame the corpus uses; the projections below are onto that same frame's axes, so the result is
    frame-free.

    Idempotent, because two kinds of corpus arrive here: the bootstrap `.npz`, which carries raw
    vectors, and one built by `datagen build-ranker-corpus`, which computed these columns from the
    clouds and has no raw vectors left to recompute them from. A table that already has all three is
    returned untouched.

    A table with some of them is refused rather than half-filled: that is a table something went
    wrong while writing, and quietly completing it would produce a corpus whose columns came from
    two different places.
    """
    derived = [name for name in GRASP_FRAME_FEATURES[1:] if name in table]
    if len(derived) == len(GRASP_FRAME_FEATURES) - 1:
        return dict(table)
    if derived:
        raise ValueError(
            f"table carries {sorted(derived)} but not the rest of {list(GRASP_FRAME_FEATURES[1:])}; "
            f"half-derived, so completing it would mix columns from two sources"
        )

    centre = np.column_stack([
        np.asarray(table["centre_x_mm"], dtype=np.float64),
        np.asarray(table["centre_y_mm"], dtype=np.float64),
        np.asarray(table["centre_z_mm"], dtype=np.float64),
    ])
    axis = np.column_stack([
        np.asarray(table["axis_x"], dtype=np.float64),
        np.asarray(table["axis_y"], dtype=np.float64),
        np.asarray(table["axis_z"], dtype=np.float64),
    ])
    approach = np.column_stack([
        np.asarray(table["approach_x"], dtype=np.float64),
        np.asarray(table["approach_y"], dtype=np.float64),
        np.asarray(table["approach_z"], dtype=np.float64),
    ])
    axis = axis / np.maximum(np.linalg.norm(axis, axis=1, keepdims=True), 1e-12)
    approach = approach / np.maximum(np.linalg.norm(approach, axis=1, keepdims=True), 1e-12)
    binormal = np.cross(axis, approach)
    binormal = binormal / np.maximum(np.linalg.norm(binormal, axis=1, keepdims=True), 1e-12)

    out = dict(table)
    out["centre_along_axis_mm"] = (centre * axis).sum(axis=1)
    out["centre_along_approach_mm"] = (centre * approach).sum(axis=1)
    out["centre_along_binormal_mm"] = (centre * binormal).sum(axis=1)
    return out
