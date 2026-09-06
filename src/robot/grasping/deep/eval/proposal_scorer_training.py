"""Stage 3.6: a scorer trained on the generator's own proposals, with physics as the target.

Named `proposal_scorer` and not `scorer`: two different things here are called a scorer.
`ranker.runtime` holds `GbtRanker`, the one a cell loads and reads during a pick; this module fits
a model offline. Only the names keep them apart.

The target is the physics referee, not the analytic verdict. On the same rows, the same folds and
the same features, changing only what the model is asked to predict, the physics target scores
higher in every rejection stratum and pooled over all of them.

`finger_collision` is the stratum to read: collision is computed exactly here, so a learner should
need no physics for it, and the analytic verdict still loses. On `too_wide` the analytic verdict is
close to a coin toss.

The proposals are the generator's own. Outside work measures +6.5 % AUC from training a
discriminator on the generator's proposals rather than on the offline label set, and names the
cause: the shift is concentrated in the negatives, because an offline set contains only
collision-free grasps while a generator produces slightly colliding ones. A scorer that never sees
what its own generator gets wrong cannot learn to refuse it.

This is not a ranker over labelled candidates and must not be trained as one: that is the mimicry
loop, where a model trained on labels derived from the analytic verdict is then graded by that
verdict. It answers one question about one pose the model itself chose, with the simulator as the
judge.

It is useless without negatives. A proposal set in which everything held, or nothing did, trains a
constant. `fit_scorer` refuses one rather than returning a model whose AUC is undefined and whose
predictions are a single number, because that model would look trained and rank nothing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

__all__ = ["ScorerData", "fit_scorer", "load_pairs"]

#: The pose features a scorer reads beside the crop embedding. Deliberately small and hand-made: a
#: compact hand-designed description of a grasp can carry more than a learned point encoder does,
#: so a small feature set is not a compromise here.
_FEATURE_NAMES: tuple[str, ...] = (
    "width_mm", "confidence", "approach_z", "axis_z", "approach_tilt", "height_mm")


@dataclass(frozen=True, slots=True)
class ScorerData:
    """Proposals joined to their physics verdicts, ready to fit.

    The join is by (scene, rank), not by pose. A pose join is structurally ambiguous, because a
    label and the row describing it carry identical poses; it does not fail loudly, it silently
    drops the rows that do not match and reports a smaller dataset.
    """

    features: np.ndarray          # (N, F)
    held: np.ndarray              # (N,) bool
    groups: np.ndarray            # (N,) scene id, so a split can be scene-disjoint
    names: tuple[str, ...]

    @property
    def positive_share(self) -> float:
        return float(self.held.mean()) if len(self.held) else 0.0


def _features_of(proposal: dict[str, Any]) -> list[float]:
    approach = np.asarray(proposal["approach"], dtype=np.float64)
    axis = np.asarray(proposal["closing_axis"], dtype=np.float64)
    return [
        float(proposal["width_mm"]),
        float(proposal.get("confidence", 0.0)),
        float(approach[2]),
        float(abs(axis[2])),                      # antipodal: the sign carries nothing
        float(np.degrees(np.arccos(np.clip(-approach[2], -1.0, 1.0)))),
        float(np.asarray(proposal["position_mm"], dtype=np.float64)[2]),
    ]


def load_pairs(proposals: str | Path, verdicts: str | Path) -> ScorerData:
    """Join a proposal file to the physics file that judged it.

    A proposal without a verdict is dropped and counted, never given a default. A refused trial is
    not a failed grasp: it is a trial the harness would not stand behind, and scoring it as a negative
    would teach the scorer that the harness's own problems are properties of the grasp.
    """
    rows = [json.loads(line) for line in
            Path(proposals).read_text(encoding="utf-8").splitlines() if line.strip()]
    judged: dict[tuple[str, int], bool] = {}
    for line in Path(verdicts).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        # `startswith`, the same check the writer uses, not a substring search. The writer marks a
        # refusal by starting the note with the word; a substring match would also drop any future
        # note that merely mentions one.
        if "controls" in row or str(row.get("note", "")).startswith("refused"):
            continue
        judged[(str(row["scene_id"]), int(row.get("row_index", -1)))] = bool(row.get("held"))

    features, held, groups = [], [], []
    for row in rows:
        key = (str(row["scene_id"]), int(row.get("rank", -1)))
        if key not in judged:
            continue
        features.append(_features_of(row))
        held.append(judged[key])
        groups.append(str(row["scene_id"]))
    return ScorerData(np.asarray(features, dtype=np.float64).reshape(len(features), -1),
                      np.asarray(held, dtype=bool), np.asarray(groups, dtype=object),
                      _FEATURE_NAMES)


def fit_scorer(data: ScorerData, *, folds: int = 5, seed: int = 0) -> dict[str, Any]:
    """Fit and score out of fold, on a scene-disjoint split. Returns the report.

    Scene-disjoint, because two proposals on one scene share a point cloud. A random split puts the
    same cloud on both sides and reports a number about memorising a scene rather than about judging
    a grasp. The sibling mistake is a held-out set taken as a prefix of a group-ordered array: it is
    a handful of asset groups, with a ceiling of its own that is not the population's.

    A degenerate target is refused. If everything held, or nothing did, there is no ranking to
    learn: the fit returns a constant, AUC is undefined, and the report still looks like a trained
    model. Refused by name.
    """
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
    from sklearn.metrics import roc_auc_score  # noqa: PLC0415
    from sklearn.model_selection import GroupKFold  # noqa: PLC0415

    if len(data.held) < folds * 2:
        raise ValueError(f"{len(data.held)} judged proposal(s) is too few to fit and score")
    positives = int(data.held.sum())
    if positives == 0 or positives == len(data.held):
        raise ValueError(
            f"the target is degenerate: {positives} of {len(data.held)} proposals held. A scorer "
            f"fitted here would be a constant, its AUC undefined, and its report indistinguishable "
            f"from a trained one")

    scores = np.zeros(len(data.held), dtype=np.float64)
    splitter = GroupKFold(n_splits=min(folds, len(set(data.groups.tolist()))))
    for train, test in splitter.split(data.features, data.held, groups=data.groups):
        if len(set(data.held[train].tolist())) < 2:
            continue
        model = LogisticRegression(max_iter=2000, C=1.0)
        model.fit(data.features[train], data.held[train])
        scores[test] = model.predict_proba(data.features[test])[:, 1]

    auc = float(roc_auc_score(data.held, scores))
    # The floor beside the score, always. Confidence is one of the features, so a scorer that has
    # learned nothing beyond "trust the head" still beats 0.5.
    confidence = data.features[:, data.names.index("confidence")]
    width = data.features[:, data.names.index("width_mm")]
    return {
        "proposals": int(len(data.held)),
        "held": positives,
        "hold_rate": round(float(data.held.mean()), 4),
        "scenes": int(len(set(data.groups.tolist()))),
        "auc": round(auc, 4),
        "auc_head_confidence_alone": round(float(roc_auc_score(data.held, confidence)), 4),
        "auc_width_alone": round(float(roc_auc_score(data.held, width)), 4),
        "features": list(data.names),
    }


def fit_from_files(proposals: str | Path, verdicts: str | Path, *, folds: int = 5,
                   seed: int = 0) -> dict[str, Any]:
    """The whole stage 3.6 measurement, from the two files the pipeline already writes."""
    data = load_pairs(proposals, verdicts)
    report = fit_scorer(data, folds=folds, seed=seed)
    report["proposal_file"] = str(proposals)
    report["verdict_file"] = str(verdicts)
    return report


def unjudged(proposals: str | Path, verdicts: str | Path) -> int:
    """How many proposals never got a verdict. Reported rather than silently dropped."""
    data = load_pairs(proposals, verdicts)
    total = sum(1 for line in Path(proposals).read_text(encoding="utf-8").splitlines()
                if line.strip())
    return total - len(data.held)


def summarise(report: dict[str, Any], lines: Sequence[str] = ()) -> list[str]:
    """The report as text, with the floors beside the score because one without them misleads."""
    out = list(lines)
    out.append(f"  {report['proposals']} judged proposal(s) over {report['scenes']} scene(s)")
    out.append(f"  referee success rate  {report['hold_rate'] * 100:5.1f} %")
    out.append(f"  scorer AUC            {report['auc']:.4f}")
    out.append(f"    head confidence alone {report['auc_head_confidence_alone']:.4f}")
    out.append(f"    width alone           {report['auc_width_alone']:.4f}")
    return out
