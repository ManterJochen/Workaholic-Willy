"""Fitting the grasp ranker: offline, and it reports what it is worth before it writes anything.

sklearn lives here and nowhere near a pick. The artifact this writes is read by `ranker/runtime.py`
in numpy alone, which is the same fence `scoring/success_probability` keeps.

The split is grouped by object, not by row. One object contributes many rows, so a row-level split
puts the same object on both sides and inflates every number reported from it. That is the leakage
the dataset gate refuses, and it would be self-inflicted here.

`GradientBoostingClassifier` rather than the faster histogram variant, because
`HistGradientBoostingClassifier` bins its features and keeps no node-wise tree to serialise. What is
measured and what ships have to be the same family, so the two are compared on the same folds, and
the exportable one gives up nothing worth keeping. 400 trees of depth 3 keeps the signal and can be
written down, so that is what ships.

Ranking metrics, not accuracy. At a low base rate a model that says "no" to everything is accurate
and useless. What a ranker is for is the top of its own list, so this reports AUROC over the whole
out-of-fold set and precision@k at several k, and the straw man it must beat is `width_mm` alone,
which already predicts this label: the hold rate climbs across the width bands.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.robot.grasping.deep.ranker.features import FeatureSpec, derive_grasp_frame_columns
from src.robot.grasping.deep.ranker.runtime import (
    ARTIFACT_KIND,
    ARTIFACT_VERSION,
    _payload_sha256,
)

__all__ = ["TrainingReport", "export_ranker", "fit_ranker", "report_at_k"]

#: The k values every report quotes. At k = 100 a difference of 0.02 is two rows, so the larger k
#: are what a claim gets made on.
REPORT_K: tuple[int, ...] = (100, 250, 500, 1000)


@dataclass(frozen=True, slots=True)
class TrainingReport:
    """What the fit is worth, out of fold. Written into the artifact card, never computed in fold."""

    rows: int
    groups: int
    positives: int
    auroc: float
    precision_at_k: dict[int, float]
    baseline_auroc: float
    baseline_precision_at_k: dict[int, float]

    @property
    def base_rate(self) -> float:
        return self.positives / self.rows if self.rows else 0.0

    def summary(self) -> str:
        lines = [
            f"{self.rows} row(s), {self.groups} group(s), {self.positives} positive "
            f"({100 * self.base_rate:.2f} %); a random ranker scores that at every k",
            f"  AUROC   model {self.auroc:.4f}   width-only baseline {self.baseline_auroc:.4f}",
        ]
        for k in sorted(self.precision_at_k):
            lines.append(
                f"  P@{k:<5d} model {self.precision_at_k[k]:.3f}   "
                f"width-only {self.baseline_precision_at_k[k]:.3f}")
        return "\n".join(lines)


def report_at_k(y_true: np.ndarray, score: np.ndarray, k: int) -> float:
    """Of the k highest-scored rows, what fraction held. ``nan`` when there are fewer than k rows."""
    if len(score) < k:
        return float("nan")
    return float(np.asarray(y_true)[np.argsort(-np.asarray(score))[:k]].mean())


def _matrix(table: dict[str, Any], spec: FeatureSpec) -> np.ndarray:
    return np.column_stack([np.asarray(table[f], dtype=np.float64) for f in spec.features])


def _out_of_fold(matrix: np.ndarray, y: np.ndarray, groups: np.ndarray, make: Any, folds: int):
    from sklearn.model_selection import GroupKFold  # noqa: PLC0415 (offline only)

    oof = np.full(len(y), np.nan)
    for train_idx, test_idx in GroupKFold(n_splits=folds).split(matrix, y, groups):
        model = make()
        model.fit(matrix[train_idx], y[train_idx])
        oof[test_idx] = model.decision_function(matrix[test_idx])
    return oof


def fit_ranker(
    table: dict[str, Any], spec: FeatureSpec, *,
    folds: int = 5, n_estimators: int = 400, learning_rate: float = 0.05, max_depth: int = 3,
    random_state: int = 0,
) -> tuple[Any, TrainingReport]:
    """Fit on everything and score honestly out of fold. Returns ``(fitted model, report)``.

    Two fits, deliberately: the out-of-fold pass earns the numbers, and the final model is refit on all
    rows because that is the one that ships. Reporting the shipped model's in-sample score instead
    would be reporting how well it memorised its own training rows.
    """
    from sklearn.ensemble import GradientBoostingClassifier  # noqa: PLC0415 (offline only)
    from sklearn.metrics import roc_auc_score  # noqa: PLC0415

    if spec.derived:
        table = derive_grasp_frame_columns(table)
    matrix = _matrix(table, spec)
    y = (np.asarray(table[spec.label], dtype=np.float64) > 0.5).astype(int)
    groups = np.asarray(table[spec.group])

    def make() -> Any:
        return GradientBoostingClassifier(
            n_estimators=n_estimators, learning_rate=learning_rate,
            max_depth=max_depth, random_state=random_state)

    oof = _out_of_fold(matrix, y, groups, make, folds)
    # The straw man gets the same folds and the same family; an easier baseline would flatter the
    # model for a reason unrelated to the features.
    width = matrix[:, [spec.features.index("width_mm")]] if "width_mm" in spec.features \
        else matrix[:, :1]
    oof_width = _out_of_fold(width, y, groups, make, folds)

    report = TrainingReport(
        rows=len(y), groups=int(np.unique(groups).size), positives=int(y.sum()),
        auroc=float(roc_auc_score(y, oof)),
        precision_at_k={k: report_at_k(y, oof, k) for k in REPORT_K},
        baseline_auroc=float(roc_auc_score(y, oof_width)),
        baseline_precision_at_k={k: report_at_k(y, oof_width, k) for k in REPORT_K},
    )
    final = make()
    final.fit(matrix, y)
    return final, report


def export_ranker(model: Any, spec: FeatureSpec, report: TrainingReport) -> dict[str, Any]:
    """Serialise a fitted `GradientBoostingClassifier` into the runtime artifact.

    ``init_score`` is recovered rather than read off a private attribute: it is
    ``decision_function(x) - learning_rate * sum(leaf(x))`` for a probe row, which is exact by the
    model's own definition. A recovery is only allowed while something checks the reconstruction
    against sklearn's `decision_function` over many rows.
    """
    trees = []
    for stage in model.estimators_:
        tree = stage[0].tree_
        trees.append({
            "feature": [int(v) for v in tree.feature],
            "threshold": [float(v) for v in tree.threshold],
            "left": [int(v) for v in tree.children_left],
            "right": [int(v) for v in tree.children_right],
            "value": [float(v) for v in tree.value.reshape(-1)],
        })

    probe = np.zeros((1, len(spec.features)), dtype=np.float64)
    leaf_total = 0.0
    for stage in model.estimators_:
        leaf_total += float(stage[0].predict(probe)[0])
    init_score = float(model.decision_function(probe)[0]) - float(model.learning_rate) * leaf_total

    payload: dict[str, Any] = {
        "kind": ARTIFACT_KIND,
        "artifact_version": ARTIFACT_VERSION,
        "spec": spec.name,
        "frame": spec.frame,
        "features": list(spec.features),
        "init_score": init_score,
        "learning_rate": float(model.learning_rate),
        "trees": trees,
        "base_rate": report.base_rate,
        # The card. Not decoration: a score with no provenance is a number nobody can defend later,
        # and "ranking only" has to travel with the model or it becomes a probability by assumption.
        "card": {
            "claims": "ranking only: not a calibrated probability",
            # Computed, not a constant. The same trainer fits at one base rate on the bootstrap
            # corpus and at a very different one on a stratified physics draw, so a card that
            # carries a reason belonging to a different corpus reads as provenance and is fiction.
            "why_no_probability": (
                f"fitted at a {100 * report.base_rate:.2f} %-class base rate on THIS corpus; a cell's "
                "rate differs and is unknown, so a calibrated P(hold) would be wrong for any other "
                "prior. If the corpus was a stratified draw, its base rate describes the sampler and "
                "not any population, which is a second and independent reason"
            ),
            "rows": report.rows,
            "groups": report.groups,
            "positives": report.positives,
            "split": f"GroupKFold by {spec.group!r} -- never by row",
            "auroc_out_of_fold": report.auroc,
            "precision_at_k_out_of_fold": {str(k): v for k, v in report.precision_at_k.items()},
            "baseline": "width_mm alone, same folds and same model family",
            "baseline_auroc": report.baseline_auroc,
            "baseline_precision_at_k": {
                str(k): v for k, v in report.baseline_precision_at_k.items()},
        },
    }
    payload["sha256"] = _payload_sha256(payload)
    return payload


def write_artifact(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write the artifact, pretty and sorted, so a diff between two models is readable."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out
