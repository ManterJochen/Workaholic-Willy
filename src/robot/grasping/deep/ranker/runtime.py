"""The learned grasp ranker at runtime: numpy only, ranking only, and nothing it cannot justify.

Ranking, not probability. A corpus holds whatever hold rate its draw produced; a cell's rate is
different and unknown, so a probability calibrated on the training corpus is wrong for any other
prior by construction, and a number that looks like a probability and is not one is the shape of
column the dataset gate refuses. So `score` returns the raw ensemble output: it orders candidates
and claims nothing else.
The base rate it was fitted at is in the artifact card, so a later calibration has what it needs;
until a corpus from the target cell exists, there is nothing honest to calibrate against. A sigmoid
would change nothing about the ordering, which is why none is applied.

No sklearn at runtime. The trainer is offline and uses it; this reads a serialised ensemble and
evaluates it in numpy, the same fence `scoring/success_probability` keeps: a vendored fitting
library has no business inside a pick.

This is a second implementation of a tree walk, and the hazard is that it agrees with itself and
disagrees with the model that was fitted. It stays only while something checks it against sklearn's
own `decision_function` on the very model that was exported, against `success_probability`'s
independent reader fed the same ensemble, and on a hand-built tree whose answers are known by
inspection.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

__all__ = [
    "ARTIFACT_KIND",
    "ARTIFACT_VERSION",
    "GbtRanker",
    "RankerTree",
    "load_ranker",
    "ranker_from_dict",
]

#: What this file is, written into every artifact so a wrong one fails loudly instead of scoring.
ARTIFACT_KIND: Final[str] = "grasp_gbt_ranker"
#: Bumped when the serialised shape changes, never for a retrain. A model is identified by its SHA.
ARTIFACT_VERSION: Final[int] = 1


@dataclass(frozen=True, slots=True)
class RankerTree:
    """One regression tree. A leaf has ``feature == -1``; an internal node goes left on ``<=``.

    Exactly the convention sklearn's `tree_` arrays use, so the export is a copy rather than a
    translation and there is no place for an off-by-one to hide between them.
    """

    feature: np.ndarray      # (nodes,) int, -1 at a leaf
    threshold: np.ndarray    # (nodes,) float
    left: np.ndarray         # (nodes,) int
    right: np.ndarray        # (nodes,) int
    value: np.ndarray        # (nodes,) float, only meaningful at a leaf


@dataclass(frozen=True, slots=True)
class GbtRanker:
    """``score = init_score + learning_rate * sum(tree leaf values)``. Higher ranks better."""

    spec: str
    features: tuple[str, ...]
    init_score: float
    learning_rate: float
    trees: tuple[RankerTree, ...]
    #: What the corpus it was fitted on actually held, so nobody has to guess later. Not used in the
    #: score; see the module docstring on why no probability is claimed.
    base_rate: float = 0.0
    #: SHA-256 over the serialised payload. The model's identity; a retrain changes it.
    sha256: str = ""

    def score(self, matrix: np.ndarray) -> np.ndarray:
        """Raw ensemble score for each row of ``matrix``, whose columns must be `features` in order."""
        rows = np.atleast_2d(np.asarray(matrix, dtype=np.float64))
        if rows.shape[1] != len(self.features):
            raise ValueError(
                f"ranker {self.spec!r} expects {len(self.features)} feature(s) "
                f"{self.features}, got {rows.shape[1]}"
            )
        if not np.isfinite(rows).all():
            # Refused rather than scored. A NaN walks a tree by falling through every comparison to
            # the same side, which produces a confident number from a missing measurement.
            raise ValueError("feature matrix holds NaN or infinite values")
        total = np.zeros(len(rows), dtype=np.float64)
        for tree in self.trees:
            total += _leaf_values(tree, rows)
        return self.init_score + self.learning_rate * total


def _leaf_values(tree: RankerTree, rows: np.ndarray) -> np.ndarray:
    """Walk every row to its leaf at once. Iterative because a tree is shallow and wide here."""
    node = np.zeros(len(rows), dtype=np.int64)
    active = tree.feature[node] >= 0
    while active.any():
        index = np.nonzero(active)[0]
        current = node[index]
        go_left = rows[index, tree.feature[current]] <= tree.threshold[current]
        node[index] = np.where(go_left, tree.left[current], tree.right[current])
        active = tree.feature[node] >= 0
    return tree.value[node]


def _payload_sha256(payload: dict[str, Any]) -> str:
    """Hash the payload with its own `sha256` field removed, so the field can hold its own hash."""
    without = {k: v for k, v in payload.items() if k != "sha256"}
    return hashlib.sha256(
        json.dumps(without, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def ranker_from_dict(payload: dict[str, Any], *, verify_sha: bool = True) -> GbtRanker:
    """Build a ranker from its serialised form, refusing anything that is not one.

    Every check is fail-closed: a wrong `kind` means someone pointed this at another model, a wrong
    `artifact_version` means the shape changed under it, and a wrong SHA means the file was edited
    after it was fitted, which is the one failure that would otherwise score silently.
    """
    kind = payload.get("kind")
    if kind != ARTIFACT_KIND:
        raise ValueError(f"artifact kind is {kind!r}, expected {ARTIFACT_KIND!r}")
    version = int(payload.get("artifact_version", -1))
    if version != ARTIFACT_VERSION:
        raise ValueError(
            f"artifact_version {version} is not {ARTIFACT_VERSION}; the serialised shape differs "
            f"from what this reader understands, so the scores would be wrong rather than absent"
        )
    recorded = str(payload.get("sha256", ""))
    if verify_sha:
        actual = _payload_sha256(payload)
        if recorded and recorded != actual:
            raise ValueError(
                f"artifact SHA mismatch: recorded {recorded[:16]}..., computed {actual[:16]}...; the "
                f"file changed after it was fitted"
            )

    trees = tuple(
        RankerTree(
            feature=np.asarray(t["feature"], dtype=np.int64),
            threshold=np.asarray(t["threshold"], dtype=np.float64),
            left=np.asarray(t["left"], dtype=np.int64),
            right=np.asarray(t["right"], dtype=np.int64),
            value=np.asarray(t["value"], dtype=np.float64),
        )
        for t in payload["trees"]
    )
    if not trees:
        raise ValueError("artifact holds no trees")
    return GbtRanker(
        spec=str(payload["spec"]),
        features=tuple(str(f) for f in payload["features"]),
        init_score=float(payload["init_score"]),
        learning_rate=float(payload["learning_rate"]),
        trees=trees,
        base_rate=float(payload.get("base_rate", 0.0)),
        sha256=recorded or _payload_sha256(payload),
    )


def load_ranker(path: str | Path, *, verify_sha: bool = True) -> GbtRanker:
    """Read an artifact from disk. Raises rather than returning ``None``: a caller that asked for a
    specific model and got nothing would fall back to an unranked order without noticing."""
    return ranker_from_dict(
        json.loads(Path(path).read_text(encoding="utf-8")), verify_sha=verify_sha)
