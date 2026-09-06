"""Shared logistic feature primitives (``_project_feature`` / ``_sigmoid``) for the candidate + ranking policies. A neutral leaf, no rl/ imports."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def _executed_candidate_features(rec: Mapping[str, Any]) -> dict[str, Any]:
    """The executed candidate's per-candidate feature row from ``extra['rl_candidate_features']``.

    Sim records log per-candidate features in that list (each row ``{candidate_id, rank, executed, features}``),
    not as flat top-level or ``extra`` keys, so the trainers' flat-key projection reads all-zeros on sim data
    and the sim ranker and candidate policies degenerate. This surfaces the executed (else rank-0) candidate's
    ``features`` dict as a fallback source. Returns ``{}`` when a record carries no ``rl_candidate_features``,
    so callers fall through to their existing flat-key sources unchanged.
    """
    extra = rec.get("extra")
    rows = extra.get("rl_candidate_features") if isinstance(extra, Mapping) else None
    if not isinstance(rows, list):
        return {}
    cand = [r for r in rows if isinstance(r, Mapping) and isinstance(r.get("features"), Mapping)]
    if not cand:
        return {}
    chosen = next((r for r in cand if r.get("executed")), None)
    if chosen is None:
        chosen = min(cand, key=lambda r: r.get("rank", 1_000_000))
    feats = chosen.get("features")
    return dict(feats) if isinstance(feats, Mapping) else {}


def _project_feature(value: Any) -> float:
    """SAR-extractor projection rules; bool is tested before int."""

    if value is None:
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _sigmoid(z: float) -> float:
    # Saturating arithmetic avoids float overflow on extreme inputs.
    if z >= 0.0:
        ez = math.exp(-z) if z < 700.0 else 0.0
        return 1.0 / (1.0 + ez)
    ez = math.exp(z) if z > -700.0 else 0.0
    return ez / (1.0 + ez)
