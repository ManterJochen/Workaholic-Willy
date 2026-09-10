"""(state, action, reward) projection.

One stable extractor: ``BaselineSARExtractor``. Additional
extractors may register against the :class:`SARExtractor` protocol;
this baseline must remain stable so older artifacts replay
deterministically.

State features: a flat dict of scalars derived from the telemetry
extras (uncertainty + fusion + latency channels). Missing fields are
projected to ``0.0`` deterministically: every state vector has the
same key set so downstream code can build dense arrays without
per-record schema branches.

Action: a stable string token derived from the executed grasp pose,
else from the last recovery action's token, else ``reject`` when
``final_outcome`` is ``decision_fail_closed``, ``no_valid_grasp`` or
``unsafe_recovery_refused``, else ``noop``. The record's
``verification`` block is not read. The token is intentionally coarse.

Reward: ``1.0`` for ``final_outcome=="succeeded"``, ``0.0``
otherwise; kept binary so the extractor is reproducible from any
artifact.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

#: Stable order of state-feature keys. Part of the contract; do not
#: reorder. New features go at the end and bump
#: ``SAR_EXTRACTOR_VERSION``.
STATE_FEATURE_KEYS: tuple[str, ...] = (
    "uncertainty_score",
    "uncertainty_disagreement",
    "fused_view_count",
    "fusion_evidence_quality",
    "predicted_success_probability",
    "decision_latency_ms",
    "ranking_latency_ms",
    "fusion_latency_ms",
    "drift_severity",
    "ood_flagged",
    "degraded_mode_active",
    "multi_view_occlusion_reduced",
)

ACTION_NOOP: str = "noop"
ACTION_GRASP_PREFIX: str = "grasp"
ACTION_RECOVERY_PREFIX: str = "recovery"
ACTION_REJECT: str = "reject"

SAR_EXTRACTOR_VERSION: int = 1


@dataclass(frozen=True)
class SAR:
    """Single (state, action, reward) tuple plus identity metadata."""

    attempt_id: str
    mode: str
    state: dict[str, float]
    action: str
    reward: float
    extractor: str
    extractor_version: int

    def to_json(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "mode": self.mode,
            "state": dict(self.state),
            "action": self.action,
            "reward": float(self.reward),
            "extractor": self.extractor,
            "extractor_version": int(self.extractor_version),
        }


class SARExtractor(Protocol):
    """Stable interface for (s, a, r) extraction."""

    name: str
    version: int

    def extract(self, record: Mapping[str, Any]) -> SAR: ...


def _project_state(record: Mapping[str, Any]) -> dict[str, float]:
    extra = record.get("extra") or {}
    if not isinstance(extra, Mapping):
        extra = {}
    out: dict[str, float] = {}
    for key in STATE_FEATURE_KEYS:
        v = extra.get(key)
        if isinstance(v, bool):
            out[key] = 1.0 if v else 0.0
        elif isinstance(v, (int, float)):
            out[key] = float(v)
        else:
            out[key] = 0.0
    return out


def _stable_grasp_token(grasp: Any) -> str:
    if not isinstance(grasp, Mapping):
        return f"{ACTION_GRASP_PREFIX}:unknown"
    payload = json.dumps(grasp, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"{ACTION_GRASP_PREFIX}:{digest}"


def _project_action(record: Mapping[str, Any]) -> str:
    """The action token. ``selected_grasp`` first, because that is the grasp the outcome graded.

    This branch had no producer until 2026-09-10, and the repair was a writer rather than a
    deletion. ``record_logging.to_attempt_record`` is the only thing in ``src`` that turns a real
    pick into a record, and it set none of the three grasp fields, so every production record fell
    through to ``ACTION_NOOP``: one constant, on every row, in the column an RL dataset exists to
    vary. Deleting the branches would have made that permanent and would have changed how existing
    artifacts extract, since the replay fixtures carry ``selected_grasp``, which this extractor
    promises not to do. ``to_attempt_record`` now stamps the executed grasp into ``selected_grasp``.

    ``refined_grasp`` and ``initial_grasp`` still have no producer. They are reachable from an
    artifact written outside this repository and from the closed-loop path own report, which no
    record writer reads yet; they are kept because removing a branch an old artifact can still reach
    would silently change how that artifact replays.
    """
    selected = record.get("selected_grasp")
    if isinstance(selected, Mapping):
        return _stable_grasp_token(selected)
    refined = record.get("refined_grasp")
    if isinstance(refined, Mapping):
        return _stable_grasp_token(refined)
    initial = record.get("initial_grasp")
    if isinstance(initial, Mapping):
        return _stable_grasp_token(initial)
    recoveries = record.get("recovery_actions")
    if isinstance(recoveries, Sequence) and recoveries:
        last = recoveries[-1]
        if isinstance(last, Mapping):
            tok = last.get("kind") or last.get("action") or last.get("type")
            if isinstance(tok, str) and tok:
                return f"{ACTION_RECOVERY_PREFIX}:{tok}"
        return f"{ACTION_RECOVERY_PREFIX}:unknown"
    if record.get("final_outcome") in {
        "decision_fail_closed",
        "no_valid_grasp",
        "unsafe_recovery_refused",
    }:
        return ACTION_REJECT
    return ACTION_NOOP


def _project_reward(record: Mapping[str, Any]) -> float:
    return 1.0 if record.get("final_outcome") == "succeeded" else 0.0


@dataclass
class BaselineSARExtractor:
    """Stable baseline extractor."""

    name: str = "v1_baseline"
    version: int = SAR_EXTRACTOR_VERSION

    def extract(self, record: Mapping[str, Any]) -> SAR:
        attempt_id = record.get("attempt_id")
        if not isinstance(attempt_id, str) or not attempt_id:
            raise ValueError("record missing attempt_id; cannot extract SAR")
        mode = record.get("mode")
        if not isinstance(mode, str):
            raise ValueError(
                f"record {attempt_id}: missing string 'mode' "
                f"required for SAR extraction"
            )
        return SAR(
            attempt_id=attempt_id,
            mode=mode,
            state=_project_state(record),
            action=_project_action(record),
            reward=_project_reward(record),
            extractor=self.name,
            extractor_version=self.version,
        )


__all__ = (
    "ACTION_GRASP_PREFIX",
    "ACTION_NOOP",
    "ACTION_RECOVERY_PREFIX",
    "ACTION_REJECT",
    "SAR",
    "SARExtractor",
    "SAR_EXTRACTOR_VERSION",
    "STATE_FEATURE_KEYS",
    "BaselineSARExtractor",
)
