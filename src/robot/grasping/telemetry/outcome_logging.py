"""Outcome logging for autonomous grasp attempts.

This module is a pure recorder. It does not call perception, the robot or the
verification stack, and only serialises typed reports produced elsewhere into a
JSONL-safe :class:`GraspAttemptRecord`.

What it is for
--------------
* Replay-quality logs with no hardware in the loop.
* A stable, machine-readable schema, with no ``repr`` strings.
* A lossless round-trip through :meth:`GraspAttemptRecord.to_dict` and
  :meth:`GraspAttemptRecord.from_dict`.
* No numpy or dataclass leaking into the JSON payload.
* No unknown field dropped silently: a caller stashes what it needs in the free-form
  ``extra`` bag.

Scope
-----
The record stores summaries, meaning positions, scores, telemetry counters and outcome
strings, and not the raw perception payload. Images, depth maps and full point clouds
are deliberately omitted, and replay tooling that needs pixels references the perception
system separately through the frame fingerprint.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Iterator, Mapping, Optional, Sequence

import numpy as np

__all__ = [
    "GraspAttemptRecord",
    "append_jsonl",
    "execution_metadata_from",
    "frame_metadata_from",
    "grasp_metadata_from",
    "iter_jsonl",
    "json_safe",
    "profile_metadata_from",
    "recovery_metadata_from",
    "refinement_metadata_from",
    "target_metadata_from",
    "verification_metadata_from",
]


# ---------------------------------------------------------------------------
# JSON sanitisation
# ---------------------------------------------------------------------------


def json_safe(value: Any) -> Any:
    """Return a JSON-serialisable equivalent of ``value``.

    It handles the data shapes the grasp stack emits:

    * ``None``, ``bool``, ``int``, ``float`` and ``str`` pass through, with a ``NaN`` or
      an infinity coerced to ``None`` so a strict JSON parser does not choke.
    * A ``numpy.ndarray`` is converted recursively through ``.tolist()``.
    * A ``numpy.integer``, ``numpy.floating`` or ``numpy.bool_`` is coerced to a native
      Python scalar.
    * An ``Enum`` becomes its ``.value``, recursed.
    * A ``Mapping`` is recursed into a ``dict`` keyed by ``str(key)``.
    * A ``Sequence``, list or tuple, is recursed into a ``list``.
    * Anything with a ``to_dict()`` method becomes that dict, recursed.
    * Anything else falls back to ``str(value)``.
    """

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return json_safe(float(value))
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Enum):
        return json_safe(value.value)
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(v) for v in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return json_safe(to_dict())
        except Exception:  # pragma: no cover (defensive)
            return str(value)
    return str(value)


# ---------------------------------------------------------------------------
# Builders for typed reports produced upstream
# ---------------------------------------------------------------------------


def profile_metadata_from(profile: Any) -> dict[str, Any]:
    """Snapshot a :class:`GraspBehaviorProfile` into a JSON-safe dict.

    It is duck-typed, so any object with the same attributes works.
    """

    return {
        "mode": json_safe(getattr(profile, "mode", None)),
        "sampling_mode": json_safe(getattr(profile, "sampling_mode", None)),
        "refinement_enabled": bool(getattr(profile, "refinement_enabled", False)),
        "verification_enabled": bool(
            getattr(profile, "verification_enabled", False)
        ),
        "recovery_allowed_actions": [
            json_safe(a)
            for a in tuple(getattr(profile, "recovery_allowed_actions", ()) or ())
        ],
    }


def frame_metadata_from(frame: Any) -> dict[str, Any]:
    """Summarise a :class:`PerceptionFrame` without persisting pixels.

    It keeps the shape fingerprints, the timestamp, the segmentation count and the
    intrinsics, which is enough to correlate back to a persisted frame.
    """

    depth = getattr(frame, "depth_map", None)
    rgb = getattr(frame, "rgb", None)
    intrinsics = getattr(frame, "intrinsics", None)
    segmentations = tuple(getattr(frame, "segmentations", ()) or ())

    depth_shape: Optional[list[int]] = None
    if depth is not None:
        depth_shape = [int(d) for d in np.asarray(depth).shape]

    rgb_shape: Optional[list[int]] = None
    if rgb is not None:
        rgb_shape = [int(d) for d in np.asarray(rgb).shape]

    intrinsics_list: Optional[list[list[float]]] = None
    if intrinsics is not None:
        intrinsics_list = json_safe(np.asarray(intrinsics).tolist())

    return {
        "timestamp": json_safe(getattr(frame, "timestamp", None)),
        "depth_shape": depth_shape,
        "rgb_shape": rgb_shape,
        "has_rgb": rgb is not None,
        "segmentation_count": len(segmentations),
        "intrinsics": intrinsics_list,
    }


def target_metadata_from(target: Any) -> Optional[dict[str, Any]]:
    """Summarise a :class:`TargetIdentity`, its mask shape and centroid."""

    if target is None:
        return None
    mask = getattr(target, "mask", None)
    mask_shape: Optional[list[int]] = None
    if mask is not None:
        mask_shape = [int(d) for d in np.asarray(mask).shape]
    centroid = getattr(target, "centroid_xy", None)
    return {
        "mask_shape": mask_shape,
        "centroid_xy": json_safe(centroid),
        "area_px": int(getattr(target, "area_px", 0) or 0),
        "label": getattr(target, "label", None),
    }


def grasp_metadata_from(grasp: Any) -> Optional[dict[str, Any]]:
    """Summarise a :class:`GraspPoint`-like object.

    It prefers the object own ``to_dict()``, so the canonical schema is reused verbatim.
    """

    if grasp is None:
        return None
    to_dict = getattr(grasp, "to_dict", None)
    if callable(to_dict):
        try:
            return json_safe(to_dict())
        except Exception:  # pragma: no cover (defensive)
            pass
    return {
        "position": json_safe(getattr(grasp, "position", None)),
        "approach": json_safe(getattr(grasp, "approach", None)),
        "axis": json_safe(getattr(grasp, "axis", None)),
        "grip_width_mm": json_safe(getattr(grasp, "grip_width_mm", None)),
        "score": json_safe(getattr(grasp, "score", None)),
        "frame": json_safe(getattr(grasp, "frame", None)),
        "label": getattr(grasp, "label", None),
        "metadata": json_safe(getattr(grasp, "metadata", {}) or {}),
    }


def refinement_metadata_from(report: Any) -> Optional[dict[str, Any]]:
    """Summarise a :class:`RefinementReport`."""

    if report is None:
        return None
    return {
        "outcome": json_safe(getattr(report, "outcome", None)),
        "matched_segmentation_index": json_safe(
            getattr(report, "matched_segmentation_index", None)
        ),
        "match_iou": json_safe(getattr(report, "match_iou", None)),
        "position_delta_mm": json_safe(
            getattr(report, "position_delta_mm", None)
        ),
        "orientation_delta_deg": json_safe(
            getattr(report, "orientation_delta_deg", None)
        ),
        "grip_width_delta_mm": json_safe(
            getattr(report, "grip_width_delta_mm", None)
        ),
        "failure_reason": json_safe(getattr(report, "failure_reason", None)),
        "telemetry": json_safe(getattr(report, "telemetry", {}) or {}),
    }


def verification_metadata_from(report: Any) -> Optional[dict[str, Any]]:
    """Summarise a :class:`GraspVerificationReport`."""

    if report is None:
        return None
    return {
        "outcome": json_safe(getattr(report, "outcome", None)),
        "reason": getattr(report, "reason", ""),
        "telemetry": json_safe(getattr(report, "telemetry", {}) or {}),
    }


def execution_metadata_from(report: Any) -> Optional[dict[str, Any]]:
    """Summarise the executed-grasp outcome, the ``execution`` block, from a report.

    It takes an ``AutonomousGraspReport`` or a bare pick report, and is duck-typed:
    it reads ``pick_report.outcome`` and ``pick_report.executed_grasp``.

    It returns ``None`` where nothing executed, so the telemetry audit flags a
    ``succeeded`` or ``execution_failed`` record that genuinely lacks execution
    telemetry."""

    if report is None:
        return None
    pick = getattr(report, "pick_report", None)
    if pick is None:
        pick = report
    outcome = getattr(pick, "outcome", None)
    executed = getattr(pick, "executed_grasp", None)
    if outcome is None and executed is None:
        return None
    md: dict[str, Any] = {}
    if outcome is not None:
        md["outcome"] = json_safe(getattr(outcome, "value", outcome))
    if executed is not None:
        eg = grasp_metadata_from(executed)
        if eg is not None:
            md["executed_grasp"] = eg
    return md or None


def recovery_metadata_from(report: Any) -> Optional[dict[str, Any]]:
    """Summarise a :class:`SceneRecoveryReport`."""

    if report is None:
        return None
    plan = getattr(report, "plan", None)
    plan_dict: Optional[dict[str, Any]] = None
    if plan is not None:
        plan_dict = {
            "action": json_safe(getattr(plan, "action", None)),
            "reason": getattr(plan, "reason", ""),
            "nudge_offset_mm": json_safe(
                getattr(plan, "nudge_offset_mm", None)
            ),
            "agitate_amplitude_mm": json_safe(
                getattr(plan, "agitate_amplitude_mm", 0.0)
            ),
            "telemetry": json_safe(getattr(plan, "telemetry", {}) or {}),
        }
    return {
        "plan": plan_dict,
        "executed": bool(getattr(report, "executed", False)),
        "outcome": getattr(report, "outcome", ""),
        "telemetry": json_safe(getattr(report, "telemetry", {}) or {}),
    }


# ---------------------------------------------------------------------------
# The record itself
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GraspAttemptRecord:
    """A JSONL-safe summary of one autonomous grasp attempt.

    Every field is a plain Python primitive, a ``dict`` or ``list`` of primitives, or
    ``None``. There is no numpy array, dataclass or Enum member on the record itself, and
    the ``*_metadata_from`` helpers convert a typed report into the expected shape.

    Mandatory fields
    ----------------
    timestamp
        Epoch seconds, as a float. Omitted at construction, :meth:`new` stamps it from
        :func:`time.time`.
    attempt_id
        An operator-supplied string identifying the attempt. The recorder generates no
        id of its own, which would hide the correlation with an upstream system.
    mode
        The :class:`GraspMode` value, as a string. It is always present.
    final_outcome
        The top-level result string. The keys are stable and safe to filter on.

    Optional fields
    ---------------
    profile, frame, target, initial_grasp, initial_telemetry,
    refined_grasp, refinement, selected_grasp, execution, verification
        The per-stage summaries, each ``None`` where the stage did not run.
    recovery_actions
        An ordered list of recovery summaries, empty where no recovery was attempted.
    extra
        The free-form bag. A caller stashes anything the schema does not cover, and the
        values pass through :func:`json_safe` on serialisation.
    """

    timestamp: float
    attempt_id: str
    mode: str
    final_outcome: str
    profile: Optional[dict[str, Any]] = None
    frame: Optional[dict[str, Any]] = None
    target: Optional[dict[str, Any]] = None
    initial_grasp: Optional[dict[str, Any]] = None
    initial_telemetry: Mapping[str, Any] = field(default_factory=dict)
    refined_grasp: Optional[dict[str, Any]] = None
    refinement: Optional[dict[str, Any]] = None
    selected_grasp: Optional[dict[str, Any]] = None
    execution: Optional[dict[str, Any]] = None
    verification: Optional[dict[str, Any]] = None
    recovery_actions: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    extra: Mapping[str, Any] = field(default_factory=dict)

    SCHEMA_VERSION: ClassVar[int] = 1

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp, (int, float)):
            raise TypeError(
                f"timestamp must be a number; got {type(self.timestamp).__name__}"
            )
        if self.timestamp != self.timestamp or self.timestamp < 0.0:
            raise ValueError(
                f"timestamp must be a finite, non-negative epoch second; "
                f"got {self.timestamp!r}"
            )
        if not isinstance(self.attempt_id, str) or not self.attempt_id:
            raise ValueError("attempt_id must be a non-empty string")
        if not isinstance(self.mode, str) or not self.mode:
            raise ValueError("mode must be a non-empty string")
        if not isinstance(self.final_outcome, str) or not self.final_outcome:
            raise ValueError("final_outcome must be a non-empty string")

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------

    @classmethod
    def new(
        cls,
        *,
        attempt_id: str,
        mode: str,
        final_outcome: str,
        timestamp: Optional[float] = None,
        **kwargs: Any,
    ) -> "GraspAttemptRecord":
        """Construct a record, with ``timestamp`` defaulting to the wall clock."""

        ts = float(timestamp) if timestamp is not None else float(time.time())
        return cls(
            timestamp=ts,
            attempt_id=attempt_id,
            mode=mode,
            final_outcome=final_outcome,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """The JSON-safe ``dict`` representation, in a stable key order."""

        return {
            "schema_version": int(self.SCHEMA_VERSION),
            "timestamp": json_safe(self.timestamp),
            "attempt_id": self.attempt_id,
            "mode": self.mode,
            "final_outcome": self.final_outcome,
            "profile": json_safe(self.profile),
            "frame": json_safe(self.frame),
            "target": json_safe(self.target),
            "initial_grasp": json_safe(self.initial_grasp),
            "initial_telemetry": json_safe(self.initial_telemetry),
            "refined_grasp": json_safe(self.refined_grasp),
            "refinement": json_safe(self.refinement),
            "selected_grasp": json_safe(self.selected_grasp),
            "execution": json_safe(self.execution),
            "verification": json_safe(self.verification),
            "recovery_actions": [
                json_safe(a) for a in tuple(self.recovery_actions or ())
            ],
            "extra": json_safe(self.extra),
        }

    def to_json(self) -> str:
        """Single-line JSON, safe to append to a ``.jsonl`` file."""

        return json.dumps(
            self.to_dict(), separators=(",", ":"), sort_keys=False
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GraspAttemptRecord":
        """The reverse of :meth:`to_dict`. It ignores an unknown top-level key."""

        try:
            return cls(
                timestamp=float(data["timestamp"]),
                attempt_id=str(data["attempt_id"]),
                mode=str(data["mode"]),
                final_outcome=str(data["final_outcome"]),
                profile=_optional_dict(data.get("profile")),
                frame=_optional_dict(data.get("frame")),
                target=_optional_dict(data.get("target")),
                initial_grasp=_optional_dict(data.get("initial_grasp")),
                initial_telemetry=dict(data.get("initial_telemetry") or {}),
                refined_grasp=_optional_dict(data.get("refined_grasp")),
                refinement=_optional_dict(data.get("refinement")),
                selected_grasp=_optional_dict(data.get("selected_grasp")),
                execution=_optional_dict(data.get("execution")),
                verification=_optional_dict(data.get("verification")),
                recovery_actions=tuple(
                    dict(a) for a in (data.get("recovery_actions") or ())
                ),
                extra=dict(data.get("extra") or {}),
            )
        except KeyError as exc:
            raise ValueError(
                f"GraspAttemptRecord.from_dict missing key {exc}"
            ) from exc

    @classmethod
    def from_json(cls, line: str) -> "GraspAttemptRecord":
        """The reverse of :meth:`to_json`."""

        return cls.from_dict(json.loads(line))


def _optional_dict(value: Any) -> Optional[dict[str, Any]]:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(
            f"expected mapping or None; got {type(value).__name__}"
        )
    return dict(value)


# ---------------------------------------------------------------------------
# JSONL file helpers
# ---------------------------------------------------------------------------


def append_jsonl(
    record: GraspAttemptRecord | Mapping[str, Any],
    path: str | os.PathLike[str],
) -> Path:
    """Atomically append ``record`` as a single line to ``path``.

    It creates the parent directory where one does not exist and returns the resolved
    :class:`Path`. It appends in line-buffered text mode, so each record is one
    ``\\n``-terminated line, which is what downstream JSONL tooling expects.
    """

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(record, GraspAttemptRecord):
        payload = record.to_json()
    else:
        payload = json.dumps(
            json_safe(record), separators=(",", ":"), sort_keys=False
        )
    with p.open("a", encoding="utf-8") as fh:
        fh.write(payload)
        fh.write("\n")
    return p


def iter_jsonl(
    path: str | os.PathLike[str],
) -> Iterator[GraspAttemptRecord]:
    """Yield :class:`GraspAttemptRecord` instances from a JSONL file.

    A blank line is skipped. A malformed line raises ``ValueError``, wrapped from
    :meth:`from_dict` or :func:`json.loads`, and carrying the 1-indexed line number,
    because replay tooling needs the failure location rather than a silent skip.
    """

    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                yield GraspAttemptRecord.from_json(line)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"{p}: malformed record at line {line_no}: {exc}"
                ) from exc
