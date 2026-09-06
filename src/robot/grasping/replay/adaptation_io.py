"""Apply-mode I/O: overlay sidecar + audit JSONL.

The filesystem side lives here so :mod:`adaptation` stays pure with no
filesystem side effects. Apply / rollback / verify CLI surfaces compose
``adaptation`` (planning + validation) with this module (writing
overlay + audit).
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Iterator, Mapping

import yaml

from src.robot.grasping.constants import (
    REPLAY_ADAPTATION_IO_LOG_FILE,
    create_grasping_logger,
)
from src.robot.grasping.replay.adaptation import (
    AdaptationPlan,
    plan_to_overlay_mapping,
)

#: The overlay this module writes is merged into ``robot.*`` at the next startup,
#: where, by design, it becomes indistinguishable from a hand-edited
#: ``robot.yaml`` value. The audit JSONL is the only provenance; this log is its
#: read-side twin, so "who changed that key" survives a lost audit file too.
logger = create_grasping_logger("AdaptationIO", REPLAY_ADAPTATION_IO_LOG_FILE)


#: Bump on any contract-level change to audit entry layout.
AUDIT_SCHEMA_VERSION: Final[int] = 1

#: Default repo-relative audit path. Callers may override it.
DEFAULT_AUDIT_PATH: Final[str] = "logs/adaptation/adaptation_audit.jsonl"

#: Default repo-relative overlay sidecar location. The active overlay
#: lives at ``DEFAULT_ACTIVE_OVERLAY_NAME``; per-plan archives live
#: alongside as ``adaptation_<plan_id>.yaml``.
DEFAULT_OVERLAY_DIR: Final[str] = "configs/overlays"
DEFAULT_ACTIVE_OVERLAY_NAME: Final[str] = "adaptation_active.yaml"


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _atomic_write_text(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` atomically via ``os.replace``."""

    _ensure_parent(path)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def overlay_sidecar_path(overlay_dir: Path, plan: AdaptationPlan) -> Path:
    return overlay_dir / f"adaptation_{plan.plan_id}.yaml"


def active_overlay_path(overlay_dir: Path) -> Path:
    return overlay_dir / DEFAULT_ACTIVE_OVERLAY_NAME


def write_overlay_sidecar(plan: AdaptationPlan, overlay_dir: Path) -> Path:
    """Write the per-plan overlay YAML, refresh the active pointer, return the sidecar path."""

    mapping = plan_to_overlay_mapping(plan)
    payload = yaml.safe_dump(
        mapping, sort_keys=True, default_flow_style=False, allow_unicode=True
    )
    sidecar = overlay_sidecar_path(overlay_dir, plan)
    _atomic_write_text(sidecar, payload)
    # A copy (not a symlink) for portability across filesystems and CI.
    _atomic_write_text(active_overlay_path(overlay_dir), payload)
    logger.info(
        "Wrote overlay for plan %s to %s (%d bytes, %d key(s)) and refreshed %s",
        plan.plan_id,
        sidecar,
        len(payload.encode("utf-8")),
        len(plan.changes),
        active_overlay_path(overlay_dir),
    )
    return sidecar


def clear_active_overlay(overlay_dir: Path) -> None:
    """Remove the active overlay pointer (sidecar archive is kept)."""

    path = active_overlay_path(overlay_dir)
    if path.exists():
        path.unlink()
        logger.info("Cleared the active overlay at %s (archive kept)", path)
    else:
        # Not an error: a rollback that found nothing to clear means the
        # config it was meant to revert was never active.
        logger.warning("No active overlay to clear at %s", path)


@dataclass(frozen=True, slots=True)
class AuditEntry:
    """A single append-only audit record."""

    schema_version: int
    plan_id: str
    timestamp_ns: int
    mode: str
    strategy: str
    action: str  # 'plan' | 'verify' | 'apply' | 'rollback'
    applied: bool
    source_baseline_sha: str | None
    source_taxonomy_sha: str | None
    rollback_of: str | None
    changes: tuple[Mapping[str, Any], ...]
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "timestamp_ns": self.timestamp_ns,
            "mode": self.mode,
            "strategy": self.strategy,
            "action": self.action,
            "applied": self.applied,
            "source_baseline_sha": self.source_baseline_sha,
            "source_taxonomy_sha": self.source_taxonomy_sha,
            "rollback_of": self.rollback_of,
            "changes": [dict(c) for c in self.changes],
            "notes": self.notes,
        }


def build_audit_entry(
    plan: AdaptationPlan,
    *,
    action: str,
    applied: bool,
    rollback_of: str | None = None,
    notes: str = "",
) -> AuditEntry:
    if action not in {"plan", "verify", "apply", "rollback"}:
        raise ValueError(
            f"audit action {action!r} not in "
            "{'plan', 'verify', 'apply', 'rollback'}"
        )
    return AuditEntry(
        schema_version=AUDIT_SCHEMA_VERSION,
        plan_id=plan.plan_id,
        timestamp_ns=plan.created_at_ns,
        mode=plan.mode,
        strategy=plan.strategy,
        action=action,
        applied=applied,
        source_baseline_sha=plan.source_baseline_sha,
        source_taxonomy_sha=plan.source_taxonomy_sha,
        rollback_of=rollback_of,
        changes=tuple(c.to_dict() for c in plan.changes),
        notes=notes,
    )


def append_audit_entry(entry: AuditEntry, audit_path: Path) -> None:
    """Append a single JSON-line entry. Append-only; never rewrites."""

    _ensure_parent(audit_path)
    line = json.dumps(entry.to_dict(), sort_keys=True, separators=(",", ":"))
    with audit_path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    logger.info(
        "Audit %s: plan %s (mode %s, strategy %s, applied=%s, %d change(s)) -> %s",
        entry.action,
        entry.plan_id,
        entry.mode,
        entry.strategy,
        entry.applied,
        len(entry.changes),
        audit_path,
    )


def iter_audit_entries(audit_path: Path) -> Iterator[dict[str, Any]]:
    if not audit_path.exists():
        return
    with audit_path.open("r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            yield json.loads(ln)


def load_audit_entries(audit_path: Path) -> tuple[dict[str, Any], ...]:
    return tuple(iter_audit_entries(audit_path))


def find_plan_in_audit(
    plan_id: str, audit_path: Path
) -> dict[str, Any] | None:
    """Return the most recent ``action='apply'`` entry for ``plan_id``."""

    latest: dict[str, Any] | None = None
    for entry in iter_audit_entries(audit_path):
        if entry.get("plan_id") == plan_id and entry.get("action") == "apply":
            latest = entry
    return latest


__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "AuditEntry",
    "DEFAULT_ACTIVE_OVERLAY_NAME",
    "DEFAULT_AUDIT_PATH",
    "DEFAULT_OVERLAY_DIR",
    "active_overlay_path",
    "append_audit_entry",
    "build_audit_entry",
    "clear_active_overlay",
    "find_plan_in_audit",
    "iter_audit_entries",
    "load_audit_entries",
    "overlay_sidecar_path",
    "write_overlay_sidecar",
]
