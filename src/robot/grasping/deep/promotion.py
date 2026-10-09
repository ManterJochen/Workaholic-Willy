"""Whether a trained generator may drive a cell: the promotion record beside its artifact.

The owner's decision of 2026-10-09: no finished models ship, every customer trains their own, and so a
cell never grasps with a trained generator that has not passed its proof. The proof leaves a record
beside the weights, ``<stem>.promotion.json`` (``set_grasp_generator_v1.promotion.json`` beside
``set_grasp_generator_v1.pt``, as the card is ``set_grasp_generator_v1.card.json``), and
:func:`why_not_deployable` is its reader: the calculator factory asks it before it builds a deep
calculator for a cell, and ``deep inspect`` prints its answer first.

    kind               "set_grasp_generator_promotion", so another component's record is not read as one
    artifact_sha256    the SHA-256 of the artifact's bytes, so a record vouches for one file only
    verdict            "pass" or "fail"
    phase              "shadow", "ab" or "active": how far the proof has gone
    protocol           the name of the proof it went through, and its version (protocol_version)
    evidence           the proof's summary, a free mapping this module does not read
    promoted_by        who wrote the record
    promoted_at        when, in UTC

An artifact may drive a cell when its record is readable, of this kind, written for its bytes, says
"pass" and has reached "active", and when its own card does not say it is a smoke-tier or a control run,
which no record can make deployable. Anything else refuses, and so does a record or a card that cannot be
read: the gate fails closed.

Nothing writes a record yet except tests. The proof itself, ``deep judge`` and ``deep promote``, comes in
the week after 2026-10-09, so until then every artifact refuses a cell, and that is the point: the best
full run so far did not beat "straight down" on objects it never saw (the 2026-10-09 review). The
evaluation paths never ask this module. The ladder, ``deep propose``, the simulation runners and the
offline sweeps build or load an artifact to measure it, which is how its proof will be made.

The shape follows the success model's gate (``calibration/model_promotion.py``): a record beside the
artifact, a SHA-256 over its bytes, a verdict and a phase. What that gate has and this one does not yet is
the exam itself, frozen with plan-locked thresholds; ``deep judge`` brings it. Like that gate, this one
refuses an accident, a stale record or a smoke run or a hand-copied file, and not a forgery: every field
is written by whoever wrote the file.

Importing this costs no torch. The record is JSON and the hash is ``hashlib``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Mapping

__all__ = [
    "PHASES",
    "PROMOTION_KIND",
    "PROMOTION_SUFFIX",
    "VERDICTS",
    "GeneratorPromotion",
    "PromotionUnreadable",
    "artifact_sha256",
    "promotion_path",
    "read_promotion",
    "why_not_deployable",
]

#: What a generator's record says it is. Not the artifact's own kind and not the success model's
#: ``promotion.json``: a record written for another component is refused by name rather than half-read.
PROMOTION_KIND: Final[str] = "set_grasp_generator_promotion"

#: Beside the weights, under the artifact's own stem, as the card is.
PROMOTION_SUFFIX: Final[str] = ".promotion.json"

VERDICTS: Final[tuple[str, ...]] = ("pass", "fail")

#: How far a proof has gone, in order: beside the analytic generator, which decides (``shadow``); in a
#: randomised comparison with it (``ab``); the cell's generator (``active``). This build has no shadow
#: seam and no comparison harness for a generator yet, so a cell grasps with one at ``active`` alone.
PHASES: Final[tuple[str, ...]] = ("shadow", "ab", "active")

#: The one phase ``calculator: deep`` builds a cell at.
_ACTIVE: Final[str] = "active"


class PromotionUnreadable(ValueError):
    """The record beside an artifact cannot be read as a generator's promotion. The message names the file."""


@dataclass(frozen=True, slots=True)
class GeneratorPromotion:
    """One promotion record, as read from beside an artifact (the fields are the module docstring's)."""

    artifact_sha256: str
    verdict: str
    phase: str
    protocol: str
    protocol_version: int
    promoted_by: str
    promoted_at: str
    evidence: Mapping[str, Any] = field(default_factory=dict)
    kind: str = PROMOTION_KIND

    def to_dict(self) -> dict[str, Any]:
        """The record as JSON holds it; :meth:`from_dict` reads it back."""
        return {
            "kind": self.kind,
            "artifact_sha256": self.artifact_sha256,
            "verdict": self.verdict,
            "phase": self.phase,
            "protocol": self.protocol,
            "protocol_version": self.protocol_version,
            "evidence": dict(self.evidence),
            "promoted_by": self.promoted_by,
            "promoted_at": self.promoted_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, source: str = "the record") -> "GeneratorPromotion":
        """Read one record, refusing with :class:`PromotionUnreadable` rather than guessing a field.

        ``source`` names the file in the refusal. Every field is required: a record that leaves out its
        phase or its protocol cannot say what it vouches for, and a default here would say it for it.
        """
        kind = payload.get("kind")
        if kind != PROMOTION_KIND:
            raise PromotionUnreadable(
                f"{source} is a record of another kind ({kind!r}; a generator's is {PROMOTION_KIND!r})")
        for key in ("artifact_sha256", "verdict", "phase", "protocol", "protocol_version", "evidence",
                    "promoted_by", "promoted_at"):
            if key not in payload:
                raise PromotionUnreadable(f"{source} lacks {key!r}")
        sha = payload["artifact_sha256"]
        if not (isinstance(sha, str) and len(sha) == 64 and all(c in "0123456789abcdefABCDEF" for c in sha)):
            raise PromotionUnreadable(f"{source} gives artifact_sha256 as {sha!r}, not 64 hex digits")
        verdict, phase = payload["verdict"], payload["phase"]
        if verdict not in VERDICTS:
            raise PromotionUnreadable(f"{source} gives verdict as {verdict!r}; a verdict is {' or '.join(VERDICTS)}")
        if phase not in PHASES:
            raise PromotionUnreadable(f"{source} gives phase as {phase!r}; the phases are {', '.join(PHASES)}")
        version = payload["protocol_version"]
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise PromotionUnreadable(f"{source} gives protocol_version as {version!r}, not a version number")
        evidence = payload["evidence"]
        if not isinstance(evidence, Mapping):
            raise PromotionUnreadable(f"{source} gives evidence as a {type(evidence).__name__}, not a mapping")
        for key in ("protocol", "promoted_by", "promoted_at"):
            if not (isinstance(payload[key], str) and payload[key].strip()):
                raise PromotionUnreadable(f"{source} gives {key} as {payload[key]!r}, not a name")
        return cls(artifact_sha256=sha.lower(), verdict=str(verdict), phase=str(phase),
                   protocol=str(payload["protocol"]), protocol_version=int(version),
                   promoted_by=str(payload["promoted_by"]), promoted_at=str(payload["promoted_at"]),
                   evidence=dict(evidence))


def promotion_path(artifact: str | Path) -> Path:
    """Where the record of ``artifact`` lives: ``<stem>.promotion.json`` beside it."""
    return Path(artifact).with_suffix(PROMOTION_SUFFIX)


def artifact_sha256(artifact: str | Path) -> str:
    """The SHA-256 of an artifact's bytes, lowercase hex, as a record vouches for it.

    The bytes and nothing else, so a record survives a copy to the cell's machine and refuses a file
    retrained, re-saved or swapped since its proof.
    """
    digest = hashlib.sha256()
    with Path(artifact).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_promotion(artifact: str | Path) -> GeneratorPromotion:
    """The record beside ``artifact``.

    Raises ``FileNotFoundError`` where there is none, and :class:`PromotionUnreadable` where the file
    cannot be read as a generator's promotion: not JSON, not an object, another kind, a missing or a
    malformed field.
    """
    record = promotion_path(artifact)
    try:
        text = record.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise
    except (OSError, UnicodeDecodeError) as exc:
        raise PromotionUnreadable(f"{record.name} cannot be opened ({exc})") from exc
    try:
        payload = json.loads(text)
    except ValueError as exc:
        raise PromotionUnreadable(f"{record.name} is not JSON ({exc})") from exc
    if not isinstance(payload, dict):
        raise PromotionUnreadable(f"{record.name} holds a {type(payload).__name__}, not a record")
    return GeneratorPromotion.from_dict(payload, source=record.name)


def _what_its_card_says(artifact: Path) -> str:
    """Why the artifact's own card rules it out of a cell, or "" where it does not.

    A smoke-tier run proves that the chain closes and says nothing about grasp quality, and a control run
    is fitted to synthetic labels derived from an input channel. Neither is a model any proof can pass,
    so the card is read before the record: a record that vouches for one is an accident, and the card is
    the better sentence for its owner. A card that is not there says nothing either way, since the record
    vouches for the bytes; a card that is there and cannot be read refuses, because what wrote the
    artifact is then unknown.
    """
    card = artifact.with_suffix(".card.json")
    if not card.is_file():
        return ""
    try:
        payload = json.loads(card.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return f"{artifact.name} has a card that cannot be read ({card.name}: {exc})"
    report = payload.get("report", {}) if isinstance(payload, dict) else None
    if not isinstance(report, dict):
        return f"{artifact.name} has a card that cannot be read ({card.name} holds no report a run writes)"
    if report.get("tier") == "smoke":
        return (f"{artifact.name} is a smoke-tier run by its own card ({card.name}), which proves the chain "
                f"closes and says nothing about grasp quality")
    if report.get("control"):
        return (f"{artifact.name} is a control run by its own card ({card.name}, control "
                f"{report['control']!r}), fitted to synthetic labels derived from an input channel")
    return ""


def why_not_deployable(artifact: str | Path) -> str:
    """Why ``artifact`` may NOT drive a cell, as one clause that starts with its file name; "" where it may.

    Answered in this order, the first that holds:

    1. its card says it is a smoke-tier or a control run, or the card cannot be read;
    2. no record beside it, or a record that cannot be read, or one of another kind;
    3. a record written for other bytes than the file's;
    4. a record whose proof failed;
    5. a record whose phase is below ``active``.

    Never raises: a file that cannot be hashed is a reason too. ``build_calculator`` turns a reason into
    its refusal, and ``deep inspect`` prints it after ``NOT DEPLOYABLE:``.
    """
    path = Path(artifact)
    name = path.name
    if not path.is_file():
        return f"there is no artifact at {path}"
    card_says = _what_its_card_says(path)
    if card_says:
        return card_says
    try:
        record = read_promotion(path)
    except FileNotFoundError:
        return f"{name} carries no promotion"
    except PromotionUnreadable as exc:
        return f"{name} carries a promotion record that cannot be read ({exc})"
    try:
        actual = artifact_sha256(path)
    except OSError as exc:
        return f"{name} cannot be read to check its promotion ({exc})"
    if actual != record.artifact_sha256:
        return (f"{name} carries a promotion written for other bytes ({promotion_path(path).name} vouches for "
                f"sha256 {record.artifact_sha256[:12]}, the file hashes to {actual[:12]}: it changed after its "
                f"proof, or the record is another file's)")
    under = f"{record.protocol} v{record.protocol_version}, {record.promoted_at}"
    if record.verdict != "pass":
        return f"{name} carries a promotion whose proof failed (verdict {record.verdict!r} under {under})"
    if record.phase != _ACTIVE:
        return (f"{name} is promoted only to phase {record.phase!r} (under {under}), and a cell grasps with a "
                f"trained generator at {_ACTIVE!r} alone")
    return ""
