"""Deterministic replay environments.

Two replay environments:

* :class:`RecordedObservationReplayEnv` yields the SAR sequence
  derived from recorded outcome records as they stand. This is the
  cheap, deterministic, byte-identical replay that offline policy
  evaluation sits on.
* :class:`GeometricRerunReplayEnv` re-projects each record through
  the SAR extractor without re-invoking the live grasping pipeline.
  There is no rigid-body simulation here: the projection is
  deterministic and the interface is frozen.

Both environments are read-only: they never mutate inputs and never
call into runtime safety/grasping code paths. Behavior in
``geometry_only`` and ``hybrid_ml`` modes is unaffected.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

from .sar import SAR, SARExtractor, BaselineSARExtractor


@dataclass
class ReplayEnvBase:
    """Common contract for the replay environments."""

    name: str
    extractor: SARExtractor

    def iter_sar(self, records: Sequence[Mapping[str, Any]]) -> Iterator[SAR]:
        for record in records:
            yield self._project(record)

    def _project(self, record: Mapping[str, Any]) -> SAR:  # pragma: no cover (abstract)
        raise NotImplementedError

    def fingerprint(self, records: Sequence[Mapping[str, Any]]) -> str:
        """sha256 of the SAR sequence; identical across runs for identical records."""

        h = hashlib.sha256()
        for sar in self.iter_sar(records):
            payload = json.dumps(sar.to_json(), sort_keys=True, separators=(",", ":"))
            h.update(payload.encode("utf-8"))
            h.update(b"\n")
        return h.hexdigest()


@dataclass
class RecordedObservationReplayEnv(ReplayEnvBase):
    """Replay observations exactly as they were recorded."""

    name: str = "recorded_observation"
    extractor: SARExtractor = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.extractor is None:
            self.extractor = BaselineSARExtractor()

    def _project(self, record: Mapping[str, Any]) -> SAR:
        return self.extractor.extract(record)


@dataclass
class GeometricRerunReplayEnv(ReplayEnvBase):
    """Deterministic projection mirroring a geometric rerun.

    Carries a distinct env ``name`` (``geometric_rerun``), not a
    distinct ``extractor`` tag: both envs default to
    :class:`BaselineSARExtractor`, so ``sar.extractor`` reads the same
    value in either and ``fingerprint()`` over the same records is
    identical. Overriding :meth:`_project` attaches a real geometric
    rerun while the interface and its determinism stay frozen.
    """

    name: str = "geometric_rerun"
    extractor: SARExtractor = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.extractor is None:
            self.extractor = BaselineSARExtractor()

    def _project(self, record: Mapping[str, Any]) -> SAR:
        return self.extractor.extract(record)


def build_default_envs() -> tuple[RecordedObservationReplayEnv, GeometricRerunReplayEnv]:
    """Both replay envs, each with the baseline extractor."""

    return RecordedObservationReplayEnv(), GeometricRerunReplayEnv()


__all__ = (
    "GeometricRerunReplayEnv",
    "RecordedObservationReplayEnv",
    "ReplayEnvBase",
    "build_default_envs",
)
