"""Where a dataset lands, in formats a human can open, and a run that survives being interrupted.

Three properties this file exists to guarantee:

* Lossless, inspectable files. RGB as PNG, depth as 16-bit PNG in millimetres, instance maps as
  16-bit PNG ids. Any tool opens them, nothing is quietly compressed away, and a person can look at
  a scene without writing code, which is how rendering bugs actually get caught.
* Scene-level resume. A path-traced run of tens of thousands of scenes takes days, and a crash at
  hour eighteen must not cost eighteen hours. Every finished scene is appended to an index, and a
  re-run skips what is already there. Because scenes are deterministic in ``(seed, index)``,
  resuming produces the same dataset, not a similar one.
* Rejections are recorded, not swallowed. A scene dropped for instability is written to the index
  with its reason, so the reject rate per family is a measurable property of the dataset instead of
  a silent gap in the numbering.

Depth in a 16-bit PNG holds 0 to 65535 mm at 1 mm resolution, which covers every distance this cell
works at. Zero means invalid, exactly as a real sensor reports it, so the noisy and the clean map
mark missing data the same way.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.utility.log_cfg import create_logger

from datagen.constants import DATAGEN_LOG_DIR, WRITER_LOG_FILE

__all__ = ["DEPTH_SCALE_MM", "RENDER_ERROR_BUDGET", "DatasetWriter", "SceneRecord",
           "decode_depth_png", "encode_depth_png"]

logger = create_logger("datagen.writer", WRITER_LOG_FILE, log_dir=DATAGEN_LOG_DIR)

#: One PNG unit = one millimetre. Stated as a constant because a reader who assumes metres gets a
#: dataset that is wrong by 1000x and looks plausible.
DEPTH_SCALE_MM = 1.0
_MAX_DEPTH_MM = 65535

#: How many recorded render failures a scene may accumulate before a resume stops re-attempting it.
#: Three, and each one is already `build._RENDER_ATTEMPTS` (3) renders, so a scene is given up on
#: after nine attempts across at least three separate runs.
#:
#: The number balances two facts. Isaac's colour buffer is intermittent, so a scene that failed can
#: render perfectly on a later run, which argues for retrying. A reproducibly-failing scene blocks an
#: unattended run forever, which argues for stopping. Nine attempts is well past the intermittent
#: case and well short of forever.
RENDER_ERROR_BUDGET = 3


@dataclass(frozen=True, slots=True)
class SceneRecord:
    """One line of the index: what happened to this scene, whether or not it produced images."""

    scene_id: str
    index: int
    family: str
    #: "ok" or a rejection reason ("unstable", "no_view_rendered", ...). Never empty.
    status: str
    views: tuple[str, ...] = ()
    view_outcomes: dict[str, str] | None = None
    objects: int = 0
    seconds: float = 0.0
    note: str = ""


def encode_depth_png(depth_mm: np.ndarray) -> np.ndarray:
    """Millimetre depth to uint16 for PNG. Out-of-range and non-finite values become 0 (invalid).

    Clipping rather than wrapping: a 70 m reading is wrong, and a wrapped 4 m reading is wrong and
    plausible, which is worse.
    """
    depth = np.asarray(depth_mm, dtype=np.float64)
    out = np.where(np.isfinite(depth) & (depth > 0.0) & (depth <= _MAX_DEPTH_MM), depth, 0.0)
    return np.round(out).astype(np.uint16)


def decode_depth_png(encoded: np.ndarray) -> np.ndarray:
    """uint16 PNG to millimetre depth as float, with 0 preserved as invalid."""
    return np.asarray(encoded, dtype=np.float64) * DEPTH_SCALE_MM


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {key: _to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


class DatasetWriter:
    """Owns one dataset directory. Constructing it creates the layout; nothing else writes paths."""

    def __init__(self, root: str | Path, name: str) -> None:
        self.root = Path(root) / name
        self.name = name
        self.scenes_dir = self.root / "scenes"
        self.index_path = self.root / "index.jsonl"
        self.scenes_dir.mkdir(parents=True, exist_ok=True)
        logger.info("dataset %s -> %s", self.name, self.root)

    # --- resume ------------------------------------------------------------
    def completed(self, *, render_error_budget: int = RENDER_ERROR_BUDGET) -> dict[str, SceneRecord]:
        """Every scene already in the index, by id, including the rejected ones but not the crashed.

        A rejection counts as completed on purpose: re-running a scene that was rejected for
        instability would reject it again on the same seed and the same physics, so retrying it is
        pure cost.

        A ``render_error`` is the opposite. It is not a verdict about the scene, it is a bug in the
        renderer, and the whole point of fixing the bug is to render that scene. Treating a crash as
        a finished scene would mean every fix silently skips exactly the scenes it repaired.

        But not forever. Retrying a crashed scene on every resume makes an unattended run
        non-terminating: `build` gives up after `_CONSECUTIVE_ERROR_LIMIT` scenes fail in a row, so
        three reproducibly-failing scenes sitting next to each other in the pending list abort
        every run at the same place, while the index grows more `render_error` rows per restart and
        the scene count stands still.

        So a scene is given up on after ``render_error_budget`` recorded failures, which is that many
        times ``_RENDER_ATTEMPTS`` actual renders. It is returned as its own failed record rather
        than silently dropped, so a summary still counts it as an error and nothing reads it as `ok`.
        Pass a larger budget (or `math.inf`) after fixing a renderer bug.
        """
        if not self.index_path.exists():
            return {}
        out: dict[str, SceneRecord] = {}
        failures: dict[str, int] = {}
        exhausted: dict[str, SceneRecord] = {}
        for line in self.index_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            record = SceneRecord(**payload)
            if record.status == "render_error":
                out.pop(record.scene_id, None)
                failures[record.scene_id] = failures.get(record.scene_id, 0) + 1
                if failures[record.scene_id] >= render_error_budget:
                    exhausted[record.scene_id] = record
                continue
            # A later success clears the tally. An intermittent scene that finally rendered is not
            # exhausted, and leaving it in `exhausted` would then hide a real record behind a stale
            # failure count.
            failures.pop(record.scene_id, None)
            exhausted.pop(record.scene_id, None)
            out[record.scene_id] = record
        for scene_id, record in exhausted.items():
            if scene_id not in out:
                logger.warning("giving up on %s: %d render_error(s) recorded, budget %d",
                               scene_id, failures[scene_id], render_error_budget)
                out[scene_id] = record
        # The resume decision, at the moment it is made. A run that "did nothing" because every scene
        # was already in the index and a run that found an empty index look identical from outside.
        logger.info("resume: %d scene(s) already complete in %s", len(out), self.index_path)
        return out

    def previously_failed(self) -> set[str]:
        """Scene ids with a recorded ``render_error`` that have not since rendered.

        Exists so `build` can tell a known-bad backlog from a broken session. Its
        `_CONSECUTIVE_ERROR_LIMIT` guard means "three in a row means Isaac is broken, stop wasting
        the night", and that reading is wrong for a scene that already failed on a previous run,
        because its failure is evidence about the scene, not about this session.

        It matters because of where those scenes sit. A resume's pending list is every index not yet
        complete, in order, so the previously-failed ones come first, ahead of every scene that has
        never been attempted. Without this, a handful of them at the head of the list aborts the run
        before it reaches anything new, on every restart.
        """
        if not self.index_path.exists():
            return set()
        failed: set[str] = set()
        for line in self.index_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("status") == "render_error":
                failed.add(str(payload["scene_id"]))
            else:
                failed.discard(str(payload["scene_id"]))
        return failed

    def append_index(self, record: SceneRecord) -> None:
        """Append one scene to the index and flush, so an interrupted run loses at most one scene."""
        with self.index_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_to_jsonable(record), sort_keys=True) + "\n")
            handle.flush()
        # One line per scene, which is minutes of path tracing apart, so not a tight loop, and it is
        # the only per-scene progress that survives the process. A rejection is logged as a warning
        # rather than a count nobody reads: the reject rate per family is a property of the dataset.
        if record.status == "ok":
            logger.info("scene %s (%s): ok, %d object(s), %d view(s), %.1f s",
                        record.scene_id, record.family, record.objects, len(record.views),
                        record.seconds)
        else:
            logger.warning("scene %s (%s): %s after %.1f s; %s",
                           record.scene_id, record.family, record.status, record.seconds,
                           record.note or "no reason recorded")

    # --- per-scene artefacts ----------------------------------------------
    def scene_dir(self, scene_id: str) -> Path:
        path = self.scenes_dir / scene_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_json(self, scene_id: str, name: str, payload: Any) -> Path:
        path = self.scene_dir(scene_id) / name
        path.write_text(json.dumps(_to_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
        return path

    def write_png(self, scene_id: str, name: str, image: np.ndarray) -> Path:
        """Write a PNG (uint8 RGB or uint16 single channel). Imported lazily: only rendering needs it."""
        import cv2  # type: ignore[import-not-found]

        path = self.scene_dir(scene_id) / name
        array = np.asarray(image)
        if array.ndim == 3 and array.shape[2] == 3:
            array = array[..., ::-1]  # RGB in, BGR for cv2
        if not cv2.imwrite(str(path), array):
            raise OSError(f"failed to write {path}")
        return path

    # --- dataset-level artefacts ------------------------------------------
    def write_provenance(self, stamp: Any) -> Path:
        path = self.root / "provenance.json"
        text = stamp.to_json() if hasattr(stamp, "to_json") else json.dumps(_to_jsonable(stamp), indent=2)
        path.write_text(text, encoding="utf-8")
        logger.info("wrote %s (%d bytes)", path, len(text.encode("utf-8")))
        return path

    def write_attribution(self, text: str) -> Path:
        path = self.root / "ATTRIBUTION"
        path.write_text(text, encoding="utf-8")
        logger.info("wrote %s (%d bytes)", path, len(text.encode("utf-8")))
        return path

    def summary(self) -> dict[str, Any]:
        """Counts by status and by family: the first thing anyone asks of a finished run."""
        records = list(self.completed().values())
        by_status: dict[str, int] = {}
        by_family: dict[str, dict[str, int]] = {}
        for record in records:
            by_status[record.status] = by_status.get(record.status, 0) + 1
            family = by_family.setdefault(record.family, {})
            family[record.status] = family.get(record.status, 0) + 1
        return {
            "dataset": self.name,
            "scenes_in_index": len(records),
            "by_status": dict(sorted(by_status.items())),
            "by_family": {key: dict(sorted(value.items())) for key, value in sorted(by_family.items())},
        }
