"""Measuring the one drift the byte-identity family is allowed to skip for.

The old shape of this decision was a 27-entry node-id allow-list in ``tests/conftest.py`` gated on
an environment variable nobody set, which meant "skip always". The shape here is a measurement: a
canonical pack is regenerated in memory and compared line by line against the committed file, and
the verdict decides.

* ``identical``   the goldens reproduce here; nothing is skipped.
* ``float_only``  every differing line differs ONLY in the decimal spelling of the same quantity
                  (relative gap below :data:`_ULP_TOLERANCE`). MEASURED 2026-09-10 on the Windows
                  RTX 5080 box: 42 lines of 1240, 51 differing leaves across four fields
                  (``decision_latency_ms`` 21, ``ranking_latency_ms`` 17, ``attempt_wall_time_s``
                  8, ``fusion_latency_ms`` 5), every one of them drawn by ``random.gauss``, which
                  reaches libm through ``math.log`` and ``math.cos``. Example:
                  ``27.840233759192085`` on disk against ``27.84023375919208`` here. Two
                  byte-comparison tests cannot hold under that.
* ``structural``  anything else. NOT skipped: a changed outcome, a changed field, a changed record
                  count or a float that moved by more than a re-spelling is the regression the
                  family exists to catch, and it must fail loudly on every box.

So the guard fires everywhere. Off the origin platform it fires and reports which tests it had
to stand down for and why; a real regression fails the suite on the same box that used to print
"27 skipped".
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterator, Literal

#: Three tests assert byte-identity of a payload whose floats come out of ``random.gauss`` or a
#: logistic fit. Everything else in the old 27-entry list either already passed, or failed for a
#: reason with a fix (CRLF in the working tree, a stale provenance hash, stale prose in a golden)
#: and was repaired rather than skipped.
#:
#: MEASURED 2026-09-10 in this tree: only the two canonical-pack entries actually stand down here.
#: The ranking artifact was re-blessed on this box in the same pass (its committed prose could no
#: longer be produced by its own generator), so its probe reports ``identical`` and the test runs.
#: It keeps its entry because the fit still lands on a different libm elsewhere.
#:
#: Each entry names the PROBE that has to agree before it stands down, so no test is skipped on
#: another test's evidence. A proxy would be the old mistake in a smaller costume: a box where the
#: packs happen to reproduce but the logistic fit does not would silently skip the wrong one.
PLATFORM_FLOAT_LOCKED_NODEIDS: dict[str, str] = {
    # 42 lines of 1240; every one a `decision_latency_ms` or `ranking_latency_ms`.
    "tests/test_canonical_determinism.py::CanonicalPackDeterminismTests"
    "::test_each_pack_regenerates_to_committed_bytes": "canonical_packs",
    "tests/test_u0_telemetry_contract.py::CanonicalPacksTests"
    "::test_pack_bytes_match_spec_render": "canonical_packs",
    # 3 of 63 pairwise-logistic weights move from box to box; measured here before the re-bless,
    # -2.3114748262842606e-05 committed against -2.3114748262843775e-05 fresh. A CLI writes the
    # file and the test hashes it, so there is no room for a tolerance the way there is in the
    # promotion-report comparison. The substance of the claim is kept by
    # `test_cli_regenerates_the_same_numbers`, which runs everywhere.
    "tests/test_ranking_shadow_and_pairwise_logistic.py::CommittedArtifactSha256Tests"
    "::test_cli_regenerates_byte_identical_artifact": "ranking_artifact",
}

#: Paths whose FILE BYTES are hashed by a committed golden, so a CRLF checkout breaks them.
#: ``docs/baselines/**`` and ``assets/models/**`` were already pinned; the replay packs were the
#: hole, and the manifest inside them stores the sha256 of its own siblings.
LF_LOCKED_GOLDEN_GLOBS: tuple[str, ...] = ("tests/data/replay/**",)

#: Relative gap under which two decimal spellings are treated as the same quantity. A double
#: carries ~2.2e-16 of relative precision; a handful of libm operations composed together stays
#: within a few of those. Anything larger is a different number, not a different spelling.
_ULP_TOLERANCE: float = 1e-12


def repo_root() -> Path:
    """Return the repository root (this file lives in ``<root>/tests``)."""

    return Path(__file__).resolve().parents[1]


def iter_lf_locked_goldens() -> Iterator[Path]:
    """Yield every tracked file under the LF-locked globs."""

    root = repo_root()
    for glob in LF_LOCKED_GOLDEN_GLOBS:
        base = root / glob.replace("/**", "")
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file():
                yield path


DriftKind = Literal["identical", "float_only", "structural"]


@dataclass(frozen=True, slots=True)
class PackDriftVerdict:
    """What a regeneration of one committed golden looks like against its bytes on disk."""

    kind: DriftKind
    differing_lines: int
    total_lines: int
    example: str
    subject: str = "canonical packs"

    def render(self) -> str:
        """Describe the verdict to a person as one ASCII line, no trailing newline."""

        if self.kind == "identical":
            return (
                f"{self.subject} reproduce byte-identically here "
                f"({self.total_lines} lines compared)"
            )
        return (
            f"{self.subject} drift {self.kind} on this platform: "
            f"{self.differing_lines}/{self.total_lines} lines; {self.example}"
        )


def _numbers_match(a: object, b: object) -> bool:
    """True when two JSON scalars are the same quantity, allowing a re-spelling of a float."""

    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a == b:
            return True
        left, right = float(a), float(b)
        if not (math.isfinite(left) and math.isfinite(right)):
            return False
        scale = max(abs(left), abs(right), 1.0)
        return abs(left - right) <= _ULP_TOLERANCE * scale
    return bool(a == b)


def _flatten(obj: object, prefix: str = "") -> dict[str, object]:
    """Flatten a decoded JSON record to ``dotted.path -> scalar``."""

    if isinstance(obj, dict):
        out: dict[str, object] = {}
        for key, value in obj.items():
            out.update(_flatten(value, f"{prefix}{key}."))
        return out
    if isinstance(obj, list):
        out = {}
        for index, value in enumerate(obj):
            out.update(_flatten(value, f"{prefix}{index}."))
        return out
    return {prefix.rstrip("."): obj}


def _drift_between_lines(on_disk: str, fresh: str) -> DriftKind:
    """Classify the difference between two JSON documents that are not byte-equal."""

    if on_disk == fresh:
        return "identical"
    try:
        left = _flatten(json.loads(on_disk))
        right = _flatten(json.loads(fresh))
    except json.JSONDecodeError:
        return "structural"
    if left.keys() != right.keys():
        return "structural"
    respelled = False
    for key, value in left.items():
        other = right[key]
        if value == other:
            continue
        if not _numbers_match(value, other):
            return "structural"
        respelled = True
    if not respelled:
        # Every value is exactly equal, yet the bytes are not. That is key order, indent, escaping
        # or a trailing newline having moved, which is precisely what a byte-identity test is for
        # and has nothing to do with the platform. It gets no stand-down.
        return "structural"
    return "float_only"


@lru_cache(maxsize=1)
def classify_canonical_pack_drift() -> PackDriftVerdict:
    """Regenerate every canonical pack in memory and classify the drift against the committed bytes.

    Cached: the packs take a couple of seconds to render and the answer cannot change inside a run.
    Nothing is written to disk here; the comparison reads the committed files in text mode, which
    also means a CRLF working tree cannot masquerade as drift.
    """

    from src.robot.grasping.replay.canonical_datasets import (
        CANONICAL_PACKS,
        render_pack_jsonl,
    )

    root = repo_root()
    total = 0
    differing = 0
    kind: DriftKind = "identical"
    example = ""
    for pack in CANONICAL_PACKS:
        path = root / pack.relative_path
        if not path.exists():
            return PackDriftVerdict("structural", 0, 0, f"missing pack {pack.relative_path}")
        disk_lines = path.read_text(encoding="utf-8").splitlines()
        fresh_lines = render_pack_jsonl(pack).splitlines()
        total += max(len(disk_lines), len(fresh_lines))
        if len(disk_lines) != len(fresh_lines):
            return PackDriftVerdict(
                "structural",
                abs(len(disk_lines) - len(fresh_lines)),
                total,
                f"{pack.name}: {len(disk_lines)} committed lines vs {len(fresh_lines)} fresh",
            )
        for index, (left, right) in enumerate(zip(disk_lines, fresh_lines)):
            line_kind = _drift_between_lines(left, right)
            if line_kind == "identical":
                continue
            differing += 1
            if line_kind == "structural":
                return PackDriftVerdict(
                    "structural",
                    differing,
                    total,
                    f"{pack.name} line {index}: committed {left[:120]!r} vs fresh {right[:120]!r}",
                )
            if not example:
                example = f"first at {pack.name} line {index}"
            kind = "float_only"
    return PackDriftVerdict(kind, differing, total, example)


#: Digests of the LF-locked goldens as they were when the session started, filled in by
#: ``pytest_configure`` before a single test has run. MEASURED 2026-09-10: a test called
#: ``regenerate_all(REPO_ROOT)`` and rewrote five committed files mid-run, and the tests that ran
#: after it silently compared against the rewrite. Comparing against a start-of-session snapshot
#: catches exactly that, and stays quiet about an edit a person made before the run.
_SESSION_GOLDEN_DIGESTS: dict[str, str] = {}


def snapshot_lf_locked_goldens() -> None:
    """Record the goldens' digests at session start. Idempotent within a session."""

    if _SESSION_GOLDEN_DIGESTS:
        return
    root = repo_root()
    for path in iter_lf_locked_goldens():
        key = str(path.relative_to(root)).replace("\\", "/")
        _SESSION_GOLDEN_DIGESTS[key] = hashlib.sha256(path.read_bytes()).hexdigest()


def goldens_rewritten_during_session() -> list[str]:
    """Return the LF-locked goldens whose bytes changed since the session started."""

    root = repo_root()
    changed: list[str] = []
    for key, digest in sorted(_SESSION_GOLDEN_DIGESTS.items()):
        path = root / key
        if not path.exists():
            changed.append(f"{key} (deleted)")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            changed.append(key)
    return changed


@lru_cache(maxsize=1)
def classify_ranking_artifact_drift() -> PackDriftVerdict:
    """Retrain the ranking policy through its own CLI and classify the drift against the committed file.

    Its own probe rather than the canonical-pack one: the weights come out of a logistic fit, not
    out of ``random.gauss``, and a box that reproduces one need not reproduce the other. Cached,
    because it costs a subprocess.
    """

    import subprocess
    import sys
    import tempfile

    root = repo_root()
    committed_path = root / "docs/baselines/rl_policies/v3_ranking_baseline_v1.json"
    subject = "the committed ranking artifact"
    if not committed_path.exists():
        return PackDriftVerdict(
            "structural", 0, 0, f"missing artifact {committed_path.name}", subject
        )
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "ranking.json"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.robot.grasping.rl",
                "train-ranking-policy",
                "--dataset-id",
                "v1_bootstrap",
                "--output",
                str(out_path),
            ],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0 or not out_path.exists():
            # The trainer refusing is not a licence to skip: let the test run and say so itself.
            return PackDriftVerdict(
                "structural", 0, 0, "train-ranking-policy did not produce a file", subject
            )
        fresh_bytes = out_path.read_bytes()
        fresh_text = out_path.read_text(encoding="utf-8")
    committed_bytes = committed_path.read_bytes()
    committed_text = committed_path.read_text(encoding="utf-8")
    if committed_bytes == fresh_bytes:
        return PackDriftVerdict("identical", 0, 1, "", subject)
    # The identical check is on BYTES, deliberately. `read_text` folds CRLF to LF, so comparing
    # text would report "identical" for a file that differs only in line ending, and then this
    # probe would wave a byte-identity test through onto a difference it exists to catch.
    kind = _drift_between_lines(committed_text, fresh_text)
    if kind == "identical":
        kind = "structural"
    return PackDriftVerdict(
        kind,
        1,
        1,
        f"{committed_path.name} regenerates with {kind} drift",
        subject,
    )


#: Probe name -> the measurement that has to say "float_only" before its tests stand down.
DRIFT_PROBES: dict[str, "Callable[[], PackDriftVerdict]"] = {
    "canonical_packs": classify_canonical_pack_drift,
    "ranking_artifact": classify_ranking_artifact_drift,
}
