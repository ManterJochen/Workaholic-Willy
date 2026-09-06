"""What a dataset has to be able to say about itself, or two runs that differ are indistinguishable.

A generated dataset is a measurement instrument, and an instrument with no serial number is an
opinion. The stamp records the four things that determine what came out: the config, the code, the
assets and the renderer. A number measured on a dataset can then be traced to what produced it.

Deliberately not a hash of the outputs: that says two datasets differ, which is easy to see anyway.
The useful question is which input changed, and that is what these four fields answer.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from typing import Any

from src.utility.log_cfg import create_logger

from datagen.constants import DATAGEN_LOG_DIR, PROVENANCE_LOG_FILE

__all__ = ["ProvenanceStamp", "provenance_stamp"]

logger = create_logger("datagen.provenance", PROVENANCE_LOG_FILE, log_dir=DATAGEN_LOG_DIR)


@dataclass(frozen=True, slots=True)
class ProvenanceStamp:
    """The record that travels with a dataset."""

    dataset_name: str
    created_utc: str
    #: Full generator config, so the dataset can be rebuilt without guessing which flags were used.
    config: dict[str, Any]
    config_sha256: str
    #: Repository head, and whether the tree was dirty. A dataset built from uncommitted code is a
    #: valid thing to do and an invalid thing to hide.
    code_commit: str
    code_dirty: bool
    #: Content hash over the asset manifest rows: the same assets under the same licences.
    asset_manifest_sha256: str
    renderer: str
    platform: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


#: Public, and named without a leading underscore because `eval/ladder.py` and `prompts/build.py`
#: import it across a package boundary. A leading underscore is a claim that nothing outside uses
#: the name, no export sweep can see such a crossing, and a refactor renames a private name freely.
def sha256_of(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _git_state() -> tuple[str, bool]:
    """``(commit, dirty)``; ``("unknown", True)`` when git cannot answer.

    Unknown counts as dirty on purpose: the honest default for a question that cannot be answered is
    the one that makes a reader check, not the one that reassures them.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        return commit, bool(status)
    except Exception as exc:  # noqa: BLE001 (no git, no repo, or a timeout: report unknown, never crash a run)
        # Warning, not info: every dataset stamped from here says "dirty" without being dirty, and the
        # reason it says so is this line. Without it the stamp is unexplainable months later.
        logger.warning("git could not be read (%s: %s); stamping commit=unknown, dirty=True",
                       type(exc).__name__, exc)
        return "unknown", True


def provenance_stamp(
    *,
    dataset_name: str,
    config: Any,
    asset_rows: Any,
    renderer: str,
    created_utc: str,
) -> ProvenanceStamp:
    """Build the stamp. ``created_utc`` is passed in rather than read, so callers stay deterministic."""
    config_dict = config.model_dump() if hasattr(config, "model_dump") else dict(config)
    commit, dirty = _git_state()
    # The code half of the stamp, said out loud at the moment it is decided. The config is not logged:
    # it is a full dump and it already travels inside the stamp. ``asset_rows`` is not counted here
    # either: it may be an iterator, and consuming it to write a log line would empty the hash below.
    logger.info("stamping %s: commit %s%s, renderer %s",
                dataset_name, commit[:12], " (DIRTY)" if dirty else "", renderer)
    return ProvenanceStamp(
        dataset_name=dataset_name,
        created_utc=created_utc,
        config=config_dict,
        config_sha256=sha256_of(config_dict),
        code_commit=commit,
        code_dirty=dirty,
        asset_manifest_sha256=sha256_of(list(asset_rows)),
        renderer=renderer,
        platform=f"{platform.system()} {platform.release()} / python {platform.python_version()}",
    )
