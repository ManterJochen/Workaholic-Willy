"""Fetch model weights into the shared Hugging Face cache, so the inference tests have something to run.

    python scripts/model_weights/fetch.py --list
    python scripts/model_weights/fetch.py vlm-4b
    python scripts/model_weights/fetch.py --all

Exit codes: ``0`` everything asked for is in the cache, ``1`` at least one fetch failed, ``2`` an
unknown catalogue key was named.

Why this is a script and not "the tests will download it", for three reasons:

* The sim runners set ``HF_HUB_OFFLINE=1`` before the first model import, because a detector that
  pings the hub at boot can take the whole cell down with it. A test that expected to download would
  fail under a runner in a way that reads as a missing model.
* The downloads are multi-gigabyte. A test suite that silently pulls 9 GB on someone's laptop is a bug.
* Two interpreters on one machine can disagree about the hub. The interpreter that loads a model, the
  GPU one, may fail TLS verification with ``CERTIFICATE_VERIFY_FAILED`` where the project ``.venv``
  succeeds. The cache is shared, so the split works: fetch with the interpreter that can reach the
  network, load with the one that owns the GPU. That is what this script is for, and it is why it does
  not run under the GPU interpreter itself.

  The root cause of that split has a fix: ``pip install truststore``. See :func:`use_system_trust_store`.

Weights land in the standard cache, ``HF_HOME`` or else ``~/.cache/huggingface``, and nothing is
written into the repository. That is the ambient cache, and it is not the only one here: the paraphraser
in ``datagen/prompts/model.py`` deliberately points ``HF_HOME`` at ``assets/models/hf`` inside the
repository, so its downloads travel with the clone. Whether a checkpoint is already present is then
answered in three separate places, none of them this file: ``api/routers/diagnostics.py`` and one
guard in the mirrored test tree each run their own ``snapshot_download(local_files_only=True)`` probe
against the ambient cache, and ``datagen.prompts.model.weights_present`` answers it against the
in-repository one. One library answer, living in ``src/models/`` beside the perception spec and called
by all three, is what that wants; it is recorded here rather than made here.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

__all__ = ["CATALOGUE", "fetch"]


@dataclass(frozen=True, slots=True)
class Weights:
    key: str
    repo_id: str
    approx_gb: float
    note: str


#: Everything the `inference`-marked tests can exercise. Sizes are approximate download totals.
CATALOGUE: tuple[Weights, ...] = (
    Weights("dino-tiny", "IDEA-Research/grounding-dino-tiny", 0.7, "the M2 default detector"),
    Weights("dino-base", "IDEA-Research/grounding-dino-base", 1.0, "used by run_dense_pick"),
    Weights("sam2", "facebook/sam2-hiera-large", 0.9, "the segmenter on every route"),
    Weights("vlm-2b", "Qwen/Qwen3-VL-2B-Instruct", 4.5, "the smallest VLM, the quickest smoke"),
    Weights("vlm-4b", "Qwen/Qwen3-VL-4B-Instruct", 9.0, "bf16, loads with no extra dependency"),
    Weights("vlm-8b", "Qwen/Qwen3-VL-8B-Instruct", 17.0, "the documented alternative to compare against"),
    # FP8 is deliberately not the first thing fetched: Qwen ships it in compressed-tensors format, and
    # `compressed-tensors` is absent from the validated Isaac environment. Installing packages into
    # that environment puts the validated pick result at risk, so bf16 is measured first and FP8 only
    # where the VRAM headroom actually calls for it.
    Weights("vlm-4b-fp8", "Qwen/Qwen3-VL-4B-Instruct-FP8", 5.0, "needs `compressed-tensors` installed"),
)

_BY_KEY = {w.key: w for w in CATALOGUE}


def use_system_trust_store() -> str:
    """Point Python's TLS at the operating system's certificate store. Returns what happened.

    This is why the download fails on some networks. A proxy terminating TLS and re-signing it with
    its own CA is the usual cause: the CA is legitimately installed in the OS trust store, while
    ``certifi``, the bundle Python ships with and ``huggingface_hub`` uses, has never heard of it. Two
    interpreters on one machine then disagree, simply because one of them was told about the CA and
    the other was not.

    It does not disable verification, and that distinction is the whole point. Certificates are still
    checked in full; only the set of trusted roots changes, from a shipped bundle to the one an
    administrator curates on this machine. Turning verification off would let anyone on the path serve
    altered weights, and these weights run in a robot's perception path, where a silently modified
    detector is a safety problem rather than an inconvenience.

    Optional by design. The project requirements pin ``truststore``, so the project environment has it;
    the interpreter that runs this script may be a different one, and without the package installed
    this returns and the fetch behaves exactly as it did before.
    """
    try:
        import truststore
    except ImportError:
        return "truststore not installed; verifying against certifi's bundled roots, the default"
    truststore.inject_into_ssl()
    return "TLS verified against the OS trust store (truststore)"


def fetch(weights: Weights) -> str:
    """Download one entry, returning its local snapshot path. Resumes an interrupted fetch."""
    print(f"    tls: {use_system_trust_store()}", flush=True)
    from huggingface_hub import snapshot_download

    print(f"[{weights.key}] {weights.repo_id}  (~{weights.approx_gb:g} GB)", flush=True)
    path = snapshot_download(repo_id=weights.repo_id)
    print(f"    cached at {path}", flush=True)
    return str(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python scripts/model_weights/fetch.py",
        description="Fetch model weights into the shared Hugging Face cache.",
    )
    parser.add_argument("keys", nargs="*", help=f"one or more of: {', '.join(_BY_KEY)}")
    parser.add_argument("--all", action="store_true", help="fetch everything in the catalogue")
    parser.add_argument("--list", action="store_true", help="print the catalogue and exit")
    args = parser.parse_args(argv)

    if args.list or (not args.keys and not args.all):
        total = sum(w.approx_gb for w in CATALOGUE)
        print(f"{'key':<12} {'~GB':>5}  repo id and note")
        for weights in CATALOGUE:
            print(f"{weights.key:<12} {weights.approx_gb:>5.1f}  {weights.repo_id} ({weights.note})")
        print(f"{'':12} {total:>5.1f}  total if --all")
        return 0

    selected = CATALOGUE if args.all else tuple(_BY_KEY[k] for k in args.keys if k in _BY_KEY)
    unknown = [k for k in args.keys if k not in _BY_KEY]
    if unknown:
        print(f"unknown key(s): {', '.join(unknown)}; run --list", file=sys.stderr)
        return 2

    failed = 0
    for weights in selected:
        try:
            fetch(weights)
        # A bare except because any failure at all, of any type, is a failed fetch, and one failure
        # must not abort the rest of a batch: the download that did land is worth keeping.
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"    FAILED {weights.repo_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
    if failed:
        print(f"{failed} of {len(selected)} fetches failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
