"""Fetch model weights into the repository, so a cell and the inference tests have something to load.

    python scripts/model_weights/fetch.py --list
    python scripts/model_weights/fetch.py vlm-4b
    python scripts/model_weights/fetch.py --all
    python scripts/model_weights/fetch.py --mediapipe

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

Every byte lands under ``assets/models/hf`` inside the repository, and deleting the checkout takes
the weights with it. That is the property this location exists for. The default is the opposite: a
download goes to ``~/.cache/huggingface``, measured at 12 GB on the workstation this was written on,
where it outlives every checkout, is shared silently between them, and cannot be reasoned about from
inside the tree. The paraphraser in ``datagen/prompts/model.py`` already pointed its downloads there;
this script and the loaders now do the same, through one definition in ``src/utility/paths.py``.

The fence has to run before ``huggingface_hub`` is imported. The hub reads its cache locations into
module level constants at import time, so setting ``HF_HOME`` afterwards is accepted by ``os.environ``
and changes nothing. That is why :func:`~src.utility.paths.fence_model_downloads` is called at the
top of :func:`main`, before any branch, and why every hub import in this file sits inside a function.

``HF_HOME`` rather than ``HF_HUB_CACHE``, for the same reason: the chunk store, the downloaded
``modules/`` and the token file all derive from ``HF_HOME``, so setting only the narrower variable
leaves pieces of the download in the user profile.

The MediaPipe hand and gesture bundles are downloads too, and ``--mediapipe`` puts them under the
same root. They are not Hugging Face repositories, so they get a plain HTTPS fetch into their own
sub-directory rather than a cache layout.

Whether a checkpoint is already present is still answered in three separate places, none of them this
file: ``api/routers/diagnostics.py`` and one guard in the mirrored test tree each run their own
``snapshot_download(local_files_only=True)`` probe, and ``datagen.prompts.model.weights_present``
answers it by looking at the directory. One library answer, living in ``src/models/`` beside the
perception spec and called by all three, is what that wants; it is recorded here rather than made here.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

# This runs as `python scripts/model_weights/fetch.py`, so the repository root is not on sys.path
# and `src` is not importable yet. Three lines of bootstrap rather than a second copy of the weights
# root: the location every download lands in has one definition, in `paths.py`, and a script that
# computed its own would be free to disagree with the loaders that read it back.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utility.paths import fence_model_downloads  # noqa: E402

__all__ = ["CATALOGUE", "fetch", "fetch_mediapipe"]


@dataclass(frozen=True, slots=True)
class Weights:
    key: str
    repo_id: str
    approx_gb: float
    note: str
    #: Sub-directory under the weights root. The categories match the config blocks that name these
    #: models, so a path in `models.*.model_path` reads as what it is rather than as a hash.
    category: str = "misc"
    #: Files to leave on the hub, per model and never as a blanket rule. Several repositories ship
    #: the same weights twice, once as `.safetensors` and once as `.bin`, and `transformers` reads
    #: only what the index names. Measured: `openai/whisper-small` is 3.87 GB whole and 0.97 GB as
    #: safetensors alone, and `IDEA-Research/grounding-dino-tiny` is 1.38 against 0.69.
    #:
    #: It cannot be a blanket pattern. `shi-labs/oneformer_coco_swin_large` ships no safetensors at
    #: all, measured 0.00 GB of them against 1.83 GB whole, so a global `*.bin` filter would fetch it
    #: empty and the failure would arrive much later as a transformers error about a missing weight.
    ignore: tuple[str, ...] = ()

    def local_dir(self, root: Path) -> Path:
        """Where this model lands. Stable and readable, so a config key can name it."""
        return root / self.category / self.repo_id.replace("/", "--")


#: Every model the shipped config names, plus what the `inference`-marked tests can exercise. Sizes
#: are approximate download totals before the ignore list is applied.
_NO_DUPLICATE_FORMATS = ("*.bin", "*.h5", "*.msgpack", "*.ckpt")

CATALOGUE: tuple[Weights, ...] = (
    Weights("dino-tiny", "IDEA-Research/grounding-dino-tiny", 0.7, "the default detector",
            "detection", _NO_DUPLICATE_FORMATS),
    Weights("dino-base", "IDEA-Research/grounding-dino-base", 1.0, "used by run_dense_pick",
            "detection", _NO_DUPLICATE_FORMATS),
    Weights("rtdetr", "PekingU/rtdetr_r50vd", 0.2, "the closed-set detector, safetensors only",
            "detection"),
    Weights("sam2", "facebook/sam2-hiera-large", 0.9, "the segmenter on every route",
            "segmentation", _NO_DUPLICATE_FORMATS),
    # No ignore list: this one publishes its weights only as .bin, measured.
    Weights("oneformer", "shi-labs/oneformer_coco_swin_large", 1.8, "the research segmenter",
            "segmentation"),
    Weights("whisper", "openai/whisper-small", 1.0, "speech to text; 3.9 GB whole, 1.0 filtered",
            "speech", _NO_DUPLICATE_FORMATS),
    Weights("vlm-2b", "Qwen/Qwen3-VL-2B-Instruct", 4.5, "the smallest VLM, the quickest smoke",
            "vlm", _NO_DUPLICATE_FORMATS),
    Weights("vlm-4b", "Qwen/Qwen3-VL-4B-Instruct", 9.0, "bf16, loads with no extra dependency",
            "vlm", _NO_DUPLICATE_FORMATS),
    Weights("vlm-8b", "Qwen/Qwen3-VL-8B-Instruct", 17.0, "the documented alternative",
            "vlm", _NO_DUPLICATE_FORMATS),
    # FP8 is deliberately not the first thing fetched: Qwen ships it in compressed-tensors format, and
    # `compressed-tensors` is absent from the validated Isaac environment. Installing packages into
    # that environment puts the validated pick result at risk, so bf16 is measured first and FP8 only
    # where the VRAM headroom actually calls for it.
    Weights("vlm-4b-fp8", "Qwen/Qwen3-VL-4B-Instruct-FP8", 5.0, "needs `compressed-tensors`",
            "vlm", _NO_DUPLICATE_FORMATS),
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


def fetch(weights: Weights, root: Path) -> str:
    """Download one entry into its own directory under ``root``. Resumes an interrupted fetch.

    ``local_dir`` rather than the content-addressed cache, because a config key has to be able to
    name this. The cache stores a snapshot under a commit hash, which no `model_path` can carry, so
    `local: true` needs a directory whose name is stable across revisions.
    """
    print(f"    tls: {use_system_trust_store()}", flush=True)
    from huggingface_hub import snapshot_download

    into = weights.local_dir(root)
    print(f"[{weights.key}] {weights.repo_id}  (~{weights.approx_gb:g} GB)", flush=True)
    path = snapshot_download(
        repo_id=weights.repo_id,
        local_dir=str(into),
        ignore_patterns=list(weights.ignore) or None,
    )
    # A filter that matched everything leaves a directory holding metadata and no weights. Said
    # here, the failure names the ignore list; left alone, it arrives much later as a transformers
    # error about a missing file.
    kept = [p for p in into.rglob("*") if p.suffix in (".safetensors", ".bin", ".pt", ".pth")]
    if not kept:
        raise RuntimeError(
            f"{weights.repo_id} downloaded no weight file into {into}. The ignore list "
            f"{weights.ignore} matched every format this repository ships."
        )
    print(f"    at {path}", flush=True)
    return str(path)


def fetch_mediapipe(root: Path) -> int:
    """Download the two MediaPipe ``.task`` bundles into ``<root>/mediapipe``. Returns 0 on success.

    The URLs come from :mod:`src.models.handdetection.constants`, which is where the loaders read
    them when they refuse a missing bundle. One source, so a moved URL cannot leave the fetch and the
    error message pointing at different files.
    """
    import urllib.request

    from src.models.handdetection.constants import GESTURE_MODEL_URL, HAND_LANDMARK_MODEL_URL

    target = root / "mediapipe"
    target.mkdir(parents=True, exist_ok=True)
    for url in (HAND_LANDMARK_MODEL_URL, GESTURE_MODEL_URL):
        name = url.rsplit("/", 1)[-1]
        into = target / name
        print(f"[mediapipe] {name}", flush=True)
        if into.is_file():
            print(f"    at {into} (already here, {into.stat().st_size} bytes)", flush=True)
            continue
        # Downloaded to a temporary name and renamed, so an interrupted fetch cannot leave a
        # half-written bundle that MediaPipe would open and fail inside its C++ graph.
        partial = into.with_suffix(into.suffix + ".part")
        try:
            urllib.request.urlopen(url, timeout=60)  # noqa: S310 - a pinned https URL from constants
        except Exception as exc:  # noqa: BLE001 - report the URL rather than a bare traceback
            print(f"    FAILED {url}: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        urllib.request.urlretrieve(url, partial)  # noqa: S310 - same URL, checked above
        partial.replace(into)
        print(f"    at {into} ({into.stat().st_size} bytes)", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python scripts/model_weights/fetch.py",
        description="Fetch model weights into assets/models/hf inside the repository.",
    )
    parser.add_argument("keys", nargs="*", help=f"one or more of: {', '.join(_BY_KEY)}")
    parser.add_argument("--all", action="store_true", help="fetch everything in the catalogue")
    parser.add_argument("--list", action="store_true", help="print the catalogue and exit")
    parser.add_argument(
        "--mediapipe", action="store_true",
        help="fetch the MediaPipe hand and gesture .task bundles into the same root",
    )
    args = parser.parse_args(argv)

    # Before any hub import and before any other branch: the module docstring says why that order is
    # load bearing rather than tidy.
    root = fence_model_downloads()

    if args.list or (not args.keys and not args.all and not args.mediapipe):
        total = sum(w.approx_gb for w in CATALOGUE)
        print(f"{'key':<12} {'~GB':>5}  repo id and note")
        for weights in CATALOGUE:
            print(f"{weights.key:<12} {weights.approx_gb:>5.1f}  {weights.repo_id} ({weights.note})")
        print(f"{'':12} {total:>5.1f}  total if --all")
        print(f"\nEverything lands under {root}. Deleting the repository deletes the weights.")
        return 0

    if args.mediapipe and (failed := fetch_mediapipe(root)):
        return failed

    selected = CATALOGUE if args.all else tuple(_BY_KEY[k] for k in args.keys if k in _BY_KEY)
    unknown = [k for k in args.keys if k not in _BY_KEY]
    if unknown:
        print(f"unknown key(s): {', '.join(unknown)}; run --list", file=sys.stderr)
        return 2

    failed = 0
    for weights in selected:
        try:
            fetch(weights, root)
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
