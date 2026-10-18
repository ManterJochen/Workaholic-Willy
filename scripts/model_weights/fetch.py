"""Fetch model weights into the weights folder, so a cell and the inference tests have something to load.

    python scripts/model_weights/fetch.py --list
    python scripts/model_weights/fetch.py vlm-4b
    python scripts/model_weights/fetch.py whisper-turbo silero-vad
    python scripts/model_weights/fetch.py --all
    python scripts/model_weights/fetch.py --mediapipe
    python scripts/model_weights/fetch.py sam2 --root D:/models

Exit codes: ``0`` everything asked for is in the folder, ``1`` at least one fetch failed, ``2`` an
unknown catalogue key was named.

Why this is a script and not "the tests will download it", for three reasons:

* The sim runners set ``HF_HUB_OFFLINE=1`` before the first model import, because a detector that
  pings the hub at boot can take the whole cell down with it. A test that expected to download would
  fail under a runner in a way that reads as a missing model.
* The downloads are multi-gigabyte. A test suite that silently pulls 9 GB on someone's laptop is a bug.
* Two interpreters on one machine can disagree about the hub. The interpreter that loads a model, the
  GPU one, may fail TLS verification with ``CERTIFICATE_VERIFY_FAILED`` where the project ``.venv``
  succeeds. The folder is shared, so the split works: fetch with the interpreter that can reach the
  network, load with the one that owns the GPU. That is what this script is for, and it is why it does
  not run under the GPU interpreter itself.

  The root cause of that split has a fix: ``pip install truststore``. See :func:`use_system_trust_store`.

Every byte lands in the weights folder, ``models.weights_root`` of the tree ``WILLY_PROFILE`` names, which is
``assets/models/hf`` inside the repository unless the tree says otherwise; ``--root`` names another. A model lands
at ``<root>/<kind>/<org>--<name>``, which is where every model reads its ``model_id`` from
(:mod:`src.models.weights`, which holds the catalogue). Inside the repository, deleting the checkout takes the
weights with it. The default of the libraries is the opposite: a download goes to ``~/.cache/huggingface``,
measured at 12 GB on the workstation this was written on, where it outlives every checkout, is shared silently
between them, and cannot be reasoned about from inside the tree.

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
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# This runs as `python scripts/model_weights/fetch.py`, so the repository root is not on sys.path
# and `src` is not importable yet. Three lines of bootstrap rather than a second copy of the
# catalogue: where every download lands has one definition, in `src/models/weights.py`, and a script
# that computed its own would be free to disagree with the loaders that read it back.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.weights import (  # noqa: E402
    BUNDLES,
    CATALOGUE,
    PACKAGED_FILES,
    Bundle,
    PackagedFile,
    Weights,
)
from src.utility.paths import fence_model_downloads, weights_root  # noqa: E402

__all__ = ["BUNDLES", "CATALOGUE", "PACKAGED_FILES", "Bundle", "PackagedFile", "Weights", "fetch", "fetch_bundle",
           "fetch_file", "fetch_mediapipe", "the_trees_root"]

_BY_KEY = {w.key: w for w in CATALOGUE}
_FILES_BY_KEY = {f.key: f for f in PACKAGED_FILES}
_BUNDLES_BY_KEY = {b.key: b for b in BUNDLES}


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


def the_trees_root() -> tuple[Path, str]:
    """The weights folder of the tree ``WILLY_PROFILE`` names, and where that answer came from.

    The tree is read for its one key alone (``load_weights_root``), so a camera or robot file that does not load
    cannot stop a fetch. A tree that cannot be read at all, or an interpreter without the config's packages, fetches
    into the repository's folder and says so: a fetch is what a fresh checkout runs first.
    """
    try:
        from src.config.loader import load_weights_root  # noqa: PLC0415 (pyyaml and pydantic, the venv's)

        return Path(load_weights_root()), "models.weights_root of the tree WILLY_PROFILE names"
    except Exception as exc:  # noqa: BLE001 (any failure to read the tree falls back to the repository's folder)
        return weights_root(), f"the repository's folder; the tree's models.weights_root was not read ({exc})"


def fetch(weights: Weights, root: Path) -> str:
    """Download one entry into its own directory under ``root``. Resumes an interrupted fetch.

    ``local_dir`` rather than the content-addressed cache, because a model id is read from this folder
    (``src/models/weights.py``). The cache stores a snapshot under a commit hash, which no reader names.
    """
    print(f"    tls: {use_system_trust_store()}", flush=True)
    from huggingface_hub import snapshot_download

    into = weights.local_dir(root)
    print(f"[{weights.key}] {weights.repo_id}  (~{weights.approx_gb:g} GB)", flush=True)
    print(f"    revision: {weights.revision or 'unpinned, the default branch today'}", flush=True)
    path = snapshot_download(
        repo_id=weights.repo_id,
        revision=weights.revision,
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


def fetch_file(entry: PackagedFile, root: Path) -> str:
    """Download ``entry``'s wheel and write the one member both hashes vouch for under ``root``.

    A wheel or a member whose sha256 differs from the pin is refused and nothing is written: the file
    runs in the cell's speech path. A file already there with the pinned hash is left alone.
    """
    import hashlib
    import io
    import urllib.request
    import zipfile

    print(f"    tls: {use_system_trust_store()}", flush=True)
    into = entry.local_path(root)
    print(f"[{entry.key}] {entry.member} from {entry.package} {entry.version}", flush=True)
    if into.is_file() and hashlib.sha256(into.read_bytes()).hexdigest() == entry.sha256:
        print(f"    at {into} (already here at the pinned sha256)", flush=True)
        return str(into)
    # A pinned https URL, and the bytes are checked against two hashes before anything is written.
    with urllib.request.urlopen(entry.wheel_url, timeout=60) as response:  # noqa: S310
        wheel = response.read()
    digest = hashlib.sha256(wheel).hexdigest()
    if digest != entry.wheel_sha256:
        raise RuntimeError(
            f"{entry.wheel_url} downloaded a wheel with sha256 {digest}, and the catalogue pins "
            f"{entry.wheel_sha256}. Nothing was written."
        )
    with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
        member = archive.read(entry.member)
    digest = hashlib.sha256(member).hexdigest()
    if digest != entry.sha256:
        raise RuntimeError(
            f"{entry.member} in {entry.package} {entry.version} has sha256 {digest}, and the catalogue "
            f"pins {entry.sha256}. Nothing was written."
        )
    into.parent.mkdir(parents=True, exist_ok=True)
    partial = into.with_suffix(into.suffix + ".part")
    partial.write_bytes(member)
    partial.replace(into)
    print(f"    at {into} ({len(member)} bytes, sha256 {digest})", flush=True)
    return str(into)


def fetch_bundle(entry: Bundle, root: Path) -> str:
    """Download one bundle as it is into ``<root>/<category>/``. A bundle already there is left alone.

    Downloaded to a temporary name and renamed, so an interrupted fetch cannot leave a half-written
    bundle that MediaPipe would open and fail inside its C++ graph.
    """
    import urllib.request

    into = entry.local_path(root)
    print(f"[{entry.key}] {entry.file_name}", flush=True)
    if into.is_file():
        print(f"    at {into} (already here, {into.stat().st_size} bytes)", flush=True)
        return str(into)
    into.parent.mkdir(parents=True, exist_ok=True)
    partial = into.with_suffix(into.suffix + ".part")
    urllib.request.urlretrieve(entry.url, partial)  # noqa: S310 - a pinned https URL from the catalogue
    partial.replace(into)
    print(f"    at {into} ({into.stat().st_size} bytes)", flush=True)
    return str(into)


def fetch_mediapipe(root: Path) -> int:
    """Download the two MediaPipe ``.task`` bundles into ``<root>/mediapipe``. Returns 0 on success."""
    for entry in BUNDLES:
        try:
            fetch_bundle(entry, root)
        except Exception as exc:  # noqa: BLE001 - report the URL rather than a bare traceback
            print(f"    FAILED {entry.url}: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python scripts/model_weights/fetch.py",
        description="Fetch model weights into the weights folder (models.weights_root, assets/models/hf as shipped).",
    )
    parser.add_argument("keys", nargs="*",
                        help=f"one or more of: {', '.join([*_BY_KEY, *_FILES_BY_KEY, *_BUNDLES_BY_KEY])}")
    parser.add_argument("--all", action="store_true", help="fetch everything in the catalogue")
    parser.add_argument("--list", action="store_true", help="print the catalogue and exit")
    parser.add_argument(
        "--mediapipe", action="store_true",
        help="fetch the MediaPipe hand and gesture .task bundles into the same root",
    )
    parser.add_argument("--root", default=None,
                        help="the weights folder to fetch into; left out, the tree's models.weights_root")
    args = parser.parse_args(argv)

    chosen, said = (Path(args.root), "--root") if args.root else the_trees_root()
    # Before any hub import and before any other branch: the module docstring says why that order is
    # load bearing rather than tidy.
    root = fence_model_downloads(chosen)

    if args.list or (not args.keys and not args.all and not args.mediapipe):
        total = sum(w.approx_gb for w in CATALOGUE) + sum(f.approx_gb for f in PACKAGED_FILES)
        print(f"{'key':<18} {'~GB':>5}  {'revision':<8}  repo id and note")
        for weights in CATALOGUE:
            pin = (weights.revision or "unpinned")[:8]
            print(f"{weights.key:<18} {weights.approx_gb:>5.1f}  {pin:<8}  "
                  f"{weights.repo_id} ({weights.note})")
        for entry in PACKAGED_FILES:
            print(f"{entry.key:<18} {entry.approx_gb:>5.3f}  {entry.sha256[:8]:<8}  "
                  f"{entry.package} {entry.version}, {entry.member} ({entry.note})")
        for bundle in BUNDLES:
            print(f"{bundle.key:<18} {'':>5}  {'latest':<8}  {bundle.file_name} ({bundle.note})")
        print(f"{'':18} {total:>5.1f}  {'':8}  total if --all")
        print(f"\nEverything lands under {root} ({said}).")
        return 0

    if args.mediapipe and (failed := fetch_mediapipe(root)):
        return failed

    known = {*_BY_KEY, *_FILES_BY_KEY, *_BUNDLES_BY_KEY}
    unknown = [k for k in args.keys if k not in known]
    if unknown:
        print(f"unknown key(s): {', '.join(unknown)}; run --list", file=sys.stderr)
        return 2
    selected = CATALOGUE if args.all else tuple(_BY_KEY[k] for k in args.keys if k in _BY_KEY)
    files = PACKAGED_FILES if args.all else tuple(_FILES_BY_KEY[k] for k in args.keys if k in _FILES_BY_KEY)
    bundles = BUNDLES if args.all else tuple(_BUNDLES_BY_KEY[k] for k in args.keys if k in _BUNDLES_BY_KEY)

    failed = 0
    for weights in selected:
        try:
            fetch(weights, root)
        # A bare except because any failure at all, of any type, is a failed fetch, and one failure
        # must not abort the rest of a batch: the download that did land is worth keeping.
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"    FAILED {weights.repo_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
    for entry in files:
        try:
            fetch_file(entry, root)
        # A bare except for the same reason as above: one failure must not abort the rest of a batch.
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"    FAILED {entry.key}: {type(exc).__name__}: {exc}", file=sys.stderr)
    for bundle in bundles:
        try:
            fetch_bundle(bundle, root)
        # A bare except for the same reason as above: one failure must not abort the rest of a batch.
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"    FAILED {bundle.key}: {type(exc).__name__}: {exc}", file=sys.stderr)
    if failed:
        print(f"{failed} of {len(selected) + len(files) + len(bundles)} fetches failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
