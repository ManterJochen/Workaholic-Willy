"""Where every model's weights are read from: one folder, one layout, one resolver.

    from src.models.weights import resolve_weights, weights_present

    found = resolve_weights("facebook/sam2-hiera-large", kind="segmentation")   # <root>/segmentation/facebook--sam2-...
    found.source, found.local                                                  # what from_pretrained is handed, and how

A model is named by its ``model_id`` alone (the owner, 2026-10-17): a Hugging Face id, read from the folder
``scripts/model_weights/fetch.py`` writes it to, ``<root>/<kind>/<org>--<name>``, or the folder a path names (a
``DetectorTraining`` out_dir). The root is set once, ``models.weights_root``, and is the repository's
``assets/models/hf`` unless a tree says otherwise (:func:`src.utility.paths.weights_root`). Every model wrapper asks
here: RT-DETR, SAM2, the VLM, Whisper, Silero and the two MediaPipe bundles, so a path is written in one place and the
fetch writes to the same one.

A run reads this machine alone: the fetched folder first, then the Hugging Face cache, which holds what an earlier
download put there, and else a refusal naming the folder it looked in and the fetch that fills it. Nothing is
downloaded on its own, as every block of the owner's cell said with ``local: true``. Only a caller that asks for it
(``local=False``: :meth:`~src.models.detection.object_detector.ObjectDetector.from_weights`, the detector training's
base model) lets transformers download an id it does not find.

The catalogue below is what the fetch script downloads. It lives here rather than in the script so that a refusal
names the key that fetches its model and the resolver knows the fixed files: Silero is one file inside a PyPI wheel
and the MediaPipe bundles are plain downloads, so neither has an ``<org>--<name>``.

Stdlib only at import: the Hugging Face cache is asked lazily, and only for an id whose folder is not there.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final, Literal, TypeAlias

__all__ = [
    "BUNDLES",
    "CATALOGUE",
    "PACKAGED_FILES",
    "Bundle",
    "PackagedFile",
    "Weights",
    "WeightsKind",
    "WeightsLocation",
    "fetch_hint",
    "fixed_file",
    "locate_weights",
    "missing_sentence",
    "model_folder",
    "names_a_folder",
    "resolve_weights",
    "weights_note",
    "weights_present",
]

#: What a model is for, and the folder under the root its weights land in.
WeightsKind: TypeAlias = Literal["detection", "segmentation", "speech", "vlm"]

#: A Hugging Face id as the Hub spells one: a name, or a namespace and a name.
_HUB_ID: Final = re.compile(r"[A-Za-z0-9][\w.-]*(/[A-Za-z0-9][\w.-]*)?")


# ---------------------------------------------------------------------------------------------------------------------
# The catalogue: what `scripts/model_weights/fetch.py` downloads
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Weights:
    """One Hugging Face repository the fetch script downloads, and where it lands."""

    key: str
    repo_id: str
    approx_gb: float
    note: str
    #: The folder under the weights root: the model's kind, so a folder reads as what it holds.
    category: str = "misc"
    #: Files to leave on the hub, per model and never as a blanket rule. Several repositories ship
    #: the same weights twice, once as `.safetensors` and once as `.bin`, and `transformers` reads
    #: only what the index names. Measured: `openai/whisper-small` is 3.87 GB whole and 0.97 GB as
    #: safetensors alone. It cannot be a blanket pattern: a repository that publishes its weights as
    #: `.bin` alone would be fetched empty, and the failure would arrive much later as a transformers
    #: error about a missing weight.
    ignore: tuple[str, ...] = ()
    #: The hub commit to download, as a full 40 character id. ``None`` takes whatever the default
    #: branch holds on the day, into a folder whose name does not change with it, so a second fetch
    #: can replace the bytes under it without a trace. Every speech entry is pinned: NOTICE states the
    #: licence its model card gave at that commit, and a bake-off result belongs to those weights.
    revision: str | None = None

    def local_dir(self, root: Path) -> Path:
        """Where this model lands: ``<root>/<category>/<org>--<name>``, the folder its id is read from."""
        return root / self.category / self.repo_id.replace("/", "--")


@dataclass(frozen=True, slots=True)
class PackagedFile:
    """One model file that ships inside a Python package on PyPI rather than in a Hub repository.

    Fetched by downloading the pinned wheel, checking the wheel's sha256, reading the one member out of
    it and checking that member's sha256. Nothing is installed: the file is what this repository uses,
    not the package's code.
    """

    key: str
    #: The distribution name on PyPI.
    package: str
    #: The release the wheel belongs to.
    version: str
    #: The wheel's URL on files.pythonhosted.org.
    wheel_url: str
    #: sha256 of the whole wheel, as PyPI lists it.
    wheel_sha256: str
    #: The file's path inside the wheel.
    member: str
    #: sha256 of that file, as the wheel's RECORD lists it.
    sha256: str
    #: Size of that file in bytes.
    size: int
    note: str
    #: Sub-directory under the weights root, as for `Weights`.
    category: str = "misc"

    @property
    def approx_gb(self) -> float:
        """The file's size in gigabytes, for the listing."""
        return self.size / 1e9

    def local_path(self, root: Path) -> Path:
        """Where this file lands: under a folder named for the release, so a new release gets a new path."""
        return root / self.category / f"{self.package}-{self.version}" / PurePosixPath(self.member).name


@dataclass(frozen=True, slots=True)
class Bundle:
    """One model file downloaded as it is from a URL: the MediaPipe ``.task`` bundles, which no hub hosts."""

    key: str
    url: str
    note: str
    #: Sub-directory under the weights root.
    category: str = "mediapipe"

    @property
    def file_name(self) -> str:
        """The file's name, the URL's last segment."""
        return self.url.rsplit("/", 1)[-1]

    def local_path(self, root: Path) -> Path:
        """Where this bundle lands: ``<root>/<category>/<file name>``."""
        return root / self.category / self.file_name


#: Formats a repository may publish beside its safetensors, left on the hub where the safetensors are there.
_NO_DUPLICATE_FORMATS: Final = ("*.bin", "*.h5", "*.msgpack", "*.ckpt")

#: Every model a shipped tree names, plus what the `inference`-marked tests can exercise. Sizes are approximate
#: download totals before the ignore list is applied.
CATALOGUE: Final[tuple[Weights, ...]] = (
    Weights("rtdetr", "PekingU/rtdetr_r50vd", 0.2, "the closed-set detector, safetensors only",
            "detection"),
    Weights("sam2", "facebook/sam2-hiera-large", 0.9, "the segmenter on every stack",
            "segmentation", _NO_DUPLICATE_FORMATS),
    # The speech entries carry a commit. Measured at this commit with the filter applied: 13 files
    # and 1,622,466,054 bytes. openai/whisper-small, apache-2.0, is no longer in the catalogue and
    # nothing in the tree names it.
    Weights("whisper-turbo", "openai/whisper-large-v3-turbo", 1.6, "speech to text, the main engine",
            "speech", _NO_DUPLICATE_FORMATS, revision="41f01f3fe87f28c78e2fbf8b568835947dd65ed9"),
    Weights("vlm-2b", "Qwen/Qwen3-VL-2B-Instruct", 4.5, "the smallest VLM, the quickest smoke",
            "vlm", _NO_DUPLICATE_FORMATS),
    Weights("vlm-4b", "Qwen/Qwen3-VL-4B-Instruct", 9.0, "bf16, loads with no extra dependency",
            "vlm", _NO_DUPLICATE_FORMATS),
    Weights("vlm-8b", "Qwen/Qwen3-VL-8B-Instruct", 17.0, "the owner's cell's",
            "vlm", _NO_DUPLICATE_FORMATS),
    # FP8 is deliberately not the first thing fetched: Qwen ships it in compressed-tensors format, and
    # `compressed-tensors` is absent from the validated Isaac environment. Installing packages into
    # that environment puts the validated pick result at risk, so bf16 is measured first and FP8 only
    # where the VRAM headroom actually calls for it.
    Weights("vlm-4b-fp8", "Qwen/Qwen3-VL-4B-Instruct-FP8", 5.0, "needs `compressed-tensors`",
            "vlm", _NO_DUPLICATE_FORMATS),
)

#: Model files that live inside a PyPI wheel. Measured: PyPI lists the silero-vad 6.2.1 wheel with the
#: first sha256 below, the downloaded wheel hashes to it, and its `silero_vad/data/silero_vad.jit`
#: hashes to the second, the value the installed package's RECORD gives for the same file. The wheel's
#: LICENSE is the MIT licence.
PACKAGED_FILES: Final[tuple[PackagedFile, ...]] = (
    PackagedFile(
        "silero-vad", "silero-vad", "6.2.1",
        "https://files.pythonhosted.org/packages/0b/2b/"
        "48566f29a8b53d856ceb1994f209122749b3fda0a733a07e82047257de7a/silero_vad-6.2.1-py3-none-any.whl",
        "09de93c4d874bb19c53e62a47dd38be5f163cedad2b5599583231f2a84ef79cb",
        "silero_vad/data/silero_vad.jit",
        "e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720",
        2_272_526,
        "the voice detector, Silero VAD as TorchScript",
        "speech",
    ),
)

#: The two MediaPipe bundles, downloaded as they are, from the URLs the hand detectors name when they refuse a
#: missing bundle (``src.models.handdetection.constants``), so a moved URL cannot leave the fetch and the refusal
#: pointing at different files.
BUNDLES: Final[tuple[Bundle, ...]] = (
    Bundle("mediapipe-hand",
           "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/"
           "hand_landmarker.task",
           "21 landmarks per hand, no gesture head: the palm detector's"),
    Bundle("mediapipe-gesture",
           "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/"
           "gesture_recognizer.task",
           "the canned gesture classifier with its own hand landmarker: thumbs up and down"),
)

_BY_REPO: Final = {weights.repo_id: weights for weights in CATALOGUE}


# ---------------------------------------------------------------------------------------------------------------------
# The resolver
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WeightsLocation:
    """Where one model's weights are read from on this machine, decided without loading them.

    Attributes:
        model_id (str): The id or path the model was named by.
        folder (Path): The folder that holds them, or would: the one the fetch writes an id to, or the one a path
            names.
        source (str): What ``from_pretrained`` is handed: the folder, or the id where the Hugging Face cache holds
            it (or, with ``local=False``, where it may be downloaded).
        local (bool): Whether the model is read with ``local_files_only``: this machine alone.
        present (bool): Whether the weights are on this machine: the folder is there, or the cache holds the id.
    """

    model_id: str
    folder: Path
    source: str
    local: bool
    present: bool

    @property
    def from_the_folder(self) -> bool:
        """Whether the source is the folder rather than an id (the cache's or the Hub's)."""
        return self.source == self.folder.as_posix()

    def describe(self) -> str:
        """Where the weights are read from, as a person reads it."""
        if self.from_the_folder:
            return self.source
        return f"{self.source} (from the Hugging Face {'cache' if self.present else 'Hub'})"


def _root(root: str | os.PathLike[str] | None) -> Path:
    if root is not None and os.fspath(root):
        return Path(root)
    from src.utility.paths import weights_root  # noqa: PLC0415 (light; kept off this module's import)

    return weights_root()


def names_a_folder(model_id: str | os.PathLike[str]) -> bool:
    """Whether ``model_id`` names a folder rather than a Hugging Face id: absolute, from home, relative by a dot, or
    written with a backslash. A plain ``org/name`` is an id, never a path.

    Args:
        model_id (str | os.PathLike[str]): A model id as a block or a caller names it.

    Returns:
        bool: ``True`` for a path; a ``PathLike`` always is one.
    """
    if isinstance(model_id, os.PathLike):
        return True
    text = str(model_id).strip()
    return (Path(text).anchor != "" or text.startswith(("~", "./", "../", ".\\", "..\\")) or "\\" in text
            or text in (".", ".."))


def model_folder(model_id: str | os.PathLike[str], *, kind: WeightsKind,
                 root: str | os.PathLike[str] | None = None) -> Path:
    """The folder ``model_id``'s weights are read from on this machine, whether it is there or not.

    Args:
        model_id (str | os.PathLike[str]): A Hugging Face id, or a folder's path.
        kind (WeightsKind): What the model is for, the folder under the root an id lands in.
        root (str | os.PathLike[str] | None): The weights root; ``None`` is the repository's
            (``src.utility.paths.weights_root``) (default: None).

    Returns:
        Path: ``<root>/<kind>/<org>--<name>`` for an id, the folder itself for a path.

    Raises:
        FileNotFoundError: ``model_id`` is neither a path nor a Hugging Face id.
    """
    text = os.fspath(model_id).strip()
    if names_a_folder(model_id):
        return Path(text).expanduser()
    if not _HUB_ID.fullmatch(text) or ".." in text:
        raise FileNotFoundError(f"{text!r} is neither a folder nor a Hugging Face id (org/name), so there are no "
                                f"weights to read")
    return _root(root) / kind / text.replace("/", "--")


def locate_weights(model_id: str | os.PathLike[str], *, kind: WeightsKind,
                   root: str | os.PathLike[str] | None = None, local: bool = True) -> WeightsLocation:
    """Where ``model_id``'s weights are read from, decided without loading them and without refusing.

    A path is read as the folder it names. An id is read from ``<root>/<kind>/<org>--<name>`` where that folder is
    there, else from the Hugging Face cache where it holds the id; else from the Hub where ``local`` is false, or, with
    ``local`` true, from nowhere (``present`` false, ``source`` the folder looked in), which :func:`resolve_weights`
    refuses.

    Args:
        model_id (str | os.PathLike[str]): A Hugging Face id, or a folder's path.
        kind (WeightsKind): What the model is for.
        root (str | os.PathLike[str] | None): The weights root; ``None`` is the repository's (default: None).
        local (bool): Read this machine alone, as every model built from a tree does; ``False`` lets transformers
            download an id that is nowhere here (default: True).

    Returns:
        WeightsLocation: The folder, the source, whether it is read locally and whether it is here.

    Raises:
        FileNotFoundError: ``model_id`` is neither a path nor a Hugging Face id.
    """
    folder = model_folder(model_id, kind=kind, root=root)
    text = os.fspath(model_id).strip()
    if names_a_folder(model_id) or folder.is_dir():
        return WeightsLocation(model_id=text, folder=folder, source=folder.as_posix(), local=True,
                               present=folder.is_dir())
    if _in_the_cache(text):
        return WeightsLocation(model_id=text, folder=folder, source=text, local=True, present=True)
    if not local:
        return WeightsLocation(model_id=text, folder=folder, source=text, local=False, present=False)
    return WeightsLocation(model_id=text, folder=folder, source=folder.as_posix(), local=True, present=False)


def resolve_weights(model_id: str | os.PathLike[str], *, kind: WeightsKind,
                    root: str | os.PathLike[str] | None = None, local: bool = True,
                    key: str = "") -> WeightsLocation:
    """:func:`locate_weights`, refused where the weights are not on this machine and may not be downloaded.

    Args:
        model_id (str | os.PathLike[str]): A Hugging Face id, or a folder's path.
        kind (WeightsKind): What the model is for.
        root (str | os.PathLike[str] | None): The weights root; ``None`` is the repository's (default: None).
        local (bool): Read this machine alone (default: True).
        key (str): The config key that named the model, for the refusal (default: "").

    Returns:
        WeightsLocation: Where the weights are read from.

    Raises:
        FileNotFoundError: The weights are not on this machine and ``local`` is true, naming the folder looked in and
            the fetch that writes it; or ``model_id`` names nothing.
    """
    found = locate_weights(model_id, kind=kind, root=root, local=local)
    if found.present or not found.local:
        return found
    raise FileNotFoundError(missing_sentence(found, key=key))


def weights_present(model_id: str | os.PathLike[str], *, kind: WeightsKind,
                    root: str | os.PathLike[str] | None = None) -> bool:
    """Whether ``model_id``'s weights are on this machine. Never downloads, never loads, never raises.

    Args:
        model_id (str | os.PathLike[str]): A Hugging Face id, or a folder's path.
        kind (WeightsKind): What the model is for.
        root (str | os.PathLike[str] | None): The weights root; ``None`` is the repository's (default: None).

    Returns:
        bool: ``True`` where the folder is there or the Hugging Face cache holds the id.
    """
    try:
        return locate_weights(model_id, kind=kind, root=root).present
    except Exception:  # noqa: BLE001 (a name that names nothing is not here, which is the answer)
        return False


def missing_sentence(found: WeightsLocation, *, key: str = "") -> str:
    """The refusal for weights that are not on this machine: the key, the folder looked in, and the fetch.

    Args:
        found (WeightsLocation): Where the weights would be.
        key (str): The config key that named the model (default: "").

    Returns:
        str: One sentence, ending in a full stop.
    """
    named = f"{key} is {found.model_id!r}" if key else f"{found.model_id!r} is named"
    cache = "" if names_a_folder(found.model_id) else " and nothing in the Hugging Face cache"
    return (f"{named}, and its weights are not on this machine: no folder at {found.folder.as_posix()}"
            f"{cache}.{weights_note(found.folder)} Nothing is downloaded at run time; "
            f"{fetch_hint(found.model_id, folder=found.folder)}.")


def weights_note(path: str | os.PathLike[str] | None) -> str:
    """Why a path a model is read from is the path it is, as a sentence for a refusal, or ``""``.

    A path a tree wrote is explained as written (:func:`src.config.paths.path_note`). One built under the weights
    folder (``<weights_root>/<kind>/<org>--<name>``, or a bundle's fixed place) is explained by that folder's own
    note, so a relative ``models.weights_root`` read against the config folder reads as what it is, as a relative
    ``model_path`` did before the folder was set once.

    Args:
        path (str | os.PathLike[str] | None): The path a refusal names.

    Returns:
        str: The sentence, with a space in front, or ``""`` where no tree resolved the path or a folder above it.
    """
    from src.config.paths import path_note  # noqa: PLC0415 (light; kept off this module's import)

    if not path:
        return ""
    note = path_note(path)
    if note:
        return note
    for parent in Path(os.fspath(path)).parents:
        note = path_note(parent)
        if note:
            return f" The weights folder is {Path(os.path.normpath(parent)).as_posix()}.{note}"
    return ""


def fetch_hint(model_id: str, *, folder: Path | None = None) -> str:
    """How the weights of ``model_id`` come onto this machine, as the end of a sentence.

    Args:
        model_id (str): A Hugging Face id, or a folder's path.
        folder (Path | None): The folder the weights would be read from, for an id the catalogue lacks
            (default: None).

    Returns:
        str: The fetch command for a catalogue id, else what writes such a folder.
    """
    entry = _BY_REPO.get(str(model_id).strip())
    if entry is not None:
        return f"`python scripts/model_weights/fetch.py {entry.key}` writes them"
    if names_a_folder(model_id):
        return "name a folder save_pretrained wrote the model to (for the closed-set detector, a DetectorTraining out_dir)"
    where = f" into {folder.as_posix()}" if folder is not None else ""
    return (f"`python scripts/model_weights/fetch.py --list` names what it fetches, and any other checkpoint is "
            f"copied{where} as save_pretrained writes one")


def fixed_file(key: str, *, root: str | os.PathLike[str] | None = None) -> Path:
    """The fixed path of a model file that has no ``<org>--<name>``: Silero (``silero-vad``) and the MediaPipe bundles
    (``mediapipe-hand``, ``mediapipe-gesture``), where the fetch writes them under the root.

    Args:
        key (str): The catalogue key.
        root (str | os.PathLike[str] | None): The weights root; ``None`` is the repository's (default: None).

    Returns:
        Path: Where the file is, or would be.

    Raises:
        KeyError: ``key`` is no fixed file of the catalogue.
    """
    base = _root(root)
    entries: tuple[PackagedFile | Bundle, ...] = (*PACKAGED_FILES, *BUNDLES)
    for entry in entries:
        if entry.key == key:
            return entry.local_path(base)
    raise KeyError(f"{key!r} is no fixed file of the weights catalogue; it holds "
                   f"{', '.join(entry.key for entry in entries)}")


def _in_the_cache(repo_id: str) -> bool:
    """Whether the Hugging Face cache holds ``repo_id``: its ``config.json`` is there. Asks no network."""
    try:
        from huggingface_hub import try_to_load_from_cache  # noqa: PLC0415 (only for an id whose folder is not there)
    except ImportError:
        return False
    try:
        found = try_to_load_from_cache(repo_id, "config.json")
    except Exception:  # noqa: BLE001 (a cache that cannot be read holds nothing for this purpose)
        return False
    return isinstance(found, str) and Path(found).is_file()
