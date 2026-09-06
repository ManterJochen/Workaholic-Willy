"""Real meshes for the scene generator, and the attribution that makes them shippable.

The procedural library renders sixteen kind words as three convex solids: a "cup" has no handle and
a "bowl" is a cuboid. That is enough for measuring geometry and not enough for teaching a network
what a mug is, which is why real meshes exist beside it. `assets.gso_weight`, `ycb_weight` and
`objaverse_weight` are the config knobs that draw from them.

What may be imported is a licensing question rather than a matter of taste. GSO and YCB are
institutional datasets with one known author each, so their CC-BY attribution is a fact about the
collection and is recorded once. Objaverse is individually authored, so its attribution is per
object and travels beside the meshes in ATTRIBUTION.tsv. `audit_asset_rows` refuses a CC-BY row that
carries no attribution string, because every render is a derivative work.

Meshes are not vendored into the repository. They are fetched into a gitignored directory, exactly
as the MediaPipe bundles are, and `python -m datagen.assets --check` answers fail-closed whether
this machine has them.
"""

from __future__ import annotations

import csv
import hashlib
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from src.utility.log_cfg import create_logger

from datagen.assets.manifest import AssetRecord
from datagen.constants import DATAGEN_LOG_DIR, LICENSING_LOG_FILE, MESH_LIBRARY_DIR

__all__ = [
    "ATTRIBUTIONS",
    "MESH_LIBRARY_DIR",
    "MeshEntry",
    "MeshLibrary",
    "SUPPORTED_SOURCES",
    "import_from_directory",
]

logger = create_logger("AssetLibrary", LICENSING_LOG_FILE, log_dir=DATAGEN_LOG_DIR)

#: Re-exported from `datagen.constants`, which is where it is declared: `assets/fetch.py` needs the
#: same value and cannot import this module, because importing it pulls numpy. See the note there.

#: Every mesh source the library can import, and the file extension each one ships.
#:
#: `custom` is the half the product rests on: without it a customer cannot bring a mesh of their
#: own, because `import_from_directory` gates on this map. Its extension is `*`, since a customer's
#: CAD exports are whatever their tool writes and `trimesh` reads all of them; refusing a `.stl`
#: because the two research datasets ship `.obj` and `.ply` would be an arbitrary rule.
#:
#: A source the fetcher can download but this map does not list is a source the pipeline never sees.
#: `screen_meshes` and `prepare_assets` iterate this dict when no collection is named, so an
#: unlisted source is skipped in silence: the meshes are on disk, the commands report success, and
#: nothing is screened. This map and the fetcher's source list have to stay in step.
SUPPORTED_SOURCES: Final[dict[str, str]] = {
    "gso": "obj", "ycb": "ply", "thingi10k": "obj", "asos": "ply",
    "objaverse": "obj", "custom": "*",
}

#: What `custom` may actually contain. Not a whitelist of vendors but a list of what the mesh reader
#: can open, which is a different and checkable thing. It is what the whole rail can carry, not what
#: one importer will accept, which is why `prepare.MESH_SUFFIXES` derives from it. A suffix accepted
#: here that the reader cannot open is imported, counted ready by `available()` and `is_ready()`,
#: placed by the layout, and then skipped in silence by both `normalise_meshes` and `screen_meshes`:
#: it never reaches a corpus and nothing says so, which on a customer rail is the worst place for
#: that failure to live.
#:
#: `.dae` is absent, and measured rather than assumed. trimesh round-trips obj, stl, ply, off, glb
#: and gltf; `.dae` raises `ImportError: missing 'pip install pycollada'`. Add `pycollada` to the
#: requirements and put the suffix back here, in that order.
CUSTOM_SUFFIXES: Final[tuple[str, ...]] = ("obj", "stl", "ply", "off", "glb", "gltf")

#: The CC-BY attribution each collection obliges: author, title and URL, what the licence asks for,
#: recorded once per collection because that is the granularity at which these are authored.
#: Objaverse is the exception, and its entry says so, because its models are authored per object.
ATTRIBUTIONS: Final[dict[str, str]] = {
    # Per object, and this string says so rather than naming one author. Objaverse is
    # artist-uploaded, so the obliged credit is a different name for every mesh and cannot be a
    # constant. The fetch writes ATTRIBUTION.tsv beside the meshes with the author and profile for
    # each, and this entry sends a reader there rather than letting the map imply one author.
    "objaverse": (
        "Objaverse 1.0 (Deitke et al., Allen Institute for AI), "
        "https://objaverse.allenai.org/, ODC-By-1.0 over the collection; "
        "each object is CC-BY-4.0 or CC0-1.0 and its author is named in "
        "assets/meshes/objaverse/ATTRIBUTION.tsv"
    ),
    "gso": (
        "Google Scanned Objects (Downs et al., Google Research), "
        "https://goo.gle/scanned-objects, CC-BY-4.0"
    ),
    "ycb": (
        "YCB Object and Model Set (Calli et al., Yale / CMU / UC Berkeley), "
        "https://www.ycbbenchmarks.com/, CC-BY-4.0"
    ),
    # CC0 and public domain oblige nothing, so this string is a courtesy and not a licence term. It
    # is recorded anyway because `MeshEntry.attribution` reads this map for every source, not only
    # the obliged ones. Only the no-attribution slice is ever fetched: the collection is
    # individually authored user uploads and its metadata carries no author column, so its CC-BY
    # rows cannot be credited at all. See `datagen/assets/fetch.py`.
    "thingi10k": (
        "Thingi10K (Zhou and Jacobson), https://ten-thousand-models.appspot.com/, "
        "CC0-1.0, public-domain slice only"
    ),
    "asos": (
        "Coles Object Set (Chumbley, Monash University), "
        "https://doi.org/10.26180/20179550, CC-BY-4.0"
    ),
}

#: The licence each source ships under, kept next to the attribution so the two cannot drift.
#:
#: `custom` has no entry on purpose: nobody but the customer can know what their own parts are
#: licensed as, and guessing would put a string into the fail-closed licence audit as if it had been
#: checked. `import_from_directory` requires it to be passed for `custom` and refuses without one.
#:
#: Objaverse is the one entry here that is not a collection-wide term. Its licence is per object,
#: CC-BY-4.0 or CC0-1.0, and the fetcher refuses NonCommercial and ShareAlike objects before
#: download. The string below is the stricter of the two admitted, because it is the one that
#: obliges naming the author: over-crediting a CC0 object costs nothing, under-crediting a CC-BY one
#: is the failure this repository fences everywhere. The per-object truth travels beside the meshes
#: in ATTRIBUTION.tsv, written by the fetch, and a LICENSE.txt carries the same declaration for
#: `declared_license`.
LICENSES: Final[dict[str, str]] = {
    "gso": "CC-BY-4.0", "ycb": "CC-BY-4.0", "thingi10k": "CC0-1.0", "asos": "CC-BY-4.0",
    "objaverse": "CC-BY-4.0",
}


@dataclass(frozen=True, slots=True)
class MeshEntry:
    """One mesh on disk, with everything the manifest needs to justify rendering it."""

    asset_id: str
    source: str
    path: Path
    sha256: str

    @property
    def license(self) -> str:
        return LICENSES[self.source]

    @property
    def attribution(self) -> str:
        return ATTRIBUTIONS[self.source]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


#: Sources that are optional for a machine to be "ready". `custom` is a customer's own parts: almost
#: no machine has any, and treating their absence as a missing dependency would turn `--check` red
#: for every operator who uses only the research datasets.
OPTIONAL_SOURCES: Final[frozenset[str]] = frozenset({"custom"})

#: Filename the `custom` licence declaration is written to, beside the meshes it describes.
LICENSE_DECLARATION = "LICENSE.txt"


def declared_license(directory: Path | str) -> tuple[str, str]:
    """The `(license, attribution)` an operator declared for a directory of custom meshes.

    Read from disk, not from a map. `LICENSES` can answer for `gso` and `ycb` because those two
    datasets publish collection-wide terms; nobody but the customer knows what their own parts are
    licensed as, so for `custom` the answer travels with the meshes. `import_from_directory` writes
    it, this reads it, and manifest, audit and attribution text all get the same string.

    Returns empty strings when there is no declaration, which is the honest answer for a directory
    populated by hand. The licence audit then reports it as undeclared rather than passing it.
    """
    path = Path(directory) / LICENSE_DECLARATION
    if not path.is_file():
        return ("", "")
    fields: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip() in ("license", "attribution"):
            fields[key.strip()] = value.strip()
    attribution = fields.get("attribution", "")
    return (fields.get("license", ""), "" if attribution == "(none declared)" else attribution)


class MeshLibrary:
    """The meshes this machine actually has, per source.

    Reads the directory rather than a checked-in list on purpose: the list would be a second source
    of truth that goes stale the moment somebody fetches a different revision, and the audit cares
    about what will be rendered, not about what was once intended.
    """

    def __init__(self, root: Path | str = MESH_LIBRARY_DIR) -> None:
        self.root = Path(root)

    def available(self, source: str) -> list[Path]:
        """Mesh files present for one source, sorted for determinism."""
        suffix = SUPPORTED_SOURCES.get(source)
        if suffix is None:
            return []
        directory = self.root / source
        if not directory.is_dir():
            return []
        if suffix == "*":
            # Not `glob("*.*")`. A `custom` directory also holds the LICENSE.txt this library writes
            # beside the meshes, and a glob on "anything with a dot" would hand that text file to
            # the mesh reader as an asset. Match the suffixes trimesh can actually open instead.
            return sorted(path for path in directory.iterdir()
                          if path.suffix.lower().lstrip(".") in CUSTOM_SUFFIXES)
        return sorted(directory.glob(f"*.{suffix}"))

    def entries(self, source: str) -> Iterator[MeshEntry]:
        for path in self.available(source):
            yield MeshEntry(
                asset_id=f"{source}_{path.stem}",
                source=source,
                path=path,
                sha256=_sha256(path),
            )

    def counts(self) -> dict[str, int]:
        return {source: len(self.available(source)) for source in SUPPORTED_SOURCES}

    def is_ready(self, source: str, *, minimum: int = 1) -> bool:
        return len(self.available(source)) >= minimum

    def records(
        self, source: str, *, extent_mm: tuple[float, float, float], mass_kg: float
    ) -> list[AssetRecord]:
        """`AssetRecord`s for the manifest.

        `extent_mm` and `mass_kg` are placeholders the caller overrides per mesh once the geometry
        is read. They are arguments rather than guesses inside this module: a wrong size silently
        changes what "graspable" means, so it may not have a default hiding in a library.
        """
        return [
            AssetRecord(
                asset_id=entry.asset_id,
                source=entry.source,
                license=entry.license,
                kind="mesh",
                extent_mm=extent_mm,
                mass_kg=mass_kg,
                attribution=entry.attribution,
                mesh_path=str(entry.path),
            )
            for entry in self.entries(source)
        ]


def import_from_directory(
    source: str,
    origin: Path | str,
    *,
    destination: Path | str = MESH_LIBRARY_DIR,
    limit: int | None = None,
    license: str | None = None,
    attribution: str = "",
) -> list[MeshEntry]:
    """Copy meshes for one source into the library, recording a SHA for each.

    A local copy rather than a download, for meshes already on this machine: re-downloading hundreds
    of files to end up byte-identical is cost without benefit. The SHA is what makes the copy
    checkable either way, so an operator who fetches from the original source can confirm they hold
    the same bytes.
    """
    if source not in SUPPORTED_SOURCES:
        raise ValueError(f"unsupported source {source!r}; expected one of {sorted(SUPPORTED_SOURCES)}")

    # The licence is the customer's to declare, and refusing without it is the point rather than
    # friction. The licence audit reads these strings; a default here would put one into that audit
    # that nobody checked, and the audit would then pass a dataset on a guess. `own` is the ordinary
    # answer for parts a customer designed, and it passes.
    if source == "custom" and not (license or "").strip():
        raise ValueError(
            "importing `custom` meshes needs an explicit licence; nobody but you knows what your "
            "parts are licensed as, and this repo's CI audits that string. Pass license='own' for "
            "parts you designed, or the actual identifier (e.g. 'CC0-1.0') for anything you did not.")

    origin_dir = Path(origin)
    if not origin_dir.is_dir():
        raise FileNotFoundError(f"mesh origin not found: {origin_dir}")

    suffix = SUPPORTED_SOURCES[source]
    if suffix == "*":
        files = sorted(path for path in origin_dir.iterdir()
                       if path.suffix.lower().lstrip(".") in CUSTOM_SUFFIXES)
    else:
        files = sorted(origin_dir.glob(f"*.{suffix}"))
    if limit is not None:
        files = files[:limit]
    if not files:
        wanted = ("/".join(CUSTOM_SUFFIXES) if suffix == "*" else suffix)
        raise FileNotFoundError(f"no *.{wanted} meshes in {origin_dir}")

    target = Path(destination) / source
    target.mkdir(parents=True, exist_ok=True)

    imported: list[MeshEntry] = []
    for path in files:
        out = target / path.name
        if not out.exists() or out.stat().st_size != path.stat().st_size:
            shutil.copy2(path, out)
        imported.append(
            MeshEntry(
                asset_id=f"{source}_{out.stem}", source=source, path=out, sha256=_sha256(out)
            )
        )

    declared = (license or "").strip() or LICENSES.get(source, "")
    logger.info(
        "imported %d %s mesh(es) into %s (licence %s%s)",
        len(imported), source, target, declared or "UNDECLARED",
        f", attributed to {attribution}" if attribution else "",
    )
    if source == "custom":
        # Written beside the meshes, not held in memory. The declaration is what the manifest and
        # the licence audit read later, and a licence that lived only in the argument list of the
        # command that imported them would be gone by the next run.
        (Path(destination) / source / "LICENSE.txt").write_text(
            f"license: {declared}\nattribution: {attribution or '(none declared)'}\n"
            f"meshes: {len(imported)}\n"
            f"\nDeclared at import time by the operator. datagen audits this string; it does not\n"
            f"verify it. If it is wrong, every dataset built from these meshes carries it.\n",
            encoding="utf-8")
    return imported


def write_sha_index(entries: list[MeshEntry], path: Path | str) -> Path:
    """An `id,source,sha256,filename` index next to the meshes.

    Its job is verification, not bookkeeping: an operator who fetched from the original source can
    confirm they hold the same bytes this library recorded.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "source", "sha256", "filename"])
        for entry in sorted(entries, key=lambda e: e.asset_id):
            writer.writerow([entry.asset_id, entry.source, entry.sha256, entry.path.name])
    return out
