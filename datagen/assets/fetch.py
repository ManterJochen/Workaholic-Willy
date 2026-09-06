"""Fetch the real object meshes the scene generator draws from, into the gitignored mesh library.

    python -m datagen.assets.fetch --list
    python -m datagen.assets.fetch gso --limit 200
    python -m datagen.assets.fetch ycb
    python -m datagen.assets.fetch --all --limit 50        # a trial run, every source

Why a script and not a line in the README. The collections are published as per-object archives on
different services, in different container formats, with the mesh buried at a different depth in
each. "Download GSO and put the .obj files in assets/meshes/gso" is a morning of work the first time
and a footnote nobody follows the second.

Every endpoint below is probed rather than read off a wiki:

  gso   `https://fuel.gazebosim.org/1.0/GoogleResearch/models?page=N&per_page=M` lists them, with
        X-Total-Count 1033, and carries `license_name` per model. `.../models/<name>.zip` returns a
        3.5 MB zip holding `meshes/model.obj`, which is exactly what the library wants.
  ycb   `https://ycb-benchmarks.s3.amazonaws.com/data/objects.json` lists 103 objects, and
        `.../data/google/<id>_google_16k.tgz` returns an 11.7 MB tarball holding
        `<id>/google_16k/nontextured.ply`, which is what the library wants.

The licence is read from the source, not from this repository's map, for every source that publishes
one. `LICENSES` says CC-BY-4.0 for both collections and the asset audit checks that string; if a
model on Fuel ever ships under something else, believing the map is how a non-commercial mesh enters
a customer's dataset with a CC-BY label on it. Fuel states the licence per model, so this script
checks it and skips anything that is not CC0/CC-BY. YCB publishes no per-object licence, so its
meshes carry the collection's published terms and this script says so rather than implying it
verified something it did not.

Nothing is written into the repository: meshes land in `assets/meshes/<source>/`, which is
gitignored. `python -m datagen.assets --check` is what turns "they should be there" into an exit
code.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

__all__ = ["SOURCES", "fetch"]

# The line below is load-bearing. `python -m datagen.assets.fetch` puts the repository root on
# `sys.path` and does not need it, but an operator who runs the file itself gets only
# `datagen/assets/` on the path, and then `import datagen` fails. This file deliberately depends on
# the repository's own licence judgement (`is_noncommercial`) rather than carrying a second opinion
# about what NonCommercial means, so that import has to succeed on both routes. Running the module
# and running the file are two different environments, and only one of them needs this line.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_EXIT_OK, _EXIT_PROBLEM, _EXIT_USAGE = 0, 1, 2

#: Where the meshes go. One declaration, read from `datagen.constants`, which is stdlib-only. A copy
#: here would be a second path to hold in step with the one `assets/library.py` re-exports, and it
#: buys nothing: `datagen/assets/__init__.py` already pulls numpy on any import of this file. See
#: the note in `datagen.constants`.
from datagen.constants import MESH_LIBRARY_DIR as _DEFAULT_LIBRARY

#: Licences a fetched mesh may carry. The repo's own audit allows CC0 / CC-BY / own; a mesh that
#: arrives under anything else is skipped here rather than downloaded and refused later, because the
#: download is the expensive half.
_ACCEPTABLE = ("cc0", "cc-by", "creative-commons-attribution", "public-domain")


def _normalise_licence(text: str) -> str:
    """Lowercase, with every run of non-alphanumerics collapsed to a single hyphen.

    Without this the punctuation does the gating. A token check against raw lowercased text does not
    find `"cc-by"` in `"CC BY 4.0"`, which is exactly what one repository API returns, and does not
    find `"creative commons attribution"` in `"Creative Commons - Attribution"` either: two
    plainly permissive licences read as forbidden because of a space and a hyphen.
    """
    out = "".join(c if c.isalnum() else "-" for c in text.strip().lower())
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-")


def _licence_is_acceptable(licence: str) -> bool:
    """Whether a mesh under this licence may be fetched at all.

    The refusal runs first, and that ordering is the whole function. A permissive substring test on
    its own accepts "Creative Commons Attribution-NonCommercial 4.0", because the acceptable phrase
    "creative commons attribution" is literally inside the forbidden one. A gate built to keep
    non-commercial meshes out would then wave through every NC model Fuel publishes, and the dataset
    audit downstream would see a licence string that reads CC-BY.

    `is_noncommercial` is the repository's own judgement, reused rather than re-implemented: one
    place decides what NC means, and this script is not a second one.
    """
    from datagen.assets.licensing import is_noncommercial      # noqa: PLC0415 (keeps --list light)

    if not licence.strip():
        return False                    # silence is not permission
    text = _normalise_licence(licence)
    if is_noncommercial(text):
        return False
    if "nd" in text.split("-") or "noderiv" in text.replace("-", ""):
        # A rendered scene is a derivative work. NoDerivatives cannot be satisfied by attribution.
        return False
    if "sa" in text.split("-") or "sharealike" in text.replace("-", ""):
        # ShareAlike is copyleft: a derivative has to ship under the same terms, and a rendered
        # corpus is a derivative. The rule is explicit rather than lucky, because punctuation used
        # to refuse "Creative Commons - Attribution - Share Alike" by accident, and teaching the
        # gate to read separators would otherwise admit a whole collection it had been refusing.
        return False
    return any(token in text for token in _ACCEPTABLE)

_TIMEOUT_SECONDS = 180
_USER_AGENT = "workaholic-willy-mesh-fetch/1.0"


def use_system_trust_store() -> str:
    """Point Python's TLS at the OS certificate store. Returns what happened.

    A corporate proxy terminating TLS and re-signing with its own CA raises
    `CERTIFICATE_VERIFY_FAILED` here, and an enterprise network is exactly where this script will be
    run. Verification stays fully on; only the set of trusted roots changes, from the `certifi`
    bundle to the one an administrator actually curates on this machine.

    Optional: without `truststore` installed this returns and the fetch behaves as it did before.
    """
    try:
        import truststore
    except ImportError:
        return "truststore not installed; using certifi's bundled roots (the default)"
    truststore.inject_into_ssl()
    return "using the OS trust store (verification still ON, only the roots changed)"


def _encode(url: str) -> str:
    """Percent-encode the non-ASCII characters a model name may contain.

    Fuel ships model names with accented characters, and `urllib.request.Request` encodes the URL as
    ASCII, so an unencoded name kills a full fetch mid-run with `UnicodeEncodeError`. `safe` keeps
    the URL's own structure (scheme, path separators, query) intact and only escapes what actually
    needs it.
    """
    from urllib.parse import quote  # noqa: PLC0415

    return quote(url, safe=":/?&=%")


def _get(url: str, *, retries: int = 3, allow_missing: bool = False) -> bytes | None:
    """One GET, with retries. Raises the last error when they run out.

    ``allow_missing`` returns None on a 404 instead of raising, for callers where "not there" is an
    answer rather than a failure, which is exactly what walking off the end of a paginated catalogue
    is. Everything else still raises.
    """
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urlopen(Request(_encode(url), headers={"User-Agent": _USER_AGENT}),
                         timeout=_TIMEOUT_SECONDS) as response:
                return bytes(response.read())
        except (HTTPError, URLError, TimeoutError) as exc:                # noqa: PERF203
            last = exc
            if isinstance(exc, HTTPError) and exc.code == 404 and allow_missing:
                return None
            if isinstance(exc, HTTPError) and exc.code in (401, 403, 404):
                break            # not a transient failure; retrying just wastes the operator's time
            print(f"    retry {attempt + 1}/{retries}: {type(exc).__name__}: {exc}", flush=True)
    raise RuntimeError(f"GET {url} failed: {last}")


# --------------------------------------------------------------------------------- GSO, from Fuel

_FUEL = "https://fuel.gazebosim.org/1.0/GoogleResearch/models"


def _gso_catalogue(limit: int | None) -> list[tuple[str, str]]:
    """`(name, license_name)` for the models to fetch, in the order Fuel lists them."""
    out: list[tuple[str, str]] = []
    page = 1
    while limit is None or len(out) < limit:
        # Past the last page Fuel answers 404, not an empty list, and that is an answer rather than
        # an error. At 100 per page a 1,033-model catalogue ends on page 11 and page 12 is a 404;
        # treating it as a failure refuses the whole source after listing every page successfully,
        # so a full fetch downloads nothing at all.
        body = _get(f"{_FUEL}?page={page}&per_page=100", allow_missing=True)
        if body is None:
            break
        entries = json.loads(body.decode("utf-8"))
        if not entries:
            break
        for entry in entries:
            out.append((str(entry.get("name", "")), str(entry.get("license_name", ""))))
            if limit is not None and len(out) >= limit:
                break
        page += 1
    return out


def _gso_fetch(destination: Path, limit: int | None, report: Callable[[str], None]) -> dict[str, int]:
    counts = {"fetched": 0, "skipped_present": 0, "skipped_license": 0, "failed": 0}
    destination.mkdir(parents=True, exist_ok=True)
    catalogue = _gso_catalogue(limit)
    report(f"  {len(catalogue)} model(s) listed on Fuel")

    for index, (name, licence) in enumerate(catalogue, start=1):
        target = destination / f"{name}.obj"
        if target.exists():
            counts["skipped_present"] += 1
            continue
        # The source's own licence, checked before the download rather than after. See the module
        # docstring: trusting this repository's map here is how a mesh enters a customer's dataset
        # mislabelled.
        if not _licence_is_acceptable(licence):
            report(f"  [{index}/{len(catalogue)}] {name}: SKIPPED, licence {licence!r}")
            counts["skipped_license"] += 1
            continue
        try:
            archive = _get(f"{_FUEL}/{name}.zip")
            assert archive is not None                # only `allow_missing=True` can return None
            with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
                inner = next((n for n in bundle.namelist() if n.endswith("meshes/model.obj")), None)
                if inner is None:
                    report(f"  [{index}/{len(catalogue)}] {name}: no meshes/model.obj in the archive")
                    counts["failed"] += 1
                    continue
                target.write_bytes(bundle.read(inner))
        except (RuntimeError, zipfile.BadZipFile, OSError) as exc:
            report(f"  [{index}/{len(catalogue)}] {name}: {type(exc).__name__}: {exc}")
            counts["failed"] += 1
            continue
        counts["fetched"] += 1
        if counts["fetched"] % 25 == 0:
            report(f"  [{index}/{len(catalogue)}] {counts['fetched']} fetched")
    return counts


# ---------------------------------------------------------------------------------- YCB, from S3

_YCB = "https://ycb-benchmarks.s3.amazonaws.com/data"


def _ycb_fetch(destination: Path, limit: int | None, report: Callable[[str], None]) -> dict[str, int]:
    counts = {"fetched": 0, "skipped_present": 0, "skipped_license": 0, "skipped_absent": 0,
              "failed": 0}
    destination.mkdir(parents=True, exist_ok=True)
    index_body = _get(f"{_YCB}/objects.json")
    assert index_body is not None                     # only `allow_missing=True` can return None
    objects = list(json.loads(index_body.decode("utf-8")).get("objects", []))
    if limit is not None:
        objects = objects[:limit]
    report(f"  {len(objects)} object(s) listed")

    for index, object_id in enumerate(objects, start=1):
        target = destination / f"{object_id}.ply"
        if target.exists():
            counts["skipped_present"] += 1
            continue
        try:
            archive = _get(f"{_YCB}/google/{object_id}_google_16k.tgz")
            assert archive is not None                # only `allow_missing=True` can return None
        except RuntimeError:
            # Expected, not an error: not every YCB object has a Google scan, some were captured
            # only on the Berkeley rig. Counting these as failures would make a healthy run look
            # broken, and hide the ones that really did fail.
            report(f"  [{index}/{len(objects)}] {object_id}: no google_16k scan, skipping")
            counts["skipped_absent"] += 1
            continue
        try:
            with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
                inner = next((m for m in bundle.getmembers()
                              if m.name.endswith("google_16k/nontextured.ply")), None)
                if inner is None:
                    report(f"  [{index}/{len(objects)}] {object_id}: no nontextured.ply in the archive")
                    counts["failed"] += 1
                    continue
                handle = bundle.extractfile(inner)
                if handle is None:
                    counts["failed"] += 1
                    continue
                target.write_bytes(handle.read())
        except (tarfile.TarError, OSError) as exc:
            report(f"  [{index}/{len(objects)}] {object_id}: {type(exc).__name__}: {exc}")
            counts["failed"] += 1
            continue
        counts["fetched"] += 1
        if counts["fetched"] % 10 == 0:
            report(f"  [{index}/{len(objects)}] {counts['fetched']} fetched")
    return counts


# ------------------------------------------------------------------------------------- catalogue

# ------------------------------------------------------------------- Thingi10K, the CC0/PD slice

_THINGI = "https://huggingface.co/datasets/Thingi10K/Thingi10K/resolve/main"

#: The collection authors in millimetres; `measure_mesh` reads metres.
_MM_TO_M = 0.001

#: The filter is an exact match, and it admits only the two licences that oblige no attribution.
#:
#: Exact, because "Creative Commons - Attribution" is a prefix of "Creative Commons - Attribution -
#: Non-Commercial". One `in` where an `==` belongs classifies most of the collection as permitted,
#: NC, ND and SA rows included.
#:
#: Only these two, because the metadata carries no author. Its columns are ID, Thing ID, License,
#: Link and geometry flags, and nothing else. These are individually authored user uploads, so a
#: CC-BY row cannot be attributed from anything on hand, and `library.ATTRIBUTIONS` already records
#: why that is a refusal rather than a gap to route around. CC0 and public domain oblige nothing, so
#: they are the part of this collection that can be honoured.
_THINGI_NO_ATTRIBUTION_REQUIRED = frozenset({
    "Creative Commons - Public Domain Dedication",
    "Public Domain",
})


def _thingi_rows(limit: int | None) -> list[dict[str, str]]:
    """The fetchable rows: licence-clean, closed, one component, manifold, one file per design."""
    import csv                                        # noqa: PLC0415 (keeps `--list` light)

    body = _get(f"{_THINGI}/metadata/input_summary.csv")
    assert body is not None                           # only `allow_missing=True` can return None
    rows = list(csv.DictReader(io.StringIO(body.decode("utf-8", errors="replace"))))

    def yes(row: dict, field: str) -> bool:
        return str(row.get(field, "")).strip().lower() in ("true", "1", "yes")

    kept: dict = {}
    for row in rows:
        if row.get("License", "") not in _THINGI_NO_ATTRIBUTION_REQUIRED:
            continue
        # `Closed` is the collection's own watertight flag. The loader re-checks it, but downloading
        # a mesh already known to be open is paying for a refusal.
        if not (yes(row, "Closed") and yes(row, "Single Component")
                and yes(row, "Edge manifold") and yes(row, "Vertex manifold")):
            continue
        # 10,000 files are only 2,011 designs. The same design ships as several parts, so without
        # this the bank carries near-duplicates and the held-out split leaks across them.
        kept.setdefault(str(row["Thing ID"]), row)
    out = sorted(kept.values(), key=lambda r: int(r["ID"]))
    return out[:limit] if limit else out


def _thingi_fetch(destination: Path, limit: int | None,
                  report: Callable[[str], None]) -> dict[str, int]:
    counts = {"fetched": 0, "skipped_present": 0, "skipped_license": 0, "failed": 0}
    destination.mkdir(parents=True, exist_ok=True)
    catalogue = _thingi_rows(limit)
    report(f"  {len(catalogue)} design(s) after licence, solidity and de-duplication")

    for index, row in enumerate(catalogue, start=1):
        file_id = str(row["ID"])
        target = destination / f"thing{row['Thing ID']}_{file_id}.obj"
        if target.exists():
            counts["skipped_present"] += 1
            continue
        # Belt and braces: the exact-match filter above already decided, and this is the shared gate
        # every other source runs through. If the two ever disagree, no download happens.
        if not _licence_is_acceptable(row.get("License", "")):
            counts["skipped_license"] += 1
            continue
        try:
            # The `Link` column is dead. It points at the original bucket and returns HTTP 403 on
            # every attempt. The mirror's npz carries `vertices` and `facets` and is the live route.
            import numpy as np                        # noqa: PLC0415 (keeps `--list` light)

            body = _get(f"{_THINGI}/npz/{file_id}.npz", allow_missing=True)
            if body is None:
                report(f"  [{index}/{len(catalogue)}] {file_id}: absent from the mirror")
                counts["failed"] += 1
                continue
            with np.load(io.BytesIO(body)) as bundle:
                vertices = np.asarray(bundle["vertices"], dtype=np.float64)
                facets = np.asarray(bundle["facets"], dtype=np.int64)
            if facets.ndim != 2 or facets.shape[1] != 3 or not len(vertices):
                report(f"  [{index}/{len(catalogue)}] {file_id}: not a triangle mesh")
                counts["failed"] += 1
                continue
            # Millimetres in, metres out. This collection authors in real millimetres, and
            # `measure_mesh` reads every mesh as metres and scales by 1000, so writing the vertices
            # through unchanged lands a 127 mm bracket at 127,000 mm and a mass wrong by the cube of
            # the factor. The units guard refuses that by name on the first load, which is the only
            # reason it does not reach the corpus silently.
            vertices = vertices * _MM_TO_M
            lines = [f"v {x:.6g} {y:.6g} {z:.6g}" for x, y, z in vertices]
            lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in facets]   # OBJ indexes from 1
            target.write_text("\n".join(lines) + "\n", encoding="ascii")
        except (RuntimeError, ValueError, OSError) as exc:
            report(f"  [{index}/{len(catalogue)}] {file_id}: {type(exc).__name__}: {exc}")
            counts["failed"] += 1
            continue
        counts["fetched"] += 1
        if counts["fetched"] % 10 == 0:
            report(f"  [{index}/{len(catalogue)}] {counts['fetched']} fetched")
    return counts


# ------------------------------------------------------------ Coles Object Set, from the repository

_ASOS_ARTICLE = "https://api.figshare.com/v2/articles/20179550"


def _asos_fetch(destination: Path, limit: int | None,
                report: Callable[[str], None]) -> dict[str, int]:
    """Supermarket objects: boxes, cylinders and packets, which is the jaw's own geometry.

    The licence is in the repository record and nowhere else. Neither the paper nor the project page
    states one, so reading either alone produces a fail-closed refusal. The API answers
    `license.name = "CC BY 4.0"` for the whole record and it names an author, which is what makes
    this a collection-level attribution like the other two rather than a per-file problem.

    These are dense scans, and they need decimating before a convex decomposition can run on them:
    almost all of an archive's size is triangle count.
    """
    counts = {"fetched": 0, "skipped_present": 0, "skipped_license": 0, "failed": 0}
    destination.mkdir(parents=True, exist_ok=True)
    record = json.loads((_get(_ASOS_ARTICLE) or b"{}").decode("utf-8"))
    licence = str((record.get("license") or {}).get("name", ""))
    if not _licence_is_acceptable(licence):
        report(f"  REFUSED: the record reports licence {licence!r}")
        return {**counts, "skipped_license": 1}
    report(f"  licence {licence!r} from the repository record, author "
           f"{', '.join(a.get('full_name', '?') for a in record.get('authors', []))}")

    archives = sorted(record.get("files", []), key=lambda f: str(f.get("name", "")))
    archives = archives[:limit] if limit else archives
    for index, meta in enumerate(archives, start=1):
        name = str(meta.get("name", ""))
        try:
            body = _get(str(meta["download_url"]))
            assert body is not None
            digest = hashlib.md5(body).hexdigest()    # noqa: S324 (integrity, not a secret)
            if digest != str(meta.get("computed_md5", digest)):
                report(f"  [{index}/{len(archives)}] {name}: md5 mismatch, discarded")
                counts["failed"] += 1
                continue
            with zipfile.ZipFile(io.BytesIO(body)) as bundle:
                members = [n for n in bundle.namelist() if n.lower().endswith(".ply")]
                if not members:
                    report(f"  [{index}/{len(archives)}] {name}: no .ply inside")
                    counts["failed"] += 1
                    continue
                for member in members:
                    target = destination / Path(member).name
                    if target.exists():
                        counts["skipped_present"] += 1
                        continue
                    target.write_bytes(bundle.read(member))
                    counts["fetched"] += 1
        except (RuntimeError, zipfile.BadZipFile, OSError, KeyError) as exc:
            report(f"  [{index}/{len(archives)}] {name}: {type(exc).__name__}: {exc}")
            counts["failed"] += 1
            continue
        report(f"  [{index}/{len(archives)}] {name}: {counts['fetched']} mesh(es) so far")
    return counts



# ------------------------------------------------------------------ Objaverse 1.0, per-object licence

_OBJAVERSE = "https://huggingface.co/datasets/allenai/objaverse/resolve/main"

#: How many metadata shards to look at, each holding 5,000 entries. Read lazily and only as far as a
#: run needs: 800k entries is about 160 MB of gzip and no run wants all of them.
#:
#: Taking the first N shards is a sample, not a prefix in the sense this repository warns about.
#: Objects are sharded by uid, which is a hash, so shard order carries no semantics. Worth stating
#: because "a prefix is not a sample" is a rule here, and this is the exception that has a reason.
_OBJAVERSE_SHARDS = 160

#: The collection's short licence codes, mapped to strings the shared gate understands.
#:
#: An unknown code raises rather than defaulting. `"by"` alone is not in the acceptable list, so an
#: unmapped code would be refused anyway, but silently, as though the object were non-commercial. A
#: code never seen before means the collection changed and somebody should look, not that thousands
#: of objects should vanish from a fetch without a word.
#:
#: The mapped strings spell every clause out, so ShareAlike and NoDerivatives reach the gate as
#: their own rules. A naive `startswith("cc-by")` test admits tens of thousands of ShareAlike
#: objects here, and ShareAlike is copyleft over a corpus that is a derivative work.
_OBJAVERSE_LICENCES: dict[str, str] = {
    "by": "cc-by-4.0",
    "cc0": "cc0-1.0",
    "by-sa": "cc-by-sa-4.0",
    "by-nc": "cc-by-nc-4.0",
    "by-nc-sa": "cc-by-nc-sa-4.0",
    "by-nd": "cc-by-nd-4.0",
    "by-nc-nd": "cc-by-nc-nd-4.0",
}


def _objaverse_rows(limit, shards):
    """Candidate objects: licence-clean, attributable, one row each.

    Attribution is a fetch condition here, not a later worry. CC-BY obliges naming the author, and
    the sibling collection in this file is capped to its no-attribution slice for exactly the reason
    that its metadata carries none. This one does carry it: every entry holds `user.displayName` and
    `user.profileUrl` beside the model's own uri. A row missing either is skipped rather than
    downloaded and credited to nobody.
    """
    import gzip                                        # noqa: PLC0415 (keeps `--list` light)

    kept = []
    unknown = {}
    for shard in range(shards):
        body = _get(f"{_OBJAVERSE}/metadata/{shard:03d}-000.json.gz", allow_missing=True)
        if body is None:
            continue
        entries = json.loads(gzip.decompress(body).decode("utf-8"))
        for uid, entry in entries.items():
            code = str(entry.get("license", ""))
            licence = _OBJAVERSE_LICENCES.get(code)
            if licence is None:
                unknown[code] = unknown.get(code, 0) + 1
                continue
            if not _licence_is_acceptable(licence):
                continue
            user = entry.get("user") or {}
            author = str(user.get("displayName") or user.get("username") or "")
            profile = str(user.get("profileUrl") or "")
            if not author or not profile:
                continue                                # cannot be attributed, so cannot be fetched
            kept.append({"uid": uid, "name": str(entry.get("name", "")), "licence": licence,
                         "author": author, "profile": profile, "uri": str(entry.get("uri", ""))})
            if limit and len(kept) >= limit:
                return kept
    if unknown:
        raise RuntimeError(f"unmapped licence code(s) {sorted(unknown)}; the collection has changed "
                           f"and _OBJAVERSE_LICENCES needs a decision rather than a default")
    return kept


def _objaverse_fetch(destination: Path, limit, report) -> dict:
    """Artist-uploaded models, licensed per object, converted from glb to obj on the way in.

    `limit` is not a convenience here, it is most of the interface: the whole collection is
    terabytes, and a run without a limit would try to pull all of it. Each model averages about
    11 MB.
    """
    import gzip                                        # noqa: PLC0415 (keeps `--list` light)

    import trimesh                                     # noqa: PLC0415 (heavy, and only needed here)

    counts = {"fetched": 0, "skipped_present": 0, "skipped_license": 0, "failed": 0}
    destination.mkdir(parents=True, exist_ok=True)
    rows = _objaverse_rows(limit, _OBJAVERSE_SHARDS)
    report(f"  {len(rows)} object(s) after licence and attributability")
    if not rows:
        return counts

    paths_body = _get(f"{_OBJAVERSE}/object-paths.json.gz")
    assert paths_body is not None
    paths = json.loads(gzip.decompress(paths_body).decode("utf-8"))

    # The declaration the pipeline audits, and without it the whole fetch is wasted. Every asset row
    # is checked for a `license` before a build boots the renderer, and the string comes from a
    # LICENSE.txt beside the meshes for any collection whose terms are not a per-source constant.
    # Objaverse is exactly that case: with no such file, downloaded, normalised and screened meshes
    # are refused at build time with "missing license", after the operator has paid for all three
    # steps.
    #
    # The strictest licence present, not an average and not the commonest. Every object here is
    # either CC-BY-4.0 or CC0-1.0, and CC-BY is the one that obliges naming the author, so declaring
    # it applies that obligation to all of them. Over-crediting a CC0 object costs nothing;
    # under-crediting a CC-BY one is the failure this repository fences everywhere.
    declaration = destination / "LICENSE.txt"
    if not declaration.exists():
        declaration.write_text(
            "license: cc-by-4.0\n"
            "attribution: per object, see ATTRIBUTION.tsv beside this file\n"
            "note: every object is CC-BY-4.0 or CC0-1.0; "
            "the stricter of the two is declared here\n",
            encoding="utf-8")

    attribution = destination / "ATTRIBUTION.tsv"
    if not attribution.exists():
        attribution.write_text("uid\tname\tauthor\tprofile\turi\tlicence\n", encoding="utf-8")
    credited = {line.split("\t")[0] for line in
                attribution.read_text(encoding="utf-8").splitlines()[1:] if line}

    for index, row in enumerate(rows, start=1):
        uid = row["uid"]
        target = destination / f"objaverse_{uid}.obj"
        if target.exists():
            counts["skipped_present"] += 1
            continue
        remote = paths.get(uid)
        if not remote:
            counts["failed"] += 1
            continue
        try:
            body = _get(f"{_OBJAVERSE}/{remote}", allow_missing=True)
            if body is None:
                report(f"  [{index}/{len(rows)}] {uid}: absent from the mirror")
                counts["failed"] += 1
                continue
            mesh = trimesh.load(io.BytesIO(body), file_type="glb", force="mesh")
            # Both attributes, not just `vertices`. A glb can hold a point cloud or a scene: it
            # passes a vertices-only test, has no faces, and `mesh.faces` is read four lines down,
            # so it would fall into the `except AttributeError` below and be counted as "failed"
            # with the exception's own text in place of a reason.
            if not hasattr(mesh, "vertices") or not len(mesh.vertices):
                report(f"  [{index}/{len(rows)}] {uid}: no geometry in the glb")
                counts["failed"] += 1
                continue
            faces = getattr(mesh, "faces", None)
            if faces is None or not len(faces):
                report(f"  [{index}/{len(rows)}] {uid}: the glb holds points but no faces, "
                       f"so it cannot be closed into a solid")
                counts["failed"] += 1
                continue
            # Metres out, like every other source here. glTF fixes the unit as metres in its own
            # spec, so nothing is scaled. The rule is stated anyway because the sibling collection
            # authors in millimetres, and writing its vertices through unchanged lands a 127 mm
            # bracket at 127 metres, caught by the units guard rather than by the corpus.
            lines = [f"v {x:.6g} {y:.6g} {z:.6g}" for x, y, z in mesh.vertices]
            lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces]
            target.write_text("\n".join(lines) + "\n", encoding="ascii")
        except (RuntimeError, ValueError, OSError, KeyError, AttributeError) as exc:
            report(f"  [{index}/{len(rows)}] {uid}: {type(exc).__name__}: {exc}")
            counts["failed"] += 1
            continue
        if uid not in credited:
            with attribution.open("a", encoding="utf-8") as handle:
                handle.write(f"{uid}\t{row['name']}\t{row['author']}\t{row['profile']}\t"
                             f"{row['uri']}\t{row['licence']}\n")
            credited.add(uid)
        counts["fetched"] += 1
        if counts["fetched"] % 25 == 0:
            report(f"  [{index}/{len(rows)}] {counts['fetched']} fetched")
    return counts


@dataclass(frozen=True, slots=True)
class Source:
    key: str
    title: str
    objects: int
    approx_mb_each: float
    licence: str
    licence_is_verified: bool
    fetch: Callable[[Path, int | None, Callable[[str], None]], dict[str, int]]

    @property
    def approx_gb(self) -> float:
        return self.objects * self.approx_mb_each / 1000.0


#: Fuel reports X-Total-Count 1033 and a 3.5 MB archive for one model; YCB's objects.json lists 103
#: and one archive is 11.7 MB. The per-object figures are the archive size, which is what the
#: network pays; the extracted mesh is much smaller.
SOURCES: dict[str, Source] = {
    "objaverse": Source(
        key="objaverse", title="Objaverse 1.0 (Deitke et al., AI2), per-object Creative Commons",
        objects=724500, approx_mb_each=11.0,
        licence="ODC-By-1.0 / per object CC-BY-4.0 or CC0-1.0",
        licence_is_verified=True, fetch=_objaverse_fetch),
    "gso": Source(
        key="gso", title="Google Scanned Objects (Downs et al., Google Research)",
        objects=1033, approx_mb_each=3.5, licence="CC-BY-4.0",
        licence_is_verified=True, fetch=_gso_fetch),
    "ycb": Source(
        key="ycb", title="YCB Object and Model Set (Calli et al., Yale / CMU / UC Berkeley)",
        objects=103, approx_mb_each=11.7, licence="CC-BY-4.0",
        licence_is_verified=False, fetch=_ycb_fetch),
    # The collection advertises 10,000 files across 2,011 designs. The count below is the slice that
    # obliges no attribution and also passes closed, single-component and manifold. Thirty is the
    # honest number: the rest is CC-BY with no author column to attribute it from.
    "thingi10k": Source(
        key="thingi10k", title="Thingi10K (Zhou and Jacobson), CC0 / public-domain slice only",
        objects=30, approx_mb_each=2.0, licence="CC0-1.0",
        licence_is_verified=True, fetch=_thingi_fetch),
    # Two figures differ by nearly a factor of two. The repository API reports 8 archives, which is
    # what the network pays; the extracted meshes are what the disk pays, and the estimate below is
    # the disk figure, because that is the one that runs a machine out of room. Roughly 280 MB per
    # dense scan, which is also why these need decimating before a convex decomposition can run.
    "asos": Source(
        key="asos", title="Coles Object Set (Chumbley, Monash University)",
        objects=50, approx_mb_each=280.0, licence="CC-BY-4.0",
        licence_is_verified=True, fetch=_asos_fetch),
}


def fetch(keys: list[str], *, library: Path, limit: int | None,
          report: Callable[[str], None] = print) -> int:
    """Fetch each named source. Returns a process exit code.

    The work is in `assets.service.MeshPreparation.fetch`, which returns a `FetchReport`. This is
    the thin shell half: a shell needs an exit code and a program needs a report, and the same run
    produces both.
    """
    from datagen.assets.service import MeshPreparation           # noqa: PLC0415

    result = MeshPreparation.from_sources(keys).fetch(
        library=library, limit=limit, report=report)
    report(result.render())
    return _EXIT_OK if result.ok else _EXIT_PROBLEM


def _list(report: Callable[[str], None] = print) -> int:
    report(f"  {'source':8} {'objects':>8} {'~GB':>7}  licence")
    for source in SOURCES.values():
        mark = "verified per model" if source.licence_is_verified else "collection terms only"
        report(f"  {source.key:8} {source.objects:8d} {source.approx_gb:7.1f}  "
               f"{source.licence} ({mark})")
    report("\n  --limit N takes the first N of a collection, for a trial run.")
    report("  Already-present meshes are skipped, so an interrupted fetch resumes.")
    return _EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m datagen.assets.fetch",
        description="Fetch the object meshes the scene generator draws from.")
    parser.add_argument("sources", nargs="*", choices=[*SOURCES, []],
                        help="which collections to fetch")
    parser.add_argument("--all", action="store_true", help="fetch every source")
    parser.add_argument("--list", action="store_true", help="what is available, and what it costs")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="fetch at most N objects per source (a trial run)")
    parser.add_argument("--library", default=str(_DEFAULT_LIBRARY), metavar="DIR",
                        help="where meshes land (default: assets/meshes, gitignored)")
    args = parser.parse_args(argv)

    if args.list:
        return _list()
    keys = list(SOURCES) if args.all else list(args.sources)
    if not keys:
        parser.print_help()
        print("\nNothing named. Try --list, or a source name, or --all.", file=sys.stderr)
        return _EXIT_USAGE
    if shutil.disk_usage(Path(args.library).parent if Path(args.library).parent.exists()
                         else Path(".")).free < 2_000_000_000:
        print("less than 2 GB free on this volume; these collections are gigabytes", file=sys.stderr)
        return _EXIT_PROBLEM
    return fetch(keys, library=Path(args.library), limit=args.limit)


if __name__ == "__main__":                                           # pragma: no cover
    raise SystemExit(main())
