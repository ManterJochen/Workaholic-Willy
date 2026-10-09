"""Getting a customer's meshes ready, as one noun with four verbs, callable without a command line.

Every step a customer performs on their own parts before a single scene is rendered belongs here:
which collections to walk, how to draw a sample from them, warming the convex-decomposition cache,
and reading a screen report back to ask why a mesh earns no grasp. `decompose`, `normalise-meshes`,
`screen-meshes` and `why-no-jaw` are the four verbs, and what decides the list of meshes is part of
the noun rather than of a command-line handler.

The trap that decision keeps out of handlers: asset ids sort by source and then by name, so
`sorted(...)[:N]` is not a sample of the library, it is one collection. `why-no-jaw --limit 20`
taking the first twenty returns twenty meshes from the first source and none from any other, and
answers "why does this mesh earn no jaw label" about one dataset while wearing the name of all of
them. A held-out set drawn the same way measures one collection too.

:func:`even_sample` is that draw, as a named function rather than four lines inside a handler that
the next caller will write again.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:  # pragma: no cover (imported for annotations only)
    from datagen.config import DatagenConfig

__all__ = ["MeshPreparation", "FetchReport", "SourceFetch", "DecompositionReport",
           "ScreenReport", "JawDiagnosis", "even_sample", "library_sources",
           "available_sources"]

_NEWLINE = chr(10)


def library_sources(named: Sequence[str] | None = None) -> list[str]:
    """The collections to walk: the ones named, or every supported one except `custom`.

    One rule for `screen-meshes`, `normalise-meshes` and `why-no-jaw` alike. A source the fetcher
    can download but `SUPPORTED_SOURCES` does not carry is invisible to all three, so those two
    lists have to stay in step.
    """
    from datagen.assets.library import SUPPORTED_SOURCES  # noqa: PLC0415

    return list(named) if named else [s for s in SUPPORTED_SOURCES if s != "custom"]


def even_sample(items: Sequence[Any], limit: int) -> list[Any]:
    """`limit` items spread across the whole list, never a prefix of it.

    Asset ids sort by source, so a prefix of a sorted asset list is one collection: it describes
    that one collection while calling itself the library.
    """
    if limit <= 0 or len(items) <= limit:
        return list(items)
    import numpy as np  # noqa: PLC0415

    picks = np.linspace(0, len(items) - 1, limit).astype(int)
    return [items[int(i)] for i in picks]


def available_sources() -> "list[dict[str, Any]]":
    """Every public mesh collection that can be downloaded, how big it is, and on what licence.

    Returns:
        list[dict[str, Any]]: One dict per collection: ``key``, ``title``, ``objects``, ``approx_gb``, ``licence`` and
            ``licence_verified``. ``False`` there means the collection publishes terms for itself and states nothing per
            object, so a dataset shipped on it inherits the collection's claim rather than a verification.
    """
    from datagen.assets.fetch import SOURCES  # noqa: PLC0415

    return [{"key": s.key, "title": s.title, "objects": s.objects, "approx_gb": s.approx_gb,
             "licence": s.licence, "licence_verified": s.licence_is_verified}
            for s in SOURCES.values()]


@dataclass(frozen=True, slots=True)
class SourceFetch:
    """What one collection's download did. Counts, not a verdict."""

    key: str
    fetched: int
    already_present: int
    skipped_licence: int
    skipped_absent: int
    failed: int
    licence: str
    licence_verified: bool
    refused: str = ""

    @property
    def ok(self) -> bool:
        """True when nothing failed and nothing was refused.

        `skipped_absent` is not a failure: some objects in a collection have no scan at this source
        at all. Counting those as errors makes a healthy run look broken and buries the ones that
        really did fail.
        """
        return not self.failed and not self.refused


@dataclass(frozen=True, slots=True)
class FetchReport:
    """Every collection asked for, and the attribution obligation that comes with them."""

    library: Path
    sources: tuple[SourceFetch, ...]

    @property
    def ok(self) -> bool:
        return all(s.ok for s in self.sources)

    @property
    def fetched(self) -> int:
        return sum(s.fetched for s in self.sources)

    def render(self) -> str:
        lines = [f"library: {self.library}"]
        for s in self.sources:
            if s.refused:
                lines.append(f"  {s.key:10s} REFUSED: {s.refused}")
                continue
            parts = [f"{s.fetched} fetched", f"{s.already_present} already present",
                     f"{s.skipped_licence} skipped on licence"]
            if s.skipped_absent:
                parts.append(f"{s.skipped_absent} have no scan at this source")
            parts.append(f"{s.failed} failed")
            lines.append(f"  {s.key:10s} " + ", ".join(parts))
            checked = ("CHECKED per model against the source" if s.licence_verified
                       else "from the collection's published terms, nothing here verified a model")
            lines.append(f"  {'':10s} licence {s.licence}, {checked}")
        lines.append("Attribution is obligatory for CC-BY. Run "
                     "`python -m datagen.assets --attribution` for the text that must ship with any "
                     "dataset built from these meshes.")
        return _NEWLINE.join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {"library": str(self.library), "ok": self.ok, "fetched": self.fetched,
                "sources": [{"key": s.key, "fetched": s.fetched,
                             "already_present": s.already_present,
                             "skipped_licence": s.skipped_licence,
                             "skipped_absent": s.skipped_absent, "failed": s.failed,
                             "licence": s.licence, "licence_verified": s.licence_verified,
                             "refused": s.refused, "ok": s.ok} for s in self.sources]}


@dataclass(frozen=True, slots=True)
class DecompositionReport:
    """What the convex-decomposition cache warm produced."""

    assets: int
    computed: int
    already_cached: int
    parts: int
    cache_dir: Path

    @property
    def parts_per_asset(self) -> float:
        return self.parts / self.assets if self.assets else 0.0

    def render(self) -> str:
        return _NEWLINE.join([
            f"{self.parts} convex part(s) over {self.assets} asset(s), "
            f"{self.parts_per_asset:.1f} per asset",
            f"  {self.computed} computed, {self.already_cached} already cached -> {self.cache_dir}"])

    def as_dict(self) -> dict[str, Any]:
        return {"assets": self.assets, "computed": self.computed,
                "already_cached": self.already_cached, "parts": self.parts,
                "parts_per_asset": self.parts_per_asset, "cache_dir": str(self.cache_dir)}


@dataclass(frozen=True, slots=True)
class ScreenReport:
    """What a graspability screen wrote, and where."""

    screen_path: Path
    ids_path: Path | None
    rows: int
    graspable: int
    raw: dict[str, Any] | None = None

    def render(self) -> str:
        lines = [f"{self.graspable} of {self.rows} mesh(es) earn a jaw label",
                 f"wrote {self.screen_path}"]
        if self.ids_path is not None:
            lines.append(f"      {self.ids_path}")
        return _NEWLINE.join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {"screen_path": str(self.screen_path),
                "ids_path": str(self.ids_path) if self.ids_path else None,
                "rows": self.rows, "graspable": self.graspable}


@dataclass(frozen=True, slots=True)
class JawDiagnosis:
    """Why a set of meshes earns no grasp.

    The pooled rejection histogram refutes the obvious guess. Thin flat objects look as though the
    fingers must meet the table underneath them, but `below_table` is a small share of the
    rejections and `too_wide` dominates. What refuses them is the width along the line tried.
    """

    raw: dict[str, Any]
    probed_meshes: int

    @property
    def reasons(self) -> dict[str, Any]:
        return dict(self.raw.get("reasons", {}))

    def render(self) -> str:
        raw = self.raw
        lines = [f"{raw['probed']} probed, {raw['unreadable']} unreadable "
                 f"(a mesh that cannot be closed into a solid is not a labelling failure)",
                 f"{raw['with_no_label_in_any_pose']} earned NOTHING in any rest pose",
                 f"{raw['rescued_by_a_different_pose']} were rescued by lying the object down",
                 "why the upright pose refused them, pooled over every candidate tried:"]
        for name, block in list(self.reasons.items())[:8]:
            lines.append(f"  {name:24s} {block['count']:>10,}  {block['share'] * 100:5.1f} %")
        return _NEWLINE.join(lines)

    def as_dict(self) -> dict[str, Any]:
        """The diagnosis without `verdicts`, which is the per-candidate table.

        That table is the bulk of the object, and a summary that carries it is not a summary.
        """
        return {k: v for k, v in self.raw.items() if k != "verdicts"}


@dataclass
class MeshPreparation:
    """Your mesh collections, fetched, normalised, screened for graspability, decomposed and diagnosed.

        prep = MeshPreparation.from_sources(["custom"])
        print(prep.describe())
        print(prep.screen("screen.json"))          # which meshes a jaw can grasp at all

    The ``custom`` collection, your own meshes, is excluded from the default sweep: nobody else can re-fetch it, so a
    bulk operation never touches it unless it is named.

    Attributes:
        sources (tuple[str, ...]): The collections to walk.
        library (Any): The mesh library folder; ``None`` is the repository's ``MESH_LIBRARY_DIR`` (default: None).
    """

    sources: tuple[str, ...]
    library: Any = None

    @classmethod
    def from_sources(cls, sources: Sequence[str] | None = None,
                     *, library: Any = None) -> "MeshPreparation":
        """The preparation over the named collections.

        Args:
            sources (Sequence[str] | None): The collections, such as ``["custom"]`` or ``["ycb", "gso"]``; ``None`` is
                every supported one but ``custom`` (default: None).
            library (Any): The mesh library folder; ``None`` the repository's (default: None).

        Returns:
            MeshPreparation: The preparation; nothing is read yet.
        """
        return cls(sources=tuple(library_sources(sources)), library=library)

    def describe(self) -> str:
        """Which collections this would touch, before anything reads a mesh.

        Returns:
            str: ASCII, one line per fact.
        """
        return _NEWLINE.join([
            f"mesh preparation over {len(self.sources)} collection(s)",
            f"  sources        {', '.join(self.sources)}"])

    def entries(self) -> list[tuple[str, str]]:
        """Every mesh in the chosen collections, in library order.

        Returns:
            list[tuple[str, str]]: ``(asset_id, path)`` pairs.
        """
        from datagen.assets.library import MeshLibrary  # noqa: PLC0415

        library = self.library if self.library is not None else MeshLibrary()
        return [(entry.asset_id, str(entry.path))
                for source in self.sources for entry in library.entries(source)]

    def fetch(self, *, library: str | Path | None = None, limit: int | None = None,
              report: Any = None) -> FetchReport:
        """Download the chosen collections: step zero of everything else here. A mesh already present is skipped, so an
        interrupted fetch resumes.

        Args:
            library (str | Path | None): Where the meshes go; ``None`` is the repository's library (default: None).
            limit (int | None): Take the first n of each collection: a trial run, not a sample (``even_sample`` is the
                sample); ``None`` all of them (default: None).
            report (Any): Called with one progress line at a time; ``None`` says nothing (default: None).

        Returns:
            FetchReport: The library folder and, per collection, what was fetched, skipped and refused.
        """
        from datagen.assets.fetch import SOURCES, use_system_trust_store  # noqa: PLC0415
        from datagen.assets.library import MESH_LIBRARY_DIR  # noqa: PLC0415

        root = Path(library) if library is not None else Path(MESH_LIBRARY_DIR)
        emit = report if report is not None else (lambda _line: None)
        emit(f"TLS: {use_system_trust_store()}")
        results: list[SourceFetch] = []
        for key in self.sources:
            source = SOURCES[key]
            licence: str = source.licence
            verified: bool = source.licence_is_verified
            try:
                counts = source.fetch(root / key, limit, emit)
            except (RuntimeError, OSError) as exc:
                results.append(SourceFetch(
                    key=key, fetched=0, already_present=0, skipped_licence=0, skipped_absent=0,
                    failed=0, licence=licence, licence_verified=verified,
                    refused=f"{type(exc).__name__}: {exc}"))
                continue
            results.append(SourceFetch(
                key=key, fetched=counts["fetched"],
                already_present=counts["skipped_present"],
                skipped_licence=counts["skipped_license"],
                skipped_absent=counts.get("skipped_absent", 0),
                failed=counts["failed"], licence=licence, licence_verified=verified))
        return FetchReport(library=root, sources=tuple(results))

    def normalise(self, **kwargs: Any) -> dict[str, Any]:
        """Scale every chosen collection into metres and decimate it into the face budget. Idempotent.

        Args:
            **kwargs (Any): Passed to ``normalise_meshes``: ``faces`` (the face budget, default 20000), ``scale`` (a
                factor to apply, ``None`` to infer), ``jobs`` (default 4) and ``report`` (a progress callback).

        Returns:
            dict[str, Any]: Per collection, how many meshes were read and changed.
        """
        from datagen.assets.prepare import normalise_meshes  # noqa: PLC0415

        return {source: normalise_meshes(source, library=self.library, **kwargs)
                for source in self.sources}

    def screen(self, out: str | Path, *, write_ids: bool = True, **kwargs: Any) -> ScreenReport:
        """Label every mesh alone, in every rest pose, and write what the labeller found, with the asset-id list a
        downstream config reads beside it.

        Args:
            out (str | Path): Where the screen report goes.
            write_ids (bool): Also write ``<out>_asset_ids.json``, the graspable ids (default: True).
            **kwargs (Any): Passed to ``screen_meshes``: ``density`` (default ``"grid"``), ``jobs`` (default 10),
                ``want_graspable`` (stop once that many graspable meshes were found) and ``report``.

        Returns:
            ScreenReport: The screen's and the ids' paths, how many meshes were screened and how many are graspable.
        """
        from datagen.assets.prepare import asset_ids_from_screen, screen_meshes  # noqa: PLC0415

        out_path = Path(out)
        rows = screen_meshes(list(self.sources), library=self.library, out=out_path, **kwargs)
        ids_path: Path | None = None
        if write_ids:
            ids_path = out_path.with_name(f"{out_path.stem}_asset_ids.json")
            # `indent=1`, matching what the CLI writes. This file is committed and read through
            # `assets.mesh_asset_ids_path`; reformatting it would put a whole-file diff in front of
            # the next reader for nothing.
            ids_path.write_text(json.dumps(asset_ids_from_screen(rows), indent=1, sort_keys=True),
                                encoding="utf-8")
        graspable = sum(1 for row in rows if row.get("status") == "ok" and row.get("jaw", 0))
        return ScreenReport(screen_path=out_path, ids_path=ids_path, rows=len(rows),
                            graspable=graspable)

    def decompose(self, config: "DatagenConfig", *, jobs: int = 8,
                  scenes: int | None = None) -> DecompositionReport:
        """Fill the convex-decomposition cache for everything a config will place, in parallel: before a MuJoCo build,
        so the build does not decompose each mesh one at a time inside its render loop.

        Args:
            config (DatagenConfig): The dataset's settings; exactly the assets it places are warmed.
            jobs (int): Processes in parallel (default: 8).
            scenes (int | None): Warm only the assets of this many scenes; ``None`` the config's count (default: None).

        Returns:
            DecompositionReport: How many assets, how many computed, how many already cached, the parts, the cache
                folder.
        """
        from datagen.build import build_manifest  # noqa: PLC0415
        from datagen.pool import process_pool  # noqa: PLC0415
        from datagen.render.convex_decomposition import (  # noqa: PLC0415
            CACHE_DIR, decomposition_refusal, warm_asset,
        )

        refusal = decomposition_refusal()
        if refusal is not None:
            # The reason, not "not installed". A blocked library is installed, and telling its
            # operator to reinstall it is a false instruction that costs them the policy name the
            # raw error would have given.
            raise RuntimeError(f"nothing to warm: {refusal}")

        records = [r for r in build_manifest(config, scenes) if getattr(r, "mesh_path", "")]
        computed = skipped = parts = 0
        if records:
            with process_pool(jobs) as pool:
                for _asset_id, asset_parts, seconds in pool.map(warm_asset, records):
                    parts += asset_parts
                    # A cache hit returns in microseconds; the threshold separates "read" from "ran".
                    if seconds < 0.05:
                        skipped += 1
                    else:
                        computed += 1
        return DecompositionReport(assets=len(records), computed=computed, already_cached=skipped,
                                   parts=parts, cache_dir=CACHE_DIR)

    def why_no_jaw(self, *, limit: int = 40, density: str = "grid",
                   from_screen: str | Path | None = None,
                   report: Any = None) -> JawDiagnosis:
        """Why do these meshes earn no grasp? The question a user with their own parts asks.

        Args:
            limit (int): How many meshes to ask about (default: 40).
            density (str): The label density to probe with, ``"default"``, ``"dense"`` or ``"grid"`` (default: "grid").
            from_screen (str | Path | None): A screen report: ask only about the meshes it scored at zero, the sharper
                question; ``None`` samples everything (default: None).
            report (Any): Called with one progress line at a time; ``None`` says nothing (default: None).

        Returns:
            JawDiagnosis: Per mesh, what kept a jaw from it (too wide, too thin, no antipodal pair, ...).
        """
        from datagen.assets.diagnose import why_no_jaw as diagnose  # noqa: PLC0415
        from datagen.assets.prepare import screen_rows  # noqa: PLC0415
        from datagen.grasps.labels import DENSITIES  # noqa: PLC0415

        available = self.entries()
        if from_screen is not None:
            # The rows come out of the writer's own shape. This read the parsed JSON as a bare
            # list, which on the wrapper dict `screen_meshes` writes iterates the keys: measured
            # 2026-09-10 against the committed screen.json as `'str' object has no attribute 'get'`.
            rows = screen_rows(json.loads(Path(from_screen).read_text(encoding="utf-8")))
            zero = {str(row.get("asset_id")) for row in rows
                    if row.get("status") == "ok" and not row.get("jaw", 0)}
            if not zero:
                raise ValueError(f"{from_screen} lists no mesh that earns zero jaw labels, "
                                 f"so there is nothing to ask about")
            available = [pair for pair in available if pair[0] in zero]
        meshes = even_sample(available, limit)
        if not meshes:
            raise ValueError("no mesh found; fetch one first with `python -m datagen.assets.fetch`")
        raw = diagnose(meshes, density=DENSITIES[density],
                       **({"report": report} if report is not None else {}))
        return JawDiagnosis(raw=raw, probed_meshes=len(meshes))
