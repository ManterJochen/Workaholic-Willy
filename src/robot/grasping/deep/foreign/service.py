"""Training on somebody else's corpus, as one noun with three verbs, callable without a command line.

A user with no simulator, no GPU-hours and no cell still has to be able to train something. That is
what `foreign/` is for, and until this module existed the only door to it was
`python -m src.robot.grasping.deep import-foreign`: the capability shipped, and no program could
reach it.

    from willy import GeneratorTraining, PublicCorpus

    corpus = PublicCorpus.from_source(out_dir="logs/dl/clouds/public")
    print(corpus.describe())                  # the source, its licence, its size; fetches nothing
    print(corpus.fetch(limit=200))            # 200 scenes as .npz, by range request
    run = GeneratorTraining.from_recipe(corpus=corpus.out_dir, recipe="v1", tier="full",
                                        out_dir="logs/dl/models/public")
    print(run.train())

The import writes the same `.npz` scene files `datagen` writes, so the second line above is the only
one that knows where the data came from. The folds, the probes, `build_sample` and every metric run
on it unchanged, which is the whole design: two producers, one consumer, and the consumer cannot tell
them apart.

This is not a second importer. `grasp_anything.import_scenes` keeps its name, its signature and its
module, and this wraps it, owns the glue the CLI handler owned privately, and returns something typed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from collections.abc import Callable  # noqa: F401 (the reader registry's value type)

    from src.robot.grasping.deep.foreign.grasp_anything import ForeignSource

__all__ = ["ImportReport", "PublicCorpus", "public_sources"]

#: Built without a backslash escape, as the other renderers in this repository are.
_NEWLINE = chr(10)

def _grasp_anything() -> Any:
    """The Grasp-Anything-6D reader, imported when it is asked for rather than at module import.

    A real import statement, not `import_module(<a string>)`: `tests/test_deep_imports_resolve.py`
    walks the AST of every live file to catch an import naming a module that no longer exists, and a
    module path held as a string is invisible to it. A registry keyed by loader keeps both the table
    and that guard.
    """
    from src.robot.grasping.deep.foreign import grasp_anything  # noqa: PLC0415 (pulls numpy)

    return grasp_anything


#: Every public corpus this build can read, and the callable that loads its reader. One entry today;
#: the key is what a caller names, and it is the key the imported scenes are stamped with, so a
#: corpus that mixes producers can be split on it afterwards.
_READERS: "dict[str, Callable[[], Any]]" = {
    "grasp_anything_6d": _grasp_anything,
}

#: The key `from_source()` uses when a caller names none. There is one reader, and naming it should
#: not be the price of the common case.
DEFAULT_SOURCE = "grasp_anything_6d"


def _reader(key: str) -> Any:
    if key not in _READERS:
        raise ValueError(f"unknown public corpus {key!r}; expected one of {sorted(_READERS)}")
    return _READERS[key]()


def public_sources() -> list[dict[str, Any]]:
    """What can be imported, how big it is, and on what licence. Opens no network connection.

    The licence column is the source's own, recorded where the reader was written against the
    published terms. A corpus is somebody else's work: the obligation travels with the data, and a
    model trained on it inherits it.
    """
    rows = []
    for key in sorted(_READERS):
        source: "ForeignSource" = _reader(key).SOURCE
        rows.append({"key": source.key, "title": source.title, "licence": source.licence,
                     "url": source.url, "scenes": source.scenes})
    return rows


@dataclass(frozen=True, slots=True)
class ImportReport:
    """What one import did: counts, the corpus it owes attribution to, and where it landed.

    `ok` is about this run, not about the corpus. A run that wrote nothing because every scene was
    already on disk is fine; a run that wrote nothing because the fetch failed is not.
    """

    out_dir: Path
    source: str
    licence: str
    url: str
    gripper: str
    written: int
    skipped_present: int
    no_usable_grasp: int
    failed: int
    training_units: int
    #: How many written scenes were read back through the corpus loader and put through the sample
    #: contract, and what that found. Empty when `validate=0` asked for none.
    contract_checked: int = 0
    contract_failures: tuple[str, ...] = ()
    #: The provenance block the importer wrote beside the scenes, verbatim.
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def scenes(self) -> int:
        """Scenes this corpus directory now holds, whoever wrote them."""
        return self.written + self.skipped_present

    @property
    def ok(self) -> bool:
        return not self.failed and not self.contract_failures and self.scenes > 0

    def render(self) -> str:
        lines = [
            f"{self.source}: {self.scenes} scene(s) in {self.out_dir}",
            f"  licence    {self.licence}  ({self.url})",
            f"  gripper    {self.gripper}",
            f"  written    {self.written}, {self.skipped_present} already present, "
            f"{self.no_usable_grasp} with no usable grasp, {self.failed} failed",
            f"  units      {self.training_units} training unit(s)",
        ]
        if self.contract_checked:
            lines.append(f"  contract   {self.contract_checked} scene(s) checked, "
                         f"{len(self.contract_failures)} failure(s)")
            lines.extend(f"    ! {failure}" for failure in self.contract_failures[:5])
        if not self.scenes:
            lines.append("  nothing was imported; see the lines above for what each scene refused")
        return _NEWLINE.join(lines)

    def __str__(self) -> str:
        return self.render()

    def as_dict(self) -> dict[str, Any]:
        return {"out_dir": str(self.out_dir), "source": self.source, "licence": self.licence,
                "ok": self.ok, "scenes": self.scenes, "written": self.written,
                "skipped_present": self.skipped_present, "failed": self.failed,
                "training_units": self.training_units,
                "contract_failures": list(self.contract_failures)}


@dataclass(frozen=True, slots=True)
class PublicCorpus:
    """A published grasp corpus, read into the scene files this training loop already eats."""

    key: str
    out_dir: Path
    #: Which jaw profile the imported labels belong to. The source's own hand, unless a caller says
    #: otherwise; a corpus is stamped with it and the trained artifact claims it.
    gripper: str

    @classmethod
    def from_source(cls, key: str = DEFAULT_SOURCE, *, out_dir: str | Path,
                    gripper: str | None = None) -> "PublicCorpus":
        """The named corpus, landing in `out_dir`. Reads nothing and opens no connection.

        The gripper is validated here rather than after the fetch: the names live in `JAW_GEOMETRY`,
        and finding out that a typo was a typo should not cost a download.
        """
        from src.robot.grasping.deep.net.gripper import JAW_GEOMETRY  # noqa: PLC0415 (pulls torch)

        module = _reader(key)
        chosen = gripper if gripper is not None else str(module.DEFAULT_GRIPPER)
        if chosen not in JAW_GEOMETRY:
            raise ValueError(f"unknown gripper {chosen!r}; choose from {', '.join(JAW_GEOMETRY)}")
        return cls(key=key, out_dir=Path(out_dir), gripper=chosen)

    @property
    def source(self) -> "ForeignSource":
        """What the corpus says about itself: title, licence, url, published size."""
        source: "ForeignSource" = _reader(self.key).SOURCE
        return source

    def describe(self) -> str:
        """The source, its licence and what is already on disk, before anything is fetched."""
        source = self.source
        return _NEWLINE.join([
            f"public corpus {source.key}",
            f"  title      {source.title}",
            f"  licence    {source.licence}  ({source.url})",
            f"  published  {source.scenes} scene(s)",
            f"  gripper    {self.gripper}",
            f"  out_dir    {self.out_dir} ({self.scenes_present()} scene(s) already imported)",
        ])

    def scenes_present(self) -> int:
        """How many scenes `out_dir` already holds. A fetch resumes rather than starting over."""
        return sum(1 for _ in self.out_dir.glob("*.npz")) if self.out_dir.is_dir() else 0

    def fetch(self, *, limit: int = 200, jobs: int = 8, validate: int = 3,
              centre_offset_m: float | None = None, cache_dir: str | Path | None = None,
              report: "Callable[[str], None] | None" = None) -> ImportReport:
        """Fetch `limit` scenes and write them as the `.npz` files the training loop reads.

        By range request, not by download. The source is 229 GB and its clouds are one Zip64 split
        across five parts, so looking at it costs roughly 200 KB per scene rather than the archive.

        Needs the network, and raises what the network raises: `OSError` (which `URLError` is),
        `RuntimeError` for an archive that will not address, `ValueError` for a member that will not
        parse. Each scene is written as it converts, so an interrupted fetch keeps what it got and
        the next call continues from there.

        `validate` reads that many written scenes back through the corpus loader and puts them
        through the sample contract, which is the cheapest place to catch a frame error; `0` skips
        it. `report` is called with one progress line at a time, `None` swallows them.
        """
        provenance = _reader(self.key).import_scenes(
            self.out_dir, limit=limit, jobs=jobs, validate=validate,
            centre_offset_m=centre_offset_m, cache_dir=cache_dir, gripper=self.gripper,
            report=report if report is not None else (lambda _line: None))
        return self._report(provenance)

    def report(self) -> ImportReport | None:
        """The last import's provenance, read back from disk. `None` where none has run."""
        stamp = self.out_dir / "provenance.json"
        if not stamp.is_file():
            return None
        try:
            return self._report(json.loads(stamp.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            return None

    def _report(self, provenance: Mapping[str, Any]) -> ImportReport:
        contract = dict(provenance.get("contract") or {})
        return ImportReport(
            out_dir=self.out_dir,
            source=str(provenance.get("source", self.key)),
            licence=str(provenance.get("licence", "")),
            url=str(provenance.get("url", "")),
            gripper=str(provenance.get("gripper", self.gripper)),
            written=int(provenance.get("written", 0)),
            skipped_present=int(provenance.get("skipped_present", 0)),
            no_usable_grasp=int(provenance.get("no_usable_grasp", 0)),
            failed=int(provenance.get("failed", 0)),
            training_units=int(provenance.get("training_units", 0)),
            contract_checked=int(contract.get("checked", 0)),
            contract_failures=tuple(str(f) for f in contract.get("failures", ())),
            provenance=dict(provenance),
        )
