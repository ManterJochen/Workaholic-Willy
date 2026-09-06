"""61: the other road to a corpus: import a public one, and the licence boundary that gates it.

    python scripts/examples/train/61_the_foreign_corpus.py            # describe, price, refuse
    python scripts/examples/train/61_the_foreign_corpus.py --index 12
    python scripts/examples/train/61_the_foreign_corpus.py --import 20 --out logs/examples/foreign

The decision: your own parts or somebody else's corpus. 60_your_own_corpus.py is the first road and
this is the second, and they end in the same place, because the importer writes the same `.npz`
scene files the local generator writes. The corpus index, the folds, the sample builder, every probe
and every metric then run on imported data with no flag and no branch: two producers, one consumer,
and the consumer cannot tell them apart.

What the second road costs, and it is not the download:

  a licence has to hold.        A research-only corpus does not survive commercial use, and the
                                best-known 6-DoF grasp datasets are research-only. The vocabulary
                                that decides lives in `datagen.assets.licensing`, and this example
                                asks it about the source it imports and about a refused one. There
                                are two boundaries in that module rather than one, and they are not
                                the same list: see the three steps below.
  no asset identity.            The scenes are generated images, so no object recurs across scenes
                                and asset-disjoint folds are not achievable. The scene id becomes
                                the fold group, which makes folds scene-disjoint. You cannot ask
                                this corpus whether a model generalises to an object it never saw.
  a convention has to be read.  A published pose matrix does not say how to read itself. Nothing
                                here assumes one: the importer measures which rotation column is
                                the approach, refits the translation offset against the cloud being
                                imported, and records both in the provenance it writes.

Nothing is vendored. The dataset is not redistributed with this repository; the importer addresses
the remote archive by range request and fetches only the members it converts, which is why the
sizes below can be read without downloading anything.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
from pathlib import Path
from typing import Final

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import Example, not_ready  # noqa: E402

EPILOG = """
exit codes
  0  the source was described and the licence boundary held
  1  the boundary did not refuse something it must refuse
  2  the host did not answer, or the layout it publishes has changed
"""

#: A licence string carrying a non-commercial clause, spelled the way a publisher spells it. Used
#: below to show the clause being read out of the identifier rather than a prefix being matched.
_NON_COMMERCIAL: Final[str] = "CC BY-NC-SA 4.0"

#: What the importer says the four central directories cost to fetch. Paid once, and the index is
#: then exact instead of accidental; the sliced index below pays nothing and is a sample.
#:
#: The figure is the download. What lands on disk is the parsed form of it and is larger, so the
#: import step below measures the cache directory afterwards rather than repeating this number: an
#: estimate printed twice is an estimate a reader takes for a measurement.
_DIRECTORY_MEGABYTES: Final[int] = 460


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "", epilog=EPILOG,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=int, default=0, metavar="N",
                        help="read a sample of the archive index, N scenes. Costs a few megabytes "
                             "and downloads no scene")
    parser.add_argument("--import", dest="import_scenes", type=int, default=0, metavar="N",
                        help="actually import N scenes. Reads the full index first, which is "
                             f"about {_DIRECTORY_MEGABYTES} MB cached once")
    parser.add_argument("--out", default="logs/examples/foreign",
                        help="where imported scenes and the cached index go")
    parser.add_argument("--gripper", default=None,
                        help="which jaw profile the imported labels belong to. Default: the one "
                             "the source names")
    args = parser.parse_args(argv)

    from src.robot.grasping.deep.foreign import grasp_anything
    from src.robot.grasping.deep.net.gripper import JAW_GEOMETRY

    source = grasp_anything.SOURCE
    gripper = args.gripper or grasp_anything.DEFAULT_GRIPPER
    # Checked here, before anything reaches the network. The importer checks it too, but only after
    # it has built the scene index, so a misspelled hand there costs the whole index first.
    if gripper not in JAW_GEOMETRY:
        return not_ready(f"unknown gripper {gripper!r}",
                         f"choose one of {', '.join(sorted(JAW_GEOMETRY))}, or leave --gripper off "
                         f"to use the one the source names")

    with Example("61 the foreign corpus", f"{source.key} by range request", hardware=False) as run:
        with run.step("what the source declares about itself") as report:
            report(f"{source.title}, licence {source.licence}")
        run.note("")
        run.note(f"  {source.url}")
        run.note(f"  {source.scenes:,} scenes as published, labels for the {gripper} jaw at "
                 f"{JAW_GEOMETRY[gripper]['aperture_mm']:.0f} mm")
        run.note("")
        run.note("  the aperture is this repository's number for that hand, not the source's. The")
        run.note("  source caps its own opening well above what the gripper it names can open to,")
        run.note("  so a label needing more than the real aperture is refused, not believed.")
        run.note("")

        from datagen.assets.licensing import (
            ALLOWED_ASSET_LICENSES, FORBIDDEN, audit_asset_rows, forbidden_hits, is_noncommercial)

        # The refused name is taken out of the list rather than typed in here, and that is not
        # tidiness. `tests/test_license_boundary.py` scans every .py, .md and .yaml in this
        # repository for those names and fails on any file that spells one out, because a paragraph
        # naming a research-only project reads as an install guide for it. A file about the
        # boundary is subject to the boundary, so the name is fetched at runtime and printed.
        refused_source = FORBIDDEN[0]

        with run.step("the dataset boundary, and this source against it") as report:
            hits = forbidden_hits(source.key) + forbidden_hits(source.title)
            report(f"non-commercial {is_noncommercial(source.licence)}, "
                   f"forbidden-name hits {hits or 'none'}")
        run.note("")
        run.note(f"  {len(FORBIDDEN)} project name(s) are forbidden outright, matched as a")
        run.note("  case-insensitive substring, so a repackaging under a new name still hits. The")
        run.note("  list is owned by the production code that has to obey it, and the guard that")
        run.note("  scans this repository imports the same tuple, so a name can never be forbidden")
        run.note("  in one place and allowed in the other. That guard covers three surfaces: an")
        run.note("  installed distribution, an import of a vendored copy, and the project named in")
        run.note("  prose, because a paragraph saying where to put the weights is a working")
        run.note("  install guide for a licence this project cannot use.")
        run.note("")

        with run.step("the same boundary refusing one") as report:
            report(f"{refused_source!r}: name hits {forbidden_hits(refused_source)}; "
                   f"{_NON_COMMERCIAL!r} non-commercial {is_noncommercial(_NON_COMMERCIAL)}")
        run.note("")
        run.note("  that is the corpus most readers think of first, and it is why this importer")
        run.note("  reads a smaller and less famous one. The licence is not the only thing that")
        run.note("  matters, but it is the one that cannot be fixed afterwards: a model trained on")
        run.note("  a research-only corpus cannot be shipped, however good it is.")
        run.note("")
        run.note("  the dataset imported here is not vendored and not redistributed. What ships is")
        run.note("  the importer; the bytes stay on the publisher host and are read from it, and")
        run.note("  the attribution file records exactly that.")
        run.note("")

        # A second audit, over asset rows, because the asset rule is a different and stricter one.
        # The rows are built here rather than read off a manifest: the point is which clause refuses
        # which row, and a manifest that happened to be clean would show none of them.
        rows = (
            {"id": "a_part_you_designed", "source": "custom", "license": "own"},
            {"id": "a_public_scan", "source": "gso", "license": "CC-BY-4.0",
             "attribution": "the publisher"},
            {"id": "a_scan_with_no_author_named", "source": "gso", "license": "CC-BY-4.0"},
            {"id": "a_research_mesh", "source": refused_source, "license": _NON_COMMERCIAL},
            {"id": "a_share_alike_mesh", "source": "somewhere", "license": "cc-by-sa-4.0",
             "attribution": "the author"},
        )
        with run.step("the asset boundary, which is stricter") as report:
            problems = audit_asset_rows(rows)
            refused = {problem.split("'")[1] for problem in problems}
            report(f"{len(rows) - len(refused)} of {len(rows)} row(s) pass, {len(refused)} "
                   f"refused for {len(problems)} reason(s)")
        run.note("")
        for problem in problems:
            run.note(f"  refused  {problem}")
        run.note("")
        if len(refused) != 3:
            # The one failure this file must not report as a tidy pass. Three of the five rows are
            # chosen because each trips a different clause, so a boundary refusing fewer of them has
            # stopped reading one of the clauses.
            run.finding("the asset boundary, which is stricter",
                        f"three rows should be refused, {len(refused)} were")
        run.note(f"  what may enter a dataset as an asset: {', '.join(ALLOWED_ASSET_LICENSES)}.")
        run.note("  That list is not the dataset list and is deliberately narrower. A rendered")
        run.note(f"  image is a derivative work of every mesh in it, so {source.licence}, which is")
        run.note("  a software licence, is not an answer to the question asked of a mesh and is")
        run.note("  not on it. Attribution is obligatory rather than polite: a CC-BY row with no")
        run.note("  author recorded is refused, because every render is a derivative work that has")
        run.note("  to name them.")
        run.note("")

        with run.step("the archives, addressed rather than downloaded") as report:
            try:
                clouds, grasps, masks, prompts = grasp_anything.archives()
            except urllib.error.URLError as error:
                return not_ready(f"the host did not answer: {error}",
                                 "this step needs network access to the publisher's host; nothing "
                                 "is cached in this repository")
            except RuntimeError as error:
                # The importer refuses a layout it does not recognise rather than guessing, which
                # is what a missing part would otherwise become: a silently shorter archive.
                return not_ready(str(error),
                                 "the source's published layout has changed and the importer needs "
                                 "a decision rather than a default")
            named = (("clouds", clouds), ("grasps", grasps), ("masks", masks),
                     ("prompts", prompts))
            report(", ".join(f"{name} {zipped.total / 1e9:.1f} GB" for name, zipped in named))
        run.note("")
        total = sum(zipped.total for _name, zipped in named)
        run.note(f"  {total / 1e9:.0f} GB across four archives, and none of it was fetched. A zip")
        run.note("  carries a directory at its end listing where every member starts, and a member")
        run.note("  is compressed on its own, so with range requests the archive reads like a")
        run.note("  filesystem: the tail, then the directory, then one member at a time.")
        run.note("")
        run.note("  the masks and the prompts matter more than their size suggests. A training")
        run.note("  unit is an object, so a scene whose points all carry one instance collapses to")
        run.note("  a single unit however many things are in it, and the masks are what separate")
        run.note("  them. The prompts are one instruction per object, which is the half of the")
        run.note("  problem a locally generated corpus cannot produce at all.")
        run.note("")

        if args.index:
            with run.step(f"a sample of the index, {args.index} scene(s)") as report:
                # No cache directory, so this reads a slice of each central directory rather than
                # all of it. Cheap, and a sample rather than a listing: the library reports its own
                # coverage below instead of leaving a caller to assume.
                triples = grasp_anything.scene_index(clouds, grasps, masks, prompts,
                                                     limit=args.index, report=run.note)
                objects = sum(triple.objects for triple in triples)
                report(f"{len(triples)} scene(s), {objects} object(s) with a grasp file")
            run.note("")
            run.note("  a slice is a sample here only because the scene ids are hashes with no")
            run.note("  semantics, so directory order carries none either. That is the exception")
            run.note("  to this repository's rule that a prefix is never a sample, and it is")
            run.note("  stated because the rule exists.")
            run.note("")

        if args.import_scenes:
            run.note(f"importing {args.import_scenes} scene(s). The exact index is read first and")
            run.note(f"  cached under {args.out}: about {_DIRECTORY_MEGABYTES} MB fetched once,")
            run.note("  against a corpus of hundreds of gigabytes. Each scene is then a couple of")
            run.note("  hundred kilobytes, and an interrupted import resumes, because every scene")
            run.note("  is written as it lands rather than at the end.")
            run.note("")
            with run.step(f"import {args.import_scenes} scene(s)") as report:
                provenance = grasp_anything.import_scenes(
                    args.out, limit=args.import_scenes, gripper=gripper, report=run.note)
                report(f"{provenance['written']} written, "
                       f"{provenance['no_usable_grasp']} with no usable grasp, "
                       f"{provenance['refused_too_wide']} label(s) refused as too wide")
            run.note("")
            for line in json.dumps(provenance, indent=2, sort_keys=True).splitlines():
                run.note(line)
            run.note("")
            cache = Path(args.out) / ".directories"
            if cache.is_dir():
                written = sum(f.stat().st_size for f in cache.glob("*")) / 1e6
                scenes = sorted(Path(args.out).glob("*.npz"))
                per_scene = (sum(f.stat().st_size for f in scenes) / len(scenes) / 1e3
                             if scenes else 0.0)
                run.note(f"  the index cache is {written:.0f} MB on disk and each scene is "
                         f"{per_scene:.0f} KB.")
                run.note("  The first number is paid once for the whole corpus and the second is")
                run.note("  what another thousand scenes would cost.")
                run.note("")
            run.note("  the provenance is written beside the scenes. It names the source, its")
            run.note("  licence, the gripper the labels were filtered for, and the offset the")
            run.note("  import fitted, so a revised source shows up later as a changed offset")
            run.note("  rather than as a silently wrong one.")
            run.note("")
            run.note("  the contract block is the import checking itself. It reads the written")
            run.note("  files back through the same loader a training run uses and puts them")
            run.note("  through the sample contract, which is the cheapest place to catch a frame")
            run.note("  error: a corpus in the wrong unit or the wrong frame trains happily to a")
            run.note("  confident wrong answer and announces nothing.")
            run.note("")
            run.note(f"Next: 63_read_the_report.py --clouds {args.out}")
        else:
            run.note("nothing was imported. `--import N` fetches N scenes; read the two costs")
            run.note("  above first, because the index is the part that is paid up front.")
        run.note("")

        run.note("three things this source does not have, each handled rather than papered over:")
        run.note("  no normals. The release ships none and the sample contract requires them, so")
        run.note("    they are estimated by local geometry and oriented toward the camera. An")
        run.note("    estimated normal is not a measured one, and the scene records that it is.")
        run.note("  no jaw contacts. The stored opening is wider than the object it holds, so the")
        run.note("    width is re-derived from where the cloud actually lies between the jaws.")
        run.note("  no asset identity. Folds become scene-disjoint rather than asset-disjoint, so")
        run.note("    a number measured on this corpus does not answer whether a model")
        run.note("    generalises to an object it has never seen. That question needs 60.")
        run.note("")
        run.note("what to pick: 60_your_own_corpus.py if you have the parts your cell handles, and")
        run.note("  you do, because a bin-picking cell runs on parts no public dataset contains.")
        run.note("  This road is for proving the chain closes without a simulator, for the")
        run.note("  language prompts, and for pretraining a backbone you then fit to your own")
        run.note("  parts. The two mix: the loop reads both as the same scene files.")
        return run.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
