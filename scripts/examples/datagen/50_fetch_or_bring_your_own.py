"""50: which objects your corpus is made of, public collections or your own parts.

    python scripts/examples/datagen/50_fetch_or_bring_your_own.py
    python scripts/examples/datagen/50_fetch_or_bring_your_own.py --parts ./my_cad --license own
    python scripts/examples/datagen/50_fetch_or_bring_your_own.py --fetch thingi10k --limit 5

The decision lives in `datagen.config.AssetSourcesConfig`, as one weight per source. The shipped
default is `procedural_weight: 1.0` and every other weight `0.0`, so a corpus built without touching
this block is made of parametric boxes, cylinders and spheres: sixteen kind words rendered as three
solids, which is enough to measure geometry and is not your cell's parts. The two ways off that
default are the two halves of this file. `gso_weight`, `ycb_weight`, `thingi10k_weight`,
`asos_weight` and `objaverse_weight` draw from public scanned collections that have to be fetched
first. `custom_weight` draws from a directory of your own meshes that you import.

Neither half is free, and they are not expensive in the same way.

A public collection costs disk and a licence obligation. Every rendered image is a derivative work
of every mesh in it, so the licence travels with the asset row into the dataset's attribution file,
and `datagen.assets.licensing.audit_asset_rows` refuses a manifest before a render rather than
after. It is fail-closed by clause and not by prefix, which matters more than it sounds: `cc-by-sa`
and `cc-by-nd` both start with `cc-by`, and both are refused here. This example makes that gate run
on rows it is meant to refuse, because the refusal is what you will meet.

Your own parts cost a declaration. `import_from_directory` has no default licence and raises
without one: nobody but you knows what your CAD is licensed as, and a default would put a string
nobody checked into a gate whose whole job is to check it. `own` is the ordinary answer.

Finally, two weights decide a corpus and only one of them is obvious. `assets.mesh_asset_ids`
restricts which meshes may be drawn, never which scenes are built, so a corpus that names your parts
and leaves `procedural_weight` at its default comes out part full of objects nobody asked for. The
last step below runs the refusal that catches it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Final

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import EXIT_FAILED, Example, not_ready  # noqa: E402

EPILOG = """
exit codes
  0  the library was reported, the import path ran, and the licence gate refused what it must
  1  the licence gate passed a row it was supposed to refuse
  2  the request could not start: an unknown source, or --parts naming nothing importable

what this writes
  nothing under assets/meshes unless you pass --library assets/meshes, which is the one root the
  scene layout reads. The default target is a throwaway under logs/examples.
"""

#: A stand-in part, used only when the reader gives no --parts. Twelve triangles of an axis-aligned
#: 40 mm cube, written as a real .obj and imported through the real library function, so what runs
#: below is the import path a customer runs and not a description of it.
_CUBE_OBJ: Final[str] = """# a 40 mm cube, stand-in for one of your parts
v -20 -20 0
v  20 -20 0
v  20  20 0
v -20  20 0
v -20 -20 40
v  20 -20 40
v  20  20 40
v -20  20 40
f 1 3 2
f 1 4 3
f 5 6 7
f 5 7 8
f 1 2 6
f 1 6 5
f 2 3 7
f 2 7 6
f 3 4 8
f 3 8 7
f 4 1 5
f 4 5 8
"""

#: Rows the licence gate is asked to judge, and what each one is there to prove. Every string here
#: is a licence identifier, not a source: the gate reads the identifier, which is why the spelling
#: of one is a fact about the row and not about whoever typed it.
_AUDIT_ROWS: Final[tuple[tuple[str, dict[str, str], bool], ...]] = (
    ("your own parts", {"id": "custom_bracket", "source": "custom", "license": "own"}, True),
    ("public domain", {"id": "thingi10k_42", "source": "thingi10k", "license": "CC0-1.0"}, True),
    ("attribution carried", {"id": "gso_mug", "source": "gso", "license": "CC-BY-4.0",
                             "attribution": "Google Scanned Objects, CC-BY-4.0"}, True),
    ("attribution missing", {"id": "gso_mug", "source": "gso", "license": "CC-BY-4.0"}, False),
    ("non-commercial", {"id": "other_part", "source": "other", "license": "CC BY-NC 4.0"}, False),
    ("share-alike", {"id": "other_part", "source": "other", "license": "cc-by-sa-4.0"}, False),
    ("no-derivatives", {"id": "other_part", "source": "other", "license": "cc-by-nd-4.0"}, False),
    ("licence absent", {"id": "hand_dropped", "source": "custom", "license": ""}, False),
)


def _first_line(error: Exception, width: int = 96) -> str:
    """One line of a refusal, for the step's verdict. The rest is printed as notes.

    Not `str(error).split(".")[0]`: these messages name config keys, so the first full stop is
    inside `assets.mesh_asset_ids` and the verdict comes out as the word `assets`.
    """
    return " ".join(str(error).split())[:width]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "", epilog=EPILOG,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parts", default=None,
                        help="directory of your own meshes to import. Without it a stand-in cube "
                             "is authored, and the same import function runs on that")
    parser.add_argument("--license", default="own",
                        help="what your parts are licensed as. `own` for parts you designed, the "
                             "identifier itself for anything you did not. There is no default in "
                             "the library: this flag has one so the example can run")
    parser.add_argument("--attribution", default="",
                        help="who to credit for your parts, recorded beside them")
    parser.add_argument("--library", default="logs/examples/mesh_library",
                        help="where the import lands. A real import passes `assets/meshes`, which "
                             "is the only root the scene layout reads")
    parser.add_argument("--fetch", default=None,
                        help="download one public collection, e.g. `thingi10k`. Off by default: "
                             "these are gigabytes and this example is meant to be cheap")
    parser.add_argument("--limit", type=int, default=5,
                        help="meshes to take from --fetch. The first n of the collection, which is "
                             "a trial and not a sample")
    args = parser.parse_args(argv)

    from datagen.assets.library import (
        LICENSES,
        MESH_LIBRARY_DIR,
        MeshLibrary,
        declared_license,
        import_from_directory,
    )
    from datagen.assets.licensing import audit_asset_rows
    from datagen.assets.service import MeshPreparation, available_sources

    with Example("50 assets", "public meshes, or the parts your cell actually handles",
                 hardware=False) as run:

        # Every table below is collected inside its step and printed after it. A step's
        # announcement is an open line on a terminal, so anything the body prints lands on it and
        # takes the verdict with it.
        present: list[str] = []
        with run.step("what this machine already holds") as report:
            counts = MeshLibrary().counts()
            for source, number in sorted(counts.items()):
                licence = LICENSES.get(source) or declared_license(MESH_LIBRARY_DIR / source)[0]
                present.append(f"  {source:10} {number:6d} mesh(es)   licence "
                               f"{licence or 'undeclared'}")
            report(f"{sum(counts.values())} mesh(es) under {MESH_LIBRARY_DIR}")
        run.note("")
        for line in present:
            run.note(line)
        run.note("")
        run.note("A count of zero is not a failure. The meshes are fetched, never vendored, and")
        run.note("`custom` is empty on almost every machine because it is yours to fill.")
        run.note("")

        offered: list[str] = [f"  {'source':10} {'objects':>8} {'GB':>8}  licence"]
        with run.step("side one: what the public collections offer") as report:
            rows = available_sources()
            for row in rows:
                offered.append(f"  {row['key']:10} {row['objects']:8,} {row['approx_gb']:8.2f}  "
                               f"{row['licence']}")
            unverified = [row["key"] for row in rows if not row["licence_verified"]]
            report(f"{len(rows)} collection(s), {len(unverified)} with an unverified licence")
        run.note("")
        for line in offered:
            run.note(line)
        run.note("")
        run.note("The licence column is two different claims wearing one name. Where")
        run.note("`licence_verified` is false the collection publishes terms for itself and states")
        run.note("nothing per object, so a dataset built on it inherits a claim rather than a")
        if unverified:
            run.note(f"check. Unverified here: {', '.join(unverified)}.")
        run.note("")

        if args.fetch:
            # Named explicitly, never swept in. A fetch is gigabytes over somebody's connection,
            # and `MeshPreparation.from_sources(None)` means every research collection.
            with run.step(f"fetch {args.fetch}") as report:
                fetched = MeshPreparation.from_sources([args.fetch]).fetch(
                    library=args.library, limit=args.limit)
                report(f"{fetched.fetched} mesh(es) fetched, ok={fetched.ok}")
            for line in fetched.render().splitlines():
                run.note(line)
            run.note("")
        else:
            run.note("Nothing was downloaded. The call is")
            run.note("  MeshPreparation.from_sources(['gso']).fetch(limit=200)")
            run.note("and it returns counts per collection rather than a process exit code. Pass")
            run.note("--fetch <source> to run it here on a small limit.")
            run.note("")

        origin = Path(args.parts) if args.parts else Path("logs/examples/example_parts")
        if args.parts is None:
            origin.mkdir(parents=True, exist_ok=True)
            (origin / "example_part.obj").write_text(_CUBE_OBJ, encoding="utf-8")
            run.note(f"No --parts given, so a 40 mm cube was authored at {origin} as a stand-in.")
            run.note("Everything below runs the real import on it.")
            run.note("")
        elif not origin.is_dir():
            return not_ready(f"--parts names {origin}, which is not a directory",
                             "point it at the folder your exported meshes are in; the readable "
                             "suffixes are obj, stl, ply, off, glb and gltf")

        with run.step("side two: the licence has no default, and that is the refusal") as report:
            try:
                import_from_directory("custom", origin, destination=args.library)
            except ValueError as refusal:
                # The expected answer, so it is a step that passed rather than a finding. An import
                # that succeeded here would be the failure: it would put an unchecked licence into
                # the gate the next step runs.
                report(f"refused: {_first_line(refusal, 86)}")
            else:
                run.finding("licence gate", "the import accepted a mesh with no licence declared")
                return EXIT_FAILED

        with run.step("import them, with the declaration") as report:
            entries = import_from_directory("custom", origin, destination=args.library,
                                            license=args.license, attribution=args.attribution)
            report(f"{len(entries)} mesh(es) into {Path(args.library) / 'custom'}")
        with run.step("read the declaration back off the disk") as report:
            # Read back rather than remembered. The declaration is written to LICENSE.txt beside the
            # meshes because the manifest and the audit read it months later, and a licence that
            # lived only in the argument list of the import would be gone by the next session.
            licence, credit = declared_license(Path(args.library) / "custom")
            report(f"licence {licence or 'undeclared'}, attribution {credit or 'none declared'}")

        run.note("")
        judged: list[str] = []
        with run.step("the gate, on rows it is meant to refuse") as report:
            wrong = 0
            for what, row, should_pass in _AUDIT_ROWS:
                problems = audit_asset_rows([row])
                passed = not problems
                mark = "passes" if passed else "refused"
                detail = "" if passed else problems[0].split(": ", 1)[-1]
                judged.append(f"  {what:22} {row['license'] or '(none)':16} {mark:8} "
                              f"{detail[:58]}")
                wrong += int(passed is not should_pass)
            report(f"{len(_AUDIT_ROWS)} row(s) judged, {wrong} judged wrongly")
        run.note("")
        for line in judged:
            run.note(line)
        if wrong:
            # Not a finding. A licence gate that admits a non-commercial mesh is the one failure
            # this repository cannot ship, so it is worth an exit code of its own.
            run.finding("licence gate", f"{wrong} row(s) came out the wrong way")
            return EXIT_FAILED
        run.note("")
        run.note("Read the last four lines again: two of the refusals start with `cc-by`. A prefix")
        run.note("test admits both, which is why `licensing.py` exports is_sharealike and")
        run.note("is_noderivatives beside is_noncommercial and reads the clause instead.")
        run.note("")

        with run.step("the weight that decides nothing, and the one that decides") as report:
            from datagen.config import AssetSourcesConfig, DatagenConfig
            from datagen.scenes import layout_scene

            named = DatagenConfig(
                scenes=2, seed=0,
                assets=AssetSourcesConfig(custom_weight=1.0,
                                          mesh_asset_ids=("custom_example_part",),
                                          refuse_procedural_fallback=True))
            try:
                layout_scene(named, 0)
            except ValueError as refusal:
                report(_first_line(refusal))
            else:
                run.finding("asset restriction", "a named mesh subset drew procedural objects and "
                                                 "nothing said so")
        run.note("")
        run.note("`mesh_asset_ids` restricts the mesh draw, not the scene. With procedural_weight")
        run.note("still at 1.0 the scenes fill up with authored objects that are not in your list,")
        run.note("and a corpus you built to hold out your own parts is contaminated instead. Zero")
        run.note("both procedural_weight and composite_weight when the point of the dataset is")
        run.note("which objects are in it. `refuse_procedural_fallback: true` turns the warning")
        run.note("above into the refusal you just read.")
        run.note("")

        run.note(f"The layout draws from {MESH_LIBRARY_DIR} and nowhere else, so with the parts")
        run.note(f"imported into {args.library} the next step is expected to refuse. Pass")
        run.note(f"--library {MESH_LIBRARY_DIR} to put them where scenes can draw them.")
        drew_custom = False
        with run.step("and the second refusal, which is where your parts have to live") as report:
            only_custom = DatagenConfig(
                scenes=2, seed=0,
                assets=AssetSourcesConfig(custom_weight=1.0, procedural_weight=0.0,
                                          composite_weight=0.0,
                                          refuse_procedural_fallback=True))
            try:
                scene = layout_scene(only_custom, 0)
                drew_custom = True
                report(f"drew {len(scene.objects)} object(s), all from assets/meshes/custom")
            except ValueError as refusal:
                report(_first_line(refusal))
        if not drew_custom:
            run.note("")
            run.note("That refusal is the good outcome of a bad configuration: the weights ask for")
            run.note("custom meshes, none is placeable where the layout looks, and falling back to")
            run.note("procedural objects would fill the dataset with what a model has seen already.")

        run.note("")
        run.note("Which side to pick. A bin-picking cell runs on the parts that cell handles, and")
        run.note("no public collection contains them, so the corpus you deploy on is built from")
        run.note("`custom`. The public collections are worth their gigabytes for variety a single")
        run.note("part range cannot give you, and for one thing your own parts cannot: an asset a")
        run.note("model has never seen. `datagen.heldout.held_out_assets(dataset_root)` lists")
        run.note("those and `format_report` prints the list.")
        run.note("")
        run.note("Two more verbs on the same noun, for when a part earns no grasp:")
        run.note("  MeshPreparation.from_sources(['custom']).screen('screen.json')")
        run.note("  MeshPreparation.from_sources(['custom']).why_no_jaw(from_screen='screen.json')")
        run.note("A fetched collection is also not yet a placeable one: meshes arrive in whatever")
        run.note("units their author used, and `normalise` is what puts them into metres.")
        run.note("")
        run.note("Next: 51_author_a_scene.py, which decides what a scene made of them looks like.")
        return run.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
