"""The licensing boundary, enforced in CI — because "we removed it" is a state, not a guarantee.

On 2026-08-11 a learned suction scorer was deleted from this repo: it wrapped a published RGB-D network
whose training data ships under a **non-commercial** licence. Deleting it was one commit. Keeping it
deleted is this file. Without a check, the next person to want a learned suction score re-adds the same
adapter in an afternoon, and nobody notices until it matters commercially — which is exactly when it is
most expensive to discover.

**Three surfaces, because a dependency is not the only way a project gets in.**

* **Installed distributions** — the obvious one: ``pip install graspnetAPI``.
* **Imports** — the sneaky one: a vendored copy in the source tree imports fine with nothing in
  ``requirements.txt``. This is what the removed adapter did (``import DeepLabV3Plus.network``),
  bridged by an env var so no dependency file ever mentioned it.
* **Prose** — deliberately included (owner decision, 2026-08-11). A doc saying "set
  ``WILLY_<X>_WEIGHTS`` to the checkpoint" is a working install guide for a licence we cannot use, and it
  reads as endorsement. Catching the name in prose catches the intent before the code exists.

**The asset rule is separate and stricter.** Meshes and textures carry their own licences, and
"the dataset is famous" is not a licence. Only CC0, CC-BY, or our own procedural assets; never NC. That
rule has no enforcement target yet — the scene generator is the thing that will produce a manifest for it
— so :func:`audit_asset_rows` is written and tested here now, and wired to the real manifest when one
exists. Better than discovering the rule was needed after 50k scenes are generated.

Mirrors the boundary the Aletheia data engine enforces (same owner, same commercial intent), so the two
repos cannot drift into different answers about the same dataset.
"""

from __future__ import annotations

import re
import unittest
from collections.abc import Iterable
from pathlib import Path

from datagen.assets.licensing import (
    FORBIDDEN,
    audit_asset_rows,
    forbidden_hits,
    is_noncommercial,
)

REPO = Path(__file__).resolve().parents[1]

#: The vocabulary lives in ``datagen/assets/licensing.py`` -- the code that actually has to obey it,
#: because that is where an asset row is judged before it can be rendered. Imported here rather than
#: duplicated so a name can never be forbidden in one place and allowed in the other. If datagen is
#: ever removed, this import fails loudly and the vocabulary must be re-homed, which is the correct
#: outcome: a licensing boundary should not be able to disappear quietly.

#: Files that MUST contain the forbidden names to do their job: this test, and any release note that
#: documents the boundary itself. Kept explicit and short -- an allowlist that grows without argument is
#: how a guard stops guarding.
ALLOWLIST: frozenset[str] = frozenset({
    "tests/test_license_boundary.py",
    # Attribution belongs in the file whose job is attribution. NOTICE names the projects whose
    # published EQUATIONS this repo re-implements (suction seal / wrench), and records what was removed
    # and why -- each next to its licence, which is precisely the context a scattered docstring mention
    # lacks. One reviewed file, not a hole: an install guide does not belong here either.
    "NOTICE",
    # Owns the FORBIDDEN vocabulary this scanner imports; it necessarily names every entry.
    "datagen/assets/licensing.py",
})

#: Text files worth scanning. Binary assets are not scanned: a name inside a mesh is not an install guide.
_SCANNED_SUFFIXES = (".py", ".md", ".yaml", ".yml", ".toml", ".txt", ".cfg", ".ini", ".ps1", ".bat")

#: Directories that are not ours to police (installed payloads, caches, build output, local logs).
_SKIP_DIRS = frozenset({
    ".git", ".venv", "venv", "ext_deps", "logs", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", "node_modules", "htmlcov", ".idea", ".vscode",
})

#: Repo-relative path prefixes that hold downloaded third-party payloads.
#:
#: Same reason as ``ext_deps`` above: a file this repository did not write is not this repository's
#: prose, and a scanner that reads it is reporting on somebody else's project. It is a prefix rather
#: than a name in ``_SKIP_DIRS`` because the membership test there runs per path component, so an
#: entry of "hf" would exempt every directory called ``hf`` anywhere in the tree.
#:
#: Measured, which is why the entry exists rather than being a precaution. A BERT wordpiece
#: vocabulary holds one token per line, and ``vocab.txt`` in ``IDEA-Research/grounding-dino-tiny``
#: holds ``acronym``. The forbidden-name check is a case-insensitive substring match, ACRONYM is a
#: forbidden grasp dataset, and ``.txt`` is scanned, so the first fetch of that detector turns this
#: gate red on an ordinary English noun inside a tokenizer.
#:
#: The skip is paired with a guard, because a skip list is how a hole gets in. See
#: ``test_the_weights_directory_holds_only_downloaded_files``: everything under the prefix has to
#: match the layout the downloaders themselves produce, so an authored file placed there is a
#: failure rather than an exemption. That is the counter-measure ``_SKIP_DIR_EXCEPTIONS`` below
#: records the need for, learned when ``ext_deps`` hid our own install guide.
_SKIP_PREFIXES = ("assets/models/hf/",)

#: Files inside a skipped directory that ARE ours and must be scanned anyway.
#:
#: MEASURED HOLE (2026-08-20). ``ext_deps/`` is skipped because it holds third-party payloads that are
#: not ours to police -- but the folder also carries OUR install guide, and that README shipped a
#: step-by-step recipe for the very network this boundary exists to keep out. The scanner reported
#: green the whole time. A skip list written for payloads silently covered the one authored file among
#: them, which is exactly the shape of hole a prose rule is supposed to catch.
_SKIP_DIR_EXCEPTIONS = frozenset({
    "ext_deps/README.md",
})

_IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+([a-zA-Z0-9_.]+)", re.MULTILINE)
_NORMALISED_FORBIDDEN = tuple(name.replace("-", "").replace("_", "") for name in FORBIDDEN)


def audit_imports(py_files: Iterable[Path]) -> list[str]:
    """Source files importing a forbidden top-level module (the vendored-copy route)."""
    problems: list[str] = []
    for path in py_files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in _IMPORT_RE.finditer(text):
            top = match.group(1).split(".")[0].lower().replace("_", "").replace("-", "")
            if any(name in top for name in _NORMALISED_FORBIDDEN):
                problems.append(f"{_rel(path)}: imports forbidden module {match.group(1)!r}")
    return problems


def audit_text(files: Iterable[Path]) -> list[str]:
    """Files naming a forbidden project in prose or code (the install-guide route)."""
    problems: list[str] = []
    for path in files:
        if _rel(path) in ALLOWLIST:
            continue
        for name in forbidden_hits(path.read_text(encoding="utf-8", errors="ignore")):
            problems.append(f"{_rel(path)}: mentions forbidden project {name!r}")
    return problems


def audit_distributions(dist_names: Iterable[str]) -> list[str]:
    """Installed distributions matching a forbidden project."""
    return [
        f"forbidden dependency {name!r} (matches {hit!r})"
        for name in dist_names
        for hit in forbidden_hits(name)
    ]


def _rel(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def _repo_files(suffixes: tuple[str, ...]) -> list[Path]:
    out: list[Path] = []
    for path in REPO.rglob("*"):
        if path.suffix.lower() not in suffixes or not path.is_file():
            continue
        rel = path.relative_to(REPO).as_posix()
        if rel not in _SKIP_DIR_EXCEPTIONS and any(
            part in _SKIP_DIRS for part in path.relative_to(REPO).parts
        ):
            continue
        if rel.startswith(_SKIP_PREFIXES):
            continue
        out.append(path)
    return out


class DownloadedWeightsAreNotOursToPoliceTests(unittest.TestCase):
    """The prefix above exempts downloaded payloads. This is what stops it exempting anything else.

    A directory the scanner does not read is a directory where a forbidden install guide could sit
    unseen, which is exactly what ``ext_deps`` did with our own README. The difference here is that
    the layout under the prefix is imposed by the downloaders rather than chosen by a person, so "did
    somebody author this" is a question the tree can answer: ``hub/`` and ``xet/`` are Hugging Face's
    own cache, ``mediapipe/`` holds the two ``.task`` bundles, ``torch/`` is ``TORCH_HOME``, and
    anything else was put there by hand.
    """

    #: What the fetchers produce. ``hub`` and ``xet`` come from ``HF_HOME``, ``mediapipe`` from
    #: ``scripts/model_weights/fetch.py --mediapipe``, ``torch`` from ``TORCH_HOME``.
    _EXPECTED_TOP_LEVEL = frozenset({"hub", "xet", "mediapipe", "torch"})

    def test_the_weights_directory_holds_only_downloaded_files(self) -> None:
        from src.utility.paths import weights_root

        root = weights_root()
        if not root.is_dir():
            self.skipTest(f"{root} does not exist; nothing has been fetched on this box")
        unexpected = sorted(
            child.name for child in root.iterdir()
            if child.name not in self._EXPECTED_TOP_LEVEL and not child.name.startswith(".")
        )
        self.assertEqual(
            unexpected, [],
            f"{root} is skipped by the licence scanner because it holds downloaded payloads. These "
            f"entries are not something a downloader produces, so they are authored files sitting "
            f"inside a directory nothing reads. Move them out, or add them to _SKIP_DIR_EXCEPTIONS "
            f"so they are scanned where they are.",
        )

    def test_the_skip_prefix_is_a_prefix_and_not_a_directory_name(self) -> None:
        """A name in ``_SKIP_DIRS`` matches per path component, so "hf" would exempt every directory
        called ``hf`` anywhere. The prefix form exempts one place."""
        for prefix in _SKIP_PREFIXES:
            with self.subTest(prefix=prefix):
                self.assertIn("/", prefix.rstrip("/"))
                self.assertTrue(prefix.endswith("/"))
                self.assertNotIn(prefix.rstrip("/").rsplit("/", 1)[-1], _SKIP_DIRS)

    def test_the_same_bytes_are_exempt_here_and_caught_elsewhere(self) -> None:
        """The discriminating control. One file, two locations, opposite outcomes, so the exemption
        is proved to be scoped rather than global and cannot quietly widen.

        The content is a real tokenizer line. ``acronym`` is an ordinary English noun and also a
        forbidden grasp dataset, and the check is a case-insensitive substring, which is how a BERT
        wordpiece vocabulary trips a licence gate.
        """
        from src.utility.paths import weights_root

        root = weights_root() / "hub"
        root.mkdir(parents=True, exist_ok=True)
        inside = root / "_licence_control_vocab.txt"
        outside = REPO / "assets" / "_licence_control_vocab.txt"
        self.addCleanup(inside.unlink, True)
        self.addCleanup(outside.unlink, True)
        for target in (inside, outside):
            target.write_text("acronym" + chr(10) + "anvil" + chr(10), encoding="utf-8")

        scanned = {_rel(p) for p in _repo_files((".txt",))}
        self.assertNotIn(_rel(inside), scanned, "a downloaded payload is being scanned")
        self.assertIn(_rel(outside), scanned, "the exemption reaches past the weights directory")

    def test_the_scanner_still_reads_the_rest_of_assets(self) -> None:
        """The control on scope. The decision was to exempt downloaded weights specifically, so a
        file elsewhere under ``assets/`` must still be scanned. If the prefix were widened to
        ``assets/``, this is what says so."""
        scanned = {_rel(p) for p in _repo_files((".md", ".txt", ".py"))}
        self.assertTrue(
            any(r.startswith("assets/") for r in scanned),
            "nothing under assets/ is scanned any more, so the exemption grew past weights",
        )
        self.assertFalse(
            any(r.startswith("assets/models/hf/") for r in scanned),
            "the weights exemption is not taking effect",
        )


class LicenseBoundaryTests(unittest.TestCase):
    def test_no_forbidden_imports(self) -> None:
        problems = audit_imports(_repo_files((".py",)))
        self.assertEqual(problems, [], "\n".join(["forbidden imports:", *problems]))

    def test_the_ext_deps_install_guide_is_scanned(self) -> None:
        """The one authored file inside a skipped payload directory must not escape the prose rule.

        Pinned because it DID escape it: `ext_deps/README.md` carried a full install recipe for a
        non-commercially-licensed network for months while this suite reported green, because
        `ext_deps/` is skipped wholesale as a third-party payload directory.
        """
        scanned = {_rel(p) for p in _repo_files(_SCANNED_SUFFIXES)}
        self.assertIn("ext_deps/README.md", scanned)

    def test_no_forbidden_project_is_named_anywhere(self) -> None:
        problems = audit_text(_repo_files(_SCANNED_SUFFIXES))
        self.assertEqual(
            problems, [],
            "\n".join([
                "a forbidden project is named in the tree:",
                *problems,
                "",
                "These projects are non-commercial or research-only (see FORBIDDEN in this file for the "
                "reason next to each name). If you are documenting the boundary itself rather than using "
                "the project, add the file to ALLOWLIST -- and say in the commit message why that is not "
                "a way around the rule.",
            ]),
        )

    def test_no_forbidden_distribution_is_installed(self) -> None:
        from importlib.metadata import distributions

        names = [dist.metadata["Name"] or "" for dist in distributions()]
        self.assertEqual(audit_distributions(names), [])

    def test_the_scanner_actually_scans_something(self) -> None:
        """A guard that silently matches zero files passes forever. Assert it has real inputs."""
        self.assertGreater(len(_repo_files((".py",))), 200)
        self.assertGreater(len(_repo_files(_SCANNED_SUFFIXES)), 300)

    def test_the_scanner_would_catch_a_reintroduction(self) -> None:
        """The positive control: the same adapter, re-added, must be caught by both surfaces."""
        self.assertEqual(forbidden_hits("import SuctionNet.network as net"), ["suctionnet"])
        self.assertTrue(forbidden_hits("weights from the GraspNet-1Billion release"))
        self.assertTrue(forbidden_hits("see docs/contact-graspnet-setup.md"))

    def test_noncommercial_detection(self) -> None:
        for license_id in ("CC BY-NC-SA 4.0", "cc-by-nc", "CC_BY_NC_SA_4.0", "Non-Commercial"):
            with self.subTest(license_id=license_id):
                self.assertTrue(is_noncommercial(license_id))
        for license_id in ("CC0-1.0", "CC-BY-4.0", "Apache-2.0", "MIT", "own"):
            with self.subTest(license_id=license_id):
                self.assertFalse(is_noncommercial(license_id))

    def test_asset_rows_are_audited(self) -> None:
        clean = [
            {"id": "cube_procedural", "license": "own", "source": "procedural"},
            # CC-BY obliges naming the author, including in derivative works -- and every rendered
            # image is one. A row without `attribution` is refused, which this fixture used to trip
            # over before the rule moved into datagen and got stricter than this test's old stub.
            {"id": "gso_mug", "license": "CC-BY-4.0", "source": "google-scanned-objects",
             "attribution": "Google Scanned Objects -- CC-BY 4.0"},
        ]
        self.assertEqual(audit_asset_rows(clean), [])

        dirty = [
            {"id": "a", "license": "", "source": "somewhere"},
            {"id": "b", "license": "CC BY-NC-SA 4.0", "source": "somewhere"},
            {"id": "c", "license": "GPL-3.0", "source": "somewhere"},
        ]
        problems = audit_asset_rows(dirty)
        self.assertEqual(len(problems), 3, problems)
        self.assertIn("missing license", problems[0])
        self.assertIn("non-commercial", problems[1])
        self.assertIn("not CC0 / CC-BY / own", problems[2])

    def test_a_forbidden_asset_source_is_caught_even_with_a_clean_licence(self) -> None:
        """The trap this closes: relabelling a research dataset's mesh as CC-BY does not make it CC-BY."""
        problems = audit_asset_rows([{"id": "x", "license": "CC-BY-4.0", "source": "ShapeNetCore v2"}])
        self.assertTrue(any("forbidden source" in p for p in problems), problems)


if __name__ == "__main__":
    unittest.main()
