"""The mesh downloader, tested without a network.

⚠ WHAT THIS FILE CANNOT DO is verify the endpoints — that needs the internet, and a test suite that
reaches out is a suite that fails in an air-gapped clone for a reason unrelated to the code. The
endpoints were probed by hand on 2026-08-27 and the evidence is in the module docstring: Fuel
returned a 3.5 MB zip holding `meshes/model.obj`, and YCB an 11.7 MB tarball holding
`google_16k/nontextured.ply`. Five real meshes were fetched, measured and passed the licence audit.

What IS testable without a network is everything that decides what to do with those bytes, and that
is where the interesting failures live: the licence gate that must not be bypassed, the archive
layout the extractor depends on, and the distinction between "this object has no scan here" and
"this download failed" — which the exit code has to agree with.
"""

from __future__ import annotations

import io
import json
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from typing import Any
from unittest import mock

# ⭐ A PLAIN PACKAGE IMPORT SINCE 2026-09-04. This used to put `scripts/meshes/` on `sys.path`
# and `import fetch`, because the module lived outside any package. It is `datagen/assets/fetch.py`
# now, so the import needs no help and cannot pick up a different `fetch` from somewhere else.

from datagen.assets import fetch as mesh_fetch


def _gso_zip(*, with_mesh: bool = True) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.writestr("model.config", "<model/>")
        bundle.writestr("thumbnails/0.jpg", "not a jpeg")
        if with_mesh:
            bundle.writestr("meshes/model.obj", "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    return buffer.getvalue()


def _ycb_tgz(object_id: str, *, with_mesh: bool = True) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
        payload = b"ply\nformat ascii 1.0\nend_header\n"
        name = (f"{object_id}/google_16k/nontextured.ply" if with_mesh
                else f"{object_id}/google_16k/textured.obj")
        info = tarfile.TarInfo(name)
        info.size = len(payload)
        bundle.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


class TheLicenceGateTests(unittest.TestCase):
    """⛔ THE ONE THAT MATTERS. Fuel publishes a licence per model and this repo's CI audits licence
    strings; believing our own map instead is how a non-commercial mesh enters a customer's dataset
    wearing a CC-BY label."""

    def _run(self, licence: str) -> dict[str, int]:
        listing = json.dumps([{"name": "thing", "license_name": licence}]).encode()

        def responses(url: str, **_: object) -> bytes:
            return listing if "page=" in url else _gso_zip()

        with mock.patch.object(mesh_fetch, "_get", side_effect=responses):
            return mesh_fetch._gso_fetch(Path(tempfile.mkdtemp()), 1, lambda _m: None)

    def test_a_noncommercial_model_is_skipped_before_it_is_downloaded(self) -> None:
        counts = self._run("Creative Commons Attribution-NonCommercial 4.0")
        self.assertEqual(counts["skipped_license"], 1)
        self.assertEqual(counts["fetched"], 0)

    def test_a_model_with_no_stated_licence_is_skipped(self) -> None:
        """Silence is not permission. An empty `license_name` must not read as acceptable."""
        self.assertEqual(self._run("")["skipped_license"], 1)

    def test_cc_by_and_cc0_pass(self) -> None:
        for licence in ("Creative Commons Attribution 4.0 International",
                        "CC0 1.0 Universal", "Public Domain"):
            with self.subTest(licence=licence):
                self.assertEqual(self._run(licence)["fetched"], 1)


class ItFindsTheMeshInsideTheArchiveTests(unittest.TestCase):
    def test_gso_pulls_meshes_model_obj_out_of_the_zip(self) -> None:
        listing = json.dumps([{"name": "shark", "license_name": "CC-BY 4.0"}]).encode()
        library = Path(tempfile.mkdtemp())
        with mock.patch.object(mesh_fetch, "_get",
                               side_effect=lambda u, **_: listing if "page=" in u else _gso_zip()):
            counts = mesh_fetch._gso_fetch(library, 1, lambda _m: None)
        self.assertEqual(counts["fetched"], 1)
        self.assertTrue((library / "shark.obj").is_file())
        self.assertIn("f 1 2 3", (library / "shark.obj").read_text(encoding="utf-8"))

    def test_an_archive_without_the_mesh_is_a_failure_not_a_silent_skip(self) -> None:
        listing = json.dumps([{"name": "shark", "license_name": "CC-BY 4.0"}]).encode()
        with mock.patch.object(
                mesh_fetch, "_get",
                side_effect=lambda u, **_: listing if "page=" in u else _gso_zip(with_mesh=False)):
            counts = mesh_fetch._gso_fetch(Path(tempfile.mkdtemp()), 1, lambda _m: None)
        self.assertEqual(counts["failed"], 1)

    def test_ycb_pulls_nontextured_ply_out_of_the_tarball(self) -> None:
        index = json.dumps({"objects": ["002_master_chef_can"]}).encode()
        library = Path(tempfile.mkdtemp())
        with mock.patch.object(
                mesh_fetch, "_get",
                side_effect=lambda u, **_: index if u.endswith("objects.json")
                else _ycb_tgz("002_master_chef_can")):
            counts = mesh_fetch._ycb_fetch(library, None, lambda _m: None)
        self.assertEqual(counts["fetched"], 1)
        self.assertTrue((library / "002_master_chef_can.ply").is_file())


class AnAbsentScanIsNotAFailureTests(unittest.TestCase):
    """⚠ MEASURED AGAINST THE REAL SERVICE: `001_chips_can` has no google_16k scan -- it was captured
    only on the Berkeley rig. Counting that as an error makes a healthy run look broken and buries
    the downloads that really did fail, and the EXIT CODE has to agree with the sentence printed."""

    def _counts(self) -> dict[str, int]:
        index = json.dumps({"objects": ["001_chips_can", "002_master_chef_can"]}).encode()

        def responses(url: str, **_: object) -> bytes:
            if url.endswith("objects.json"):
                return index
            if "001_chips_can" in url:
                raise RuntimeError("GET ... failed: HTTP Error 404: Not Found")
            return _ycb_tgz("002_master_chef_can")

        with mock.patch.object(mesh_fetch, "_get", side_effect=responses):
            return mesh_fetch._ycb_fetch(Path(tempfile.mkdtemp()), None, lambda _m: None)

    def test_it_is_counted_apart_from_failures(self) -> None:
        counts = self._counts()
        self.assertEqual(counts["skipped_absent"], 1)
        self.assertEqual(counts["failed"], 0)
        self.assertEqual(counts["fetched"], 1)

    def test_the_run_still_exits_zero(self) -> None:
        index = json.dumps({"objects": ["001_chips_can"]}).encode()

        def responses(url: str, **_: object) -> bytes:
            if url.endswith("objects.json"):
                return index
            raise RuntimeError("HTTP Error 404: Not Found")

        with mock.patch.object(mesh_fetch, "_get", side_effect=responses):
            code = mesh_fetch.fetch(["ycb"], library=Path(tempfile.mkdtemp()), limit=None,
                                    report=lambda _m: None)
        self.assertEqual(code, 0)


class ItResumesTests(unittest.TestCase):
    def test_an_already_present_mesh_is_not_downloaded_again(self) -> None:
        """These are gigabytes over a network. A fetch that restarts from zero after an interruption
        is a fetch nobody runs twice."""
        library = Path(tempfile.mkdtemp())
        library.mkdir(parents=True, exist_ok=True)
        (library / "shark.obj").write_text("already here", encoding="utf-8")
        listing = json.dumps([{"name": "shark", "license_name": "CC-BY 4.0"}]).encode()

        def responses(url: str, **_: object) -> bytes:
            if "page=" in url:
                return listing
            raise AssertionError("it downloaded a mesh that was already on disk")

        with mock.patch.object(mesh_fetch, "_get", side_effect=responses):
            counts = mesh_fetch._gso_fetch(library, 1, lambda _m: None)
        self.assertEqual(counts["skipped_present"], 1)
        self.assertEqual((library / "shark.obj").read_text(encoding="utf-8"), "already here")


class TheCatalogueTellsTheTruthTests(unittest.TestCase):
    def test_it_declares_which_licence_it_actually_verified(self) -> None:
        """`gso` checks a per-model licence at the source; `ycb` cannot, because the source states
        none. Reporting both the same way would claim a check that never happened."""
        self.assertTrue(mesh_fetch.SOURCES["gso"].licence_is_verified)
        self.assertFalse(mesh_fetch.SOURCES["ycb"].licence_is_verified)

    def test_the_written_suffix_matches_what_the_library_globs_for(self) -> None:
        """⭑ THE SEAM. The fetcher writes `<name>.obj` / `<id>.ply` and `MeshLibrary.available()`
        globs per source. If those two ever disagree the download succeeds and the library reports
        an empty bank -- a failure with no error message anywhere."""
        from datagen.assets.library import SUPPORTED_SOURCES
        for key in mesh_fetch.SOURCES:
            with self.subTest(source=key):
                self.assertIn(key, SUPPORTED_SOURCES)
        self.assertEqual(SUPPORTED_SOURCES["gso"], "obj")
        self.assertEqual(SUPPORTED_SOURCES["ycb"], "ply")

    def test_listing_exits_zero_and_names_both_collections(self) -> None:
        lines: list[str] = []
        self.assertEqual(mesh_fetch._list(lines.append), 0)
        text = "\n".join(lines)
        self.assertIn("gso", text)
        self.assertIn("ycb", text)

    def test_naming_nothing_is_a_usage_error_not_a_silent_success(self) -> None:
        self.assertEqual(mesh_fetch.main([]), 2)


class TheRetryPolicyTests(unittest.TestCase):
    def test_a_404_is_not_retried(self) -> None:
        """Three attempts at an object that does not exist is three times the wait for the same
        answer, and YCB alone has ~20 of them."""
        from urllib.error import HTTPError
        attempts = []

        def urlopen(*_a: object, **_k: object) -> None:
            attempts.append(1)
            raise HTTPError("http://x", 404, "Not Found", {}, None)  # type: ignore[arg-type]

        with mock.patch.object(mesh_fetch, "urlopen", side_effect=urlopen):
            with self.assertRaises(RuntimeError):
                mesh_fetch._get("http://x")
        self.assertEqual(len(attempts), 1)

    def test_a_timeout_is_retried(self) -> None:
        attempts = []

        def urlopen(*_a: object, **_k: object) -> None:
            attempts.append(1)
            raise TimeoutError("slow")

        with mock.patch.object(mesh_fetch, "urlopen", side_effect=urlopen):
            with self.assertRaises(RuntimeError):
                mesh_fetch._get("http://x", retries=3)
        self.assertEqual(len(attempts), 3)


if __name__ == "__main__":                                           # pragma: no cover
    unittest.main()


class TheLicenceGateRefusesBeforeItAcceptsTests(unittest.TestCase):
    """⛔ THE DEFECT THIS FILE CAUGHT. A permissive substring test accepts "Creative Commons
    Attribution-NonCommercial 4.0", because the acceptable phrase is inside the forbidden one. The
    gate built to keep NC meshes out would have waved through every NC model Fuel publishes."""

    def test_every_flavour_of_noncommercial_is_refused(self) -> None:
        for licence in (
            "Creative Commons Attribution-NonCommercial 4.0",
            "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International",
            "CC BY-NC 4.0",
            "CC-BY-NC-SA-4.0",
            "cc by nc nd 3.0",
        ):
            with self.subTest(licence=licence):
                self.assertFalse(mesh_fetch._licence_is_acceptable(licence))

    def test_noderivatives_is_refused_because_a_render_is_a_derivative(self) -> None:
        self.assertFalse(mesh_fetch._licence_is_acceptable("CC BY-ND 4.0"))

    def test_the_permissive_ones_still_pass(self) -> None:
        for licence in ("Creative Commons Attribution 4.0 International", "CC-BY-4.0",
                        "CC0 1.0 Universal", "Public Domain Dedication"):
            with self.subTest(licence=licence):
                self.assertTrue(mesh_fetch._licence_is_acceptable(licence))

    def test_it_reuses_the_repos_own_judgement_rather_than_a_second_one(self) -> None:
        """One place decides what non-commercial means. A downloader with its own opinion is how the
        two drift apart, and the drift is invisible until a dataset ships."""
        from datagen.assets.licensing import is_noncommercial
        for licence in ("CC BY-NC 4.0", "CC-BY-NC-SA-4.0", "Attribution-NonCommercial"):
            with self.subTest(licence=licence):
                self.assertTrue(is_noncommercial(licence))
                self.assertFalse(mesh_fetch._licence_is_acceptable(licence))


class ThePaginationEndTests(unittest.TestCase):
    """⛔ THE DEFECT THAT MADE A FULL FETCH DOWNLOAD NOTHING. Fuel's catalogue is 1,033 models, so
    page 11 is the last and page 12 answers **404, not an empty list**. The loop treated that as a
    failure and refused the whole source — after listing every real page successfully. Walking off the
    end of a paginated catalogue is an ANSWER, not an error."""

    def test_a_404_on_a_listing_page_ends_the_catalogue(self) -> None:
        from urllib.error import HTTPError

        pages = {1: [{"name": "a", "license_name": "CC-BY 4.0"}],
                 2: [{"name": "b", "license_name": "CC-BY 4.0"}]}

        def responses(url: str, **kwargs: object) -> bytes | None:
            page = int(url.split("page=")[1].split("&")[0])
            if page in pages:
                return json.dumps(pages[page]).encode()
            if kwargs.get("allow_missing"):
                return None
            raise HTTPError(url, 404, "Not Found", {}, None)  # type: ignore[arg-type]

        with mock.patch.object(mesh_fetch, "_get", side_effect=responses):
            self.assertEqual([name for name, _ in mesh_fetch._gso_catalogue(None)], ["a", "b"])

    def test_a_404_elsewhere_still_raises(self) -> None:
        """The control: `allow_missing` must not turn every 404 into silence."""
        from urllib.error import HTTPError

        def urlopen(*_a: object, **_k: object) -> None:
            raise HTTPError("http://x", 404, "Not Found", {}, None)  # type: ignore[arg-type]

        with mock.patch.object(mesh_fetch, "urlopen", side_effect=urlopen):
            with self.assertRaises(RuntimeError):
                mesh_fetch._get("http://x")
            self.assertIsNone(mesh_fetch._get("http://x", allow_missing=True))


class ItRunsAsASCRIPTTests(unittest.TestCase):
    """⛔ THE ENVIRONMENT THE TESTS DO NOT REPRODUCE. Every test above IMPORTS this module, and pytest
    runs with the repo root already on `sys.path`. An operator runs the FILE — and then only the
    file's OWN directory is on the path, so the lazy `from datagen.assets.licensing import ...` inside
    the licence gate raises ModuleNotFoundError.

    ⚠ MOVING THE FILE INTO THE PACKAGE DID NOT RETIRE THIS. `python -m datagen.assets.fetch` is safe,
    but `python datagen/assets/fetch.py` puts only `datagen/assets/` on the path and is exactly as
    exposed as the old `python scripts/meshes/fetch.py` was. The module still carries the
    `sys.path.insert` that makes both work, and this test is what keeps it honest. MEASURED: a full fetch listed all 1,033 models and then
    died on the first licence check, with the whole suite green.

    Running it as a subprocess is the only way to test the thing the operator actually does.
    """

    def _run(self, *args: str) -> Any:
        import subprocess
        import sys as _sys
        from pathlib import Path as _Path

        # ⚠ THE WHOLE PATH, NOT ASSEMBLED FROM SEGMENTS. Built as `"scripts" / "meshes" / "fetch.py"`
        # the string `scripts/meshes/fetch.py` existed nowhere in this file, so when the module moved
        # into the package on 2026-09-04 no rename pass could see it: not the AST sweep over imports,
        # not the dotted-string sweep, not the prose sweep. It failed at RUN time on a path spelled
        # one segment at a time.
        script = _Path(__file__).resolve().parents[1] / "datagen/assets/fetch.py"
        return subprocess.run([_sys.executable, str(script), *args], capture_output=True,
                              text=True, timeout=180, cwd=str(script.parents[2]))

    def test_the_licence_gate_is_importable_from_a_bare_script_run(self) -> None:
        """`--list` alone would not catch it -- the import is inside the gate. This runs the gate."""
        import subprocess

        result = self._run("--list")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("gso", result.stdout)

        # And the gate itself, in the same interpreter the operator gets.
        from pathlib import Path as _Path
        import sys as _sys
        script_dir = _Path(__file__).resolve().parents[1] / "datagen/assets"
        probe = subprocess.run(
            [_sys.executable, "-c",
             "import sys; sys.path.insert(0, r'%s'); import fetch; "
             "print(fetch._licence_is_acceptable('CC-BY 4.0'), "
             "fetch._licence_is_acceptable('CC BY-NC 4.0'))" % script_dir],
            capture_output=True, text=True, timeout=180)
        self.assertEqual(probe.returncode, 0, probe.stderr)
        self.assertIn("True False", probe.stdout)

    def test_naming_nothing_exits_two_as_a_script(self) -> None:
        self.assertEqual(self._run().returncode, 2)


class TheUrlEncodingTests(unittest.TestCase):
    """⛔ THE THIRD DEFECT THIS FETCHER HAD, and the third one only a real run could find. GSO ships
    model names with accented characters; `urllib.request.Request` encodes the URL as ASCII, so a full
    fetch died mid-run on `\xe9`. Every fixture in this file used ASCII names — which is exactly why
    the suite stayed green through it."""

    def test_a_non_ascii_model_name_is_percent_encoded(self) -> None:
        encoded = mesh_fetch._encode(
            "https://fuel.gazebosim.org/1.0/GoogleResearch/models/Caf\u00e9_Mug.zip")
        self.assertNotIn("\u00e9", encoded)
        self.assertIn("Caf%C3%A9_Mug.zip", encoded)

    def test_the_url_structure_survives(self) -> None:
        """`safe` must keep the query and the path separators, or the request goes somewhere else."""
        url = "https://x/models?page=2&per_page=100"
        self.assertEqual(mesh_fetch._encode(url), url)

    def test_an_ascii_url_is_unchanged(self) -> None:
        url = "https://x/models/Plain_Name.zip"
        self.assertEqual(mesh_fetch._encode(url), url)

    def test_an_already_encoded_url_is_not_double_encoded(self) -> None:
        """`%` is in the safe set: re-encoding `%C3%A9` into `%25C3%25A9` would 404 every retry."""
        url = "https://x/models/Caf%C3%A9.zip"
        self.assertEqual(mesh_fetch._encode(url), url)
