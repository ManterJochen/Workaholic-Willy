"""Every model path the shipped config names must be one the fetch script can produce.

Measured on 2026-09-08, before this existed, three of them could not be. The base profile set
``local: true`` on `objectdetector`, `segmenter` and `stt`, pointing at
``src/models/{detection,segmentation,speech}/model``, and none of those directories exists in the
repository or is created by anything. The detector's ``model_id`` was the empty string as well, so it
could load neither locally nor from the hub. A cell built from the shipped tree could not assemble a
perception stack at all.

The check asks the fetch script, it does not keep a copy of the answer. A list of expected paths
written here would be a second declaration of the layout, maintained by whoever remembers this file
exists. `Weights.local_dir` computes where a model lands, so the config is compared against the thing
that puts it there.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import io
import sys
import tempfile
import types
import unittest
import zipfile
from collections.abc import Callable
from pathlib import Path
from unittest import mock

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from scripts.model_weights import fetch as fetch_script  # noqa: E402
from scripts.model_weights.fetch import CATALOGUE, PACKAGED_FILES  # noqa: E402
from src.utility.paths import weights_root  # noqa: E402


def _blocks() -> list[tuple[str, str | None, str | None, bool | None]]:
    """Every config block that names a model, as (name, model_id, model_path, local)."""
    from src.config.loader import load_config

    models = load_config().models
    out = []
    for name in ("objectdetector", "segmenter", "stt", "rtdetr", "oneformer"):
        block = getattr(models, name, None)
        if block is not None:
            out.append((name, getattr(block, "model_id", None),
                        getattr(block, "model_path", None), getattr(block, "local", None)))
    vlm = models.pipeline.zero_shot.vlm if models.pipeline is not None else None
    if vlm is not None:
        out.append(("vlm", vlm.model_id, vlm.model_path, vlm.local))
    return out


class EveryNamedModelIsInTheCatalogueTests(unittest.TestCase):
    def test_every_model_id_the_config_names_can_be_fetched(self) -> None:
        """A `local: true` path that no command produces is worse than `local: false`, which at
        least downloads. This is the half that makes `local: true` honest."""
        catalogue = {w.repo_id for w in CATALOGUE}
        missing = sorted(
            f"{name}: {model_id}" for name, model_id, _path, _local in _blocks()
            if model_id and model_id not in catalogue
        )
        self.assertEqual(
            missing, [],
            "the config names a model the fetch script cannot download, so its local path can "
            "never come into existence. Add it to CATALOGUE in scripts/model_weights/fetch.py.",
        )

    def test_no_block_names_a_model_by_nothing(self) -> None:
        """`objectdetector` shipped with `model_id: ""` and a path to a directory that does not
        exist, which is neither a local model nor a remote one."""
        for name, model_id, path, _local in _blocks():
            with self.subTest(block=name):
                self.assertTrue(model_id or path, f"{name} names neither a model id nor a path")


class EveryLocalPathIsWhereTheFetchPutsItTests(unittest.TestCase):
    def test_a_local_block_points_at_the_directory_the_fetch_script_writes(self) -> None:
        root = weights_root()
        by_id = {w.repo_id: w for w in CATALOGUE}
        checked = 0
        for name, model_id, path, local in _blocks():
            if not local or not model_id or model_id not in by_id:
                continue
            with self.subTest(block=name):
                expected = by_id[model_id].local_dir(root).relative_to(_REPO).as_posix()
                self.assertEqual(
                    Path(path).as_posix(), expected,
                    f"{name} is local: true at a path the fetch script does not produce",
                )
                checked += 1
        self.assertGreaterEqual(checked, 3, "a test over no local block passes loudest")

    def test_the_speech_section_names_the_voice_detector_where_the_fetch_puts_it(self) -> None:
        from src.config.loader import load_config

        stt = load_config().models.stt
        entry = next(f for f in PACKAGED_FILES if f.key == "silero-vad")
        self.assertEqual(
            Path(stt.vad_model_path).as_posix(),
            entry.local_path(weights_root()).relative_to(_REPO).as_posix(),
            "models.stt.vad_model_path is not the file `fetch.py silero-vad` writes",
        )

    def test_every_local_path_is_inside_the_weights_root(self) -> None:
        """The whole point of the location: deleting the repository takes the weights with it. A
        path outside it is a download that survives, which is what this move was for."""
        root = weights_root().relative_to(_REPO).as_posix()
        for name, _model_id, path, local in _blocks():
            if not local or not path:
                continue
            with self.subTest(block=name):
                self.assertTrue(
                    Path(path).as_posix().startswith(root),
                    f"{name} is local: true outside {root}, so a repo delete leaves it behind",
                )


class TheIgnoreListCannotEmptyAModelTests(unittest.TestCase):
    def test_the_one_repository_with_no_safetensors_has_no_ignore_list(self) -> None:
        """Measured against the hub: `shi-labs/oneformer_coco_swin_large` is 1.83 GB whole and
        0.00 GB of safetensors, so a blanket `*.bin` filter fetches it empty. The filter is per
        model for that reason, and this is the entry that proves it cannot be global."""
        oneformer = next(w for w in CATALOGUE if "oneformer" in w.repo_id)
        self.assertEqual(oneformer.ignore, ())

    def test_the_filtered_ones_still_keep_a_weight_format(self) -> None:
        """A control on the entry above: every other model is filtered, and none of the filters
        removes safetensors."""
        filtered = [w for w in CATALOGUE if w.ignore]
        self.assertGreaterEqual(len(filtered), 5)
        for w in filtered:
            with self.subTest(model=w.key):
                self.assertNotIn("*.safetensors", w.ignore)


def _speech_entries() -> list:
    """The catalogue entries that land under `speech/`, asked of the catalogue each time."""
    return [w for w in CATALOGUE if w.category == "speech"]


def _speech_files() -> list:
    """The packaged files that land under `speech/`, asked of the fetch script each time."""
    return [f for f in PACKAGED_FILES if f.category == "speech"]


def _speech_disagreements(root: Path) -> list[str]:
    """Every way the speech weights under `root` differ from the fetch script, one sentence each.

    The hub writes one record per downloaded file into `<local_dir>/.cache/huggingface/download`,
    and the first line of each `.metadata` record is the commit that file came from. MEASURED
    2026-09-11 on both speech directories of the main checkout: all records of one directory name
    the same commit. A packaged file (the voice detector, from a PyPI wheel) carries no such record,
    so its directory is compared by the sha256 of the file instead.
    """
    by_dir = {w.local_dir(root): w for w in _speech_entries()}
    by_file_dir = {f.local_path(root).parent: f for f in _speech_files()}
    found: list[str] = []
    speech = root / "speech"
    present = sorted(p for p in speech.iterdir() if p.is_dir() and not p.name.startswith("."))
    for directory in present:
        packaged = by_file_dir.get(directory)
        if packaged is not None:
            held_file = packaged.local_path(root)
            if not held_file.is_file():
                found.append(f"{directory.name}: holds no {held_file.name}")
            elif (digest := hashlib.sha256(held_file.read_bytes()).hexdigest()) != packaged.sha256:
                found.append(f"{directory.name}: {held_file.name} has sha256 {digest}, "
                             f"PACKAGED_FILES pins {packaged.sha256}")
            continue
        weights = by_dir.get(directory)
        if weights is None:
            found.append(f"{directory.name}: not a directory fetch.py produces; add it to CATALOGUE "
                         f"or PACKAGED_FILES with what it holds")
            continue
        records = sorted((directory / ".cache" / "huggingface" / "download").rglob("*.metadata"))
        if not records:
            found.append(f"{directory.name}: carries no hub download record, so the commit it "
                         f"holds cannot be read")
            continue
        held = sorted({(r.read_text(encoding="utf-8").splitlines() or [""])[0] for r in records})
        pin = getattr(weights, "revision", None)
        if held != [pin]:
            found.append(f"{directory.name}: holds commit {held}, CATALOGUE pins {pin}")
    return found


class EverySpeechEntryIsPinnedTests(unittest.TestCase):
    """A speech entry names the hub commit it downloads, and the download asks the hub for it.

    ⛔ `local_dir` IS STABLE ACROSS REVISIONS ON PURPOSE, so a config path can name it, and the same
    stability lets a second fetch replace the bytes under that path without a trace. The licence
    NOTICE gives for a model was read from one commit's card, and a bake-off measures one commit's
    weights, so both are claims about a commit that an unpinned entry cannot keep.
    """

    def test_every_speech_entry_carries_a_full_commit(self) -> None:
        entries = _speech_entries()
        self.assertTrue(entries, "no speech entry in CATALOGUE, so this would pass over nothing")
        for weights in entries:
            with self.subTest(model=weights.key):
                revision = getattr(weights, "revision", None)
                self.assertIsNotNone(
                    revision, f"{weights.repo_id} fetches whatever its default branch holds that day"
                )
                self.assertRegex(
                    str(revision), r"\A[0-9a-f]{40}\Z",
                    "a branch or tag can move; only a full commit id names one set of bytes",
                )

    def test_the_fetch_asks_the_hub_for_the_pinned_commit(self) -> None:
        """A pin that never reaches the download reads like a guarantee and is none."""
        for weights in _speech_entries():
            with self.subTest(model=weights.key), tempfile.TemporaryDirectory() as tmp:
                calls: list[dict] = []

                def download(**kwargs: object) -> str:
                    calls.append(kwargs)
                    into = Path(str(kwargs["local_dir"]))
                    into.mkdir(parents=True, exist_ok=True)
                    (into / "model.safetensors").write_bytes(b"")
                    return str(into)

                # A stand-in module rather than a patch on the real one: importing huggingface_hub
                # here would fix its cache constants before any later fence could move them.
                hub = types.SimpleNamespace(snapshot_download=download)
                with mock.patch.dict(sys.modules, {"huggingface_hub": hub}), mock.patch.object(
                    fetch_script, "use_system_trust_store", return_value="not asked in a test"
                ), contextlib.redirect_stdout(io.StringIO()):
                    fetch_script.fetch(weights, Path(tmp))

                self.assertEqual(len(calls), 1)
                self.assertIn("revision", calls[0], "the download was not handed a revision at all")
                self.assertEqual(calls[0]["revision"], getattr(weights, "revision", "<no field>"))


class TheSpeechWeightsOnThisBoxAreTheCatalogueTests(unittest.TestCase):
    """⭐ THE PIN IS COMPARED WITH THE BYTES ON DISK, NOT WITH A SECOND COPY OF ITSELF.

    Weights can reach `speech/` without this script, and the bake-off weights of 2026-09-11 did:
    a direct `snapshot_download` call put them into the directories the catalogue names.
    """

    def test_every_speech_model_on_this_box_is_a_catalogue_entry_at_its_pin(self) -> None:
        root = weights_root()
        if not (root / "speech").is_dir():
            self.skipTest(f"{root / 'speech'} does not exist; no speech weights were fetched here")
        self.assertEqual(_speech_disagreements(root), [])

    def test_the_comparison_names_each_way_to_disagree(self) -> None:
        """⭐ THE CONTROL. The test above skips wherever nothing was fetched, which is CI and most
        boxes, so the comparison is proved here on a built tree: one directory that agrees and one
        of each kind that does not."""
        entry = _speech_entries()[0]
        pin = getattr(entry, "revision", None) or "unpinned"
        other = "0" * 40
        cases: tuple[tuple[str, Callable[[Path], Path], str | None, str | None], ...] = (
            ("agrees", entry.local_dir, pin, None),
            ("another commit", entry.local_dir, other, f"holds commit ['{other}']"),
            ("not in the catalogue", lambda r: r / "speech" / "an_unlisted_model", pin,
             "not a directory fetch.py produces"),
            ("no download record", entry.local_dir, None, "carries no hub download record"),
        )
        for name, place, commit, expected in cases:
            with self.subTest(case=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                directory = place(root)
                directory.mkdir(parents=True)
                if commit is not None:
                    download = directory / ".cache" / "huggingface" / "download"
                    download.mkdir(parents=True)
                    (download / "config.json.metadata").write_text(
                        f"{commit}\nsome-etag\n", encoding="utf-8"
                    )
                found = _speech_disagreements(root)
                if expected is None:
                    self.assertEqual(found, [])
                else:
                    self.assertEqual(len(found), 1, found)
                    self.assertIn(directory.name, found[0])
                    self.assertIn(expected, found[0])

    def test_a_packaged_file_is_compared_by_its_bytes(self) -> None:
        """⭐ THE CONTROL for the voice detector: a file from a wheel, whose pin is a sha256."""
        entry = _speech_files()[0]
        stand_in = b"silero stand-in"
        pinned = dataclasses.replace(entry, sha256=hashlib.sha256(stand_in).hexdigest())
        cases: tuple[tuple[str, bytes | None, str | None], ...] = (
            ("agrees", stand_in, None),
            ("other bytes", b"something else", "PACKAGED_FILES pins"),
            ("no file", None, f"holds no {pinned.local_path(Path('.')).name}"),
        )
        this_module = sys.modules[__name__]
        for name, content, expected in cases:
            with self.subTest(case=name), tempfile.TemporaryDirectory() as tmp, \
                    mock.patch.object(this_module, "_speech_files", return_value=[pinned]):
                root = Path(tmp)
                target = pinned.local_path(root)
                target.parent.mkdir(parents=True)
                if content is not None:
                    target.write_bytes(content)
                found = _speech_disagreements(root)
                if expected is None:
                    self.assertEqual(found, [])
                else:
                    self.assertEqual(len(found), 1, found)
                    self.assertIn(expected, found[0])


class TheVoiceDetectorFileIsPinnedTests(unittest.TestCase):
    """Silero's model is a file inside the silero-vad wheel on PyPI, not a Hub repository.

    Its entry pins the wheel by sha256 and the file inside it by sha256, and the fetch keeps only bytes
    both hashes vouch for. MEASURED 2026-09-11: the wheel PyPI serves for silero-vad 6.2.1 hashes to the
    pinned value, and its `silero_vad/data/silero_vad.jit` to the value the installed package's RECORD
    gives for the same file.
    """

    def test_every_packaged_file_pins_its_wheel_and_the_member_by_sha256(self) -> None:
        self.assertTrue(PACKAGED_FILES, "no packaged file, so this would pass over nothing")
        for entry in PACKAGED_FILES:
            with self.subTest(file=entry.key):
                self.assertRegex(entry.wheel_sha256, r"\A[0-9a-f]{64}\Z")
                self.assertRegex(entry.sha256, r"\A[0-9a-f]{64}\Z")
                self.assertTrue(entry.wheel_url.startswith("https://files.pythonhosted.org/"),
                                entry.wheel_url)
                wheel = entry.wheel_url.rsplit("/", 1)[-1]
                self.assertTrue(wheel.startswith(f"{entry.package.replace('-', '_')}-{entry.version}-"),
                                wheel)

    def test_the_fetch_keeps_only_bytes_both_hashes_vouch_for(self) -> None:
        entry = PACKAGED_FILES[0]
        member = b"torchscript stand-in"
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as wheel:
            wheel.writestr(entry.member, member)
        wheel_bytes = archive.getvalue()
        good = dataclasses.replace(
            entry, wheel_sha256=hashlib.sha256(wheel_bytes).hexdigest(),
            sha256=hashlib.sha256(member).hexdigest(), size=len(member),
        )
        cases: tuple[tuple[str, object, str | None], ...] = (
            ("both agree", good, None),
            ("another wheel", dataclasses.replace(good, wheel_sha256="0" * 64), "wheel"),
            ("another member", dataclasses.replace(good, sha256="0" * 64), entry.member),
        )
        for name, pinned, refused in cases:
            with self.subTest(case=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                with mock.patch("urllib.request.urlopen", return_value=io.BytesIO(wheel_bytes)) as opened, \
                        mock.patch.object(fetch_script, "use_system_trust_store", return_value="not asked"), \
                        contextlib.redirect_stdout(io.StringIO()):
                    if refused is None:
                        fetch_script.fetch_file(pinned, root)
                    else:
                        with self.assertRaises(RuntimeError) as caught:
                            fetch_script.fetch_file(pinned, root)
                self.assertEqual(opened.call_args.args[0], good.wheel_url)
                target = good.local_path(root)
                if refused is None:
                    self.assertEqual(target.read_bytes(), member)
                else:
                    self.assertIn(refused, str(caught.exception))
                    self.assertFalse(target.exists(), "a refused download left a file behind")

    def test_the_script_fetches_the_voice_detector_by_its_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(fetch_script, "fetch_file", return_value=tmp) as fetched, \
                mock.patch.object(fetch_script, "fence_model_downloads", return_value=Path(tmp)), \
                contextlib.redirect_stdout(io.StringIO()) as listing:
            self.assertEqual(fetch_script.main(["silero-vad"]), 0)
            self.assertEqual(fetch_script.main(["--list"]), 0)
        self.assertEqual(fetched.call_args.args[0].key, "silero-vad")
        self.assertIn("silero-vad", listing.getvalue())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
