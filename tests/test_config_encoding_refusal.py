"""A YAML saved in the wrong encoding must REFUSE by name, not raise a traceback.

MEASURED, before the guard existed. ``python -m src.config --data <tree>`` against a copy of
``config`` whose ``camera/cam.yaml`` had been re-saved the way Notepad's "Save as ANSI"
saves it (one umlaut in a COMMENT is enough) printed::

    File "...\\src\\config\\loader.py", line 528, in _load_yaml
        text = path.read_text(encoding="utf-8")
    UnicodeDecodeError: 'utf-8' codec can't decode byte 0xf6 in position 9: invalid start byte

No file name in the last line, a byte offset instead, and a traceback out of the one tool whose whole
job is to validate the tree and print a refusal. The SAME tree broken the ordinary way (bad YAML
indentation, still UTF-8) produced the designed ``config error: YAML parse error in <file>``, so the
machinery to say it properly already existed and was being walked past.

⛔ **THE GUARD COULD NOT FIRE, AND LOOKED LIKE IT COULD.** ``_load_yaml`` wrapped the read in
``except OSError``. ``UnicodeDecodeError.__mro__`` is ``(UnicodeDecodeError, UnicodeError, ValueError,
Exception, BaseException, object)``: there is no ``OSError`` in it. Two shipped CLIs
(``src.robot.drivers.ur``, ``src.robot.execution.real_cell.calibrate``) carry
``ConfigError`` handlers written for exactly this moment and could never reach them.
:meth:`EncodingGuardCannotBeNarrowedTests.test_unicode_decode_error_is_not_an_oserror` is that fact
as an assertion, so narrowing the clause back to ``OSError`` alone fails here with the reason.
"""

from __future__ import annotations

import io
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from src.config import ConfigError, load_config, reload_config
from src.config.loader import set_active_profile

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "config"

# The file every one of these trees breaks. It is loaded on the REQUIRED path, so a tree carrying it
# cannot come up half-way and hide the refusal behind a later schema error.
BROKEN_RELATIVE = Path("camera") / "cam.yaml"


def _tree_with_encoding(tmp_dir: str, encoding: str, *, errors: str = "strict") -> tuple[Path, Path]:
    """Copy the real data tree and re-save ONE file in ``encoding``. Returns (root, broken file).

    A comment line carrying two umlauts goes in at the top, because that is the operator's actual
    mistake: nothing about the CONFIG changed, only the bytes a text editor wrote it in.
    """
    root = Path(tmp_dir) / "data"
    shutil.copytree(DATA_DIR, root)
    broken = root / BROKEN_RELATIVE
    # Written as escapes so THIS file stays ASCII: the bytes under test are the point, and a source
    # file carrying them raw is one editor away from becoming the very bug it tests for.
    text = "# Kamerah\u00f6he \u00fcber dem Tisch\n" + broken.read_text(encoding="utf-8")
    broken.write_bytes(text.encode(encoding, errors=errors))
    # ⚠ RESOLVED, or this test fails on the message it is asking for. `tempfile` hands back the 8.3
    # short name on this box (`...\TIMKAC~1\...`) and the loader resolves its root to the long one
    # (`...\Tim Kackstein\...`), so an unresolved comparison rejects a refusal that names the file
    # perfectly well.
    return root, broken.resolve()


class MisEncodedConfigRefusalTests(unittest.TestCase):
    def setUp(self) -> None:
        self._previous_profile = os.environ.get("WILLY_PROFILE")
        set_active_profile(None)
        reload_config()

    def tearDown(self) -> None:
        set_active_profile(self._previous_profile)
        reload_config()

    def _assert_names_the_file_and_the_cure(self, message: str, broken: Path) -> None:
        self.assertIn(str(broken), message, "the refusal must name the FILE, not a byte offset")
        self.assertIn("UTF-8", message, "the refusal must say what the file is not")
        self.assertIn("re-save", message.lower(), "the refusal must say what the operator can DO")

    def test_cp1252_yaml_is_refused_and_names_the_file(self) -> None:
        # Notepad "Save as ANSI": unrepresentable characters become '?', the umlauts become
        # single cp1252 bytes. This is the reproduction, not a synthetic byte.
        with tempfile.TemporaryDirectory() as tmp_dir:
            root, broken = _tree_with_encoding(tmp_dir, "cp1252", errors="replace")

            with self.assertRaises(ConfigError) as caught:
                load_config(data_dir=root)

            message = str(caught.exception)
            self._assert_names_the_file_and_the_cure(message, broken)
            # The byte that stopped the read is worth keeping: it is what an editor's encoding menu
            # is searched for. Losing it would make the refusal unactionable on a file whose bad
            # character is not visible in a diff.
            self.assertIn("0xf6", message.lower())

    def test_utf16_bom_yaml_is_refused_and_names_the_encoding(self) -> None:
        # A BOM-carrying file is a different symptom (it dies on byte 0 of the file, not on the
        # umlaut) with the same cause, and it is what Notepad's "Unicode" and PowerShell's
        # `-Encoding Unicode` both write.
        with tempfile.TemporaryDirectory() as tmp_dir:
            root, broken = _tree_with_encoding(tmp_dir, "utf-16")

            with self.assertRaises(ConfigError) as caught:
                load_config(data_dir=root)

            message = str(caught.exception)
            self._assert_names_the_file_and_the_cure(message, broken)
            self.assertIn("UTF-16", message, "a BOM is a positive identification; say what it is")

    def test_utf16_without_a_bom_is_not_diagnosed_as_cp1252(self) -> None:
        # MEASURED while building the guard: this file DOES reach the refusal (its umlaut is
        # `\xf6\x00`, and `\xf6` is an invalid UTF-8 start byte), but the no-mark branch called it
        # "usually cp1252", a confident wrong answer that sends the operator to the one menu entry
        # that cannot help. A NUL in the first four bytes rules every single-byte encoding out.
        with tempfile.TemporaryDirectory() as tmp_dir:
            root, broken = _tree_with_encoding(tmp_dir, "utf-16-le")

            with self.assertRaises(ConfigError) as caught:
                load_config(data_dir=root)

            message = str(caught.exception)
            self._assert_names_the_file_and_the_cure(message, broken)
            self.assertIn("UTF-16", message)
            self.assertNotIn("cp1252", message, "a NUL byte rules out every single-byte encoding")

    def test_utf8_with_a_bom_still_loads(self) -> None:
        # ⚠ THE NEGATIVE HALF, and it is the one a guard like this gets wrong. A UTF-8 file with a
        # byte-order mark IS valid UTF-8 (it is what Notepad writes by DEFAULT), and PyYAML
        # accepts the mark. Refusing it would turn this repair into a worse bug than the one it
        # fixes: every Windows operator's ordinary save would stop the cell.
        with tempfile.TemporaryDirectory() as tmp_dir:
            root, _ = _tree_with_encoding(tmp_dir, "utf-8-sig")

            cfg = load_config(data_dir=root)

            self.assertEqual(cfg.camera.cameras.primary_rig_id, "webcam_main")

    def test_a_utf8_bom_file_with_a_later_bad_byte_is_not_blamed_on_its_encoding(self) -> None:
        # ⚠ THE ROW THAT GARBLED. A UTF-8 byte-order mark is VALID UTF-8, so reaching the BOM table
        # with one means the file STARTS right and something pasted another encoding in further
        # down. The first version slotted every mark into one shared sentence and told this operator
        # to re-save a file whose encoding was already correct, and said "Notepad calls that
        # 'Unicode'", which is true of UTF-16 and false here.
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "data"
            shutil.copytree(DATA_DIR, root)
            broken = root / BROKEN_RELATIVE
            good = broken.read_text(encoding="utf-8")
            broken.write_bytes(good.encode("utf-8-sig") + b"# H\xf6he\n")

            with self.assertRaises(ConfigError) as caught:
                load_config(data_dir=root)

            message = str(caught.exception)
            self.assertIn(str(broken.resolve()), message)
            self.assertIn("UTF-8 byte-order mark", message)
            self.assertNotIn("Unicode'", message, "that is the UTF-16 sentence, not this one")
            self.assertNotIn("byte-order mark byte-order mark", message)

    def test_the_refusal_is_ascii_because_a_cp1252_console_prints_it(self) -> None:
        # The operator meeting this message is on the console that caused the problem. A refusal
        # carrying a character that console cannot encode would raise UnicodeEncodeError on its way
        # out: the config tool crashing on the config, which this repo has already measured once.
        with tempfile.TemporaryDirectory() as tmp_dir:
            root, _ = _tree_with_encoding(tmp_dir, "cp1252", errors="replace")

            with self.assertRaises(ConfigError) as caught:
                load_config(data_dir=root)

            message = str(caught.exception)
            # The PATH is the operator's own and may be anything; the sentences we wrote must not be.
            authored = message.replace(str(root), "")
            self.assertTrue(authored.isascii(), ascii(authored))

    def test_refusal_does_not_re_decode_the_tree_in_another_encoding(self) -> None:
        # ⛔ The tempting "fix" is to retry in cp1252 and carry on. A tree that loads differently
        # depending on the machine's code page is worse than one that refuses, so a mis-encoded tree
        # must NEVER produce an AppConfig, on any code page.
        with tempfile.TemporaryDirectory() as tmp_dir:
            root, _ = _tree_with_encoding(tmp_dir, "cp1252", errors="replace")

            with self.assertRaises(ConfigError):
                load_config(data_dir=root)

    def test_cli_prints_a_refusal_and_exits_1(self) -> None:
        # The surface the operator actually meets. `python -m src.config` is the tool whose whole
        # job is to validate the tree; it was the one printing the traceback.
        from src.config.__main__ import main

        with tempfile.TemporaryDirectory() as tmp_dir:
            root, broken = _tree_with_encoding(tmp_dir, "cp1252", errors="replace")

            out, err = io.StringIO(), io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                exit_code = main(["--data", str(root)])

            printed = out.getvalue() + err.getvalue()
            self.assertEqual(exit_code, 1, printed)
            self.assertIn("config error:", printed)
            self._assert_names_the_file_and_the_cure(printed, broken)


class EncodingGuardCannotBeNarrowedTests(unittest.TestCase):
    """The MRO fact the guard rests on, asserted so that re-narrowing it fails loudly."""

    def test_unicode_decode_error_is_not_an_oserror(self) -> None:
        # This is why `except OSError` around `path.read_text(encoding="utf-8")` read like a guard
        # for years and caught nothing. If a later edit narrows the clause back on the belief that
        # OSError covers a bad decode, this assertion states the measurement that refutes it.
        self.assertNotIsInstance(
            UnicodeDecodeError("utf-8", b"\xf6", 0, 1, "invalid start byte"), OSError
        )
        self.assertNotIn(OSError, UnicodeDecodeError.__mro__)

    def test_config_error_does_not_inherit_the_decode_error(self) -> None:
        # Guards the tests above from passing for the wrong reason: if ConfigError ever grew
        # UnicodeDecodeError as a base, `assertRaises(ConfigError)` would go green on the raw
        # traceback this whole file exists to prevent.
        self.assertNotIn(UnicodeDecodeError, ConfigError.__mro__)


if __name__ == "__main__":  # pragma: no cover - manual runs
    unittest.main()
