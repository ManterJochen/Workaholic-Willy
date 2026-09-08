"""What a machine without administrator rights can and cannot get through, held in place.

The install script's own docstring promises "no system conda, no PATH or registry changes, no admin
rights". Measured on 2026-09-08, three things contradicted it, and the one nobody would predict was
a registry key nothing mentions.

MAX_PATH. The deepest path the install produced was 281 characters against a 260 limit, and it
worked on the reference workstation only because ``HKLM\\SYSTEM\\CurrentControlSet\\Control\\
FileSystem\\LongPathsEnabled`` is 1, which needs administrator rights to set. All 40 over-length
paths belonged to two packages, ``cuda-nvvp`` and ``nsight-compute``, a visual profiler and a
profiler GUI that the planner never loads. Dropping them and their metapackages from the lock took
the deepest path to 242 on this root. That moves the gate rather than removing it: the deepest
survivor is 219 characters relative to the root, so a checkout root longer than 39 characters still
needs the key.

What is not proven here. These tests read the lock and the environment that a previous install
produced. The proof that the trimmed lock still builds a working planner is a ``-Clean`` rebuild,
which needs the network and several minutes, and has not been run. What is proven is narrower and
still worth holding: the removed packages contributed no file the build compiles or links against.
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_LOCK = _REPO / "ext_deps" / "locks" / "curobo_env.win-64.lock"
_ENV = _REPO / "ext_deps" / "curobo_env"

#: Windows refuses a path longer than this unless LongPathsEnabled is set, which needs admin. The
#: limit counts the terminating null, so the usable length is one less.
_MAX_PATH = 260

#: How long a checkout root the trimmed install still fits into.
#:
#: The trim moves this gate, it does not remove it, and the first version of this file claimed
#: otherwise. Measured after dropping the two profilers, the deepest path is 219 characters relative
#: to the root, and 39 + 1 separator + 219 is 259, the longest path Windows will accept. This
#: checkout is ``D:/dev/Workaholic-Willy``, 23 characters, which went from over the limit to 20
#: under it. A home directory checkout such as ``C:/Users/<name>/dev/Workaholic-Willy`` is around 42
#: and does not fit.
#:
#: The deepest survivor is a boost cmake file rather than a CUDA package, so trimming further would
#: mean touching something the build actually reads.
_ROOT_BUDGET = 39

#: Where the compile actually looks. ``build_compiled_backend.bat`` sets CUDA_HOME and LIB to
#: ``<env>/Library``, so a header or import library the build reads is one directly under
#: ``Library/include`` or ``Library/lib``. Anything deeper that merely happens to contain a directory
#: called ``lib`` belongs to somebody else's layout: a looser version of this pattern matched
#: ``libnvvp/plugins/.../lib/ant.jar`` and reported a Java profiler as a build dependency.
_BUILD_READS = re.compile(r"^Library/(include|lib)/")


def _lock_packages() -> list[str]:
    text = _LOCK.read_text(encoding="utf-8-sig")
    return [
        line.rsplit("/", 1)[-1].split(".conda")[0].split(".tar.bz2")[0]
        for line in text.splitlines()
        if line.startswith("https")
    ]


class TheLockIsSafeToEditByLineTests(unittest.TestCase):
    def test_the_lock_is_explicit_so_a_removed_line_removes_only_that_package(self) -> None:
        """This is why eight lines could be deleted without re-locking. An ``@EXPLICIT`` lock is a
        literal list of URLs that conda installs as given, with no solve, so removing one entry
        cannot change what any other entry resolves to. Re-locking would re-pin all of them, and
        every re-pin is a new binary whose reputation an application-control policy has not yet
        formed, which is the risk ``docs/code-integrity.md`` exists to avoid."""
        head = _LOCK.read_text(encoding="utf-8-sig").splitlines()[:6]
        self.assertIn("@EXPLICIT", head)

    def test_the_lock_keeps_its_byte_order_mark(self) -> None:
        """The house re-lock recipe pipes through ``Out-File -Encoding utf8``, which writes a BOM on
        PowerShell 5.1. A reader that opens this file with plain ``utf-8`` gets a stray character on
        the first line, so anything that rewrites it has to put the BOM back."""
        self.assertTrue(_LOCK.read_bytes().startswith(b"\xef\xbb\xbf"))


class NothingTheBuildNeedsWasRemovedTests(unittest.TestCase):
    """Derived from the environment rather than from a list of names somebody trusts.

    A hand-written "these packages are safe to drop" would be a second declaration of what the build
    needs, and it would be written by the person least able to check it. conda records the file list
    of every installed package in ``conda-meta``, so the tree can answer instead: a package that
    contributed a header or an import library is one the compile reads, and it must still be in the
    lock.
    """

    def _installed(self) -> dict[str, list[str]]:
        meta = _ENV / "conda-meta"
        if not meta.is_dir():
            self.skipTest(f"no environment at {_ENV}; run scripts/ext_deps/install.ps1")
        out: dict[str, list[str]] = {}
        for path in meta.glob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            name = payload.get("name")
            if isinstance(name, str):
                out[name] = [str(f) for f in payload.get("files", [])]
        return out

    def test_every_package_that_provides_a_header_or_import_library_is_still_locked(self) -> None:
        locked = {p.rsplit("-", 2)[0] for p in _lock_packages()}
        needed = {
            name for name, files in self._installed().items()
            if any(_BUILD_READS.match(f) for f in files)
        }
        self.assertTrue(needed, "no package appears to provide headers or import libraries")
        self.assertEqual(
            sorted(needed - locked), [],
            "a package the compile reads is installed here but no longer named in the lock, so a "
            "fresh install would not get it",
        )

    def test_the_packages_that_were_dropped_contributed_only_profiling_tools(self) -> None:
        """The measurement the removal rests on. Each dropped package either ships nothing at all
        (a metapackage) or ships a profiler, and none of them puts a file where the build looks."""
        installed = self._installed()
        dropped = (
            "cuda-nvvp", "nsight-compute", "cuda-visual-tools", "cuda-tools", "cuda-toolkit",
            "cuda-nvprof", "cuda-sanitizer-api", "cuda-command-line-tools",
        )
        present = [name for name in dropped if name in installed]
        if not present:
            self.skipTest("this environment predates the trim, or was built from the trimmed lock")
        for name in present:
            with self.subTest(package=name):
                stray = [f for f in installed[name] if _BUILD_READS.match(f)]
                self.assertEqual(stray, [], f"{name} puts files where the build looks")

    def test_none_of_them_is_named_in_the_lock_any_more(self) -> None:
        names = {p.rsplit("-", 2)[0] for p in _lock_packages()}
        for gone in ("cuda-nvvp", "nsight-compute", "cuda-toolkit"):
            with self.subTest(package=gone):
                self.assertNotIn(gone, names)


class TheDeepestPathFitsWithoutTheAdminKeyTests(unittest.TestCase):
    def test_the_installed_tree_stays_under_max_path_for_a_plausible_root(self) -> None:
        """The gate nobody predicts. Over MAX_PATH the install fails with an error that names a file
        rather than the limit, and the remedy is a registry key under HKLM.

        Measured against what is on disk, which still holds the two profilers' unpacked payloads in
        the package cache from before the trim. Those directories are what the trim stops arriving,
        so they are excluded here by name: the assertion is about what a fresh install produces.
        """
        ext = _REPO / "ext_deps"
        if not ext.is_dir():
            self.skipTest("nothing installed")
        stale = ("cuda-nvvp", "nsight-compute")
        worst, longest = None, 0
        for path in ext.rglob("*"):
            text = path.relative_to(_REPO).as_posix()
            if any(s in text for s in stale):
                continue
            if len(text) > longest:
                worst, longest = text, len(text)
        room = _MAX_PATH - longest - 1
        self.assertGreaterEqual(
            room, _ROOT_BUDGET,
            f"the deepest install path is {longest} characters ({worst}), which leaves {room} for "
            f"the checkout root. Beyond that the install needs HKLM LongPathsEnabled, and setting "
            f"that needs administrator rights.",
        )


class TheEmergencyFallbackIsNamedAsOneTests(unittest.TestCase):
    """The compiled backend is the normal case and the fallback is an emergency, so the script has
    to say which is which. A missing toolchain must not fail the install, and a failed build must."""

    def _script(self) -> str:
        return (_REPO / "scripts" / "ext_deps" / "install.ps1").read_text(encoding="utf-8")

    def test_a_missing_toolchain_continues_and_a_failed_build_does_not(self) -> None:
        text = self._script()
        self.assertIn("$rc -eq 2", text, "exit 2 from the build script is the no-toolchain case")
        self.assertIn("EMERGENCY FALLBACK", text)
        self.assertIn('throw "compiled backend build failed', text)

    def test_the_build_script_accepts_an_already_active_x64_toolchain(self) -> None:
        """The value and not the presence. A plain Developer Command Prompt sets this to x86, and
        setuptools would then hand torch a 32-bit environment for a 64-bit interpreter."""
        bat = (_REPO / "scripts" / "curobo" / "build_compiled_backend.bat").read_text(
            encoding="utf-8"
        )
        self.assertIn('"%VSCMD_ARG_TGT_ARCH%"=="x64"', bat)

    def test_the_script_writes_no_cache_outside_the_tree(self) -> None:
        """``git lfs install`` wrote the user's global .gitconfig and pip cached wheels in the user
        profile, 607 MB of it, both of which outlive a deleted ext_deps/."""
        text = self._script()
        self.assertIn("PIP_CACHE_DIR", text)
        self.assertNotIn("git lfs install |", text)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
