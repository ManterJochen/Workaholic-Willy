"""The Linux environments are the Windows ones at the same versions, wherever the platform lets them be (P0h h5).

The Windows locks are the environments measured to plan on the reference box. The Linux locks were made in
Ubuntu-24.04 on 2026-09-26 by asking conda-forge for the Windows versions by name (Python, every part of the CUDA
toolkit, Coal and its octomap, Coal at the Windows build number), and by resolving cuRobo's pip half with the
Windows pip lock as constraints. Each lock checked on its own would stay green through a re-lock that moved Linux to
another CUDA or another Coal, and the Coal agreement across platforms (P0h h6) and the planner's numbers rest on the
two being the same. So the pair is held here.

What differs has a reason the platform gives, and each reason is named where it is allowed:

* the compilers and the system libraries: MSVC's runtime on Windows, GCC 13 and glibc's neighbours on Linux, which
  no test here holds;
* four CUDA packages that exist on Linux only: ``cuda-driver-dev`` and its ``_linux-64`` part (the driver stub the
  link resolves ``libcuda`` against; Windows links ``cuda.lib`` from ``cuda-cudart-dev``) and ``libcufile`` with its
  headers (GPUDirect Storage, which NVIDIA ships for Linux only);
* Coal's ``eigen-abi``: the same Coal 3.0.4 build 0 was compiled against Eigen ABI 5.0.1.100 on Windows and
  5.0.1.80 on Linux, which a lock cannot choose;
* pip: on Linux ``torch==2.7.0+cu128`` depends on the CUDA 12.8 runtime wheels (``nvidia-*-cu12``) and ``triton``,
  while on Windows torch bundles its CUDA and cuRobo's ``[cu12]`` pulled the 12.9 wheels; that is why the Windows
  pip lock does not resolve on Linux (measured: ``nvidia-cuda-nvrtc-cu12==12.9.86`` against torch). ``colorama`` is
  a Windows-only dependency of the console libraries.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

_LOCKS = Path(__file__).resolve().parents[1] / "ext_deps" / "locks"
_URL = re.compile(r"/(?:win-64|linux-64|noarch)/(?P<file>[^/#]+?)\.(?:conda|tar\.bz2)#")
#: conda-forge names a few CUDA packages per target (``cuda-cudart_win-64``, ``cuda-cudart_linux-64``).
_TARGET = re.compile(r"_(?:win|linux)-64$")

#: The conda packages the planner's numbers come from.
_COAL = ("python", "coal", "coal-python", "libcoal", "octomap", "assimp", "qhull", "libboost")
#: Present on Linux only, each for the reason the module docstring gives.
_LINUX_ONLY_CUDA = frozenset({"cuda-driver-dev", "cuda-driver-dev_<target>", "libcufile", "libcufile-dev"})
#: The pip packages torch decides on Linux, and the one Windows alone needs.
_PIP_PLATFORM = re.compile(r"^(nvidia-.*-cu12|triton|colorama)$")


def _conda(name: str, platform: str) -> dict[str, tuple[str, str]]:
    """``{package: (version, build)}`` of one conda lock, the per-target suffix spelled ``_<target>`` on both.

    Replaced, not dropped: ``cuda-cudart`` and ``cuda-cudart_win-64`` are two packages, and folding the suffix away
    let the second overwrite the first, which would hide a version that differs in one of them."""
    out: dict[str, tuple[str, str]] = {}
    for line in (_LOCKS / f"{name}.{platform}.lock").read_text(encoding="utf-8-sig").splitlines():
        match = _URL.search(line)
        if match:
            package, version, build = match["file"].rsplit("-", 2)
            out[_TARGET.sub("_<target>", package)] = (version, build)
    return out


def _pip(platform: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in (_LOCKS / f"curobo_env.{platform}.pip.lock").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            name, version = line.split("==")
            out[name.lower()] = version
    return out


def _cuda(name: str) -> bool:
    return name.startswith(("cuda-", "libcu", "libnv"))


class TheCudaToolkitIsTheSameTests(unittest.TestCase):
    def test_every_cuda_package_windows_runs_is_on_linux_at_its_version(self) -> None:
        win, lin = _conda("curobo_env", "win-64"), _conda("curobo_env", "linux-64")
        cuda = sorted(name for name in win if _cuda(name))
        self.assertGreater(len(cuda), 40)
        for name in cuda:
            with self.subTest(name):
                self.assertIn(name, lin)
                self.assertEqual(lin[name][0], win[name][0])

    def test_linux_adds_only_the_cuda_packages_windows_has_no_build_of(self) -> None:
        win, lin = _conda("curobo_env", "win-64"), _conda("curobo_env", "linux-64")
        self.assertEqual({name for name in lin if _cuda(name)} - set(win), _LINUX_ONLY_CUDA)

    def test_the_interpreter_is_the_same(self) -> None:
        self.assertEqual(_conda("curobo_env", "linux-64")["python"][0], _conda("curobo_env", "win-64")["python"][0])


class CoalIsTheSameBuildTests(unittest.TestCase):
    def test_coal_and_what_it_links_are_the_windows_versions(self) -> None:
        win, lin = _conda("coal_env", "win-64"), _conda("coal_env", "linux-64")
        for name in _COAL:
            with self.subTest(name):
                self.assertIn(name, win)
                self.assertIn(name, lin)
                self.assertEqual(lin[name][0], win[name][0])

    def test_coal_is_the_same_build_number(self) -> None:
        """A build number is a revision of the recipe; the solver had picked Linux's rebuild 1 while Windows runs
        build 0, and conda-forge has build 0 for Linux too."""
        win, lin = _conda("coal_env", "win-64"), _conda("coal_env", "linux-64")
        for name in ("coal", "coal-python", "libcoal"):
            with self.subTest(name):
                self.assertEqual(lin[name][1].rsplit("_", 1)[-1], win[name][1].rsplit("_", 1)[-1])


class ThePipHalfIsTheSameTests(unittest.TestCase):
    def test_every_package_both_platforms_install_is_at_one_version(self) -> None:
        win, lin = _pip("win-64"), _pip("linux-64")
        for name in sorted(win.keys() & lin.keys()):
            if _PIP_PLATFORM.match(name):
                continue
            with self.subTest(name):
                self.assertEqual(lin[name], win[name])

    def test_what_one_platform_installs_alone_is_what_torch_or_the_console_decides(self) -> None:
        win, lin = _pip("win-64"), _pip("linux-64")
        alone = sorted(win.keys() ^ lin.keys())
        self.assertTrue(alone)
        for name in alone:
            with self.subTest(name):
                self.assertRegex(name, _PIP_PLATFORM)

    def test_the_cuda_runtime_wheels_differ_only_where_torch_pins_them_on_linux(self) -> None:
        """Measured: torch 2.7.0+cu128 on Linux requires the 12.8 wheels, so the Windows pip lock's 12.9 ones cannot
        resolve there. Every ``nvidia-*-cu12`` wheel on Linux is a 12.8 build or a library with its own numbering."""
        lin = _pip("linux-64")
        self.assertEqual(lin["nvidia-cuda-nvrtc-cu12"], "12.8.61")
        self.assertEqual(lin["nvidia-cuda-runtime-cu12"], "12.8.57")
        self.assertEqual(lin["torch"], _pip("win-64")["torch"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
