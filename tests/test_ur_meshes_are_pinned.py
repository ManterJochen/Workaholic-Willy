"""Every UR arm the stack knows has its collision meshes pinned, and the fetch writes all of them or nothing.

A cuRobo descriptor is built from Isaac's UR description, whose mesh references resolve under the cuRobo asset root.
On 2026-09-15 that root held meshes for ur5e and ur10e only, so ur3, ur3e and ur5 refused to build and a fresh
`install.ps1` stopped at its ur3e descriptor. The meshes come from Universal Robots' ROS 2 description at one commit,
the source cuRobo's own `ur_description` LICENSE names, and every file is held to a committed sha256.

`fetch_ur_meshes.py` runs in the cuRobo environment as the build does, so it is loaded by path, and it touches the
network only when no local source is handed in. Nothing here does.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest import mock

from src.robot.safety._ur_kinematics import UR_DH_TABLES_M

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts" / "curobo"
_LINKS = ("base", "shoulder", "upperarm", "forearm", "wrist1", "wrist2", "wrist3")


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class TheManifestPinsEveryArmTests(unittest.TestCase):

    def setUp(self) -> None:
        self.fetch = _load("fetch_ur_meshes")
        self.pinned = self.fetch.read_manifest(self.fetch.MANIFEST)

    def test_every_arm_with_a_dh_table_has_its_seven_collision_meshes_pinned(self) -> None:
        missing = []
        for model in UR_DH_TABLES_M:
            for link in _LINKS:
                owner = self.fetch.mesh_owner(model, link)
                path = f"meshes/{owner}/collision/{link}.stl"
                if path not in self.pinned:
                    missing.append(f"{model}: {path}")
        self.assertEqual(missing, [])

    def test_the_manifest_names_one_commit_and_its_licence(self) -> None:
        header = [line for line in self.fetch.MANIFEST.read_text(encoding="utf-8").splitlines() if line.startswith("#")]
        text = "\n".join(header)
        self.assertIn(self.fetch.COMMIT, text)
        self.assertIn("BSD-3-Clause", text)
        self.assertRegex(self.fetch.COMMIT, r"^[0-9a-f]{40}$")

    def test_every_pinned_hash_is_a_sha256(self) -> None:
        self.assertTrue(self.pinned)
        bad = {path: digest for path, digest in self.pinned.items() if not re.fullmatch(r"[0-9a-f]{64}", digest)}
        self.assertEqual(bad, {})


class TheFetchWritesAllOrNothingTests(unittest.TestCase):

    def setUp(self) -> None:
        self.fetch = _load("fetch_ur_meshes")
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.source = self.tmp / "source"
        self.dest = self.tmp / "dest"
        self.files = {"meshes/ur3/collision/base.stl": b"base", "meshes/ur3/collision/wrist1.stl": b"wrist1"}
        for relative, data in self.files.items():
            path = self.source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        self.manifest = self.tmp / "manifest.sha256"
        self.manifest.write_text(
            "# a test manifest\n" + "".join(f"{_sha(data)}  {relative}\n" for relative, data in self.files.items()),
            encoding="utf-8")

    def test_a_verified_source_is_installed_and_then_checks(self) -> None:
        written = self.fetch.install(source=self.source, dest=self.dest, manifest=self.manifest)

        self.assertEqual(sorted(written), sorted(self.files))
        for relative, data in self.files.items():
            self.assertEqual((self.dest / relative).read_bytes(), data)
        self.assertEqual(self.fetch.check(self.dest, manifest=self.manifest), [])

    def test_a_tampered_source_file_writes_nothing(self) -> None:
        (self.source / "meshes/ur3/collision/wrist1.stl").write_bytes(b"not wrist1")

        with self.assertRaises(self.fetch.MeshPinError) as raised:
            self.fetch.install(source=self.source, dest=self.dest, manifest=self.manifest)

        self.assertIn("meshes/ur3/collision/wrist1.stl", str(raised.exception))
        self.assertFalse((self.dest / "meshes/ur3/collision/base.stl").exists())

    def test_a_different_installed_file_is_refused_and_left_as_it_is(self) -> None:
        installed = self.dest / "meshes/ur3/collision/base.stl"
        installed.parent.mkdir(parents=True)
        installed.write_bytes(b"someone else's base")

        with self.assertRaises(self.fetch.MeshPinError) as raised:
            self.fetch.install(source=self.source, dest=self.dest, manifest=self.manifest)

        self.assertIn("meshes/ur3/collision/base.stl", str(raised.exception))
        self.assertEqual(installed.read_bytes(), b"someone else's base")
        self.assertFalse((self.dest / "meshes/ur3/collision/wrist1.stl").exists())

    def test_check_names_a_missing_mesh_and_exits_1(self) -> None:
        problems = self.fetch.check(self.dest, manifest=self.manifest)

        self.assertTrue(any("meshes/ur3/collision/base.stl" in problem for problem in problems), problems)
        self.assertEqual(
            self.fetch.main(["--check", "--dest", str(self.dest), "--manifest", str(self.manifest)]), 1)


class AFetchWritesTheUpstreamBytesTests(unittest.TestCase):
    """A checkout's line endings are not a different mesh.

    Measured on 2026-09-15: this box sets `core.autocrlf=true` system wide, so a checkout wrote UR's `wrist3.dae` with
    100 carriage returns, 66,177 bytes against the blob's 66,077, and a manifest hashed from that checkout pinned bytes
    no Linux machine would ever fetch. The fetch has to write what the commit holds, whatever the machine's git says.
    """

    def setUp(self) -> None:
        if shutil.which("git") is None:
            self.skipTest("git is not on PATH")
        self.fetch = _load("fetch_ur_meshes")
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _git(self, cwd: Path, *args: str) -> str:
        import subprocess

        return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()

    def test_a_text_mesh_is_written_as_the_commit_holds_it_under_autocrlf(self) -> None:
        upstream = self.tmp / "upstream"
        upstream.mkdir()
        relative = "meshes/ur3/visual/base.dae"
        content = b"<COLLADA>\n  <mesh/>\n</COLLADA>\n"
        (upstream / relative).parent.mkdir(parents=True)
        (upstream / relative).write_bytes(content)
        self._git(upstream, "init", "-q")
        self._git(upstream, "-c", "core.autocrlf=false", "add", relative)
        self._git(upstream, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "mesh")
        commit = self._git(upstream, "rev-parse", "HEAD")
        global_config = self.tmp / "gitconfig"
        global_config.write_text("[core]\n\tautocrlf = true\n", encoding="utf-8")

        workdir = self.tmp / "work"
        workdir.mkdir()
        with mock.patch.object(self.fetch, "REPO_URL", upstream.as_uri()), \
                mock.patch.object(self.fetch, "COMMIT", commit), \
                mock.patch.dict(os.environ, {"GIT_CONFIG_GLOBAL": str(global_config)}):
            source = self.fetch.fetch_source(workdir, [relative])

        self.assertEqual((source / relative).read_bytes(), content)


class TheInstallBuildsEveryArmTests(unittest.TestCase):
    """The install builds a descriptor for every arm, with every hand it can place without a bench measurement."""

    def setUp(self) -> None:
        self.script = (_REPO / "scripts" / "ext_deps" / "install.ps1").read_text(encoding="utf-8")

    def _list(self, name: str) -> list[str]:
        match = re.search(rf"\${name}\s*=\s*@\(([^)]*)\)", self.script)
        self.assertIsNotNone(match, f"install.ps1 declares no ${name} list")
        assert match is not None
        return re.findall(r"'([^']+)'", match.group(1))

    def test_every_arm_that_can_carry_a_judged_retract_is_built(self) -> None:
        """A descriptor carries the retract the rule judged on the exact meshes, so an arm with no committed bundle
        has no row and no descriptor (UM5, B4). Since B6 the list holds every arm the DH tables know: all seven
        have a bundle baked from UR's own STLs and a sphere map fitted to it."""
        from src.robot.safety.planning.environment import collision_mesh_bundle

        # An arm is built when this repository can build it CORRECTLY: its own collision bundle, and a sphere
        # map fitted to that bundle. Until B6 the second half was the VENDOR's map, and two arms were left out
        # because theirs describes a different arm (ur16e by 60.8 mm, ur10 by 75.9). The fit replaced it, so
        # the condition is now the committed fit and both arms are in.
        maps = _REPO / "src" / "robot" / "safety" / "planning" / "robot"

        buildable = sorted(arm for arm in UR_DH_TABLES_M
                           if collision_mesh_bundle(arm).is_file()
                           and (maps / f"{arm}_arm_spheres.yml").is_file())
        self.assertEqual(sorted(self._list("UR_ARMS")), buildable)
        for arm in sorted(arm for arm in UR_DH_TABLES_M
                          if not collision_mesh_bundle(arm).is_file()
                          or not (maps / f"{arm}_arm_spheres.yml").is_file()):
            with self.subTest(arm=arm):
                self.assertNotIn(f"'{arm}'", self.script)
                self.assertIn(arm, self.script, "an arm left out of the build is named with a reason")

    def test_an_arm_with_no_bundle_would_be_left_out(self) -> None:
        """⭐ THE CONTROL, rebuilt because it rotted by growth.

        It read "every arm has a bundle, so this assertion can no longer fail" and said to
        delete it rather than weaken it. The rule is the same either way, so it is proven against a mesh
        directory with nothing in it instead of against whichever arm happens to be unbaked.
        """
        import tempfile
        from pathlib import Path
        from unittest import mock

        from src.robot.safety.planning.environment import collision_mesh_bundle

        with tempfile.TemporaryDirectory() as empty, mock.patch(
            "src.robot.safety.planning.environment.COLLISION_MESH_DIR", Path(empty)
        ):
            buildable = sorted(arm for arm in UR_DH_TABLES_M if collision_mesh_bundle(arm).is_file())
        self.assertEqual(buildable, [], "a mesh directory with no bundles still answered with arms")
        self.assertNotEqual(sorted(self._list("UR_ARMS")), buildable)

    def test_the_install_builds_every_arm_and_names_no_hand(self) -> None:
        """One descriptor per arm (UM lane S11): the sidecar adds the hand a cell names when it starts, so the install
        builds no hand, and a hand flag is refused by the builder."""
        for name in ("$FLANGE_HANDS", "--gripper", "--coupling-mm"):
            with self.subTest(absent=name):
                self.assertNotIn(name, self.script)
        # The model, and since B5 S10 the description source, and nothing else: no hand, no plate. Since B6
        # every arm takes UR's own description, so the flag is a literal rather than a per-model variable.
        self.assertRegex(self.script, r"build_ur_config\.py'\)\s+\$model\s+'--urdf-from'\s+'ur'\s*\|")

    def test_the_meshes_are_fetched_before_the_first_descriptor(self) -> None:
        fetch = self.script.find("fetch_ur_meshes.py")
        build = self.script.find("build_ur_config.py")
        self.assertGreater(fetch, -1, "install.ps1 never fetches the UR meshes")
        self.assertLess(fetch, build)


if __name__ == "__main__":
    unittest.main()
