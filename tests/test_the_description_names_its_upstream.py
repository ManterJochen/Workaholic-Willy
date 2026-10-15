"""Every arm description names the pin its derived half was read from, as data (P0f F6b, owner decisions 9 and D6).

Until F6 the pin lived in prose: a `source` sentence and a comment on line 9 of each file, both spelling the commit by
hand, twelve copies of one 40-character string that nothing compared. `upstream` states it as five fields a test can
hold: the build adapter that reads it, the repository, the full commit, the licence as one SPDX identifier, and a hash
over the adapter manifest's rows. The adapter (`scripts/arms/fetch.py` and the manifest) stays the authority for the
bytes, and every description is checked against it here, so a pin bump that forgets one arm fails by name.

The rows hash is over the manifest's `<digest>  <path>` rows sorted by path, header excluded: the header's prose
changed in the very commit that moved the manifest (F6a), and a CRLF checkout would change the file's hash too, while
neither changes a pinned byte.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from src.config.arms import available_arms, load_arm
from src.config.schema.arms import ArmSpec

_ROOT = Path(__file__).resolve().parents[1]
_ARMS = _ROOT / "scripts" / "arms"


def _fetch():
    spec = importlib.util.spec_from_file_location("_fetch_for_the_upstream_tests", _ARMS / "fetch.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows_sha256(manifest: Path) -> str:
    """The recipe, restated here rather than taken from the code it checks."""
    rows = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        digest, _, path = line.partition("  ")
        rows.append((path.strip(), digest.strip()))
    return hashlib.sha256("".join(f"{digest}  {path}\n" for path, digest in sorted(rows)).encode("utf-8")).hexdigest()


def _header(manifest: Path, key: str) -> str:
    (line,) = [line for line in manifest.read_text(encoding="utf-8").splitlines() if line.startswith(f"# {key}:")]
    return line


def _adapters() -> tuple[str, ...]:
    """``scripts/arms/_urdf_source.SOURCES``, read by AST: the module imports its sibling by bare name."""
    tree = ast.parse((_ARMS / "_urdf_source.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "SOURCES" for t in node.targets):
            return tuple(ast.literal_eval(node.value))
    raise AssertionError("_urdf_source.py assigns no SOURCES")


def _differences(model: str, upstream: object, pin: dict[str, str]) -> list[str]:
    return [f"{model}.upstream.{field} {getattr(upstream, field)!r}, the pin {value!r}"
            for field, value in pin.items() if getattr(upstream, field) != value]


class EveryDescriptionNamesThePinTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fetch = _fetch()
        cls.pin = {
            "adapter": "ur",
            "repo": cls.fetch.REPO_URL,
            "commit": cls.fetch.COMMIT,
            "licence": "BSD-3-Clause",
            "manifest_rows_sha256": _rows_sha256(cls.fetch.DESCRIPTION_MANIFEST),
        }

    def test_every_arm_names_the_pin_the_adapter_keeps(self) -> None:
        differing = [line for model in available_arms()
                     for line in _differences(model, load_arm(model).upstream, self.pin)]
        self.assertEqual([], differing)
        self.assertEqual(8, len(available_arms()), "the eight UR arms this repository describes")

    def test_the_commit_inside_every_rendered_urdf_is_the_pin(self) -> None:
        """`_commit()` is what `urdf_sha256` hashes, through the URDF header."""
        spec = importlib.util.spec_from_file_location("_ur_description_for_the_upstream_tests",
                                                      _ARMS / "_ur_description.py")
        assert spec is not None and spec.loader is not None
        ur = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = ur
        spec.loader.exec_module(ur)
        self.assertEqual(self.fetch.COMMIT, ur._commit())

    def test_the_manifest_header_names_the_same_repository_and_licence(self) -> None:
        manifest = self.fetch.DESCRIPTION_MANIFEST
        self.assertIn(f"# Source: {self.fetch.REPO_URL.removesuffix('.git')} at commit {self.fetch.COMMIT}",
                      _header(manifest, "Source"))
        self.assertIn("BSD-3-Clause", _header(manifest, "Licence"))
        licence = (self.fetch.DESCRIPTION / "LICENSE").read_text(encoding="utf-8")
        self.assertIn("Neither the name of the copyright holder", licence, "the third clause, which makes it BSD-3")

    def test_every_adapter_named_is_one_the_build_has(self) -> None:
        self.assertLessEqual({load_arm(model).upstream.adapter for model in available_arms()}, set(_adapters()))

    def test_a_bumped_commit_is_named_by_arm_and_field(self) -> None:
        """THE CONTROL: the comparison above sees one arm left behind by a pin bump."""
        arm = load_arm("ur10")
        stale = arm.upstream.model_copy(update={"commit": "0" * 40})
        self.assertEqual([f"ur10.upstream.commit {'0' * 40!r}, the pin {self.fetch.COMMIT!r}"],
                         _differences("ur10", stale, self.pin))

    def test_the_rows_hash_sees_a_digest_and_not_a_header(self) -> None:
        """THE CONTROL: one flipped digest moves the hash; a rewritten header and CRLF line endings do not."""
        text = self.fetch.DESCRIPTION_MANIFEST.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as tmp:
            header_edit = Path(tmp) / "header.sha256"
            header_edit.write_text(text.replace("# Licence:", "# Licence (reworded):"), encoding="utf-8")
            crlf = Path(tmp) / "crlf.sha256"
            crlf.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
            first_row = next(line for line in text.splitlines() if line and not line.startswith("#"))
            flipped = Path(tmp) / "flipped.sha256"
            flipped.write_text(text.replace(first_row, ("0" if first_row[0] != "0" else "1") + first_row[1:]),
                               encoding="utf-8")
            self.assertEqual(self.pin["manifest_rows_sha256"], _rows_sha256(header_edit))
            self.assertEqual(self.pin["manifest_rows_sha256"], _rows_sha256(crlf))
            self.assertNotEqual(self.pin["manifest_rows_sha256"], _rows_sha256(flipped))


class TheSchemaRefusesAPinItCannotHoldTests(unittest.TestCase):
    def _refusal(self, **upstream: object) -> str:
        data = load_arm("ur5e").model_dump(mode="json")
        data["upstream"].update(upstream)
        with self.assertRaises(ValidationError) as caught:
            ArmSpec.model_validate(data)
        return str(caught.exception)

    def test_a_description_without_upstream_is_refused(self) -> None:
        data = load_arm("ur5e").model_dump(mode="json")
        del data["upstream"]
        with self.assertRaises(ValidationError) as caught:
            ArmSpec.model_validate(data)
        self.assertIn("upstream", str(caught.exception))

    def test_a_commit_that_can_move_is_refused(self) -> None:
        for commit in ("89bbe79", "89BBE795F38A7AB00FB66FE8831DFFF79DC99EDF", "main", "v2.4.0"):
            with self.subTest(commit=commit):
                self.assertIn("commit", self._refusal(commit=commit))

    def test_a_licence_written_as_prose_is_refused(self) -> None:
        self.assertIn("licence", self._refusal(licence="BSD 3-Clause, Universal Robots"))

    def test_a_repository_not_fetched_over_https_is_refused(self) -> None:
        self.assertIn("repo", self._refusal(repo="git@github.com:UniversalRobots/Universal_Robots_ROS2_Description"))

    def test_the_ur_adapter_reads_only_a_ur(self) -> None:
        data = load_arm("ur5e").model_dump(mode="json")
        data["vendor"] = "franka"
        with self.assertRaises(ValidationError) as caught:
            ArmSpec.model_validate(data)
        self.assertIn("adapter 'ur'", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
