"""``PATCH /v1/config`` answers every refusal the config writer can give with its own code and the catalog's status.

The route maps ``WriteRefused`` to an HTTP status; a member with no entry turned a correct refusal into a KeyError and a
500, and a guard that becomes a server error the first time it fires is not a guard. The pose door added three
members (``no_layer``, ``invalid_name``, ``invalid_label``) that the generic writer does not return today, because a
pose key is refused ``not_writable`` first; this pins that the map stays total anyway, and that each status is the one
the console's catalog gives the code (``api.codes.REFUSAL_STATUS``), so the two never say different things.

A pose key is refused by its own door: typed joints would skip the hand and the screen, so the route answers 403
``not_writable`` with the writer's sentence, and no byte of the tree changes (owner decision Q10; the library side is
``tests/test_a_pose_is_written_only_through_its_own_door.py``).
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover - the console is an optional extra
    TestClient = None  # type: ignore[assignment,misc]

from src.config.loader import active_profile, reload_config, set_active_profile

_SHIPPED = Path(__file__).resolve().parents[1] / "config"


@unittest.skipIf(TestClient is None, "fastapi is unavailable; requirements.txt pins fastapi and httpx")
class EveryRefusalOfTheWriterIsAnsweredTests(unittest.TestCase):
    def setUp(self) -> None:
        from api.app import create_app
        from api.cell import Console, set_console

        self.tmp = Path(tempfile.mkdtemp()) / "data"
        shutil.copytree(_SHIPPED, self.tmp)
        self._previous_profile = active_profile()
        self._previous_console = set_console(Console(root=self.tmp, profile=None))
        self.client = TestClient(create_app())
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        from api.cell import set_console

        set_console(self._previous_console)
        set_active_profile(self._previous_profile)
        reload_config()
        shutil.rmtree(self.tmp.parent, ignore_errors=True)

    def test_each_refusal_comes_back_with_its_code_and_the_catalogs_status(self) -> None:
        """Red before for no_layer, invalid_name and invalid_label: the route raised KeyError, a 500."""
        from api.codes import REFUSAL_STATUS, RefusalCode
        from src.config.edit import WriteRefused, WriteResult

        for refused in WriteRefused:
            result = WriteResult(applied=False, keys=("robot.safety.payload.mass_kg",), refused=refused,
                                 refused_key="robot.safety.payload.mass_kg", message=f"refused: {refused.value}")
            with self.subTest(refused=refused.value), \
                    mock.patch("src.config.tree.ConfigTree.write", return_value=result):
                response = self.client.patch("/v1/config", json={"robot.safety.payload.mass_kg": 1.15})
                self.assertEqual(REFUSAL_STATUS[RefusalCode(refused.value)], response.status_code, response.text)
                body = response.json()
                self.assertEqual(refused.value, body["code"])
                self.assertEqual(f"refused: {refused.value}", body["message"])


def _tree_hash(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file()}


#: A chain whose last layer is the cell's own (``robot.cell.yaml``), and a pose a person might type.
_CHAIN = "ur10,hande,cell"
_DROP = [-60.0, -95.0, -120.0, -55.0, 90.0, 0.0]


@unittest.skipIf(TestClient is None, "fastapi is unavailable; requirements.txt pins fastapi and httpx")
class AConsolePatchOfAPoseIsRefusedTests(unittest.TestCase):
    """``PATCH /v1/config`` is the generic writer's door in the console: it answers 403 for a pose key, and the tree it
    writes to keeps every byte, the cell's own layer included (moved from the library's pose test, which imports no web
    framework)."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp()) / "data"
        shutil.copytree(_SHIPPED, self.tmp)
        cell_layer = self.tmp / "robot" / "robot.cell.yaml"
        cell_layer.write_text("# The cell's own layer: what this cell measured and taught. Not in git.\n", encoding="utf-8")
        self._previous_profile = active_profile()
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        set_active_profile(self._previous_profile)
        reload_config()
        shutil.rmtree(self.tmp.parent, ignore_errors=True)

    def test_the_patch_is_refused_with_403_and_the_file_is_untouched(self) -> None:
        from api.app import create_app
        from api.cell import Console, set_console

        previous = set_console(Console(root=self.tmp, profile=_CHAIN))
        self.addCleanup(set_console, previous)
        before = _tree_hash(self.tmp)

        response = TestClient(create_app()).patch("/v1/config", json={"robot.named_poses.drop_left.joints_deg": _DROP})

        self.assertEqual(403, response.status_code, response.text)
        body = response.json()
        self.assertEqual("not_writable", body["code"])
        self.assertIn("taught by hand", body["message"])
        self.assertEqual(before, _tree_hash(self.tmp))

if __name__ == "__main__":  # pragma: no cover
    unittest.main()
