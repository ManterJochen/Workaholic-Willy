"""A rebake that produces the committed arrays leaves the committed file as it is (P0f F3).

The retract table records the sha256 of every bundle file it was judged on (``retract.yaml``, 54 rows), so a file
that is rewritten with the very same arrays still invalidates the table if its bytes move. Measured 2026-09-23 in aurora
(numpy 2.4.4): that day's baker writes the committed bytes back for 7 of 7 arms, because its key order is the
committed one and every zip member is dated 1980. That holds by coincidence of order and zlib: F7 to F9 rewrite the
baker's link tables from the arm description, and P0h bakes on Linux. So ``--write`` now writes only when an array
changed, and says so when none did.

No pinned meshes are needed: the bake itself is stood in for with the committed arrays, which is exactly the case
under test, and the write is watched rather than performed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_BAKER = _ROOT / "scripts" / "arms" / "bake.py"


def _baker():
    name = "bake_for_f3"
    spec = importlib.util.spec_from_file_location(name, _BAKER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, str(_BAKER.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(_BAKER.parent))
    return module


class TheBakerWritesOnlyWhatChangedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.baker = _baker()
        self.committed = self.baker.DATA / "ur5e_collision_meshes.npz"
        with np.load(self.committed, allow_pickle=False) as held:
            self.arrays = {key: np.array(held[key]) for key in held.files}
        hand = self.baker.GRIPPER_KEYS
        #: What bake() returns: the six links; main() adds the 2F-85 arrays before it writes.
        self.links = {key: value for key, value in self.arrays.items() if key.split("__")[0] not in hand}
        self.sha = hashlib.sha256(self.committed.read_bytes()).hexdigest()

    def _main(self, links: dict, *args: str) -> tuple[int, str, mock.Mock]:
        writer = mock.Mock()
        said = io.StringIO()
        with mock.patch.object(self.baker, "bake", return_value={k: np.array(v) for k, v in links.items()}), \
                mock.patch.object(self.baker.np, "savez_compressed", writer), redirect_stdout(said):
            code = self.baker.main(["ur5e", "--write", *args])
        return code, said.getvalue(), writer

    def test_the_committed_arrays_are_not_written_again(self) -> None:
        code, said, writer = self._main(self.links)
        self.assertEqual(0, code)
        writer.assert_not_called()
        self.assertIn("left as it is", said)
        self.assertEqual(self.sha, hashlib.sha256(self.committed.read_bytes()).hexdigest())

    def test_a_changed_array_is_still_written(self) -> None:
        """The control: a baker that never writes would pass the test above."""
        moved = {key: np.array(value) for key, value in self.links.items()}
        moved["forearm__v"][0, 0] += 1.0
        code, _, writer = self._main(moved, "--expect-change=the control of the F3 test")
        self.assertEqual(0, code)
        writer.assert_called_once()
        self.assertEqual(self.committed, writer.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
