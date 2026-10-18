"""One request carries the whole scene, and every client writes its field to a file of its own.

Two defects, both measured in the code rather than on the box. The world and the field went to the
sidecar as two requests, and the second one rebuilt the scene from the boxes it remembered, so a
declared mesh disappeared whenever a field was sent. And the field was written to a file named after
the process alone, so two arms in one process wrote the same file and each could register the other's
scene. Neither needs a GPU to see: the first is what the client sends, the second is a file name.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from typing import Any

from src.robot.safety.planning.curobo_client import CuroboPlanClient, SceneRegistration

_BENCH = {"name": "support_plane", "dims_m": [1.2, 1.0, 0.05], "pose": [0.4, 0.0, -0.025, 1, 0, 0, 0]}
_TOTE = {"name": "tote", "file_path": "assets/tote.obj", "pose": [0.5, 0.0, 0.0, 1, 0, 0, 0]}
_FIELD = {
    "path": "scene.npy",
    "dims_m": [0.81, 0.81, 0.69],
    "voxel_size_m": 0.03,
    "pose": [0.4, 0.0, 0.345, 1.0, 0.0, 0.0, 0.0],
}


class _Wire:
    """Stands in for the pipe: records what the client sends and answers with one reply."""

    def __init__(self, reply: "dict[str, Any] | None") -> None:
        self.reply = reply
        self.sent: list[dict[str, Any]] = []

    def send(self, request: dict[str, Any]) -> int:
        self.sent.append(request)
        return 7

    def recv(self, timeout_s: float, *, want: "int | None" = None) -> "dict[str, Any] | None":
        return None if self.reply is None else {**self.reply, "id": want}


class SetSceneTests(unittest.TestCase):
    def _client(self, wire: _Wire) -> CuroboPlanClient:
        client = CuroboPlanClient(python_path="no-python-is-spawned")
        client._proc = object()  # type: ignore[assignment]  # noqa: SLF001 - nothing is spawned
        client._send = wire.send  # type: ignore[method-assign]  # noqa: SLF001
        client._recv = wire.recv  # type: ignore[method-assign]  # noqa: SLF001
        self.addCleanup(setattr, client, "_proc", None)
        return client

    def test_the_boxes_the_meshes_and_the_field_go_in_one_request(self) -> None:
        wire = _Wire({"world_set": 2, "voxels_set": 16767})

        registration = self._client(wire).set_scene([_BENCH], [_TOTE], dict(_FIELD))

        self.assertEqual(len(wire.sent), 1, "a second request is the moment the meshes were lost")
        (request,) = wire.sent
        self.assertEqual(request["cmd"], "set_world")
        self.assertEqual(request["cuboids"], [_BENCH])
        self.assertEqual(request["meshes"], [_TOTE])
        self.assertEqual(request["voxels"], _FIELD)
        self.assertEqual(registration, SceneRegistration(world_set=2, voxels_set=16767, reason=""))

    def test_the_sidecar_reason_is_kept_word_for_word(self) -> None:
        why = "this planner was started with no voxel storage; set WILLY_CUROBO_VOXEL_GRID before it starts"
        wire = _Wire({"world_set": 1, "voxels_set": None, "reason": why})

        registration = self._client(wire).set_scene([_BENCH], [], dict(_FIELD))

        self.assertIsNone(registration.voxels_set)
        self.assertEqual(registration.reason, why)

    def test_no_reply_registers_nothing_and_says_so(self) -> None:
        registration = self._client(_Wire(None)).set_scene([_BENCH], [_TOTE], dict(_FIELD))

        self.assertEqual(registration.world_set, 0)
        self.assertIsNone(registration.voxels_set)
        self.assertIn("no reply", registration.reason)


class LiveSceneFileTests(unittest.TestCase):
    def test_two_clients_write_two_scene_files(self) -> None:
        first, second = CuroboPlanClient(), CuroboPlanClient()

        self.assertNotEqual(first.live_scene_path, second.live_scene_path)
        for path in (first.live_scene_path, second.live_scene_path):
            with self.subTest(path=path):
                self.assertEqual(Path(path).parent, Path(tempfile.gettempdir()))
                self.assertIn(str(os.getpid()), Path(path).name)
                self.assertTrue(path.endswith(".npy"))


if __name__ == "__main__":
    unittest.main()
