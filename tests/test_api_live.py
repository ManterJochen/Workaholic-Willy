"""The live image (build plan 1.7, OD 11): a display frame of each camera the cell's picks see through, never a
measurement.

``GET /v1/camera/live`` reads through ``Camera.peek`` of the service's camera owners (``service_cameras``), which never
waits and never grabs for a measurement: while a measuring grab holds a rig it answers nothing, and the answer says
``measuring`` so the browser keeps its last frame. What is held here:

* a frame is JPEG, downscaled to ``max_width`` with its aspect kept, at the runtime's frame quality, and carries when it
  was taken and how old it is, with every rig the cell sees through (mounting, which is primary);
* the primary rig is shown unless ``rig`` names another; a rig the cell does not hold is ``no_rig``;
* a peek that answers nothing is ``measuring``, and keeps the last frame's time; a camera that cannot be read is
  ``no_camera``; a frame that cannot be encoded is ``encode_failed``; no cell is ``not_built``; none of them is an error;
* the rehearsal cell has no camera: its source's own picture is ``synthetic``, never ``camera``;
* the route writes at the cell's own frame quality and bounds the width (32 to 4096 pixels);
* it reads during a run too: a peek takes nothing from a pick;
* each rig's last frame keeps its age, and ``refresh_stale`` peeks only a rig whose frame is older than the bound, so a
  rig nobody shows on the stage never reads as stale to the ready bar.

``GET /v1/camera`` and its tests (``tests/test_api_viewfinder.py``) stay as they are.

Honesty bucket (2): camera owners are doubles with ``Camera.peek``'s contract; the rehearsal cell is real.
"""

from __future__ import annotations

import base64
import re
import time
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover - the console is an optional extra
    TestClient = None  # type: ignore[assignment,misc]

from src.camera.orchestration.camera import CameraNotOpen, PeekFrame
from tests.test_api_jaws import ScratchCell


class _Owner:
    """A camera owner as ``service_cameras`` finds it: a rig id, the rig's config, and ``peek`` that never waits."""

    def __init__(self, rig_id: str, mounting: str | None, *, colour: Any = None, shape: tuple[int, int] = (480, 640),
                 taken: float | None = None) -> None:
        self.rig_id = rig_id
        self.rig = SimpleNamespace(rig_id=rig_id, body=None,
                                   extrinsics=None if mounting is None else SimpleNamespace(mounting_mode=mounting))
        self.colour = colour if colour is not None else np.full((*shape, 3), 90, dtype=np.uint8)
        self.taken = taken
        self.peeks = 0
        self.answer: str = "frame"   # "frame" | "measuring" | "closed"

    def peek(self) -> PeekFrame | None:
        self.peeks += 1
        if self.answer == "measuring":
            return None
        if self.answer == "closed":
            raise CameraNotOpen(f"rig {self.rig_id!r} is not open")
        return PeekFrame(color=self.colour, captured_at_s=time.time() if self.taken is None else self.taken)

    def grab(self) -> Any:  # pragma: no cover - a look must never measure
        raise AssertionError("the live image grabbed a measuring frame")


def _service(*owners: _Owner) -> Any:
    """A built service whose primary perception holds the first owner and whose planner world opened the rest."""
    first, *rest = owners
    orchestrator = SimpleNamespace(perception=SimpleNamespace(streamer=SimpleNamespace(camera=first)),
                                   planner_world_cameras=SimpleNamespace(cameras=list(rest)))
    return SimpleNamespace(runtime=SimpleNamespace(orchestrator=orchestrator))


def _jpeg(body: dict[str, Any]) -> bytes:
    data = base64.b64decode(body["image_base64"])
    assert data[:2] == b"\xff\xd8", "not a JPEG"
    return data


class TheFramesTests(unittest.TestCase):
    """``api.live.LiveFrames`` on its own: the frame, its age, and what it says where there is none."""

    def setUp(self) -> None:
        from api.live import LiveFrames

        self.live = LiveFrames()
        self.wrist = _Owner("wrist", "eye_in_hand")
        self.overhead = _Owner("overhead", "eye_to_hand", shape=(400, 300))
        self.service = _service(self.wrist, self.overhead)

    def test_the_primary_rig_is_shown_downscaled_with_its_aspect_and_its_time(self) -> None:
        taken = time.time() - 0.4
        self.wrist.taken = taken
        frame = self.live.frame(self.service, rig=None, max_width=320)
        body = frame.model_dump()
        self.assertEqual(("wrist", "camera", ""), (frame.rig_id, frame.source, frame.reason))
        self.assertEqual((320, 240), (frame.width, frame.height))
        self.assertEqual(taken, frame.captured_at)
        self.assertIsNotNone(frame.age_s)
        assert frame.age_s is not None
        self.assertAlmostEqual(0.4, frame.age_s, delta=0.3)
        self.assertGreater(len(_jpeg(body)), 100)
        self.assertEqual([("wrist", "wrist", True), ("overhead", "fixed", False)],
                         [(r.rig_id, r.mounting, r.primary) for r in frame.rigs])
        self.assertEqual(1, self.wrist.peeks)
        self.assertEqual(0, self.overhead.peeks, "a rig nobody asked for was read")

    def test_a_frame_narrower_than_the_bound_keeps_its_size(self) -> None:
        frame = self.live.frame(self.service, rig="overhead", max_width=960)
        self.assertEqual(("overhead", 300, 400), (frame.rig_id, frame.width, frame.height))

    def test_a_rig_the_cell_does_not_hold_is_no_rig(self) -> None:
        frame = self.live.frame(self.service, rig="ceiling", max_width=960)
        self.assertEqual(("none", "no_rig", None), (frame.source, frame.reason, frame.image_base64))
        self.assertEqual(["wrist", "overhead"], [r.rig_id for r in frame.rigs])

    def test_a_peek_that_answers_nothing_is_measuring_and_keeps_the_last_frame_s_time(self) -> None:
        first = self.live.frame(self.service, rig=None, max_width=960)
        self.wrist.answer = "measuring"
        frame = self.live.frame(self.service, rig=None, max_width=960)
        self.assertEqual(("camera", "measuring", None), (frame.source, frame.reason, frame.image_base64))
        self.assertEqual(first.captured_at, frame.captured_at)

    def test_a_camera_that_cannot_be_read_is_no_camera_never_an_error(self) -> None:
        self.wrist.answer = "closed"
        frame = self.live.frame(self.service, rig=None, max_width=960)
        self.assertEqual(("none", "no_camera"), (frame.source, frame.reason))

    def test_a_frame_that_cannot_be_encoded_is_said(self) -> None:
        with patch("cv2.imencode", return_value=(False, None)):
            frame = self.live.frame(self.service, rig=None, max_width=960)
        self.assertEqual(("none", "encode_failed", None), (frame.source, frame.reason, frame.image_base64))

    def test_no_cell_is_not_built_and_a_cell_with_no_camera_is_no_camera(self) -> None:
        self.assertEqual("not_built", self.live.frame(None, rig=None, max_width=960).reason)
        blind = SimpleNamespace(runtime=SimpleNamespace(orchestrator=SimpleNamespace(perception=object())))
        self.assertEqual(("none", "no_camera"), (self.live.frame(blind, rig=None, max_width=960).source,
                                                 self.live.frame(blind, rig=None, max_width=960).reason))

    def test_the_jpeg_is_written_at_the_quality_it_is_handed(self) -> None:
        import cv2

        seen: list[Any] = []
        real = cv2.imencode

        def recording(ext: str, image: Any, params: Any = None) -> Any:
            seen.append(list(params or ()))
            return real(ext, image, params)

        self.live.quality = 37
        with patch("cv2.imencode", side_effect=recording):
            self.live.frame(self.service, rig=None, max_width=960)
        self.assertEqual([[int(cv2.IMWRITE_JPEG_QUALITY), 37]], seen)

    def test_each_rig_keeps_its_age_and_only_a_stale_one_is_peeked_again(self) -> None:
        self.assertEqual({}, self.live.ages())
        self.wrist.taken = time.time() - 5.0
        self.live.frame(self.service, rig=None, max_width=960)
        self.assertEqual({"wrist"}, set(self.live.ages()))
        self.assertGreaterEqual(self.live.ages()["wrist"], 5.0)
        self.wrist.taken = None
        peeks = (self.wrist.peeks, self.overhead.peeks)

        ages = self.live.refresh_stale(self.service, max_age_s=2.0)

        self.assertEqual((peeks[0] + 1, peeks[1] + 1), (self.wrist.peeks, self.overhead.peeks),
                         "a stale rig and a rig with no frame were not peeked")
        self.assertLess(ages["wrist"], 2.0)
        self.assertIn("overhead", ages)
        again = self.live.refresh_stale(self.service, max_age_s=2.0)
        self.assertEqual((peeks[0] + 1, peeks[1] + 1), (self.wrist.peeks, self.overhead.peeks),
                         "a fresh rig was peeked again")
        self.assertEqual(set(ages), set(again))

    def test_a_rig_measuring_during_a_refresh_keeps_no_new_age(self) -> None:
        self.wrist.answer = "measuring"
        self.assertNotIn("wrist", self.live.refresh_stale(self.service, max_age_s=2.0))


@unittest.skipIf(TestClient is None, "fastapi is unavailable; requirements.txt pins fastapi and httpx")
class TheLiveRouteTests(ScratchCell):
    def test_no_cell_is_an_answer_not_an_error(self) -> None:
        answered = self.client.get("/v1/camera/live")
        self.assertEqual(200, answered.status_code, answered.text)
        self.assertEqual(("none", "not_built", None), (answered.json()["source"], answered.json()["reason"],
                                                       answered.json()["image_base64"]))

    def test_the_rehearsal_cell_shows_its_own_picture_as_synthetic_never_as_a_camera(self) -> None:
        self.build()
        answered = self.client.get("/v1/camera/live", params={"max_width": 320})
        self.assertEqual(200, answered.status_code, answered.text)
        body = answered.json()
        self.assertEqual(("synthetic", "", []), (body["source"], body["reason"], body["rigs"]))
        self.assertEqual((320, 240), (body["width"], body["height"]))
        _jpeg(body)

    def test_the_route_reads_the_cell_s_owners_during_a_run_too(self) -> None:
        wrist, overhead = _Owner("wrist", "eye_in_hand"), _Owner("overhead", "eye_to_hand")
        self.cell.session.service = _service(wrist, overhead)
        self.cell.active_run_id = "run-picking"
        answered = self.client.get("/v1/camera/live", params={"rig": "overhead"})
        self.assertEqual(200, answered.status_code, answered.text)
        self.assertEqual(("overhead", "camera"), (answered.json()["rig_id"], answered.json()["source"]))
        self.assertEqual(1, overhead.peeks)
        self.assertIn("overhead", self.cell.live.ages(), "the console's own frames did not keep the rig's age")

    def test_the_width_is_bounded(self) -> None:
        self.assertEqual(422, self.client.get("/v1/camera/live", params={"max_width": 0}).status_code)
        self.assertEqual(422, self.client.get("/v1/camera/live", params={"max_width": 31}).status_code)
        self.assertEqual(422, self.client.get("/v1/camera/live", params={"max_width": 4097}).status_code)

    def test_the_route_writes_at_the_runtime_s_frame_quality(self) -> None:
        """``runtime.image_encoding.frame_quality`` of the cell's own tree, as ``GET /v1/camera`` writes its frames."""
        import cv2

        runtime = self.tmp / "app" / "runtime.yaml"
        text, hits = re.subn(r"^(\s*)frame_quality:\s*\d+", r"\g<1>frame_quality: 37",
                             runtime.read_text(encoding="utf-8"), count=1, flags=re.MULTILINE)
        self.assertEqual(1, hits, "the frame quality line was not found")
        runtime.write_text(text, encoding="utf-8")
        self.cell.session.service = _service(_Owner("wrist", "eye_in_hand"))
        seen: list[Any] = []
        real = cv2.imencode

        def recording(ext: str, image: Any, params: Any = None) -> Any:
            seen.append(list(params or ()))
            return real(ext, image, params)

        with patch("cv2.imencode", side_effect=recording):
            answered = self.client.get("/v1/camera/live")

        self.assertEqual(200, answered.status_code, answered.text)
        self.assertEqual("camera", answered.json()["source"])
        self.assertEqual([[int(cv2.IMWRITE_JPEG_QUALITY), 37]], seen)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
