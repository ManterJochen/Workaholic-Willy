"""The live view, and the four ways it says "no picture" without saying "error".

WHAT THIS FILE IS DEFENDING. A console that shows an image labelled "camera" is making a claim, and
there are THREE different pictures it could be showing:

* a colour frame from a physical device, taken now;
* a synthetic scene this process DREW -- the rehearsal cell has no camera at all;
* the grasp overlay -- segmentation, projected gripper, chosen contacts -- rendered during a pick and
  overwritten once per segmentation, so it can be minutes old.

Rendering any of them as "the camera" is how a stale segmentation, or a picture of a room that does
not exist, ends up on a screen an operator reads as live. So every assertion here is about the payload
SAYING WHICH, and about the no-picture states being ordinary answers rather than faults: a cell with
no camera is a legitimate cell, and a 404 for it would make an everyday state look broken.

The synthetic case is the one that took a correction. The first version of this endpoint answered
`source: "camera"` for a rehearsal cell and said "Live from the cell's camera" underneath it -- on the
one profile that runs with no hardware attached, which is the only profile anyone can run today. The
kind is now DECLARED by the source (`colour_source_kind`) rather than inferred, and an undeclared
source is described as unknown, never as a camera.

The second property, and the reason this endpoint refuses rather than tries: **the pick owns the
camera.** ``backend/src/camera/`` holds no lock (measured: ``grep -ri thread`` there is empty) and a
pick runs on its own daemon thread, so two ``grab()`` calls on one ``rs.pipeline`` would split the
frame stream between the viewer and the robot. While a run is active the viewfinder does not touch the
device at all.

Honesty bucket ②: a real console, a real rehearsal cell, real HTTP. No camera has ever been attached
to this path -- ``RehearsalPerceptionSource`` stands in, and it is deliberately obvious about it.
"""

from __future__ import annotations

import base64
import re
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover - the console is an optional extra
    TestClient = None  # type: ignore[assignment,misc]

from api.viewfinder import ViewfinderFrame, read_viewfinder
from src.config.loader import active_profile, reload_config, set_active_profile
from src.robot.execution.autonomous_grasp.rehearsal import RehearsalPerceptionSource
from src.robot.perception.viewfinder import colour_source_kind, peek_color_of

_SHIPPED = Path(__file__).resolve().parents[1] / "backend" / "config" / "data"

#: A 1x1 PNG, so the overlay path can be exercised without rendering one.
_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000a49444154789c6300010000050001"
    "0d0a2db40000000049454e44ae426082"
)


# ---------------------------------------------------------------------------------------------
# stand-ins: the smallest thing shaped like a built service


class _Calculator:
    def __init__(self) -> None:
        self.render_debug_images = False
        self.last_debug_image_png: bytes | None = None


class _Service:
    """Mirrors the two accessors ``AutonomousGraspService`` exposes to a viewer, and nothing else."""

    def __init__(self, perception: object) -> None:
        self.calculator = _Calculator()
        orchestrator = type("_O", (), {})()
        orchestrator.perception = perception
        orchestrator.calculator = self.calculator
        self.runtime = type("_R", (), {})()
        self.runtime.orchestrator = orchestrator

    @property
    def debug_image_rendering_enabled(self) -> bool:
        return bool(self.calculator.render_debug_images)

    @property
    def last_debug_image_png(self) -> bytes | None:
        return self.calculator.last_debug_image_png


class _Console:
    def __init__(self, service: object | None) -> None:
        self.session = type("_S", (), {})()
        self.session.service = service
        self.active_run_id: str | None = None


class _Blind:
    """A perception source with no ``peek_color`` -- what an Isaac source deliberately is."""

    def acquire(self) -> None:  # pragma: no cover - never called
        raise AssertionError("the viewfinder must never call acquire()")


class _Undeclared:
    """Can peek, does not say what of. Must never be promoted to "camera"."""

    def peek_color(self):  # noqa: ANN201
        return np.full((4, 6, 3), 128, dtype=np.uint8)


# ---------------------------------------------------------------------------------------------


class ThePeekContractTests(unittest.TestCase):
    """``peek_color`` is optional BY DESIGN, and the absence must degrade the view, not the console."""

    def test_a_source_without_peek_color_yields_none_rather_than_raising(self) -> None:
        self.assertIsNone(peek_color_of(_Blind()))

    def test_a_source_whose_peek_raises_is_treated_as_having_no_picture(self) -> None:
        """A camera unplugged mid-session must cost the picture, never the pick."""

        class _Broken:
            def peek_color(self):  # noqa: ANN202
                raise OSError("device disconnected")

        self.assertIsNone(peek_color_of(_Broken()))

    def test_a_wrong_shaped_return_is_refused_rather_than_forwarded(self) -> None:
        class _Greyscale:
            def peek_color(self):  # noqa: ANN202
                return np.zeros((10, 10), dtype=np.uint8)

        self.assertIsNone(peek_color_of(_Greyscale()))

    def test_a_source_that_does_not_declare_its_kind_is_UNKNOWN_never_camera(self) -> None:
        """The default has to be the one that cannot mislead."""
        self.assertEqual(colour_source_kind(_Undeclared()), "unknown")
        self.assertEqual(colour_source_kind(object()), "unknown")

    def test_an_out_of_vocabulary_kind_falls_back_to_unknown(self) -> None:
        class _Liar:
            colour_source_kind = "definitely_a_camera"

            def peek_color(self):  # noqa: ANN202
                return None

        self.assertEqual(colour_source_kind(_Liar()), "unknown")

    def test_the_two_shipped_sources_declare_themselves(self) -> None:
        from src.robot.perception import RealSenseVisionPerceptionSource

        self.assertEqual(RealSenseVisionPerceptionSource.colour_source_kind, "camera")
        self.assertEqual(colour_source_kind(RehearsalPerceptionSource()), "synthetic")

    def test_the_rehearsal_source_peeks_without_advancing_its_own_frame_counter(self) -> None:
        """The contract's whole point: peeking must not change what the next acquire() returns."""
        source = RehearsalPerceptionSource()
        before = source.frames_served
        image = peek_color_of(source)
        self.assertIsNotNone(image)
        assert image is not None
        self.assertEqual(image.shape, (480, 640, 3))
        self.assertEqual(image.dtype, np.uint8)
        self.assertEqual(source.frames_served, before, "peeking served a frame it should not have")


class TheFourNoPictureAnswersTests(unittest.TestCase):
    """Each of these is an ANSWER. None of them is an error, and each names itself."""

    def test_an_unbuilt_cell_says_so(self) -> None:
        frame = read_viewfinder(_Console(None))
        self.assertEqual((frame.source, frame.reason), ("none", "not_built"))
        self.assertIsNone(frame.image)
        self.assertTrue(frame.human)

    def test_a_source_that_cannot_be_peeked_at_says_WHY_not_just_that(self) -> None:
        frame = read_viewfinder(_Console(_Service(_Blind())))
        self.assertEqual((frame.source, frame.reason), ("none", "no_colour_source"))
        # The sentence has to carry the reason a simulated cell shows nothing, or the honest refusal
        # reads to an operator as a bug in the console.
        self.assertIn("step the simulator", frame.human)

    def test_a_running_pick_owns_the_camera_and_the_viewfinder_stands_down(self) -> None:
        """The safety property. The perception source here CAN peek -- and must not be asked to."""
        console = _Console(_Service(RehearsalPerceptionSource()))
        console.active_run_id = "run-7"
        frame = read_viewfinder(console)
        self.assertEqual((frame.source, frame.reason), ("none", "pick_owns_camera"))
        self.assertIsNone(frame.image)

    def test_while_a_pick_runs_the_message_depends_on_whether_the_overlay_is_ON(self) -> None:
        """"Nothing rendered yet" and "rendering is off" are different things to do about it."""
        console = _Console(_Service(RehearsalPerceptionSource()))
        console.active_run_id = "run-7"
        self.assertIn("Switch the grasp overlay on", read_viewfinder(console).human)

        console.session.service.calculator.render_debug_images = True
        second = read_viewfinder(console)
        self.assertIn("nothing has been rendered yet", second.human)
        self.assertTrue(second.overlay_enabled)


class TheTwoPicturesAreNeverBlurredTests(unittest.TestCase):
    def test_an_idle_REHEARSAL_cell_is_SYNTHETIC_and_the_sentence_denies_a_camera(self) -> None:
        """The correction, pinned. This is the only profile that runs with no hardware attached."""
        frame = read_viewfinder(_Console(_Service(RehearsalPerceptionSource())), jpeg_quality=60)
        self.assertEqual(frame.source, "synthetic")
        self.assertEqual(frame.media_type, "image/jpeg")
        self.assertEqual((frame.width, frame.height), (640, 480))
        self.assertEqual(frame.age_s, 0.0, "a frame taken now is not allowed to claim an age")
        assert frame.image is not None
        self.assertEqual(frame.image[:2], b"\xff\xd8", "not a JPEG")
        # Not merely missing the word "camera" -- actively contradicting it.
        self.assertIn("not a camera", frame.human)

    def test_a_source_declaring_CAMERA_gets_the_live_wording_and_only_it(self) -> None:
        class _Device:
            colour_source_kind = "camera"

            def peek_color(self):  # noqa: ANN202
                return np.zeros((12, 16, 3), dtype=np.uint8)

        frame = read_viewfinder(_Console(_Service(_Device())))
        self.assertEqual(frame.source, "camera")
        self.assertIn("Live from the cell's camera", frame.human)

    def test_an_undeclared_source_gets_a_picture_but_NOT_the_word_camera(self) -> None:
        frame = read_viewfinder(_Console(_Service(_Undeclared())))
        self.assertEqual(frame.source, "synthetic", "anything not a declared camera is not a camera")
        self.assertIsNotNone(frame.image)
        self.assertIn("will not call it a camera", frame.human)

    def test_during_a_pick_the_overlay_is_served_and_is_NOT_called_a_camera_frame(self) -> None:
        console = _Console(_Service(RehearsalPerceptionSource()))
        console.session.service.calculator.last_debug_image_png = _PNG
        console.active_run_id = "run-7"

        frame = read_viewfinder(console)
        self.assertEqual(frame.source, "overlay")
        self.assertEqual(frame.media_type, "image/png")
        self.assertEqual(frame.image, _PNG)
        # The sentence must contradict "live", not merely omit it.
        self.assertIn("not what the camera shows now", frame.human)

    def test_the_overlay_age_is_a_LOWER_BOUND_until_this_process_watched_it_change(self) -> None:
        """The correction to my own criticism of ``/v1/overlay``.

        That socket seeds its clock at connect time and reports ``age_s: 0.0`` for an image rendered
        minutes earlier. The first version of THIS module did the same on its first read, while its
        comment accused the socket of exactly that. The fix is not a comment -- it is a distinction the
        payload now carries, because nothing anywhere stamps a render time:

          * the first overlay a process sees may predate it -> the age is a FLOOR, ``age_is_exact`` False
          * one it watched replace another                  -> the age is real, ``age_is_exact`` True
        """
        import api.viewfinder as module

        module._OVERLAY_SEEN = ("", 0.0, False)          # a fresh process
        console = _Console(_Service(RehearsalPerceptionSource()))
        console.session.service.calculator.last_debug_image_png = _PNG
        console.active_run_id = "run-7"

        first = read_viewfinder(console)
        self.assertEqual(first.age_s, 0.0)
        self.assertFalse(first.age_is_exact, "an unseen overlay cannot be dated, only bounded")
        self.assertIn("AT LEAST this old", first.human)

        # Now this process WATCHES the image change, so the next age is real to within a poll.
        console.session.service.calculator.last_debug_image_png = _PNG + bytes([0])
        second = read_viewfinder(console)
        self.assertEqual(second.age_s, 0.0)
        self.assertTrue(second.age_is_exact)
        self.assertNotIn("AT LEAST this old", second.human)

    def test_the_jpeg_quality_argument_actually_reaches_the_encoder(self) -> None:
        """Pinned because the config key it comes from had no reader at all before this endpoint."""
        console = _Console(_Service(RehearsalPerceptionSource()))
        low = read_viewfinder(console, jpeg_quality=5)
        high = read_viewfinder(console, jpeg_quality=95)
        assert low.image is not None and high.image is not None
        self.assertLess(len(low.image), len(high.image))


@unittest.skipIf(TestClient is None, "requirements/api.txt is not installed (optional extra)")
class TheEndpointTests(unittest.TestCase):
    """Over real HTTP, against a real rehearsal cell built through the real build path."""

    def setUp(self) -> None:
        from api.app import create_app
        from api.cell import Console, set_console

        self.tmp = Path(tempfile.mkdtemp()) / "data"
        shutil.copytree(_SHIPPED, self.tmp)
        robot = self.tmp / "robot" / "robot.yaml"
        text = robot.read_text(encoding="utf-8")
        text, arm = re.subn(
            r'^(\s*)vendor:\s*"ur"$', r'\g<1>vendor: "dummy"', text, count=1, flags=re.MULTILINE
        )
        text, grip = re.subn(
            r'^(\s*)vendor:\s*"robotiq"$', r'\g<1>vendor: "none"', text, count=1, flags=re.MULTILINE
        )
        assert arm == 1 and grip == 1, "the dummy substitution found nothing"
        robot.write_text(text, encoding="utf-8")

        self._previous_profile = active_profile()
        self.cell = Console(root=self.tmp, profile=None)
        self._previous_console = set_console(self.cell)
        self.client = TestClient(create_app())
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        from api.cell import set_console

        try:
            self.cell.session.disconnect()
        except Exception:  # pragma: no cover
            pass
        set_console(self._previous_console)
        set_active_profile(self._previous_profile)
        reload_config()
        shutil.rmtree(self.tmp.parent, ignore_errors=True)

    def test_before_a_build_it_answers_200_with_no_picture_not_404(self) -> None:
        response = self.client.get("/v1/camera")
        self.assertEqual(response.status_code, 200, "a cameraless cell is a state, not a fault")
        body = response.json()
        self.assertEqual(body["source"], "none")
        self.assertEqual(body["reason"], "not_built")
        self.assertIsNone(body["image_base64"])

    def test_after_a_rehearsal_build_it_serves_a_decodable_jpeg(self) -> None:
        self.assertEqual(
            self.client.post("/v1/cell/build", params={"rehearse": True}).status_code, 200
        )
        body = self.client.get("/v1/camera").json()
        # A rehearsal build has no camera. Over real HTTP, through the real build path, the wire says so.
        self.assertEqual(body["source"], "synthetic")
        self.assertEqual(body["media_type"], "image/jpeg")
        self.assertEqual((body["width"], body["height"]), (640, 480))

        raw = base64.b64decode(body["image_base64"])
        self.assertEqual(raw[:2], b"\xff\xd8")
        # Round-trip it: a payload the browser cannot decode is not a picture.
        import cv2

        decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded.shape, (480, 640, 3))

    def test_the_overlay_switch_is_reported_back_so_a_ui_never_has_to_guess(self) -> None:
        self.client.post("/v1/cell/build", params={"rehearse": True})
        self.assertFalse(self.client.get("/v1/camera").json()["overlay_enabled"])

        self.client.post("/v1/overlay/enable", params={"enabled": True})
        self.assertTrue(self.client.get("/v1/camera").json()["overlay_enabled"])

    def test_the_wire_shape_matches_the_dataclass_field_for_field(self) -> None:
        """Guards the one drift that would be invisible: a renamed field silently dropped in mapping."""
        self.client.post("/v1/cell/build", params={"rehearse": True})
        body = self.client.get("/v1/camera").json()
        expected = {f for f in ViewfinderFrame.__dataclass_fields__} - {"image"} | {"image_base64"}
        self.assertEqual(set(body), expected)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
