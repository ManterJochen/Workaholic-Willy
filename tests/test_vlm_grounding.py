"""The VLM route's testable half: response parsing and the on_unavailable contract.

No weights, no GPU, no network. Every rejection asserted here is a box that would otherwise have become
a grasp pose, so the tests are written from that angle rather than as JSON-parsing trivia.
"""

from __future__ import annotations

import logging
import unittest

from src.models.perception_backend import PerceivedObject
from src.models.vlm import (
    VLM_NOMINAL_SCORE,
    GuardedVlmBackend,
    VlmUnavailableError,
    extract_json_payload,
    parse_grounding_response,
)
from src.models.vlm.parsing import CoordinateSpace

W, H = 640, 480


def parse(text: str, label: str = "a red cube", space=CoordinateSpace.ABSOLUTE):
    """Structural cases use ABSOLUTE so the numbers in the assertions are the numbers in the JSON.

    The default space is GRID_1000 (what Qwen3-VL actually emits) -- that is covered on its own in
    :class:`CoordinateSpaceTests`, with the raw strings the model really produced.
    """
    return parse_grounding_response(
        text, image_width=W, image_height=H, fallback_label=label, space=space,
    )


class CoordinateSpaceTests(unittest.TestCase):
    """The most dangerous failure in the route: grid values that LOOK like plausible pixels.

    Every string here was emitted by ``Qwen3-VL-4B-Instruct`` on this box (2026-08-10). Treating them
    as pixels produces a well-formed, in-range, sanely-sized box in the wrong place -- nothing throws,
    nothing is logged, the gripper simply goes somewhere else. So the space is declared, not inferred,
    and these tests pin the declaration.
    """

    #: Measured: a 100x100 square at (100,100) in a 640x480 frame came back as this.
    MEASURED_SYNTHETIC = '[{"bbox_2d": [144, 198, 312, 418], "label": "red square"}]'
    #: Measured on a 1280x720 frame -- note 999/1000, which CANNOT be pixels of a 720px-tall image.
    MEASURED_REAL = ('[{"bbox_2d": [238, 288, 680, 714], "label": "bin"},'
                     ' {"bbox_2d": [474, 500, 999, 999], "label": "bin"}]')

    def test_grid_values_are_rescaled_to_pixels(self) -> None:
        [det] = parse_grounding_response(
            self.MEASURED_SYNTHETIC, image_width=640, image_height=480,
            fallback_label="red square", space=CoordinateSpace.GRID_1000,
        )
        x0, y0, x1, y1 = det.box
        # ground truth was (100, 100, 200, 200); the model is accurate to a few pixels
        self.assertAlmostEqual(x0, 92.2, delta=2.0)
        self.assertAlmostEqual(y0, 95.0, delta=2.0)
        self.assertAlmostEqual(x1, 199.7, delta=2.0)
        self.assertAlmostEqual(y1, 200.6, delta=2.0)

    def test_the_same_answer_read_as_pixels_is_silently_wrong(self) -> None:
        """Documents the trap: no error, no warning -- just a box in the wrong place."""
        [det] = parse_grounding_response(
            self.MEASURED_SYNTHETIC, image_width=640, image_height=480,
            fallback_label="red square", space=CoordinateSpace.ABSOLUTE,
        )
        self.assertEqual(det.box, [144.0, 198.0, 312.0, 418.0])   # well-formed...
        self.assertGreater(det.box[0], 140.0)                      # ...and ~50 px off in x
        self.assertGreater(det.box[3], 400.0)                      # ...and ~200 px off in y

    def test_grid_values_beyond_the_image_height_still_land_inside_it(self) -> None:
        """999/1000 on a 720px-tall frame is the observation that settled the coordinate space."""
        dets = parse_grounding_response(
            self.MEASURED_REAL, image_width=1280, image_height=720,
            fallback_label="bin", space=CoordinateSpace.GRID_1000,
        )
        self.assertEqual(len(dets), 2)
        for det in dets:
            self.assertLessEqual(det.box[2], 1280.0)
            self.assertLessEqual(det.box[3], 720.0)

    def test_the_default_space_is_the_measured_one(self) -> None:
        """A caller who says nothing gets Qwen's actual convention, not a hopeful assumption."""
        default = parse_grounding_response(
            self.MEASURED_SYNTHETIC, image_width=640, image_height=480, fallback_label="x",
        )
        explicit = parse_grounding_response(
            self.MEASURED_SYNTHETIC, image_width=640, image_height=480, fallback_label="x",
            space=CoordinateSpace.GRID_1000,
        )
        self.assertEqual(default, explicit)

    def test_scaling_happens_before_clamping(self) -> None:
        """Clamp-then-scale would pin every right-hand box against the frame edge and flatten it."""
        [det] = parse_grounding_response(
            '[{"bbox_2d": [900, 900, 990, 990]}]', image_width=1280, image_height=720,
            fallback_label="x", space=CoordinateSpace.GRID_1000,
        )
        self.assertAlmostEqual(det.box[0], 1152.0, delta=1.0)
        self.assertAlmostEqual(det.box[2], 1267.2, delta=1.0)
        self.assertGreater(det.box[2] - det.box[0], 100.0, "the box must not be flattened")


class PayloadExtractionTests(unittest.TestCase):
    """Instruct models wrap JSON in whatever they feel like. All of it has to come back."""

    def test_bare_json(self) -> None:
        self.assertEqual(extract_json_payload('[{"a": 1}]'), [{"a": 1}])

    def test_markdown_fence(self) -> None:
        self.assertEqual(extract_json_payload('```json\n[{"a": 1}]\n```'), [{"a": 1}])

    def test_unlabelled_fence(self) -> None:
        self.assertEqual(extract_json_payload('```\n[{"a": 1}]\n```'), [{"a": 1}])

    def test_json_embedded_in_prose(self) -> None:
        text = 'Sure! Here is what I found: [{"a": 1}] Let me know if you need more.'
        self.assertEqual(extract_json_payload(text), [{"a": 1}])

    def test_unparseable_returns_none_rather_than_raising(self) -> None:
        for text in ("", "   ", "I could not find anything.", "{not json", "```json\n{oops\n```"):
            with self.subTest(text=text):
                self.assertIsNone(extract_json_payload(text))


class BoxParsingTests(unittest.TestCase):
    def test_the_documented_format(self) -> None:
        [det] = parse('[{"bbox_2d": [10, 20, 110, 220], "label": "red cube"}]')
        self.assertEqual(det.box, [10.0, 20.0, 110.0, 220.0])
        self.assertEqual(det.label, "red cube")
        self.assertEqual(det.x_center, 60.0)
        self.assertEqual(det.y_center, 120.0)
        self.assertEqual(det.score, VLM_NOMINAL_SCORE)

    def test_multiple_boxes_keep_the_models_order(self) -> None:
        """Output order is the only ranking a VLM gives us -- it must not be reshuffled."""
        dets = parse('[{"bbox_2d":[0,0,50,50],"label":"a"},'
                     ' {"bbox_2d":[60,60,90,90],"label":"b"},'
                     ' {"bbox_2d":[100,100,150,150],"label":"c"}]')
        self.assertEqual([d.label for d in dets], ["a", "b", "c"])

    def test_alternate_key_names(self) -> None:
        for key in ("bbox_2d", "bbox", "box_2d", "box"):
            with self.subTest(key=key):
                self.assertEqual(len(parse(f'[{{"{key}": [10, 20, 110, 220]}}]')), 1)

    def test_wrapped_in_an_object(self) -> None:
        self.assertEqual(len(parse('{"objects": [{"bbox_2d": [10, 20, 110, 220]}]}')), 1)

    def test_a_lone_object_is_one_detection(self) -> None:
        self.assertEqual(len(parse('{"bbox_2d": [10, 20, 110, 220], "label": "cube"}')), 1)

    def test_missing_label_falls_back_to_the_prompt(self) -> None:
        """Detection rejects an empty label, so a nameless box needs SOMETHING meaningful."""
        [det] = parse('[{"bbox_2d": [10, 20, 110, 220]}]', label="a red cube")
        self.assertEqual(det.label, "a red cube")

    def test_empty_array_is_the_honest_absent_answer(self) -> None:
        self.assertEqual(parse("[]"), [])

    def test_inverted_corners_are_repaired(self) -> None:
        """x1<x0 has exactly one possible intent -- unlike anything else here, repair is not a guess."""
        [det] = parse('[{"bbox_2d": [110, 220, 10, 20]}]')
        self.assertEqual(det.box, [10.0, 20.0, 110.0, 220.0])

    def test_out_of_frame_boxes_are_clamped(self) -> None:
        [det] = parse('[{"bbox_2d": [-50, -50, 700, 900]}]')
        self.assertEqual(det.box, [0.0, 0.0, float(W), float(H)])

    def test_a_box_entirely_off_frame_is_dropped(self) -> None:
        """Clamping collapses it; a hallucinated location must not survive as an edge sliver."""
        self.assertEqual(parse('[{"bbox_2d": [700, 900, 800, 1000]}]'), [])

    def test_values_outside_the_declared_space_are_dropped_not_reinterpreted(self) -> None:
        """Under ABSOLUTE, fractions are not silently re-read as a normalised box.

        Deciding per-box that a value "must have meant" something else is the same guess that the
        explicit CoordinateSpace exists to remove -- made once per detection instead of once per
        backend, where nobody can review it.
        """
        self.assertEqual(parse('[{"bbox_2d": [0.1, 0.2, 0.5, 0.6]}]'), [])

    def test_degenerate_and_malformed_boxes_are_dropped(self) -> None:
        for payload in (
            '[{"bbox_2d": [10, 20, 10, 220]}]',        # zero width
            '[{"bbox_2d": [10, 20, 110]}]',            # three values
            '[{"bbox_2d": [10, 20, 110, "x"]}]',       # non-numeric
            '[{"bbox_2d": null}]',                     # null
            '[{"label": "cube"}]',                     # no box at all
            '[{"bbox_2d": [10, 20, 110, 1e999]}]',     # non-finite
            '["not a dict"]',                          # wrong element type
        ):
            with self.subTest(payload=payload):
                self.assertEqual(parse(payload), [])

    def test_good_boxes_survive_alongside_bad_ones(self) -> None:
        """One malformed entry must not discard a cluttered scene's other objects."""
        dets = parse('[{"bbox_2d": [10, 20, 110, 220], "label": "good"},'
                     ' {"bbox_2d": [1, 2]},'
                     ' {"bbox_2d": [200, 200, 300, 300], "label": "also good"}]')
        self.assertEqual([d.label for d in dets], ["good", "also good"])

    def test_prose_only_answer_yields_nothing(self) -> None:
        self.assertEqual(parse("I'm sorry, I cannot see a red cube in this image."), [])


class _FakeBackend:
    def __init__(self, objects=(), error: BaseException | None = None) -> None:
        self.objects, self.error, self.calls = objects, error, 0

    def perceive(self, image_bgr, prompt):  # noqa: ANN001
        self.calls += 1
        if self.error is not None:
            raise self.error
        return tuple(self.objects)


class OnUnavailableTests(unittest.TestCase):
    """refuse | degrade -- the owner's stated contract, including that degrade must be loud."""

    def test_working_vlm_is_passed_straight_through(self) -> None:
        sentinel = (PerceivedObject(detection=None, segmentation=None),)  # type: ignore[arg-type]
        guard = GuardedVlmBackend(vlm=_FakeBackend(sentinel), model_id="qwen")
        self.assertEqual(guard.perceive(None, "the broken part"), sentinel)
        self.assertFalse(guard.degraded)

    def test_refuse_raises_rather_than_returning_empty(self) -> None:
        """'no objects found' and 'I could not look' are different answers."""
        guard = GuardedVlmBackend(vlm=_FakeBackend(error=OSError("no weights")), model_id="qwen")
        with self.assertRaises(VlmUnavailableError) as caught:
            guard.perceive(None, "the broken part")
        self.assertIn("qwen", str(caught.exception))
        self.assertIn("on_unavailable=degrade", str(caught.exception))

    def test_degrade_falls_back_and_warns_every_single_time(self) -> None:
        """A one-shot warning scrolls away and the run then looks normal for an hour."""
        fallback = _FakeBackend(())
        guard = GuardedVlmBackend(
            vlm=_FakeBackend(error=OSError("no weights")), model_id="qwen",
            degrade=True, fallback_factory=lambda: fallback,
        )
        with self.assertLogs("backend.src.models.vlm.availability", level=logging.WARNING) as first:
            guard.perceive(None, "the broken part")
        with self.assertLogs("backend.src.models.vlm.availability", level=logging.WARNING) as second:
            guard.perceive(None, "the broken part")
        self.assertTrue(guard.degraded)
        self.assertEqual(fallback.calls, 2)
        for captured in (first, second):
            self.assertTrue(any("DEGRADED" in line and "WRONG object" in line
                                for line in captured.output))

    def test_a_failed_load_is_not_retried_on_every_pick(self) -> None:
        vlm = _FakeBackend(error=OSError("no weights"))
        guard = GuardedVlmBackend(vlm=vlm, model_id="qwen", degrade=True,
                                  fallback_factory=lambda: _FakeBackend(()))
        for _ in range(3):
            guard.perceive(None, "the broken part")
        self.assertEqual(vlm.calls, 1, "a multi-second load must not be re-attempted per pick")

    def test_the_fallback_is_never_built_while_the_vlm_works(self) -> None:
        """The reason it is a factory: eager construction loads GroundingDINO AND SAM2 on a healthy
        cell, spending seconds and VRAM on a path that never runs."""
        built = []

        def factory():
            built.append(1)
            return _FakeBackend(())

        guard = GuardedVlmBackend(vlm=_FakeBackend(()), model_id="qwen",
                                  degrade=True, fallback_factory=factory)
        guard.perceive(None, "the broken part")
        self.assertEqual(built, [], "the fallback must not be constructed unless it is used")

    def test_the_fallback_is_built_once_across_many_degraded_picks(self) -> None:
        built = []

        def factory():
            built.append(1)
            return _FakeBackend(())

        guard = GuardedVlmBackend(vlm=_FakeBackend(error=OSError("no weights")), model_id="qwen",
                                  degrade=True, fallback_factory=factory)
        for _ in range(3):
            guard.perceive(None, "the broken part")
        self.assertEqual(len(built), 1)

    def test_degrade_without_a_fallback_is_refused_at_construction(self) -> None:
        with self.assertRaises(ValueError):
            GuardedVlmBackend(vlm=_FakeBackend(()), model_id="qwen", degrade=True)

    def test_unexpected_errors_are_not_swallowed_into_a_fallback(self) -> None:
        """Only missing-dependency / missing-weights / out-of-VRAM shapes degrade. A bug must surface."""
        guard = GuardedVlmBackend(
            vlm=_FakeBackend(error=ValueError("a real bug")), model_id="qwen",
            degrade=True, fallback_factory=lambda: _FakeBackend(()),
        )
        with self.assertRaises(ValueError):
            guard.perceive(None, "the broken part")


class ImportHygieneTests(unittest.TestCase):
    """The VLM package must import on CI, on macOS, and on a box with no GPU."""

    def test_importing_the_package_does_not_pull_in_torch_or_transformers(self) -> None:
        """Measured in a FRESH interpreter, because this one has already imported torch elsewhere.

        Checking ``sys.modules`` in-process would pass no matter what this package does -- other tests
        populate it. A subprocess is the only honest way to ask the question.
        """
        import subprocess
        import sys

        probe = (
            "import sys;"
            "import src.models.vlm as v;"
            "import src.models.vlm.qwen;"
            "heavy=[m for m in ('torch','transformers') if m in sys.modules];"
            "print(','.join(heavy) or 'clean')"
        )
        done = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, check=False, timeout=180,
        )
        self.assertEqual(done.returncode, 0, done.stderr[-600:])
        self.assertEqual(
            done.stdout.strip(), "clean",
            "importing the vlm package loaded heavy dependencies -- they must stay inside _ensure_loaded",
        )


if __name__ == "__main__":
    unittest.main()
