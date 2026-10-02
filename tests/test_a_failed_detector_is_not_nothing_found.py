"""A detector that raised has not found nothing: every such perceive is counted.

`TwoStageBackend` turns a model exception into "no objects", so a pick does not die with a traceback while the
cell holds something. That rule stays. What changes is that it is no longer silent: each perceive that returned
nothing BECAUSE a model raised (the detector, or the segmenter on every detection) adds one to `failures` and
names itself in `last_failure`. A caller that must tell "the scene is empty" from "the model could not look" reads
the count before and after: the console's task ends `detector_failed` instead of `nothing_left`.

The guarded VLM backend and the routed backend pass the counts through, so the number is the same whichever
stack a cell built. A missing VLM is the case that matters most on the owner's cell: its load error is raised
inside the detector, and until now it read as an empty table on every look.
"""

from __future__ import annotations

import unittest
from unittest import mock

from src.models.perception_backend import TwoStageBackend, failures_of, last_failure_of
from src.models.routed_backend import RoutedPerceptionBackend
from src.models.vlm import GuardedVlmBackend, Qwen3VLGrounder, VlmUnavailableError, shared_vlm


class _Detection:
    def __init__(self, label: str) -> None:
        self.label = label
        self.box = [0.0, 0.0, 10.0, 10.0]


class _Detector:
    def __init__(self, labels: tuple[str, ...] = (), *, raises: BaseException | None = None) -> None:
        self.labels = labels
        self.raises = raises

    def detect_all(self, image: object, prompt: str) -> list[_Detection]:
        if self.raises is not None:
            raise self.raises
        return [_Detection(label) for label in self.labels]


class _Segmenter:
    def __init__(self, *, fail_on: frozenset[str] = frozenset()) -> None:
        self.fail_on = fail_on

    def segment_detection(self, image: object, detection: _Detection) -> str:
        if detection.label in self.fail_on:
            raise RuntimeError(f"no mask for {detection.label}")
        return f"mask:{detection.label}"


def _two_stage(labels: tuple[str, ...] = (), *, raises: BaseException | None = None,
               fail_on: frozenset[str] = frozenset()) -> TwoStageBackend:
    return TwoStageBackend(_Detector(labels, raises=raises), _Segmenter(fail_on=fail_on))


class TheTwoStageBackendCountsItsFailuresTests(unittest.TestCase):
    def test_a_detector_exception_is_a_counted_failure(self) -> None:
        backend = _two_stage(raises=RuntimeError("the detector fell over"))
        self.assertEqual(backend.perceive(object(), "a cube"), (), "still nothing rather than a traceback")
        self.assertEqual(backend.failures, 1)
        self.assertIn("RuntimeError", backend.last_failure)
        self.assertIn("the detector fell over", backend.last_failure)

    def test_an_honest_empty_answer_is_no_failure(self) -> None:
        backend = _two_stage(())
        self.assertEqual(backend.perceive(object(), "a cube"), ())
        self.assertEqual(backend.failures, 0)
        self.assertEqual(backend.last_failure, "")

    def test_a_segmenter_that_fails_on_every_detection_is_a_failure(self) -> None:
        """The detector saw two parts; nothing came back because the masks all raised, not because the bin is empty."""
        backend = _two_stage(("cube", "mug"), fail_on=frozenset({"cube", "mug"}))
        self.assertEqual(backend.perceive(object(), "everything"), ())
        self.assertEqual(backend.failures, 1)
        self.assertIn("segmenter", backend.last_failure)
        self.assertIn("no mask for cube", backend.last_failure)

    def test_one_failed_mask_among_good_ones_is_no_failure(self) -> None:
        backend = _two_stage(("cube", "mug"), fail_on=frozenset({"mug"}))
        self.assertEqual(len(backend.perceive(object(), "everything")), 1)
        self.assertEqual(backend.failures, 0)

    def test_the_count_only_grows_and_cannot_be_reset_by_assignment(self) -> None:
        backend = _two_stage(raises=OSError("weights gone"))
        seen = []
        for _ in range(3):
            backend.perceive(object(), "a cube")
            seen.append(backend.failures)
        self.assertEqual(seen, [1, 2, 3])
        with self.assertRaises(AttributeError):
            backend.failures = 0  # type: ignore[misc]

    def test_the_latest_failure_is_the_one_named(self) -> None:
        detector = _Detector(raises=RuntimeError("first"))
        backend = TwoStageBackend(detector, _Segmenter())
        backend.perceive(object(), "a cube")
        detector.raises = OSError("second")
        backend.perceive(object(), "a cube")
        self.assertIn("second", backend.last_failure)
        self.assertNotIn("first", backend.last_failure)

    def test_a_backend_that_counts_nothing_reads_zero(self) -> None:
        self.assertEqual(failures_of(object()), 0)
        self.assertEqual(last_failure_of(object()), "")
        self.assertEqual(failures_of(mock.Mock()), 0, "a stand-in without a real count is no count")


class TheCountsPassThroughTests(unittest.TestCase):
    def test_the_guarded_vlm_backend_reports_its_route_failures(self) -> None:
        inner = _two_stage(raises=RuntimeError("generate failed"))
        guard = GuardedVlmBackend(vlm=inner, model_id="qwen")
        self.assertEqual(guard.perceive(object(), "the broken part"), ())
        self.assertEqual(guard.failures, 1)
        self.assertIn("generate failed", guard.last_failure)

    def test_a_degraded_guard_counts_its_fallback_too(self) -> None:
        class _Unavailable:
            failures = 0

            def perceive(self, image: object, prompt: str) -> tuple[()]:
                raise VlmUnavailableError("qwen", OSError("no weights"))

        fallback = _two_stage(raises=RuntimeError("dino fell over"))
        guard = GuardedVlmBackend(vlm=_Unavailable(), model_id="qwen", degrade=True,
                                  fallback_factory=lambda: fallback)
        with self.assertLogs("src.models.vlm.availability", level="WARNING"):
            self.assertEqual(guard.perceive(object(), "the broken part"), ())
        self.assertEqual(guard.failures, 1)
        self.assertIn("dino fell over", guard.last_failure)

    def test_the_routed_backend_sums_its_built_routes(self) -> None:
        vlm_route = _two_stage(raises=RuntimeError("qwen fell over"))
        simple_route = _two_stage(("cube",))
        routed = RoutedPerceptionBackend(simple_factory=lambda: simple_route, vlm_factory=lambda: vlm_route)
        self.assertEqual(routed.failures, 0, "nothing built, nothing failed")
        routed.perceive(object(), "der kaputte Würfel")       # the VLM route, which raises
        routed.perceive(object(), "a red cube")               # the simple route, which works
        self.assertEqual(routed.failures, 1)
        self.assertIn("qwen fell over", routed.last_failure)


class TheComposedVlmStackTests(unittest.TestCase):
    """The owner's stack, built from config: a VLM that cannot load is a counted failure on every look."""

    def setUp(self) -> None:
        shared_vlm().forget()
        self.addCleanup(shared_vlm().forget)

    def test_a_vlm_that_cannot_load_is_counted_not_nothing(self) -> None:
        from src.config.loader import load_config
        from src.config.schema.models.models_schema import PipelineConfig
        from src.models import factory

        models = load_config().models.model_copy(update={"pipeline": PipelineConfig(
            zero_shot={"backend": "vlm"}, router={"enabled": False})})
        with mock.patch.object(factory, "_build_named_segmenter", return_value=_Segmenter()):
            backend = factory.build_perception(models)
        with mock.patch.object(Qwen3VLGrounder, "_load", side_effect=OSError("no weights on disk")):
            for _ in range(2):
                self.assertEqual(backend.perceive(object(), "the red cube"), ())
        self.assertEqual(backend.failures, 2)
        self.assertIn("no weights on disk", backend.last_failure)


if __name__ == "__main__":
    unittest.main()
