"""The route, once chosen, reaches the operator and the log — and stays invisible when nothing routed.

Two claims, and the second matters as much as the first. A routed cell must SAY which model grounded
the frame, because "why was that slow" and "why did it grasp the wrong thing" are usually answered by
the route. An un-routed cell must produce byte-identical events and records, because that is what keeps
the U12 soak baseline and every existing consumer valid.
"""

from __future__ import annotations

import unittest
from unittest import mock

from src.robot.grasping.loop.progress import PickProgress, PickStage


class _Decision:
    def __init__(self, route: str, reason: str) -> None:
        self.route, self.reason = route, reason


class _Backend:
    def __init__(self, decision=None) -> None:
        self.last_decision = decision


class SourceRouteExposureTests(unittest.TestCase):
    """Both sim sources expose the decision the same way, via one shared duck-typed reader."""

    def test_a_routed_backend_reports_its_decision(self) -> None:
        from src.willy_sim.perception.vision import _last_route

        self.assertEqual(_last_route(_Backend(_Decision("vlm", "non_english"))), ("vlm", "non_english"))

    def test_an_unrouted_backend_reports_nothing_rather_than_guessing_simple(self) -> None:
        """Stamping 'simple' would read as 'the router chose this' when no router ran."""
        from src.willy_sim.perception.vision import _last_route

        self.assertIsNone(_last_route(_Backend()))
        self.assertIsNone(_last_route(None))
        self.assertIsNone(_last_route(object()))


class ProgressEventTests(unittest.TestCase):
    def test_route_fields_default_to_none(self) -> None:
        """Additive: every existing construction site keeps working and says nothing about routing."""
        event = PickProgress(stage=PickStage.PERCEIVED, segmentation_count=3)
        self.assertIsNone(event.route)
        self.assertIsNone(event.route_reason)


class ConsoleRenderingTests(unittest.TestCase):
    def test_the_sentence_names_the_route_when_there_is_one(self) -> None:
        from api.runs import _sentence

        _, sentence = _sentence(PickProgress(
            stage=PickStage.PERCEIVED, attempt=0, segmentation_count=2,
            route="vlm", route_reason="non_english",
        ))
        self.assertIn("2 object(s) segmented", sentence)
        self.assertIn("via the vlm route (non_english)", sentence)

    def test_the_sentence_is_unchanged_when_nothing_routed(self) -> None:
        """The byte-identity claim, at the copy level: no dangling 'via ...' on an un-routed cell."""
        from api.runs import _sentence

        _, sentence = _sentence(PickProgress(
            stage=PickStage.PERCEIVED, attempt=0, segmentation_count=2,
        ))
        self.assertEqual(sentence, "Camera frame acquired: 2 object(s) segmented.")

    def test_the_payload_carries_the_route_and_omits_it_when_absent(self) -> None:
        from api.runs import _payload

        routed = _payload(PickProgress(
            stage=PickStage.PERCEIVED, segmentation_count=1, route="simple",
            route_reason="plain_noun_phrase",
        ))
        self.assertEqual(routed["route"], "simple")
        self.assertEqual(routed["route_reason"], "plain_noun_phrase")

        plain = _payload(PickProgress(stage=PickStage.PERCEIVED, segmentation_count=1))
        self.assertNotIn("route", plain, "an un-routed cell must not gain keys on the wire")
        self.assertNotIn("route_reason", plain)


class RecordExtraTests(unittest.TestCase):
    """extra is the additive channel; the record contract itself is frozen and stays untouched."""

    @staticmethod
    def _service(route):
        from src.robot.execution.autonomous_grasp.service import AutonomousGraspService

        service = object.__new__(AutonomousGraspService)
        service._record_provenance = {"robot_vendor": "ur"}  # noqa: SLF001
        source = mock.Mock()
        source.last_route = route
        orchestrator = mock.Mock()
        orchestrator.perception = source
        runtime = mock.Mock()
        runtime.orchestrator = orchestrator
        service.runtime = runtime
        return service

    def test_a_routed_pick_stamps_the_route_alongside_the_provenance(self) -> None:
        extra = self._service(("vlm", "state_word"))._extra_with_route()
        self.assertEqual(extra["perception_route"], "vlm")
        self.assertEqual(extra["perception_route_reason"], "state_word")
        self.assertEqual(extra["robot_vendor"], "ur", "provenance must survive the merge")

    def test_an_unrouted_pick_logs_exactly_what_it_logged_before(self) -> None:
        """This is the U12-soak-baseline claim: no new keys appear on a cell that does not route."""
        extra = self._service(None)._extra_with_route()
        self.assertEqual(extra, {"robot_vendor": "ur"})

    def test_a_broken_perception_object_never_breaks_logging(self) -> None:
        from src.robot.execution.autonomous_grasp.service import AutonomousGraspService

        service = object.__new__(AutonomousGraspService)
        service._record_provenance = None  # noqa: SLF001
        service.runtime = None
        self.assertIsNone(service._extra_with_route())  # noqa: SLF001


if __name__ == "__main__":
    unittest.main()
