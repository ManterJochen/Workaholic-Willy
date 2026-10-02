"""The grasp overlays a task captures: "what the robot decided", kept as its own image, never drawn over the live one.

Build plan 1.7 and item 15: the calculator renders its overlay when the grasp is decided, before ``pick.executing``;
the task's progress listener keeps it there, and after each pick, only where it is a new image (the identity rule of
``PickRun``: an image that already stood before the pick is no picture of this pick's grasp). ``pick.executing`` and
``pick_result`` carry its URL, ``GET /v1/runs/{id}/overlays/{n}`` serves it, and the store is bounded in memory. What
this file pins:

* the store keeps PNGs by run and number, evicts the oldest beyond its bound, and forgets a run whole;
* a task on console_dummy captures one overlay per grasp, at ``pick.executing``, served as ``image/png``;
* an image that stood before the pick is never claimed for it; the overlay switch is put back as found;
* ``404 no_such_run`` and ``404 no_overlay``; a run the registry forgets takes its overlays with it.

Honesty bucket (2): the real store, console and ``run_task``; the rehearsal's real calculator renders the overlays.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from tests._console_task_fakes import ConsoleCase, Scripted, events_of, wait_for_event


class TheStoreTests(unittest.TestCase):
    def test_it_keeps_by_run_and_number_and_says_where_each_is_served(self) -> None:
        from api.overlays import OverlayStore

        store = OverlayStore(capacity_bytes=1000)
        self.assertEqual("/v1/runs/run-a/overlays/1", store.put("run-a", 1, b"\x89PNG one"))
        self.assertEqual("/v1/runs/run-a/target/overlay", store.put("run-a", "target", b"\x89PNG bin"))
        self.assertEqual(b"\x89PNG one", store.get("run-a", 1))
        self.assertEqual(b"\x89PNG bin", store.get("run-a", "target"))
        self.assertIsNone(store.get("run-a", 2))
        self.assertIsNone(store.get("run-b", 1))
        store.forget("run-a")
        self.assertIsNone(store.get("run-a", 1))
        self.assertIsNone(store.put("run-a", 3, b""), "an empty image was kept")

    def test_it_is_bounded_and_lets_the_oldest_go_first(self) -> None:
        from api.overlays import OverlayStore

        store = OverlayStore(capacity_bytes=250)
        for n in range(1, 4):
            store.put("run-a", n, bytes([n]) * 100)
        self.assertIsNone(store.get("run-a", 1), "the oldest overlay was kept beyond the bound")
        self.assertIsNotNone(store.get("run-a", 2))
        self.assertIsNotNone(store.get("run-a", 3))
        self.assertLessEqual(store.size_bytes, 250)
        self.assertIsNone(store.put("run-a", 9, b"x" * 300), "an overlay larger than the whole store was kept")
        self.assertIsNotNone(store.get("run-a", 3), "a refused overlay evicted the ones it could never replace")


class ATaskCapturesItsOverlaysTests(ConsoleCase):
    def test_one_overlay_per_grasp_as_it_was_decided(self) -> None:
        self.build_dummy()
        service = self.cell.session.service
        self.assertFalse(service.debug_image_rendering_enabled)
        from tests._console_task_fakes import EmptyingScene

        EmptyingScene.install(service, after=2)
        run = self.task(object="", scope="until_empty")
        body = self.finished(run["id"])
        self.assertEqual(2, body["parts_placed"])
        executing = [event.data.get("overlay") for event in events_of(self.cell, run["id"])
                     if event.type == "pick.executing"]
        self.assertEqual([f"/v1/runs/{run['id']}/overlays/1", f"/v1/runs/{run['id']}/overlays/2"], executing)
        results = [event.data.get("overlay") for event in events_of(self.cell, run["id"]) if event.type == "pick_result"]
        self.assertEqual(executing + [None, None], results)
        for url in executing:
            image = self.client.get(url)
            self.assertEqual(200, image.status_code, image.text)
            self.assertEqual("image/png", image.headers["content-type"])
            self.assertTrue(image.content.startswith(b"\x89PNG"))
        self.assertFalse(service.debug_image_rendering_enabled, "the overlay switch was left as the task set it")

    def test_an_image_that_stood_before_the_pick_is_never_claimed_for_it(self) -> None:
        cell = self.scripted([Scripted("part")])
        cell.service.rendered = b"\x89PNG an older pick's overlay"
        run = self.task(object="green cube", options={"overlay": False})
        self.finished(run["id"])
        executing = wait_for_event(self.cell, run["id"], "pick.executing").data
        self.assertNotIn("overlay", executing)
        self.assertIsNone(wait_for_event(self.cell, run["id"], "pick_result").data["overlay"])
        self.refused(self.client.get(f"/v1/runs/{run['id']}/overlays/1"), 404, "no_overlay")

    def test_the_two_refusals(self) -> None:
        self.refused(self.client.get("/v1/runs/run-nobody/overlays/1"), 404, "no_such_run")
        self.refused(self.client.get("/v1/runs/run-nobody/target/overlay"), 404, "no_such_run")
        self.build_dummy()
        run = self.task(object="")
        self.finished(run["id"])
        self.refused(self.client.get(f"/v1/runs/{run['id']}/overlays/7"), 404, "no_overlay")
        self.refused(self.client.get(f"/v1/runs/{run['id']}/target/overlay"), 404, "no_overlay")

    def test_a_run_the_console_forgets_takes_its_overlays_with_it(self) -> None:
        self.build_dummy()
        with patch("api.runs.RETAINED_RUNS", 1):
            first = self.task(object="")
            self.finished(first["id"])
            self.assertIsNotNone(self.cell.overlays.get(first["id"], 1))
            second = self.task(object="")
            self.finished(second["id"])
        self.assertIsNone(self.cell.overlays.get(first["id"], 1), "a forgotten run's overlay is still kept")
        self.assertIsNotNone(self.cell.overlays.get(second["id"], 1))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
