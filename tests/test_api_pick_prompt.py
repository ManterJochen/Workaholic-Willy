"""The console's prompt reaches the detector, and the cell's own phrase is back when the run ends (Q6, owner 2026-09-18).

`POST /v1/pick {prompt}` used to hand the text to `service.set_target_label` alone. The console builds a
real cell with the phrase "object" (`api/cell.py`), so a typed or spoken "the red cube" never reached the
detector: every frame was still grounded with "object", and the run filtered on a label no phrase
grounder returns. The run now goes through `service.set_prompt` and puts the phrase back in its
`finally`.

A rehearsal cell grounds no phrase, so after the build the rehearsal's own scene source is replaced by
the live camera source over a stand-in streamer that serves the same scene, with a phrase grounder
double as its backend. Everything else is the console as an operator drives it: the route, the run
registry, the run thread and the service.

Honesty bucket (2): a dummy cell, real threads, the real route; no hardware and no weights.
"""

from __future__ import annotations

import re
import shutil
import tempfile
import time
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover - the console is an optional extra
    TestClient = None  # type: ignore[assignment,misc]

from src.config.loader import active_profile, reload_config, set_active_profile

_SHIPPED = Path(__file__).resolve().parents[1] / "config"
_NO_FRAME = "Frame didn't arrive within 5000"


def _dummy_tree(target: Path) -> None:
    """The shipped tree with a dummy arm and no gripper, as `tests/test_api_pick.py` builds it."""
    shutil.copytree(_SHIPPED, target)
    robot = target / "robot" / "robot.yaml"
    text = robot.read_text(encoding="utf-8")
    text, arm_hits = re.subn(
        r'^(\s*)vendor:\s*"ur"$', r'\g<1>vendor: "dummy"', text, count=1, flags=re.MULTILINE
    )
    text, gripper_hits = re.subn(
        r'^(\s*)vendor:\s*"robotiq"$', r'\g<1>vendor: "none"', text, count=1, flags=re.MULTILINE
    )
    assert arm_hits == 1 and gripper_hits == 1, "the dummy substitution found nothing"
    robot.write_text(text, encoding="utf-8")


@dataclass(frozen=True)
class _Seg:
    mask: np.ndarray
    label: str


class _PhraseGrounder:
    """GroundingDINO's part: the rehearsal box, labelled with the words it matched, every caption kept."""

    _ARTICLES = frozenset({"the", "a", "an"})

    def __init__(self, mask: np.ndarray) -> None:
        self._mask = mask
        self.captions: list[str] = []

    def perceive(self, image_bgr: Any, prompt: str) -> tuple[Any, ...]:
        self.captions.append(prompt)
        label = " ".join(w for w in prompt.split() if w.lower() not in self._ARTICLES) or prompt
        rows, cols = np.nonzero(self._mask)
        box = [float(cols.min()), float(rows.min()), float(cols.max()) + 1.0, float(rows.max()) + 1.0]
        detection = SimpleNamespace(box=box, label=label, score=0.9)
        return (SimpleNamespace(detection=detection, segmentation=_Seg(mask=self._mask.copy(), label=label)),)


class _SceneStreamer:
    """The rehearsal scene as an RGB-D streamer would deliver it: colour, depth in millimetres, and K."""

    def __init__(self, color: np.ndarray, depth: np.ndarray, k: np.ndarray) -> None:
        from src.camera.setup.image_taking.frames import RGBDFrame

        self._frame = RGBDFrame(color=color, depth=depth)
        self._k = k

    def grab(self) -> Any:
        return self._frame

    def get_intrinsics(self) -> np.ndarray:
        return self._k.copy()


class _DeadCamera:
    def acquire(self) -> Any:
        raise RuntimeError(_NO_FRAME)


@unittest.skipIf(TestClient is None, "fastapi is unavailable; requirements.txt pins fastapi and httpx")
class TheConsolePromptReachesTheDetectorTests(unittest.TestCase):

    def setUp(self) -> None:
        from api.app import create_app
        from api.cell import Console, set_console

        self.tmp = Path(tempfile.mkdtemp())
        _dummy_tree(self.tmp / "data")
        self._previous_profile = active_profile()
        self.cell = Console(root=self.tmp / "data", profile=None)
        # A scratch record log: the console's default is under logs/, and a test must not append to it.
        self.cell.record_log_path = self.tmp / "records.jsonl"
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
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _connected(self) -> None:
        self.assertEqual(self.client.post("/v1/cell/build", params={"rehearse": True}).status_code, 200)
        token = self.client.get("/v1/cell/connect-preview").json()["token"]
        self.assertEqual(self.client.post("/v1/cell/connect", json={"token": token}).status_code, 200)

    def _orchestrator(self) -> Any:
        return self.cell.session.service.runtime.orchestrator

    def _live_camera(self) -> tuple[Any, _PhraseGrounder]:
        """The live camera source over the rehearsal scene, built with the console's phrase."""
        from src.robot.execution.autonomous_grasp.rehearsal import (
            RehearsalPerceptionSource,
            rehearsal_intrinsics,
        )
        from src.robot.perception import RealSenseVisionPerceptionSource

        scene = RehearsalPerceptionSource()
        frame = scene.acquire()
        streamer = _SceneStreamer(
            color=scene.peek_color(), depth=np.asarray(frame.depth_map).astype(np.uint16),
            k=rehearsal_intrinsics(),
        )
        grounder = _PhraseGrounder(np.asarray(frame.segmentations[0].mask, dtype=np.uint8))
        source = RealSenseVisionPerceptionSource(
            streamer=streamer, backend=grounder, prompt=self.cell.prompt, warmup_grabs=0,
        )
        self._orchestrator().perception = source
        return source, grounder

    def _await_run(self, run_id: str, *, timeout: float = 20.0) -> dict:
        """The run's record once its thread has published `run_finished`.

        Waiting on the event rather than on the state: the state flips before the run's `finally`
        puts the prompt back, and the event is the last thing that `finally` does.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if any(e.type == "run_finished" for e in self.cell.hub.since(run_id, 0)[0]):
                return self.client.get(f"/v1/runs/{run_id}").json()
            time.sleep(0.02)
        self.fail(f"run {run_id} did not finish within {timeout}s")

    def test_a_typed_prompt_reaches_the_detector_and_the_build_phrase_comes_back(self) -> None:
        """Red before this step: the run set the label filter alone, so the detector read "object"."""
        self._connected()
        source, grounder = self._live_camera()

        response = self.client.post("/v1/pick", json={"prompt": "the red cube", "picks": 1})
        self.assertEqual(response.status_code, 202)
        body = self._await_run(response.json()["id"])

        self.assertTrue(grounder.captions, body)
        self.assertEqual(set(grounder.captions), {"the red cube"}, "the typed prompt never reached the detector")
        self.assertEqual(source.prompt, "object", "the cell's own phrase did not come back")
        self.assertEqual(source.object_labels, ())
        self.assertIsNone(self._orchestrator().target_label, "the label filter outlived the run")

    def test_an_empty_prompt_keeps_the_build_phrase(self) -> None:
        """A control, green before and after: no prompt grounds the build phrase and filters nothing."""
        self._connected()
        source, grounder = self._live_camera()

        self._await_run(self.client.post("/v1/pick", json={"picks": 1}).json()["id"])

        self.assertEqual(set(grounder.captions), {"object"})
        self.assertIsNone(self._orchestrator().target_label)

    def test_a_fault_of_the_cell_still_ends_the_run_and_the_label_is_put_back(self) -> None:
        """The run fails on a camera that stopped delivering, as it did while `pick()` raised it, and it
        does not ask the dead camera for the remaining picks.

        Red before this step: the label filter was never cleared when a run ended, so it outlived this one.
        """
        self._connected()
        self._orchestrator().perception = _DeadCamera()

        body = self._await_run(self.client.post("/v1/pick", json={"prompt": "the red cube", "picks": 3}).json()["id"])

        self.assertEqual(body["state"], "failed")
        self.assertEqual(body["error"], f"RuntimeError: {_NO_FRAME}")
        self.assertEqual(body["attempted"], 0)
        self.assertIsNone(self._orchestrator().target_label, "the label filter outlived the run")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
