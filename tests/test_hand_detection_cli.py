"""`python -m src.models.handdetection --rig RIG_ID`: which rig it opens, and which it refuses.

⚠ NOTHING IMPORTED THIS MODULE BEFORE. Its own source carries a note saying so, written after a
`FrameProvider` call that had been wrong since the day it shipped. These tests close that: the CLI is
now exercised without a camera, a model bundle or MediaPipe.

MEASURED 2026-09-10 against the code as it stood:

* `--rig does_not_exist` built a `FrameProvider` over EVERY configured rig, opened all of them, and
  then reported "no single hand with usable depth found" (exit 1) because `HandFinder` iterates the
  `rig_ids` it was handed and an id that names nothing simply never matches. A typo therefore read as
  "no hand in the workspace", which on this path is the answer a robot is allowed to move on.
* `--rig <a rig with enabled: false>` did the same, plus it opened the disabled rig's device.
* A cell with three rigs opened three cameras to look at one, fighting the console for devices the
  command does not need. `real_cell/calibrate.py` had already solved exactly this with
  `open_rig` / `release_rig`; this command was the one that had not.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from src.models.handdetection import __main__ as cli


class _Rig:
    def __init__(self, rig_id: str, enabled: bool = True) -> None:
        self.rig_id = rig_id
        self.enabled = enabled
        self.source = "rgbd"


def _config(*rigs: _Rig) -> SimpleNamespace:
    return SimpleNamespace(
        camera=SimpleNamespace(cameras=SimpleNamespace(rigs=list(rigs))),
        models=SimpleNamespace(
            handdetect=SimpleNamespace(model_path="does/not/matter.task"),
            gesturedetect=SimpleNamespace(model_path="does/not/matter.task"),
        ),
    )


class _ProviderSpy:
    """Records what the CLI asked for. Constructing one is the event the refusal tests forbid."""

    instances: list["_ProviderSpy"] = []

    def __init__(self, rigs, stereo=None) -> None:  # noqa: ANN001 - stand-in for FrameProvider
        self.rig_ids = [r.rig_id for r in rigs]
        self.opened_all = False
        self.opened: list[str] = []
        self.released: list[str] = []
        _ProviderSpy.instances.append(self)

    def open(self) -> None:
        self.opened_all = True

    def open_rig(self, rig_id: str) -> None:
        self.opened.append(rig_id)

    def release_rig(self, rig_id: str) -> None:
        self.released.append(rig_id)

    def release(self) -> None:
        self.opened_all = False

    def __enter__(self) -> "_ProviderSpy":
        self.open()
        return self

    def __exit__(self, *_exc) -> bool:
        self.release()
        return False


class _NothingFoundFinder:
    def find_hand(self):
        return None, None


class RigSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        _ProviderSpy.instances = []
        patch_provider = mock.patch(
            "src.camera.orchestration.frame_provider.FrameProvider", _ProviderSpy
        )
        patch_finder = mock.patch(
            "src.models.handdetection.factory.build_hand_finder",
            return_value=_NothingFoundFinder(),
        )
        patch_provider.start()
        patch_finder.start()
        self.addCleanup(patch_provider.stop)
        self.addCleanup(patch_finder.stop)

    def _run(self, config, argv):
        with mock.patch.object(cli, "load_config", return_value=config):
            return cli.main(argv)

    def test_an_unknown_rig_is_a_SETUP_problem_not_an_empty_workspace(self) -> None:
        """Exit 1 means "no hand". A caller that treats a typo as an empty workspace is being told
        the fence is clear by a command that never looked."""
        code = self._run(_config(_Rig("overhead"), _Rig("wrist")), ["--rig", "front"])
        self.assertEqual(code, cli._SETUP_PROBLEM)
        self.assertEqual(_ProviderSpy.instances, [], "a refused rig id still opened cameras")

    def test_the_refusal_lists_the_rigs_there_are(self) -> None:
        with mock.patch("sys.stderr") as err:
            self._run(_config(_Rig("overhead"), _Rig("wrist")), ["--rig", "front"])
        printed = " ".join(str(c.args[0]) for c in err.write.call_args_list if c.args)
        self.assertIn("overhead", printed)
        self.assertIn("wrist", printed)

    def test_a_DISABLED_rig_is_refused_by_name(self) -> None:
        """`enabled: false` is how the shipped base profile ships its only RGB-D rig. Opening it
        anyway is the defect `CameraSystemConfig` documents on its own primary_rig_id field."""
        with mock.patch("sys.stderr") as err:
            code = self._run(_config(_Rig("overhead", enabled=False)), ["--rig", "overhead"])
        printed = " ".join(str(c.args[0]) for c in err.write.call_args_list if c.args)
        self.assertEqual(code, cli._SETUP_PROBLEM)
        self.assertEqual(_ProviderSpy.instances, [], "a disabled rig was still opened")
        self.assertIn("enabled", printed)

    def test_only_the_named_rig_is_opened(self) -> None:
        """Three configured cameras, one asked for. `open()` claims every device in the cell."""
        code = self._run(
            _config(_Rig("overhead"), _Rig("wrist"), _Rig("side", enabled=False)),
            ["--rig", "wrist"],
        )
        self.assertEqual(code, cli._NOTHING_FOUND)
        self.assertEqual(len(_ProviderSpy.instances), 1)
        provider = _ProviderSpy.instances[0]
        self.assertFalse(provider.opened_all, "the CLI opened every configured rig")
        self.assertEqual(provider.opened, ["wrist"])
        self.assertEqual(provider.released, ["wrist"], "the rig was not handed back")


if __name__ == "__main__":
    unittest.main()
