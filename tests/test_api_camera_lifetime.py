"""Every camera this server opens, it closes — and a refused rebuild opens nothing at all.

WHY THIS FILE EXISTS. `Console.build()` opens an `rs.pipeline` and loads two models onto the GPU.
Nothing else in the server ever closed one: `CellSession.adopt()` overwrote `self.service` and the old
service went on the floor with its device still streaming, inside a reference cycle
(service -> runtime -> orchestrator) that refcounting would not have collected even if a destructor
had existed. There is none: `RealSenseRGBDStreamer` has `release()` but no `__del__` and no
context-manager protocol.

That is not a slow leak, it is a hard stop on real hardware. A second `pipeline.start()` on a device
the first build is still streaming FAILS, and the loop it breaks is the ordinary one:

    build  ->  read the refusal  ->  fix a config key  ->  build again

Measured before the fix, on the shipped tree: three builds produced three `open()` calls and zero
`release()` calls, and the third was refused with `wrong_state` *after* it had already claimed the
device.

THE ORDERING MATTERS AS MUCH AS THE RELEASE, and the first fix got only half of it. Releasing inside
`adopt()` happens AFTER the replacement streamer is already open — which closes the leak and still
fails on one physical camera, because that is exactly the double-open librealsense refuses. It shipped
green because this file's fake counted `open()` calls instead of modelling a device that can be held
only once. `_CountingStreamer` now raises `DeviceBusy`, and `Console.build()` releases before it
acquires. Three rules, in order: check the state, release the old device, then open the new one.

Honesty bucket ②: a real `Console`, the real build path, a fake streamer. No physical camera has been
on this path — a fake is what makes the counting exact, and it is now shaped to fail the way the real
device fails, which is the only property that made this test worth having.
"""

from __future__ import annotations

import pathlib
import re
import shutil
import tempfile
import unittest

import numpy as np

from api.cell import Console, set_console
from api.lifecycle import CellState, CellTransitionError, release_perception

_SHIPPED = pathlib.Path(__file__).resolve().parents[1] / "config"


class DeviceBusy(RuntimeError):
    """What librealsense does when a second pipeline starts on a device that is already streaming."""


class _CountingStreamer:
    """Shaped like ``RealSenseRGBDStreamer``, keeps score -- and MODELS AN EXCLUSIVE DEVICE.

    The exclusivity is the point. The first version of this fake only appended to a list, so it
    happily reported two simultaneous opens and the suite went green while the real ordering bug --
    open the new camera, then release the old one -- was still in the code. A fake that cannot fail
    the way the real thing fails is not testing the thing it is named after.

    ``held`` is shared by every instance because a cell has ONE camera: two streamers here stand for
    two *attempts on the same device*, not two devices.
    """

    def __init__(self, tag: str, opens: list[str], releases: list[str], held: list[str]) -> None:
        self.tag, self._opens, self._releases, self._held = tag, opens, releases, held

    def open(self) -> None:
        if self._held:
            raise DeviceBusy(
                f"cannot start a pipeline for {self.tag}: {self._held[0]} is still streaming"
            )
        self._held.append(self.tag)
        self._opens.append(self.tag)

    def release(self) -> None:
        if self.tag in self._held:
            self._held.remove(self.tag)
        self._releases.append(self.tag)

    def grab(self):  # noqa: ANN201 - only the attributes the source reads
        return type("_F", (), {"color": np.zeros((8, 8, 3), np.uint8),
                               "depth": np.zeros((8, 8), np.uint16)})()

    def get_intrinsics(self) -> np.ndarray:
        return np.eye(3)


class CameraLifetimeTests(unittest.TestCase):
    def setUp(self) -> None:
        import src.robot.execution.autonomous_grasp.cells as components

        self.opens: list[str] = []
        self.releases: list[str] = []
        #: The one physical device. Non-empty means somebody is holding it.
        self.held: list[str] = []
        self._n = 0

        from src.robot.grasping.generation.calculator import GraspCalculator
        from src.robot.perception import RealSenseVisionPerceptionSource

        def _fake_build_real(robot_cfg, prompt):  # noqa: ANN001, ANN202
            self._n += 1
            streamer = _CountingStreamer(f"cam{self._n}", self.opens, self.releases, self.held)
            source = RealSenseVisionPerceptionSource(
                streamer=streamer, backend=object(), prompt=prompt
            )
            streamer.open()          # exactly where build_real_components does it
            calculator = GraspCalculator(
                camera_matrix=np.eye(3),
                max_grip_width_mm=robot_cfg.gripper.max_width_mm,
                min_grip_width_mm=robot_cfg.gripper.min_width_mm,
            )
            # Five, matching `build_real_components`: the last is the per-camera calculator
            # map, and `None` is right here because this cell has one camera.
            return calculator, source, None, None, None

        self._real = components.build_real_components
        components.build_real_components = _fake_build_real
        self.addCleanup(setattr, components, "build_real_components", self._real)

        self.tmp = pathlib.Path(tempfile.mkdtemp()) / "data"
        shutil.copytree(_SHIPPED, self.tmp)
        robot = self.tmp / "robot" / "robot.yaml"
        text = robot.read_text(encoding="utf-8")
        text, arm = re.subn(
            r'^(\s*)vendor:\s*"ur"$', r'\g<1>vendor: "dummy"', text, count=1, flags=re.M
        )
        text, grip = re.subn(
            r'^(\s*)vendor:\s*"robotiq"$', r'\g<1>vendor: "none"', text, count=1, flags=re.M
        )
        assert arm == 1 and grip == 1, "the dummy substitution found nothing"
        robot.write_text(text, encoding="utf-8")

        self.cell = Console(root=self.tmp, profile=None)
        self._previous = set_console(self.cell)
        self.addCleanup(set_console, self._previous)
        self.addCleanup(shutil.rmtree, self.tmp.parent, True)

    # -- the leak ----------------------------------------------------------------------------------

    def test_rebuilding_closes_the_camera_the_previous_build_opened(self) -> None:
        for _ in range(3):
            self.cell.build(rehearse=False)
        self.assertEqual(self.opens, ["cam1", "cam2", "cam3"])
        # Two, not three: the live one is still open, which is the point of having built it.
        self.assertEqual(self.releases, ["cam1", "cam2"])

    def test_the_OLD_camera_is_released_BEFORE_the_new_one_is_opened(self) -> None:
        """The ordering, which the leak fix alone did not give.

        A cell has one camera. Releasing inside ``adopt()`` -- i.e. after the new streamer is already
        open -- closes the leak and still fails on real hardware, because ``pipeline.start()`` on a
        device another pipeline is streaming is refused. `_CountingStreamer` now models that refusal,
        so this test fails against the earlier ordering instead of passing through it.
        """
        self.cell.build(rehearse=False)
        self.cell.build(rehearse=False)          # would raise DeviceBusy under the old order
        self.assertEqual(self.opens, ["cam1", "cam2"])
        self.assertEqual(self.releases, ["cam1"])
        self.assertEqual(self.held, ["cam2"], "exactly one device is held at a time")

    def test_at_most_one_device_is_ever_held_across_many_rebuilds(self) -> None:
        for _ in range(5):
            self.cell.build(rehearse=False)
            self.assertLessEqual(len(self.held), 1, f"two pipelines at once: {self.held}")

    def test_a_refused_rebuild_claims_NOTHING(self) -> None:
        """The ordering half. Asking after acquiring meant a refusal still took the device."""
        self.cell.build(rehearse=False)
        self.cell.session.state = CellState.CONNECTED

        opened_before = len(self.opens)
        with self.assertRaises(CellTransitionError) as caught:
            self.cell.build(rehearse=False)
        self.assertEqual(str(caught.exception.reason), "wrong_state")
        self.assertEqual(len(self.opens), opened_before, "a refused build opened a camera")

    def test_releasing_the_session_balances_every_open(self) -> None:
        """What the server's lifespan calls on the way out."""
        self.cell.build(rehearse=False)
        self.cell.build(rehearse=False)
        self.cell.session.release()
        self.assertEqual(len(self.releases), len(self.opens))
        self.assertIsNone(self.cell.session.service)
        self.assertIs(self.cell.session.state, CellState.DISCONNECTED)

    def test_release_is_idempotent(self) -> None:
        self.cell.build(rehearse=False)
        self.cell.session.release()
        self.cell.session.release()
        self.assertEqual(self.releases, ["cam1"], "the second release closed something twice")

    # -- the helper's own contract -----------------------------------------------------------------

    def test_a_rehearsal_build_owns_no_device_and_that_is_not_an_error(self) -> None:
        """Most sources own nothing. `release_perception` must be quiet about it, not defensive."""
        self.cell.build(rehearse=True)
        self.cell.session.release()
        self.assertEqual(self.opens, [])
        self.assertEqual(self.releases, [])

    def test_release_perception_survives_anything_it_is_handed(self) -> None:
        class _Angry:
            def close(self) -> None:
                raise OSError("the device is on fire")

        for case in (None, object(), _Angry()):
            with self.subTest(case=type(case).__name__):
                service = type("_S", (), {})()
                service.runtime = type("_R", (), {})()
                service.runtime.orchestrator = type("_O", (), {})()
                service.runtime.orchestrator.perception = case
                release_perception(service)          # must not raise
        release_perception(None)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
