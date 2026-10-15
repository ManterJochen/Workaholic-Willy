"""A cell that fuses two cameras is rehearsed on the calibrations its rigs declare, opening no camera (2026-09-26).

``real_cell --rehearse --profile eth2`` was refused at the build: the fusion block names ``cam_left`` and ``cam_right``,
their calibrations are declared on their rigs in the camera half of the tree, and a rehearsal handed the build no
camera half, so the refusal asked for ``camera=``, which no operator can pass. The command line and the console hold
both halves and now hand the camera half in (``Cell.rehearsal(..., app_config=...)``, ``build_rehearsal_cell``): a
calibrated two-camera cell rehearses its picks on the synthetic scene, single view, and an uncalibrated one is refused
naming the calibration that does not load.
"""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from src.config import ConfigTree


def _calibrated(directory: Path) -> dict[str, str]:
    """Identity calibrations for both rigs of ``eth2``, written into ``directory``, as values to load the tree with."""
    from src.calibration.extrinsics import Extrinsics
    from src.calibration.serialization import save_extrinsics
    from src.geometry import Frame, Transform

    values = {}
    for index, rig in enumerate(("cam_left", "cam_right")):
        path = directory / f"{rig}.json"
        save_extrinsics(path, Extrinsics(
            transform=Transform.identity(from_frame=Frame.CAMERA, to_frame=Frame.BASE), rmse_mm=1.0,
            max_error_mm=2.0, num_samples=10, captured_at=datetime(2026, 1, 1, tzinfo=timezone.utc), rig_id=rig))
        values[f"camera.cameras.rigs[{index}].extrinsics.artifact_path"] = path.as_posix()
    return values


class AFusedCellRehearsesTests(unittest.TestCase):
    def test_a_calibrated_two_camera_cell_rehearses_its_picks(self) -> None:
        from src.robot.execution.cell import Cell
        from src.robot.execution.pick_run import PickOutcome, PickRun, Recording

        with tempfile.TemporaryDirectory() as tmp:
            loaded = ConfigTree.from_directory(profile="console_dummy,eth2").load().with_values(_calibrated(Path(tmp)))
            self.assertTrue(loaded.ok, loaded.error)
            cell = Cell.rehearsal(loaded.robot, data_dir=loaded.root, app_config=loaded.app_config)
            service = cell.build()
            with cell.connected():
                report = PickRun.from_service(service, runs=2, recording=Recording.off()).execute()
        self.assertEqual([attempt.outcome for attempt in report.attempts], [PickOutcome.SUCCEEDED] * 2, report.render())

    def test_the_command_line_rehearsal_is_refused_naming_the_calibration_that_does_not_load(self) -> None:
        """The shipped ``eth2`` tree declares calibrations nobody has swept here: refused for that, not for ``camera=``."""
        from src.robot.execution.real_cell.__main__ import main

        said = io.StringIO()
        with contextlib.redirect_stdout(said):
            code = main(["--rehearse", "--profile", "eth2", "--dry-run"])
        self.assertNotEqual(code, 0, said.getvalue())
        self.assertIn("[build] REFUSED", said.getvalue())
        self.assertIn("camera.cameras.rigs['cam_left'].extrinsics names", said.getvalue())
        self.assertIn("does not load: there is no file at that path", said.getvalue())
        self.assertNotIn("Pass camera=", said.getvalue())

    def test_the_console_rehearsal_reads_the_same_camera_half(self) -> None:
        """The console builds through ``build_rehearsal_cell`` with the tree it resolved, both halves from one read."""
        from api.cell import Console

        with self.assertRaises(RuntimeError) as caught:
            Console(profile="eth2").build(rehearse=True)
        self.assertIn("camera.cameras.rigs['cam_left'].extrinsics names", str(caught.exception))
        self.assertIn("does not load: there is no file at that path", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
