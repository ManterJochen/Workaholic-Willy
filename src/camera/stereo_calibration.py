"""A stereo rig's own calibration: the pair solved against a ChArUco board, so that the rig gives depth.

```python
from willy import StereoCalibration, load_tree

calibration = StereoCalibration.from_tree(load_tree(), rig_id="webcam_main")
print(calibration)                 # the rig, the board, the pairs on disk and the file the calibration goes to
calibration.record()               # a window per eye: s keeps the pair on screen, ESC ends
report = calibration.calibrate()   # solves the pairs and writes the file
print(report)                      # the pairs kept, the residual, the baseline, the depths searched, the file
```

A webcam pair (``source: webcam_pair``) or a side by side stereo device (``source: single_device``) gives depth through
this calibration (the owner, 2026-10-17): both cameras' lenses, where the right one stands from the left, and the
rectification that puts a point on one row of both images, solved by ``StereoCalibrator`` from pairs of a printed
ChArUco board, the one ``camera.cameras.stereo_calibration`` describes. It is written to the rig's
``calibration_paths.stereo_map_file``, where the rig's owner reads it as it opens (``StereoDepthStreamer``). It says
nothing of where the pair stands in the cell: that is the rig's hand-eye calibration, swept afterwards as an RGB-D
rig's is (``HandEyeCalibration``), and once its ``extrinsics`` are declared and ``camera.cameras.primary_rig_id`` names
it, the pair is the cell's camera.

A calibration whose residual is poor, above 2 pixels (``calibration.quality``), is not written: its depth would be as
wrong as its rows are, and nothing downstream could tell. Measure the printed board: its square sets the scale of the
baseline, and so of every depth.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.calibration.exceptions import StereoCalibrationError
from src.calibration.quality import DEFAULT_BANDS_PX, classify_rmse
from src.calibration.stereo.repository import StereoRigRepository
from src.calibration.stereo.sub_modules.calibrator import StereoCalibrator
from src.calibration.stereo.sub_modules.depth import StereoDepth, working_depth_range_mm
from src.camera.setup.image_taking.stereo_depth import eye_size
from src.config.schema.camera import CalibrationConfig, SingleDeviceRigConfig, WebcamPairRigConfig
from src.contracts import UNSET, Maybe, chosen

__all__ = ["StereoCalibration", "StereoCalibrationReport"]

StereoRig = WebcamPairRigConfig | SingleDeviceRigConfig


@dataclass(frozen=True, slots=True)
class StereoCalibrationReport:
    """What one stereo calibration came to: what it rests on, what it gives, and where it was written.

    Attributes:
        rig_id (str): The rig calibrated.
        refusal (str): Why nothing was written, one sentence; empty when the file was (default: "").
        written (str): The stereo map file written; empty when nothing was (default: "").
        pairs_found (int): The image pairs on disk (default: 0).
        pairs_used (int): The pairs whose board both cameras saw well enough to be solved with (default: 0).
        rms_px (float): The pair's reprojection residual, pixels; NaN before a solve (default: NaN).
        quality (str): The residual's band: excellent, good, marginal, poor or unknown (default: "unknown").
        baseline_mm (float): The distance between the two cameras' centres, millimetres (default: NaN).
        focal_px (float): The rectified pair's focal length, pixels (default: NaN).
        frame_size (tuple[int, int]): The size of one eye, ``(width, height)`` pixels (default: (0, 0)).
        near_mm (float): The nearest depth the pair is read at where no robot section says otherwise (default: NaN).
        disparities (tuple[int, int]): The smallest and the largest disparity searched there, pixels
            (default: (0, 0)).
    """

    rig_id: str
    refusal: str = ""
    written: str = ""
    pairs_found: int = 0
    pairs_used: int = 0
    rms_px: float = float("nan")
    quality: str = "unknown"
    baseline_mm: float = float("nan")
    focal_px: float = float("nan")
    frame_size: tuple[int, int] = (0, 0)
    near_mm: float = float("nan")
    disparities: tuple[int, int] = (0, 0)

    @property
    def ok(self) -> bool:
        """Whether the calibration was written."""
        return not self.refusal

    @property
    def exit_code(self) -> int:
        """0 written, 1 refused and nothing written."""
        return 0 if self.ok else 1

    def __str__(self) -> str:
        """What ``print()`` shows: the text :meth:`render` returns."""
        return self.render()

    def render(self) -> str:
        """The report as a person reads it.

        Returns:
            str: ASCII, no trailing newline.
        """
        if not self.ok:
            return f"stereo calibration of {self.rig_id!r}: REFUSED, nothing written: {self.refusal}"
        return (
            f"stereo calibration of {self.rig_id!r}: written to {self.written}\n"
            f"  pairs       {self.pairs_used} of {self.pairs_found} solved with\n"
            f"  residual    {self.rms_px:.3f} px ({self.quality})\n"
            f"  baseline    {self.baseline_mm:.1f} mm, focal length {self.focal_px:.1f} px, "
            f"{self.frame_size[0]} x {self.frame_size[1]} per eye\n"
            f"  depth       read from {self.near_mm:.0f} mm on, disparities {self.disparities[0]} to "
            f"{self.disparities[1]}, wherever no robot section says where the work starts\n"
            "  next        calibrate the rig against the robot (examples/real_robot/07 to 10), declare its "
            "extrinsics, and name it camera.cameras.primary_rig_id"
        )

    def to_dict(self) -> dict[str, Any]:
        """The report as plain data.

        Returns:
            dict[str, Any]: Every field, and ``ok``; ``json.dumps`` safe but for a NaN, which it writes as NaN.
        """
        return {**dataclasses.asdict(self), "ok": self.ok}


class StereoCalibration:
    """One stereo rig's calibration against the ChArUco board its camera section describes.

    Build it with :meth:`from_tree`; building touches no device. :meth:`record` opens the pair and records board
    pairs interactively, :meth:`calibrate` solves the pairs on disk and writes the rig's stereo map file.

    Args:
        rig (StereoRig): The rig, a ``webcam_pair`` or ``single_device`` entry of ``camera.cameras.rigs``.
        board (CalibrationConfig): The printed board, ``camera.cameras.stereo_calibration``.
    """

    def __init__(self, rig: StereoRig, board: CalibrationConfig) -> None:
        self._rig = rig
        self._board = board

    @classmethod
    def from_tree(cls, tree: Any, *, rig_id: Maybe[str] = UNSET) -> StereoCalibration:
        """The calibration of one stereo rig of a loaded tree, not yet run.

        Args:
            tree (Any): A loaded tree, ``load_tree()``.
            rig_id (Maybe[str]): Which rig of ``camera.cameras.rigs``; unset is the one the cell runs on,
                ``camera.cameras.primary_rig_id`` (default: UNSET).

        Returns:
            StereoCalibration: The calibration; nothing is opened.

        Raises:
            ConfigError: The tree did not load.
            ValueError: No such rig, a rig that is not a stereo pair, or a section that describes no board.
        """
        cameras = tree.app_config.camera.cameras
        wanted = rig_id if chosen(rig_id) else cameras.primary_rig_id
        rig = next((each for each in cameras.rigs if each.rig_id == wanted), None)
        if rig is None:
            raise ValueError(f"no rig {wanted!r} in camera.cameras.rigs; configured: "
                             f"{sorted(each.rig_id for each in cameras.rigs)}")
        if not isinstance(rig, (WebcamPairRigConfig, SingleDeviceRigConfig)):
            raise ValueError(f"rig {wanted!r} is a {rig.source!r} rig, whose depth is its own: only a stereo pair, "
                             "a webcam_pair or a single_device rig, is calibrated here")
        if cameras.stereo_calibration is None:
            raise ValueError("camera.cameras.stereo_calibration is not set, and it is the board the pair is solved "
                             "against: describe the printed ChArUco board there, its squares measured on the print")
        return cls(rig, cameras.stereo_calibration)

    @property
    def rig_id(self) -> str:
        """The rig's id, as ``camera.cameras.rigs`` names it."""
        return str(self._rig.rig_id)

    @property
    def stereo_map_file(self) -> Path:
        """Where the calibration is written, and where the rig's owner reads it: ``calibration_paths.stereo_map_file``."""
        return Path(str(self._rig.calibration_paths.stereo_map_file))

    @property
    def pairs(self) -> int:
        """How many image pairs are on disk, the fewer of the left and the right images."""
        paths = self._rig.calibration_paths
        return min(len(list(Path(str(paths.left_images_dir)).glob("*.png"))),
                   len(list(Path(str(paths.right_images_dir)).glob("*.png"))))

    def record(self) -> int:
        """Open the pair and record board pairs: a window per eye, ``s`` keeps the pair on screen and ESC ends.

        The images are written to the rig's ``calibration_paths.left_images_dir`` and ``right_images_dir``, beside the
        ones already there; the session ends by itself at the rig's ``max_pairs``.

        Returns:
            int: How many pairs are on disk once the session ended.

        Raises:
            OSError: The pair does not open.
            ValueError: The pair does not deliver the size or the frame rate its rig configures.
        """
        if isinstance(self._rig, SingleDeviceRigConfig):
            from src.camera.setup.devices.stereocamera import StereoVisionCalibrationSingleDevice  # noqa: PLC0415

            StereoVisionCalibrationSingleDevice(dev_cfg=self._rig).forward()
        else:
            from src.camera.setup.devices.webcam import StereoVisionCalibrationWebcams  # noqa: PLC0415

            StereoVisionCalibrationWebcams(wc_cfg=self._rig).forward()
        return self.pairs

    def calibrate(self) -> StereoCalibrationReport:
        """Solve the pairs on disk and write the rig's stereo map file, unless the solve is refused.

        Returns:
            StereoCalibrationReport: What the calibration rests on and gives; a refusal, nothing written, where too few
                pairs show the board, the residual is poor, or the solve puts the cameras where no depth is read along
                rows.
        """
        board, rig = self._board, self._rig
        calibrator = StereoCalibrator(board.charuco_squares_x, board.charuco_squares_y, board.charuco_square_length_mm,
                                      board.charuco_marker_length_mm, board.aruco_dict_name, board.rectify_alpha)
        paths = rig.calibration_paths
        try:
            result = calibrator.calibrate(eye_size(rig), str(paths.left_images_glob), str(paths.right_images_glob))
            depth = StereoDepth(result, working_depth_range_mm())
        except StereoCalibrationError as exc:
            return StereoCalibrationReport(rig_id=self.rig_id, refusal=str(exc), pairs_found=self.pairs)
        solve = calibrator.last_solve
        assert solve is not None  # set by every calibrate() that returned
        width, height = result.frame_size
        report = StereoCalibrationReport(
            rig_id=self.rig_id, pairs_found=solve.pairs_found, pairs_used=solve.pairs_used, rms_px=solve.rms_stereo_px,
            quality=classify_rmse(solve.rms_stereo_px, DEFAULT_BANDS_PX), baseline_mm=depth.baseline_mm,
            focal_px=depth.focal_px, frame_size=(int(width), int(height)), near_mm=depth.depth_range_mm[0],
            disparities=(depth.search.min_disparity, depth.search.max_disparity - 1))
        if report.quality == "poor":
            return dataclasses.replace(report, refusal=(
                f"the pair's residual is {solve.rms_stereo_px:.2f} px, above the {DEFAULT_BANDS_PX.marginal:.1f} px a "
                "calibration may have: the board moved or blurred, or its corners were misread. Record the pairs "
                "again, the board held still and filling a good part of both views"))
        StereoRigRepository().save(self.stereo_map_file, result)
        return dataclasses.replace(report, written=str(self.stereo_map_file))

    def __str__(self) -> str:
        """What ``print()`` shows: the text :meth:`render` returns."""
        return self.render()

    def render(self) -> str:
        """The calibration as a person reads it, before it runs.

        Returns:
            str: ASCII, no trailing newline.
        """
        board, rig = self._board, self._rig
        width, height = eye_size(rig)
        written = "written" if self.stereo_map_file.is_file() else "not written yet"
        return (
            f"stereo calibration of {self.rig_id!r} ({rig.source}, {width} x {height} per eye)\n"
            f"  board       ChArUco {board.charuco_squares_x} x {board.charuco_squares_y} squares of "
            f"{board.charuco_square_length_mm:g} mm, markers {board.charuco_marker_length_mm:g} mm, "
            f"{board.aruco_dict_name}\n"
            f"  pairs       {self.pairs} on disk ({rig.calibration_paths.left_images_dir}), "
            f"{rig.min_pairs} needed, a session ends at {rig.max_pairs}\n"
            f"  calibration {self.stereo_map_file} ({written})"
        )

    def to_dict(self) -> dict[str, Any]:
        """The calibration as plain data.

        Returns:
            dict[str, Any]: ``rig_id``, ``source``, ``pairs``, ``stereo_map_file`` and whether it is written.
        """
        return {"rig_id": self.rig_id, "source": str(self._rig.source), "pairs": self.pairs,
                "stereo_map_file": str(self.stereo_map_file), "written": self.stereo_map_file.is_file()}
