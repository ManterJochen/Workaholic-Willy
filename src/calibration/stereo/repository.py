from pathlib import Path

from .config import CalibrationResult
from .sub_modules.calib_store import StereoCalibrationStore

__all__ = ["StereoRigRepository"]

class StereoRigRepository:
    """Persistence boundary for stereo calibration artifacts.

    Takes the stereomap path as `str` or `Path` and hands it on as `str` to
    `StereoCalibrationStore`, which owns the file format and its schema version.
    """

    def __init__(self, store: StereoCalibrationStore | None = None):
        self.store = store if store is not None else StereoCalibrationStore()

    def load(self, stereomap_file: str | Path) -> CalibrationResult:
        """Reads a stereomap: the pair's calibration."""
        return self.store.load(str(stereomap_file))

    def save(self, stereomap_file: str | Path, result: CalibrationResult) -> None:
        """Writes the calibration, the file rewritten whole."""
        self.store.save(str(stereomap_file), result)
