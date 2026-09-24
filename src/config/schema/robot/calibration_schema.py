"""Hand-eye calibration config schema."""

from __future__ import annotations

from pydantic import Field, model_validator

from .._base import StrictModel


class RobotCalibrationQualityBandsMm(StrictModel):
    """RMSE thresholds (mm) that classify a hand-eye calibration result.

    Each boundary is an inclusive upper bound: an RMSE of exactly ``good``
    lands in the "good" band, not in "marginal".
    """

    excellent: float = Field(default=1.0, gt=0.0)
    good: float = Field(default=2.5, gt=0.0)
    marginal: float = Field(default=5.0, gt=0.0)

    @model_validator(mode="after")
    def _check_ordering(self) -> RobotCalibrationQualityBandsMm:
        if not (self.excellent < self.good < self.marginal):
            raise ValueError(
                f"quality bands must be strictly ordered: "
                f"{self.excellent} < {self.good} < {self.marginal}"
            )
        return self


class RobotCalibrationConfig(StrictModel):
    """Tunable parameters for the hand-eye calibration routine.

    Two keys this block deliberately does not offer:

    * ``quality_threshold_mm``, an auto-apply gate that would hold an RMSE above it for operator
      review. No such gate exists anywhere: ``save_extrinsics`` writes whatever it is given,
      judging the RMSE is the operator's job, and :data:`quality_bands_mm` is what labels it.
    * ``speed_scale``, an arm-speed scale for the duration of the routine. There is no speed
      governor; ``robot.motion_limits`` is what the driver layer enforces, and the routine takes
      a ``motion_limits=`` argument.

    ``quality_bands_mm`` is the opposite case: both calibration runners (``run_eth_calibrate.py``,
    ``run_eih_calibrate.py``) pass it straight into :func:`src.calibration.quality.classify_rmse`.

    Nor does it offer anything that shapes generated poses (a spread, a retry budget, a pose box):
    nothing generates calibration stations. They are written down, taught, or guided by hand.
    """

    # Mechanical settle time after each pose move, before image capture.
    settle_time_s: float = Field(default=0.5, ge=0.0)

    #: How many counted samples a hand-guided calibration collects before it solves (``--freedrive``), unless the
    #: run says otherwise (``--samples``, ``SweepOptions(samples=)``). The person may finish earlier; the solve still
    #: needs the hand-eye block's ``min_samples``.
    freedrive_samples: int = Field(default=15, ge=1)

    #: Structurally identical to :class:`src.calibration.quality.QualityBandsMm`. The config layer
    #: redeclares it rather than importing it: config sits below the source tree and must not
    #: depend on it.
    quality_bands_mm: RobotCalibrationQualityBandsMm = Field(
        default_factory=RobotCalibrationQualityBandsMm
    )
