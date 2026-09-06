"""Grasping-side calibration tooling for the uncertainty fusion layer.

It parallels the camera-side :mod:`src.calibration`, but fits behavioural
mappings from a channel to a label rather than metric or geometric ones.

Two modules are deliberately not re-exported. ``model_promotion`` and
``success_model_calibration`` pull numpy and, further in, scikit-learn, and
importing this package has to stay cheap enough for the uncertainty runtime, so
the promotion half is reached through its own module path. ``fits`` is
re-exported, because it adds only the fitter that is already here plus
``src.contracts``, which is stdlib.
"""

from .fits import (
    ChannelFit,
    FitVerdict,
    LabelPolicy,
    UncertaintyFit,
    UncertaintyFitReport,
)
from .uncertainty_calibration import fit_uncertainty_calibration, usable_samples

__all__ = [
    "ChannelFit",
    "FitVerdict",
    "LabelPolicy",
    "UncertaintyFit",
    "UncertaintyFitReport",
    "fit_uncertainty_calibration",
    "usable_samples",
]
