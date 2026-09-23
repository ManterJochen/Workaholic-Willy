"""UR driver connection config schema."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from .._base import StrictModel
from ._ur_models import UR_MODEL_KEYS


class URConfig(StrictModel):
    """UR-specific connection settings.

    Read only when ``robot.vendor == "ur"``. Vendor blocks are siblings
    under ``robot:`` so each driver owns its own typed surface.
    """

    #: Which UR this cell drives, ``ur5e`` by default. Not cosmetic: the key selects the safety DH
    #: chain, the exact-mesh collision bundle and the cuRobo robot config. A wrong key plans and
    #: collision-checks a real arm against another model's link lengths (ur5e a2/a3 -425/-392.2 mm
    #: against ur3e -243.55/-213.2) with nothing anywhere saying so.
    model: str = Field(default="ur5e")
    ip: str = Field(default="192.168.1.100", min_length=1)
    #: RTDE update rate in Hz. ``0.0``, the default, reads as not configured: let the controller
    #: pick its own rate, 500 Hz on the e-Series. That is a local convention and not ur_rtde's,
    #: whose sentinel for the same thing is -1.0 and which takes 0.0 as literally zero hertz and
    #: fails synchronisation. Against URSim 5.26.0, 0.0 failed in 6.1 s while both -1.0 and 500.0
    #: connected, so the UR connection translates 0.0 -> -1.0 at the driver boundary.
    rtde_frequency: float = Field(default=0.0, ge=0.0)
    # Approach-phase planner, "curobo" by default. It plans a global collision-free trajectory in
    # the process-isolated cuRobo planner (safety.planning), to the goal configuration nearest the
    # arm, and runs it over ur_rtde as one moveJ per kept waypoint: the plan is shortened to the
    # fewest of its own waypoints whose legs stay within the path gate's step of it, the local path
    # gate and the planner both judge those legs, and exactly that list runs. "ik" is the
    # controller's calibrated IK and a moveJ/moveL straight line, which knows nothing about the cell
    # and will drive through anything in it.
    #
    # The "curobo" path is fail-closed: the move returns CONTROLLER_REJECTED when the planner is
    # unavailable and TIMEOUT when no collision-free plan exists. It never degrades to blind IK, so
    # a cell without a cuRobo environment does not move at all. Check that environment with
    # `python -m src.robot.safety.planning --doctor` before commissioning.
    motion_planner: Literal["ik", "curobo"] = "curobo"

    @field_validator("model")
    @classmethod
    def _known_model(cls, v: str) -> str:
        """Reject a typo at config load (<1 s) rather than at the moment an arm is asked to move."""
        if v not in UR_MODEL_KEYS:
            raise ValueError(f"unknown UR model {v!r}; supported: {list(UR_MODEL_KEYS)}")
        return v
