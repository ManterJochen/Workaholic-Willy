"""PayloadGuard, the tool and payload envelope safety guard.

Per-move evaluation is deliberately cheap: it verifies that the currently configured
payload still fits the operator-declared envelope. The heavy lift, pushing the payload
mass and centre of gravity into the UR controller through the RTDE ``setPayload``,
happens in :meth:`URRobotArm.connect`, so the controller protective-stop calculation
sees the same values this stack believes.

What each vendor actually does
------------------------------
* UR: the driver pushes ``mass_kg`` and ``cog_mm`` to the controller through
  :meth:`URConnection.set_payload` on connect, so after a successful connect the
  configured payload and the running controller cannot disagree.
* KUKA: EKI exposes no runtime payload write. The shipped ``robot.web.yaml`` keeps the
  same validation, but only the operator setting the controller-side ``$LOAD`` makes
  it real. The guard surface is unchanged and the YAML comment records the gap.
* Sim: there is no real payload, and the guard still validates the envelope so an
  operator gets the same diagnostics on the simulator as on hardware.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .decision import SafetyDecision, SafetyReason
from .guard import SafetyContext

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.config.schema.robot import PayloadSafetyConfig

__all__ = ["PayloadGuard"]


class PayloadGuard:
    """Mass / CoG / inertia envelope guard.

    Stateless across calls. :meth:`SafetyPreflight.from_safety_config` constructs it
    when ``safety.payload.enforce`` is ``True``.
    """

    name = "payload"

    def __init__(self, config: "PayloadSafetyConfig") -> None:
        self._config = config

    def evaluate(self, ctx: SafetyContext) -> SafetyDecision:
        # The Pydantic ``PayloadSafetyConfig`` is the primary validator and already
        # rejects a negative mass, a mass over the maximum and a negative inertia at
        # construction. The three runtime checks below are a deliberate second line:
        # they still fail closed for a frozen config assembled by a path that bypassed
        # schema validation, such as a grasping-preset overlay or a hand-built config,
        # rather than letting a bad payload reach the controller. They are not
        # redundant; do not delete them.
        cfg = self._config

        if cfg.mass_kg < 0.0:
            return SafetyDecision.reject(
                self.name,
                SafetyReason.PAYLOAD,
                message=(
                    f"payload.mass_kg ({cfg.mass_kg:.6f}) must be >= 0"
                ),
                detail={
                    "reason": "negative_mass",
                    "mass_kg": f"{cfg.mass_kg:.6f}",
                },
            )

        if cfg.mass_kg > cfg.max_mass_kg:
            return SafetyDecision.reject(
                self.name,
                SafetyReason.PAYLOAD,
                message=(
                    f"payload.mass_kg ({cfg.mass_kg:.6f}) exceeds "
                    f"max_mass_kg ({cfg.max_mass_kg:.6f})"
                ),
                detail={
                    "reason": "over_mass",
                    "mass_kg": f"{cfg.mass_kg:.6f}",
                    "max_mass_kg": f"{cfg.max_mass_kg:.6f}",
                },
            )

        for axis_label, val in zip(("Ixx", "Iyy", "Izz"), cfg.inertia_kgm2):
            if val < 0.0:
                return SafetyDecision.reject(
                    self.name,
                    SafetyReason.PAYLOAD,
                    message=(
                        f"payload.inertia_kgm2.{axis_label} "
                        f"({val:.6f}) must be >= 0"
                    ),
                    detail={
                        "reason": "negative_inertia",
                        "axis": axis_label,
                        "value": f"{val:.6f}",
                    },
                )

        return SafetyDecision.accept(self.name)
