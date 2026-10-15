"""UR driver connection config schema."""

from __future__ import annotations

from pydantic import Field, field_validator

from .._base import StrictModel


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
    #: The controller's address, with no default (aurora's P0c, 2026-10-13): it used to be ``192.168.1.100`` and could
    #: not be emptied, so every tree that never had an address typed into it carried one that looked real, and the cell
    #: lock, keyed on it, printed that placeholder as if it were the arm in the room. A cell says where its controller
    #: is, or it is not a cell that can drive one. The owner's base tree names the UR10 cell's own address.
    ip: str | None = Field(default=None, min_length=1)
    #: RTDE update rate in Hz. ``0.0``, the default, reads as not configured: let the controller
    #: pick its own rate, 500 Hz on the e-Series. That is a local convention and not ur_rtde's,
    #: whose sentinel for the same thing is -1.0 and which takes 0.0 as literally zero hertz and
    #: fails synchronisation. Against URSim 5.26.0, 0.0 failed in 6.1 s while both -1.0 and 500.0
    #: connected, so the UR connection translates 0.0 -> -1.0 at the driver boundary.
    rtde_frequency: float = Field(default=0.0, ge=0.0)
    #: What "halt now" in the console (``POST /v1/cell/brake``) does to a move already in flight. Off, as shipped (the
    #: owner's answer of 2026-09-30): every move is sent exactly as before, one synchronous ``moveJ`` or ``moveL`` that
    #: returns at its end, and a halt latches the arm, so the move in flight ends as it would have and nothing after it
    #: is sent, no output switched either. On: every move is sent asynchronously and watched by the thread that sent
    #: it, which brakes it under control with ``stopJ`` or ``stopL`` on a halt, at max(2.0, the move's own
    #: acceleration), and answers ``True`` only once the arm stands at the target. Either way "halt now" is not an
    #: emergency stop; the red button is. Switch it on in the cell's own profile only after
    #: ``scripts/ursim/probe_halt.py`` has passed against URSim.
    brake_on_halt: bool = False

    @field_validator("model")
    @classmethod
    def _known_model(cls, v: str) -> str:
        """Reject a typo at config load (<1 s) rather than at the moment an arm is asked to move."""
        # Admission is read from the arm registry (P0d D3f). Lazily, because the registry's loader imports this schema,
        # and at validation time both are loaded. The repository's registry is the authority a deployment tree may only
        # repeat (src.config.arms.tree_arm_refusal), so it is the one asked here. Only the UR's own descriptions (P0g
        # slice 11): a second vendor's arm admitted beside them is not a UR model.
        from src.config.arms import admitted_arms  # noqa: PLC0415

        admitted = admitted_arms(vendor="ur")
        if v not in admitted:
            raise ValueError(
                f"unknown UR model {v!r}; supported: {admitted}. A cell may name an arm whose description in "
                f"config/arms/ says admitted: true"
            )
        return v
