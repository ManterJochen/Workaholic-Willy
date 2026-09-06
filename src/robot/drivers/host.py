"""Is this machine ready to drive a cell: which vendor drivers exist here, and which SDKs import.

Readiness has to have one definition. Two of them disagree in both directions: a table
whose `ready` means registered and every SDK importable, and a gate whose `ready` means
every SDK importable, with a mock-mode escape the table has no concept of. Measured:

    require_arm_vendor_ready(franka)             passes     `--require franka`  exits 1
    require_arm_vendor_ready(SIM, mock_mode=True) passes     `--require sim`     exits 1

The first is the dangerous one. Franka has no registered driver and no SDK entry, so no
modules are missing is trivially true and a startup gate reading only that waves it
through; the build then fails later, deeper, with a worse message. The second is the
gate being right and the terminal having no way to say that mock mode is intended. One
set of rows answers both questions here, and `require()` is a query over those rows
rather than a second implementation.

The gripper half is derived rather than hand-kept for the same reason. Reading
`registered` off membership in `_GRIPPER_VENDOR_SDKS` reports any vendor missing from
that dict as having no driver, whether or not one exists, and that has already happened
twice: `vacuum` was added to the dict to fix it, and `jaw_io` then landed without a line
and read as unregistered, while `available_gripper_vendors()` sat one import away.
`registered` comes from the registries now, so the next vendor cannot repeat it.

It is the cheapest question in the stack and it stays that way: no `connect`, no
`build`, no network, just `find_spec` and two registry lookups. Its value is that an
operator can ask it before anything is powered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.robot.core.vendor import RobotVendor
    from src.robot.drivers.doctor import VendorReadiness

__all__ = ["Host", "ReadinessReport"]


@dataclass(frozen=True, slots=True)
class ReadinessReport:
    """Every arm and gripper vendor this checkout knows, and whether this host can drive it."""

    #: Composed once. A caller that builds this list and then asks for it again probes the
    #: same interpreter twice per invocation, and gives a table and a verdict printed
    #: together two chances to disagree.
    rows: tuple["VendorReadiness", ...]

    @property
    def ready(self) -> tuple["VendorReadiness", ...]:
        return tuple(r for r in self.rows if r.ready)

    @property
    def exit_code(self) -> int:
        """Always 0, because a readiness table is an answer and not a gate.

        The verdict lives on `require`. That is the CLI contract too: `doctor` with no
        arguments prints the table and exits 0 even where nothing on the host is ready,
        because a box with no vendor SDKs is a fact an operator asked for rather than a
        failure.
        """
        return 0

    # --- the gate, as a query over these same rows ---------------------------------------------

    def row_for(self, kind: str, vendor: str) -> "VendorReadiness | None":
        return next((r for r in self.rows if r.kind == kind and r.vendor == vendor), None)

    def ready_for(self, vendor: "RobotVendor | str", *, mock_mode: bool = False) -> bool:
        """Can this host drive that arm vendor?

        One definition, read by both doors. The row `ready` means registered and every
        SDK importable, and `mock_mode` skips the SDK half for the vendors whose mock
        path drives no real device.

        It does not skip the `registered` half. A vendor with no driver cannot be built
        in mock mode either, and pretending otherwise is what passes franka.
        """
        from src.robot.drivers.doctor import _MOCKABLE_ARM_VENDORS  # noqa: PLC0415

        name = str(getattr(vendor, "value", vendor))
        row = self.row_for("arm", name)
        if row is None:
            return False
        if not row.registered:
            return False
        if mock_mode and any(name == str(v.value) for v in _MOCKABLE_ARM_VENDORS):
            return True
        return row.ready

    def require(self, vendor: "RobotVendor | str", *, mock_mode: bool = False) -> None:
        """Raise unless this host can drive that arm vendor. This is the fail-early startup gate.

        The message names the missing modules and what to do, because this is the last
        cheap moment before a bring-up dies inside `connect()` with less context.
        """
        from src.robot.core.errors import RobotConnectionError  # noqa: PLC0415

        name = str(getattr(vendor, "value", vendor))
        if self.ready_for(vendor, mock_mode=mock_mode):
            return
        row = self.row_for("arm", name)
        if row is not None and not row.registered:
            raise RobotConnectionError(
                f"host not ready for arm vendor {name!r}: no driver is registered for it. It is a "
                f"reserved slot in the vendor enum, not an implemented driver. "
                f"Run `python -m src.robot.drivers.doctor` for the full table."
            )
        missing = [s.module for s in (row.sdks if row else ()) if not s.importable]
        raise RobotConnectionError(
            f"host not ready for arm vendor {name!r}: missing SDK module(s) {missing}. "
            "Install the vendor extra (e.g. `pip install ur_rtde`, or the Isaac bundled python for "
            "the sim) and re-run, or `python -m src.robot.drivers.doctor` for the full table."
        )

    # --- the report halves ---------------------------------------------------------------------

    def render(self) -> str:
        """The readiness table. ASCII, no trailing newline, no arguments."""
        lines = [f"{'kind':<8}{'vendor':<14}{'ready':<7}note"]
        for r in self.rows:
            lines.append(f"{r.kind:<8}{r.vendor:<14}{'yes' if r.ready else 'NO':<7}{r.note}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Plain data, a pure view over the `to_dict` of each row.

        Transcribing the fields by hand instead, six keys per row plus three per SDK,
        leaves a field added to the dataclass reaching neither surface.
        """
        return {"rows": [r.to_dict() for r in self.rows]}


@dataclass(frozen=True, slots=True)
class Host:
    """This machine, as something that can be asked whether it could drive a cell.

        from src.robot.drivers.host import Host

        report = Host.local().readiness()
        print(report.render())
        report.require("ur")          # raises with the missing modules named

    It is a `Host` and not a `DriverDoctor`, because an operator asks whether this host
    is ready and never asks to run a doctor object. A single noun owning both this and
    the digital-I/O bench would answer whether the box can import ur_rtde and which pin
    closes the jaws behind one name, which is a runner in a doctor coat.
    """

    #: `None` means the maps this checkout ships, read at probe time rather than at
    #: construction. A caller that swaps an entry in those module-level dicts is
    #: therefore still seen, where freezing a copy here would ignore it silently.
    arm_sdks: "Mapping[Any, Sequence[str]] | None" = None
    gripper_sdks: "Mapping[Any, Sequence[str]] | None" = None

    @classmethod
    def local(cls) -> "Host":
        """This interpreter, with the vendor maps this checkout ships."""
        return cls()

    @classmethod
    def from_sdk_map(
        cls,
        *,
        arm_sdks: "Mapping[Any, Sequence[str]]",
        gripper_sdks: "Mapping[Any, Sequence[str]]",
    ) -> "Host":
        """A hypothetical host: what a box with these SDKs would be able to drive.

        It is the plain-Python door, and answering the question without mutating a
        module global is the point.
        """
        return cls(arm_sdks=dict(arm_sdks), gripper_sdks=dict(gripper_sdks))

    def readiness(self) -> ReadinessReport:
        """Probe the host: `find_spec` per SDK module and two registry lookups, importing nothing."""
        from src.robot.core.gripper_vendor import GripperVendor  # noqa: PLC0415
        from src.robot.core.vendor import RobotVendor  # noqa: PLC0415
        from src.robot.drivers import doctor  # noqa: PLC0415

        arm_map = self.arm_sdks if self.arm_sdks is not None else doctor._ARM_VENDOR_SDKS
        grip_map = (
            self.gripper_sdks if self.gripper_sdks is not None else doctor._GRIPPER_VENDOR_SDKS
        )

        rows: list[VendorReadiness] = [
            doctor._readiness(
                v.value, "arm", tuple(arm_map.get(v, ())), doctor._is_arm_registered(v)
            )
            for v in RobotVendor
        ]
        # Derived from the registry rather than from membership in the SDK map. That
        # membership check reports a driver as unregistered for as long as nobody
        # remembers to add a line to a dict, which has happened to `vacuum` and to
        # `jaw_io`.
        registered_grippers = _registered_gripper_names()
        rows.extend(
            doctor._readiness(
                v.value, "gripper", tuple(grip_map.get(v, ())), v.value in registered_grippers
            )
            for v in GripperVendor
        )
        return ReadinessReport(rows=tuple(rows))


def _registered_gripper_names() -> frozenset[str]:
    """Which gripper vendors have a factory, read from the registry that owns that fact."""
    from src.robot.grippers.registry import available_gripper_vendors  # noqa: PLC0415

    return frozenset(str(getattr(v, "value", v)) for v in available_gripper_vendors())
