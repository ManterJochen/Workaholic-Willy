"""The driver readiness doctor.

An optional vendor SDK, such as ``ur_rtde`` for UR or ``isaacsim`` for the sim, is
imported lazily inside a driver factory or at ``connect()``, so a missing one fails
late with a cryptic error. This module probes which vendor SDKs are importable up
front and reports a per-vendor readiness table. It also exposes
:func:`require_arm_vendor_ready`, the fail-early startup gate the composition root
calls, so a misconfigured host fails with a message naming the vendor rather than deep
inside ``connect()``.

It is import-safe: it never imports a vendor SDK at module top level and probes through
``importlib.util`` instead, which keeps the import discipline and keeps heavy
dependencies out of a host that has none.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any

from src.robot.constants import DRIVER_DOCTOR_LOG_FILE, create_robot_logger
from src.robot.core.gripper_vendor import GripperVendor
from src.robot.core.vendor import RobotVendor

__all__ = [
    "SdkStatus",
    "VendorReadiness",
    "arm_vendor_readiness",
    "gripper_vendor_readiness",
    "probe_module",
    "readiness_table",
    "require_arm_vendor_ready",
]

# Vendor to the SDK modules that must import. An empty tuple means no third-party SDK,
# so the vendor is always importable. KUKA speaks EKI and KRL over a plain TCP socket,
# and dummy and NONE are pure Python.
_ARM_VENDOR_SDKS: dict[RobotVendor, tuple[str, ...]] = {
    RobotVendor.UR: ("rtde_control", "rtde_receive"),
    RobotVendor.KUKA: (),
    RobotVendor.SIM: ("isaacsim",),
    RobotVendor.DUMMY: (),
}
_GRIPPER_VENDOR_SDKS: dict[GripperVendor, tuple[str, ...]] = {
    # Robotiq needs no SDK. Naming `robotiq_gripper` here would be unsatisfiable, because
    # it is not a package: `pip index versions robotiq_gripper` answers that no matching
    # distribution was found, and the installed `ur_rtde` ships nothing by that name. It
    # is one example file from SDU's repository that an operator drops on the path by
    # hand, so such a row would read as an uninstalled SDK on every box, correctly, and
    # no install would fix it. `grippers/robotiq_socket.py` speaks the URCap port-63352
    # grammar directly.
    GripperVendor.ROBOTIQ: (),
    GripperVendor.DUMMY: (),
    GripperVendor.NONE: (),
    # Vacuum ships a real driver in grippers/vacuum.py and needs no SDK, because it
    # drives the arm digital I/O. Omitting it makes the doctor report no driver
    # registered for a driver that exists, which is a worse bug than a missing
    # capability would be.
    GripperVendor.VACUUM: (),
    # OnRobot needs no SDK either, because `grippers/onrobot_modbus.py` speaks Modbus TCP
    # to the Compute Box with `socket` and `struct`. Do not invent an SDK name for a
    # dependency-free driver: the doctor would then report the vendor as not ready on
    # every box.
    GripperVendor.ONROBOT: (),
}
# The vendors whose SDK need not be present in mock mode, because no real device is
# driven.
_MOCKABLE_ARM_VENDORS: frozenset[RobotVendor] = frozenset({RobotVendor.SIM, RobotVendor.DUMMY})

# The gate below is the last cheap moment before a cryptic late failure, so what it saw
# is worth keeping. A bring-up that dies inside connect() is diagnosed faster from a
# record that the doctor passed with these SDK versions than from the traceback alone.
logger = create_robot_logger("DriverDoctor", DRIVER_DOCTOR_LOG_FILE)


@dataclass(frozen=True, slots=True)
class SdkStatus:
    """Whether one SDK module is importable, with its version where that is discoverable."""

    module: str
    importable: bool
    version: str | None

    def to_dict(self) -> dict[str, Any]:
        return {"module": self.module, "importable": self.importable, "version": self.version}


@dataclass(frozen=True, slots=True)
class VendorReadiness:
    """Readiness of one vendor: registered as a driver, with every SDK module importable."""

    vendor: str
    kind: str  # "arm" | "gripper"
    registered: bool
    sdks: tuple[SdkStatus, ...]
    ready: bool
    note: str

    def to_dict(self) -> dict[str, Any]:
        """Plain data, safe for `json.dumps` with no custom encoder.

        It exists so that no consumer transcribes the fields. A wire model that names
        every attribute by hand, as `api/routers/diagnostics.py` would otherwise do,
        drops a field added here silently until somebody notices it missing.
        """
        # `kind` comes first. The field order of a serialiser claims nothing, and this
        # order is the one `--json` already emits.
        return {
            "kind": self.kind,
            "vendor": self.vendor,
            "registered": self.registered,
            "ready": self.ready,
            "note": self.note,
            "sdks": [sdk.to_dict() for sdk in self.sdks],
        }


def probe_module(module: str) -> SdkStatus:
    """Is ``module`` importable? Best-effort, and free of import side effects: it uses find_spec."""
    importable = False
    try:
        importable = importlib.util.find_spec(module) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        importable = False
    version: str | None = None
    if importable:
        try:
            from importlib.metadata import PackageNotFoundError, version as _pkg_version

            try:
                version = _pkg_version(module)
            except PackageNotFoundError:
                version = None
        except Exception:  # noqa: BLE001 (version detection is best-effort only)
            version = None
    return SdkStatus(module=module, importable=importable, version=version)


def _is_arm_registered(vendor: RobotVendor) -> bool:
    from src.robot.drivers.registry import is_vendor_registered

    return bool(is_vendor_registered(vendor.value))


def _readiness(vendor_value: str, kind: str, sdk_modules: tuple[str, ...], registered: bool) -> VendorReadiness:
    sdks = tuple(probe_module(m) for m in sdk_modules)
    sdks_ok = all(s.importable for s in sdks)
    ready = registered and sdks_ok
    if not registered:
        note = "no driver registered (reserved slot / not implemented)"
    elif not sdks_ok:
        missing = ", ".join(s.module for s in sdks if not s.importable)
        note = f"SDK not installed: {missing}"
    else:
        note = "ready"
    return VendorReadiness(
        vendor=vendor_value, kind=kind, registered=registered, sdks=sdks, ready=ready, note=note,
    )


def arm_vendor_readiness() -> list[VendorReadiness]:
    """Readiness for every known arm vendor. A shim over :class:`Host`."""
    from src.robot.drivers.host import Host

    return [r for r in Host.local().readiness().rows if r.kind == "arm"]


def gripper_vendor_readiness() -> list[VendorReadiness]:
    """Readiness for every known gripper vendor. A shim over :class:`Host`.

    `registered` is derived from the registry. Reading it off membership in
    `_GRIPPER_VENDOR_SDKS` instead reports a vendor missing from that dict as an
    unimplemented reserved slot whether or not a driver exists, which has happened to
    `vacuum` and to `jaw_io` while `available_gripper_vendors()` listed them.
    """
    from src.robot.drivers.host import Host

    return [r for r in Host.local().readiness().rows if r.kind == "gripper"]


def readiness_table() -> str:
    """The readiness table an operator reads. A shim over :meth:`ReadinessReport.render`."""
    from src.robot.drivers.host import Host

    return Host.local().readiness().render()


def require_arm_vendor_ready(vendor: RobotVendor, *, mock_mode: bool = False) -> None:
    """The fail-early startup gate: raise unless this host can drive ``vendor``.

    It is a query over the rows of the readiness table rather than a second definition
    of ready. A gate that asks only whether any SDK modules are missing answers no for
    franka, which has no SDK entry, and a vendor with no registered driver then sails
    through the last cheap check before a bring-up.

    It is fail-closed: a config naming franka or ros2 is refused here, by name, instead
    of failing later inside the driver factory with less context. ``mock_mode`` skips
    the SDK half for the vendors whose mock path drives no real device, and does not
    skip the registered half, because a driver that does not exist cannot be mocked
    either.
    """
    from src.robot.drivers.host import Host

    Host.local().readiness().require(vendor, mock_mode=mock_mode)


def main(argv: list[str] | None = None) -> int:
    """Print the readiness table, and with ``--require vendor`` gate on that arm vendor.

    Exit codes: ``0`` where the table printed or the required vendor is ready, ``1``
    where a required vendor is not ready, and ``2`` for bad arguments, meaning an
    unknown vendor.
    """
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="python -m src.robot.drivers.doctor",
        description="Probe which vendor SDKs are importable + report driver readiness.",
    )
    parser.add_argument("--require", metavar="VENDOR", default=None,
                        help="exit non-zero unless this arm vendor is ready (e.g. ur, sim, kuka).")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of the human table.")
    args = parser.parse_args(argv)

    # One probe, one set of rows. Building the list here and then calling
    # `readiness_table()` would build it again from scratch: two probes of the same
    # interpreter per invocation, and two chances for a table and a verdict printed side
    # by side to disagree.
    from src.robot.drivers.host import Host

    report = Host.local().readiness()
    rows = report.rows
    logger.info(
        "readiness probe: %d/%d vendors ready; not ready: %s",
        len(report.ready), len(rows),
        ", ".join(f"{r.kind}/{r.vendor} ({r.note})" for r in rows if not r.ready) or "none",
    )
    if args.json:
        # The row own `to_dict`, not six keys named by hand. Naming them by hand leaves a
        # field added to `VendorReadiness` reaching neither this CLI nor the console.
        print(json.dumps([r.to_dict() for r in rows], indent=2))
    else:
        print(report.render())

    if args.require is not None:
        try:
            vendor = RobotVendor.from_string(args.require)
        except Exception:  # noqa: BLE001 (unknown vendor string -> bad-args exit)
            print(f"unknown arm vendor: {args.require!r}")
            return 2
        return 0 if report.ready_for(vendor) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
