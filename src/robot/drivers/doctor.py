"""The driver readiness doctor.

An optional vendor SDK, such as ``ur_rtde`` for UR or ``isaacsim`` for the sim, is
imported lazily inside a driver factory or at ``connect()``, so a missing one fails
late with a cryptic error. This module probes which vendor SDKs are importable up
front and reports a per-vendor readiness table. It also exposes
:func:`require_arm_vendor_ready`, the fail-early startup gate the composition root
calls, so a misconfigured host fails with a message naming the vendor rather than deep
inside ``connect()``.

It is import-safe: it never imports a vendor SDK at module top level, which keeps the
import discipline and keeps heavy dependencies out of a host that has none. The probe
itself does import, inside the function, when someone asks, because that is the only way
to learn whether a present SDK actually loads. See :func:`probe_module`. Isaac is the one
exception and is still resolved by spec.
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
# so the vendor is always importable; dummy and NONE are pure Python.
_ARM_VENDOR_SDKS: dict[RobotVendor, tuple[str, ...]] = {
    RobotVendor.UR: ("rtde_control", "rtde_receive"),
    RobotVendor.SIM: ("isaacsim",),
    RobotVendor.DUMMY: (),
}


def _gripper_vendor_sdks() -> dict[GripperVendor, tuple[str, ...]]:
    """Gripper vendor to the SDK modules its driver imports, read off each driver's spec (P0g slice 8).

    Read from the specs rather than kept here: a table kept here once missed ``vacuum`` and then ``jaw_io``, because
    nobody remembered to add a line. A vendor with no spec (a reserved name) has no row, and the readiness reads
    ``()`` for it as before.
    """
    from src.robot.grippers import gripper_drivers  # noqa: PLC0415

    return {driver.vendor: tuple(driver.sdk_modules) for driver in gripper_drivers()}


# The vendors whose SDK need not be present in mock mode, because no real device is
# driven.
_MOCKABLE_ARM_VENDORS: frozenset[RobotVendor] = frozenset({RobotVendor.SIM, RobotVendor.DUMMY})

#: Which systems a vendor's driver runs on, by ``sys.platform`` name. A vendor absent from this map carries no
#: platform claim, which is not the same as running everywhere: the repository says nothing, and the row reads as it
#: did before this map existed.
#:
#: "SDK not installed" is the wrong sentence for a driver that cannot be installed here: it sends an operator to
#: ``pip install`` something that has no wheel for their system. So the doctor says plainly what runs only on Linux.
#: The entries are the measured restrictions of aurora's vendor research of 2026-09-19, and only those. UR and the two
#: simulated arms carry no entry, because nothing measured a system they refuse.
_ARM_VENDOR_PLATFORMS: dict[RobotVendor, frozenset[str]] = {
    # franky/libfranka publish Linux wheels only, and a real cell wants PREEMPT_RT besides.
    RobotVendor.FRANKA: frozenset({"linux"}),
    # KUKA returns over ROS 2 only, and `kuka_drivers` is an apt package.
    RobotVendor.KUKA: frozenset({"linux"}),
    # `rclpy` is built for the distribution's Python on Linux; there is no Windows Jazzy.
    RobotVendor.ROS2: frozenset({"linux"}),
}

#: How to say a ``sys.platform`` to a person. An unknown value is printed as it is, rather than guessed.
_PLATFORM_NAMES: dict[str, str] = {"win32": "Windows", "linux": "Linux", "darwin": "macOS"}


def platform_name(platform: str) -> str:
    """``sys.platform`` as a person writes it, or the raw value where this repository has no name."""
    return _PLATFORM_NAMES.get(platform, platform)


# The gate below is the last cheap moment before a cryptic late failure, so what it saw
# is worth keeping. A bring-up that dies inside connect() is diagnosed faster from a
# record that the doctor passed with these SDK versions than from the traceback alone.
logger = create_robot_logger("DriverDoctor", DRIVER_DOCTOR_LOG_FILE)


@dataclass(frozen=True, slots=True)
class SdkStatus:
    """Whether one SDK module imports, with its version where discoverable and why it did not."""

    module: str
    importable: bool
    version: str | None
    #: Why it is not importable. Empty means it is not installed, the ordinary and expected
    #: absence. Non-empty means the module is there and refused, and the text is whatever
    #: refused, quoted.
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"module": self.module, "importable": self.importable, "version": self.version,
                "detail": self.detail}


@dataclass(frozen=True, slots=True)
class VendorReadiness:
    """Readiness of one vendor: registered as a driver, with every SDK module importable."""

    vendor: str
    kind: str  # "arm" | "gripper"
    registered: bool
    sdks: tuple[SdkStatus, ...]
    ready: bool
    note: str
    #: The systems this driver runs on, by ``sys.platform`` name, sorted. Empty means no claim: this
    #: repository has measured no system that refuses it (:data:`_ARM_VENDOR_PLATFORMS`), not that it
    #: runs nowhere.
    platforms: tuple[str, ...] = ()

    def runs_on(self, platform: str) -> bool:
        """Whether this driver runs on ``platform``. True wherever no claim is made."""
        return not self.platforms or platform in self.platforms

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
            "platforms": list(self.platforms),
            "sdks": [sdk.to_dict() for sdk in self.sdks],
        }


#: Probed by spec only, because importing these is not a probe. ``isaacsim`` is a multi-GB
#: install whose import takes tens of seconds and starts a renderer, and a readiness table
#: that did that is a table nobody would run before powering a cell. Everything else here is
#: a thin client library that imports in milliseconds, measured 2026-09-10 on this box at
#: 0.00 s each for ``rtde_control`` and ``rtde_receive``, so the honest probe costs nothing
#: worth saving.
_PROBE_BY_SPEC_ONLY: frozenset[str] = frozenset({"isaacsim"})


def probe_module(module: str) -> SdkStatus:
    """Does ``module`` actually import on this host?

    It is imported rather than looked up, and that is the defect this closes. ``find_spec``
    builds a spec without executing anything, so a package whose native extension the OS
    refuses answers yes. Measured 2026-09-10 on this box: ``find_spec("mujoco")`` said yes
    while ``import mujoco`` raised ``OSError: [WinError 4551] Eine
    Anwendungssteuerungsrichtlinie hat diese Datei blockiert``. In this table that reads
    ``arm ur yes ready`` for a host that cannot open a UR session, and
    :func:`require_arm_vendor_ready`, the last cheap gate before a bring-up, cannot fire on
    it. The identical repair landed in ``datagen/render/engine.py``'s ``engine_is_available``
    first, and the asymmetry travelled with it, see :data:`_PROBE_BY_SPEC_ONLY`: a cheap probe
    is right exactly where the import is expensive.
    """
    importable, detail = _import_probe(module)
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
    return SdkStatus(module=module, importable=importable, version=version, detail=detail)


def _import_probe(module: str) -> tuple[bool, str]:
    """``(importable, why not)``. An empty reason means the ordinary "it is not installed"."""
    if module in _PROBE_BY_SPEC_ONLY:
        try:
            found = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError, ModuleNotFoundError):
            found = False
        return found, ""
    try:
        importlib.import_module(module)
    except ModuleNotFoundError as exc:
        # ``exc.name`` separates two opposite answers that both arrive as
        # ModuleNotFoundError. The SDK itself being absent is "not installed", while the SDK
        # being present and failing on a dependency it imports is an installed, broken
        # package, and telling that operator to pip install what they already have teaches
        # them not to trust the message.
        if exc.name in (module, None):
            return False, ""
        return False, f"installed and will not import: {type(exc).__name__}: {exc}"
    except Exception as exc:  # noqa: BLE001 (report whatever refused, do not classify it)
        # A blocked DLL, a broken build, an ABI mismatch against numpy. The cause is quoted
        # rather than named, because the remedy differs for every one of them.
        return False, f"installed and will not import: {type(exc).__name__}: {exc}"
    return True, ""


def _is_arm_registered(vendor: RobotVendor) -> bool:
    from src.robot.drivers.registry import is_vendor_registered

    return bool(is_vendor_registered(vendor.value))


def _readiness(
    vendor_value: str,
    kind: str,
    sdk_modules: tuple[str, ...],
    registered: bool,
    platforms: tuple[str, ...] = (),
    platform: str = "",
) -> VendorReadiness:
    """One row. ``platform`` is the ``sys.platform`` the question is asked about, empty to skip it.

    The platform is decided before the SDK is probed, and the probe is then not run at all. A module
    with no wheel for this system can only answer "not installed", which is true and useless: it names
    a repair that does not exist here. The row says what the operator can act on, which is that this
    driver is driven from another system.
    """
    runs_here = not platforms or not platform or platform in platforms
    if not runs_here:
        where = " or ".join(platform_name(p) for p in platforms)
        # The reserved-slot wording is kept as it is where it still applies: an operator who has read
        # this table once reads the sentence, not the row, and two words for one state stop a table
        # being read.
        head = (f"no driver registered (reserved slot / not implemented), and when one lands it "
                f"drives from {where} only" if not registered else f"drives from {where} only")
        return VendorReadiness(
            vendor=vendor_value, kind=kind, registered=registered, sdks=(), ready=False,
            note=f"{head}; this host is {platform_name(platform)}", platforms=platforms,
        )
    sdks = tuple(probe_module(m) for m in sdk_modules)
    sdks_ok = all(s.importable for s in sdks)
    ready = registered and sdks_ok
    if not registered:
        note = "no driver registered (reserved slot / not implemented)"
    elif not sdks_ok:
        # "SDK not installed" was the only sentence this could say, and it is the wrong one
        # for a present SDK the OS refuses. The per-module reason is carried through instead,
        # so the row that fails a bring-up also names what to do about it.
        note = "SDK not usable: " + "; ".join(
            f"{s.module} ({s.detail})" if s.detail else f"{s.module} not installed"
            for s in sdks if not s.importable
        )
    else:
        note = "ready"
    return VendorReadiness(
        vendor=vendor_value, kind=kind, registered=registered, sdks=sdks, ready=ready, note=note,
        platforms=platforms,
    )


def arm_vendor_readiness() -> list[VendorReadiness]:
    """Readiness for every known arm vendor. A shim over :class:`Host`."""
    from src.robot.drivers.host import Host

    return [r for r in Host.local().readiness().rows if r.kind == "arm"]


def gripper_vendor_readiness() -> list[VendorReadiness]:
    """Readiness for every known gripper vendor. A shim over :class:`Host`.

    `registered` is derived from the registry. Reading it off membership in the SDK
    table, a dict kept by hand until P0g slice 8, reported a vendor missing from it as an
    unimplemented reserved slot whether or not a driver existed, which happened to
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
                        help="exit non-zero unless this arm vendor is ready (e.g. ur, sim, dummy).")
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
