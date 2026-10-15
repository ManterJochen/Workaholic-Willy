"""Gap L3.5 — driver readiness doctor: SDK probe, readiness table, fail-early startup gate, CLI.

Env-independent: the missing-SDK cases monkeypatch the vendor->SDK map so the test does not depend on
which vendor SDKs happen to be installed in the test environment.
"""

from __future__ import annotations

import contextlib
import importlib
import sys
from pathlib import Path
from typing import Iterator

import pytest

from src.robot.core.errors import RobotConnectionError
from src.robot.core.vendor import RobotVendor
from src.robot.drivers import doctor


class TestProbe:
    def test_importable_stdlib_module(self) -> None:
        st = doctor.probe_module("json")
        assert st.importable is True

    def test_absent_module(self) -> None:
        st = doctor.probe_module("definitely_not_a_real_module_xyz_123")
        assert st.importable is False
        assert st.version is None


class TestReadiness:
    def test_arm_table_covers_all_vendors(self) -> None:
        rows = doctor.arm_vendor_readiness()
        vendors = {r.vendor for r in rows}
        assert {"ur", "sim", "dummy"} <= vendors

    def test_dummy_has_no_sdk_dependency(self) -> None:
        rows = {r.vendor: r for r in doctor.arm_vendor_readiness()}
        # DUMMY carries no third-party SDK, so readiness == registered (sdks all trivially ok).
        assert rows["dummy"].ready is True
        assert rows["dummy"].sdks == ()

    def test_kuka_is_a_reserved_name_and_not_ready(self) -> None:
        """The EthernetKRL driver was deleted (2026-09-19); the name stays until the ROS 2 driver lands."""
        rows = {r.vendor: r for r in doctor.arm_vendor_readiness()}
        assert rows["kuka"].ready is False
        assert rows["kuka"].registered is False

    def test_table_renders(self) -> None:
        text = doctor.readiness_table()
        assert "vendor" in text and "ur" in text and "sim" in text


class TestStartupGate:
    def test_dummy_always_passes(self) -> None:
        doctor.require_arm_vendor_ready(RobotVendor.DUMMY)  # no raise

    def test_kuka_is_refused_as_a_reserved_slot(self) -> None:
        with pytest.raises(RobotConnectionError, match="reserved slot"):
            doctor.require_arm_vendor_ready(RobotVendor.KUKA)

    def test_sim_mock_mode_skips_sdk_probe(self) -> None:
        # mock_mode=True for the sim must NOT require isaacsim (mock drives no real device).
        doctor.require_arm_vendor_ready(RobotVendor.SIM, mock_mode=True)  # no raise

    def test_missing_sdk_raises_clear_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Pretend DUMMY needs an absent SDK -> the gate must fail early with a clear message.
        monkeypatch.setitem(doctor._ARM_VENDOR_SDKS, RobotVendor.DUMMY, ("nope_xyz_123",))
        with pytest.raises(RobotConnectionError, match="host not ready"):
            doctor.require_arm_vendor_ready(RobotVendor.DUMMY, mock_mode=False)


class TestCli:
    def test_no_require_exits_zero(self) -> None:
        assert doctor.main([]) == 0

    def test_require_dummy_exits_zero(self) -> None:
        assert doctor.main(["--require", "dummy"]) == 0

    def test_require_unknown_vendor_exits_two(self) -> None:
        assert doctor.main(["--require", "not_a_vendor"]) == 2

    def test_json_output_runs(self) -> None:
        assert doctor.main(["--json"]) == 0


#: The message Windows raised on this workstation on 2026-09-10 when an application-control policy
#: refused a DLL a present, correct package loads at import time (measured for `mujoco/plugin`, and
#: recorded in the CodeIntegrity log for `ext_deps/coal_env/Library/bin/coal.dll` the same morning).
_BLOCK = ("[WinError 4551] Eine Anwendungssteuerungsrichtlinie hat diese Datei blockiert: "
          r"'C:\site-packages\rtde_control\_rtde_control.pyd'")


@contextlib.contextmanager
def _sdk_that_is_present_and_will_not_import(tmp_path: Path, name: str) -> "Iterator[None]":
    """A real module on a real `sys.path` whose import raises what the OS raised.

    Not a mock of the probe: `find_spec` finds it (a spec is built without executing anything) and
    `import_module` fails, which is exactly the state a blocked native extension is in.
    """
    body = "raise OSError(%r)" % (_BLOCK,)
    (tmp_path / f"{name}.py").write_text(body, encoding="utf-8")
    saved = list(sys.path)
    sys.path.insert(0, str(tmp_path))
    importlib.invalidate_caches()
    try:
        yield
    finally:
        sys.path[:] = saved
        sys.modules.pop(name, None)
        importlib.invalidate_caches()


class TestPlatformIsNotAMissingInstall:
    """A driver that cannot run here says which system drives it, instead of naming a missing wheel.

    ⚠ TODAY THESE VENDORS ARE SILENT ABOUT THE SYSTEM BY ACCIDENT: franka, kuka and ros2 name no
    SDK, so their rows read "no driver registered" and nothing worse. The trap opens when P1 gives
    franka its `franky` entry, and the row on Windows would then read "franky not installed", which
    sends an operator to install a wheel that is not built for their machine. That case is exercised
    here with the entry P1 will add, rather than waited for. The owner asked on 2026-09-19 for the
    opposite of that sentence: name plainly, from the start, what runs on Linux only.

    The platform is asked as a QUESTION (`Host.running`), so both answers are measured here and
    neither depends on which box the suite runs on.
    """

    @staticmethod
    def _arm_rows(platform: str) -> dict:
        from src.robot.drivers.host import Host

        return {r.vendor: r for r in Host.running(platform).readiness().rows if r.kind == "arm"}

    def test_a_linux_only_vendor_on_windows_names_the_system(self) -> None:
        row = self._arm_rows("win32")["franka"]
        assert row.ready is False
        assert "Linux only" in row.note
        assert "this host is Windows" in row.note

    def test_the_sdk_p1_will_name_does_not_become_the_answer_on_the_wrong_system(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """⭐ THE CASE THIS MAP EXISTS FOR, with the entry P1 adds, so it is measured before P1 not after."""
        monkeypatch.setitem(doctor._ARM_VENDOR_SDKS, RobotVendor.FRANKA, ("franky",))
        row = self._arm_rows("win32")["franka"]
        assert "Linux only" in row.note
        assert "franky" not in row.note, "the operator was sent to install a wheel not built for them"
        assert "not installed" not in row.note
        assert row.sdks == (), "a module with no wheel here was probed anyway, which can only fail"

    def test_the_same_sdk_entry_is_probed_normally_on_its_own_system(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """⭐ THE CONTROL: the map suppresses the probe on the wrong system only, never everywhere."""
        monkeypatch.setitem(doctor._ARM_VENDOR_SDKS, RobotVendor.FRANKA, ("franky",))
        row = self._arm_rows("linux")["franka"]
        assert [s.module for s in row.sdks] == ["franky"]

    def test_the_same_vendor_on_linux_is_judged_on_its_driver_alone(self) -> None:
        """⭐ THE CONTROL: the platform half is a fact about the HOST, not a property stuck on the row."""
        row = self._arm_rows("linux")["franka"]
        assert "Linux only" not in row.note
        assert "reserved slot" in row.note, "on its own system it is judged like any other missing driver"

    def test_the_reserved_slot_wording_survives_on_both_systems(self) -> None:
        """One vocabulary. A table read twice is read by its sentences, not by its columns."""
        for platform in ("win32", "linux"):
            assert "reserved slot" in self._arm_rows(platform)["kuka"].note

    def test_a_vendor_this_repository_measured_no_refusal_for_carries_no_claim(self) -> None:
        """⭐ THE NEGATIVE CONTROL: the map touches only what it declares, and empty is not "nowhere"."""
        for platform in ("win32", "linux", "darwin"):
            for vendor in ("ur", "sim", "dummy"):
                row = self._arm_rows(platform)[vendor]
                assert row.platforms == ()
                assert row.runs_on(platform) is True
                assert "only" not in row.note

    def test_the_connect_refusal_reads_the_same_sentence(self) -> None:
        from src.robot.drivers.host import Host

        report = Host.running("win32").readiness()
        with pytest.raises(RobotConnectionError) as caught:
            report.require(RobotVendor.FRANKA)
        message = str(caught.value)
        assert "Linux only" in message
        assert "pip install" not in message, "nothing installable on this box fixes a system mismatch"

    def test_every_declared_platform_is_a_name_this_repository_knows(self) -> None:
        """⛔ A TYPO HERE MAKES A VENDOR RUN NOWHERE, SILENTLY. `Linux` and `win` are not sys.platform."""
        known = set(doctor._PLATFORM_NAMES)
        for vendor, platforms in doctor._ARM_VENDOR_PLATFORMS.items():
            assert isinstance(vendor, RobotVendor)
            assert platforms, f"{vendor.value} declares an empty platform set, which is 'runs nowhere'"
            assert set(platforms) <= known, f"{vendor.value} names a system no sys.platform reports: {platforms}"

    def test_the_field_reaches_the_console_rather_than_being_dropped(self) -> None:
        """The wire model takes `to_dict()` as keywords, so a field it does not declare is dropped in silence."""
        from api.schemas import VendorReadinessOut

        row = self._arm_rows("win32")["franka"]
        out = VendorReadinessOut(**row.to_dict())
        assert out.platforms == ["linux"]

class TestPresentButUnimportableSdk:
    """⛔ THE SAME DEFECT `engine_is_available` HAD FOR MuJoCo, in the driver table.

    `probe_module` asks `find_spec`, which builds a spec without running the module, so a vendor SDK
    whose native extension the OS refuses is reported importable. The table then prints
    `arm ur yes ready`, and `require_arm_vendor_ready(UR)` (the last cheap gate before a bring-up)
    waves the host through to fail inside `connect()` instead.
    """

    def test_the_probe_reports_a_blocked_module_as_importable(self, tmp_path: Path) -> None:
        with _sdk_that_is_present_and_will_not_import(tmp_path, "willy_blocked_sdk_a"):
            status = doctor.probe_module("willy_blocked_sdk_a")
        assert status.importable is False, "find_spec answered for an import that raises"

    def test_the_table_says_ready_for_a_host_that_cannot_drive_the_arm(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(doctor._ARM_VENDOR_SDKS, RobotVendor.UR, ("willy_blocked_sdk_b",))
        with _sdk_that_is_present_and_will_not_import(tmp_path, "willy_blocked_sdk_b"):
            row = {r.vendor: r for r in doctor.arm_vendor_readiness()}["ur"]
        assert row.ready is False, f"ready with a refused SDK, note was {row.note!r}"

    def test_the_startup_gate_cannot_fire_on_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The whole point of the gate: fail HERE, with the reason, not deep inside connect()."""
        monkeypatch.setitem(doctor._ARM_VENDOR_SDKS, RobotVendor.UR, ("willy_blocked_sdk_c",))
        with _sdk_that_is_present_and_will_not_import(tmp_path, "willy_blocked_sdk_c"):
            with pytest.raises(RobotConnectionError, match="host not ready"):
                doctor.require_arm_vendor_ready(RobotVendor.UR)

    def test_the_note_says_installed_and_refusing_rather_than_not_installed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A note reading "SDK not installed" sends an operator to reinstall a package they already have, which
        teaches them not to trust the message. The refusal is quoted instead."""
        monkeypatch.setitem(doctor._ARM_VENDOR_SDKS, RobotVendor.UR, ("willy_blocked_sdk_d",))
        with _sdk_that_is_present_and_will_not_import(tmp_path, "willy_blocked_sdk_d"):
            row = {r.vendor: r for r in doctor.arm_vendor_readiness()}["ur"]
        assert "not installed" not in row.note
        assert "4551" in row.note

    def test_isaac_is_still_probed_by_spec_because_importing_it_starts_a_renderer(self) -> None:
        """The asymmetry `engine_is_available` states: a cheap probe is right where the import is
        expensive. Isaac is a multi-GB install whose import takes tens of seconds and boots a
        renderer, so a readiness table that imported it is a table nobody would run."""
        assert "isaacsim" in doctor._PROBE_BY_SPEC_ONLY
