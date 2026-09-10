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
        assert {"ur", "kuka", "sim", "dummy"} <= vendors

    def test_dummy_and_kuka_have_no_sdk_dependency(self) -> None:
        rows = {r.vendor: r for r in doctor.arm_vendor_readiness()}
        # DUMMY/KUKA carry no third-party SDK, so readiness == registered (sdks all trivially ok).
        assert rows["dummy"].ready is True
        assert rows["dummy"].sdks == ()
        assert rows["kuka"].sdks == ()

    def test_table_renders(self) -> None:
        text = doctor.readiness_table()
        assert "vendor" in text and "ur" in text and "sim" in text


class TestStartupGate:
    def test_dummy_and_kuka_always_pass(self) -> None:
        doctor.require_arm_vendor_ready(RobotVendor.DUMMY)  # no raise
        doctor.require_arm_vendor_ready(RobotVendor.KUKA)   # no raise

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
