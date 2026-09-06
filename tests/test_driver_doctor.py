"""Gap L3.5 — driver readiness doctor: SDK probe, readiness table, fail-early startup gate, CLI.

Env-independent: the missing-SDK cases monkeypatch the vendor->SDK map so the test does not depend on
which vendor SDKs happen to be installed in the test environment.
"""

from __future__ import annotations

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
