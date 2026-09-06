"""R8.2 Seam-0 pin: GateResult.to_dict() reproduces each runner's exact legacy bare-dict.

The runner ``run_gate`` returns were heterogeneous bare dicts consumed by ``write_run_result``
(``{**meta, **dict(result)}`` with ``sort_keys=True``) and, via the persisted ``--result-json``,
by the mode-matrix subprocess aggregator. This locks that ``GateResult.to_dict()`` emits the same
key set per runner so the persisted artifact stays byte-identical, plus the shared gate math.
"""

from __future__ import annotations

from src.willy_sim.harness.gate import GateResult, gate_passed


def test_to_dict_m1_eih_key_set() -> None:
    # m1 / eih: {runs, passed, gate_passed, results}
    res = [{"run": 0, "succeeded": True, "lift_mm": 99.6, "passed": True}]
    gr = GateResult(runs=10, passed=10, results=res, gate_passed=True)
    assert gr.to_dict() == {"runs": 10, "passed": 10, "gate_passed": True, "results": res}


def test_to_dict_m2_omits_gate_passed() -> None:
    # m2 scores on lift only -> {runs, passed, results} (NO gate_passed)
    res = [{"run": 0, "succeeded": True, "lift_mm": 80.0, "passed": True}]
    gr = GateResult(runs=10, passed=8, results=res)
    assert gr.to_dict() == {"runs": 10, "passed": 8, "results": res}
    assert "gate_passed" not in gr.to_dict()


def test_to_dict_dense_adds_target() -> None:
    # dense: {runs, passed, gate_passed, target, results}
    res = [{"run": 0, "passed": True, "lift_mm": 82.0}]
    gr = GateResult(runs=5, passed=5, results=res, gate_passed=True, target="the red cube")
    assert gr.to_dict() == {
        "runs": 5, "passed": 5, "gate_passed": True, "target": "the red cube", "results": res,
    }


def test_to_dict_fused_clutter_adds_distractor_lifts() -> None:
    # fused clutter gate: {runs, passed, distractor_lifts, gate_passed, results}
    res = [{"run": 0, "passed": True}]
    gr = GateResult(runs=5, passed=4, results=res, gate_passed=False, distractor_lifts=1)
    d = gr.to_dict()
    assert d == {"runs": 5, "passed": 4, "gate_passed": False, "distractor_lifts": 1, "results": res}


def test_gate_passed_math() -> None:
    # n_pass >= max(1, int(pass_fraction * runs))
    assert gate_passed(8, 10, 0.8) is True       # 8 >= max(1, int(8.0)) = 8
    assert gate_passed(7, 10, 0.8) is False      # 7 >= 8 -> False
    assert gate_passed(1, 10, 0.0) is True        # max(1, 0) = 1
    assert gate_passed(0, 10, 0.0) is False
    assert gate_passed(5, 5, 1.0) is True
