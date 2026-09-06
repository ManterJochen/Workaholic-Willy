"""The mode-matrix harness: easy, auto and closed_loop across the m1, m2 and eih scenes, aggregated.

``SimulationApp`` is a process singleton (one Isaac session per process), so each ``(scene, mode)`` cell
runs as its own subprocess, one boot per cell. The driver holds three on-box rules:

* each child's stdout/stderr is redirected to a log file, never a pipe (a pipe destabilizes Kit's boot);
* HF offline is forced, so the vision scene reads the cached GroundingDINO/SAM2 and cannot hit the
  httpx-client-closed crash;
* a per-cell timeout records a hung boot as failed and moves on, killing the whole child tree
  (``taskkill /T`` on Windows) so a stuck Isaac never leaks RAM into the next cell.

Two ways to run:

    # 1) aggregate existing per-cell result JSONs into a matrix (pure, robust, no Isaac):
    python -m src.willy_sim.run_mode_matrix --aggregate-only logs/mode_matrix --out logs/mode_matrix/matrix.json

    # 2) run the cells then aggregate (on-box; one Isaac boot per cell):
    <isaac python> -m src.willy_sim.run_mode_matrix --run --runs 3 \
        --scenes m1,m2,eih --modes easy,auto,closed_loop --out logs/mode_matrix/matrix.json

The aggregator (:func:`matrix_from_cells`, :func:`cell_command`) is pure; the subprocess driver
(:func:`run_matrix`) is on-box only.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from src.utility.log_cfg import create_logger
from src.willy_sim.constants import MODE_MATRIX_LOG_FILE, WILLY_SIM_LOG_DIR

#: The driver's own narrative, which by construction exists in no cell's log: each cell's stdout goes
#: to its redirect, so a cell that hung, was tree-killed and recorded failed leaves nothing behind
#: that says the matrix moved on. These lines are the run's index over its children.
_LOG = create_logger("ModeMatrix", log_file=MODE_MATRIX_LOG_FILE, log_dir=WILLY_SIM_LOG_DIR)

#: Each scene names its runner module and the scene-specific extra CLI args it needs.
SCENES: dict[str, dict[str, Any]] = {
    "m1": {"module": "src.willy_sim.run_m1_pick", "extra": []},
    "m2": {"module": "src.willy_sim.run_m2_pick", "extra": ["--prompt", "a red cube"]},
    "eih": {"module": "src.willy_sim.run_eih_pick", "extra": []},
}
MODES: tuple[str, ...] = ("easy", "auto", "closed_loop")

#: The robot the runners drive when no model is named. A cell on this robot keeps its bare scene label,
#: so a matrix taken on this robot alone carries no robot prefix at all.
DEFAULT_ROBOT: str = "ur5e"


def scene_label(scene: str, robot: str = DEFAULT_ROBOT) -> str:
    """How a cell is named in the matrix: ``m1`` on the default robot, ``ur3e/m1`` on another.

    The matrix is keyed by this label, which gives the harness a robot axis without changing its
    shape: two robots appear as two rows. Nothing else guards the second robot, because the soak gate
    is robot-agnostic and synthetic and stays green through a completely broken UR3e cell. This
    harness is the only place both cells are measured together.
    """
    return scene if robot == DEFAULT_ROBOT else f"{robot}/{scene}"

#: Cells known to be limited: they run anyway, and the matrix annotates them rather than reporting a
#: silent green. closed_loop fails on all three scenes, because the two-scan image-space IoU refiner
#: returns target_lost whenever the standoff re-perceive shifts the object's image location (m1: the
#: arm occludes the no-park overhead GT cam; m2: the standoff re-perceive falls under the IoU
#: threshold; eih: the wrist cam moves between scans). The cause is the refiner, not a per-scene
#: quirk: closed_loop needs a world-space, non-occluding re-perceive. easy and auto work on all three.
KNOWN_LIMITED: dict[tuple[str, str], str] = {
    ("m1", "closed_loop"): "NOT-VIABLE: refiner re-perceive occludes the no-park overhead GT cam -> target_lost (P1.C)",
    ("m2", "closed_loop"): "NOT-VIABLE: two-scan refine IoU<threshold on the standoff re-perceive -> target_lost (P1.C)",
    ("eih", "closed_loop"): "NOT-VIABLE: the moving wrist cam breaks the image-space IoU tracker (G13/P1.C)",
}


def cell_command(
    python: str,
    scene: str,
    mode: str,
    *,
    runs: int,
    result_json: str,
    record_log: Optional[str] = None,
    debug_frames: Optional[str] = None,
    robot_model: Optional[str] = None,
    radial_closing: bool = False,
) -> list[str]:
    """Build the argv that runs one ``(scene, mode)`` cell as a subprocess.

    ``robot_model`` is omitted for the default robot, so that robot's argv never changes; naming
    another model selects that robot's config layer in the runner. ``radial_closing`` is the
    shorter-arm grasp fix: a UR3e cannot reach a tangential top-down close at any azimuth, so its
    matrix should be taken with it on.
    """

    spec = SCENES[scene]
    cmd = [python, "-m", spec["module"], "--mode", mode, "--runs", str(runs),
           "--result-json", result_json, *spec["extra"]]
    if robot_model and robot_model != DEFAULT_ROBOT:
        cmd += ["--robot-model", robot_model]
    if radial_closing:
        cmd += ["--radial-closing"]
    if record_log:
        cmd += ["--record-log", record_log]
    if debug_frames:
        cmd += ["--debug-frames", debug_frames]
    return cmd


def _median(xs: Sequence[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def matrix_from_cells(cells: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pure aggregator: turns per-cell dicts into ``{matrix: {scene: {mode: summary}}, rows: [...]}``.

    Each summary carries ``status`` (ok / timeout / no_result), ``passed``/``runs``, ``pass_rate``, the
    median measured ``lift_mm``, and any ``KNOWN_LIMITED`` note. A cell that never produced a result
    (timeout or crash) still appears, carrying its failure status; it is never silently dropped.
    """

    matrix: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for c in cells:
        scene = c.get("scene")
        mode = c.get("mode")
        passed = c.get("passed")
        runs = c.get("runs")
        lifts = [r.get("lift_mm") for r in c.get("results", []) if r.get("lift_mm") is not None]
        pass_rate = (passed / runs) if (isinstance(passed, (int, float)) and isinstance(runs, int) and runs) else None
        summary = {
            "status": c.get("status", "ok"),
            "passed": passed,
            "runs": runs,
            "pass_rate": pass_rate,
            "lift_mm_median": _median(lifts) if lifts else None,
            # The note is keyed by the bare scene: these limits are properties of the refiner, not of
            # the arm, so they hold for every robot running that scene.
            "note": KNOWN_LIMITED.get((str(scene).rsplit("/", 1)[-1], str(mode))),
        }
        matrix.setdefault(str(scene), {})[str(mode)] = summary
        rows.append({"scene": scene, "mode": mode, **summary})
    return {"matrix": matrix, "rows": rows}


def format_table(agg: dict[str, Any]) -> str:
    """A compact text table of the matrix: one row per scene and mode, with status, pass and lift."""

    lines = ["scene  mode          status         pass     lift_med  note"]
    for r in agg.get("rows", []):
        pr = "" if r["passed"] is None else f"{r['passed']}/{r['runs']}"
        lm = "" if r["lift_mm_median"] is None else f"{r['lift_mm_median']:.0f}mm"
        note = r.get("note") or ""
        lines.append(f"{str(r['scene']):<6} {str(r['mode']):<13} {str(r['status']):<14} {pr:<8} {lm:<9} {note}")
    return "\n".join(lines)


def _kill_tree(pid: int) -> None:
    """Best-effort kill of a child process tree (Windows Isaac spawns kit.exe grandchildren)."""

    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        else:  # pragma: no cover (dev box is Windows)
            os.kill(pid, 9)
    except Exception:  # noqa: BLE001 (cleanup must never raise)
        pass


def run_matrix(
    *,
    scenes: Sequence[str],
    modes: Sequence[str],
    runs: int,
    robots: Sequence[str] = (DEFAULT_ROBOT,),
    radial_closing: bool = False,
    out: str,
    logdir: str,
    timeout_s: float,
    record_dir: Optional[str] = None,
    frames_dir: Optional[str] = None,
    python: Optional[str] = None,
) -> dict[str, Any]:
    """On-box: subprocess each cell (file-redirect, HF offline, timeout+tree-kill), then aggregate."""

    python = python or sys.executable
    Path(logdir).mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    cells: list[dict[str, Any]] = []
    plan = [(r, sc, m) for r in robots for sc in scenes for m in modes]
    # One line that bounds the session: a full matrix is hours of Isaac boots, and the plan is what a
    # later reader needs to tell "this cell failed" from "this cell was never in the run".
    _LOG.info(
        "matrix: %d cell(s) planned (robots=%s scenes=%s modes=%s), %d run(s) each, timeout %.0f s, "
        "child logs -> %s", len(plan), list(robots), list(scenes), list(modes), runs, timeout_s, logdir,
    )
    for robot, scene, mode in plan:
        label = scene_label(scene, robot)
        tag = label.replace("/", "_") + f"_{mode}"
        result_json = str(Path(logdir) / f"{tag}.result.json")
        log_path = str(Path(logdir) / f"{tag}.log")
        Path(result_json).unlink(missing_ok=True)
        cmd = cell_command(
            python, scene, mode, runs=runs, result_json=result_json,
            record_log=(str(Path(record_dir) / f"{tag}.jsonl") if record_dir else None),
            debug_frames=(str(Path(frames_dir) / tag) if frames_dir else None),
            robot_model=robot, radial_closing=radial_closing,
        )
        cell: dict[str, Any] = {"scene": label, "robot": robot, "mode": mode, "log": log_path}
        print(f"[matrix] {tag}: launching (timeout {timeout_s:.0f}s) -> {log_path}", flush=True)
        # The argv at debug: it carries the per-cell flags (--radial-closing, --record-log, the robot
        # layer) that decide what the cell measured, and the child cannot report its own argv.
        _LOG.debug("cell %s argv: %s", tag, " ".join(cmd))
        _started = time.perf_counter()
        with open(log_path, "w", encoding="utf-8") as lf:
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
            try:
                rc = proc.wait(timeout=timeout_s)
                cell["returncode"] = rc
                if Path(result_json).exists():
                    cell.update(json.loads(Path(result_json).read_text(encoding="utf-8")))
                    cell["status"] = "ok"
                else:
                    cell["status"] = f"no_result(rc={rc})"
            except subprocess.TimeoutExpired:
                _kill_tree(proc.pid)
                proc.wait()
                cell["status"] = "timeout"
                # A killed cell is the one outcome whose evidence is destroyed with the tree: its own
                # log stops mid-boot with no ending, so the fact that the budget (not the pick) ended
                # it lives only here.
                _LOG.warning(
                    "cell %s: TIMEOUT after %.0f s; killed pid %d and its tree, recorded FAILED, "
                    "continuing (child log: %s)", tag, timeout_s, proc.pid, log_path,
                )
                print(f"[matrix] {tag}: TIMEOUT -> killed tree, recorded FAILED, continuing", flush=True)
        print(f"[matrix] {tag}: status={cell['status']} passed={cell.get('passed')}/{cell.get('runs')}", flush=True)
        _elapsed = time.perf_counter() - _started
        if cell["status"] == "ok":
            _LOG.info(
                "cell %s: ok in %.0f s; passed=%s/%s (rc=%s, result %s)",
                tag, _elapsed, cell.get("passed"), cell.get("runs"), cell.get("returncode"), result_json,
            )
        elif cell["status"] != "timeout":   # the timeout branch above already said what happened
            # "no result" is not the same failure as "timeout": the child exited on its own and wrote
            # nothing, which is a crash or a refusal, and the reason is in its log, named here.
            _LOG.error(
                "cell %s: status=%s after %.0f s; no result JSON at %s; the cause is in %s",
                tag, cell["status"], _elapsed, result_json, log_path,
            )
        cells.append(cell)
    return _write_and_print(cells, out)


def aggregate_only(logdir: str, out: str) -> dict[str, Any]:
    """Pure: read every ``*.result.json`` already in ``logdir`` and aggregate (no Isaac)."""

    cells = []
    for p in sorted(Path(logdir).glob("*.result.json")):
        try:
            cells.append({"status": "ok", **json.loads(p.read_text(encoding="utf-8"))})
        except Exception as exc:  # noqa: BLE001
            # A skipped file shrinks the matrix silently: the aggregate simply has one cell fewer,
            # and nothing in it says a result was dropped rather than never taken.
            _LOG.warning("aggregate: skipping unreadable result %s (%s)", p, exc)
            print(f"[matrix] skip {p.name}: {exc}", flush=True)
    return _write_and_print(cells, out)


def _write_and_print(cells: Sequence[dict[str, Any]], out: str) -> dict[str, Any]:
    agg = matrix_from_cells(cells)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    _text = json.dumps(agg, indent=2, sort_keys=True)
    Path(out).write_text(_text, encoding="utf-8")
    _LOG.info(
        "matrix written -> %s (%d bytes, %d cell(s): %s)", out, len(_text.encode("utf-8")), len(cells),
        ", ".join(f"{r['scene']}/{r['mode']}={r['status']}" for r in agg.get("rows", [])) or "none",
    )
    print("\n" + format_table(agg), flush=True)
    print(f"\n[matrix] written -> {out}", flush=True)
    return agg


def main() -> None:
    ap = argparse.ArgumentParser(description="Willy mode-matrix harness (EASY/AUTO/CLOSED_LOOP x M1/M2/EIH).")
    ap.add_argument("--run", action="store_true", help="run the cells as subprocesses (on-box); else aggregate-only")
    ap.add_argument("--aggregate-only", type=str, default=None, metavar="LOGDIR",
                    help="skip running; aggregate the *.result.json already in LOGDIR")
    ap.add_argument("--robots", type=str, default=DEFAULT_ROBOT,
                    help=f"comma-separated robot models to run every cell on (default: {DEFAULT_ROBOT}). "
                         "Cells on a non-default robot are labelled '<robot>/<scene>', so a single-robot "
                         "matrix is unchanged. NOTHING else guards a second robot: the U12 soak gate is "
                         "robot-agnostic and synthetic and stays green through a completely broken cell.")
    ap.add_argument("--radial-closing", action="store_true",
                    help="aim a symmetric object's jaw along the RADIAL axis. A shorter arm (UR3e) cannot "
                         "reach a TANGENTIAL top-down close at any azimuth, so its matrix needs this on.")
    ap.add_argument("--scenes", type=str, default="m1,m2,eih")
    ap.add_argument("--modes", type=str, default="easy,auto,closed_loop")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", type=str, default="logs/mode_matrix/matrix.json")
    ap.add_argument("--logdir", type=str, default="logs/mode_matrix")
    ap.add_argument("--timeout-s", type=float, default=600.0, help="per-cell wall-clock budget")
    ap.add_argument("--record-dir", type=str, default=None, help="P0: per-cell GraspAttemptRecord JSONL dir")
    ap.add_argument("--frames-dir", type=str, default=None, help="P0: per-cell grasp-viewer PNG dir")
    ap.add_argument("--python", type=str, default=None, help="python to launch cells with (default: this one)")
    args = ap.parse_args()

    if args.aggregate_only:
        aggregate_only(args.aggregate_only, args.out)
        return
    if not args.run:
        ap.error("pass --run to launch cells, or --aggregate-only LOGDIR to aggregate existing results")
    run_matrix(
        scenes=[s.strip() for s in args.scenes.split(",") if s.strip()],
        modes=[m.strip() for m in args.modes.split(",") if m.strip()],
        robots=[r.strip() for r in args.robots.split(",") if r.strip()],
        radial_closing=args.radial_closing,
        runs=args.runs, out=args.out, logdir=args.logdir, timeout_s=args.timeout_s,
        record_dir=args.record_dir, frames_dir=args.frames_dir, python=args.python,
    )


if __name__ == "__main__":
    main()
