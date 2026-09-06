"""71: a watchable MP4 of a run. The machine is the requirement, not the cell.

Isaac Sim with its bundled interpreter, the cuRobo planner and OpenCV's video writer are what
this needs. No robot, camera or gripper is involved, and no path in this file reaches one.

    python scripts/examples/sim/71_record_a_demo.py                    # check the box, film nothing
    <isaac-sim>\\python.bat scripts/examples/sim/71_record_a_demo.py --record eih
    <isaac-sim>\\python.bat scripts/examples/sim/71_record_a_demo.py --record clearing --fps 24

The decision is a clip or a number, and the two are not interchangeable. A gate perceives, picks
and then scores itself against the scene's ground truth, repeats that N times, and returns a
rate. A recorder films one run from a cinematic camera and scores nothing: it records the pick
that happened, whether it worked or not, and the resulting file contains no claim at all. So
handing a video to somebody who asked whether the cell works answers a different question than
the one they asked. 70_sim_pick.py is the number. This is the picture.

Take the picture when the audience is a person and the thing to convey is what the cell does:
which tool it chose, where it reached, whether the motion looks like a machine that knows what it
is doing. Take the number when the question is whether it works. A clip is worth recording
because motion is hard to describe and trivial to show, and because a reviewer watching a run
notices things no gate scores, a flinch before the grasp, a path that grazes a wall.

What the picture costs. One Isaac boot and minutes of wall clock, for exactly one run. A file:
1280x720 frames encoded with the mp4v fourcc, plus keyframe PNGs written alongside for stills.
And the cuRobo planner, which is a hard requirement rather than a preference here: without it the
arm plans with blind IK, whose paths visibly flicker, and one of the two recorders below refuses
to start rather than film that. The other reaches the same refusal one layer down, through the
shared sim bootstrap, so neither will quietly hand you a clip of a degraded stack.

Frames per second here is playback speed, not capture rate. Capture is `--capture-every`, which
takes one frame in every N simulator steps and trades smoothness against file size and encode
time; `--fps` then decides how fast those frames are played back, and lower is easier to watch
because the simulator runs faster than an eye follows. The defaults are the recorder's own.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import EXIT_FAILED, Example, not_ready  # noqa: E402


def _record_eih(out: str, *, headless: bool, fps: int, capture_every: int) -> dict[str, Any]:
    """One eye-in-hand pick, filmed from a fixed cinematic three-quarter view."""
    from src.willy_sim.run_eih_demo import record_demo

    # `out_path`, and the hold is `hold_steps`. The other recorder takes `out` positionally and
    # calls the same idea `hold`. The two signatures are read off the two functions rather than
    # assumed from each other, which is why this dispatch is two small functions and not one
    # table of keyword names.
    return record_demo(out_path=out, headless=headless, fps=fps, capture_every=capture_every)


def _record_clearing(out: str, *, headless: bool, fps: int, capture_every: int) -> dict[str, Any]:
    """A source bin emptied object by object, with the tool choice drawn on each object."""
    from src.willy_sim.run_bin_clearing_demo import record_clearing

    # `max_picks` and `prompt` are left at their defaults. `prompt` is the language-driven mode:
    # it runs the detector over the overhead frame and clears only the objects that match, which
    # needs the detector weights already in the local HuggingFace cache and forces the detector
    # onto the CPU so it does not starve the renderer's depth annotator. That is a second set of
    # preconditions for a second subject, so it belongs in its own example rather than behind a
    # flag here.
    return record_clearing(out, headless=headless, fps=fps, capture_every=capture_every)


@dataclass(frozen=True, slots=True)
class Recorder:
    """One recorder this example fronts, and what its file will contain."""

    call: Callable[..., dict[str, Any]]
    subject: str
    filmed: str
    #: The recorder's own default output path, so nothing here invents a second one.
    default_out: str
    #: Stills the recorder writes next to the MP4, evenly spaced across the clip.
    keyframes: int
    #: The recorder's own default playback fps.
    default_fps: int


RECORDERS: dict[str, Recorder] = {
    "eih": Recorder(_record_eih, "one eye-in-hand pick",
                    "the wrist camera approaches, perceives from where it moved, grasps and lifts",
                    "logs/demo/eih_pick_demo.mp4", keyframes=4, default_fps=18),
    "clearing": Recorder(_record_clearing, "a bin cleared, object by object",
                         "per-object tool choice as a floating tag, then suction or jaw, until "
                         "only the rejects are left",
                         "logs/demo/bin_clearing_demo.mp4", keyframes=6, default_fps=24),
}

EPILOG = """
exit codes
  0  a clip was recorded and the file is on disk
  1  the recorder ran and produced nothing readable
  2  the box is not ready: no Isaac in this interpreter, no OpenCV, or a motion engine the
     recorder refuses to film without

Isaac's python.bat does not pass these through. It reports its own status, and every non-zero
code arrives at the shell as 1.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "", epilog=EPILOG,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record", choices=sorted(RECORDERS), default=None,
                        help="which demo to film. Omitted: check the box and stop")
    parser.add_argument("--out", default=None,
                        help="where the MP4 goes. Default: the recorder's own path")
    parser.add_argument("--fps", type=int, default=None,
                        help="playback speed. Default: the recorder's own, which is slower than "
                             "the simulator on purpose")
    parser.add_argument("--capture-every", type=int, default=1,
                        help="film one frame in every N simulator steps (default 1)")
    parser.add_argument("--gui", action="store_true",
                        help="also open the live window. The clip is filmed from the cinematic "
                             "camera either way, so this costs frame rate and shows nothing extra")
    args = parser.parse_args(argv)

    # `hardware=False` for the same reason as 70: `Example.hardware` asks whether this file can
    # touch a robot, and the sim config builds the Isaac driver and nothing else. Announcing a
    # rehearsal would claim something was held back that never existed.
    with Example("71 record_a_demo", args.record or "readiness check", hardware=False) as run:
        with run.step("is Isaac importable in THIS interpreter") as report:
            # `find_spec` rather than an import: importing Isaac takes tens of seconds and starts
            # a renderer, and the question here is only whether it could be imported.
            isaac = importlib.util.find_spec("isaacsim") is not None
            report("yes" if isaac else "no, this is the repository venv rather than Isaac's")
        with run.step("is there a video writer") as report:
            # OpenCV writes the file. It is a plain pip dependency, so it is usually present and
            # is checked anyway: a recorder that boots Isaac, films a full run and then discovers
            # it cannot encode has spent minutes to reach an ImportError.
            opencv = importlib.util.find_spec("cv2") is not None
            report("cv2 present, mp4v fourcc" if opencv else "no cv2, nothing can encode the clip")

        if not isaac:
            return not_ready(
                "Isaac Sim is not importable in this interpreter",
                r"run this file with Isaac's own Python: "
                r"<isaac-sim>\python.bat scripts/examples/sim/71_record_a_demo.py --record eih")
        if not opencv:
            return not_ready("OpenCV is not importable, so no file can be written",
                             "pip install -r requirements.txt into this interpreter")

        run.note("")
        for key, recorder in sorted(RECORDERS.items()):
            run.note(f"    {key:9} {recorder.subject}: {recorder.filmed}")
            run.note(f"    {'':9} lands at {recorder.default_out}, plus {recorder.keyframes} "
                     f"keyframe PNGs, at {recorder.default_fps} fps")
        run.note("")

        if args.record is None:
            run.note("Pick one with --record. Neither scores anything: for a rate rather than a")
            run.note("clip, 70_sim_pick.py runs the same scenes through their gates.")
            return run.exit_code

        chosen = RECORDERS[args.record]
        out = args.out or chosen.default_out
        fps = args.fps if args.fps is not None else chosen.default_fps

        from src.willy_sim.harness.bootstrap import DegradedMotionStackError

        run.note(f"    filming {args.record}: {chosen.filmed}.")
        run.note("    One Isaac boot and one full run, so this takes minutes and narrates itself")
        run.note("    at length below. The verdict is the two steps after the narration.")
        run.note("")
        # Deliberately not inside a `run.step`. Two reasons, and both are why the recorder is
        # called here and only reported on afterwards. A step opens a line that the body's own
        # output lands on, and this body prints hundreds of lines. And a refusal raised inside a
        # step is recorded as a failed step, which would report "the film failed" for a box that
        # was never able to start filming; those are the two different answers the exit codes
        # exist to separate.
        try:
            # The recorder's own return value, not a directory listing taken afterwards: it
            # reports the path it wrote, and reading that back is what makes the location below a
            # fact rather than a restatement of the argument.
            summary = chosen.call(out, headless=not args.gui, fps=fps,
                                  capture_every=args.capture_every)
        except SystemExit as refusal:
            # `SystemExit` is not an `Exception`, so it passes straight through `Example.step` and
            # would kill this file before its summary printed. Both recorders raise it: one when
            # the cuRobo environment is absent, because it will not film a blind-IK run, and both
            # when the cinematic camera captured no frames at all.
            return not_ready(str(refusal.code),
                             "install the motion engines with scripts/ext_deps/install.ps1, or "
                             "point WILLY_CUROBO_PYTHON and WILLY_COAL_PREFIX at a built pair; "
                             "`python -m src.robot.safety.planning --check` says which is missing")
        except DegradedMotionStackError as refusal:
            # The shared bootstrap refuses a cell whose configured motion engines are missing, and
            # it raises rather than exits. Reported as not-ready rather than as a failure: nothing
            # was filmed, so there is no result to have come out wrong.
            return not_ready(str(refusal),
                             "install the motion engines with scripts/ext_deps/install.ps1, "
                             "or set WILLY_ALLOW_DEGRADED_MOTION=1 and own the clip you get")

        landed = Path(str(summary.get("out", out))).resolve()
        with run.step(f"filmed {args.record}") as report:
            report(f"{summary.get('frames')} frames at {fps} fps, "
                   f"{float(summary.get('duration_s', 0.0)):.1f} s of clip")
        with run.step("where the MP4 landed") as report:
            if landed.is_file():
                report(f"{landed} ({landed.stat().st_size / 1_000_000.0:.1f} MB)")
            # Nothing is reported on the other branch, on purpose. `Example.step` records a step
            # that reported no detail as a failure, and a recorder that returned a path with no
            # file behind it is exactly that.
        stills = sorted(landed.parent.glob(f"{landed.stem}_key*.png"))
        run.note("")
        run.note(f"    {len(stills)} still(s) alongside it: "
                 f"{', '.join(p.name for p in stills) if stills else 'none'}")
        run.note("")
        run.note("WARN the clip scores nothing. It filmed the run that happened, including a")
        run.note("  failed one, and the file carries no claim about whether the cell works.")
        run.note("  For a rate, run 70_sim_pick.py, which scores the same scenes against ground")
        run.note("  truth and returns how many of N picks passed.")
        run.note("  Segments are separate Isaac boots, because the simulator app is a singleton.")
        run.note("  `python -m src.willy_sim.run_sorting_demo --concat a.mp4,b.mp4 --out all.mp4`")
        run.note("  stitches finished clips into one file without Isaac; the title card it draws")
        run.note("  before each segment carries that demo's own text.")
        return run.exit_code if landed.is_file() else EXIT_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
