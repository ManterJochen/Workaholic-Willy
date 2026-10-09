"""Every pick a cell ran, split into its stages from the cell's own logs; two log folders, A against B, stage by stage.

    python scripts/cell/pick_timeline.py <logs>                  the stages of every pick: best, median, worst
    python scripts/cell/pick_timeline.py <logs A> <logs B>       A against B, stage by stage
    python scripts/cell/pick_timeline.py <logs>#last             only the runs of the last connect (#2: the second)
    python scripts/cell/pick_timeline.py <logs>@09:30-10:15      only the runs that start in that window
    python scripts/cell/pick_timeline.py <logs> --picks          one line per pick besides
    python scripts/cell/pick_timeline.py <logs> --json out.json  everything, every pick's stages, for a spreadsheet
    python scripts/cell/pick_timeline.py --stages                what each stage holds

``<logs>`` is a cell's logs folder as the console writes it, copied whole: ``robot/robot.log`` (or the per-module files
under ``robot/``), ``models/*.log`` and ``api/*.log``, rotated files (``robot.log.1``, ...) included. It reads them and
writes nothing but the ``--json`` file, and it needs nothing but Python. What a folder lacks it leaves out: without
``models/`` a grounding is timed inside its look, and without ``api/`` the picks are told apart by their grasp records.

How a stage is timed: from the line that starts it to the line that starts the next one. Most lines are written when
what they name begins ("try 1 of 12", "move to approach_00: ...", "Moving to 'retreat'", "actuating jaws CLOSED"). A
few are written when it ends, with its length in them, and are dated back by it: a grounding ("perceived 3 object(s)
... in 13165.6 ms"), a planner world refresh ("806.2 ms to build, 245.7 ms to register"), a command read. A motion
whose end no line names (the lines, the carry, the return) runs until the first line of what comes after it, so it
holds the steady gate and what the controller is asked before that line; ``--stages`` says it per stage.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

#: Every stage this script times, in the order it prints them: (name, what it is part of, what it holds).
STAGES: tuple[tuple[str, str, str], ...] = (
    ("command", "task", "the sentence read: by Qwen, or a known sentence, or an answer from memory"),
    ("to_start", "task", "from the reading to the task's start: the card and the person's click, or Enter"),
    ("survey", "task", "from the task's start to its first pick: the countdown, a restart's way home, the bin found"),
    ("look", "pick", "a look judged and driven to (or skipped where the arm stands there), its frame taken"),
    ("grounding", "pick", "the detector on the look's frame: Qwen and SAM2, and a colour Qwen is asked to name"),
    ("grasp_search", "pick", "the grasp calculator over the parts the look saw"),
    ("judging", "pick", "the tries judged before the arm leaves the look: world, IK, guard, cuRobo; refused tries"),
    ("push", "pick", "a push of the part: its legs, and the look after it"),
    ("blocker", "pick", "a blocker searched for and taken away, its own pick and drop included"),
    ("approach", "pick", "the route to the standoff, from its send to 'judged path executed'"),
    ("standoff", "pick", "at the standoff: the steady gate, the line down solved and judged, the move's checks"),
    ("line_down", "pick", "the moveL down, then the lift solved at the part, to the first line of its judgement"),
    ("at_part", "pick", "the lift judged at the part as if the jaws held it, to the close"),
    ("close", "pick", "the jaws closed and settled, the line up judged, to its send (a URCap hand: in at_part)"),
    ("lift", "pick", "the moveL up, to the pick's end (a record written before the pick ends included)"),
    ("place_judging", "place", "from the pick's end to the place's first motion: the carry or the drop route judged"),
    ("carry", "place", "the carry to the bin's look, and the bin checked by depth there"),
    ("bin_check", "place", "the bin checked again by the detector"),
    ("drop_judging", "place", "the route to the drop's standoff judged"),
    ("to_drop", "place", "the route to the drop's standoff, from its send to 'judged path executed'"),
    ("drop_standoff", "place", "at the drop's standoff: the line in solved and judged"),
    ("line_in", "place", "the moveL in, to the jaws opening"),
    ("release", "place", "the jaws opened and settled, the line out judged, to its send"),
    ("line_out", "place", "the moveL out, to the first line of the way home's judgement"),
    ("return_judging", "place", "the way home judged"),
    ("return", "place", "the move home, to the next pick's first line or the task's end"),
)
_PICK_STAGES = tuple(name for name, part, _ in STAGES if part == "pick")
_PLACE_STAGES = tuple(name for name, part, _ in STAGES if part == "place")

#: What a pick spends inside its stages, read off lines that say their own length: (name, what it is).
INSIDE: tuple[tuple[str, str], ...] = (
    ("qwen", "Qwen grounding the look's frame (inside grounding)"),
    ("sam2", "SAM2's masks (inside grounding)"),
    ("colour", "Qwen naming a part's colour"),
    ("world", "planner world refreshes, built and registered, in every stage"),
    ("camera", "the camera read inside those refreshes"),
    ("register", "the boxes registered with cuRobo inside those refreshes"),
    ("curobo", "cuRobo checks and plans that said their length"),
)

#: The counts a pick keeps beside its stages.
COUNTS: tuple[tuple[str, str], ...] = (
    ("tries", "grasps tried"),
    ("refused", "tries refused before anything moved"),
    ("failed_sent", "tries that failed after a motion was sent"),
    ("looks", "looks the pick evaluated"),
    ("looks_skipped", "looks the arm already stood at"),
    ("parts", "parts the calculator computed"),
    ("refreshes", "planner world refreshes"),
    ("held", "motions judged in a held world"),
    ("mesh_first", "paths the exact guard alone judged clear"),
    ("curobo_judged", "paths cuRobo judged beside the exact guard"),
    ("curobo_refused", "paths cuRobo's check refused"),
    ("plans", "routes cuRobo planned"),
    ("no_plan", "plans cuRobo found none for"),
    ("route_reused", "routes run as judged ahead"),
    ("lift_reused", "lines up run on the lift judged at the part"),
    ("line_reused", "lines run as judged ahead where the arm stands"),
    ("leg_reused", "joint moves run as judged ahead (in a held world)"),
    ("local_ik", "lines solved on the controller's own kinematics"),
    ("local_ik_fallback", "lines that went back to the controller's IK"),
    ("judged_again", "motions judged again where the arm came to rest"),
    ("followed", "looks whose parts were followed, the detector not asked"),
    ("not_followed", "looks whose parts were not followed (grounded)"),
    ("over_the_rim", "drops let go over the rim instead of below it"),
    ("sfe_alone", "grasp searches made one build at a time where batched_builds is on"),
    ("guard_by_sample", "paths judged sample by sample where whole_path_judge is on"),
    ("records", "grasp records written"),
)

# ---------------------------------------------------------------------------------------------------------------------
# Reading a folder
# ---------------------------------------------------------------------------------------------------------------------

_LINE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}) - (.*?) - (DEBUG|INFO|WARNING|ERROR|CRITICAL) - (.*)$")
_LOG_FILE = re.compile(r"^(?P<base>.+\.log)(?:\.(?P<rotated>\d+))?$")


@dataclass(frozen=True)
class Record:
    """One line of a log, with the lines that continue it (a traceback) joined to its text."""

    at: datetime
    logger: str
    level: str
    text: str
    file: str


def _when(stamp: str) -> datetime:
    """``2026-10-08 07:38:15,722`` as a datetime, without strptime's cost on a day's lines."""
    return datetime(int(stamp[0:4]), int(stamp[5:7]), int(stamp[8:10]), int(stamp[11:13]), int(stamp[14:16]),
                    int(stamp[17:19]), int(stamp[20:23]) * 1000)


def _rank(name: str) -> int:
    """Which file's line comes first where two files carry lines of the same millisecond: the robot's aggregate, the
    robot's own files, the models', the console's, which writes what the others did once it is done."""
    if name == "robot/robot.log":
        return 0
    for rank, prefix in enumerate(("robot/", "models/", "api/"), start=1):
        if name.startswith(prefix):
            return rank
    return 4


def log_files(folder: Path) -> dict[str, list[Path]]:
    """Every log under ``folder`` by its name in the folder, its rotated files first, the oldest first."""
    groups: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or "views" in path.relative_to(folder).parts:
            continue
        match = _LOG_FILE.match(path.name)
        if match is None:
            continue
        name = (path.parent / match["base"]).relative_to(folder).as_posix()
        groups[name].append((int(match["rotated"] or 0), path))
    return {name: [path for _, path in sorted(paths, key=lambda item: -item[0])] for name, paths in groups.items()}


def read_folder(folder: Path) -> list[Record]:
    """Every line of every log in ``folder``, in time order, each once.

    The robot's aggregate ``robot.log`` and its per-module files carry the same lines, and a folder may hold either or
    both, rotated apart: a line is kept as many times as the one file holding it most often holds it.
    """
    rows: list[tuple[datetime, int, str, int, tuple[str, str, str, str]]] = []
    most: Counter[tuple[str, str, str, str]] = Counter()
    for name, paths in log_files(folder).items():
        rank = _rank(name)
        held: Counter[tuple[str, str, str, str]] = Counter()
        index = 0
        for path in paths:
            last: list[Any] | None = None
            for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
                match = _LINE.match(raw)
                if match is None:
                    if last is not None and raw.strip():
                        last[3] += "\n" + raw  # a traceback or a message of several lines
                    continue
                if last is not None:
                    rows.append(_row(last, rank, name))
                    held[rows[-1][4]] += 1
                last = [match[1], match[2], match[3], match[4], index]
                index += 1
            if last is not None:
                rows.append(_row(last, rank, name))
                held[rows[-1][4]] += 1
        for key, count in held.items():
            most[key] = max(most[key], count)
    rows.sort(key=lambda row: row[:4])
    kept: Counter[tuple[str, str, str, str]] = Counter()
    records: list[Record] = []
    for at, _, name, _, key in rows:
        if kept[key] < most[key]:
            kept[key] += 1
            records.append(Record(at=at, logger=key[1], level=key[2], text=key[3], file=name))
    return records


def _row(fields: list[Any], rank: int, name: str) -> tuple[datetime, int, str, int, tuple[str, str, str, str]]:
    stamp, logger, level, text, index = fields
    return _when(stamp), rank, name, index, (stamp, logger, level, text)


# ---------------------------------------------------------------------------------------------------------------------
# What a line says
# ---------------------------------------------------------------------------------------------------------------------

_P = re.compile
_S = re.DOTALL
#: The lines this script knows, first match first, by what they say rather than by who logs them (logger names
#: changed before). Each named after the event it becomes.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # The console: runs, tasks, commands, connects.
    ("task_pick", _P(r"^Task run (?P<run>\S+), part (?P<part>\d+), pick (?P<pick>\d+): (?P<outcome>\w+)\.$")),
    ("task_ended", _P(r"^Task run (?P<run>\S+) ended (?P<stop>\w+): (?P<sentence>.*)$", _S)),
    ("task_started", _P(r"^Task run (?P<run>\S+) started: (?P<object>'.*?'|\".*?\") into (?P<target>.*?), "
                        r"(?P<scope>\w+)(?:; its picks ground (?P<phrase>'.*'|\".*\") where the cell grounds a "
                        r"phrase)?\.$", _S)),
    ("task_handed", _P(r"^Task run (?P<run>\S+) is handed the ")),
    ("restarts", _P(r"^Run (?P<run>\S+) restarts run (?P<of>\S+) \(")),
    ("run_pick", _P(r"^Run (?P<run>\S+) pick (?P<pick>\d+)/(?P<of>\d+): (?P<outcome>\w+) \(", _S)),
    ("run_starting", _P(r"^Run (?P<run>\S+) starting: (?:(?P<picks>\d+) pick\(s\), prompt (?P<prompt>.*)|"
                        r"(?P<kind>\w+)(?:, restarting (?P<of>\S+))?)\.$", _S)),
    ("run_finished", _P(r"^Run (?P<run>\S+) \((?P<kind>\w+)\) (?P<verb>finished|failed): (?P<code>\w+) in "
                        r"(?P<s>[\d.]+) s\.$")),
    ("run_finished", _P(r"^Run (?P<run>\S+) (?P<verb>\w+) \((?P<code>\w+)\): (?P<ok>\d+)/(?P<of>\d+) succeeded in "
                        r"(?P<s>[\d.]+) s\.$")),
    ("abandoned", _P(r"^Run (?P<run>\S+) abandoned \(")),
    ("countdown", _P(r"^Run (?P<run>\S+) \((?P<kind>\w+)\) counts down (?P<s>\d+) s before its first motion")),
    ("halt_asked", _P(r"^Halt requested for run (?P<run>\S+)")),
    ("command", _P(r"^Command read \((?P<source>[^,]*) source, (?P<language>[^)]*)\): intent (?P<intent>\w+), "
                   r"(?P<questions>\d+) question\(s\), (?P<ms>\d+) ms(?P<how>.*)\.$", _S)),
    ("connecting", _P(r"^(?:Connecting: arm=|Connecting to UR robot at )")),
    ("disconnected", _P(r"^(?:Cell disconnected|Disconnected\.$|Cell down \()")),
    ("preview", _P(r"^Connect preview issued for .*config fingerprint (?P<fingerprint>[0-9a-f]+)")),
    ("built", _P(r"^Cell built in (?P<s>[\d.]+) s")),
    # The models.
    ("perceived", _P(r"^perceived (?P<objects>\d+) object\(s\) from (?P<detections>\d+) detection\(s\) for prompt "
                     r"(?P<prompt>.*) in (?P<ms>[\d.]+) ms$", _S)),
    ("qwen", _P(r"^VLM grounded (?:(?P<boxes>\d+) box\(es\)|nothing) for (?P<prompt>.*?) in (?P<ms>\d+) ms"
                r"(?:, (?P<tokens>\d+) token\(s\))?", _S)),
    ("colour", _P(r"^VLM \S+ named the colour of the object in .* in (?P<ms>\d+) ms$", _S)),
    ("vlm_ready", _P(r"^VLM (?P<model>\S+) (?:ready|loaded on request) in (?P<s>[\d.]+) s")),
    ("vlm_config", _P(r"^VLM (?P<model>\S+) answers with (?P<said>.*)$", _S)),
    ("sam2", _P(r"^SAM2 segmentation completed in (?P<s>[\d.]+)s")),
    ("followed", _P(r"^(?P<said>.*) in (?P<ms>\d+) ms: the detector was not asked$", _S)),
    ("not_followed", _P(r"^the parts kept are not followed on this frame")),
    ("command_read", _P(r"^read (?P<said>.*) as (?P<intent>\w+)(?: in (?P<questions>\d+) question\(s\), (?P<ms>\d+) "
                        r"ms|: a known sentence, no model asked, (?P<known_ms>[\d.]+) ms)", _S)),
    # The pick loop.
    ("try_refused", _P(r"^try (?P<k>\d+) of (?P<n>\d+) \(rank \d+\) judged from where the arm stands, nothing "
                       r"moved: (?P<why>.*)$", _S)),
    ("try_failed", _P(r"^try (?P<k>\d+) of (?P<n>\d+) \(rank \d+\) (?P<why>.*)$", _S)),
    ("try", _P(r"^try (?P<k>\d+) of (?P<n>\d+): ")),
    ("look_skipped", _P(r"^look (?P<look>.+?): the arm already stands there")),
    ("looks_end", _P(r"^(?:look .+?: a valid grasp with no rescan reason|the looks (?:stop at|end on|reached)|"
                     r"no look (?:saw anything|ranked a grasp))", _S)),
    ("next_look", _P(r"^look .+?(?:: (?:no grasp candidate|the grasp is uncertain|the contact face .* was not seen|"
                     r"nothing was segmented there)| was refused before anything was sent)", _S)),
    ("part_good", _P(r"^part \d+ is good ")),
    ("generated", _P(r"^Generated (?P<n>\d+) grasp candidates")),
    ("search", _P(r"^(?:the camera world finds what the part stands on|support plane observed at|the parts are "
                  r"computed in the order|scene obstacles: |no grasp: |the wrist camera keeps the guard's distance|"
                  r"Dense sampling overran budget|no part has a full result)")),
    ("push", _P(r"^pushing the part ")),
    ("blocker", _P(r"^(?:a blocker at |taking away the blocker|taking away all \d+ neighbour|clear the blocker|"
                   r"setting the blocker aside)")),
    ("recovery_start", _P(r"^recovery rescans nothing for a wrist pick")),
    # The UR driver and the planner.
    ("route_ahead", _P(r"^move to (?P<label>.+?): the route judged ahead runs as it was judged")),
    ("route_again", _P(r"^move to (?P<label>.+?): the route judged ahead is judged again")),
    ("route", _P(r"^(?P<verb>.+?): (?P<how>.+?); (?P<dense>\d+) waypoint\(s\) dense, (?P<sent>\d+) executed "
                 r"\((?P<legs>\d+) leg\(s\)\); joint travel", _S)),
    ("executed", _P(r"^judged path executed on UR: (?P<n>\d+) moveJ")),
    ("halted_path", _P(r"^judged path ended by the halt")),
    ("line", _P(r"^Moving to '(?P<label>.*)' \[")),
    ("world_held", _P(r"^the world is held \(")),
    ("refresh", _P(r"^planner world refreshed: (?P<boxes>\d+) box\(es\), frame (?P<age>[^,]*), (?P<build>[\d.]+) ms "
                   r"to build(?: \((?P<grab>[\d.]+) ms of it reading the camera\))?, (?P<register>[\d.]+) ms to "
                   r"register")),
    ("refresh_failed", _P(r"^planner world NOT refreshed")),
    ("mesh_first", _P(r"^mesh first: the exact guard alone judged (?P<n>\d+) sample\(s\)")),
    ("mesh_first_na", _P(r"^mesh first does not apply")),
    ("lift_judged", _P(r"^a lift to \S+ judged as if the jaws held a part")),
    ("local_ik", _P(r"^the line to .+?: \d+ configuration\(s\) solved here on the controller's own kinematics")),
    ("local_ik_fallback", _P(r"^the line to .+? is solved by the controller as before")),
    ("line_reused", _P(r"^the line to .+? runs as it was judged ahead where the arm stands")),
    ("leg_reused", _P(r"^.+?: the joint move judged ahead runs as it was judged")),
    ("judged_again", _P(r"^the arm came to rest .* from where its motion was judged from")),
    ("leg_on_a_thread", _P(r"^the joint move declared next is judged on a second thread while the line")),
    ("over_the_rim", _P(r"^the line into the box, to .+?, was refused before anything was sent")),
    ("sfe_alone", _P(r"^SFE makes its builds one at a time")),
    ("guard_by_sample", _P(r"^Safety preflight judges this path one sample at a time although")),
    ("line_up_ahead", _P(r"^the line up to \S+ runs on the lift judged at the part")),
    ("line_up_again", _P(r"^the line up to \S+ is judged again")),
    ("planner_ms", _P(r"^(?:the cuRobo check refused a joint path|(?P<none>cuRobo found NO collision-free) .*?) "
                      r"after (?P<ms>\d+) ms", _S)),
    ("planner", _P(r"^(?:the planner refuses \d+ of \d+|the planner's refusal stands|the world refused |payload "
                   r"attached to the planner|this move changes the arm's branch|the plan shortened to)")),
    ("close", _P(r"^actuating jaws CLOSED via (?P<how>\S+)")),
    ("open", _P(r"^actuating jaws OPEN via (?P<how>\S+)")),
    ("jaws_settled", _P(r"^jaws settled: .* after (?P<s>[\d.]+) s")),
    ("record", _P(r"^attempt (?P<id>\S+) \((?P<outcome>\w+)\) appended to .*: (?P<bytes>\d+) bytes written", _S)),
    ("survey", _P(r"^survey: (?P<said>.*)$", _S)),
    ("recheck", _P(r"^recheck: (?P<said>.*)$", _S)),
    ("spawn", _P(r"^spawning the cuRobo sidecar: (?P<said>.*)$", _S)),
    ("planner_ready", _P(r"^cuRobo sidecar ready after (?P<s>[\d.]+) s")),
    ("workers", _P(r"^SFE workers(?: \(robot\.grasping\.workers\))?: (?P<said>.*)$", _S)),
    ("tool_frame", _P(r"^tool frame verified: .*within (?P<mm>[\d.]+) mm / (?P<deg>[\d.]+) deg")),
    # What a connect says it switched on: the switches that log their choice.
    ("said", _P(r"^(?P<said>(?:safety\.dwell\.gate_at is 'send'|robot\.motion\.judge_next_leg is |The steady gate "
                r"reads the joint speeds|The controller's kinematics|The singularity check before a move runs on the "
                r"arm's own DH chain|VLM \S+ decodes |VLM \S+: decode graphs |SFE builds each closing line's grasps "
                r"at once|robot\.grasping\.batched_builds is on, but|the planner plans on the |the sidecar did not "
                r"build the |planning model |the sidecar says nothing of the planning model).*)$", _S)),
)

#: Lines written where what they name begins, which nothing written later may be dated back past.
_MARKERS = frozenset({
    "task_pick", "task_ended", "task_started", "run_pick", "run_starting", "run_finished", "abandoned", "countdown",
    "try", "try_refused", "try_failed", "next_look", "push", "blocker", "recovery_start", "route_ahead", "route",
    "executed", "halted_path", "line", "close", "open",
})
#: What shows the arm stands and the run thread works again: a motion whose end no line names ends at the first of these.
_WORK = frozenset({
    "world_held", "refresh_failed", "mesh_first", "mesh_first_na", "lift_judged", "line_up_ahead",
    "line_up_again", "route_again", "planner", "survey", "recheck", "look_skipped", "local_ik", "local_ik_fallback",
    "line_reused", "leg_reused", "judged_again",
})


@dataclass
class Event:
    """What a line says, taking effect ``at`` (a line that says its own length begins that long before it)."""

    at: datetime
    order: int
    kind: str
    data: dict[str, Any]
    logged: datetime


def _quoted(text: str | None) -> str:
    """A ``%r`` of a phrase as the phrase."""
    if not text:
        return ""
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        return text[1:-1]
    return text


def events_of(records: list[Record]) -> list[Event]:
    """Every line this script knows as an event, in the order they take effect.

    A line that says its own length is two events: what it names begins that long before the line, and ends at it.
    Its beginning is never dated back past the last line written where something began (:data:`_MARKERS`): the
    milliseconds a line rounds to are not allowed to reorder what the log says happened.
    """
    out: list[Event] = []
    last_marker = datetime.min
    for order, record in enumerate(records):
        text = record.text
        for kind, pattern in _PATTERNS:
            if kind == "route" and "waypoint(s) dense" not in text:
                continue  # the one pattern that would scan every line it does not fit
            match = pattern.match(text)
            if match is None:
                continue
            data = {key: value for key, value in match.groupdict().items() if value is not None}
            if "known_ms" in data:  # the reader's line for a known sentence: no model asked, a fraction of a ms
                data["ms"], data["how"] = data.pop("known_ms"), "known sentence"
            at = record.at
            if kind in _MARKERS:
                last_marker = max(last_marker, at)
            seconds = _length_s(kind, data)
            if seconds is not None:
                begin = max(at - timedelta(seconds=seconds), min(last_marker, at))
                out.append(Event(begin, order, _BEGINS.get(kind, kind) + "_begin", data, at))
            out.append(Event(at, order, kind, data, at))
            break
    out.sort(key=lambda event: (event.at, event.order))
    return out


#: The lines that say their own length, and what each begins.
_BEGINS = {"perceived": "grounding", "qwen": "grounding", "colour": "grounding", "followed": "grounding",
           "command": "command", "command_read": "command", "refresh": "work", "planner_ms": "work"}


def _length_s(kind: str, data: dict[str, Any]) -> float | None:
    if kind in ("perceived", "qwen", "colour", "followed", "command", "command_read", "planner_ms"):
        return float(data["ms"]) / 1000.0
    if kind == "refresh":
        return (float(data["build"]) + float(data["register"])) / 1000.0
    return None


# ---------------------------------------------------------------------------------------------------------------------
# Connects and runs
# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class Session:
    """One connect of the cell, to its disconnect: one configuration."""

    number: int
    start: datetime
    end: datetime | None = None
    fingerprint: str = ""
    facts: dict[str, Any] = field(default_factory=dict)


@dataclass
class Command:
    """A sentence read by the console."""

    at: datetime
    ms: float
    intent: str
    how: str


@dataclass
class Unit:
    """A stretch of a run: its survey, a pick, the place after a pick, or the time between two picks."""

    kind: str
    start: datetime
    end: datetime | None = None
    stages: dict[str, float] = field(default_factory=dict)
    inside: dict[str, float] = field(default_factory=dict)
    counts: Counter[str] = field(default_factory=Counter)
    reasons: Counter[str] = field(default_factory=Counter)
    outcome: str = ""
    part: int | None = None
    pick: int | None = None
    order: list[str] = field(default_factory=list)

    @property
    def total(self) -> float:
        return 0.0 if self.end is None else max(0.0, (self.end - self.start).total_seconds())


@dataclass
class Run:
    """One run of the console: a task, a pick run, a wave, a way home, the planner's start."""

    id: str
    kind: str
    start: datetime
    end: datetime | None = None
    code: str = ""
    seconds: float | None = None
    object: str = ""
    target: str = ""
    scope: str = ""
    phrase: str = ""
    restarts: str = ""
    session: int = 0
    command: Command | None = None
    units: list[Unit] = field(default_factory=list)


def sessions_of(events: list[Event]) -> list[Session]:
    """The connects, each with the build's facts before it and what the connect logged."""
    sessions: list[Session] = []
    open_: Session | None = None
    pending: dict[str, Any] = {}
    fingerprint = ""
    for event in events:
        kind, data = event.kind, event.data
        if kind == "preview":
            fingerprint = data["fingerprint"]
        elif kind == "built":
            pending["cell_build_s"] = float(data["s"])
        elif kind == "connecting" and open_ is None:
            open_ = Session(number=len(sessions) + 1, start=event.logged, fingerprint=fingerprint, facts=dict(pending))
            sessions.append(open_)
            pending = {}
        elif kind == "disconnected" and open_ is not None:
            open_.end = event.logged
            open_ = None
        elif open_ is not None:
            if kind == "planner_ready":
                open_.facts.setdefault("planner_start_s", float(data["s"]))
            elif kind == "spawn":
                open_.facts.setdefault("planner", data["said"])
            elif kind == "workers":
                open_.facts.setdefault("workers", data["said"])
            elif kind == "tool_frame":
                open_.facts.setdefault("tool_frame", f"{data['mm']} mm / {data['deg']} deg")
        if kind == "said":
            said = (open_.facts if open_ is not None else pending).setdefault("said", [])
            if data["said"] not in said:
                said.append(data["said"])
        if kind == "vlm_ready":
            (open_.facts if open_ is not None else pending).setdefault("qwen_load_s", float(data["s"]))
        elif kind == "vlm_config":
            (open_.facts if open_ is not None else pending).setdefault("qwen", data["said"])
    return sessions


def runs_of(events: list[Event], sessions: list[Session]) -> list[Run]:
    """The console's runs, from its own lines; with no console lines, one run over everything the robot logged."""
    runs: dict[str, Run] = {}

    def run(rid: str, at: datetime) -> Run:
        if rid not in runs:
            runs[rid] = Run(id=rid, kind="", start=at)
        found = runs[rid]
        found.start = min(found.start, at)
        return found

    for event in events:
        kind, data, at = event.kind, event.data, event.logged
        if kind == "run_starting":
            found = run(data["run"], at)
            found.kind = data.get("kind") or ("pick" if "picks" in data else found.kind)
            found.restarts = data.get("of", found.restarts)
            if "prompt" in data:
                found.phrase = _quoted(data["prompt"])
        elif kind == "task_started":
            found = run(data["run"], at)
            found.kind = found.kind or "task"
            found.object, found.target = _quoted(data.get("object")), data.get("target", "")
            found.scope, found.phrase = data.get("scope", ""), _quoted(data.get("phrase"))
        elif kind in ("run_finished", "task_ended", "abandoned") and data["run"] in runs:
            found = runs[data["run"]]
            found.end = max(found.end or at, at)
            if kind == "run_finished":
                found.code, found.seconds = data["code"], float(data["s"])
                found.kind = found.kind or data.get("kind", "")
            elif not found.code:
                found.code = data.get("stop", "abandoned")
        elif kind == "restarts" and data["run"] in runs:
            runs[data["run"]].restarts = data["of"]
    ordered = sorted(runs.values(), key=lambda found: found.start)
    for found, after in zip(ordered, [*ordered[1:], None]):
        if found.end is None:
            # A run the log never saw end (a console that stopped): it ends where the next began, or at its connect's
            # end, never past the log's last line.
            ends = [event.logged for event in events[-1:]]
            ends += [after.start] if after is not None else []
            ends += [session.end for session in sessions
                     if session.end is not None and session.start <= found.start <= session.end]
            found.end = min(ends) if ends else None
        if found.restarts in runs and not found.object:
            first = runs[found.restarts]
            found.object, found.target, found.scope, found.phrase = (first.object, first.target, first.scope,
                                                                    first.phrase)
        for session in sessions:
            if session.start <= found.start and (session.end is None or found.start <= session.end):
                found.session = session.number
    if not ordered and events:
        robot = [event for event in events if event.kind not in ("connecting", "disconnected")]
        if robot:
            ordered = [Run(id="(no console log)", kind="logs", start=robot[0].logged, end=robot[-1].logged)]
    return ordered


def commands_of(events: list[Event]) -> list[Command]:
    """The sentences the console read, from its own line or, without the console's logs, from the reader's."""
    said = [Command(at=event.logged, ms=float(event.data["ms"]), intent=event.data["intent"],
                    how=_how_read(event.data.get("how", "")))
            for event in events if event.kind == "command"]
    if said:
        return said
    return [Command(at=event.logged, ms=float(event.data["ms"]), intent=event.data["intent"],
                    how=event.data.get("how", "")) for event in events if event.kind == "command_read"]


def _how_read(how: str) -> str:
    for words, said in (("known sentence", "known sentence"), ("from memory", "from memory"),
                        ("model loaded for it", "model loaded for it")):
        if words in how:
            return said
    return ""


# ---------------------------------------------------------------------------------------------------------------------
# A run's timeline
# ---------------------------------------------------------------------------------------------------------------------

_SEARCH = frozenset({"search", "generated", "part_good", "looks_end"})
_TRY = frozenset({"try", "try_refused", "try_failed"})
_BEGINNINGS = frozenset({"work_begin", "grounding_begin"})


class _Timeline:
    """Feeds a run's events through the stages they mark, unit by unit (survey, pick, place, between)."""

    def __init__(self, run: Run) -> None:
        self.run = run
        self.units: list[Unit] = []
        self.unit: Unit | None = None
        self.phase = ""
        self.since = run.start
        self.picks = run.phrase.lower() or (f"each separate {run.object}".lower() if run.object else "")
        #: Whether the move after the line in flight is being judged on a second thread (robot.motion.judge_next_leg
        #: in_settles_and_motion): what that thread writes while the line runs is no sign that the line ended.
        self.beside = False
        #: The first work after a place's way home was sent, or between two picks: the next pick's start once a line
        #: only a pick writes follows, or nothing where the task's own way home at its end follows instead.
        self.maybe: datetime | None = None
        first = "survey" if run.kind == "task" else "between"
        self._open(first, run.start, first)

    # -- bookkeeping --

    def _charge(self, at: datetime) -> None:
        assert self.unit is not None
        if at > self.since:
            self.unit.stages[self.phase] = self.unit.stages.get(self.phase, 0.0) + (at - self.since).total_seconds()
            self.since = at

    def to(self, phase: str, at: datetime) -> None:
        """The unit's stage is ``phase`` from ``at``, never before the stage it ends began."""
        at = max(at, self.since)
        self._charge(at)
        self.phase = phase
        assert self.unit is not None
        self.unit.stages.setdefault(phase, 0.0)
        if phase not in self.unit.order:
            self.unit.order.append(phase)

    def _open(self, kind: str, at: datetime, phase: str) -> None:
        self.unit = Unit(kind=kind, start=at)
        self.units.append(self.unit)
        self.since = at
        self.phase = ""
        self.to(phase, at)

    def _close(self, at: datetime) -> None:
        assert self.unit is not None
        at = max(at, self.since)
        self._charge(at)
        self.unit.end = at

    def next_unit(self, kind: str, at: datetime, phase: str) -> None:
        at = max(at, self.since)
        self._close(at)
        self._open(kind, at, phase)

    def add(self, name: str, seconds: float) -> None:
        assert self.unit is not None
        self.unit.inside[name] = self.unit.inside.get(name, 0.0) + seconds

    # -- the events --

    def feed(self, event: Event) -> None:
        assert self.unit is not None
        kind = event.kind
        if kind == "leg_on_a_thread":
            self.beside = True  # said before the line it runs beside is sent: that line's own send does not end it
        elif (kind in _MARKERS and kind != "line") or kind == "leg_reused":
            self.beside = False
        if self.beside and (kind in _WORK or kind in _BEGINNINGS):
            self._count(event)
            return
        if kind in ("task_pick", "run_pick") or (kind == "record" and self.run.kind == "logs"):
            self._pick_ended(event)
            return
        if self.unit.kind != "pick":
            start = self._pick_start(event)
            if start is not None:
                self.next_unit("pick", start, "look")
        if self.unit.kind == "pick":
            self._pick(event)
        elif self.unit.kind == "place":
            self._place(event)
        self._count(event)

    def _pick_ended(self, event: Event) -> None:
        assert self.unit is not None
        data, at = event.data, event.at
        if self.unit.kind != "pick":
            self.next_unit("pick", self.maybe or self.since, "look")
            self.maybe = None
        pick = self.unit
        pick.outcome = data.get("outcome", "")
        pick.part = int(data["part"]) if "part" in data else None
        pick.pick = int(data["pick"]) if "pick" in data else None
        if event.kind == "record":
            pick.counts["records"] += 1
        pick.counts["looks"] += 1  # the look it ended on; every look before it said so (next_look)
        following = "place" if pick.outcome == "succeeded" and self.run.kind == "task" else "between"
        self.next_unit(following, at, "place_judging" if following == "place" else "between")

    def _pick_start(self, event: Event) -> datetime | None:
        """Where a pick starts that ``event``, outside one, shows has started; ``None`` where it shows none.

        Only a line no survey and no way home writes shows a pick: the pick loop's own (a try, a grasp search, a look
        evaluated, the service's start of a wrist pick) or a grounding of the pick's phrase. After a place's way home
        was sent, and between two picks of a task or a pick run, the first work before that line is where the pick
        began, its first look being judged: the place's way home ends there. Where the task's own way home at its
        end follows instead, that work was its judgement, and no pick began. The survey looks and judges as a pick
        does, and a folder without the console's lines has no run to say where a pick may begin: there only the
        pick's own lines start one.
        """
        assert self.unit is not None
        kind, unit = event.kind, self.unit.kind
        shown = (kind in ("recovery_start", "next_look") or kind in _SEARCH or kind in _TRY
                 or (kind == "grounding_begin" and self._grounds(event) == "pick"))
        if shown:
            start, self.maybe = self.maybe or event.at, None
            return start
        if (unit == "place" and self.phase == "return") or (unit == "between" and self.run.kind in ("task", "pick")):
            if kind == "route" and event.data.get("verb") == "move_home":
                self.maybe = None
            elif self.maybe is None and (kind in _WORK or kind in _BEGINNINGS or kind == "route"):
                self.maybe = event.at
        return None

    def _grounds(self, event: Event) -> str:
        """What a grounding is for: ``pick`` (the parts), ``bin`` (the task's target), or ``other`` (the survey's own,
        or one a folder without the console's lines cannot place)."""
        prompt = _quoted(event.data.get("prompt")).lower()
        if self.run.target and prompt == self.run.target.lower():
            return "bin"
        if prompt.startswith("each separate") or (self.picks and prompt == self.picks):
            return "pick"
        if self.unit is not None and self.unit.kind == "pick":
            return "pick"  # a later look's, whatever its phrase
        if self.run.kind == "pick" and not self.picks:
            return "pick"
        return "other"

    def _pick(self, event: Event) -> None:
        kind, data, at = event.kind, event.data, event.at
        phase = self.phase
        if kind == "next_look":
            self.to("look", at)
        elif kind == "grounding_begin":
            if phase != "grounding":
                self.to("grounding", at)
        elif kind in ("perceived", "colour", "followed"):
            if phase in ("grounding", "look"):
                self.to("grasp_search", at)
        elif kind in _SEARCH:
            if phase in ("look", "grounding"):
                self.to("grasp_search", at)
        elif kind in _TRY:
            if phase != "judging":
                self.to("judging", at)
        elif kind == "push":
            self.to("push", at)
        elif kind == "blocker":
            if phase != "blocker":
                self.to("blocker", at)
        elif phase in ("push", "blocker"):
            return  # the push's legs and the blocker's own pick and drop stay theirs, until the part is tried again
        elif kind == "route_ahead" or (kind == "route" and data.get("verb", "").startswith("move to approach")):
            if phase != "approach":
                self.to("approach", at)
        elif kind == "executed":
            if phase == "approach":
                self.to("standoff", at)
        elif kind == "line":
            label = data["label"]
            if label.startswith("approach"):
                self.to("line_down", at)
            elif label.startswith(("retreat", "lift", "standoff")):
                self.to("lift", at)
        elif kind == "close":
            self.to("close", at)
        elif phase == "line_down" and (kind in _WORK or kind == "work_begin"):
            self.to("at_part", at)

    def _place(self, event: Event) -> None:
        kind, data, at = event.kind, event.data, event.at
        phase = self.phase
        if kind == "route":
            verb = data.get("verb", "")
            if verb == "move_home" or (verb == "move_to_joints" and phase in ("release", "line_out",
                                                                             "return_judging")):
                self.to("return", at)
            elif verb == "move_to_joints" and phase == "place_judging":
                self.to("carry", at)
            elif verb.startswith("move to"):
                self.to("to_drop", at)
        elif kind == "route_ahead":
            self.to("to_drop", at)
        elif kind == "executed":
            if phase == "to_drop":
                self.to("drop_standoff", at)
        elif kind == "grounding_begin":
            if phase in ("place_judging", "carry"):
                self.to("bin_check", at)
        elif kind == "perceived":
            if phase == "bin_check":
                self.to("drop_judging", at)
        elif kind == "recheck":
            if phase in ("carry", "bin_check"):
                self.to("drop_judging", at)
        elif kind == "line":
            label = data["label"]
            if label.startswith("standoff") and phase in ("release", "line_in"):
                self.to("line_out", at)
            elif label.startswith(("retreat", "lift")):
                self.to("line_out", at)
            elif phase in ("drop_standoff", "to_drop", "drop_judging", "place_judging"):
                self.to("line_in", at)
        elif kind == "open":
            self.to("release", at)
        elif kind in _WORK or kind == "work_begin":
            if phase in ("carry", "bin_check"):
                self.to("drop_judging", at)
            elif phase == "line_out":
                self.to("return_judging", at)

    def _count(self, event: Event) -> None:
        assert self.unit is not None
        kind, data, unit = event.kind, event.data, self.unit
        counts = unit.counts
        if kind in ("try", "try_refused", "try_failed"):
            if kind == "try":
                counts["tries"] += 1
            else:
                counts["refused" if kind == "try_refused" else "failed_sent"] += 1
                unit.reasons[_reason(data.get("why", ""))] += 1
        elif kind == "next_look" and unit.kind == "pick":
            counts["looks"] += 1
        elif kind == "look_skipped":
            counts["looks_skipped"] += 1
        elif kind == "generated":
            counts["parts"] += 1
        elif kind == "refresh":
            counts["refreshes"] += 1
        elif kind == "work_begin" and "build" in data:
            self.add("world", (float(data["build"]) + float(data["register"])) / 1000.0)
            self.add("register", float(data["register"]) / 1000.0)
            if "grab" in data:
                self.add("camera", float(data["grab"]) / 1000.0)
        elif kind == "work_begin" and "ms" in data:
            self.add("curobo", float(data["ms"]) / 1000.0)
        elif kind == "planner_ms":
            counts["no_plan" if "none" in data else "curobo_refused"] += 1
        elif kind == "mesh_first":
            counts["mesh_first"] += 1
        elif kind == "mesh_first_na":
            counts["curobo_judged"] += 1
        elif kind == "route" and "plan" in data.get("how", ""):
            counts["plans"] += 1
        elif kind == "world_held":
            counts["held"] += 1
        elif kind == "route_ahead":
            counts["route_reused"] += 1
        elif kind == "line_up_ahead":
            counts["lift_reused"] += 1
        elif kind in ("followed", "not_followed", "leg_reused", "line_reused", "local_ik", "local_ik_fallback",
                      "judged_again", "over_the_rim", "sfe_alone", "guard_by_sample"):
            counts[kind] += 1
        elif kind == "record":
            counts["records"] += 1
        elif kind == "qwen":
            self.add("qwen", float(data["ms"]) / 1000.0)
        elif kind == "sam2":
            self.add("sam2", float(data["s"]))
        elif kind == "colour":
            self.add("colour", float(data["ms"]) / 1000.0)

    def finish(self, at: datetime) -> list[Unit]:
        self._close(at)
        return self.units


def _reason(why: str) -> str:
    """A refused or failed try in a few words: where it was refused, and by what."""
    low = why.lower()
    where = next((said for words, said in (("line down", "line down"), ("lift", "lift"), ("retreat", "lift"),
                                           ("route", "route"), ("standoff", "route"), ("move to", "route"))
                  if words in low), "")
    guard = re.search(r"(?P<part>[\w.-]+)\|(?P<other>[\w.:-]+): mesh distance", why)
    if guard is not None:
        other = guard["other"]
        what = ("guard: " + guard["part"] + " vs " + ("a box the camera saw" if other.startswith("fixture:seen")
                                                      else "a fixture" if other.startswith("fixture:") else other))
    else:
        what = next((said for words, said in (("planner", "planner"), ("curobo", "planner"),
                                              ("joint limit", "joint limit"), ("workspace", "workspace"),
                                              ("branch", "branch"), ("singular", "singularity"),
                                              ("inverse kinematics", "IK"), ("cancel", "stopped"),
                                              ("halt", "stopped")) if words in low), "other")
    return f"{where}: {what}" if where else what


def timeline(run: Run, events: list[Event]) -> list[Unit]:
    """``run``'s units from the events inside it, the last ending where the run said it ended."""
    end = run.end or (events[-1].logged if events else run.start)
    machine = _Timeline(run)
    for event in events:
        if event.logged < run.start or event.logged > end:
            continue
        if event.kind in ("run_starting", "task_started", "restarts", "task_handed"):
            continue
        if event.kind in ("task_ended", "run_finished", "abandoned"):
            if event.data.get("run") == run.id:
                end = event.logged  # the task's own end line, written as its last motion ended
                break
            continue
        if event.kind in ("task_pick", "run_pick") and event.data.get("run") not in (run.id, None):
            continue
        machine.feed(event)
    return machine.finish(end)


# ---------------------------------------------------------------------------------------------------------------------
# A folder, a selection of it
# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class Folder:
    """A logs folder read, and the runs of it a selection keeps."""

    path: Path
    selection: str
    sessions: list[Session]
    runs: list[Run]
    first: datetime | None
    last: datetime | None
    files: list[str]


_WINDOW = re.compile(r"^(?P<a>[\dT:-]*?)(?:-(?P<b>[\dT:]*))?$")


def _instant(text: str, day: datetime | None) -> datetime | None:
    if not text:
        return None
    text = text.replace("T", " ")
    for form in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%H:%M:%S", "%H:%M"):
        try:
            parsed = datetime.strptime(text, form)
        except ValueError:
            continue
        if form.startswith("%H") and day is not None:
            parsed = parsed.replace(year=day.year, month=day.month, day=day.day)
        return parsed
    raise SystemExit(f"{text!r} is no time: give HH:MM, HH:MM:SS or YYYY-MM-DDTHH:MM")


def split_spec(spec: str) -> tuple[Path, str]:
    """``<folder>`` and what follows its last ``#`` or ``@`` (a connect or a time window), where the whole is no
    folder of its own."""
    path = Path(spec)
    if path.exists():
        return path, ""
    for mark in ("#", "@"):
        if mark in path.name:
            head, _, tail = spec.rpartition(mark)
            return Path(head), mark + tail
    return path, ""


def load(spec: str) -> Folder:
    """The folder ``spec`` names, read, its runs timed, and the selection after ``#`` or ``@`` applied."""
    path, selection = split_spec(spec)
    if not path.is_dir():
        raise SystemExit(f"{path} is no folder: give a cell's logs folder (the one holding robot/, models/, api/)")
    records = read_folder(path)
    events = events_of(records)
    sessions = sessions_of(events)
    runs = runs_of(events, sessions)
    commands = commands_of(events)
    used: set[int] = set()
    for run in runs:
        if run.kind == "task" and not run.restarts:
            run.command = _command_before(run, commands, used)
        if run.kind in ("task", "pick", "logs"):
            run.units = timeline(run, events)
    first = records[0].at if records else None
    last = records[-1].at if records else None
    runs = _selected(runs, sessions, selection, last)
    return Folder(path=path, selection=selection, sessions=sessions, runs=runs, first=first, last=last,
                  files=sorted(log_files(path)))


def _command_before(run: Run, commands: list[Command], used: set[int]) -> Command | None:
    """The task's sentence: the last one read as a task before it started, within ten minutes, not another task's."""
    best = None
    for index, command in enumerate(commands):
        if index in used or command.intent != "task" or command.at > run.start:
            continue
        if run.start - command.at <= timedelta(minutes=10):
            best = index
    if best is None:
        return None
    used.add(best)
    return commands[best]


def _selected(runs: list[Run], sessions: list[Session], selection: str, last: datetime | None) -> list[Run]:
    if not selection:
        return runs
    if selection.startswith("#"):
        which = selection[1:]
        with_runs = sorted({run.session for run in runs if run.session})
        if which == "last":
            if not with_runs:
                raise SystemExit("no connect in this folder holds a run")
            number = with_runs[-1]
        elif which.isdigit():
            number = int(which)
        else:
            raise SystemExit(f"{selection!r}: a connect is named by its number or 'last' (#2, #last)")
        return [run for run in runs if run.session == number]
    window = _WINDOW.match(selection[1:])
    if window is None:
        raise SystemExit(f"{selection!r}: a window is @HH:MM-HH:MM (either end may be left out)")
    begin = _instant(window["a"], last)
    until = _instant(window["b"] or "", begin or last)
    return [run for run in runs
            if (begin is None or run.start >= begin) and (until is None or run.start <= until)]


# ---------------------------------------------------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------------------------------------------------


def picks_of(runs: list[Run]) -> list[Unit]:
    """The picks that ended: one cut short by a disconnect or a console that stopped is no pick to time."""
    return [unit for run in runs for unit in run.units if unit.kind == "pick" and unit.outcome]


def cut_short(runs: list[Run]) -> int:
    return sum(1 for run in runs for unit in run.units if unit.kind == "pick" and not unit.outcome)


def _placed(unit: Unit) -> bool:
    """Whether a place let its part go: its jaws opened, or the line out ran (a URCap hand logs no opening)."""
    return unit.kind == "place" and ("release" in unit.stages or "line_out" in unit.stages)


def _tried(unit: Unit) -> bool:
    """Whether a pick tried a grasp: a pick that found nothing to grip is an empty look, timed apart."""
    return bool(unit.counts["tries"]) or unit.outcome == "succeeded"


def tried(units: list[Unit]) -> list[Unit]:
    return [unit for unit in units if _tried(unit)]


def stage_values(runs: list[Run]) -> dict[str, list[float]]:
    """Every stage's seconds, one value per pick, place, task or connect that went through it."""
    values: dict[str, list[float]] = defaultdict(list)
    for run in runs:
        if run.kind == "task":
            if run.command is not None:
                values["command"].append(run.command.ms / 1000.0)
                values["to_start"].append(max(0.0, (run.start - run.command.at).total_seconds()))
            survey = next((unit for unit in run.units if unit.kind == "survey"), None)
            if survey is not None:
                values["survey"].append(survey.total)
            values["task"].append(run.seconds if run.seconds is not None else
                                  (run.end - run.start).total_seconds() if run.end else 0.0)
        units = run.units
        for index, unit in enumerate(units):
            if unit.kind == "pick" and not unit.outcome:
                continue
            if unit.kind == "pick" and _tried(unit):
                values["pick"].append(unit.total)
                for name in _PICK_STAGES:
                    if name in unit.stages:
                        values[name].append(unit.stages[name])
                for name, _ in INSIDE:
                    if name in unit.inside:
                        values["in " + name].append(unit.inside[name])
                after = units[index + 1] if index + 1 < len(units) else None
                if after is not None and _placed(after):
                    values["cycle"].append(unit.total + after.total)
            elif unit.kind == "pick":
                values["empty pick"].append(unit.total)
            elif _placed(unit):
                # A place that let no part go (a task that stopped with the part in the jaws) is no place to time.
                values["place"].append(unit.total)
                for name in _PLACE_STAGES:
                    if name in unit.stages:
                        values[name].append(unit.stages[name])
        if run.kind == "wave" and run.seconds is not None:
            values["wave"].append(run.seconds)
    return values


def cell_values(folder: Folder) -> dict[str, list[float]]:
    """The connects' own numbers, for the connects the selection's runs ran in."""
    numbers = {run.session for run in folder.runs}
    values: dict[str, list[float]] = defaultdict(list)
    for session in folder.sessions:
        if session.number in numbers or not folder.selection:
            for name in ("cell_build_s", "planner_start_s", "qwen_load_s"):
                if name in session.facts:
                    values[name[:-2]].append(float(session.facts[name]))
    return values


def quality(runs: list[Run]) -> dict[str, Any]:
    """What the speed was bought with: picks gripped, parts placed, tries, refusals, how the tasks ended."""
    picks = picks_of(runs)
    tried_ = tried(picks)
    places = [unit for run in runs for unit in run.units if unit.kind == "place"]
    reasons: Counter[str] = Counter()
    for unit in picks:
        reasons.update(unit.reasons)
    tries = [unit.counts["tries"] for unit in tried_]
    return {
        "task runs": sum(1 for run in runs if run.kind == "task"),
        "task ends": dict(Counter(run.code or "(no end logged)" for run in runs if run.kind == "task")),
        "picks with a grasp tried": len(tried_),
        "picks gripped": sum(1 for unit in tried_ if unit.outcome == "succeeded"),
        "parts placed": sum(1 for unit in places if _placed(unit)),
        "empty picks (nothing to grip)": len(picks) - len(tried_),
        "picks cut short (no end logged)": cut_short(runs),
        "tries per pick (median)": statistics.median(tries) if tries else None,
        "tries refused before moving": sum(unit.counts["refused"] for unit in picks),
        "tries failed after a motion": sum(unit.counts["failed_sent"] for unit in picks),
        "refusals by what": dict(reasons.most_common()),
        "pick outcomes": dict(Counter(unit.outcome or "(no end logged)" for unit in picks)),
        "waves": sum(1 for run in runs if run.kind == "wave"),
    }


def _row_order(values: dict[str, list[float]]) -> list[tuple[str, int]]:
    """The rows printed, in order, each with its indent."""
    rows: list[tuple[str, int]] = [("command", 0), ("to_start", 0), ("survey", 0), ("task", 0), ("pick", 0)]
    for name in _PICK_STAGES:
        rows.append((name, 1))
        if name == "grounding":
            rows += [("in qwen", 2), ("in sam2", 2), ("in colour", 2)]
    rows += [("in world", 1), ("in camera", 2), ("in register", 2), ("in curobo", 1), ("empty pick", 0),
             ("place", 0)]
    rows += [(name, 1) for name in _PLACE_STAGES]
    rows += [("cycle", 0), ("wave", 0), ("cell_build", 0), ("planner_start", 0), ("qwen_load", 0)]
    return [(name, indent) for name, indent in rows if values.get(name)]


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}"


def _stats(values: list[float]) -> tuple[int, float | None, float | None, float | None]:
    if not values:
        return 0, None, None, None
    return len(values), min(values), statistics.median(values), max(values)


# ---------------------------------------------------------------------------------------------------------------------
# What it prints
# ---------------------------------------------------------------------------------------------------------------------


def describe(folder: Folder, out: list[str]) -> None:
    """The folder, its connects and its runs."""
    span = (f"{folder.first:%Y-%m-%d %H:%M:%S} to {folder.last:%H:%M:%S}" if folder.first and folder.last
            else "no line read")
    out.append(f"{folder.path}{folder.selection}: {span}, {len(folder.files)} log file(s)")
    if not any(name.startswith("models/") for name in folder.files):
        out.append("  no models/*.log: a grounding is timed inside its look, and Qwen and SAM2 are not read")
    if not any(name.startswith("api/") for name in folder.files):
        out.append("  no api/*.log: no runs or sentences, and the picks are told apart by their grasp records")
    numbers = {run.session for run in folder.runs}
    for session in folder.sessions:
        if session.number not in numbers:
            continue
        facts = session.facts
        said = ", ".join(f"{label} {facts[key]:.1f} s" for key, label in (
            ("cell_build_s", "built in"), ("planner_start_s", "planner ready after"),
            ("qwen_load_s", "Qwen loaded in")) if key in facts)
        end = f"{session.end:%H:%M:%S}" if session.end else "still connected"
        out.append(f"  connect #{session.number}: {session.start:%H:%M:%S} to {end}"
                   + (f", config {session.fingerprint}" if session.fingerprint else "") + (f"; {said}" if said else "")
                   + f"  (select it: {folder.path}#{session.number})")
        for key, label in (("workers", "grasp workers"), ("qwen", "Qwen"), ("planner", "planner")):
            if key in facts:
                out.append(f"      {label}: {_short(str(facts[key]), 110)}")
        for said in facts.get("said", []):
            out.append(f"      said: {_short(said, 110)}")
    for run in folder.runs:
        out.append("  " + _run_line(run))


def _short(text: str, width: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 3] + "..."


def _run_line(run: Run) -> str:
    seconds = f"{run.seconds:.1f} s" if run.seconds is not None else (
        f"{(run.end - run.start).total_seconds():.1f} s" if run.end else "no end logged")
    line = f"{run.start:%H:%M:%S} {run.id} {run.kind or '?'} -> {run.code or '?'} in {seconds}"
    if run.kind == "task":
        what = f"'{run.object}' into {run.target}, {run.scope}" + (f", restarting {run.restarts}" if run.restarts
                                                                   else "")
        picks = [unit for unit in run.units if unit.kind == "pick"]
        placed = sum(1 for unit in run.units if _placed(unit))
        line += f": {what}; {len(picks)} pick(s), {placed} placed"
        if run.command is not None:
            line += f"; read in {run.command.ms / 1000.0:.1f} s" + (f" ({run.command.how})" if run.command.how
                                                                     else "")
    return line


def table(values: dict[str, list[float]], out: list[str]) -> None:
    out.append(f"{'stage':<24}{'n':>4}{'best':>9}{'median':>9}{'worst':>9}   seconds")
    for name, indent in _row_order(values):
        n, best, median, worst = _stats(values[name])
        out.append(f"{'  ' * indent + name:<24}{n:>4}{_fmt(best):>9}{_fmt(median):>9}{_fmt(worst):>9}")


def compare(a: dict[str, list[float]], b: dict[str, list[float]], out: list[str]) -> None:
    merged = {name: a.get(name, []) or b.get(name, []) for name in set(a) | set(b)}
    out.append(f"{'stage':<24}{'A n':>4}{'A median':>10}{'(best-worst)':>17}{'B n':>5}{'B median':>10}"
               f"{'(best-worst)':>17}{'B-A':>9}")
    for name, indent in _row_order(merged):
        na, besta, meda, worsta = _stats(a.get(name, []))
        nb, bestb, medb, worstb = _stats(b.get(name, []))
        delta = "" if meda is None or medb is None else f"{medb - meda:+.1f}"
        share = f"{100.0 * (medb - meda) / meda:+.0f} %" if meda and medb is not None else ""
        out.append(f"{'  ' * indent + name:<24}{na:>4}{_fmt(meda):>10}{_range(besta, worsta):>17}{nb:>5}"
                   f"{_fmt(medb):>10}{_range(bestb, worstb):>17}{delta:>9}{share:>8}".rstrip())


def _range(best: float | None, worst: float | None) -> str:
    return "" if best is None or worst is None else f"({best:.1f}-{worst:.1f})"


def quality_lines(sides: list[tuple[str, dict[str, Any]]], out: list[str]) -> None:
    keys = list(sides[0][1])
    width = 40 if len(sides) > 1 else 200
    head = f"{'quality':<34}" + "".join(f"{name:<{width}}" for name, _ in sides)
    out.append(head.rstrip())
    for key in keys:
        cells = []
        for _, said in sides:
            value = said.get(key)
            if isinstance(value, dict):
                value = ", ".join(f"{k} {v}" for k, v in value.items()) or "-"
            elif value is None:
                value = "-"
            cells.append(_short(str(value), width - 2))
        out.append(f"{key:<34}" + "".join(f"{cell:<{width}}" for cell in cells).rstrip())


def pick_lines(folder: Folder, out: list[str]) -> None:
    """One line per pick, one per place, under its run."""
    for run in folder.runs:
        if not run.units:
            continue
        out.append(_run_line(run))
        for unit in run.units:
            if unit.kind in ("between", "place") and unit.total < 0.05:
                continue
            stages = " | ".join(f"{name} {seconds:.1f}" for name in unit.order
                                if (seconds := unit.stages.get(name, 0.0)) >= 0.05 or name in ("look", "approach"))
            inside = ", ".join(f"{name} {unit.inside[name]:.1f}" for name, _ in INSIDE if unit.inside.get(name))
            counts = ", ".join(f"{name} {unit.counts[name]}" for name, _ in COUNTS if unit.counts.get(name))
            head = f"  {unit.start:%H:%M:%S} {unit.kind}"
            if unit.kind == "pick":
                head += f" {unit.pick or ''} (part {unit.part or '?'}) {unit.outcome or '?'}"
            out.append(f"{head}: {unit.total:.1f} s = {stages}")
            if inside or counts:
                out.append(f"      in it: {inside or '-'}; {counts or 'no counts'}")
            if unit.reasons:
                out.append("      refused: " + ", ".join(f"{why} {n}" for why, n in unit.reasons.most_common()))


def as_json(folder: Folder) -> dict[str, Any]:
    def unit_json(unit: Unit) -> dict[str, Any]:
        return {"kind": unit.kind, "start": unit.start.isoformat(), "end": unit.end.isoformat() if unit.end else None,
                "total_s": round(unit.total, 3), "stages_s": {k: round(v, 3) for k, v in unit.stages.items()},
                "inside_s": {k: round(v, 3) for k, v in unit.inside.items()}, "counts": dict(unit.counts),
                "refusals": dict(unit.reasons), "outcome": unit.outcome, "part": unit.part, "pick": unit.pick}

    return {
        "folder": str(folder.path), "selection": folder.selection, "files": folder.files,
        "sessions": [{"number": s.number, "start": s.start.isoformat(), "end": s.end.isoformat() if s.end else None,
                      "fingerprint": s.fingerprint, "facts": s.facts} for s in folder.sessions],
        "runs": [{"id": r.id, "kind": r.kind, "start": r.start.isoformat(),
                  "end": r.end.isoformat() if r.end else None, "code": r.code, "seconds": r.seconds,
                  "object": r.object, "target": r.target, "scope": r.scope, "restarts": r.restarts,
                  "session": r.session,
                  "command": None if r.command is None else {"at": r.command.at.isoformat(), "ms": r.command.ms,
                                                             "how": r.command.how},
                  "units": [unit_json(u) for u in r.units]} for r in folder.runs],
        "stages_s": {k: v for k, v in stage_values(folder.runs).items()},
        "quality": quality(folder.runs),
    }


def main(argv: list[str] | None = None) -> int:
    head, _, rest = (__doc__ or "").partition("\n\n")
    parser = argparse.ArgumentParser(description=head, epilog=rest, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", nargs="*", help="one logs folder, or two to set A against B (folder#N, folder@HH:MM-HH:MM)")
    parser.add_argument("--picks", action="store_true", help="one line per pick and per place as well")
    parser.add_argument("--json", type=Path, help="write everything read to this file")
    parser.add_argument("--stages", action="store_true", help="say what each stage holds, and stop")
    args = parser.parse_args(argv)
    out: list[str] = []
    if args.stages:
        for name, part, said in STAGES:
            out.append(f"{name:<16}{part:<7}{said}")
        out.append("")
        for name, said in INSIDE:
            out.append(f"in {name:<13}{'':<7}{said}")
        _say(out)
        return 0
    if not 1 <= len(args.logs) <= 2:
        parser.error("give one logs folder, or two to set A against B")
    folders = [load(spec) for spec in args.logs]
    empty = [folder for folder in folders if not folder.files]
    if empty:
        _say([f"{folder.path} holds no log file (robot/robot.log, models/*.log, api/*.log)" for folder in empty])
        return 2
    for label, folder in zip("AB" if len(folders) == 2 else " ", folders):
        if len(folders) == 2:
            out.append(f"== {label}")
        describe(folder, out)
        out.append("")
    if len(folders) == 1:
        folder = folders[0]
        values = {**stage_values(folder.runs), **cell_values(folder)}
        if not values:
            out.append("no pick, place, task or connect was found in what was selected")
        else:
            table(values, out)
        out.append("")
        quality_lines([("", quality(folder.runs))], out)
        if args.picks:
            out.append("")
            pick_lines(folder, out)
    else:
        a, b = folders
        compare({**stage_values(a.runs), **cell_values(a)}, {**stage_values(b.runs), **cell_values(b)}, out)
        out.append("")
        quality_lines([("A", quality(a.runs)), ("B", quality(b.runs))], out)
        if args.picks:
            for label, folder in (("A", a), ("B", b)):
                out.append("")
                out.append(f"== {label}, pick by pick")
                pick_lines(folder, out)
    if args.json is not None:
        payload = [as_json(folder) for folder in folders]
        args.json.write_text(json.dumps(payload[0] if len(payload) == 1 else {"A": payload[0], "B": payload[1]},
                                        indent=1, ensure_ascii=False, default=str), encoding="utf-8")
        out.append("")
        out.append(f"written: {args.json}")
    _say(out)
    return 0


def _say(lines: list[str]) -> None:
    text = "\n".join(lines) + "\n"
    try:
        sys.stdout.write(text)
    except UnicodeEncodeError:
        sys.stdout.write(text.encode("ascii", "replace").decode("ascii"))


if __name__ == "__main__":
    sys.exit(main())
