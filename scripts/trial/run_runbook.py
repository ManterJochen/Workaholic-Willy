"""Run a runbook's own commands against a plan, so the runbook is the commands that ran.

    .venv/Scripts/python.exe scripts/trial/run_runbook.py --runbook docs/runbooks/your_own_gripper.md \\
        --plan scripts/trial/customer_hand_trial.json --root D:/willy_trial/customer_hand --logs LOGDIR [--from ID]

A trial instrument, not product code. Standard library only.

The runbook. Every block the trial runs is a fenced ``bash`` or ``yaml`` block directly preceded by a marker line::

    <!-- step: fit-write -->
    <!-- step: cell-layer file: config/robot/robot.$CELL.yaml -->
    <!-- trial-skip: needs a physical controller in Remote Control -->   (on the line before a step marker)

A bash step holds one command (a trailing backslash continues it, and ``#`` lines are comments). A yaml step is
written to its file, with ``$NAME`` substituted and LF endings.

The plan, in JSON. ``env`` ``{"set": {...}, "unset": [...]}``; ``bindings`` for every phase; ``phases``, each with its
own ``bindings`` and ``steps``. A step is ``{"runbook": ID}``, ``{"id": ID, "run": COMMAND, "cwd": DIR}``,
``{"hide": PATH}`` or ``{"restore": PATH}``, with ``expect`` ``{"exit": N, "stdout_has": [...], "stdout_lacks": [...],
"stderr_has": [...], "stderr_lacks": [...], "files_exist": [...], "files_absent": [...]}`` and an optional ``timeout_s``.
``env`` values are substituted from the plan's own ``bindings``, ``ROOT`` and ``LOGS``. ``$NAME`` and ``${NAME}`` are
substituted from the bindings, with ``ROOT`` and ``LOGS`` always bound, and an unbound name anywhere refuses the plan
before anything runs. Commands run without a shell, from the copy root unless a step says otherwise, each with its own
stdout and stderr file and a timeout that kills the whole process tree (a sidecar is a child). The first unexpected result stops
the run; every hidden file is restored in a ``finally`` and its bytes verified. The record binds the run to the text:
``runbook_steps_sha256`` over every step's id, kind and text in document order.

Exit codes: 0 every step as expected, 1 stopped at a step, 2 the plan or the runbook is malformed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_STEP = re.compile(r"^<!--\s*step:\s*(?P<id>[A-Za-z0-9_.-]+)(?:\s+file:\s*(?P<file>\S+))?\s*-->\s*$")
_SKIP = re.compile(r"^<!--\s*trial-skip:\s*(?P<reason>.*?)\s*-->\s*$")
_FENCE = re.compile(r"^```(?P<lang>[A-Za-z0-9_-]*)\s*$")
_NAME = re.compile(r"\$\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|\$(?P<bare>[A-Za-z_][A-Za-z0-9_]*)")


class Malformed(Exception):
    """The runbook or the plan cannot be run as written: exit 2."""


@dataclass(frozen=True)
class Step:
    id: str
    kind: str  # "bash" or "yaml"
    text: str
    line: int
    file: str = ""
    skip: str = ""


def parse_runbook(text: str) -> list[Step]:
    """Every marked block, in document order. Refuses a duplicate id, an empty skip reason and a marker with no block."""
    lines = text.replace("\r\n", "\n").split("\n")
    steps: list[Step] = []
    seen: dict[str, int] = {}
    pending_skip = ""
    index = 0
    while index < len(lines):
        line = lines[index]
        skip = _SKIP.match(line)
        if skip:
            if not skip.group("reason"):
                raise Malformed(f"line {index + 1}: a trial-skip marker needs a reason")
            pending_skip = skip.group("reason")
            index += 1
            continue
        marker = _STEP.match(line)
        if not marker:
            if line.strip():
                pending_skip = ""
            index += 1
            continue
        step_id = marker.group("id")
        if step_id in seen:
            raise Malformed(f"line {index + 1}: step {step_id!r} is already marked at line {seen[step_id]}")
        seen[step_id] = index + 1
        fence = _FENCE.match(lines[index + 1]) if index + 1 < len(lines) else None
        if fence is None or fence.group("lang") not in ("bash", "yaml"):
            raise Malformed(f"line {index + 1}: step {step_id!r} is not followed by a ```bash or ```yaml block")
        body: list[str] = []
        cursor = index + 2
        while cursor < len(lines) and not lines[cursor].startswith("```"):
            body.append(lines[cursor])
            cursor += 1
        if cursor >= len(lines):
            raise Malformed(f"line {index + 1}: the block of step {step_id!r} is never closed")
        kind = fence.group("lang")
        if kind == "yaml" and not marker.group("file"):
            raise Malformed(f"line {index + 1}: yaml step {step_id!r} names no file")
        steps.append(Step(id=step_id, kind=kind, text="\n".join(body) + "\n", line=index + 1,
                          file=marker.group("file") or "", skip=pending_skip))
        pending_skip = ""
        index = cursor + 1
    return steps


def command_of(step: Step) -> str:
    """The one command a bash step holds: comment lines dropped, backslash continuations joined."""
    joined = re.sub(r"\\\n", " ", step.text)
    commands = [line.strip() for line in joined.split("\n") if line.strip() and not line.strip().startswith("#")]
    if len(commands) != 1:
        raise Malformed(f"line {step.line}: bash step {step.id!r} holds {len(commands)} commands, and a step is one")
    return commands[0]


def steps_sha256(steps: list[Step]) -> str:
    digest = hashlib.sha256()
    for step in steps:
        for part in (step.id, step.kind, step.text):
            digest.update(part.encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


def names_in(text: str) -> set[str]:
    return {match.group("braced") or match.group("bare") for match in _NAME.finditer(text)}


def substitute(text: str, bindings: dict[str, str]) -> str:
    return _NAME.sub(lambda match: str(bindings[match.group("braced") or match.group("bare")]), text)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _kill_tree(process: subprocess.Popen) -> None:
    # sys.platform and not os.name: the same answer on every box, and the test mypy reads, so each platform types
    # the branch it can run. os.killpg, os.getpgid and signal.SIGKILL exist only on POSIX.
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)], capture_output=True, check=False)
    else:  # pragma: no cover (the box is Windows)
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()


@dataclass
class Runner:
    runbook_path: str
    steps: list[Step]
    plan: dict[str, Any]
    root: Path
    logs: Path

    def __post_init__(self) -> None:
        self.by_id = {step.id: step for step in self.steps}
        self.hidden: dict[str, tuple[Path, str]] = {}
        self.record: dict[str, Any] = {
            "runbook": self.runbook_path, "runbook_steps_sha256": steps_sha256(self.steps), "phases": [],
            "steps": [], "skipped": [{"id": step.id, "reason": step.skip} for step in self.steps if step.skip],
            "stopped_at": None,
        }

    def _bindings(self, phase: dict[str, Any]) -> dict[str, str]:
        """ROOT, LOGS, the plan's bindings and the phase's, a later one winning; a value may name an earlier binding.

        Resolved until nothing changes, so ``MESH_DIR: $LOGS/vendor`` is a path; a name no binding gives is left for
        :meth:`check` to refuse, and a binding that names itself is refused here.
        """
        # Forward slashes, because a posix split eats a backslash: D:\willy_trial becomes D:willy_trial, a path
        # relative to the drive's current directory, which can put a copy inside the original tree.
        bound = {"ROOT": self.root.as_posix(), "LOGS": self.logs.as_posix()}
        bound.update({key: str(value) for key, value in (self.plan.get("bindings") or {}).items()})
        bound.update({key: str(value) for key, value in (phase.get("bindings") or {}).items()})
        for _ in range(len(bound) + 1):
            resolved = {key: _NAME.sub(lambda m: bound.get(m.group("braced") or m.group("bare"), m.group(0)), value)
                        for key, value in bound.items()}
            if resolved == bound:
                return bound
            bound = resolved
        raise Malformed(f"the bindings of phase {phase.get('name')!r} name each other in a circle")

    def check(self) -> None:
        """Refuse the whole plan before anything runs: an unknown runbook id, an unbound name, a step expecting nothing."""
        env_bound = self._bindings({})
        for phase in [{}, *(self.plan.get("phases") or [])]:
            for name, value in self._bindings(phase).items():
                if "\\" in value:
                    raise Malformed(f"binding {name} holds a backslash ({value!r}), which splitting a command removes: "
                                    f"write the path with forward slashes")
        for name, value in ((self.plan.get("env") or {}).get("set") or {}).items():
            unbound = sorted(names_in(str(value)) - set(env_bound))
            if unbound:
                raise Malformed(f"env {name} uses {', '.join(unbound)}, which no plan binding gives")
        for phase in self.plan.get("phases") or []:
            bound = self._bindings(phase)
            for number, step in enumerate(phase.get("steps") or []):
                where = f"phase {phase.get('name')!r} step {number + 1}"
                texts: list[str] = []
                if "runbook" in step:
                    if step["runbook"] not in self.by_id:
                        raise Malformed(f"{where} names runbook step {step['runbook']!r}, which the runbook does not mark")
                    source = self.by_id[step["runbook"]]
                    texts += [source.text, source.file]
                elif "run" in step:
                    texts += [step["run"], step.get("cwd", "")]
                elif "hide" in step or "restore" in step:
                    texts.append(step.get("hide") or step.get("restore"))
                else:
                    raise Malformed(f"{where} is none of runbook, run, hide or restore")
                expect = step.get("expect") or {}
                texts += [*expect.get("stdout_has", []), *expect.get("stdout_lacks", []),
                          *expect.get("stderr_has", []), *expect.get("stderr_lacks", []),
                          *expect.get("files_exist", []), *expect.get("files_absent", [])]
                unbound = sorted(set().union(*(names_in(text) for text in texts)) - set(bound))
                if unbound:
                    raise Malformed(f"{where} uses {', '.join(unbound)}, which no binding gives")

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        for name in (self.plan.get("env") or {}).get("unset", []):
            env.pop(name, None)
        bound = self._bindings({})
        env.update({key: substitute(str(value), bound)
                    for key, value in ((self.plan.get("env") or {}).get("set") or {}).items()})
        env.update({"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"})
        return env

    def _verdict(self, expect: dict[str, Any], bound: dict[str, str], exit_code: "int | None", out: str,
                 err: str = "") -> str:
        if exit_code is not None and "exit" in expect and exit_code != int(expect["exit"]):
            return f"exit {exit_code}, expected {expect['exit']}"
        for text in expect.get("stdout_has", []):
            if substitute(text, bound) not in out:
                return f"stdout lacks {substitute(text, bound)!r}"
        for text in expect.get("stdout_lacks", []):
            if substitute(text, bound) in out:
                return f"stdout has {substitute(text, bound)!r}"
        for text in expect.get("stderr_has", []):
            if substitute(text, bound) not in err:
                return f"stderr lacks {substitute(text, bound)!r}"
        for text in expect.get("stderr_lacks", []):
            if substitute(text, bound) in err:
                return f"stderr has {substitute(text, bound)!r}"
        for path in expect.get("files_exist", []):
            if not (self.root / substitute(path, bound)).exists():
                return f"no file at {substitute(path, bound)}"
        for path in expect.get("files_absent", []):
            if (self.root / substitute(path, bound)).exists():
                return f"a file at {substitute(path, bound)}"
        return ""

    def _restore(self, relative: str) -> str:
        if relative not in self.hidden:
            return f"{relative} was never hidden"
        place, digest = self.hidden.pop(relative)
        if _sha256(place) != digest:
            return f"the hidden copy of {relative} at {place} has other bytes than were hidden; not restored"
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(place), str(target))
        return "" if _sha256(target) == digest else f"{relative} came back with other bytes"

    def run(self, *, from_id: "str | None" = None) -> int:
        self.check()
        self.logs.mkdir(parents=True, exist_ok=True)
        started = from_id is None
        env = self._env()
        counter = 0
        try:
            for phase in self.plan.get("phases") or []:
                bound = self._bindings(phase)
                self.record["phases"].append({"name": phase.get("name"), "bindings": bound})
                for step in phase.get("steps") or []:
                    step_id = step.get("runbook") or step.get("id") or ("hide" if "hide" in step else "restore")
                    if not started:
                        started = step_id == from_id
                        if not started:
                            continue
                    counter += 1
                    entry = self._one(phase, step, step_id, bound, env, counter)
                    self.record["steps"].append(entry)
                    if entry["verdict"] != "as expected":
                        self.record["stopped_at"] = {"phase": phase.get("name"), "id": step_id}
                        return 1
        finally:
            for relative in list(self.hidden):
                said = self._restore(relative)
                self.record.setdefault("restore_problems", []).extend([said] if said else [])
            (self.logs / "record.json").write_text(json.dumps(self.record, indent=1) + "\n", encoding="utf-8",
                                                   newline="\n")
        return 0 if not self.record.get("restore_problems") else 1

    def _one(self, phase: dict, step: dict, step_id: str, bound: dict, env: dict, counter: int) -> dict[str, Any]:
        began = time.perf_counter()
        expect = step.get("expect") or {}
        entry: dict[str, Any] = {"phase": phase.get("name"), "id": step_id}
        if "hide" in step or "restore" in step:
            named = step.get("hide") or step.get("restore")
            if not isinstance(named, str):
                raise Malformed(f"phase {phase.get('name')!r} step {step_id!r} names no path to hide or restore")
            relative = substitute(named, bound)
            if "hide" in step:
                source = self.root / relative
                if not source.is_file():
                    reason = f"nothing to hide at {relative}"
                else:
                    digest = _sha256(source)
                    place = self.logs / "hidden" / digest / source.name
                    place.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(source), str(place))
                    self.hidden[relative] = (place, digest)
                    reason = ""
            else:
                reason = self._restore(relative)
            entry.update({"source": "trial", "command": f"{'hide' if 'hide' in step else 'restore'} {relative}",
                          "exit": None, "seconds": round(time.perf_counter() - began, 3),
                          "verdict": "as expected" if not reason else "unexpected", "reason": reason})
            return entry
        if "runbook" in step:
            source_step = self.by_id[step["runbook"]]
            entry["source"] = "runbook"
            if source_step.kind == "yaml":
                target = self.root / substitute(source_step.file, bound)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(substitute(source_step.text, bound).encode("utf-8"))
                reason = self._verdict({"files_exist": [substitute(source_step.file, bound)], **expect}, bound, None, "")
                entry.update({"command": f"write {target.relative_to(self.root).as_posix()}", "exit": None,
                              "seconds": round(time.perf_counter() - began, 3),
                              "verdict": "as expected" if not reason else "unexpected", "reason": reason})
                return entry
            command = substitute(command_of(source_step), bound)
            cwd = self.root
        else:
            entry["source"] = "trial"
            command = substitute(step["run"], bound)
            cwd = Path(substitute(step["cwd"], bound)) if step.get("cwd") else self.root
        stem = f"{counter:03d}_{step_id}"
        out_path, err_path = self.logs / f"{stem}.stdout", self.logs / f"{stem}.stderr"
        timeout = float(step.get("timeout_s", 1800))
        with out_path.open("wb") as out, err_path.open("wb") as err:
            process = subprocess.Popen(shlex.split(command, posix=True), cwd=str(cwd), env=env, stdout=out, stderr=err,
                                       start_new_session=sys.platform != "win32")
            try:
                code: "int | None" = process.wait(timeout=timeout)
                timed_out = False
            except subprocess.TimeoutExpired:
                _kill_tree(process)
                code, timed_out = None, True
        printed = out_path.read_text(encoding="utf-8", errors="replace")
        complained = err_path.read_text(encoding="utf-8", errors="replace")
        reason = (f"timed out after {timeout:g} s" if timed_out
                  else self._verdict(expect, bound, code, printed, complained))
        entry.update({"command": command, "exit": code, "seconds": round(time.perf_counter() - began, 3),
                      "verdict": "as expected" if not reason else "unexpected", "reason": reason})
        return entry


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description="Run a runbook's own commands against a trial plan.")
    parser.add_argument("--runbook", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--root", required=True, help="the copy the commands run in")
    parser.add_argument("--logs", required=True, help="where each step's output and the record go")
    parser.add_argument("--from", dest="from_id", default=None, help="resume at this step inside one copy")
    args = parser.parse_args(argv)
    try:
        runner = Runner(runbook_path=args.runbook,
                        steps=parse_runbook(Path(args.runbook).read_text(encoding="utf-8")),
                        plan=json.loads(Path(args.plan).read_text(encoding="utf-8")),
                        root=Path(args.root), logs=Path(args.logs))
        code = runner.run(from_id=args.from_id)
    except Malformed as exc:
        print(f"malformed: {exc}", file=sys.stderr)
        return 2
    stopped = runner.record["stopped_at"]
    last = runner.record["steps"][-1] if runner.record["steps"] else None
    if stopped and last:
        print(f"STOPPED at {stopped['phase']} {stopped['id']}: {last['reason']}")
    else:
        print(f"ran {len(runner.record['steps'])} steps as expected; record in {Path(args.logs) / 'record.json'}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
