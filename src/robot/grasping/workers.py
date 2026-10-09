"""SFE's units on a warm pool of worker processes: one part's grasp search spread over the cores, with the same answer.

The owner's cell, 2026-10-08: one boxed-in cube took 24 of a look's 30 s, every grasp the support-footprint stage tried
built one after another in the one process. A part's search is a set of independent units, one closing line at one
height each (``support_footprint.SfeUnit``), merged in the order SFE runs them, so any process may run any of them.
:class:`SfeWorkers` runs them on worker processes started once, when the cell is built (``robot.grasping.workers``),
and SFE merges what they answer as it merges its own units: the same candidates, the same refusal counts and the same
telemetry, bit for bit.

Only SFE's units leave the cell's process. The frame, the masks, the scene obstacles, the post-filters, the overlay, the
log and every motion stay in it, and the exact guard judges every sample of every path there, as before. The arm stands
still while grasps are computed, so the workers may take every core but one (``auto``); they run at the console's own
priority, each with one BLAS thread so that they do not crowd each other. A worker holds no driver, camera, GPU or cell
lock, and writes no log line. It is a process of its own that runs the worker loop alone and connects back to the
cell's process with a key only the two know: never the cell's main module, so a program without a ``__main__`` guard
that builds a cell and moves the arm is never run again in a worker.

How a part reaches them: its inputs (``support_footprint.SfeInputs``) are pickled once into shared memory, and the depth
of the frame the side approaches' seen test reads (``scene_obstacles.SeenInTheFrame``) once per look. Each worker plans
the part from them once (``plan_support_footprint``) and keeps the plan for the units that follow, a second ranking of
the same part included. A unit goes to whichever worker is free, a few at a time, and a line's heights stay together, so
where the inputs ask for it (``SfeInputs.batched``, ``robot.grasping.batched_builds``) a worker makes each line's builds
at once (``support_footprint.run_units``), as the cell's process would.

Anything that goes wrong (inputs that do not pickle, a worker that raises, dies or does not answer within
:data:`TIMEOUT_S`) has that part searched in the cell's process, from the start, as without workers, said in one log
line (``support_footprint.SfeRunnerFailed``); nothing a worker answered for it is kept. A pool that broke is started
again for the next part; one that breaks again, or cannot be started, is given up for the process, and every part is
searched in it.
"""

from __future__ import annotations

import atexit
import dataclasses
import hashlib
import logging
import math
import os
import pickle
import queue
import subprocess
import sys
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Sequence
from multiprocessing import connection as _connection
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any, Final

import numpy as np

from src.robot.grasping.generation.scene_obstacles import SeenInTheFrame
from src.robot.grasping.generation.support_footprint import (
    SfePlan,
    SfeRunnerFailed,
    SfeUnit,
    SupportFootprintCandidate,
    plan_support_footprint,
    run_units,
)

__all__ = ["MOST_WORKERS", "TIMEOUT_S", "SfeWorkers", "shared_workers", "worker_count"]

_LOG = logging.getLogger(__name__)

#: How long the workers have to answer for one search of a part (its coarse units, or its fine and rolled ones),
#: seconds. Far past the longest search measured on the owner's cell (24 s, alone in one process): only a worker that
#: hangs reaches it, and the part is then searched in the cell's process.
TIMEOUT_S: Final[float] = 60.0
#: How long every worker has to start and import the search, seconds: 0.6 s each on the desk, about 2 s on the cell.
_START_TIMEOUT_S: Final[float] = 120.0
#: How many chunks of units each worker gets of one search, about: enough that a worker done early takes more, few
#: enough that handing them over costs nothing worth measuring.
_CHUNKS_PER_WORKER: Final[int] = 4
#: How many parts' inputs the cell's process keeps in shared memory, the latest first: every part of a look, and more.
_INPUTS_KEPT: Final[int] = 16
#: How many parts' plans a worker keeps, and how many looks' depths.
_PLANS_KEPT: Final[int] = 8
_LOOKS_KEPT: Final[int] = 2
#: The most workers a pool runs: the cell's process waits on every worker's connection at once, and Windows waits on
#: 63 handles at most.
MOST_WORKERS: Final[int] = 30
#: Where ``src`` stands, for a worker's ``import src...``.
_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
#: What a worker process runs, and nothing else: the worker loop, by the module's own name. Never the cell's main
#: module, which a spawned process would run again: a program without a ``__main__`` guard that builds a cell and
#: moves the arm would build it and move it again in every worker.
_WORKER_MAIN: Final[str] = "import sys; from src.robot.grasping.workers import _serve; _serve(sys.argv[1], sys.argv[2])"


def worker_count(setting: Any) -> tuple[int, str]:
    """How many worker processes ``robot.grasping.workers`` asks for, and how that was read, for the log.

    ``0`` none: SFE searches in the cell's process. ``"auto"`` every physical core but one, read through ``psutil``
    (which the environment carries for another package) and, where it does not import, half the logical cores less
    one; never fewer than one. ``N`` that many. Never more than :data:`MOST_WORKERS`, and the sentence says so.
    Anything else is refused (``ValueError``).
    """
    if isinstance(setting, str) and setting.strip().lower() == "auto":
        physical = None
        try:
            import psutil  # noqa: PLC0415 (only an auto setting asks for the cores)

            physical = psutil.cpu_count(logical=False)
        except Exception:  # noqa: BLE001 (no psutil: the logical cores answer instead)
            physical = None
        if physical:
            count = max(1, int(physical) - 1)
            said = f"auto: {count} worker(s), the {int(physical)} physical cores but one"
        else:
            logical = int(os.cpu_count() or 2)
            count = max(1, logical // 2 - 1)
            said = f"auto: {count} worker(s), half the {logical} logical cores less one (psutil did not say)"
    elif isinstance(setting, bool) or not isinstance(setting, int) or setting < 0:
        raise ValueError(f"robot.grasping.workers is 0, 'auto' or a number of workers, not {setting!r}")
    elif setting == 0:
        return 0, "no workers: SFE searches in the cell's process"
    else:
        count, said = int(setting), f"{int(setting)} worker(s), as configured"
    if count > MOST_WORKERS:
        return MOST_WORKERS, f"{said}; {MOST_WORKERS} at most"
    return count, said


# ---------------------------------------------------------------------------------------------------------------------
# What a part's inputs are handed over as
# ---------------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _SharedRays:
    """A :class:`SeenInTheFrame` whose depth stands in shared memory, as a worker is handed it: the block's name, the
    depth's shape and type, and the rest of the seen test as it was."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    to_camera: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float
    tolerance_mm: float

    def in_the_frame(self, depth_mm: np.ndarray) -> SeenInTheFrame:
        """The seen test again, reading ``depth_mm``, the block's depth."""
        return SeenInTheFrame(depth_mm=depth_mm, to_camera=self.to_camera, fx=self.fx, fy=self.fy, cx=self.cx,
                              cy=self.cy, tolerance_mm=self.tolerance_mm)


def _buffer(block: SharedMemory) -> memoryview:
    """The memory of an open block; only a closed one has none."""
    buffer = block.buf
    if buffer is None:  # pragma: no cover (every block here is read while it is open)
        raise ValueError(f"the shared memory {block.name} is closed")
    return buffer


def _free(block: SharedMemory) -> None:
    """Let go of a block of shared memory, and never raise: it goes once every process that holds it let go."""
    try:
        block.close()
    except Exception:  # noqa: BLE001 (a view still held keeps it until the process ends)
        pass
    try:
        block.unlink()
    except Exception:  # noqa: BLE001 (already gone, or Windows, where the last handle frees it)
        pass


# ---------------------------------------------------------------------------------------------------------------------
# A worker
# ---------------------------------------------------------------------------------------------------------------------

#: The worker's own limit on its BLAS threads, kept for as long as the worker lives (``threadpoolctl``).
_LIMITS: Any = None
#: Whether this process is an SFE worker (:func:`_serve`), not the cell's process.
_IN_A_WORKER = False


def _quiet() -> None:
    """A worker writes no log line, warns of nothing and leaves Ctrl+C to the cell's process; its BLAS runs on one
    thread: the workers share the cores between them, not each among its own threads. One thread or many give the same
    grasps (the parallel map, 2026-10-08), so a limit that cannot be set changes nothing but the speed."""
    global _LIMITS  # noqa: PLW0603 (one limit for the process's life)
    import signal  # noqa: PLC0415
    import warnings  # noqa: PLC0415

    logging.disable(logging.CRITICAL)
    warnings.simplefilter("ignore")
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (ValueError, OSError):  # pragma: no cover (a worker's main thread can always set it)
        pass
    if os.name != "nt":
        # A worker reads the blocks the cell's process made and lets go of them; it never owns one. Python's resource
        # tracker would take an attached block for the worker's own and unlink it when the worker ends.
        from multiprocessing import resource_tracker  # noqa: PLC0415

        resource_tracker.register = lambda name, rtype: None  # type: ignore[assignment]
    try:
        from threadpoolctl import threadpool_limits  # noqa: PLC0415 (scikit-learn's, where it is installed)

        _LIMITS = threadpool_limits(limits=1)
    except Exception:  # noqa: BLE001 (the threads are a matter of speed, never of the answer)
        _LIMITS = None


class _WorkerPlans:
    """What a worker keeps between jobs: the plans of the parts it was handed lately, and the depths they read."""

    def __init__(self) -> None:
        self._plans: OrderedDict[str, SfePlan] = OrderedDict()
        self._reads: dict[str, str] = {}
        self._depths: OrderedDict[str, tuple[SharedMemory, np.ndarray]] = OrderedDict()

    def run(self, key: str, payload: tuple[str, int], units: Sequence[SfeUnit]) -> tuple[dict, dict[str, int]]:
        """Run ``units`` of the part ``key``: every unit's grasps, by unit, and what its coarse units refused."""
        plan = self._plan(key, payload)
        counted: dict[str, int] = {}
        return run_units(plan, units, counted if plan.inputs.counting else None), counted

    def _plan(self, key: str, payload: tuple[str, int]) -> SfePlan:
        plan = self._plans.get(key)
        if plan is not None:
            self._plans.move_to_end(key)
            return plan
        name, size = payload
        block = SharedMemory(name=name)
        try:
            inputs = pickle.loads(bytes(_buffer(block)[:size]))
        finally:
            block.close()
        seen = inputs.corridor_seen
        if isinstance(seen, _SharedRays):
            inputs = dataclasses.replace(inputs, corridor_seen=seen.in_the_frame(self._depth(seen)))
            self._reads[key] = seen.name
        planned = plan_support_footprint(inputs)
        if planned is None:
            raise ValueError("the part's cloud gave no prism here, where the cell's process planned one")
        self._plans[key] = planned
        while len(self._plans) > _PLANS_KEPT:
            gone, _ = self._plans.popitem(last=False)
            self._reads.pop(gone, None)
        return planned

    def _depth(self, seen: _SharedRays) -> np.ndarray:
        got = self._depths.get(seen.name)
        if got is not None:
            self._depths.move_to_end(seen.name)
            return got[1]
        block = SharedMemory(name=seen.name)
        depth = np.ndarray(seen.shape, dtype=np.dtype(seen.dtype), buffer=_buffer(block))
        self._depths[seen.name] = (block, depth)
        while len(self._depths) > _LOOKS_KEPT:
            name, (old, _view) = self._depths.popitem(last=False)
            for key in [key for key, read in self._reads.items() if read == name]:
                self._plans.pop(key, None)
                self._reads.pop(key, None)
            del _view
            try:
                old.close()
            except BufferError:  # pragma: no cover (a view somebody still holds keeps it until the process ends)
                pass
        return depth


def _serve(address: str, authkey: str) -> None:
    """One worker's life (:data:`_WORKER_MAIN`): quiet itself, import the search, connect to the cell's process at
    ``address`` with ``authkey``, take its import path (so a part's inputs unpickle here as they pickled there) and say
    it is ready, then answer jobs until it is told to stop or the connection closes, the cell's process gone. A job
    that raises is answered with what it raised."""
    global _IN_A_WORKER  # noqa: PLW0603 (said once, for the process's life)
    _IN_A_WORKER = True
    _quiet()
    # Warm: the side approaches' room, and the modules of the solids and the seen boxes a part's inputs carry, so the
    # first part a worker is handed does not wait for them.
    import scipy.spatial  # noqa: F401, PLC0415
    import src.robot.safety.planning.perceived  # noqa: F401, PLC0415
    import src.robot.safety.planning.support_surfaces  # noqa: F401, PLC0415

    conn = _connection.Client(address, authkey=bytes.fromhex(authkey))
    sys.path[:0] = [entry for entry in conn.recv() if entry not in sys.path]
    conn.send(("ready", os.getpid()))
    plans = _WorkerPlans()
    while True:
        try:
            job = conn.recv()
        except (EOFError, OSError, KeyboardInterrupt):
            return
        if job is None:
            return
        try:
            answer: tuple[Any, ...] = ("done", *plans.run(*job))
        except BaseException as exc:  # noqa: BLE001 (said to the cell's process, which searches the part itself)
            answer = ("failed", f"{type(exc).__name__}: {exc}")
        try:
            conn.send(answer)
        except (EOFError, OSError, KeyboardInterrupt):
            return
        except Exception as exc:  # noqa: BLE001 (an answer that does not pickle is said instead)
            try:
                conn.send(("failed", f"the answer could not be handed back: {type(exc).__name__}: {exc}"))
            except Exception:  # noqa: BLE001
                return


# ---------------------------------------------------------------------------------------------------------------------
# The pool
# ---------------------------------------------------------------------------------------------------------------------


class _Broken(Exception):
    """The pool could not answer for a part; its words say why."""


@dataclasses.dataclass(eq=False)
class _Worker:
    """One worker process, its connection and its name in the log."""

    process: subprocess.Popen  # type: ignore[type-arg]
    conn: Any
    name: str


def _worker_python() -> tuple[str, dict[str, str]]:
    """The interpreter a worker runs on and its environment: the cell's, one BLAS thread from the start (set before
    numpy loads), and ``src`` found wherever the cell's process was started from.

    In a Windows virtual environment ``python.exe`` is a launcher that starts the base interpreter as a process of its
    own; the base interpreter is started directly and told the environment's (as ``multiprocessing`` does it), so the
    process the pool ends where it must is the worker itself.
    """
    env = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[name] = "1"
    env["PYTHONPATH"] = os.pathsep.join([str(_ROOT), *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])])
    python = sys.executable
    base = getattr(sys, "_base_executable", python) or python
    if os.name == "nt" and os.path.normcase(os.path.abspath(base)) != os.path.normcase(os.path.abspath(python)):
        env["__PYVENV_LAUNCHER__"] = python
        python = base
    return python, env


def _accept(listener: Any, count: int, into: "queue.Queue[Any]") -> None:
    """Accept ``count`` connections on ``listener`` into ``into``, or what refused one."""
    for _ in range(count):
        try:
            into.put(listener.accept())
        except BaseException as exc:  # noqa: BLE001 (a refused handshake is said to the starting thread)
            into.put(exc)
            return


def _chunks(units: Sequence[SfeUnit], workers: int) -> list[list[SfeUnit]]:
    """``units`` in order, cut into consecutive runs, about :data:`_CHUNKS_PER_WORKER` per worker and each cut only
    between two closing lines: a line's heights stay together, so a worker works its line out once and, where the inputs
    ask for it, makes all its builds at once."""
    size = max(1, math.ceil(len(units) / max(1, workers * _CHUNKS_PER_WORKER)))
    chunks: list[list[SfeUnit]] = []
    for unit in units:
        if chunks and (len(chunks[-1]) < size or tuple(chunks[-1][-1][:3]) == tuple(unit[:3])):
            chunks[-1].append(unit)
        else:
            chunks.append([unit])
    return chunks


class SfeWorkers:
    """SFE's runner (``support_footprint.SfeRunner``) on ``count`` worker processes, started once and kept warm.

    Hand it to the calculator per call (``GraspCalculator.compute(sfe_workers=...)``, which the pick loop does where the
    cell started a pool: ``BinPickingOrchestrator.sfe_workers``). :meth:`start` starts the workers and waits until each
    is ready; a pool that was not started starts at the first part it is asked for. Asked for a part, it hands the
    part's inputs over once, its units to whichever worker is free, and answers every unit or raises
    ``SfeRunnerFailed`` after saying why in one log line, the part then searched in the cell's process. One part at a
    time: a second caller waits for the first.
    """

    def __init__(self, count: int, *, timeout_s: float = TIMEOUT_S, start_timeout_s: float = _START_TIMEOUT_S) -> None:
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= MOST_WORKERS:
            raise ValueError(f"a pool has 1 to {MOST_WORKERS} workers, not {count!r}")
        self.count = int(count)
        self.timeout_s = float(timeout_s)
        self.start_timeout_s = float(start_timeout_s)
        self._lock = threading.RLock()
        self._workers: list[_Worker] = []
        self._broke = False
        self._given_up = ""
        self._inputs: OrderedDict[str, SharedMemory] = OrderedDict()
        self._depth: "tuple[SharedMemory, np.ndarray] | None" = None

    def __repr__(self) -> str:
        state = (f"given up: {self._given_up}" if self._given_up
                 else "running" if self._workers else "not started")
        return f"SfeWorkers({self.count}, {state})"

    @property
    def running(self) -> bool:
        """Whether the workers run now."""
        return bool(self._workers)

    @property
    def given_up(self) -> str:
        """Why the pool was given up for this process, or ``""``: every part is then searched in the cell's process."""
        return self._given_up

    def pids(self) -> tuple[int, ...]:
        """The worker processes' ids, while they run."""
        return tuple(int(worker.process.pid) for worker in self._workers)

    def start(self) -> None:
        """Start every worker and wait until each has said it is ready, its imports done: the start paid where the cell
        is built, not at the first part. Workers that run already are left as they are. Raises ``SfeRunnerFailed``
        where they could not all be started, said in the log; the pool is then given up for this process."""
        with self._lock:
            if self._given_up:
                raise SfeRunnerFailed(f"the workers were given up: {self._given_up}")
            if self._workers:
                return
            try:
                self._start()
            except _Broken as exc:
                self._given_up = str(exc)
                _LOG.warning("SFE workers: %s; every part is searched in the cell's process", exc)
                raise SfeRunnerFailed(str(exc)) from None
            _LOG.info("SFE workers: %d ready (pids %s)", len(self._workers), ", ".join(map(str, self.pids())))

    def close(self) -> None:
        """Stop the workers and let go of the shared memory. A closed pool starts again where it is asked for a part."""
        with self._lock:
            self._stop()
            for block in self._inputs.values():
                _free(block)
            self._inputs.clear()
            if self._depth is not None:
                block = self._depth[0]
                self._depth = None
                _free(block)

    def __call__(
        self, plan: SfePlan, units: Sequence[SfeUnit],
    ) -> tuple[dict[SfeUnit, list[SupportFootprintCandidate]], dict[str, int]]:
        """Every unit of ``units`` of ``plan`` run by the workers: the grasps by unit, and what the coarse ones refused
        where the plan counts. Raises ``SfeRunnerFailed``, said in one log line, where they cannot answer for all."""
        with self._lock:
            if self._given_up:
                raise self._failed(f"the workers were given up for this process ({self._given_up})")
            if not self._workers:
                try:
                    self._start()
                except _Broken as exc:
                    self._given_up = str(exc)
                    raise self._failed(f"{exc}; they are given up for this process") from None
            try:
                key, payload = self._shared(plan)
            except Exception as exc:  # noqa: BLE001 (inputs that do not pickle are searched here)
                raise self._failed(f"the part's inputs could not be handed over ({type(exc).__name__}: {exc})") \
                    from None
            try:
                answered = self._run(key, payload, list(units))
            except Exception as exc:  # noqa: BLE001 (whatever broke the pool, the part is searched here)
                self._stop()
                if not isinstance(exc, _Broken):
                    exc = _Broken(f"the pool failed: {type(exc).__name__}: {exc}")
                if self._broke:
                    self._given_up = str(exc)
                    raise self._failed(f"{exc}; the pool broke twice in a row and is given up for this process") \
                        from None
                self._broke = True
                raise self._failed(f"{exc}; the pool is started again for the next part") from None
            self._broke = False
            return answered

    # ---- the workers ------------------------------------------------------------------------------------------------

    def _start(self) -> None:
        """Start ``count`` worker processes and wait until each has connected and said it is ready.

        Each runs the worker loop alone (:data:`_WORKER_MAIN`, on this interpreter: :func:`_worker_python`), at the
        cell's own priority, with one BLAS thread, and connects back to a listener only this process and the workers
        know the key of; a Ctrl+C at the console reaches the cell's process, never a worker.
        """
        authkey = os.urandom(32)
        listener = _connection.Listener(authkey=authkey)
        accepted: "queue.Queue[Any]" = queue.Queue()
        accepting = threading.Thread(target=_accept, args=(listener, self.count, accepted), name="willy-sfe-accept",
                                     daemon=True)
        accepting.start()
        # A Windows-only flag: read by name, so the module also type-checks where it does not exist.
        flags = int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)) if os.name == "nt" else 0
        python, env = _worker_python()
        processes: list[subprocess.Popen] = []  # type: ignore[type-arg]
        conns: dict[int, Any] = {}
        try:
            for _ in range(self.count):
                processes.append(subprocess.Popen(
                    [python, "-c", _WORKER_MAIN, str(listener.address), authkey.hex()], cwd=str(_ROOT), env=env,
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, creationflags=flags,
                    start_new_session=os.name != "nt"))
            deadline = time.monotonic() + self.start_timeout_s
            while len(conns) < self.count:
                gone = [process for process in processes if process.poll() is not None]
                if gone:
                    raise _Broken(f"a worker ended as it started (exit code {gone[0].returncode}"
                                  f"{_last_words(gone[0])})")
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise _Broken(f"the workers did not start within {self.start_timeout_s:.0f} s")
                try:
                    got = accepted.get(timeout=min(remaining, 0.25))
                except queue.Empty:
                    continue
                if isinstance(got, BaseException):
                    raise _Broken(f"a worker could not connect ({type(got).__name__}: {got})")
                got.send([str(entry) for entry in sys.path])
                if not got.poll(max(0.0, deadline - time.monotonic())):
                    got.close()
                    raise _Broken(f"a worker connected and said nothing within {self.start_timeout_s:.0f} s")
                said = got.recv()
                if not (isinstance(said, tuple) and len(said) == 2 and said[0] == "ready"):
                    got.close()
                    raise _Broken(f"a worker said {said!r} instead of being ready")
                conns[int(said[1])] = got
            by_pid = {process.pid: process for process in processes}
            if set(conns) != set(by_pid):
                raise _Broken("the workers that connected are not the ones started")
        except Exception as exc:  # noqa: BLE001 (workers that cannot be started: the cell searches itself)
            self._stop([_Worker(process, conns.get(process.pid), f"willy-sfe-{index}")
                        for index, process in enumerate(processes)])
            _unblock(listener, authkey, accepting, accepted)
            raise exc if isinstance(exc, _Broken) else _Broken(
                f"the workers could not be started ({type(exc).__name__}: {exc})") from None
        listener.close()
        self._workers = [_Worker(by_pid[pid], conn, f"willy-sfe-{index}")
                         for index, (pid, conn) in enumerate(sorted(conns.items()))]

    def _stop(self, workers: "list[_Worker] | None" = None) -> None:
        """Stop ``workers`` (the pool's own where none are named): told to, then ended where they do not."""
        stopping = self._workers if workers is None else workers
        for worker in stopping:
            try:
                if worker.conn is not None:
                    worker.conn.send(None)
            except Exception:  # noqa: BLE001 (a worker that is gone is stopped)
                pass
        for worker in stopping:
            try:
                worker.process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                worker.process.kill()
                try:
                    worker.process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:  # pragma: no cover (a process that cannot be ended is left)
                    pass
            for closing in (worker.conn, worker.process.stderr):
                try:
                    if closing is not None:
                        closing.close()
                except Exception:  # noqa: BLE001
                    pass
        if workers is None:
            self._workers = []

    def _failed(self, why: str) -> SfeRunnerFailed:
        """The refusal a part falls back on, said once."""
        _LOG.warning("SFE workers: %s; this part is searched in the cell's process", why)
        return SfeRunnerFailed(why)

    # ---- one part ---------------------------------------------------------------------------------------------------

    def _shared(self, plan: SfePlan) -> tuple[str, tuple[str, int]]:
        """The part's inputs pickled into shared memory, once per part; the seen test's depth once per look. The key is
        the inputs' own digest, so the same part asked again is the same key, and a worker's plan of it serves again."""
        inputs = plan.inputs
        seen = inputs.corridor_seen
        if isinstance(seen, SeenInTheFrame):
            # The seen test with its depth in shared memory: a worker makes it a seen test again (``in_the_frame``).
            rays: Any = self._shared_rays(seen)
            inputs = dataclasses.replace(inputs, corridor_seen=rays)
        payload = pickle.dumps(inputs, protocol=pickle.HIGHEST_PROTOCOL)
        key = hashlib.blake2b(payload, digest_size=16).hexdigest()
        block = self._inputs.get(key)
        if block is None:
            block = SharedMemory(create=True, size=max(1, len(payload)))
            _buffer(block)[:len(payload)] = payload
            self._inputs[key] = block
            while len(self._inputs) > _INPUTS_KEPT:
                _free(self._inputs.popitem(last=False)[1])
        else:
            self._inputs.move_to_end(key)
        return key, (block.name, len(payload))

    def _shared_rays(self, seen: SeenInTheFrame) -> _SharedRays:
        depth = np.asarray(seen.depth_mm)
        held = self._depth
        if not (held is not None and held[1].shape == depth.shape and held[1].dtype == depth.dtype
                and np.array_equal(held[1], depth, equal_nan=True)):
            block = SharedMemory(create=True, size=max(1, depth.nbytes))
            copy = np.ndarray(depth.shape, dtype=depth.dtype, buffer=_buffer(block))
            copy[...] = depth
            stale, self._depth = self._depth, (block, copy)
            if stale is not None:
                stale_block = stale[0]
                del stale
                _free(stale_block)
            held = self._depth
        return _SharedRays(name=held[0].name, shape=tuple(int(n) for n in depth.shape), dtype=depth.dtype.str,
                           to_camera=seen.to_camera, fx=seen.fx, fy=seen.fy, cx=seen.cx, cy=seen.cy,
                           tolerance_mm=seen.tolerance_mm)

    def _run(self, key: str, payload: tuple[str, int],
             units: list[SfeUnit]) -> tuple[dict[SfeUnit, list[SupportFootprintCandidate]], dict[str, int]]:
        pending = deque(_chunks(units, len(self._workers)))
        busy: dict[int, _Worker] = {}
        by_unit: dict[SfeUnit, list[SupportFootprintCandidate]] = {}
        counted: dict[str, int] = {}
        for worker in self._workers:
            if not pending:
                break
            self._hand(worker, (key, payload, pending.popleft()))
            busy[id(worker)] = worker
        deadline = time.monotonic() + self.timeout_s
        while busy:
            remaining = deadline - time.monotonic()
            waiting = list(busy.values())
            # A worker that ended closed its connection, which then reads as ready and answers EOF.
            ready = _connection.wait([w.conn for w in waiting], timeout=max(0.0, remaining))
            if not ready:
                raise _Broken(f"no answer within {self.timeout_s:.0f} s")
            for worker in waiting:
                if worker.conn in ready:
                    try:
                        said = worker.conn.recv()
                    except (EOFError, OSError) as exc:
                        raise _Broken(f"worker {worker.name} ended (exit code {worker.process.poll()}, "
                                      f"{type(exc).__name__})") from None
                    if not (isinstance(said, tuple) and len(said) == 3 and said[0] == "done"):
                        raise _Broken(f"worker {worker.name} could not search its units: "
                                      f"{said[1] if isinstance(said, tuple) and len(said) > 1 else said!r}")
                    _, part, part_counted = said
                    by_unit.update(part)
                    for cause, count in dict(part_counted).items():
                        counted[cause] = counted.get(cause, 0) + int(count)
                    if pending:
                        self._hand(worker, (key, payload, pending.popleft()))
                    else:
                        del busy[id(worker)]
        if len(by_unit) != len(set(units)) or any(unit not in by_unit for unit in units):
            raise _Broken(f"the workers answered {len(by_unit)} of {len(units)} units")
        return by_unit, counted

    def _hand(self, worker: _Worker, job: tuple[Any, ...]) -> None:
        try:
            worker.conn.send(job)
        except Exception as exc:  # noqa: BLE001 (a worker that cannot be handed a job breaks the pool)
            raise _Broken(f"worker {worker.name} could not be handed its units ({type(exc).__name__})") from None


def _last_words(process: Any) -> str:
    """The last line a worker that ended wrote to its stderr, as a clause, or ``""``."""
    try:
        said = process.stderr.read().decode("utf-8", "replace").strip().splitlines() if process.stderr else []
    except Exception:  # noqa: BLE001
        said = []
    return f": {said[-1]}" if said else ""


def _unblock(listener: Any, authkey: bytes, accepting: threading.Thread, accepted: "queue.Queue[Any]") -> None:
    """Let the accepting thread of a start that failed end: it is handed the connections it still waits for, and every
    connection it accepted is closed with the listener."""
    while accepting.is_alive():
        try:
            _connection.Client(listener.address, authkey=authkey).close()
        except Exception:  # noqa: BLE001 (the thread ended between the two)
            break
        accepting.join(timeout=0.2)
    while True:
        try:
            got = accepted.get_nowait()
        except queue.Empty:
            break
        if not isinstance(got, BaseException):
            got.close()
    listener.close()


# ---------------------------------------------------------------------------------------------------------------------
# One pool per process
# ---------------------------------------------------------------------------------------------------------------------

_SHARED_LOCK = threading.Lock()
_SHARED: "SfeWorkers | None" = None


def shared_workers(count: int) -> "SfeWorkers | None":
    """The process's one pool of ``count`` workers, started now where it does not run yet; ``None`` for 0.

    A cell built again keeps a pool that already runs, so the workers start once per process; one of another size
    replaces it. A pool that cannot start is said in the log and handed back all the same: it searches every part in the
    cell's process (:attr:`SfeWorkers.given_up`). Closed when the process ends.
    """
    global _SHARED  # noqa: PLW0603 (one pool per process)
    if count < 1:
        return None
    # The cell's process says what its pool does in the robot log (a bare logger writes nowhere in the console). A
    # worker process never gets here, so it stays quiet, as it must.
    from src.robot.constants import create_robot_logger  # noqa: PLC0415

    create_robot_logger(__name__, "sfe_workers.log")
    with _SHARED_LOCK:
        pool = _SHARED
        if pool is None or pool.count != count or pool.given_up:
            if pool is not None:
                pool.close()
            pool = SfeWorkers(count)
            if _SHARED is None:
                atexit.register(_close_the_shared_pool)
            _SHARED = pool
    try:
        pool.start()
    except SfeRunnerFailed:
        pass   # said in the log; every part is searched in the cell's process
    return pool


def _close_the_shared_pool() -> None:
    """At the process's end: the workers stopped, the shared memory let go."""
    pool = _SHARED
    if pool is not None:
        pool.close()
