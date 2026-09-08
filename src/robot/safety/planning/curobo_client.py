"""Client for the process-isolated cuRobo planning server, the python 3.11 side of the seam.

cuRobo cannot share the Isaac process, because it wants warp 1.14 and Isaac ships
1.8.2, so this client spawns :mod:`.curobo_planner_server` with the python of the
cuRobo environment and speaks newline-delimited JSON over the subprocess stdio. The
server loads and JIT-warms once, and this client keeps it warm and issues many ``plan``
calls of tens of milliseconds each. It imports neither cuRobo nor Isaac, so it
type-checks in this environment.

Paths default to the in-repo ``ext_deps/`` install location, whose contents are
gitignored, and ``WILLY_CUROBO_PYTHON`` and ``WILLY_CUROBO_ROBOT`` override them, so
nothing machine-specific is baked into the driver. The whole path is opt-in and is
reached only where ``motion_planner="curobo"``.
"""
from __future__ import annotations

import atexit
import json
import os
import queue
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from src.robot.constants import CUROBO_CLIENT_LOG_FILE, create_robot_logger

from ._curobo_attach import ENV_ATTACH_SPHERES
from ._curobo_margin import ENV_SELF_COLLISION_MARGIN_MM
from .environment import (
    ENV_CUROBO_MESH_CACHE,
    ENV_CUROBO_STDERR,
    ENV_CUROBO_VOXEL_GRID,
    curobo_cuboid_cache,
    curobo_env_available,
    curobo_mesh_cache,
    curobo_python_path,
    curobo_robot_config,
    curobo_voxel_grid,
)

__all__ = ["CuroboPlanClient", "CuroboUnavailableError", "curobo_env_available"]

# The server file ships in this package and is run by the external cuRobo python. It is
# never imported here.
_SERVER_SCRIPT = str(Path(__file__).with_name("curobo_planner_server.py"))

# The sidecar stderr goes to a file only where WILLY_CUROBO_STDERR is set, and the
# server cannot log here at all, being a different interpreter in a different
# environment. So this client is the only place the planner lifecycle is visible from
# this side. A per-plan line is one per motion and never per candidate, and the polling
# and reader loops stay silent.
logger = create_robot_logger("CuroboPlanClient", CUROBO_CLIENT_LOG_FILE)

# Generous, because the server JIT-warms the cuRobo kernels on first boot: about 25 s
# cold and about 8 s with a warm kernel cache.
_READY_TIMEOUT_S = 120.0
_PLAN_TIMEOUT_S = 30.0


class CuroboUnavailableError(RuntimeError):
    """The cuRobo planning server could not be started or became ready (env missing, JIT/load failure)."""


def _raise_if_call_failed(msg: dict) -> None:
    """Turn a failed call into an exception, and leave a genuine absence of a solution as ``None``.

    The two are not the same thing and must never look the same, which is why this
    function exists. A planner that searched and found nothing is a verdict a caller can
    act on: fail safe, do not move. A call that raised is a bug, and returning ``None``
    for it dresses a bug up as a verdict about the robot.

    The failure that shape produces is concrete. ``plan_js`` is not a method on this
    cuRobo build, where it is ``plan_cspace``, so every joint-space request raises
    ``AttributeError``. A broad ``except`` in the sidecar reports that as a failure, the
    client returns ``None``, and the arm reads it as cuRobo refusing the configuration,
    which sends a reader to the robot descriptor and its collision spheres. The clue is
    that the planner cannot plan from a configuration to itself. So the sidecar labels
    which of the two happened, and this raises.
    """
    if msg.get("planner_error"):
        raise CuroboUnavailableError(
            f"the cuRobo planning CALL failed (not a planning verdict): {msg.get('reason')}"
        )


def _log_at_exit(emit: "Callable[..., None]", message: str, *args: object) -> None:
    """Log without raising while the interpreter is already tearing its logging streams down.

    `close()` is registered with atexit, so it runs after the host has closed the
    streams the handlers write to. A ValueError from a shutdown message is noise that
    reads like a real failure, arriving as an I/O-operation-on-closed-file traceback
    under a call stack pointing at this line.
    """
    try:
        emit(message, *args)
    except (ValueError, OSError):  # streams already closed at interpreter exit
        pass


class CuroboPlanClient:
    """Spawns + drives the warm cuRobo planning server. ``plan`` returns a joint trajectory or ``None``."""

    def __init__(
        self,
        *,
        python_path: str | None = None,
        server_script: str | None = None,
        robot_config: str | None = None,
        scene_config: str | None = None,
        stderr_log: str | None = None,
        self_collision_margin_mm: float = 0.0,
        attach_spheres: int = 0,
        mesh_cache: int | None = None,
        voxel_grid: str | None = None,
    ) -> None:
        # Every path and knob resolves through safety.planning.environment, the single
        # anchor, so a caller overrides what it needs and the rest tracks the variables
        # documented there.
        self._python = python_path or curobo_python_path()
        self._script = server_script or _SERVER_SCRIPT
        self._robot = robot_config or curobo_robot_config()
        self._scene = scene_config or curobo_cuboid_cache()
        # The clearance the safety guard will demand of the final configuration of the
        # plan. It is handed to the sidecar so cuRobo plans with that margin instead of
        # returning paths the guard then refuses: measured, 9.44 to 9.47 mm plans against
        # a 10.000 mm guard margin. 0.0 leaves the config untouched.
        self._self_collision_margin_mm = float(self_collision_margin_mm)
        # Collision-sphere slots reserved for a carried payload. At 0 the sidecar robot
        # config is untouched and `attach_payload` refuses, which is the unchanged path.
        self._attach_spheres = max(0, int(attach_spheres))
        # Slots for the two other kinds of obstacle. Both are reserved when the planner is
        # BUILT, so they are decided here and never again: a mesh or a voxel grid sent to a
        # planner that reserved none has nowhere to go. 0 and empty leave the sidecar as it was.
        self._mesh_cache = max(0, int(mesh_cache if mesh_cache is not None else curobo_mesh_cache()))
        self._voxel_grid = str(voxel_grid if voxel_grid is not None else curobo_voxel_grid()).strip()
        # The server stderr, carrying cuRobo warmup and plan diagnostics, goes to a log
        # file where one is requested through the parameter or WILLY_CUROBO_STDERR, and
        # is discarded otherwise. It is what makes the isolated server debuggable on-box.
        self._stderr_log = stderr_log or os.environ.get(ENV_CUROBO_STDERR)
        self._proc: subprocess.Popen[str] | None = None
        self._q: "queue.Queue[dict | None]" = queue.Queue()
        #: The monotonic request counter. Every request carries it and every reply echoes
        #: it, which is what lets a late answer be dropped instead of executed.
        self._next_id = 0
        #: False once the sidecar stdout hit EOF or a write to it failed. `self._proc`
        #: alone cannot say this, because only `close()` clears it, so a client whose
        #: child had died would still believe it was running.
        self._alive = True
        self._warned_unstamped = False
        self._reader: threading.Thread | None = None
        self.joint_names: list[str] = []
        self.dt: float = 0.0

    # --- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        """Spawn the server and block until it reports ``ready`` (raises CuroboUnavailableError otherwise)."""
        if self._proc is not None:
            return
        if not Path(self._python).exists():
            raise CuroboUnavailableError(f"cuRobo env python not found: {self._python}")
        logger.info(
            "spawning the cuRobo sidecar: python=%s robot=%s cuboid_cache=%s self_collision_margin=%.3f mm",
            self._python, self._robot, self._scene, self._self_collision_margin_mm,
        )
        started = time.monotonic()
        stderr = open(self._stderr_log, "w") if self._stderr_log else subprocess.DEVNULL  # noqa: SIM115
        env = dict(os.environ)
        if self._self_collision_margin_mm > 0.0:
            env[ENV_SELF_COLLISION_MARGIN_MM] = repr(self._self_collision_margin_mm)
        if self._attach_spheres > 0:
            env[ENV_ATTACH_SPHERES] = str(self._attach_spheres)
        if self._mesh_cache > 0:
            env[ENV_CUROBO_MESH_CACHE] = str(self._mesh_cache)
        if self._voxel_grid:
            env[ENV_CUROBO_VOXEL_GRID] = self._voxel_grid
        self._alive = True
        self._proc = subprocess.Popen(
            [self._python, "-u", self._script, self._robot, self._scene],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr, text=True, bufsize=1,
            env=env,
        )
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()
        # Close on interpreter exit, so a runner that never calls disconnect() still
        # kills the server first. The Isaac picks rely on the Isaac hard teardown, and
        # without this the daemon _pump thread is left blocked on the server pipe when
        # Isaac tears the process down, which crashes the native teardown and produces a
        # 12 MB Kit dump. atexit runs before the deep teardown, and close() is
        # idempotent, so an earlier explicit disconnect is harmless.
        atexit.register(self.close)
        msg = self._recv(_READY_TIMEOUT_S)          # the handshake carries no id
        if not msg or msg.get("status") != "ready":
            reason = (msg or {}).get("reason", "no ready signal (timeout or server died)")
            self.close()
            raise CuroboUnavailableError(f"cuRobo server did not become ready: {reason}")
        self.joint_names = list(msg["joint_names"])
        self.dt = float(msg.get("dt", 0.0))
        # The warm-up cost is worth recording: about 25 s cold against about 8 s with a
        # warm kernel cache is the difference between a slow planner and a kernel cache
        # that was thrown away.
        logger.info(
            "cuRobo sidecar ready after %.1f s: %d joint(s), dt=%.4f s",
            time.monotonic() - started, len(self.joint_names), self.dt,
        )

    def _pump(self) -> None:
        """The reader thread. It pushes each JSON line from the server stdout onto the queue.

        It puts ``None`` on EOF. It captures the stream locally, because ``close()``
        nulls ``self._proc``, and swallows the read error that fires when ``close()``
        shuts the pipe under this thread during teardown, so a graceful close never
        raises here.
        """
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        try:
            for line in proc.stdout:
                line = line.strip()
                if line.startswith("{"):
                    try:
                        self._q.put(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        except (ValueError, OSError):  # stream closed under us during shutdown -> stop quietly
            pass
        # This is where the client learns its sidecar died. Only `close()` clears
        # `self._proc`, so without this a client whose child had exited keeps believing
        # it is running and the next `_send` writes into a dead pipe.
        self._alive = False
        self._q.put(None)  # EOF / server exit

    def _recv(self, timeout_s: float, *, want: "int | None" = None) -> "dict | None":
        """The reply to request ``want``, or ``None`` on a timeout or a server exit.

        A late reply is dropped here, and dropping it is the point. This is request and
        response over one queue: a call that times out stops waiting and leaves its
        request outstanding, so the late answer from the sidecar lands in the queue and
        would otherwise be handed to whatever asks next. Reproduced end to end with a
        stub sidecar, one plan timeout leaves the next motion holding the trajectory of
        the previous goal, and a real arm would drive it.

        The budget is not reset per message. Draining a backlog must not extend the
        caller timeout, or a sidecar emitting stale lines holds a motion open
        indefinitely.
        """
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return None
            try:
                msg = self._q.get(timeout=remaining)
            except queue.Empty:
                return None
            if msg is None:                       # EOF: the sidecar exited
                self._alive = False
                return None
            got = msg.get("id")
            if want is None or got is None or got == want:
                # `want is None` is the ready handshake, emitted before the server loop
                # and carrying no id. `got is None` is a reply from a server older than
                # this protocol, accepted rather than refused, because rejecting every
                # answer would be a worse failure than the one this guards, and said
                # once so it cannot pass unnoticed.
                if want is not None and got is None and not self._warned_unstamped:
                    self._warned_unstamped = True
                    logger.warning(
                        "the cuRobo sidecar answers without a request id, so a late reply cannot be "
                        "told apart from the right one; it is older than this client")
                return msg
            logger.warning(
                "dropped a late cuRobo reply for request %s while waiting for %s; it would have "
                "been used as the answer to this call", got, want)

    def _send(self, obj: dict) -> int:
        """Write one request and return the id its reply must carry.

        A dead sidecar fails typed here. Without `self._proc` being cleared and `poll()`
        consulted, the call after the sidecar dies writes to the stdin of a dead child
        and raises `OSError` straight out of the fail-closed motion path. Measured at
        0.0, 0.3 and 2.0 s after the death of the sidecar, only the 0.0 s case produces
        the typed error first, so in any realistic case the typed
        `CONTROLLER_REJECTED` never fires at all.
        """
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise CuroboUnavailableError("cuRobo server is not running")
        # The exit code is the half an operator can act on. `poll()` is consulted even
        # where `_alive` is already false, because a server that is not running and a
        # server that exited with 1 send someone to two different places, and only the
        # second names a child process.
        if not self._alive or proc.poll() is not None:
            self._alive = False
            code = proc.poll()
            raise CuroboUnavailableError(
                f"the cuRobo sidecar exited with code {code}; no plan can be trusted until it is "
                f"restarted" if code is not None else
                "the cuRobo sidecar's stream ended; no plan can be trusted until it is restarted")
        self._next_id += 1
        obj = {**obj, "id": self._next_id}
        try:
            proc.stdin.write(json.dumps(obj) + "\n")
            proc.stdin.flush()
        except OSError as exc:
            # The child can die between poll() and write(). A typed refusal keeps this
            # inside the vocabulary of the motion path instead of surfacing an OSError
            # from a pipe.
            self._alive = False
            raise CuroboUnavailableError(
                f"writing to the cuRobo sidecar failed ({type(exc).__name__}: {exc}); it is gone"
            ) from exc
        return self._next_id


    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                proc.stdin.flush()
            proc.wait(timeout=5)
            _log_at_exit(logger.info, "cuRobo sidecar shut down cleanly")
        except Exception as exc:  # noqa: BLE001 (shutdown is best-effort)
            _log_at_exit(logger.warning,
                         "cuRobo sidecar did not shut down cleanly (%s); killing it", exc)
            proc.kill()

    # --- planning ----------------------------------------------------------
    def plan(
        self,
        start_joints: list[float],
        goal_pos_m: list[float],
        goal_quat_wxyz: list[float],
    ) -> list[list[float]] | None:
        """Plan tool0 -> ``goal`` (metres, WXYZ, base frame) from ``start_joints`` (rad).

        It returns the interpolated joint trajectory ``[[6 rad], ...]`` in
        :attr:`joint_names` order, or ``None`` where cuRobo found no collision-free
        solution, and the caller then fails safe with no blind motion.
        """
        if self._proc is None:
            self.start()
        started = time.monotonic()
        want = self._send({"start_joints": list(start_joints), "goal_pos_m": list(goal_pos_m),
                           "goal_quat_wxyz": list(goal_quat_wxyz)})
        msg = self._recv(_PLAN_TIMEOUT_S, want=want)
        if msg is None:
            raise CuroboUnavailableError("cuRobo server stopped responding (timeout / died)")
        if msg.get("success"):
            traj: list[list[float]] = msg["trajectory"]
            logger.debug(
                "planned to (%.3f, %.3f, %.3f) m in %.0f ms: %d waypoint(s)",
                goal_pos_m[0], goal_pos_m[1], goal_pos_m[2],
                (time.monotonic() - started) * 1000.0, len(traj),
            )
            return traj
        _raise_if_call_failed(msg)
        # A verdict rather than a failure, but the caller turns it into no motion, so the
        # reason a pick stopped is recoverable from here alone.
        logger.warning(
            "cuRobo found NO collision-free plan to (%.3f, %.3f, %.3f) m after %.0f ms: %s",
            goal_pos_m[0], goal_pos_m[1], goal_pos_m[2],
            (time.monotonic() - started) * 1000.0, msg.get("reason", "no reason given"),
        )
        return None

    def plan_joint(
        self, start_joints: list[float], goal_joints: list[float]
    ) -> list[list[float]] | None:
        """Plan ``start_joints`` -> ``goal_joints`` (rad) collision-free, or ``None`` if it cannot.

        This is the joint-space twin of :meth:`plan`, for goals that are joint
        configurations, as park and home are stored. Going through FK and a Cartesian
        plan instead would let cuRobo satisfy the tool pose with any IK branch, and a
        flipped elbow branch is how 43 mm of hidden self-penetration gets into a pose
        that looks fine.

        ``None`` means there is no collision-free plan, and the caller fails safe rather
        than moving blindly, which is the reason to ask for a plan instead of
        interpolating.
        """
        if self._proc is None:
            self.start()
        started = time.monotonic()
        want = self._send({"cmd": "plan_js", "start_joints": list(start_joints),
                           "goal_joints": list(goal_joints)})
        msg = self._recv(_PLAN_TIMEOUT_S, want=want)
        if msg is None:
            raise CuroboUnavailableError("cuRobo server stopped responding (timeout / died)")
        if msg.get("success"):
            traj: list[list[float]] = msg["trajectory"]
            logger.debug(
                "planned a joint move in %.0f ms: %d waypoint(s)",
                (time.monotonic() - started) * 1000.0, len(traj),
            )
            return traj
        _raise_if_call_failed(msg)
        logger.warning(
            "cuRobo found NO collision-free joint plan after %.0f ms: %s",
            (time.monotonic() - started) * 1000.0, msg.get("reason", "no reason given"),
        )
        return None

    def set_world(self, cuboids: list[dict], meshes: "list[dict] | None" = None) -> int:
        """Replace cuRobo's collision world with these obstacles (the scene->planner world-model).

        Each cuboid is ``{"name": str, "dims_m": [x,y,z], "pose": [px,py,pz,qw,qx,qy,qz]}``
        in the base frame, in metres, with a WXYZ quaternion. Each mesh is
        ``{"name": str, "file_path": str, "pose": [...]}`` with an optional
        ``"scale": [sx,sy,sz]``, and the sidecar reads the file itself: a tote is tens of
        thousands of triangles and this is a line-based JSON pipe.

        A mesh is how a container reaches the planner as the shape it is rather than as a
        solid block. Meshes need slots reserved at spawn through ``mesh_cache``; without
        them the sidecar has nowhere to put one and says so.

        It returns the count registered, or 0 on failure. It replaces rather than extends,
        so everything the planner must keep has to be in this one call.
        """
        if self._proc is None:
            self.start()
        request = {"cmd": "set_world", "cuboids": list(cuboids)}
        if meshes:
            request["meshes"] = list(meshes)
        want = self._send(request)
        msg = self._recv(_PLAN_TIMEOUT_S, want=want)
        if msg and msg.get("world_set") is not None:
            count = int(msg["world_set"])
            sent = len(cuboids) + len(meshes or ())
            # An obstacle the planner never received is one it will route straight
            # through, so the count registered rather than the count sent is what
            # matters.
            logger.info("collision world set: %d of %d obstacle(s) registered", count, sent)
            return count
        logger.error(
            "the cuRobo sidecar did not confirm the collision world (%d cuboid(s) sent); planning "
            "continues against the PREVIOUS world",
            len(cuboids),
        )
        return 0

    def set_voxels(
        self,
        path: "str | None",
        *,
        dims_m: "Sequence[float]" = (),
        voxel_size_m: float = 0.0,
        pose: "Sequence[float]" = (),
    ) -> "int | None":
        """Register the live scene as a distance field over a grid, or clear it with ``path=None``.

        This is the channel that carries a whole cell. The cuboid channel carries as many
        obstacles as there are slots and the caller then has to choose which ones matter; a
        grid carries everything the cameras saw, at the resolution it was cut to, and
        nothing is left out.

        ``path`` names a NumPy file holding the field as one flat array in the planner's
        own voxel order. It is a file rather than numbers in the request because a 30 mm
        grid over a 2 m cell is 179,560 values, which is not something to send down this
        pipe before every motion.

        ⛔ The field is positive INSIDE an obstacle and negative in free space. That is the
        opposite of the distance-to-obstacle a person would write, and the wrong sign fails
        silently: measured on this repository, a wall written the intuitive way registered
        without an error, reported success, and the planner drove straight through it.
        :mod:`src.robot.safety.planning.perceived` builds the field, and builds it in this
        sign.

        It returns the number of values registered, or ``None`` where the sidecar refused,
        which is the state a caller must treat as no world at all.
        """
        if self._proc is None:
            self.start()
        request: dict = {"cmd": "set_voxels", "path": path}
        if path:
            request.update(
                {"dims_m": list(dims_m), "voxel_size_m": float(voxel_size_m), "pose": list(pose)}
            )
        want = self._send(request)
        msg = self._recv(_PLAN_TIMEOUT_S, want=want)
        if msg and msg.get("voxels_set") is not None:
            return int(msg["voxels_set"])
        reason = (msg or {}).get("reason", "no reply")
        logger.error(
            "the cuRobo sidecar did not register the live scene (%s); planning continues against "
            "the PREVIOUS world", reason,
        )
        return None

    def reserve_attach_spheres(self, slots: int) -> None:
        """Reserve payload collision spheres. Only takes effect before :meth:`start`."""
        if self._proc is not None:
            logger.warning(
                "reserve_attach_spheres(%d) after the sidecar started: the robot config is already "
                "built, so this has no effect until the next start", slots,
            )
            return
        self._attach_spheres = max(0, int(slots))

    def attach_payload(
        self,
        joints: list[float],
        dims_m: "Sequence[float]",
        pose: "Sequence[float]",
        *,
        name: str = "payload",
    ) -> bool:
        """Hang a box on the tool so later plans route the carried part around the world too.

        ``dims_m`` are full side lengths, and ``pose`` is ``[x, y, z, qw, qx, qy, qz]``
        in the tool frame, so ``[0, 0, h/2, 1, 0, 0, 0]`` is a part sitting half its
        height beyond the flange. ``joints`` is the configuration the part was grasped
        in, which is where the attachment is fitted.

        It returns ``False`` where the sidecar could not attach, including the case of a
        sidecar started without a sphere budget, which therefore has no link to hang
        anything from. That is deliberately not an exception: a cell that cannot model
        its payload carries on planning without it and says so rather than stopping
        mid-pick.
        """
        if self._proc is None:
            self.start()
        want = self._send({"cmd": "attach", "joints": list(joints), "dims_m": list(dims_m),
                           "pose": list(pose), "name": name})
        msg = self._recv(_PLAN_TIMEOUT_S, want=want)
        if msg and msg.get("attached"):
            logger.info("payload attached to the planner model: dims_m=%s", list(dims_m))
            return True
        logger.error(
            "the cuRobo sidecar did not attach the payload (%s); it is planning as if the gripper "
            "were EMPTY", (msg or {}).get("reason", "no response"),
        )
        return False

    def detach_payload(self) -> bool:
        """Take the carried box off the planner's model. ``False`` if the sidecar did not confirm."""
        if self._proc is None:
            return True  # nothing was ever attached
        want = self._send({"cmd": "detach"})
        msg = self._recv(_PLAN_TIMEOUT_S, want=want)
        if msg and msg.get("detached"):
            return True
        logger.error(
            "the cuRobo sidecar did not detach the payload (%s); it will keep planning around a part "
            "the gripper no longer holds", (msg or {}).get("reason", "no response"),
        )
        return False

    def fk(self, joints: list[float]) -> tuple[list[float], list[float]] | None:
        """Tool0 FK of ``joints`` in radians, as (pos_m, quat_wxyz), or ``None`` on failure."""
        if self._proc is None:
            self.start()
        want = self._send({"cmd": "fk", "joints": list(joints)})
        msg = self._recv(_PLAN_TIMEOUT_S, want=want)
        if msg and msg.get("fk_pos_m") is not None:
            return msg["fk_pos_m"], msg["fk_quat_wxyz"]
        return None

    def __enter__(self) -> "CuroboPlanClient":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
