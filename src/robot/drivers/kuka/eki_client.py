"""The KUKA EthernetKRL client: a TCP socket and the XML protocol driver.

Architecture
------------
``EkiClient`` owns three things:

* one TCP socket, where ``server`` mode listens for an incoming KRL connection and
  ``client`` mode dials out;
* a reader thread that parses incoming :mod:`.protocol` frames continuously and
  dispatches each either to a thread-safe state cache, for telemetry, or to the
  in-flight request registry, for an FK or IK response or a command ack;
* a heartbeat thread that sends ``<Echo>`` periodically and watches for ``<EchoAck>``,
  marking the link stale where the round-trip exceeds the configured timeout.

The public surface is synchronous and thread-safe:

* :meth:`connect` and :meth:`disconnect` are the lifecycle.
* :attr:`is_connected` reports a live link.
* :meth:`get_state` returns the last cached telemetry.
* :meth:`send_move_cartesian`, :meth:`send_movej`, :meth:`send_stop` and
  :meth:`send_home` are the motion commands. Each blocks for the controller ``<Ack>``
  or raises :class:`RobotConnectionError` on timeout.
* :meth:`request_fk` and :meth:`request_ik` are the synchronous round-trip to the
  controller KRL FK and IK utilities.
* :meth:`wait_until_steady` blocks until the cached telemetry reports ``Steady=1`` or
  the timeout elapses.
"""

from __future__ import annotations

import socket
import threading
import time
import uuid
from dataclasses import dataclass

from src.robot.constants import KUKA_EKI_LOG_FILE, create_robot_logger
from src.robot.core import RobotConnectionError

from .pose_convert import KukaCartesian
from .protocol import (
    EkiAck,
    EkiEchoAck,
    EkiFkResult,
    EkiIkResult,
    EkiState,
    EkiUnknown,
    decode_frame,
    encode_echo,
    encode_fk_request,
    encode_home,
    encode_ik_request,
    encode_move,
    encode_movej,
    encode_stop,
    iter_frames,
)

__all__ = ["EkiCachedState", "EkiClient"]


logger = create_robot_logger("KukaEkiClient", KUKA_EKI_LOG_FILE)


@dataclass
class EkiCachedState:
    """Snapshot of the most-recent ``<State>`` frame."""

    pose: KukaCartesian | None = None
    joints_deg: tuple | None = None
    steady: bool = False
    last_update_ts: float = 0.0


class _PendingRequest:
    """Cell waiting on a single keyed reply from the controller."""

    __slots__ = ("event", "result")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.result: object = None


class _PendingAck:
    """Cell waiting on the next generic ``<Ack>`` for a given command."""

    __slots__ = ("ack", "event")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.ack: EkiAck | None = None


class EkiClient:
    """The TCP and XML driver for KUKA EthernetKRL.

    Parameters
    ----------
    role
        ``"server"`` listens here and lets KRL dial in, which is the recommended
        direction. ``"client"`` dials KRL instead.
    host, port
        The TCP endpoint. In server mode ``host`` is the local bind address.
    timeout_s
        The per-operation timeout, covering connect, request and reply, and ack.
    heartbeat_s
        The period of the background ``<Echo>`` probes. A very high value, or ``0.0``,
        disables them.
    buffer_size
        The reader recv buffer.
    """

    def __init__(
        self,
        *,
        role: str = "server",
        host: str = "0.0.0.0",
        port: int = 7000,
        timeout_s: float = 5.0,
        heartbeat_s: float = 0.5,
        buffer_size: int = 65536,
    ) -> None:
        if role not in ("server", "client"):
            raise ValueError(f"role must be 'server' or 'client', got {role!r}")
        self._role = role
        self._host = host
        self._port = int(port)
        self._timeout_s = float(timeout_s)
        self._heartbeat_s = float(heartbeat_s)
        self._buffer_size = int(buffer_size)

        self._sock: socket.socket | None = None
        self._listener: socket.socket | None = None
        self._reader: threading.Thread | None = None
        self._heartbeat: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._send_lock = threading.Lock()

        self._state = EkiCachedState()
        self._state_lock = threading.Lock()
        self._state_event = threading.Event()

        self._pending_fk: dict[str, _PendingRequest] = {}
        self._pending_ik: dict[str, _PendingRequest] = {}
        self._pending_acks: list[_PendingAck] = []
        self._pending_echo: dict[str, threading.Event] = {}
        self._pending_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._sock is not None and not self._stop_event.is_set()

    def connect(self) -> None:
        """Open the TCP link and start the reader / heartbeat threads."""
        if self.is_connected:
            return
        self._stop_event.clear()
        if self._role == "server":
            self._sock = self._accept_inbound()
        else:
            self._sock = self._dial_outbound()
        self._sock.settimeout(0.5)  # so the reader recv wakes regularly

        self._reader = threading.Thread(
            target=self._reader_loop, name="EkiClient.reader", daemon=True,
        )
        self._reader.start()

        if self._heartbeat_s > 0.0:
            self._heartbeat = threading.Thread(
                target=self._heartbeat_loop,
                name="EkiClient.heartbeat",
                daemon=True,
            )
            self._heartbeat.start()
        logger.info("EKI link up (role=%s, %s:%d).", self._role, self._host, self._port)

    def disconnect(self) -> None:
        """Stop threads, close the TCP link. Idempotent."""
        if self._stop_event.is_set() and self._sock is None:
            return
        self._stop_event.set()
        sock, self._sock = self._sock, None
        listener, self._listener = self._listener, None
        for s in (sock, listener):
            if s is None:
                continue
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                s.close()
            except OSError:
                pass
        for t in (self._reader, self._heartbeat):
            if t is not None and t.is_alive():
                t.join(timeout=1.0)
        self._reader = None
        self._heartbeat = None
        self._wake_all_pending()

    def _wake_all_pending(self) -> None:
        """Wake and drop every in-flight waiter, so a caller fails fast.

        Both :meth:`disconnect` and the reader loop call it when the link dies. Waking
        waiters only from ``disconnect`` leaves a reader thread that died on its own,
        from a dropped KRC link or a peer close, holding every ``_send_with_ack``, FK
        and IK caller until each individual timeout elapses.
        """
        with self._pending_lock:
            for pr in list(self._pending_fk.values()) + list(self._pending_ik.values()):
                pr.event.set()
            for pa in self._pending_acks:
                pa.event.set()
            for ev in self._pending_echo.values():
                ev.set()
            self._pending_fk.clear()
            self._pending_ik.clear()
            self._pending_acks.clear()
            self._pending_echo.clear()

    def __enter__(self) -> EkiClient:
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.disconnect()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def get_state(self) -> EkiCachedState:
        """Return a copy of the latest cached telemetry."""
        with self._state_lock:
            return EkiCachedState(
                pose=self._state.pose,
                joints_deg=self._state.joints_deg,
                steady=self._state.steady,
                last_update_ts=self._state.last_update_ts,
            )

    def wait_until_steady(self, timeout_s: float) -> bool:
        """Block until the cached state reports ``Steady=1`` or timeout."""
        deadline = time.monotonic() + max(0.0, timeout_s)
        while True:
            with self._state_lock:
                if self._state.steady:
                    return True
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return False
            self._state_event.wait(timeout=min(remaining, 0.1))
            self._state_event.clear()

    def wait_for_first_state(self, timeout_s: float | None = None) -> bool:
        """Block until at least one ``<State>`` frame has arrived.

        A caller uses it directly after :meth:`connect`, so the cache is populated
        before it is read.
        """
        deadline = (
            time.monotonic() + (timeout_s if timeout_s is not None else self._timeout_s)
        )
        while True:
            with self._state_lock:
                if self._state.pose is not None:
                    return True
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return False
            self._state_event.wait(timeout=min(remaining, 0.1))
            self._state_event.clear()

    # ------------------------------------------------------------------
    # Outbound commands
    # ------------------------------------------------------------------

    def send_move_cartesian(
        self,
        target: KukaCartesian,
        *,
        mode: str = "PTP",
        vel: float | None = None,
        acc: float | None = None,
    ) -> EkiAck:
        """Send a Cartesian move command and wait for the controller ack."""
        return self._send_with_ack(
            encode_move(target, mode=mode, vel=vel, acc=acc),
            expected_command="Move",
        )

    def send_movej(
        self,
        joints_deg: list,
        *,
        vel: float | None = None,
        acc: float | None = None,
    ) -> EkiAck:
        """Send a joint-space move command and wait for ack."""
        return self._send_with_ack(
            encode_movej(list(joints_deg), vel=vel, acc=acc),
            expected_command="MoveJ",
        )

    def send_stop(self) -> None:
        """Fire-and-forget stop. Does not wait for ack."""
        self._send_bytes(encode_stop())

    def send_home(self) -> EkiAck:
        """Send a home command and wait for ack."""
        return self._send_with_ack(encode_home(), expected_command="Home")

    # ------------------------------------------------------------------
    # Round-trip requests
    # ------------------------------------------------------------------

    def request_fk(self, joints_deg: list) -> KukaCartesian:
        """Synchronously ask KRL to compute FK on ``joints_deg``."""
        request_id = uuid.uuid4().hex
        slot = _PendingRequest()
        with self._pending_lock:
            self._pending_fk[request_id] = slot
        try:
            self._send_bytes(encode_fk_request(request_id, list(joints_deg)))
            if not slot.event.wait(timeout=self._timeout_s):
                raise RobotConnectionError(
                    f"KUKA FkRequest {request_id!r} timed out after {self._timeout_s:.2f}s"
                )
            result = slot.result
            if not isinstance(result, EkiFkResult) or result.status != "ok" or result.pose is None:
                detail = getattr(result, "detail", None) or "no detail"
                raise RobotConnectionError(
                    f"KUKA FkRequest {request_id!r} failed: {detail}"
                )
            return result.pose
        finally:
            with self._pending_lock:
                self._pending_fk.pop(request_id, None)

    def request_ik(
        self,
        target: KukaCartesian,
        seed_deg: list,
    ) -> list:
        """Ask KRL synchronously to compute IK for ``target`` near ``seed_deg``.

        It returns the joint solution in degrees, the KUKA wire convention.
        """
        request_id = uuid.uuid4().hex
        slot = _PendingRequest()
        with self._pending_lock:
            self._pending_ik[request_id] = slot
        try:
            self._send_bytes(encode_ik_request(request_id, target, list(seed_deg)))
            if not slot.event.wait(timeout=self._timeout_s):
                raise RobotConnectionError(
                    f"KUKA IkRequest {request_id!r} timed out after {self._timeout_s:.2f}s"
                )
            result = slot.result
            if not isinstance(result, EkiIkResult) or result.status != "ok" or result.joints_deg is None:
                detail = getattr(result, "detail", None) or "no detail"
                raise RobotConnectionError(
                    f"KUKA IkRequest {request_id!r} failed: {detail}"
                )
            return list(result.joints_deg)
        finally:
            with self._pending_lock:
                self._pending_ik.pop(request_id, None)

    # ------------------------------------------------------------------
    # Internal: networking
    # ------------------------------------------------------------------

    def _accept_inbound(self) -> socket.socket:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.settimeout(self._timeout_s)
        try:
            listener.bind((self._host, self._port))
            listener.listen(1)
            self._listener = listener
            sock, peer = listener.accept()
        except TimeoutError as exc:
            try:
                listener.close()
            except OSError:
                pass
            self._listener = None
            raise RobotConnectionError(
                f"KUKA EKI: no inbound connection on {self._host}:{self._port} "
                f"within {self._timeout_s:.1f}s"
            ) from exc
        except OSError as exc:
            try:
                listener.close()
            except OSError:
                pass
            self._listener = None
            raise RobotConnectionError(
                f"KUKA EKI: failed to bind on {self._host}:{self._port}: {exc}"
            ) from exc
        logger.info("EKI accepted KRL connection from %s.", peer)
        return sock

    def _dial_outbound(self) -> socket.socket:
        try:
            sock = socket.create_connection(
                (self._host, self._port), timeout=self._timeout_s,
            )
        except OSError as exc:
            raise RobotConnectionError(
                f"KUKA EKI: failed to dial {self._host}:{self._port}: {exc}"
            ) from exc
        return sock

    def _send_bytes(self, payload: bytes) -> None:
        if self._sock is None or self._stop_event.is_set():
            raise RobotConnectionError("KUKA EKI link is not connected.")
        with self._send_lock:
            try:
                self._sock.sendall(payload)
            except OSError as exc:
                raise RobotConnectionError(f"KUKA EKI send failed: {exc}") from exc

    def _send_with_ack(self, payload: bytes, *, expected_command: str) -> EkiAck:
        slot = _PendingAck()
        with self._pending_lock:
            self._pending_acks.append(slot)
        try:
            self._send_bytes(payload)
            if not slot.event.wait(timeout=self._timeout_s):
                raise RobotConnectionError(
                    f"KUKA <Ack> for {expected_command!r} timed out after {self._timeout_s:.2f}s"
                )
            ack = slot.ack
            if ack is None:
                raise RobotConnectionError(
                    f"KUKA <Ack> for {expected_command!r} not received."
                )
            if ack.status != "ok":
                raise RobotConnectionError(
                    f"KUKA controller rejected {expected_command!r}: "
                    f"{ack.detail or 'no detail'}"
                )
            return ack
        finally:
            with self._pending_lock:
                if slot in self._pending_acks:
                    self._pending_acks.remove(slot)

    # ------------------------------------------------------------------
    # Internal: threads
    # ------------------------------------------------------------------

    def _reader_loop(self) -> None:
        buffer = bytearray()
        while not self._stop_event.is_set():
            sock = self._sock
            if sock is None:
                break
            try:
                chunk = sock.recv(self._buffer_size)
            except TimeoutError:
                continue
            except OSError as exc:
                logger.warning("EKI reader: recv failed (%s); closing.", exc)
                break
            if not chunk:
                logger.info("EKI reader: peer closed.")
                break
            buffer.extend(chunk)
            if len(buffer) > self._buffer_size * 8:
                logger.error(
                    "EKI reader: buffer overflow (%d bytes without newline); dropping link.",
                    len(buffer),
                )
                break
            for frame in iter_frames(buffer):
                try:
                    decoded = decode_frame(frame)
                except ValueError as exc:
                    logger.warning("EKI reader: malformed frame: %s", exc)
                    continue
                self._dispatch(decoded)
        # Mark the link dead and wake any in-flight waiter, so it fails fast rather
        # than blocking for its full timeout with the reader gone.
        self._stop_event.set()
        self._wake_all_pending()

    def _dispatch(self, message: object) -> None:
        if isinstance(message, EkiState):
            with self._state_lock:
                self._state.pose = message.pose
                self._state.joints_deg = message.joints_deg
                self._state.steady = message.steady
                self._state.last_update_ts = time.monotonic()
            self._state_event.set()
            return
        if isinstance(message, EkiAck):
            with self._pending_lock:
                if self._pending_acks:
                    slot = self._pending_acks[0]
                    slot.ack = message
                    slot.event.set()
            return
        if isinstance(message, EkiFkResult):
            with self._pending_lock:
                fk_slot = self._pending_fk.get(message.request_id)
            if fk_slot is not None:
                fk_slot.result = message
                fk_slot.event.set()
            return
        if isinstance(message, EkiIkResult):
            with self._pending_lock:
                ik_slot = self._pending_ik.get(message.request_id)
            if ik_slot is not None:
                ik_slot.result = message
                ik_slot.event.set()
            return
        if isinstance(message, EkiEchoAck):
            with self._pending_lock:
                ev = self._pending_echo.pop(message.token, None)
            if ev is not None:
                ev.set()
            return
        if isinstance(message, EkiUnknown):
            logger.debug("EKI reader: ignoring unknown frame <%s>", message.tag)
            return
        logger.debug("EKI reader: unhandled message type %s", type(message).__name__)

    def _heartbeat_loop(self) -> None:
        miss_window = max(2.0 * self._heartbeat_s, self._timeout_s)
        while not self._stop_event.is_set():
            token = uuid.uuid4().hex[:8]
            ev = threading.Event()
            with self._pending_lock:
                self._pending_echo[token] = ev
            try:
                self._send_bytes(encode_echo(token))
            except RobotConnectionError as exc:
                logger.warning("EKI heartbeat send failed (%s); dropping link.", exc)
                self._stop_event.set()
                return
            got = ev.wait(timeout=miss_window)
            with self._pending_lock:
                self._pending_echo.pop(token, None)
            if not got:
                logger.warning(
                    "EKI heartbeat: no <EchoAck> within %.2fs; dropping link.",
                    miss_window,
                )
                self._stop_event.set()
                return
            self._stop_event.wait(timeout=self._heartbeat_s)
