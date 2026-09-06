"""The KUKA EthernetKRL wire protocol: XML message encode and decode.

Willy and KRL exchange newline-terminated XML frames over a single TCP socket. There
are two top-level tag families:

* ``<Sen Type="Willy">`` carries frames from the controller, meaning KRL, to Willy. It
  carries telemetry: the current pose, the joints, the steady flag, and the responses
  to FK, IK and move-complete requests.
* ``<Cmd Type="Willy">`` carries frames from Willy to the controller. It carries motion
  commands and FK and IK requests.

Wire format
-----------
* Each frame is a complete XML element followed by a single ``\\n`` byte.
* Whitespace inside an attribute value is preserved, and element text is unused.
* Frames are independent: there is no global session header.

Frame catalogue
---------------
From Willy to KRL, as ``<Cmd>``:

* ``<Move Mode="PTP|LIN" X=".." Y=".." Z=".." A=".." B=".." C=".."
  Vel=".." Acc=".."/>`` is a Cartesian move, PTP or LIN.
* ``<MoveJ J1=".." J2=".." ... J6=".." Vel=".." Acc=".."/>`` is a joint move, the KRL
  ``PTP $AXIS_ACT ...``. The joints are in degrees.
* ``<Stop/>`` aborts the current motion, through the KRL ``BRAKE`` and ``$STOPMESS``.
* ``<Home/>`` moves to the configured controller home.
* ``<FkRequest Id=".." J1=".." ... J6=".."/>`` asks KRL to compute FK.
* ``<IkRequest Id=".." X=".." ... C=".." S1=".." ... S6=".."/>`` asks KRL to compute
  IK, where ``S1..S6`` are the seed joints in degrees.
* ``<Echo Token=".."/>`` is the heartbeat and liveness probe.

From KRL to Willy, as ``<Sen>``:

* ``<State X=".." Y=".." Z=".." A=".." B=".." C=".." J1=".." ... J6=".."
  Steady="0|1"/>`` is the periodic full-state telemetry.
* ``<FkResult Id=".." X=".." ... C=".." Status="ok|error" Detail=".."/>``
* ``<IkResult Id=".." J1=".." ... J6=".." Status="ok|error" Detail=".."/>``
* ``<EchoAck Token=".."/>`` is the heartbeat reply.
* ``<Ack Cmd="Move|MoveJ|Stop|Home" Status="ok|error" Detail=".."/>`` is the generic
  command acknowledgement.

The KRL-side template itself ships under ``config/robot/templates/kuka/``.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

from .pose_convert import KukaCartesian

__all__ = [
    # Framing helpers
    "FRAME_TERMINATOR",
    "EkiAck",
    "EkiEchoAck",
    "EkiFkResult",
    "EkiIkResult",
    # Inbound
    "EkiState",
    "EkiUnknown",
    "decode_frame",
    "encode_echo",
    "encode_fk_request",
    "encode_home",
    "encode_ik_request",
    # Outbound
    "encode_move",
    "encode_movej",
    "encode_stop",
    "iter_frames",
]


FRAME_TERMINATOR = b"\n"


# ---------------------------------------------------------------------------
# Outbound encoders, from Willy to KRL
# ---------------------------------------------------------------------------


def _root_cmd() -> ET.Element:
    return ET.Element("Cmd", attrib={"Type": "Willy"})


def _fmt(value: float) -> str:
    """Format a float as a fixed-precision string KRL parses cheaply."""
    return f"{value:.6f}"


def _serialise(root: ET.Element) -> bytes:
    return ET.tostring(root, encoding="us-ascii", xml_declaration=False) + FRAME_TERMINATOR


def encode_move(
    target: KukaCartesian,
    *,
    mode: str = "PTP",
    vel: float | None = None,
    acc: float | None = None,
) -> bytes:
    """Encode a Cartesian move command frame, PTP or LIN."""
    if mode not in ("PTP", "LIN"):
        raise ValueError(f"mode must be 'PTP' or 'LIN', got {mode!r}")
    root = _root_cmd()
    attrib = {
        "Mode": mode,
        "X": _fmt(target.x), "Y": _fmt(target.y), "Z": _fmt(target.z),
        "A": _fmt(target.a), "B": _fmt(target.b), "C": _fmt(target.c),
    }
    if vel is not None:
        attrib["Vel"] = _fmt(float(vel))
    if acc is not None:
        attrib["Acc"] = _fmt(float(acc))
    ET.SubElement(root, "Move", attrib=attrib)
    return _serialise(root)


def encode_movej(
    joints_deg: list[float],
    *,
    vel: float | None = None,
    acc: float | None = None,
) -> bytes:
    """Encode a joint-space move command, the KRL ``PTP``."""
    if len(joints_deg) != 6:
        raise ValueError(f"MoveJ requires 6 joints in degrees; got {len(joints_deg)}")
    root = _root_cmd()
    attrib = {f"J{i + 1}": _fmt(float(j)) for i, j in enumerate(joints_deg)}
    if vel is not None:
        attrib["Vel"] = _fmt(float(vel))
    if acc is not None:
        attrib["Acc"] = _fmt(float(acc))
    ET.SubElement(root, "MoveJ", attrib=attrib)
    return _serialise(root)


def encode_stop() -> bytes:
    """Encode a ``<Stop/>`` command, which brakes on the controller side."""
    root = _root_cmd()
    ET.SubElement(root, "Stop")
    return _serialise(root)


def encode_home() -> bytes:
    """Encode a ``<Home/>`` command, a move to the home KRL is configured with."""
    root = _root_cmd()
    ET.SubElement(root, "Home")
    return _serialise(root)


def encode_fk_request(request_id: str, joints_deg: list[float]) -> bytes:
    """Encode an ``<FkRequest>`` command, asking KRL to compute FK."""
    if len(joints_deg) != 6:
        raise ValueError(f"FkRequest requires 6 joints; got {len(joints_deg)}")
    root = _root_cmd()
    attrib = {"Id": str(request_id)}
    attrib.update({f"J{i + 1}": _fmt(float(j)) for i, j in enumerate(joints_deg)})
    ET.SubElement(root, "FkRequest", attrib=attrib)
    return _serialise(root)


def encode_ik_request(
    request_id: str,
    target: KukaCartesian,
    seed_deg: list[float],
) -> bytes:
    """Encode an ``<IkRequest>`` command, asking KRL to compute IK."""
    if len(seed_deg) != 6:
        raise ValueError(f"IkRequest seed requires 6 joints; got {len(seed_deg)}")
    root = _root_cmd()
    attrib = {
        "Id": str(request_id),
        "X": _fmt(target.x), "Y": _fmt(target.y), "Z": _fmt(target.z),
        "A": _fmt(target.a), "B": _fmt(target.b), "C": _fmt(target.c),
    }
    attrib.update({f"S{i + 1}": _fmt(float(j)) for i, j in enumerate(seed_deg)})
    ET.SubElement(root, "IkRequest", attrib=attrib)
    return _serialise(root)


def encode_echo(token: str) -> bytes:
    """Encode an ``<Echo>`` heartbeat probe."""
    root = _root_cmd()
    ET.SubElement(root, "Echo", attrib={"Token": str(token)})
    return _serialise(root)


# ---------------------------------------------------------------------------
# Inbound decoded messages, from KRL to Willy
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EkiState:
    """Periodic state telemetry from the controller."""

    pose: KukaCartesian
    joints_deg: tuple
    steady: bool


@dataclass(frozen=True, slots=True)
class EkiAck:
    """Generic command acknowledgement."""

    command: str
    status: str  # "ok" | "error"
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class EkiFkResult:
    """Reply to an FK request."""

    request_id: str
    pose: KukaCartesian | None
    status: str
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class EkiIkResult:
    """Reply to an IK request."""

    request_id: str
    joints_deg: tuple | None
    status: str
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class EkiEchoAck:
    """Reply to a heartbeat echo."""

    token: str


@dataclass(frozen=True, slots=True)
class EkiUnknown:
    """The fallback for a tag this protocol does not model."""

    tag: str
    raw: str


def decode_frame(payload: bytes) -> object:
    """Parse one inbound frame into a typed message object.

    Raises
    ------
    ValueError
        If ``payload`` is not well-formed XML or the root element is
        not ``<Sen>``.
    """
    if not payload.strip():
        raise ValueError("empty EKI frame")
    try:
        root = ET.fromstring(payload.decode("us-ascii", errors="replace"))
    except ET.ParseError as exc:
        raise ValueError(f"malformed EKI XML frame: {exc}") from exc
    if root.tag != "Sen":
        raise ValueError(f"expected <Sen> root, got <{root.tag}>")
    children = list(root)
    if not children:
        return EkiUnknown(tag="<Sen/>", raw=root.attrib.get("Type", ""))
    child = children[0]
    if child.tag == "State":
        return _decode_state(child)
    if child.tag == "Ack":
        return EkiAck(
            command=child.attrib.get("Cmd", "?"),
            status=child.attrib.get("Status", "error"),
            detail=child.attrib.get("Detail"),
        )
    if child.tag == "FkResult":
        return _decode_fk(child)
    if child.tag == "IkResult":
        return _decode_ik(child)
    if child.tag == "EchoAck":
        return EkiEchoAck(token=child.attrib.get("Token", ""))
    return EkiUnknown(tag=child.tag, raw=ET.tostring(child, encoding="us-ascii").decode("us-ascii"))


def _decode_state(elem: ET.Element) -> EkiState:
    pose = KukaCartesian(
        x=float(elem.attrib["X"]),
        y=float(elem.attrib["Y"]),
        z=float(elem.attrib["Z"]),
        a=float(elem.attrib["A"]),
        b=float(elem.attrib["B"]),
        c=float(elem.attrib["C"]),
    )
    joints = tuple(float(elem.attrib[f"J{i + 1}"]) for i in range(6))
    steady_attr = elem.attrib.get("Steady", "0").strip()
    steady = steady_attr in ("1", "true", "True")
    return EkiState(pose=pose, joints_deg=joints, steady=steady)


def _decode_fk(elem: ET.Element) -> EkiFkResult:
    status = elem.attrib.get("Status", "error")
    request_id = elem.attrib.get("Id", "")
    pose: KukaCartesian | None
    if status == "ok":
        pose = KukaCartesian(
            x=float(elem.attrib["X"]),
            y=float(elem.attrib["Y"]),
            z=float(elem.attrib["Z"]),
            a=float(elem.attrib["A"]),
            b=float(elem.attrib["B"]),
            c=float(elem.attrib["C"]),
        )
    else:
        pose = None
    return EkiFkResult(
        request_id=request_id, pose=pose, status=status,
        detail=elem.attrib.get("Detail"),
    )


def _decode_ik(elem: ET.Element) -> EkiIkResult:
    status = elem.attrib.get("Status", "error")
    request_id = elem.attrib.get("Id", "")
    joints: tuple | None
    if status == "ok":
        joints = tuple(float(elem.attrib[f"J{i + 1}"]) for i in range(6))
    else:
        joints = None
    return EkiIkResult(
        request_id=request_id, joints_deg=joints, status=status,
        detail=elem.attrib.get("Detail"),
    )


# ---------------------------------------------------------------------------
# Framing helper for the reader thread
# ---------------------------------------------------------------------------


def iter_frames(buffer: bytearray) -> list[bytes]:
    """Pull complete ``\\n``-terminated frames out of ``buffer`` in place.

    It returns the list of complete frames without their trailing newline, and leaves
    the trailing partial frame at the start of ``buffer`` for the next read.
    """
    frames: list[bytes] = []
    while True:
        nl = buffer.find(FRAME_TERMINATOR)
        if nl < 0:
            return frames
        frame = bytes(buffer[:nl])
        del buffer[: nl + 1]
        if frame.strip():
            frames.append(frame)
