"""Writing a measured value back into the config tree, without destroying the file it lands in.

Three of the numbers this stack depends on cannot be read out of anything: payload mass, the
flange-to-grasp-centre transform, and which physical camera is on which rig. A person measures them
at a bench with a scale, a caliper and ``rs-enumerate-devices``, and the cell refuses to run until
they are typed in. Putting such a measurement where the loader will find it is the whole job here.
This is not a config editor.

Only :data:`WRITABLE` may be written. Everything under ``safety.*`` beyond the payload measurement,
every workspace and motion limit and every threshold stays YAML-only: those values sit next to the
comments that carry the evidence for them ("6/6 up to 6 mm and 0/6 at 10 mm"), and a value changed
without reading its comment is a value changed without knowing what it was for. The payload and the
tool frame are measurements rather than policy, which is what makes them the exception.

Nothing here round-trips YAML. A ``safe_load``/``dump`` cycle silently deletes every comment in the
file, which in this tree is most of the knowledge, so the editor only rewrites the value on a single
existing line or inserts new lines; every other byte is copied through untouched.

After writing, the whole tree is reloaded through the real loader and the real validators. On any
failure (a typo, a cross-field validator, a rejected enum, a bug in the line editor itself) the
original file is restored and the validation error is returned, so the tree is never left in a state
the loader would refuse.

A pose taught by hand in the console is no measurement either, and it has a door of its own:
:func:`set_named_pose` and :func:`set_default_place_pose` write ``robot.named_poses`` and
``robot.default_place_pose``, through the same transaction, into the last layer of the profile chain
alone (the owner, Q10: the cell's own, untracked layer). Inside a git work tree that layer must be one
git keeps out, ignored and tracked nowhere (:func:`pose_layer_refusal`), so one cell's poses never land
in a file every clone shares. The joints are where a person guided the arm, screened by the exact guard
and the planner while it held; typed into the generic writer they would skip both, so :func:`set_keys`
refuses every pose key with a sentence of its own (:data:`POSE_KEYS`).
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from ._provenance import section_sources
from ._schema_index import schema_index
from .schema.robot.robot_schema import pose_label_refusal, pose_name_refusal

__all__ = [
    "MISSING",
    "POSE_KEYS",
    "WRITABLE",
    "WRITTEN_SCREENS",
    "WriteRefused",
    "WriteResult",
    "Writable",
    "is_pose_key",
    "pose_layer_refusal",
    "pose_target_file",
    "read_key",
    "set_default_place_pose",
    "set_key",
    "set_keys",
    "set_named_pose",
    "target_file",
    "writable",
]


class WriteRefused(StrEnum):
    """Why a write did not happen. Typed, since the console renders its own form for each member."""

    #: A real key that a tool may not write. The module docstring says which keys stay YAML-only.
    NOT_WRITABLE = "not_writable"
    #: The schema does not accept the key at all.
    UNKNOWN_KEY = "unknown_key"
    #: The value was written, the loader rejected the result, the file was restored.
    INVALID_VALUE = "invalid_value"
    #: The file the value would land in does not exist, or the key's section is not file-backed.
    NO_TARGET = "no_target"
    #: The key decides which machine moves, and something is currently connected to one.
    CELL_CONNECTED = "cell_connected"
    #: A taught pose, and the chain names no layer of the cell's own: none at all, so it would land in the shared base
    #: file every cell reads, or a last layer git does not keep out, so it would land in the repository
    #: (:func:`pose_layer_refusal`). Only the pose door returns it.
    NO_LAYER = "no_layer"
    #: A taught pose under a name that is none (``pose_name_refusal``); nothing was written. Only the pose door.
    INVALID_NAME = "invalid_name"
    #: A taught pose under a label that is none (``pose_label_refusal``); nothing was written. Only the pose door.
    INVALID_LABEL = "invalid_label"


@dataclass(frozen=True, slots=True)
class Writable:
    """One key an operator may set, with the sentence that tells them what they are measuring."""

    path: str
    #: What to call it in a form.
    label: str
    #: The physical act that produces the number. A form shows this sentence in place of a type.
    measure: str
    unit: str = ""
    #: Only offered for this ``robot.vendor``. Empty means every cell. The controller's address is
    #: the same fact under a different key per vendor, so only the entry for the cell's vendor is
    #: offered.
    vendor: str = ""
    #: True for values that decide which machine moves or how it is driven, rather than what the tool
    #: weighs. Writable only with nothing connected; the guard is ``connected`` in :func:`set_keys`.
    requires_disconnected: bool = False


#: The complete writable set. A key belongs here only when no API can answer it and the cell blocks
#: until a human does.
WRITABLE: tuple[Writable, ...] = (
    Writable(
        path="robot.safety.payload.mass_kg",
        label="Payload mass",
        measure=(
            "Weigh the whole assembly on a bench scale: gripper plus coupling/adapter plate, every "
            "cable and hose that rides on the wrist, and any workpiece the arm carries. Not the "
            "gripper's datasheet mass."
        ),
        unit="kg",
    ),
    Writable(
        path="robot.safety.payload.cog_mm",
        label="Payload centre of gravity",
        measure=(
            "From the flange face, in the same session as the mass. Leaving it at [0,0,0] declares a "
            "multi-kilogram tool to be a point mass at the flange: at 3 kg and 132 mm that is "
            "3.88 N*m of wrist torque the controller does not model."
        ),
        unit="mm",
    ),
    Writable(
        path="robot.safety.planning_world.payload.length_mm",
        label="Carried part length",
        measure=(
            "How far your longest part hangs below the closed fingertips, in millimetres. The planner carries a "
            "box of this length under the jaws while a part is held; a task needs it before it starts."
        ),
        unit="mm",
    ),
    Writable(
        path="robot.gripper.tool_frame.source",
        label="Tool frame owner",
        measure=(
            "'willy' if the controller runs a bare flange and this driver composes the offset; "
            "'polyscope' if an operator set the TCP on the pendant and the driver only verifies it. "
            "Either way connect() derives what the controller is actually running and refuses a "
            "mismatch."
        ),
    ),
    Writable(
        path="robot.gripper.tool_frame.offset_mm",
        label="Flange -> grasp centre offset",
        measure="Where the grasp centre sits relative to the flange face, measured on the real coupling.",
        unit="mm",
    ),
    Writable(
        path="robot.gripper.tool_frame.rotation_quat_xyzw",
        label="Flange -> grasp centre rotation",
        measure=(
            "Which flange axis the jaws close along. This half never crashes: ninety degrees out "
            "produces 10/10 logged successes with the jaws closing across the wrong object axis, and "
            "hand-eye calibration returns an excellent RMSE either way."
        ),
    ),
    Writable(
        path="camera.cameras.rigs[*].serial_number",
        label="Camera serial",
        measure=(
            "rs-enumerate-devices -s. With two identical D435s the serial is the only stable identity; "
            "device_index is the SDK's enumeration order and can swap between boots. Cameras that swap "
            "do not fail; they hand back a complete, plausible scene with the views exchanged."
        ),
    ),
    Writable(
        path="camera.cameras.primary_rig_id",
        label="Primary camera",
        measure=(
            "Which of the rigs below this cell opens. The grasp is synthesised from its depth, so "
            "every other camera confirms what this one saw and moving it moves every number the "
            "cell measures. It has to be a rig with source: rgbd, and one that is enabled: a cell "
            "built on anything else is refused at build with a message naming both. Writable here "
            "for the same reason a camera serial is: it is a fact about the cell, and it is the "
            "fact you set immediately after the serials, once you know which camera is which. It "
            "cannot be changed while anything is connected, because it decides where the next "
            "grasp comes from."
        ),
        requires_disconnected=True,
    ),
    Writable(
        path="camera.cameras.rigs[*].enabled",
        label="Camera in use",
        measure=(
            "Whether this rig is part of the cell. The bench order is: read the serials with "
            "rs-enumerate-devices -s, write them in, then switch the rigs on. This is that last "
            "step, and without it an operator could set a serial through the console and still had "
            "to open the YAML by hand to use it. Read only when a cell is built, so switching a rig "
            "while one is running changes nothing until the next build, which is why it is not "
            "refused while connected: that would block the repair it exists for."
        ),
    ),
    Writable(
        path="robot.ur.ip",
        label="UR controller address",
        measure=(
            "The controller's address on your network; read it off the pendant under Settings > "
            "Network. It is a fact about the cell, like a camera serial, which is why it is writable "
            "here at all. It is also the one value that decides which machine receives every motion, "
            "so it cannot be changed while anything is connected."
        ),
        vendor="ur",
        requires_disconnected=True,
    ),
    Writable(
        path="robot.kuka.controller_ip",
        label="KUKA controller address",
        measure=(
            "The controller's address on your network. As with UR: a fact about the cell, and the one "
            "value that decides which machine moves, so it is refused while anything is connected."
        ),
        vendor="kuka",
        requires_disconnected=True,
    ),
)

_INDEXED = re.compile(r"^(?P<head>.*?)\[(?P<index>\d+)\](?P<tail>.*)$")


def writable(key: str) -> Writable | None:
    """The :class:`Writable` ``key`` names, or ``None``. An index matches the ``[*]`` template."""
    generic = re.sub(r"\[\d+\]", "[*]", key)
    for entry in WRITABLE:
        if entry.path in (key, generic):
            return entry
    return None


#: The keys a taught pose is written under. Never in :data:`WRITABLE`, ``/v1/config/writable`` or a PATCH: only the
#: pose door (:func:`set_named_pose`, :func:`set_default_place_pose`) writes them, and :func:`set_keys` refuses every
#: key under ``robot.named_poses`` and ``robot.default_place_pose`` with :data:`_A_POSE_IS_TAUGHT`.
POSE_KEYS: tuple[str, ...] = (
    "robot.named_poses.*.joints_deg",
    "robot.named_poses.*.label",
    "robot.named_poses.*.taught_at",
    "robot.named_poses.*.screen",
    "robot.named_poses.*.note",
    "robot.default_place_pose",
)
#: The screens a taught pose is written with: both the exact guard and the planner clear it, or it lies in the
#: planner's cushion band. An unscreened or refused pose never is (the owner, 2026-09-30).
WRITTEN_SCREENS: tuple[str, ...] = ("clear", "band")

_POSE_FIELD = re.compile(r"^robot\.named_poses\.[A-Za-z_][A-Za-z0-9_]*\.(?:joints_deg|label|taught_at|screen|note)$")
#: What the pose door allows: one entry for every key of :data:`POSE_KEYS`, none of which disconnects anything.
_A_TAUGHT_POSE = Writable(
    path="robot.named_poses.*",
    label="Taught pose",
    measure=("A person guides the arm there by hand in the console; the exact guard and the planner screen it while it "
             "holds, and only a clear pose or one in the planner's band is written."),
)
#: Why the generic writer refuses a pose key.
_A_POSE_IS_TAUGHT = (
    "{key} is a taught pose: a pose is taught by hand in the console (Setup, 'Pose einlernen'), screened by the exact "
    "guard and the planner while the arm holds, and written by the pose writer alone, into the cell's own layer. Typed "
    "joints would skip the hand and the screen. To rename or delete a pose, edit the YAML."
)


def is_pose_key(key: str) -> bool:
    """Whether ``key`` is under ``robot.named_poses`` or is ``robot.default_place_pose``: a taught pose's key."""
    stripped = re.sub(r"\[\d+\]", "", key)
    return stripped in ("robot.named_poses", "robot.default_place_pose") or stripped.startswith("robot.named_poses.")


def _pose_door(key: str) -> Writable | None:
    """The pose door's allowlist: a key of :data:`POSE_KEYS`, under a name, and nothing else."""
    return _A_TAUGHT_POSE if key == "robot.default_place_pose" or _POSE_FIELD.match(key) else None


def _known(key: str) -> bool:
    """Whether the schema accepts ``key``. ``rigs[0].x`` is stored under the list's element shape."""
    index = schema_index()
    if key in index:
        return True
    stripped = re.sub(r"\[\d+\]", "", key)
    return stripped in index or f"{stripped}" in {re.sub(r"\[\d+\]", "", k) for k in index}


def target_file(key: str, root: Path, layers: tuple[str, ...]) -> Path | None:
    """The file a write to ``key`` must land in, given the profile chain being run.

    The target is the overlay for the last active layer, created on write if it does not exist yet;
    only a chain with no layers at all writes the base file. A measurement belongs to the most
    specific layer: a payload weighed on the UR3e is a fact about the UR3e, and the shared base file
    would hand the same number to the UR5e and to the simulator.
    """
    for base, _top, prefix in section_sources(root):
        stripped = re.sub(r"\[\d+\]", "", key)
        if stripped == prefix or stripped.startswith(f"{prefix}."):
            if not layers:
                return base if base.exists() else None
            return base.with_name(f"{base.stem}.{layers[-1]}{base.suffix}")
    return None


@dataclass(frozen=True, slots=True)
class WriteResult:
    """What happened to the whole group. A caller has to branch on ``applied`` and nothing else."""

    applied: bool
    keys: tuple[str, ...]
    #: Read back out of the reloaded tree, not echoed from the request: the value that survived
    #: coercion and the validators is the one the cell runs with. Empty when refused.
    values: dict[str, Any] = field(default_factory=dict)
    files: tuple[Path, ...] = ()
    refused: WriteRefused | None = None
    #: The key a refusal names, when it names one. Empty when the whole tree failed validation.
    refused_key: str = ""
    #: Operator-readable. For ``INVALID_VALUE`` this is the loader's own message, unabridged; it
    #: already names the file, the line and the validator's own sentence.
    message: str = ""


# --------------------------------------------------------------------------------------------------
# YAML leaf writing. Line-based rather than a YAML round-trip: see the module docstring.
# --------------------------------------------------------------------------------------------------

def _emit(value: Any) -> str:
    """A scalar or flat list as this tree writes it: flow lists, quoted strings, ``null``."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_emit(item) for item in value) + "]"
    text = str(value)
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _block_end(lines: list[str], start: int, indent: int) -> int:
    """Index one past the block opened at ``start``. Its lines are those indented past ``indent``.

    Blanks and comments inside the block count as part of it; trailing ones do not, so an insert
    lands against the block's last real line instead of after the blank that separates the next.
    """
    end = start + 1
    last_real = start + 1
    while end < len(lines):
        stripped = lines[end].strip()
        if stripped and not stripped.startswith("#") and _indent_of(lines[end]) <= indent:
            break
        if stripped and not stripped.startswith("#"):
            last_real = end + 1
        end += 1
    return last_real


def _find_child(lines: list[str], lo: int, hi: int, name: str, indent: int) -> int | None:
    """Line index of ``name:`` at exactly ``indent`` within ``[lo, hi)``."""
    pattern = re.compile(rf"^ {{{indent}}}{re.escape(name)}:(\s|$)")
    for i in range(lo, min(hi, len(lines))):
        if pattern.match(lines[i]):
            return i
    return None


def _find_item(lines: list[str], lo: int, hi: int, position: int, indent: int) -> tuple[int, int] | None:
    """``(line, item_indent)`` of the ``position``-th ``- `` entry at ``indent`` within ``[lo, hi)``."""
    pattern = re.compile(rf"^ {{{indent}}}-(\s|$)")
    seen = 0
    for i in range(lo, min(hi, len(lines))):
        if pattern.match(lines[i]):
            if seen == position:
                # A sequence item's own keys sit at the column after "- ", whether or not the first key
                # shares the dash's line, which in this tree it always does.
                return i, indent + 2
            seen += 1
    return None


_SET_LINE = re.compile(r"^(?P<lead>\s*(?:-\s+)?)(?P<key>[A-Za-z_][\w.]*):(?P<gap>\s*)(?P<rest>.*)$")
#: A block written as an empty flow mapping, ``named_poses: {}``, with a comment after it or none.
_EMPTY_FLOW_MAP = re.compile(r"^(?P<head>\s*(?:-\s+)?[A-Za-z_][\w.]*:)\s*\{\s*\}(?P<comment>\s+#.*?)?\s*$")


def _opened(line: str) -> str:
    """``line`` with an empty flow mapping opened as a block that takes children (``named_poses:``), its comment kept;
    any other line as it is. A child inserted under ``named_poses: {}`` itself is no YAML the loader reads."""
    match = _EMPTY_FLOW_MAP.match(line)
    if match is None:
        return line
    return f"{match['head']}{match['comment'] or ''}"


def _rewrite(line: str, value: Any) -> str:
    """Replace the value on an existing ``key: value`` line, keeping indent, padding and any comment."""
    match = _SET_LINE.match(line)
    if match is None:  # pragma: no cover (callers only pass lines this matched already)
        raise ValueError(f"not a settable line: {line!r}")
    rest = match["rest"]
    comment = ""
    # A `#` starts a comment only after whitespace; one inside a quoted value does not. This module
    # writes numbers, short flow lists and bare identifiers, so the conservative split covers them.
    hit = re.search(r"(?<=\s)#.*$", rest)
    if hit:
        comment = "  " + hit.group(0).strip()
    gap = match["gap"] or " "
    return f"{match['lead']}{match['key']}:{gap}{_emit(value)}{comment}"


def _write_leaf(path: Path, top_key: str | None, dotted: str, value: Any) -> None:
    """Set ``dotted`` (relative to ``top_key``) in ``path``, creating the file or nesting if needed.

    Handles the three shapes this tree contains: a leaf already written (rewrite the line), a leaf
    missing under a parent block that exists (insert), and a parent chain that does not exist yet
    (create the nesting). A missing overlay file is created with just the path it needs.
    """
    segments: list[str | int] = []
    for raw in dotted.split("."):
        match = _INDEXED.match(raw)
        if match is None:
            segments.append(raw)
            continue
        if match["head"]:
            segments.append(match["head"])
        segments.append(int(match["index"]))

    if not path.exists():
        lines: list[str] = []
        if top_key:
            lines.append(f"{top_key}:")
    else:
        lines = path.read_text(encoding="utf-8").splitlines()

    lo, hi, indent, parent = 0, len(lines), 0, -1
    if top_key:
        found = _find_child(lines, 0, len(lines), top_key, 0)
        if found is None:
            lines.append(f"{top_key}:")
            found = len(lines) - 1
            hi = len(lines)
        else:
            lines[found] = _opened(lines[found])
            hi = _block_end(lines, found, 0)
        lo, indent, parent = found + 1, 2, found

    for position, segment in enumerate(segments):
        last = position == len(segments) - 1
        if isinstance(segment, int):
            item = _find_item(lines, lo, hi, segment, indent)
            if item is None:
                raise ValueError(f"{path.name}: no item [{segment}] under the block ending at line {hi}")
            parent, indent = item[0], item[1]
            lo, hi = parent, _block_end(lines, parent, indent - 2)
            continue

        found = _find_child(lines, lo, hi, segment, indent)
        if found is None:
            # Nothing from here down exists. It goes in as one block at the parent's insertion point.
            insert = hi if parent >= 0 else len(lines)
            block: list[str] = []
            depth = indent
            for tail in segments[position:-1]:
                block.append(" " * depth + f"{tail}:")
                depth += 2
            block.append(" " * depth + f"{segments[-1]}: {_emit(value)}")
            lines[insert:insert] = block
            break
        if last:
            lines[found] = _rewrite(lines[found], value)
            break
        lines[found] = _opened(lines[found])
        parent, lo = found, found + 1
        hi = _block_end(lines, found, indent)
        indent += 2

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def set_keys(
    items: Mapping[str, Any],
    *,
    root: Path,
    layers: tuple[str, ...] = (),
    profile: str | None = None,
    connected: bool = False,
    allowed: Callable[[str], Writable | None] = writable,
) -> WriteResult:
    """Write a group of measured values as one transaction: all of them land, or none do.

    The group is a correctness requirement rather than a convenience. The three ``tool_frame`` keys
    are one measurement and the schema knows it: ``source: "willy"`` with the offset still identity
    is rejected, because a grasp centre exactly at the flange face is the shape of an unmeasured cell
    rather than a mounted tool. Written one at a time, no order validates: the first write is always
    half a tool frame. So every file is edited, the tree is reloaded once, and on any failure every
    file is restored to the bytes it had.

    ``profile`` is the chain string the reload runs under; ``layers`` is the same chain split, used
    to pick the target files. They must describe the same chain and are checked against each other
    before anything is written: ``layers`` picks the file a value lands in and ``profile`` picks the
    tree that then has to validate, so a disagreement lands a write in a file the validation never
    reads. `ConfigTree.write()` derives both from one chain, so the disagreeing call cannot be
    constructed there at all.

    ``allowed`` is the allowlist a key is read against: :func:`writable`, the measurements of
    :data:`WRITABLE`, for every caller but the pose door, which hands its own. A taught pose's key
    that the allowlist does not take is refused ``NOT_WRITABLE`` with its own sentence, never as an
    unknown key: it is real, and only the pose door writes it.
    """
    # A mismatched pair such as layers=("ur3e",) with profile=None writes into the layer file but
    # validates the base tree: no rollback fires, the read-back reports the base tree's value rather
    # than the one written, and the tree is left unloadable under its own layer while the call
    # reports success.
    #
    # Raised rather than returned as a `WriteRefused`: every member of that enum is something an
    # operator did, and this is something a caller did, unreachable through any UI. A refusal would
    # put a programming error in front of a person who cannot act on it. This guards the raw
    # function; `ConfigTree.write()` makes the call unconstructible instead.
    from .loader import profile_layers  # noqa: PLC0415

    if tuple(profile_layers(profile)) != tuple(layers):
        raise ValueError(
            f"layers={layers!r} and profile={profile!r} describe different chains. `layers` picks "
            f"the file a value lands in and `profile` picks the tree that then has to validate; "
            f"when they disagree a write can land in a file the validation never reads. Derive both "
            f"from one chain, or use ConfigTree.write(), which does."
        )

    if not items:
        return WriteResult(applied=True, keys=())

    plan: list[tuple[str, Any, Path, str | None, str]] = []
    for key, value in items.items():
        entry = allowed(key)
        if entry is None and is_pose_key(key):
            return WriteResult(
                applied=False, keys=tuple(items), refused=WriteRefused.NOT_WRITABLE, refused_key=key,
                message=_A_POSE_IS_TAUGHT.format(key=key),
            )
        if entry is not None and entry.requires_disconnected and connected:
            # The controller address is the one writable value that does not describe the tool; it
            # decides which machine every motion goes to. Changed under a live connection it leaves
            # the process driving one arm while the config names another, and nothing downstream
            # notices: the driver holds the address it connected with.
            return WriteResult(
                applied=False, keys=tuple(items), refused=WriteRefused.CELL_CONNECTED, refused_key=key,
                message=(
                    f"{key} decides which machine receives every motion, so it cannot be changed while "
                    f"anything is connected. Disconnect the cell first."
                ),
            )
        if entry is None:
            if not _known(key):
                return WriteResult(
                    applied=False, keys=tuple(items), refused=WriteRefused.UNKNOWN_KEY,
                    refused_key=key,
                    message=(
                        f"{key} is not a config key. The schema rejects unknown keys at load, so "
                        f"writing it would break the tree rather than change anything."
                    ),
                )
            return WriteResult(
                applied=False, keys=tuple(items), refused=WriteRefused.NOT_WRITABLE, refused_key=key,
                message=(
                    f"{key} is real but not writable from a tool. Limits, thresholds and safety "
                    f"toggles live next to the comment that carries the evidence for their value; edit "
                    f"them in YAML, where that comment is readable. Writable here: "
                    + ", ".join(w.path for w in WRITABLE)
                ),
            )
        path = target_file(key, root, layers)
        section = next(
            (s for s in section_sources(root) if re.sub(r"\[\d+\]", "", key).startswith(s[2])), None
        )
        if path is None or section is None:
            return WriteResult(
                applied=False, keys=tuple(items), refused=WriteRefused.NO_TARGET, refused_key=key,
                message=f"no file backs {key} under the config root {root}.",
            )
        _base, top_key, prefix = section
        plan.append((key, value, path, top_key, key[len(prefix):].lstrip(".")))

    # Snapshot every file the group touches before editing any of them; `None` marks one that did not
    # exist, so a rollback deletes it instead of writing an empty file back. Bytes rather than text: a
    # text round trip rewrites every line ending to the platform's, so an LF file on Windows would
    # come back CRLF.
    snapshot: dict[Path, bytes | None] = {
        path: (path.read_bytes() if path.exists() else None)
        for _k, _v, path, _t, _d in plan
    }
    try:
        for _key, value, target, section_key, dotted in plan:
            _write_leaf(target, section_key, dotted, value)
        loaded = _reload(root, profile)
    except Exception as exc:  # noqa: BLE001 (any failure means the same thing: put the files back)
        for path, before in snapshot.items():
            if before is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(before)
        _reload(root, profile)
        return WriteResult(
            applied=False, keys=tuple(items), files=tuple(snapshot), refused=WriteRefused.INVALID_VALUE,
            message=f"{exc}",
        )

    read_back = {key: read_key(loaded, key) for key in items}
    absent = [key for key, value in read_back.items() if value is MISSING]
    if absent:  # pragma: no cover (a live tree that validates always exposes what it validated)
        return WriteResult(
            applied=False, keys=tuple(items), files=tuple(snapshot), refused=WriteRefused.NO_TARGET,
            refused_key=absent[0],
            message=f"wrote {absent[0]} and the reloaded tree does not expose it, refusing to claim it applied.",
        )
    return WriteResult(applied=True, keys=tuple(items), files=tuple(snapshot), values=read_back)


def set_key(
    key: str,
    value: Any,
    *,
    root: Path,
    layers: tuple[str, ...] = (),
    profile: str | None = None,
    connected: bool = False,
) -> WriteResult:
    """One key, written as a group of one. :func:`set_keys` says why the group is the primitive."""
    return set_keys(
        {key: value}, root=root, layers=layers, profile=profile, connected=connected,
    )


# --------------------------------------------------------------------------------------------------
# A taught pose's own door.
# --------------------------------------------------------------------------------------------------

def pose_target_file(root: Path, layers: tuple[str, ...]) -> Path | None:
    """The file a taught pose is written to under this chain: the robot overlay of its LAST layer.

    ``None`` for a chain with no layer, where the generic writer's target is the shared base ``robot.yaml``
    every cell reads, and a pose is refused (:attr:`WriteRefused.NO_LAYER`); ``None`` too for a tree with no robot
    section. The owner's chain ends in the cell's own layer, untracked (Q10), so this names
    ``robot/robot.cell.yaml`` there. Reads nothing and writes nothing: the console says it before the arm is freed.
    Whether the file may take a pose (git keeps it out) is :func:`pose_layer_refusal`'s to say.
    """
    if not layers:
        if _is_shipped_tree(root):
            return None
        # A cell's own tree, outside this repository's config folder (``--data``): its base robot.yaml is the cell's
        # own file, as a layer would be, and git keeps it out or no git holds it (:func:`pose_layer_refusal`).
        return target_file("robot.named_poses", root, layers)
    return target_file("robot.named_poses", root, layers)


def _is_shipped_tree(root: Path) -> bool:
    """Whether ``root`` is this repository's own config folder, whose base ``robot.yaml`` every cell reads."""
    shipped = Path(__file__).resolve().parents[2] / "config"
    try:
        return Path(root).resolve() == shipped.resolve()
    except OSError:  # pragma: no cover (a root that cannot be resolved is treated as the shared one)
        return True


def _one_line(text: str) -> str:
    """``text`` as one line a later rewrite keeps: every run of whitespace, newlines included, one space; a ``#`` after
    whitespace dropped, since the line editor takes it for a comment; characters a line cannot show dropped."""
    shown = "".join(character for character in " ".join(str(text).split()) if character.isprintable())
    return " ".join(re.sub(r"(?<=\s)#+", " ", shown).split())


#: How long git may take to say whether it keeps the cell's layer out, in seconds. A git that does not answer in time
#: keeps nothing out that anyone could vouch for.
_GIT_TIMEOUT_S = 10.0
#: How a chain gets a layer of the cell's own: the loader refuses a layer that has no file, so the file comes first.
_CELL_LAYER = ("Run the console with the cell's chain, ending in its own layer (for example --profile "
               "ur10,hande,cell): create that layer's robot overlay first, robot/robot.cell.yaml under the config root "
               "(one comment line is enough; the loader refuses a layer with no file of its own), and git-ignore it.")


def pose_layer_refusal(root: Path, layers: tuple[str, ...], *, what: str = "a taught pose") -> str:
    """Why ``what`` may not be written under this chain, in one line, or ``""`` where it may (the owner, Q10).

    A taught pose goes into the robot overlay of the chain's LAST layer (:func:`pose_target_file`), and that layer must
    be the cell's own, one git keeps out:

    * a chain with no layer is refused: its target is the shared base ``robot.yaml`` every cell reads;
    * inside a git work tree the file must be one git ignores and tracks nowhere (``git check-ignore``, which reads the
      index, ``.gitignore``, ``.git/info/exclude`` and the user's own excludes). A file git tracks is shared by every
      clone (``robot.hande.yaml`` ending ``--profile ur10,hande`` reaches every Hand-E chain on every arm), and one git
      neither tracks nor ignores goes into the repository with the next ``git add``;
    * a work tree whose git cannot be asked (git not on the PATH, a repository it refuses to read) keeps nothing out
      that anyone could vouch for, and is refused as well, never written on a guess.

    A tree in no git work tree asks git nothing. Nothing is written: the console says it before the arm is freed, and
    the pose door asks again before it writes. The sentence says what happens and what to do; the caller adds what was
    not done (nothing written, nothing freed).
    """
    if not layers and _is_shipped_tree(root):
        return (f"{what} is written into the last layer of the profile chain, the cell's own, and this chain names no "
                f"layer, so it would land in the shared robot.yaml every cell reads. {_CELL_LAYER}")
    target = pose_target_file(root, layers)
    if target is None:
        return ""  # no robot section to write into: the writer's own refusal says so (NO_TARGET)
    return _kept_out_of_git(target, what, layers)


def _kept_out_of_git(target: Path, what: str, layers: tuple[str, ...]) -> str:
    """``""`` where git keeps ``target`` out, or no git work tree holds it; else why it does not, with the remedy."""
    folder = target.parent.resolve()
    top = next((above for above in (folder, *folder.parents) if (above / ".git").exists()), None)
    if top is None:
        return ""
    try:
        shown = target.resolve().relative_to(top).as_posix()
    except ValueError:  # pragma: no cover (the work tree was found above the file itself)
        shown = str(target)
    try:
        ignored = _git(folder, "check-ignore", "-q", "--", target.name)
        # check-ignore reads the index: a tracked file is never ignored, whatever a pattern says. Which of the two it is
        # decides the remedy.
        tracked = (ignored.returncode == 1
                   and _git(folder, "ls-files", "--error-unmatch", "--", target.name).returncode == 0)
    except (OSError, subprocess.SubprocessError) as exc:
        return (f"{what} goes into {shown}, which lies in a git work tree, and git could not be asked whether it keeps "
                f"that file out ({type(exc).__name__}: {exc}): a layer git may carry into the repository is no cell's "
                "own. Put git on the console's PATH, or move the config out of the work tree, then teach.")
    if ignored.returncode == 0:
        return ""
    if ignored.returncode != 1:
        said = " ".join((ignored.stderr or "").split()) or f"it ended with {ignored.returncode}"
        return (f"{what} goes into {shown}, which lies in a git work tree, and git could not say whether it keeps that "
                f"file out ({said}): a layer git may carry into the repository is no cell's own.")
    if tracked:
        chain = ",".join(layers if layers[-1] == "cell" else (*layers, "cell"))
        return (f"{what} goes into the last layer of the chain, {shown}, and that file is tracked by git: every clone "
                "of the repository shares it, and one cell's poses do not belong there (Q10). End the chain in a layer "
                f"of the cell's own that git ignores (for example --profile {chain}), or, where this file is the cell's "
                f"own, stop tracking it (git rm --cached {shown}) and add it to .gitignore.")
    return (f"{what} goes into {shown}, the cell's own layer, and that file is not git-ignored: the next `git add` "
            f"would carry this cell's poses into the repository (Q10). Add the line {shown} to .gitignore (or to "
            ".git/info/exclude, which keeps it out on this machine alone), then teach.")


def _git(folder: Path, *args: str) -> "subprocess.CompletedProcess[str]":
    """``git <args>`` run in ``folder``, its output kept; never raises for git's own exit code."""
    return subprocess.run(["git", *args], cwd=folder, capture_output=True, encoding="utf-8", errors="replace",
                          timeout=_GIT_TIMEOUT_S, check=False)


def _no_layer(keys: tuple[str, ...], why: str) -> WriteResult:
    """The refusal of a pose write under a chain with no layer of the cell's own (:func:`pose_layer_refusal`)."""
    return WriteResult(
        applied=False, keys=keys, refused=WriteRefused.NO_LAYER, refused_key=keys[0] if keys else "",
        message=f"{why} Nothing was written.",
    )


def set_named_pose(
    name: str,
    *,
    joints_deg: Sequence[float],
    label: str,
    screen: str,
    taught_at: str,
    note: str = "",
    make_default_place: bool = False,
    root: Path,
    layers: tuple[str, ...],
    profile: str | None,
) -> WriteResult:
    """Write one taught pose, and the default place with it where asked, as one transaction into the last layer.

    The group is ``robot.named_poses.<name>``'s five keys (the joints in degrees, the label, when it was taught, the
    screen and a note), plus ``robot.default_place_pose: <name>`` with ``make_default_place``, written through
    :func:`set_keys` with the pose door's own allowlist: all land, or every file keeps the bytes it had. A pose taught
    again is rewritten in place, every comment kept; a joint list written by hand as a block cannot be rewritten so, and
    the loader's refusal comes back with the file untouched.

    Refused with nothing written: a name :func:`pose_name_refusal` refuses (``INVALID_NAME``), a label
    :func:`pose_label_refusal` refuses (``INVALID_LABEL``), and a chain with no layer of the cell's own (``NO_LAYER``,
    :func:`pose_layer_refusal`): none at all, whose target would be the shared base file, or a last layer git does not
    keep out. The note and ``taught_at`` are made one line a later rewrite keeps. A ``screen`` other than
    :data:`WRITTEN_SCREENS` raises ``ValueError``: an unscreened or refused pose is never written, and a caller that
    hands one here skipped the screen, which no operator can mend.
    """
    if screen not in WRITTEN_SCREENS:
        raise ValueError(
            f"a pose screened {screen!r} is never written: only one the exact guard and the planner clear ('clear') or "
            "one in the planner's band ('band') is; screen it first (screen_configuration while the arm holds)")
    prefix = f"robot.named_poses.{name}"
    keys = tuple(f"{prefix}.{part}" for part in ("joints_deg", "label", "taught_at", "screen", "note")) + (
        ("robot.default_place_pose",) if make_default_place else ())
    refused = pose_name_refusal(name)
    if refused:
        return WriteResult(applied=False, keys=keys, refused=WriteRefused.INVALID_NAME, refused_key=prefix,
                           message=refused)
    refused = pose_label_refusal(label)
    if refused:
        return WriteResult(applied=False, keys=keys, refused=WriteRefused.INVALID_LABEL, refused_key=f"{prefix}.label",
                           message=refused)
    refused = pose_layer_refusal(root, layers, what=f"the taught pose {name!r}")
    if refused:
        return _no_layer(keys, refused)
    items: dict[str, Any] = {
        f"{prefix}.joints_deg": [float(value) for value in joints_deg],
        f"{prefix}.label": label.strip(),
        f"{prefix}.taught_at": _one_line(taught_at),
        f"{prefix}.screen": screen,
        f"{prefix}.note": _one_line(note),
    }
    if make_default_place:
        items["robot.default_place_pose"] = name
    return set_keys(items, root=root, layers=layers, profile=profile, allowed=_pose_door)


def set_default_place_pose(
    name: str | None,
    *,
    root: Path,
    layers: tuple[str, ...],
    profile: str | None,
) -> WriteResult:
    """Choose the pose a task places at when its command names no target, or ``None`` for none, in the last layer.

    A name must be a pose name; whether the tree holds that pose is the validators' to say, with the file rolled back
    where it does not (``INVALID_VALUE``). ``None`` writes the loader's reset (``"__null__"``), not ``null``: a ``null``
    in an overlay keeps the value a lower layer set, and the default would stay. A chain with no layer of the cell's
    own is refused (``NO_LAYER``, :func:`pose_layer_refusal`) with nothing written.
    """
    from .loader import _OVERLAY_RESET  # noqa: PLC0415

    key = "robot.default_place_pose"
    if name is not None:
        refused = pose_name_refusal(name)
        if refused:
            return WriteResult(applied=False, keys=(key,), refused=WriteRefused.INVALID_NAME, refused_key=key,
                               message=refused)
    refused = pose_layer_refusal(root, layers, what="the default place pose")
    if refused:
        return _no_layer((key,), refused)
    return set_keys({key: _OVERLAY_RESET if name is None else name}, root=root, layers=layers, profile=profile,
                    allowed=_pose_door)


def _reload(root: Path, profile: str | None) -> Any:
    """Load the tree as a runner would, under ``profile``. Raises on anything the loader rejects."""
    from .loader import load_config, reload_config  # noqa: PLC0415

    # The chain goes to `load_config` as an argument, never through WILLY_PROFILE: the cache is keyed
    # per chain, so two profiles are two entries rather than one entry fought over, and no other
    # holder in the process loses its tree because a write was validated.
    #
    # The `reload_config()` is load-bearing. This runs after bytes have been written to disk, and the
    # cache would otherwise hand back the tree as it was before the edit, so the validating load
    # would validate the old file.
    reload_config()
    return load_config(root, profile=profile)


#: Returned by :func:`read_key` for a path that does not exist. Distinct from ``None``, which is a
#: real value here: an unconfigured ``serial_number`` is ``null``, and reporting that as "missing"
#: would tell an operator their write vanished when it landed exactly as asked.
#:
#: Not :data:`~src.contracts.UNSET`. That sentinel means "the caller did not choose this", a
#: question about an argument; this one means "the tree does not have this path", a question about a
#: lookup.
MISSING = object()


def read_key(cfg: Any, key: str) -> Any:
    """The value the tree holds for ``key``, or :data:`MISSING` if the path does not exist.

    Public together with the sentinel: `explain.py` imports both, and a consumer that cannot reach
    them carries a walker of its own returning ``None`` for both "missing" and "the value is None".
    45 of the default tree's 468 keys hold ``None``, and each of them prints as
    ``camera.stereomatcher.p1`` from such a walker and as
    ``camera.stereomatcher.p1 = None`` from this one: one key, two answers, across the CLI in
    `src/config/__main__.py` and the operator console, the two surfaces `KeyExplanation`
    (`explain.py`) is split out to keep in step. The drift sits one layer above that class, in what
    each surface finds out before calling it.

    Handles ``[index]`` segments.
    """
    node: Any = cfg
    for raw in key.split("."):
        match = _INDEXED.match(raw)
        name = match["head"] if match else raw
        if name:
            node = node.get(name, MISSING) if isinstance(node, dict) else getattr(node, name, MISSING)
            if node is MISSING:
                return MISSING
        if match:
            index = int(match["index"])
            if not isinstance(node, (list, tuple)) or not 0 <= index < len(node):
                return MISSING
            node = node[index]
    return node
