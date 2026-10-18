"""Bringing a config tree written for an older Willy to the keys this one reads, every comment kept.

A key that left the schema on purpose is refused at load with the sentence that says what replaced it
(:mod:`src.config.schema._removed`). This rewrites a tree's own files to match, so a person does not have to find
each line by hand: a key that left is deleted together with the comment written directly above it, a key that moved
is moved under its new parent (into the other section's file where its section changed), a value whose word changed
is rewritten, a home in radians written on one line is written in degrees under its new key, and two blocks of one key written side by side, which the loader refuses since 2026-10-16, become one.
A file whose whole section left (``camera/stereomatcher.yaml`` since 2026-10-17) is deleted with its layers, and so is a
models file left with no key (``models/hand.yaml``, whose two blocks left the same day). Every change is said with the
sentence of its key, at its file and line.

Nothing round-trips YAML, for the reason :mod:`.edit` gives: a ``safe_load``/``dump`` cycle deletes every comment, and
in a cell's tree the comments are most of what it knows. Lines are deleted, moved or rewritten whole and every other
byte is copied through.

``python -m src.config migrate PATH`` says what it would change and changes nothing; ``--write`` changes the files,
loads the tree through the real loader and puts every file back as it was if the result does not load.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from ._provenance import section_sources
from .edit import _block_end, _indent_of
from .schema._removed import removed_key_sentence

__all__ = ["CONVERTED", "MOVED", "REWORDED", "Change", "Migration", "migrate_tree"]

#: A key that moved, by its old dotted name, to its new dotted name. Each is refused at load naming its new key
#: (``schema/_removed.py``); this moves its whole block, comments included, under the new parent, and where the new
#: name starts with another section, into that section's file of the same profile layer.
MOVED: Final[dict[str, str]] = {
    "robot.ur.motion_planner": "robot.motion.planner",
    "robot.grasping.recovery.critical_parts": "runtime.task.critical_parts",
    # Every key of how a cell moves, under robot.motion (2026-10-17). The limits' block leaves (its sentence) and its
    # two keys move out of it, one level up.
    "robot.motion_limits.max_velocity": "robot.motion.max_velocity",
    "robot.motion_limits.max_acceleration": "robot.motion.max_acceleration",
    "robot.workspace_limits": "robot.motion.workspace",
    "robot.home_joint_positions_deg": "robot.motion.home_deg",
    "robot.look_joint_positions_deg": "robot.motion.look_deg",
    "robot.natural_closing_axis": "robot.motion.natural_closing_axis",
    "robot.named_poses": "robot.motion.named_poses",
    "robot.default_place_pose": "robot.motion.default_place_pose",
    "robot.place": "robot.motion.place",
    # How far the parts hang past the fingertips: the carried part's spheres left, its length stayed (2026-10-17).
    "robot.safety.planning_world.payload.length_mm": "robot.motion.place.part_hang_mm",
}


def _degrees_of(value: str) -> str | None:
    """A list of radians written on one line, ``[0.0, -1.5708]``, as degrees to a ten-thousandth; ``None`` where the
    value is no such list (a list written over several lines, a value the loader resets)."""
    match = re.fullmatch(r"\[(?P<items>[^\]\[]*)\]", value.strip())
    if match is None:
        return None
    try:
        radians = [float(item) for item in match["items"].split(",") if item.strip()]
    except ValueError:
        return None
    return "[" + ", ".join(str(round(math.degrees(r), 4)) for r in radians) + "]"


#: A key that moved and changed its unit, by its old dotted name: its new dotted name and what writes its value anew
#: (``None`` where the value cannot be read, and the key is left for a person). The home in radians is written in
#: degrees alone since 2026-10-17.
CONVERTED: Final[dict[str, tuple[str, Callable[[str], str | None]]]] = {
    "robot.home_joint_positions": ("robot.motion.home_deg", _degrees_of),
}

#: A value whose word changed, by (dotted key, old word): the new word. The line is rewritten in place.
REWORDED: Final[dict[tuple[str, str], str]] = {
    ("robot.gripper.tool_frame.source", "polyscope"): "controller",
}

#: How many passes the migration makes at most: a block merged in one pass is walked in the next, and no tree nests
#: blocks written twice deeper than this.
_MAX_PASSES: Final = 6

_KEY_LINE: Final = re.compile(r"^(?P<indent> *)(?P<key>[A-Za-z_][\w-]*):(?P<rest>(?:\s.*)?)$")
#: A value that opens a block scalar (``>-``, ``|``): the lines under it are text, not keys.
_BLOCK_SCALAR: Final = re.compile(r"^\s*[>|][+-]?\d*\s*(?:#.*)?$")
#: The columns one level of a tree's YAML is indented by.
_UNIT: Final = 2


@dataclass(frozen=True, slots=True)
class Change:
    """One change to one file: what it does, where, and the sentence that says why."""

    file: Path
    line: int  #: 1-based, in the file as the pass read it
    key: str
    action: str  #: "delete", "move", "convert", "keep", "reword" or "merge"
    said: str

    def render(self, relative_to: Path | None = None) -> str:
        """The change as one line for a person: file and line, what happens to which key, and why."""
        where = self.file
        if relative_to is not None and self.file.is_relative_to(relative_to):
            where = self.file.relative_to(relative_to)
        return f"{where.as_posix()}:{self.line}  {self.action} {self.key}: {self.said}"


@dataclass(slots=True)
class Migration:
    """What :func:`migrate_tree` found and did: every change, the new text of each file, and whether it was written."""

    root: Path
    changes: list[Change] = field(default_factory=list)
    #: Each file's new text; ``None`` for a file the migration deletes, its whole section having left.
    texts: dict[Path, str | None] = field(default_factory=dict)
    #: Each file's bytes as they were, ``None`` for a file the migration creates: what a refusal puts back.
    originals: dict[Path, bytes | None] = field(default_factory=dict)
    #: Each file's line ending as it was written, kept on every line the migration writes.
    newlines: dict[Path, str] = field(default_factory=dict)
    written: bool = False
    refused: str = ""

    @property
    def changed_files(self) -> list[Path]:
        """The files whose text the migration changes, created ones included."""
        return sorted(path for path, text in self.texts.items()
                      if (text is None and self.originals.get(path) is not None)
                      or (text is not None and text.encode("utf-8") != _as_lf(self.originals.get(path))))

    def render(self) -> str:
        """Every change, one line each, and what became of them."""
        if not self.changes:
            return f"{self.root.as_posix()}: nothing to change; the tree writes only keys this Willy reads."
        tail = ("written, and the tree loads" if self.written
                else f"NOT written: {self.refused}" if self.refused else "nothing written (add --write)")
        lines = [change.render(self.root) for change in self.changes]
        return "\n".join([*lines, f"{len(self.changes)} change(s) in {len(self.changed_files)} file(s): {tail}"])


@dataclass(frozen=True, slots=True)
class _Block:
    """One mapping key in one file: its dotted name, its own line, the comments directly above it, and where it ends."""

    key: str
    indent: int
    head: int
    lead: int
    end: int


@dataclass(slots=True)
class _File:
    """One file of the tree as the migration reads and rewrites it."""

    path: Path
    prefix: str
    top: str | None
    layer: str
    lines: list[str]
    #: The sentence of a section that left as a whole, whose file goes: ``""`` for every other file.
    removed: str = ""


def _as_lf(raw: bytes | None) -> bytes:
    """``raw`` with every line ending one line feed, the form the migration compares and builds on."""
    return b"" if raw is None else raw.replace(b"\r\n", b"\n")


def _blocks(lines: list[str], prefix: str, top: str | None) -> Iterator[_Block]:
    """Every mapping key of a section file in file order, outermost first, with its dotted name under ``prefix``.

    ``top`` is the file's one top-level key that stands for the section itself (``robot:`` in ``robot.yaml``) and is
    no segment of the dotted name; ``None`` for a file whose top-level keys are the section's own (the models files).
    Keys inside a list item and the lines of a block scalar are not walked: no key that left or moved lives in either.
    """
    stack: list[tuple[int, str]] = []
    opaque_below: int | None = None  # lines indented past this column belong to a list item or a block scalar
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = _indent_of(line)
        if opaque_below is not None:
            if indent > opaque_below:
                continue
            opaque_below = None
        if stripped.startswith("- ") or stripped == "-":
            opaque_below = indent
            continue
        match = _KEY_LINE.match(line)
        if match is None:
            continue
        while stack and stack[-1][0] >= indent:
            stack.pop()
        stack.append((indent, match["key"]))
        if _BLOCK_SCALAR.match(match["rest"] or ""):
            opaque_below = indent
        names = [name for _, name in stack]
        if top is not None:
            if names[0] != top or len(names) == 1:
                continue
            names = names[1:]
        lead = index
        while lead > 0 and lines[lead - 1].strip().startswith("#") and _indent_of(lines[lead - 1]) == indent:
            lead -= 1
        yield _Block(".".join([prefix, *names]), indent, index, lead, _block_end(lines, index, indent))


def _holds_a_key(lines: list[str]) -> bool:
    """Whether ``lines`` write any top-level key: a file that holds only comments and blank lines writes none."""
    return any(_KEY_LINE.match(line) and not line.startswith(" ") for line in lines)


def _value_of(line: str) -> str:
    """The plain value on a ``key: value`` line, its comment and quotes taken off."""
    match = _KEY_LINE.match(line)
    rest = " " + (match["rest"] if match else "")
    return re.sub(r"(?<=\s)#.*$", "", rest).strip().strip("'\"")


def _reindent(block: list[str], by: int) -> list[str]:
    """``block`` moved ``by`` columns right (left where negative), blank lines kept blank."""
    if by >= 0:
        return [(" " * by + line) if line.strip() else line for line in block]
    return [line[-by:] if line.startswith(" " * -by) else line.lstrip(" ") for line in block]


def _tidy(lines: list[str]) -> list[str]:
    """``lines`` with no two blank lines in a row and none at the end, which is what a cut can leave."""
    out: list[str] = []
    for line in lines:
        if not line.strip() and (not out or not out[-1].strip()):
            continue
        out.append(line)
    while out and not out[-1].strip():
        out.pop()
    return out


def _section_files(root: Path) -> list[_File]:
    """Every file of the tree the loader reads a section from: each base file and each of its profile layers
    (``robot.hande.yaml`` is the ``hande`` layer of ``robot.yaml``), the models files with their own top-level keys."""
    sources: list[tuple[Path, str | None, str]] = [(base, top, prefix) for base, top, prefix in section_sources(root)]
    sources += [(path, None, "models") for path in sorted((root / "models").glob("*.yaml")) if path.name.count(".") == 1]
    found: list[_File] = []
    for base, top, prefix in sources:
        for path in [base, *sorted(base.parent.glob(f"{base.stem}.*{base.suffix}"))]:
            if not path.is_file():
                continue
            layer = path.name[len(base.stem) + 1:-len(base.suffix)] if path != base else ""
            found.append(_File(path, prefix, top, layer, path.read_text(encoding="utf-8").splitlines(),
                               removed=removed_key_sentence(prefix) or ""))
    return found


def _plan(file: _File, said: list[Change], moving: list[tuple[str, list[str], int, _File]]) -> list[str]:
    """One pass over ``file``: each key that left cut out, each value reworded, each block written twice merged into
    the first, and each block that moves cut out and handed back in ``moving`` (its new key, its lines, its indent, the
    file it came from).

    A pass changes the outermost blocks only; :func:`migrate_tree` repeats passes until one changes nothing, so a key
    that left inside a block merged in one pass is cut in the next.
    """
    lines = file.lines
    if file.removed:
        if lines:
            said.append(Change(file.path, 1, file.prefix, "delete", f"the whole file: {file.removed}"))
        return []
    blocks = list(_blocks(lines, file.prefix, file.top))
    cut: set[int] = set()
    before: dict[int, list[str]] = {}
    replace: dict[int, str] = {}
    seen: dict[str, _Block] = {}
    handled: list[_Block] = []

    def take(block: _Block) -> None:
        cut.update(range(block.lead, block.end))
        handled.append(block)

    for block in blocks:
        if any(outer.head < block.head < outer.end for outer in handled):
            continue
        line = block.head + 1
        if block.key in MOVED:
            moving.append((MOVED[block.key], lines[block.lead:block.end], block.indent, file))
            said.append(Change(file.path, line, block.key, "move", f"now {MOVED[block.key]}"))
            take(block)
            continue
        if block.key in CONVERTED:
            new_key, convert = CONVERTED[block.key]
            head = lines[block.head]
            alone = block.end == block.head + 1
            if alone and _value_of(head) in ("null", "~", ""):
                # A key that says nothing (no home given): there is nothing to convert, and it goes.
                said.append(Change(file.path, line, block.key, "delete", removed_key_sentence(block.key) or ""))
                take(block)
                continue
            value = convert(_value_of(head)) if alone else None
            if value is None:
                # Left as it is, said once: the loader then refuses the tree, naming the key, and a person converts it.
                if not any(c.action == "keep" and c.file == file.path and c.key == block.key for c in said):
                    said.append(Change(file.path, line, block.key, "keep", (
                        f"not converted, its value is no list on one line: {removed_key_sentence(block.key)}")))
                seen[block.key] = block
                continue
            comment = re.search(r"\s#.*$", head)
            written = f"{head[:head.index(':') + 1]} {value}{comment.group(0) if comment else ''}"
            moving.append((new_key, [*lines[block.lead:block.head], written], block.indent, file))
            said.append(Change(file.path, line, block.key, "convert", f"now {new_key}: {value}"))
            take(block)
            continue
        sentence = removed_key_sentence(block.key)
        if sentence is not None:
            for inner in blocks:
                if block.head < inner.head < block.end and inner.key in MOVED:
                    moving.append((MOVED[inner.key], lines[inner.lead:inner.end], inner.indent, file))
                    said.append(Change(file.path, inner.head + 1, inner.key, "move", f"now {MOVED[inner.key]}"))
            said.append(Change(file.path, line, block.key, "delete", sentence))
            take(block)
            continue
        twin = seen.get(block.key)
        if twin is not None:
            children = lines[block.head + 1:block.end]
            if any(child.strip() and not child.strip().startswith("#") for child in children):
                before.setdefault(twin.end, []).extend(_reindent(children, twin.indent - block.indent))
                said.append(Change(file.path, line, block.key, "merge", (
                    f"written twice; its keys join the block at line {twin.head + 1}, and where both write a key the "
                    "later one counts")))
            else:
                replace[twin.head] = " " * twin.indent + lines[block.head].lstrip(" ")
                said.append(Change(file.path, line, block.key, "merge",
                                   f"written twice; the later value counts, at line {twin.head + 1}"))
            take(block)
            continue
        seen[block.key] = block
        old = _value_of(lines[block.head])
        word = REWORDED.get((block.key, old))
        if word is not None:
            replace[block.head] = lines[block.head].replace(old, word, 1)
            said.append(Change(file.path, line, block.key, "reword", f"{old!r} is now {word!r}"))
    if not cut and not replace and not before:
        # Nothing to change in this file: it stays as it is, byte for byte, its blank lines too.
        return lines
    out: list[str] = []
    for index, text in enumerate(lines):
        out.extend(before.get(index, ()))
        if index not in cut:
            out.append(replace.get(index, text))
    out.extend(before.get(len(lines), ()))
    return _tidy(out)


def _insert(file: _File, key: str, block: list[str], indent: int) -> None:
    """Put ``block`` (indented at ``indent`` where it came from) into ``file`` as the key ``key``, at the end of its
    parent, with every parent made where the file has none; the block's key line takes the new key's last name."""
    names = key[len(file.prefix) + 1:].split(".")
    lines = file.lines
    lo, hi, depth = 0, len(lines), 0
    if file.top is not None:
        head = next((i for i, line in enumerate(lines) if re.match(rf"^{re.escape(file.top)}:(\s|$)", line)), None)
        if head is None:
            lines.append(f"{file.top}:")
            head = len(lines) - 1
        lo, hi, depth = head + 1, _block_end(lines, head, 0), _UNIT
    for name in names[:-1]:
        found = next((i for i in range(lo, min(hi, len(lines)))
                      if re.match(rf"^ {{{depth}}}{re.escape(name)}:(\s|$)", lines[i])), None)
        if found is None:
            lines[hi:hi] = [" " * depth + f"{name}:"]
            found = hi
        lo, hi = found + 1, _block_end(lines, found, depth)
        depth += _UNIT
    moved = _reindent(block, depth - indent)
    for i, text in enumerate(moved):
        match = _KEY_LINE.match(text)
        if match is not None and len(match["indent"]) == depth:
            moved[i] = f"{match['indent']}{names[-1]}:{match['rest']}"
            break
    lines[hi:hi] = moved


def _target(folder: Path, files: list[_File], key: str, layer: str, result: Migration) -> _File:
    """The file a moved key lands in: the section its new name starts with, in the same profile layer, created where
    the tree has no such file yet."""
    for file in files:
        if file.layer == layer and key.startswith(file.prefix + "."):
            return file
    base, top, prefix = next(entry for entry in section_sources(folder) if key.startswith(entry[2] + "."))
    path = base if not layer else base.with_name(f"{base.stem}.{layer}{base.suffix}")
    created = _File(path, prefix, top, layer, [])
    result.originals[path] = None
    files.append(created)
    return created


def migrate_tree(root: str | Path, *, write: bool = False) -> Migration:
    """What it takes to bring the tree at ``root`` to the keys this Willy reads, and with ``write`` the files changed.

    Args:
        root (str | Path): The config tree's folder, the one holding ``robot/``, ``camera/``, ``models/`` and ``app/``.
        write (bool): Change the files, then load the tree under its base and every profile layer it holds; any
            refusal puts every file back as it was and is said in :attr:`Migration.refused` (default: False).

    Returns:
        Migration: Every change with its file, line and sentence, and each file's new text.
    """
    folder = Path(root).resolve()
    result = Migration(folder)
    files = _section_files(folder)
    for file in files:
        raw = file.path.read_bytes()
        result.originals[file.path] = raw
        result.newlines[file.path] = "\r\n" if b"\r\n" in raw else "\n"
    for _ in range(_MAX_PASSES):
        count = len(result.changes)
        moving: list[tuple[str, list[str], int, _File]] = []
        for file in list(files):
            file.lines = _plan(file, result.changes, moving)
        for key, block, indent, source in moving:
            _insert(_target(folder, files, key, source.layer, result), key, block, indent)
        if len(result.changes) == count:
            break
    for file in files:
        if file.top is None and not file.removed and file.path.is_file() and not _holds_a_key(file.lines):
            # A models file is a mapping of its own top-level keys, and the loader refuses one that holds none.
            file.removed = "every key it held left"
            result.changes.append(Change(file.path, 1, file.prefix, "delete", f"the whole file: {file.removed}"))
        result.texts[file.path] = None if file.removed else "\n".join(file.lines) + "\n" if file.lines else ""
    if write and result.changed_files:
        _write_and_load(result)
    return result


def _write_and_load(result: Migration) -> None:
    """Write the changed files and load the tree under its base and every layer it holds; a refusal puts every file
    back as it was."""
    from .loader import ConfigError, load_config, reload_config  # noqa: PLC0415

    for path in result.changed_files:
        text = result.texts[path]
        if text is None:
            path.unlink()
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline=result.newlines.get(path, "\n"))
    layers = sorted({path.name.split(".")[1] for path in result.texts if path.name.count(".") >= 2})
    try:
        for profile in [None, *layers]:
            reload_config()
            load_config(result.root, profile=profile)
    except ConfigError as exc:
        for path in result.changed_files:
            original = result.originals.get(path)
            if original is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(original)
        reload_config()
        result.refused = f"the migrated tree does not load, so every file is as it was: {exc}"
        return
    result.written = True
