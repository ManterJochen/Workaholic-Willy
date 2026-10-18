"""A cell of several arms: one small file names each arm and the config tree it runs on (owner, 2026-10-15).

```yaml
# <tree>/cells/clip.yaml
arms:
  halter: ur10,robotiq,zelle,halter                  # a chain in the tree that holds this file
  werker: {tree: D:/zellen/werker, profile: ur10}    # or a tree of its own, anywhere
```

A tree with one arm needs no such file: a tree is a cell of one arm, as it always was, and ``WILLY_PROFILE`` stays
the way to name it. A cell of several arms is named with ``WILLY_CELL`` (or ``--cell``): a path to the file, which
may lie anywhere (most users keep their trees outside the repository), or a bare name, read as
``<tree>/cells/<name>.yaml``.

Each arm keeps its own tree, so different arms bring their own blocks. What they share is written once, in a layer
both chains name. The first arm listed stands where the cell's frame is: a position any tree of the cell states is in
that arm's base, as a one-arm tree's always was. Where every other arm stands is measured, never written.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.contracts import UNSET, Maybe, chosen

from .tree import LoadedTree, default_data_dir, load_tree

__all__ = ["CELL_ENV_VAR", "CellArm", "LoadedCell", "active_cell", "cell_file", "load_cell"]

#: The variable that names a cell of several arms, as ``WILLY_PROFILE`` names a tree.
CELL_ENV_VAR = "WILLY_CELL"
#: The folder of a tree a bare cell name is read from.
CELLS_DIR = "cells"
#: An arm's name: it stands in logs, records, file names and the sentences a person says.
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,31}$")


def active_cell() -> str | None:
    """The cell ``WILLY_CELL`` names, or ``None`` where it names none.

    Returns:
        str | None: The variable's value, stripped; ``None`` where it is unset or empty.
    """
    value = os.environ.get(CELL_ENV_VAR, "").strip()
    return value or None


def cell_file(cell: str | Path, *, root: "Maybe[str | Path | None]" = UNSET) -> Path:
    """The file a cell is read from: ``cell`` itself where it is a path, else ``<root>/cells/<cell>.yaml``.

    Args:
        cell (str | Path): A path to the file, or a bare name.
        root (Maybe[str | Path | None]): The tree a bare name is looked up in; unset or ``None`` is the repository's
            ``config/`` (default: UNSET).

    Returns:
        Path: The file, resolved; it may not exist.
    """
    text = str(cell)
    if isinstance(cell, Path) or any(sep in text for sep in ("/", "\\")) or text.endswith((".yaml", ".yml")):
        return Path(text).resolve()
    base = Path(root).resolve() if chosen(root) and root is not None else default_data_dir()
    return base / CELLS_DIR / f"{text}.yaml"


@dataclass(frozen=True, slots=True)
class CellArm:
    """One arm of a cell: its name and its own loaded tree."""

    #: The arm's name as the cell file writes it.
    name: str
    #: The tree the arm runs on, loaded and validated.
    tree: LoadedTree


@dataclass(frozen=True, slots=True)
class LoadedCell:
    """A cell file read, and every arm's tree loaded. What :func:`load_cell` returns.

    A cell that does not load is still returned, with ``ok`` false and its refusal, as a tree is.

    Attributes:
        path (Path): The cell file.
        arms (tuple[CellArm, ...]): The arms in the file's order; the first stands at the cell's frame (default: ()).
        error (str): Why the cell does not load; empty where it does (default: "").
    """

    path: Path
    arms: tuple[CellArm, ...] = ()
    error: str = ""

    @property
    def ok(self) -> bool:
        """Whether the file read and every arm's tree loaded."""
        return not self.error

    @property
    def exit_code(self) -> int:
        """0 where the cell loaded, 1 where it did not."""
        return 0 if self.ok else 1

    @property
    def names(self) -> tuple[str, ...]:
        """The arms' names in the file's order."""
        return tuple(arm.name for arm in self.arms)

    @property
    def frame_arm(self) -> CellArm:
        """The first arm: the cell's frame is its base."""
        return self.arms[0]

    def arm(self, name: str) -> CellArm:
        """The arm called ``name``, compared without case.

        Args:
            name (str): The arm's name.

        Returns:
            CellArm: That arm.

        Raises:
            ConfigError: No arm of this cell is called so; the sentence lists the names it has.
        """
        for arm in self.arms:
            if arm.name.casefold() == name.casefold():
                return arm
        from .loader import ConfigError  # noqa: PLC0415

        raise ConfigError(f"the cell {self.path} has no arm {name!r}: its arms are {', '.join(self.names) or 'none'}")

    def where_written(self, arm: str, block: str) -> str:
        """In which frame ``arm``'s tree states ``block``: ``"cell"`` where every file that sets a key of it is one the
        first arm's tree reads too (a layer both share, or the base files), so its positions stand in the cell's frame;
        ``"own"`` where every such file is the arm's own, in its own base; ``""`` where no file sets any (its defaults).

        Args:
            arm (str): The arm's name.
            block (str): A dotted key, such as ``"robot.safety.self_collision.fixtures"``; every key under it counts.

        Returns:
            str: ``"cell"``, ``"own"`` or ``""``.

        Raises:
            ConfigError: Some keys of the block are set in a shared file and some in the arm's own, so its positions
                would stand in two frames.
        """
        from ._provenance import index_chains  # noqa: PLC0415
        from .loader import ConfigError  # noqa: PLC0415

        tree = self.arm(arm).tree
        frame = self.frame_arm.tree
        shared = {origin.file.resolve() for chain in index_chains(frame.root, frame.layers).values()
                  for origin in chain}
        files = {chain[-1].file.resolve() for key, chain in index_chains(tree.root, tree.layers).items()
                 if chain and (key == block or key.startswith(block + ".") or key.startswith(block + "["))}
        if not files:
            return ""
        own = files - shared
        if own and own != files:
            raise ConfigError(f"the arm {arm!r} sets {block} partly in a file the first arm reads too, so in the cell's "
                              f"frame, and partly in its own ({', '.join(sorted(map(str, own)))}), in its own base: "
                              "write the block in one of the two")
        return "own" if own else "cell"

    def render(self) -> str:
        """The verdict, then one line per arm: its name and its tree's chain and folder.

        Returns:
            str: ASCII lines, no trailing newline.
        """
        if not self.ok:
            return f"cell {self.path}  REFUSED: {self.error}"
        lines = [f"cell {self.path}  {len(self.arms)} arm(s), the frame stands at {self.frame_arm.name}"]
        lines += [f"  {arm.name:<12} {arm.tree.chain}  ({arm.tree.root})" for arm in self.arms]
        return "\n".join(lines)

    def __str__(self) -> str:
        """What ``print()`` shows: the text :meth:`render` returns."""
        return self.render()

    def to_dict(self) -> dict[str, Any]:
        """Plain data, ``json.dumps`` safe.

        Returns:
            dict[str, Any]: The file, the verdict and each arm's name, chain and folder.
        """
        return {"path": str(self.path), "ok": self.ok, "error": self.error,
                "arms": [{"name": arm.name, "profile": arm.tree.profile, "root": str(arm.tree.root)}
                         for arm in self.arms]}


def _arm_entries(path: Path) -> list[tuple[str, Path | None, str | None]]:
    """Each arm the file names: its name, the folder of its own tree (``None`` for the cell's tree) and its chain."""
    from .loader import ConfigError, _load_yaml  # noqa: PLC0415

    data = _load_yaml(path)
    if not isinstance(data, dict) or set(data) != {"arms"}:
        raise ConfigError(f"{path} must hold one key, 'arms:', naming each arm and its profile chain")
    arms = data["arms"]
    if not isinstance(arms, dict) or not arms:
        raise ConfigError(f"{path}: 'arms:' names no arm; write 'name: chain' for each arm of the cell")
    entries: list[tuple[str, Path | None, str | None]] = []
    seen: dict[str, str] = {}
    for name, value in arms.items():
        name = str(name)
        if not _NAME.match(name):
            raise ConfigError(f"{path}: the arm name {name!r} is not a name: a letter, then up to 31 letters, digits, "
                              "'_' or '-'")
        if name.casefold() in seen:
            raise ConfigError(f"{path}: the arms {seen[name.casefold()]!r} and {name!r} differ only in case")
        seen[name.casefold()] = name
        if isinstance(value, str) and value.strip():
            entries.append((name, None, value.strip()))
        elif isinstance(value, dict) and "tree" in value and set(value) <= {"tree", "profile"}:
            folder = Path(str(value["tree"]))
            folder = folder if folder.is_absolute() else (path.parent / folder)
            profile = value.get("profile")
            entries.append((name, folder.resolve(), str(profile).strip() if profile else None))
        else:
            raise ConfigError(f"{path}: the arm {name!r} takes a profile chain ('{name}: ur10,cell') or its own tree "
                              f"('{name}: {{tree: <folder>, profile: <chain>}}')")
    return entries


def load_cell(cell: "Maybe[str | Path]" = UNSET, *, root: "Maybe[str | Path | None]" = UNSET) -> LoadedCell:
    """Read a cell file and load every arm's tree. Connects, opens and moves nothing.

    ```python
    cell = load_cell("clip")                         # <the repository's config>/cells/clip.yaml
    cell = load_cell("D:/zellen/clip/cells/clip.yaml")
    print(cell)                                      # the verdict and one line per arm
    tree = cell.arm("werker").tree                   # that arm's LoadedTree
    ```

    Args:
        cell (Maybe[str | Path]): A path to the file or a bare name; unset is the cell ``WILLY_CELL`` names
            (default: UNSET).
        root (Maybe[str | Path | None]): The tree a bare name and a plain chain are read in; unset is the tree that
            holds the file (its ``cells/`` folder's parent) or else the repository's ``config/`` (default: UNSET).

    Returns:
        LoadedCell: The cell. One that does not load comes back with ``ok`` false and its refusal; it does not raise.

    Raises:
        ConfigError: ``cell`` is unset and ``WILLY_CELL`` names no cell.
    """
    from .loader import ConfigError  # noqa: PLC0415

    named: str | Path | None = cell if isinstance(cell, (str, Path)) else active_cell()
    if named is None:
        raise ConfigError(f"no cell named: pass one, or set {CELL_ENV_VAR} to a cell file or a name in "
                          f"<tree>/{CELLS_DIR}/")
    path = cell_file(named, root=root)
    if not path.is_file():
        return LoadedCell(path=path, error=f"no cell file at {path}")
    tree_root: Path | None
    if chosen(root) and root is not None:
        tree_root = Path(root).resolve()
    elif path.parent.name == CELLS_DIR:
        tree_root = path.parent.parent
    else:
        tree_root = None
    try:
        entries = _arm_entries(path)
    except ConfigError as exc:
        return LoadedCell(path=path, error=str(exc))
    arms: list[CellArm] = []
    controllers: dict[tuple[str, str], str] = {}
    for name, folder, profile in entries:
        tree = load_tree(profile, root=folder if folder is not None else (tree_root if tree_root else UNSET))
        if not tree.ok:
            return LoadedCell(path=path, arms=tuple(arms), error=f"the arm {name!r} ({tree.chain}): {tree.error}")
        try:
            selection = tree.robot.arm
        except ConfigError as exc:
            return LoadedCell(path=path, arms=tuple(arms), error=f"the arm {name!r} ({tree.chain}): {exc}")
        # Two arms behind one controller would share its one control program: the second connect takes it from the
        # first. A simulated or desk arm drives no controller and shares nothing.
        if selection.owns_a_controller and selection.controller:
            key = (str(tree.robot.vendor), str(selection.controller))
            if key in controllers:
                return LoadedCell(path=path, arms=tuple(arms), error=(
                    f"the arms {controllers[key]!r} and {name!r} both drive {key[0]}@{key[1]}: each arm of a cell "
                    "needs its own controller"))
            controllers[key] = name
        arms.append(CellArm(name=name, tree=tree))
    return LoadedCell(path=path, arms=tuple(arms))
