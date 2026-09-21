"""The file, line and profile layer a merged config value came from.

The loader reads up to a dozen YAML files, deep-merges a chain of profile overlays onto each and
hands back plain Python values. That merge keeps the values and drops their origin: ``_deep_merge``
returns ``42`` with no record of which of four files said ``42``. Under a chain such as
``WILLY_PROFILE=sim,ur3e,tiltcam`` the merged values alone cannot say which layer set a key, and a
schema error can name only the data directory. ``loader.py`` calls ``index_origins`` to name the
file, line and layer of each offending key instead.

This module is that record, and it is a side-car: a second, read-only walk over the same files the
loader walks, storing a line mark per leaf. It never takes part in the merge, so a bug here cannot
change a loaded value; the worst it can do is fail to explain one.

Stdlib only (``yaml.compose()`` yields ``start_mark.line`` for every node, so no ``ruamel.yaml``
dependency at the bottom of the dependency stack), and it imports nothing from the robot runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "Origin",
    "index_origins",
    "nearest_keys",
    "section_sources",
]


@dataclass(frozen=True, slots=True)
class Origin:
    """Where one leaf's winning value was written."""

    file: Path
    line: int          #: 1-based, ready to paste after a colon
    layer: str         #: "" for the base YAML, else the profile layer name
    key: str           #: the leaf's own name (not the dotted path)

    def location(self, relative_to: Path | None = None) -> str:
        """``path:line  [layer: x]``, made relative to a given root so the message stays readable."""
        path: Path | str = self.file
        if relative_to is not None:
            try:
                path = self.file.relative_to(relative_to)
            except ValueError:
                path = self.file
        suffix = f"  [layer: {self.layer}]" if self.layer else ""
        return f"{path}:{self.line}{suffix}"


def section_sources(root: Path) -> list[tuple[Path, str, str]]:
    """``(base_file, top_level_yaml_key, dotted AppConfig prefix)`` for every section the loader reads.

    Keep this table in step with ``loader._load_cached``: a file the loader does not read cannot
    explain anything, and a file it reads that is missing here leaves a key unexplained. One table
    so the two can be compared by eye.
    """
    return [
        (root / "camera" / "cam.yaml", "cameras", "camera.cameras"),
        (root / "camera" / "stereomatcher.yaml", "stereomatcher", "camera.stereomatcher"),
        (root / "camera" / "hand_eye.yaml", "hand_eye", "camera.hand_eye"),
        (root / "robot" / "robot.yaml", "robot", "robot"),
        (root / "app" / "runtime.yaml", "runtime", "runtime"),
    ]


def index_origins(root: Path, layers: tuple[str, ...] = ()) -> dict[str, Origin]:
    """Map each dotted config key to the file/line/layer whose value wins.

    The last entry of each :func:`index_chains` chain. Files that do not exist are skipped silently:
    an absent overlay is the normal case, not an error.
    """
    return {key: chain[-1] for key, chain in index_chains(root, layers).items()}


def index_chains(root: Path, layers: tuple[str, ...] = ()) -> dict[str, list[Origin]]:
    """Every write of every key the merged tree holds, in loader order, not just the one that won.

    The winner alone answers where a key is set; the chain also shows what a layer overrode and
    whether it was reached at all. The files are merged as the loader merges them (see
    :func:`_merge`), so a key the merge dropped has no chain, and the last entry is always the winner.
    """
    chains: dict[str, list[Origin]] = {}
    for base, top_key, prefix in section_sources(root):
        tree = _merged_files(base, [(base.with_name(f"{base.stem}.{layer}{base.suffix}"), layer) for layer in layers])
        section = tree.get(top_key) if isinstance(tree, dict) else None
        if section is None:
            continue
        if not isinstance(section.value, dict):
            # A layer that replaced the whole section (`stereomatcher: "__null__"`, `cameras: []`) wrote
            # the one line that explains it: nothing is left inside it to name.
            chains[prefix] = list(section.origins)
        _flatten(section.value, prefix, chains)
    # models/*.yaml contribute their top-level keys directly under `models`, each base file with its
    # overlays, and a profile may add a file with no base counterpart (the loader allows that), which
    # is folded across the layers by stem, as `loader._load_models_section` does.
    models_dir = root / "models"
    if models_dir.is_dir():
        bases = sorted(path for path in models_dir.glob("*.yaml") if "." not in path.stem)
        for base in bases:
            overlays = [(base.with_name(f"{base.stem}.{layer}{base.suffix}"), layer) for layer in layers]
            _flatten(_merged_files(base, overlays), "models", chains)
        base_stems = {path.stem for path in bases}
        profile_only: dict[str, list[tuple[Path, str]]] = {}
        for layer in layers:
            for path in sorted(models_dir.glob(f"*.{layer}.yaml")):
                stem = path.stem[: -len(f".{layer}")]
                if stem not in base_stems:
                    profile_only.setdefault(stem, []).append((path, layer))
        for files in profile_only.values():
            _flatten(_merged_files(None, files), "models", chains)
    return chains


@dataclass(slots=True)
class _Key:
    """One key of a parsed file: every line that wrote it, in merge order, and what it holds now.

    ``value`` is a ``dict`` of ``_Key`` for a mapping, a ``list`` of values for a sequence, ``None``
    for a YAML null, and :data:`_SCALAR` for any other scalar.
    """

    origins: list[Origin]
    value: Any


_SCALAR = "<scalar>"  # any non-null scalar: the index needs only that it replaces what was there


def _merged_files(base: Path | None, overlays: list[tuple[Path, str]]) -> Any:
    """``base`` with the ``(path, layer)`` overlays that exist, merged as ``loader._load_yaml_with_profile`` does.

    The overlays are folded together first and the result is merged onto the base, not applied one
    after the other: ``_deep_merge`` does not associate, and the two orders keep different keys when
    one layer replaces a block that a later layer writes as a mapping again.
    """
    folded: Any = None
    for path, layer in overlays:
        if not path.exists():
            continue
        tree = _parsed(path, layer)
        folded = tree if folded is None else _merge(folded, tree)
    tree = _parsed(base, "") if base is not None else None
    if tree is None:
        return folded
    return _merge(tree, folded)


def _parsed(path: Path, layer: str) -> Any:
    """``path`` as a tree of :class:`_Key`, or ``None`` for a file that is empty or does not parse.

    This runs while the loader is already reporting an error, so a failure here must not raise a
    second one.
    """
    try:
        node = yaml.compose(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 (explaining must never become a second failure)
        return None
    return None if node is None else _value(node, path, layer)


def _value(node: Any, path: Path, layer: str) -> Any:
    if isinstance(node, yaml.MappingNode):
        out: dict[str, _Key] = {}
        for k, v in node.value:
            name = str(getattr(k, "value", ""))
            # Container keys get a line too: an `extra_forbidden` on a nested block points at the block
            # instead of vanishing because the block is not a leaf. A key written twice keeps the last
            # write, as `yaml.safe_load` does.
            out[name] = _Key([Origin(path, int(k.start_mark.line) + 1, layer, name)], _value(v, path, layer))
        return out
    if isinstance(node, yaml.SequenceNode):
        return [_value(item, path, layer) for item in node.value]
    return None if _is_null(node) else _SCALAR


def _merge(base: Any, overlay: Any) -> Any:
    """``_merge._deep_merge`` over parsed trees, so the index holds the keys the loaded tree holds.

    The index answers "which line wrote the value the tree holds", so it has to merge the way the
    loader merges, or it answers for a value the tree does not hold. Each rule was measured by a
    review on 2026-09-21 against a tree where the index got it wrong:

    * A null under a key the merge already holds keeps the earlier value, and the null's line is not
      a write of it. `robot.gripper.max_width_mm:` with nothing after it in a layer left 70.0 from
      robot.yaml in the tree while the index named the empty line. Under a key the merge does not
      hold yet, the null is the value (`robot.sim.assets_root: null` in robot.sim.yaml), and so is
      every null inside a block that replaced what was there.
    * Anything but a mapping over a mapping, and anything over a list or a scalar, replaces it whole
      (a list, a scalar, the `__null__` reset), so every key written inside the earlier value is gone.
      A layer that restated `camera.cameras.rigs` with one RealSense left the base webcam rig's keys
      indexed, and a key the layer's rig lacked could be traced to a rig that no longer exists.
    * A mapping over a mapping merges key by key.
    """
    if overlay is None:
        return base
    if isinstance(base, dict) and isinstance(overlay, dict):
        out = dict(base)
        for name, key in overlay.items():
            earlier = out.get(name)
            if earlier is None:
                out[name] = key
            elif key.value is not None:
                out[name] = _Key(earlier.origins + key.origins, _merge(earlier.value, key.value))
        return out
    return overlay


def _flatten(value: Any, prefix: str, chains: dict[str, list[Origin]]) -> None:
    """Every key inside ``value`` under its dotted path; a list item is ``prefix[i]``, as the files count it."""
    if isinstance(value, dict):
        for name, key in value.items():
            dotted = f"{prefix}.{name}" if prefix else name
            chains[dotted] = list(key.origins)
            _flatten(key.value, dotted, chains)
    elif isinstance(value, list):
        for i, item in enumerate(value):
            _flatten(item, f"{prefix}[{i}]", chains)


def _is_null(node: Any) -> bool:
    """A YAML null (``key:``, ``key: null``, ``key: ~``). The ``__null__`` reset is a string, and replaces."""
    return isinstance(node, yaml.ScalarNode) and node.tag == "tag:yaml.org,2002:null"


def comment_above(origin: Origin) -> str:
    """The contiguous ``#`` block written immediately above ``origin``'s line, dedented.

    A YAML comment often carries the measurement behind a value ("measured by sweeping the real
    planner: 6/6 up to 6 mm and 0/6 at 10 mm"), and ``yaml.safe_load`` discards every one of them.
    Reading the block at query time surfaces that evidence without copying it anywhere: the comment
    stays in the file its author wrote it in.
    """
    try:
        lines = origin.file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    out: list[str] = []
    i = origin.line - 2  # the line above the key (origin.line is 1-based)
    while i >= 0:
        stripped = lines[i].strip()
        if not stripped.startswith("#"):
            break
        out.append(stripped.lstrip("#").strip())
        i -= 1
    return "\n".join(reversed(out))


def read_value(origin: Origin) -> str:
    """The raw text written after the key on ``origin``'s line (``""`` for a block header)."""
    try:
        line = origin.file.read_text(encoding="utf-8").splitlines()[origin.line - 1]
    except (OSError, IndexError):
        return ""
    _, _, rest = line.partition(":")
    return rest.split("#")[0].strip()


def nearest_keys(name: str, candidates: list[str], *, limit: int = 3) -> list[str]:
    """Closest ``candidates`` to ``name`` for a did-you-mean hint (stdlib difflib, no dependency).

    ``extra='forbid'`` turns every typo into a hard stop, and the most common typo is a near-miss of
    a real field, where only the spelling is missing.
    """
    import difflib

    return difflib.get_close_matches(name, candidates, n=limit, cutoff=0.6)
