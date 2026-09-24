"""Where a path in a config tree points: the rule, and the one place it is applied.

THE RULE
--------
* **A relative path in a tree is read against the tree's folder**: the directory the tree was loaded
  from, the one that holds ``robot/``, ``camera/`` and ``models/``. That is ``config/`` in this
  repository and ``D:/cells/line3`` for ``load_tree(root="D:/cells/line3")``. Not the working
  directory the program was started in, and not the subfolder of the file that writes the value:
  ``calibration/eih_wrist.json`` in ``D:/cells/line3/camera/cam.yaml`` is
  ``D:/cells/line3/calibration/eih_wrist.json``.
* **``${WILLY_PROJECT_ROOT}`` at the start of a path is the repository** (or the directory the
  ``WILLY_PROJECT_ROOT`` environment variable names, the same override
  :func:`src.utility.paths.project_root` honours). It is how a tree names what the repository holds
  rather than the cell: model weights under ``assets/models/hf``, the committed success model and
  ranker, the RL baselines under ``docs/baselines``. The shipped tree writes those with it, so a copy
  of ``config/`` moved out of the repository still finds them. **Only a path field reads it**: any
  other key that holds it is refused by :class:`~src.config.schema._base.StrictModel`, naming the key,
  because the loader leaves the anchor in the text for this rule and nothing else would expand it.
* **An absolute path is read as written**, and ``~`` at its start is the user's home folder.
* **An empty value stays empty**: it is how an optional path says "off" (``record_log_path:
  "${WILLY_RECORD_LOG:-}"`` with the variable unset).
* **A schema default is not written in the tree**, so it is never read against the tree's folder.
  Every path default is empty, ``None``, or starts with ``${WILLY_PROJECT_ROOT}``, so a default names
  the repository's file wherever the tree lives (``tests/test_config_paths_resolve_against_the_config_folder.py``
  holds every path field of the schema to that).

``${VAR}`` substitution runs on the text before any of this, so a relative path an environment
variable supplies is read against the tree's folder too: give such a variable an absolute path.
``${WILLY_PROJECT_ROOT}`` is the one name it does not substitute (``src/config/loader.py`` says how it
carries the anchor through the YAML parse instead).

WHY A NON-PATH KEY REFUSES THE ANCHOR RATHER THAN EXPANDING IT
--------------------------------------------------------------
Expanding it there too was the other choice, and it would make a second, looser rule for one name:
anywhere in any string, no "at the start", no "takes no default". The keys that are named like a path
and are not typed :data:`ConfigPath` are exactly the ones that name no file on this machine (a USD
stage path, an asset path Isaac resolves, an Isaac asset root that may be a URL), so a local folder
spliced into them would be a guess about what the operator meant. No tree ever relied on it: the
anchor exists only since the path rule did, and the shipped tree writes it only in path fields. A key
that is not a path still takes any other ``${VAR}``, which the loader substitutes as text.
A value given to :meth:`~src.config.tree.LoadedTree.with_values` is part of the tree it is given
to and is read against that tree's folder, exactly as the same value written in a layer would be.

WHERE IT IS APPLIED
-------------------
Once, at load. Every path field of the schema is typed :data:`ConfigPath`, and the loader validates a
tree with its folder as the validation context, so the validated config holds absolute paths and
every consumer (a calibration artifact, a model directory, a mesh the planning sidecar opens in its
own process) reads the file the rule names without knowing the rule. Pydantic runs a field's
validator only on a value that was given, which is exactly the line between "written in the tree"
and "the schema's default"; the few defaults that name a repository file opt in with
``validate_default=True`` and are anchored, so they come out absolute as well.

A model built in Python with no tree (a test, a sim runner declaring a rig in code) has no folder, and
a relative path in it is left as written: it is read against the working directory, as any Python
path is.

WHY NOT THE WORKING DIRECTORY
-----------------------------
It was, until 2026-09-24: every consumer did ``Path(value)``. That held only while every program ran
from the repository root with the tree at ``config/``. A tree kept in its own folder outside the
repository, with its calibration beside it, read ``calibration/eih_wrist.json`` from wherever the
program happened to be started, and refused a camera whose artifact was on disk.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Final, get_args, get_origin

from pydantic import AfterValidator, BaseModel, ValidationInfo

__all__ = [
    "CONFIG_FOLDER",
    "ConfigPath",
    "PROJECT_ROOT_ANCHOR",
    "PROJECT_ROOT_VARIABLE",
    "anchor_outside_a_path",
    "is_config_path_field",
    "path_note",
    "repository_root",
    "resolve_config_path",
]

#: The environment variable that overrides the repository root, and the name of the anchor.
PROJECT_ROOT_VARIABLE: Final = "WILLY_PROJECT_ROOT"
#: The anchor a path starts with to name a file in the repository: ``${WILLY_PROJECT_ROOT}/assets/...``.
PROJECT_ROOT_ANCHOR: Final = "${" + PROJECT_ROOT_VARIABLE + "}"
#: The anchor as a file can write it, bare or with the default it does not take: the loader leaves both
#: in the text, so both are what a key that is not a path must not hold.
_ANCHOR_WRITTEN: Final = re.compile(r"\$\{" + PROJECT_ROOT_VARIABLE + r"(?::-[^}]*)?\}")
#: The validation-context key under which the loader passes the tree's folder.
CONFIG_FOLDER: Final = "config_folder"

#: This file is src/config/paths.py, so the repository is two levels up from its folder.
_REPOSITORY: Final = Path(__file__).resolve().parents[2]

#: How each resolved path came to be, for a refusal that cannot find the file: resolved path ->
#: (the value as given, the sentence naming the rule). Bounded, because a test run loads hundreds of
#: trees; a note that fell out only makes a refusal shorter.
_NOTES: dict[str, tuple[str, str]] = {}
_NOTES_KEPT: Final = 512


def repository_root() -> Path:
    """What ``${WILLY_PROJECT_ROOT}`` stands for: the environment variable when set, else this repository."""
    raw = os.environ.get(PROJECT_ROOT_VARIABLE, "").strip()
    return Path(raw).expanduser().resolve() if raw else _REPOSITORY


def resolve_config_path(value: str, folder: "Path | str | None") -> str:
    """``value`` as the file it names under the rule the module docstring states.

    ``folder`` is the tree's folder, or ``None`` for a model built with no tree, which leaves a relative
    path as written. The result is POSIX-spelled (``D:/cells/line3/...``): Windows reads that spelling,
    and a path printed in a refusal or pasted into YAML then carries no backslash to escape.
    """
    if not value:
        return value
    if value.startswith(PROJECT_ROOT_ANCHOR):
        rest = value[len(PROJECT_ROOT_ANCHOR):]
        if rest and rest[0] not in "/\\":
            raise ValueError(
                f"{value!r} runs {PROJECT_ROOT_ANCHOR} into a name; the anchor is a folder, so write "
                f"{PROJECT_ROOT_ANCHOR}/<path inside the repository>")
        root = repository_root()
        resolved = _spelled(root / rest.lstrip("/\\"))
        _remember(resolved, value, f"{PROJECT_ROOT_ANCHOR} is the repository, {root.as_posix()}")
        return resolved
    if "${" in value:
        # The loader substitutes every other `${VAR}` before parsing and passes only the anchor through,
        # so what is left is the anchor written with a default or in the middle of a path.
        raise ValueError(
            f"{value!r} holds a variable a path cannot take: {PROJECT_ROOT_ANCHOR} opens a path and takes "
            f"no default, and any other ${{VAR}} is substituted before the tree is read")
    path = Path(value)
    if value.startswith("~"):
        home = path.expanduser()
        resolved = _spelled(home)
        _remember(resolved, value, f"~ is the home folder, {Path('~').expanduser().as_posix()}")
        return resolved
    # `anchor`, not `is_absolute()`: on Windows `\x.json` is rooted at the drive without being absolute, and
    # joining it to a folder would move it to that folder's drive rather than read it where it was written.
    if path.anchor or folder is None:
        return value
    base = Path(folder)
    resolved = _spelled(base / path)
    _remember(resolved, value, f"a relative path in a config is read against the config folder, {base.as_posix()}")
    return resolved


def path_note(path: "str | os.PathLike[str] | None") -> str:
    """Why a path read from a config is the path it is, as a sentence for a refusal, or ``""``.

    A refusal that cannot find a file names the path it looked at; this adds the value as the tree gave
    it and the rule that turned one into the other, so a relative path that resolved somewhere
    unexpected reads as what it is. ``""`` for a path no tree resolved: an absolute one written as it
    is, or one a program built.
    """
    if not path:
        return ""
    note = _NOTES.get(_spelled(Path(os.fspath(path))))
    if note is None:
        return ""
    written, rule = note
    return f" The config gives it as {written}, and {rule}."


def _spelled(path: Path) -> str:
    """``path`` with ``..`` folded away, in the POSIX spelling. Lexical: no link is followed."""
    return Path(os.path.normpath(path)).as_posix()


def _remember(resolved: str, written: str, rule: str) -> None:
    if resolved not in _NOTES and len(_NOTES) >= _NOTES_KEPT:
        try:  # the oldest note goes; two loads on two threads may race for it, and losing costs nothing
            _NOTES.pop(next(iter(_NOTES)), None)
        except (StopIteration, RuntimeError):
            pass
    _NOTES[resolved] = (written, rule)


def _resolve_field(value: Any, info: ValidationInfo) -> Any:
    """The validator behind :data:`ConfigPath`: the tree's folder comes in as the validation context."""
    if not isinstance(value, str):
        return value
    context = info.context if isinstance(info.context, dict) else {}
    return resolve_config_path(value, context.get(CONFIG_FOLDER))


#: The type of every schema field that names a file or a folder. See the module docstring for the rule.
#: A field whose default names a file in the repository anchors it (``"${WILLY_PROJECT_ROOT}/assets/..."``)
#: and sets ``validate_default=True``; pydantic does not validate a default otherwise.
ConfigPath = Annotated[str, AfterValidator(_resolve_field)]


def is_config_path_field(field: Any) -> bool:
    """Whether a pydantic field (``Model.model_fields[name]``) is typed :data:`ConfigPath`, bare or optional."""

    def carries(annotation: Any) -> bool:
        if get_origin(annotation) is Annotated:
            return any(getattr(extra, "func", None) is _resolve_field for extra in get_args(annotation)[1:])
        return any(carries(member) for member in get_args(annotation))

    metadata = getattr(field, "metadata", ())
    return any(getattr(extra, "func", None) is _resolve_field for extra in metadata) or carries(
        getattr(field, "annotation", None))


def anchor_outside_a_path(value: Any) -> str | None:
    """The refusal for ``value`` held by a key that is not a path, or ``None`` when it writes no anchor.

    :class:`~src.config.schema._base.StrictModel` asks this of every field it validates and refuses a
    field that is not typed :data:`ConfigPath` when it answers. Strings are looked for through lists,
    tuples, sets and both sides of a mapping, the containers a schema field holds text in; a nested model
    is not looked into, because its own fields are asked when it is validated. The module docstring says
    why a key that is not a path refuses the anchor instead of expanding it.
    """
    written = _first_anchored_string(value)
    if written is None:
        return None
    return (
        f"{written!r} holds {PROJECT_ROOT_ANCHOR}, and only a path field reads it: it names the repository "
        f"at the start of a file path, and this key is not one, so the text would reach its reader "
        f"unexpanded. Write the value out in full, or take it from another environment variable, which "
        f"the loader substitutes as text before the file is parsed")


def _first_anchored_string(value: Any) -> str | None:
    if isinstance(value, str):
        return value if _ANCHOR_WRITTEN.search(value) else None
    if isinstance(value, BaseModel):
        return None
    if isinstance(value, Mapping):
        members: Any = (item for pair in value.items() for item in pair)
    elif isinstance(value, (list, tuple, set, frozenset)):
        members = value
    else:
        return None
    for member in members:
        found = _first_anchored_string(member)
        if found is not None:
            return found
    return None
