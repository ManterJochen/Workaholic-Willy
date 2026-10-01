"""The boxes the camera saw, set aside in the planner's world for one judgement and put back exactly. Standard library only.

The owner's Option 1, after the guard fixes of 2026-09-30. cuRobo judges its world with a sphere cover that reaches 25 to
29 mm past the UR10's shoulder housing, so a bin the camera saw beside the base needed about 50 mm for a plan and 60 for
a straight line, where the exact guard alone takes about 26. Where only the camera's boxes refuse a sample on the
planner's world, the exact guard decides them, at ``perceived_min_distance_mm``. To know that "only", the UR driver asks
the planner to judge the refused samples once more with every box the camera saw set aside (``check_js`` with
``ignore_perceived``, ``_curobo_protocol``), and this is how the sidecar sets them aside: in the storage the planner
judges with, by the planner's own enable flag, and back exactly as each flag was before the reply goes out.

What it refuses, before anything is touched (:class:`PerceivedAsideError`, which the sidecar answers as a failed call):

* a request that names no camera box: anything but a list of distinct names that each carry the camera world's prefix;
* a request that does not name every camera box the planner holds, or names one it does not hold: a box left in is one
  the planner judges anyway, and a name it does not hold is a request about another world;
* a flag the storage will not drop: the judgement would run with a camera box still in the world.

A flag that does not come back is :class:`WorldRestoreError`: the planner's world can no longer be vouched for, and the
sidecar exits rather than plan in it. The bench, the declared fixtures and meshes and the camera's distance field are
never touched; they stay the planner's.

The sidecar's ``check_js`` branch decides with two calls of this module and nothing of its own: :func:`requested_aside`
reads the request (``None`` for a plain check, refused while the sidecar may hold a carried part, which only it models),
and :func:`judged_world` opens the world the judgement runs in (the whole world for a plain check, its storage not even
built). So a plain check is judged as check_js always judged it, and the CPU suite holds that too.

Imported from both sides of the process boundary, like ``_curobo_pairs``: as part of the Willy package under python 3.11,
and as a plain sibling module by ``curobo_planner_server.py`` under python 3.10, so the CPU suite holds the very code the
sidecar runs. The storage is anything with ``names()``, ``enabled(name)`` and ``set_enabled(name, enabled)``: the
sidecar's adapter over cuRobo's cuboid storage, the CPU replica's box list, a test's dictionary.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Mapping, Sequence
from types import TracebackType
from typing import Protocol

__all__ = [
    "CuboidFlags",
    "PerceivedAsideError",
    "SetAside",
    "WorldRestoreError",
    "judged_world",
    "requested_aside",
    "requested_names",
]


class PerceivedAsideError(ValueError):
    """The camera's boxes cannot be set aside exactly as asked, so nothing is judged and nothing was left changed."""


class WorldRestoreError(RuntimeError):
    """A box set aside did not come back as it was: the planner's world can no longer be vouched for."""


class CuboidFlags(Protocol):
    """The planner's box storage as this module reads and switches it, one enable flag per box."""

    def names(self) -> "list[str]":
        """Every box the planner holds, in its own order."""

    def enabled(self, name: str) -> bool:
        """Whether the planner judges the box ``name`` now."""

    def set_enabled(self, name: str, enabled: bool) -> None:
        """Switch the box ``name`` in or out of the planner's judgement."""


def requested_names(value: object, prefix: str) -> "tuple[str, ...]":
    """The camera boxes a request asks to set aside, sorted; or :class:`PerceivedAsideError` for anything else.

    A non-empty list of distinct names, each the camera world's ``prefix`` and more: never the bench, a declared fixture
    or a stray name, whatever the request says.
    """
    if not isinstance(value, list) or not value:
        raise PerceivedAsideError(f"the boxes to set aside are not a list of names: {value!r}")
    names: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.startswith(prefix) or len(item) <= len(prefix):
            raise PerceivedAsideError(
                f"{item!r} is not a box the camera world named ({prefix!r}...), and only those are ever set aside")
        names.append(item)
    if len(set(names)) != len(names):
        raise PerceivedAsideError(f"the boxes to set aside name one twice: {names}")
    return tuple(sorted(names))


def requested_aside(request: Mapping[str, object], *, key: str, prefix: str,
                    carrying: bool) -> "tuple[str, ...] | None":
    """The camera boxes a check_js ``request`` asks to set aside, sorted; ``None`` where it does not carry ``key``.

    ``None`` is a plain check, judged in the planner's whole world as check_js always judged it, whatever else holds.
    With the key: the names as :func:`requested_names` reads them, and :class:`PerceivedAsideError` before anything is
    touched for names that are not some of the camera's boxes, or for any while ``carrying``, the sidecar's own mark that
    it may hold a carried part. The carried part is modelled by the planner alone, and with the camera's boxes set aside
    nobody would judge it against them.
    """
    if key not in request:
        return None
    names = requested_names(request.get(key), prefix)
    if carrying:
        raise PerceivedAsideError(
            "this sidecar may hold a carried part, which only it models, so no box the camera saw is set aside")
    return names


def judged_world(names: "Sequence[str] | None", store: "Callable[[], CuboidFlags]", *,
                 prefix: str) -> "contextlib.AbstractContextManager[object]":
    """The world a check_js judgement runs in: the planner's whole world for ``names`` ``None``, a plain check, whose
    ``store`` is not even built; else :class:`SetAside` over the storage ``store`` builds, ``names`` set aside."""
    if names is None:
        return contextlib.nullcontext()
    return SetAside(store(), names, prefix=prefix)


class SetAside:
    """Inside the ``with`` block the named camera boxes are out of the planner's judgement; after it, back as they were.

    ``names`` has to be every box the storage holds whose name carries ``prefix``, and only those
    (:class:`PerceivedAsideError` otherwise, before anything is switched). Every other box is never touched. On the way
    out each flag is set back to what it was on the way in, whatever happened inside, and read back: a flag that does
    not come back raises :class:`WorldRestoreError`, over any error raised inside.
    """

    def __init__(self, store: CuboidFlags, names: "Sequence[str]", *, prefix: str) -> None:
        self._store = store
        self._names = tuple(sorted(str(name) for name in names))
        self._prefix = str(prefix)
        self._before: dict[str, bool] = {}

    @property
    def names(self) -> "tuple[str, ...]":
        """The boxes set aside, sorted."""
        return self._names

    def __enter__(self) -> "SetAside":
        held = sorted(name for name in self._store.names() if str(name).startswith(self._prefix))
        if held != list(self._names) or not self._names:
            missing = sorted(set(self._names) - set(held))
            unnamed = sorted(set(held) - set(self._names))
            raise PerceivedAsideError(
                f"the planner holds the boxes the camera saw {held} and was asked to set aside {list(self._names)}"
                + (f"; it holds no box named {missing}" if missing else "")
                + (f"; {unnamed} were not named" if unnamed else "")
                + ": the request is about another world, so nothing is set aside")
        self._before = {name: bool(self._store.enabled(name)) for name in self._names}
        try:
            for name in self._names:
                self._store.set_enabled(name, False)
            still = [name for name in self._names if self._store.enabled(name)]
            if still:
                raise PerceivedAsideError(f"the planner would not set aside {still}, so nothing is judged without them")
        except BaseException:
            self._restore()
            raise
        return self

    def __exit__(self, kind: "type[BaseException] | None", error: "BaseException | None",
                 trace: "TracebackType | None") -> None:
        self._restore()

    def _restore(self) -> None:
        """Every flag back as it was on the way in, read back; :class:`WorldRestoreError` where one is not."""
        lost: list[str] = []
        for name, was in self._before.items():
            try:
                self._store.set_enabled(name, was)
                if bool(self._store.enabled(name)) != was:
                    lost.append(name)
            except Exception:  # noqa: BLE001 (any failure to put a box back is a world nobody can vouch for)
                lost.append(name)
        if lost:
            raise WorldRestoreError(
                f"the boxes the camera saw {lost} did not come back into the planner's world as they were, so its world "
                "can no longer be vouched for")
