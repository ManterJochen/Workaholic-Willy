"""The boxes of a ``set_world`` request written into the planner's own box storage in one go. Standard library only.

Every world refresh hands the sidecar the cell's boxes, and the sidecar registered them the way cuRobo's
``MotionPlanner.update_world`` does: clear the storage, then one ``CuboidData.add`` per box, each with two copies to
the device, a pose inverse, four slot writes and a device sync. On the owner's cell that is 217 ms a refresh at 129
boxes (p10 136, p90 334; 1.6 to 2.0 ms a box, ``ur_curobo.log`` 2026-10-08), about eight refreshes a pick.

cuRobo itself writes a whole scene in one call when it builds a batch of worlds (``CuboidData.load_batch``): one
batched pose, one inverse, slice writes into the same storage, no sync per box. Measured with cuRobo's own code on the
CPU (2026-10-08, 129 boxes): 13.9 ms the box-by-box way, 0.29 ms in one go, and the two fill every slot the same, bit
for bit: the sizes, the inverse poses, the enable flags, the count and the names. The storage stays the very tensors
every solver aliases (the IK, the trajectory optimiser, the graph planner and the checker the sidecar judges with share
``scene_collision_checker.data``), so nothing that captured them needs rebuilding.

:func:`register_boxes_in_place` is that one call, plus what ``update_world`` does around its own: it keeps the scene it
loaded as the checker's and the storage's reference, and resets the graph planner's buffer. :func:`boxes_only` says
when it is the same thing as ``update_world``: a scene of boxes and nothing else, on a planner that holds no mesh and
no grid storage, which ``update_world`` would otherwise clear as well. Anything else goes through ``update_world`` as
it always did. Too many boxes raise before any slot is written, so the planner keeps the world it had; the box-by-box
way wrote the first boxes and then raised. Either way the reply is a refused registration, and the client refuses the
motion on it.

Asked for with ``WILLY_CUROBO_WORLD_IN_PLACE=1``, which the client writes from ``safety.planning_world.
register_in_place`` (off by default). Imported from both sides of the process boundary, like ``_curobo_perceived``: as
part of the Willy package under python 3.11, where the CPU suite runs it against a storage of its own, and as a plain
sibling module by ``curobo_planner_server.py`` under python 3.10. The storage is duck typed: anything with
``data.cuboids.load_batch(cuboids, env)``, ``data.scene_model`` and ``scene_model``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = [
    "ENV_WORLD_IN_PLACE",
    "InPlaceRefused",
    "boxes_only",
    "in_place_asked",
    "register_boxes_in_place",
]

#: Set to ``1`` by a client whose cell asks for its boxes registered in one go (``safety.planning_world.
#: register_in_place``); unset, or anything else, registers them box by box through ``update_world``, as always.
ENV_WORLD_IN_PLACE = "WILLY_CUROBO_WORLD_IN_PLACE"


class InPlaceRefused(RuntimeError):
    """The planner's storage holds another kind of geometry than boxes, so writing the boxes alone would leave it."""


def in_place_asked(environ: Mapping[str, str]) -> bool:
    """Whether the sidecar was started to register boxes in one go: ``WILLY_CUROBO_WORLD_IN_PLACE`` is exactly ``1``."""
    return str(environ.get(ENV_WORLD_IN_PLACE, "") or "").strip() == "1"


def boxes_only(scene: Mapping[str, Any], *, mesh_slots: int, voxel_reserved: bool) -> bool:
    """Whether registering ``scene`` in place does exactly what ``update_world`` does with it.

    ``scene`` is the dict the sidecar hands ``SceneCfg.create``: boxes under ``cuboid``, and ``mesh`` or ``voxel``
    where the request carried them. ``update_world`` clears every kind of storage the planner holds before it adds
    the scene, so a planner that reserved mesh slots or a grid keeps going through it: a mesh or a field registered
    earlier would otherwise stay where the new world has none.
    """
    return (not scene.get("mesh") and not scene.get("voxel") and int(mesh_slots) == 0
            and not bool(voxel_reserved))


def register_boxes_in_place(checker: Any, scene_cfg: Any, graph_planner: Any) -> int:
    """Write ``scene_cfg``'s boxes into ``checker``'s box storage in one call, and return how many it holds now.

    ``checker`` is the planner's ``scene_collision_checker``, ``scene_cfg`` the ``SceneCfg`` the request built, and
    ``graph_planner`` the planner's graph planner or ``None``. What ``MotionPlanner.update_world`` does, for a scene of
    boxes alone: the storage is replaced (``CuboidData.load_batch``, which raises before any write where the boxes
    outnumber the slots), the scene is kept as the storage's and the checker's own reference, as
    ``load_from_scene_cfg`` and ``load_collision_model`` keep it, and the graph planner's buffer is reset.

    :class:`InPlaceRefused` before anything is written where the storage holds meshes or a grid as well: those would
    stay, and only ``update_world`` clears them.
    """
    data = checker.data
    for kind in ("meshes", "voxels"):
        if getattr(data, kind, None) is not None:
            raise InPlaceRefused(
                f"the planner holds {kind} storage, which writing the boxes alone would leave as it was: the world is "
                "registered through update_world")
    cuboids = list(getattr(scene_cfg, "cuboid", None) or [])
    data.cuboids.load_batch(cuboids, 0)
    data.scene_model = scene_cfg
    checker.scene_model = scene_cfg
    if graph_planner is not None:
        graph_planner.reset_buffer()
    return len(cuboids)
