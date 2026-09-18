"""The real UR's planner is rooted at UR's URDF ``base_link``; the controller reports in the DH base.

Every descriptor ``scripts/curobo/build_ur_config.py`` writes carries ``base_link-base_link_inertia``
turned by pi about Z (the builder asserts it), and the controller's base is ``base_link_inertia``'s,
the frame the DH chain and RTDE share. The planner's tool0 is the DH flange turned half a turn about
Z to 0.0001 mm, and 1019 mm from it without the turn (``scripts/curobo/probe_payload_attach.py``,
``frame``).

A planner handed the controller's frame unturned plans to the goal mirrored through the base axis
and routes around a mirrored cell. The Isaac cell does not meet this, because its world, its goals
and its planner all sit in ``base_link``.

:class:`PlannerFrameClient` is the one place the UR driver talks to the planner through: every pose
handed in is turned into the planner's frame and every pose handed back is turned out of it. A half
turn is its own inverse, so both are :func:`planner_pose`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

__all__ = ["PlannerFrameClient", "planner_pose"]


def planner_pose(pose: Sequence[float]) -> list[float]:
    """A wire pose ``[x, y, z, qw, qx, qy, qz]`` turned half a turn about the base Z axis.

    From the controller's base into the planner's, and back: the half turn is its own inverse. The
    rotation is the half turn's quaternion ``(0, 0, 0, 1)`` times the pose's, so an identity comes
    back as the half turn itself.
    """
    x, y, z, qw, qx, qy, qz = (float(v) for v in pose)
    return [-x, -y, z, -qz, -qy, qx, qw]


def _turned_bodies(bodies: "Sequence[Mapping[str, Any]] | None") -> "list[dict[str, Any]] | None":
    if bodies is None:
        return None
    return [dict(body, pose=planner_pose(body["pose"])) for body in bodies]


class PlannerFrameClient:
    """A planner client seen from the controller's base.

    ``plan`` and ``set_world`` turn what they are handed; ``set_voxels``, ``set_scene`` and ``fk``
    are turned too, and exist here only where the client has them, so a caller asking whether the
    channel exists gets the client's answer. Everything else is the client's own: joints are the
    same numbers in either frame.
    """

    def __init__(self, client: Any) -> None:
        self._client = client

    @property
    def inner(self) -> Any:
        """The client this one turns for."""
        return self._client

    def plan(self, start: Sequence[float], goal_pos_m: Sequence[float], goal_quat_wxyz: Sequence[float],
             *args: Any, **kwargs: Any) -> Any:
        turned = planner_pose([*goal_pos_m, *goal_quat_wxyz])
        return self._client.plan(start, turned[:3], turned[3:], *args, **kwargs)

    def set_world(self, cuboids: Sequence[Mapping[str, Any]], *args: Any, **kwargs: Any) -> Any:
        rest = [_turned_bodies(meshes) for meshes in args]
        if "meshes" in kwargs:
            kwargs = dict(kwargs, meshes=_turned_bodies(kwargs["meshes"]))
        return self._client.set_world(_turned_bodies(cuboids), *rest, **kwargs)

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._client, name)
        if name == "set_voxels":
            return self._set_voxels(attribute)
        if name == "set_scene":
            return self._set_scene(attribute)
        if name == "fk":
            return self._fk(attribute)
        return attribute

    @staticmethod
    def _set_voxels(inner: Callable[..., Any]) -> Callable[..., Any]:
        def set_voxels(path: "str | None", **kwargs: Any) -> Any:
            if path and "pose" in kwargs:
                kwargs = dict(kwargs, pose=planner_pose(kwargs["pose"]))
            return inner(path, **kwargs)

        return set_voxels

    @staticmethod
    def _set_scene(inner: Callable[..., Any]) -> Callable[..., Any]:
        def set_scene(cuboids: Sequence[Mapping[str, Any]], meshes: "Sequence[Mapping[str, Any]] | None",
                      voxels: "Mapping[str, Any] | None") -> Any:
            field = None if voxels is None else dict(voxels, pose=planner_pose(voxels["pose"]))
            return inner(_turned_bodies(cuboids), _turned_bodies(meshes), field)

        return set_scene

    @staticmethod
    def _fk(inner: Callable[..., Any]) -> Callable[..., Any]:
        def fk(joints: Sequence[float]) -> Any:
            answer = inner(joints)
            if answer is None:
                return None
            turned = planner_pose([*answer[0], *answer[1]])
            return turned[:3], turned[3:]

        return fk
