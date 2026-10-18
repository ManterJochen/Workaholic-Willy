"""A cuRobo client with no sidecar behind it: the one planner double a core test hands a planned arm (P0g slice 5).

About twenty test files carry their own planner double, at two levels: some replace the arm's ``CuroboPlanner``, some
the client below it. This one is a client, so the real ``CuroboPlanner`` runs over it: the descriptor check, the world
registration, the frame turn and the execute over the transport are the product's, and only the sidecar is doubled.
It is handed in the way every test hands a client in, through the arm's ``curobo_client_factory``.

What it answers:

- It says who it is as the committed descriptor of its arm and hand does (``tests/_sidecar_identity.py``), so the
  planner's descriptor check passes on the arms that were measured and refuses the ones that were not, as it would.
- Every call is recorded in ``calls``, with its arguments as given.
- A plan to a pose is scripted (``script("plan", path, None, error)``): nothing plans to a pose unscripted,
  because a pose needs an inverse kinematics this double does not have, and a plan it invented would be a fixture
  richer than the artefact. A plan to a joint goal is a straight joint line in five waypoints unless scripted.
- A joint path is clear at any clearance unless ``check_joints`` or ``judge_joints`` is scripted. No part is
  attached: no client attaches one since 2026-10-17.
- A queue that runs dry raises :class:`ScriptExhausted`, as the fake transport's does.

It takes a live scene in one request (``set_scene``), as every cell's camera world sends its boxes and its distance
field since 2026-10-17, and no field on its own (``set_voxels``);
``tests/test_the_fake_transport_keeps_step_with_the_protocols.py`` holds its signatures to ``CuroboPlanClient``'s.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import Any

from src.robot.safety.planning import JointCheckVerdict
from src.robot.safety.planning.curobo_client import PathJudgement, SceneRegistration
from tests._fake_transport import Call, ScriptExhausted
from tests._sidecar_identity import arm_identity

__all__ = ["FakePlanner"]


class FakePlanner:
    """A client for the arm's ``curobo_client_factory``: ``curobo_client_factory=lambda: planner``."""

    def __init__(self, *, arm: str = "ur5e", hand: str = "robotiq_2f85", joint_names: Sequence[str] | None = None,
                 **identity: Any) -> None:
        from src.config.arms import load_arm

        self.identity = arm_identity(arm, hand=hand, **identity)
        self.joint_names = list(joint_names) if joint_names is not None else [j.name for j in load_arm(arm).joints]
        self.calls: list[Call] = []
        self.started = False
        self._scripts: dict[str, deque[Any]] = {}

    def script(self, verb: str, *answers: Any) -> None:
        """Queue ``answers`` for ``verb``: a value is returned, an exception instance raised."""
        if not callable(getattr(type(self), verb, None)):
            raise AttributeError(f"the fake planner does not offer {verb!r}")
        self._scripts.setdefault(verb, deque()).extend(answers)

    def calls_of(self, verb: str) -> list[Call]:
        return [call for call in self.calls if call.verb == verb]

    def _record(self, verb: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append(Call(verb, tuple(args), dict(kwargs)))

    def _scripted(self, verb: str) -> tuple[bool, Any]:
        queue = self._scripts.get(verb)
        if queue is None:
            return False, None
        if not queue:
            raise ScriptExhausted(f"the planner's {verb} was scripted and asked once more than its script answers")
        answer = queue.popleft()
        if isinstance(answer, BaseException):
            raise answer
        return True, answer

    # ---- the client surface the planner reaches (CuroboPlanClient) -------------------------------------------------

    def reserve_world(self, reservation: Any) -> None:
        self._record("reserve_world", reservation)

    def start(self) -> None:
        self._record("start")
        self._scripted("start")
        self.started = True

    def close(self) -> None:
        self._record("close")
        self.started = False

    def set_world(self, cuboids: list[dict], meshes: list[dict] | None = None) -> int:
        """Every obstacle registered unless scripted with the count the planner took."""
        self._record("set_world", [dict(c) for c in cuboids], None if meshes is None else [dict(m) for m in meshes])
        scripted, answer = self._scripted("set_world")
        return int(answer) if scripted else len(cuboids) + len(meshes or ())

    def set_scene(self, cuboids: Sequence[dict], meshes: Sequence[dict] | None,
                  voxels: dict | None) -> SceneRegistration:
        """Every obstacle registered and the field taken, its occupancy as written, unless scripted."""
        self._record("set_scene", [dict(c) for c in cuboids], None if meshes is None else [dict(m) for m in meshes],
                     None if voxels is None else dict(voxels))
        scripted, answer = self._scripted("set_scene")
        if scripted:
            return answer  # type: ignore[no-any-return]
        import numpy as np  # noqa: PLC0415

        taken = None if voxels is None else int(np.load(voxels["path"]).shape[0])
        return SceneRegistration(world_set=len(cuboids) + len(meshes or ()), voxels_set=taken, reason="")

    def check_joints(self, configs: Sequence[Sequence[float]], clearance_mm: float = 0.0) -> JointCheckVerdict:
        """Clear unless scripted."""
        path = [[float(v) for v in config] for config in configs]
        self._record("check_joints", path, clearance_mm=clearance_mm)
        scripted, answer = self._scripted("check_joints")
        if scripted:
            return answer  # type: ignore[no-any-return]
        return JointCheckVerdict(valid=True, first_invalid=None, checked=len(path), reason="clear")

    def judge_joints(self, configs: Sequence[Sequence[float]], *, clearance_mm: float = 0.0, name_pairs: bool = True,
                     ignore_perceived: Sequence[str] | None = None) -> PathJudgement:
        """Every configuration passes unless scripted; the boxes it was asked to set aside are said set aside."""
        path = [[float(v) for v in config] for config in configs]
        self._record("judge_joints", path, clearance_mm=clearance_mm, name_pairs=name_pairs,
                     ignore_perceived=None if ignore_perceived is None else list(ignore_perceived))
        scripted, answer = self._scripted("judge_joints")
        if scripted:
            return answer  # type: ignore[no-any-return]
        return PathJudgement(checked=len(path), clearance_mm=float(clearance_mm), refused=(), pairs_named=bool(name_pairs),
                             perceived_ignored=tuple(sorted(ignore_perceived or ())))

    def plan(self, start_joints: list[float], goal_pos_m: list[float],
             goal_quat_wxyz: list[float]) -> list[list[float]] | None:
        """Scripted, or no plan: this double has no inverse kinematics to plan a pose with."""
        self._record("plan", [float(v) for v in start_joints], [float(v) for v in goal_pos_m],
                     [float(v) for v in goal_quat_wxyz])
        scripted, answer = self._scripted("plan")
        return None if not scripted or answer is None else [[float(v) for v in w] for w in answer]

    def plan_joint(self, start_joints: list[float], goal_joints: list[float]) -> list[list[float]] | None:
        """A straight joint line in five waypoints, start and goal included, unless scripted."""
        start = [float(v) for v in start_joints]
        goal = [float(v) for v in goal_joints]
        self._record("plan_joint", start, goal)
        scripted, answer = self._scripted("plan_joint")
        if scripted:
            return None if answer is None else [[float(v) for v in w] for w in answer]
        return [[s + (g - s) * k / 4.0 for s, g in zip(start, goal, strict=True)] for k in range(5)]
