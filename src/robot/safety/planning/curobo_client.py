"""Client for the process-isolated cuRobo planning server, the python 3.11 side of the seam.

cuRobo cannot share the Isaac process, because it wants warp 1.14 and Isaac ships
1.8.2, so this client spawns :mod:`.curobo_planner_server` with the python of the
cuRobo environment and speaks newline-delimited JSON over the subprocess stdio. The
server loads and JIT-warms once, and this client keeps it warm and issues many ``plan``
calls of tens of milliseconds each. It imports neither cuRobo nor Isaac, so it
type-checks in this environment.

Paths default to the in-repo ``ext_deps/`` install location, whose contents are
gitignored, and ``WILLY_CUROBO_PYTHON`` and ``WILLY_CUROBO_ROBOT`` override them, so
nothing machine-specific is baked into the driver. The whole path is opt-in and is
reached only where ``motion_planner="curobo"``.
"""
from __future__ import annotations

import atexit
import json
import math
import os
import queue
import re
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.contracts import UNSET, Maybe, chosen
from src.robot.constants import CUROBO_CLIENT_LOG_FILE, create_robot_logger

from ._curobo_attach import ENV_ATTACH_SPHERES
from ._curobo_body_links import ENV_BODY_LINKS, ENV_DEFAULT_Q, ENV_WRIST_BODY_LINKS
from ._curobo_margin import ENV_SELF_COLLISION_MARGIN_MM
from ._curobo_plan_policy import CLEARANCE_KEY, MAX_CLEARANCE_M
from ._curobo_protocol import (
    ENV_MEASURE_ONLY,
    KIND_SELF_COLLISION,
    KINDS,
    WHERE_DEFAULT_Q,
    WHERE_GOAL,
    WHERE_PATH,
    WHERE_START,
    WHERES,
)
from .environment import (
    ENV_CUROBO_CUBOID_CACHE,
    ENV_CUROBO_MESH_CACHE,
    ENV_CUROBO_STDERR,
    ENV_CUROBO_VOXEL_GRID,
    curobo_cuboid_cache,
    curobo_env_available,
    curobo_mesh_cache,
    curobo_python_path,
    curobo_robot_config,
    curobo_voxel_grid,
)

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from .reservation import PlannerReservation

__all__ = [
    "CuroboNotReady",
    "CuroboPlanClient",
    "CuroboUnavailableError",
    "JointCheckVerdict",
    "MAX_CHECK_CONFIGURATIONS",
    "SceneRegistration",
    "SelfExplanation",
    "SidecarIdentity",
    "StateRefusal",
    "StateRefusalKind",
    "StateWhere",
    "curobo_env_available",
]

# The server file ships in this package and is run by the external cuRobo python. It is
# never imported here.
_SERVER_SCRIPT = str(Path(__file__).with_name("curobo_planner_server.py"))

# The sidecar stderr goes to a file only where WILLY_CUROBO_STDERR is set, and the
# server cannot log here at all, being a different interpreter in a different
# environment. So this client is the only place the planner lifecycle is visible from
# this side. A per-plan line is one per motion and never per candidate, and the polling
# and reader loops stay silent.
logger = create_robot_logger("CuroboPlanClient", CUROBO_CLIENT_LOG_FILE)

# Generous, because the server JIT-warms the cuRobo kernels on first boot: about 25 s
# cold and about 8 s with a warm kernel cache.
_READY_TIMEOUT_S = 120.0
_PLAN_TIMEOUT_S = 30.0


class CuroboUnavailableError(RuntimeError):
    """The cuRobo planning server could not be started or became ready (env missing, JIT/load failure)."""


class StateWhere(StrEnum):
    """Which configuration the sidecar judged: its own retract, the start of a move, its goal, or one of a path.

    ``PATH`` is what a checked path's refusal carries: the sidecar judges the configurations it is handed and does not
    know which is a start, a goal screened on its own or a sample between (found porting to dev, 2026-09-25; it read
    "the start of this move" wherever the sample sat).
    """

    DEFAULT_Q = WHERE_DEFAULT_Q
    START = WHERE_START
    GOAL = WHERE_GOAL
    PATH = WHERE_PATH


class StateRefusalKind(StrEnum):
    """What it found there. ``WORLD`` is the planner's obstacles; the other two are the robot itself."""

    SELF_COLLISION = KIND_SELF_COLLISION
    JOINT_LIMIT = "joint_limit"
    WORLD = "world"


@dataclass(frozen=True, slots=True)
class StateRefusal:
    """Why the sidecar refused one configuration, in the terms an operator can act on.

    ``link_a`` and ``link_b`` are the deepest overlapping pair, and ``depth_mm`` is how far
    into one another they reach. Each is UNSET where the sidecar could not say it: a
    descriptor that resolves its collision spheres from a file carries no per link
    ownership, so the refusal is real and the pair has no name. That costs a name, never a
    plan. ``clearance_mm`` is set where the world was judged at a clearance rather than at no
    penetration: the configuration comes closer to the world than that, and it has no depth.
    """

    where: StateWhere
    kind: StateRefusalKind
    joints: tuple[float, ...]
    link_a: Maybe[str] = UNSET
    link_b: Maybe[str] = UNSET
    depth_mm: Maybe[float] = UNSET
    clearance_mm: Maybe[float] = UNSET

    @classmethod
    def from_reply(cls, block: object) -> StateRefusal:
        """Read a sidecar ``refusal`` block, or raise: a half read refusal sends an operator to the wrong link.

        It is strict on the two words that decide what this is, ``where`` and ``kind``, and
        on the shape of anything it does carry. A missing pair or depth is not malformed, it
        is a sidecar that could not name one.
        """
        if not isinstance(block, dict):
            raise CuroboUnavailableError(f"the cuRobo sidecar sent a refusal that is not a block: {block!r}")
        where, kind = block.get("where"), block.get("kind")
        if where not in WHERES or kind not in KINDS:
            raise CuroboUnavailableError(
                f"the cuRobo sidecar refused a state in terms this client does not know (where={where!r}, "
                f"kind={kind!r}): it is not the sidecar in this tree"
            )
        joints = block.get("joints") or ()
        if not isinstance(joints, (list, tuple)) or not all(
            isinstance(value, (int, float)) and math.isfinite(value) for value in joints
        ):
            raise CuroboUnavailableError(f"the cuRobo sidecar sent a refusal whose configuration is not joints: {joints!r}")
        pair = block.get("pair")
        link_a: Maybe[str] = UNSET
        link_b: Maybe[str] = UNSET
        if pair is not None:
            if not (isinstance(pair, (list, tuple)) and len(pair) == 2 and all(isinstance(n, str) for n in pair)):
                raise CuroboUnavailableError(f"the cuRobo sidecar sent a refusal whose pair is not two links: {pair!r}")
            link_a, link_b = str(pair[0]), str(pair[1])
        depth, clearance = block.get("depth_mm"), block.get("clearance_mm")
        for name, value in (("depth", depth), ("clearance", clearance)):
            if value is not None and not (isinstance(value, (int, float)) and math.isfinite(value)):
                raise CuroboUnavailableError(f"the cuRobo sidecar sent a refusal whose {name} is not a number: {value!r}")
        return cls(
            where=StateWhere(where), kind=StateRefusalKind(kind),
            joints=tuple(float(value) for value in joints),
            link_a=link_a,
            link_b=link_b,
            depth_mm=float(depth) if depth is not None else UNSET,
            clearance_mm=float(clearance) if clearance is not None else UNSET,
        )

    def render(self) -> str:
        """One sentence: what was judged, what it found, and where it reached that deep."""
        where = {
            StateWhere.DEFAULT_Q: "the descriptor's own retract (default_q)",
            StateWhere.START: "the start of this move",
            StateWhere.GOAL: "the goal of this move",
            StateWhere.PATH: "a configuration of the path it was asked to judge",
        }[self.where]
        depth = f"{self.depth_mm:.1f} mm" if chosen(self.depth_mm) else "an unreported depth"
        if self.kind is StateRefusalKind.SELF_COLLISION:
            found = (
                f"{self.link_a} and {self.link_b} overlap by {depth}" if chosen(self.link_a) and chosen(self.link_b)
                else f"a self collision of {depth} between an unnamed pair, because this descriptor does not say "
                     "which link owns which collision sphere"
            )
        elif self.kind is StateRefusalKind.JOINT_LIMIT:
            found = f"a joint sits outside the limits the planner was built with, by {depth}"
        elif chosen(self.clearance_mm):
            found = f"it comes closer than {self.clearance_mm:.1f} mm to the world the planner holds"
        else:
            found = f"it reaches {depth} into the world the planner holds"
        joints = ", ".join(f"{value:.4f}" for value in self.joints)
        return f"cuRobo refuses {where}: {found}. Joints (rad): [{joints}]"

    def to_dict(self) -> dict[str, Any]:
        """The same fields as plain data, ``null`` for each the sidecar did not report."""
        return {
            "where": str(self.where),
            "kind": str(self.kind),
            "joints": list(self.joints),
            "link_a": self.link_a if chosen(self.link_a) else None,
            "link_b": self.link_b if chosen(self.link_b) else None,
            "depth_mm": self.depth_mm if chosen(self.depth_mm) else None,
            "clearance_mm": self.clearance_mm if chosen(self.clearance_mm) else None,
        }


class CuroboNotReady(CuroboUnavailableError):
    """The sidecar started, judged its own retract and refused it. :attr:`refusal` says which links touch."""

    def __init__(self, message: str, *, refusal: StateRefusal) -> None:
        super().__init__(message)
        self.refusal = refusal


def _refusal_or_logged(block: object, *, what: str) -> "StateRefusal | None":
    """The refusal in a reply, or ``None`` with a warning where this client cannot read it.

    A refusal is a diagnostic on a verdict that already stands. Raising here would turn "no
    collision free plan" into "the planner is unavailable" over the spelling of a field, and
    take a driver's status with it.
    """
    if block is None:
        return None
    try:
        return StateRefusal.from_reply(block)
    except CuroboUnavailableError as exc:
        logger.warning("%s, so it is reported without a typed reason: %s (block: %r)", what, exc, block)
        return None


def _raise_if_call_failed(msg: dict) -> None:
    """Turn a failed call into an exception, and leave a genuine absence of a solution as ``None``.

    The two are not the same thing and must never look the same, which is why this
    function exists. A planner that searched and found nothing is a verdict a caller can
    act on: fail safe, do not move. A call that raised is a bug, and returning ``None``
    for it dresses a bug up as a verdict about the robot.

    The failure that shape produces is concrete. ``plan_js`` is not a method on this
    cuRobo build, where it is ``plan_cspace``, so every joint-space request raises
    ``AttributeError``. A broad ``except`` in the sidecar reports that as a failure, the
    client returns ``None``, and the arm reads it as cuRobo refusing the configuration,
    which sends a reader to the robot descriptor and its collision spheres. The clue is
    that the planner cannot plan from a configuration to itself. So the sidecar labels
    which of the two happened, and this raises.
    """
    if msg.get("planner_error"):
        raise CuroboUnavailableError(
            f"the cuRobo planning CALL failed (not a planning verdict): {msg.get('reason')}"
        )


#: The most configurations one batch check request carries. A longer path is split across
#: requests and never thinned, because the samples a thinned path leaves out are samples
#: nobody checked.
MAX_CHECK_CONFIGURATIONS = 1000

#: What a sidecar older than check_js answers it with. The request falls through into that
#: sidecar's plan branch, which reads start_joints first and raises KeyError on it.
_OLD_SIDECAR_MARK = "start_joints"


@dataclass(frozen=True, slots=True)
class JointCheckVerdict:
    """What the cuRobo sidecar said about a joint path, sample by sample.

    ``valid`` is true only when every sample passed. ``first_invalid`` is the 0-based index
    of the first sample that did not, and None exactly when ``valid``. ``checked`` is how
    many samples the sidecar judged, which is always the number sent. ``reason`` says the
    same in a sentence.

    A pass means cuRobo's collision spheres penetrate neither the planner's world nor the
    robot itself, and no joint limit is exceeded. It is not a clearance: the guard margin
    lives in the derived robot the sidecar plans with, and exact meshes stay the
    self-collision guard's job.
    """

    valid: bool
    first_invalid: int | None
    checked: int
    reason: str
    #: Why the sidecar refused ``first_invalid``, where it said. None when it passed, and
    #: None when the sidecar is older than the typed reason or could not name one: the
    #: verdict above stands either way.
    refusal: "StateRefusal | None" = None


@dataclass(frozen=True, slots=True)
class SelfExplanation:
    """What the sidecar found in each of a set of configurations, judged but not planned.

    There is one entry per configuration sent, in that order: whether the robot's own
    spheres overlap, whether every joint is inside the planner's bounds, and, where the
    descriptor says which link owns which sphere, the deepest pair and its depth. ``pairs``
    and ``depths_mm`` hold ``None`` per entry where there was nothing to name.

    They are ``UNSET`` where nobody asked, which is a different thing from ``None`` and has
    to stay different. Naming a pair is pairwise arithmetic over every sphere of the robot,
    on the CPU, per configuration: on a ur5 with the EGU-50, at 590 spheres, that is
    173,755 pairs and 8.29 ms a pose, which is 51 minutes over one candidate family of the
    retract rule, and that rule reads the two verdicts and throws the names away. So a
    caller may ask for the verdicts alone. Were not-asked also ``None``, an evidence file
    would record that family as overlapping nothing, which is a sentence about the robot
    nobody measured.

    The matrix gate compares this against the exact meshes. A sample where the cuRobo term
    says it collides and the pair says nothing, or the other way round, is an attribution
    disagreement, and that is the number the evidence records.
    """

    self_collides: tuple[bool, ...]
    bound_ok: tuple[bool, ...]
    pairs: "Maybe[tuple[tuple[str, str] | None, ...]]"
    depths_mm: "Maybe[tuple[float | None, ...]]"

    @property
    def pairs_named(self) -> bool:
        """Whether the names were asked for at all. Everything that reports has to say which it is."""
        return chosen(self.pairs)

    def __len__(self) -> int:
        return len(self.self_collides)

    def joined(self, other: SelfExplanation) -> SelfExplanation:
        """This explanation followed by ``other``: how a batched question becomes one answer, in order.

        Two halves that were not asked the same question are refused rather than joined:
        joining a named half onto an unnamed one would either invent names for the half that
        has none or drop the half that has them, and both read as a measurement.
        """
        if self.pairs_named != other.pairs_named:
            raise ValueError(
                f"one half of this answer was asked for link names and the other was not "
                f"({self.pairs_named} and {other.pairs_named}): there is no way to join them that is true"
            )
        if not self.pairs_named:
            return SelfExplanation(
                self_collides=self.self_collides + other.self_collides,
                bound_ok=self.bound_ok + other.bound_ok, pairs=UNSET, depths_mm=UNSET,
            )
        assert chosen(self.pairs) and chosen(other.pairs)  # noqa: S101 (narrowed by pairs_named above)
        assert chosen(self.depths_mm) and chosen(other.depths_mm)  # noqa: S101
        return SelfExplanation(
            self_collides=self.self_collides + other.self_collides,
            bound_ok=self.bound_ok + other.bound_ok,
            pairs=self.pairs + other.pairs,
            depths_mm=self.depths_mm + other.depths_mm,
        )

    @classmethod
    def from_reply(cls, msg: dict, *, sent: int, named: bool = True) -> SelfExplanation:
        """Read an ``explain_js`` reply covering ``sent`` configurations, or raise: a partial answer is not an answer.

        ``named`` is what the question asked for. A reply with no names to a question that
        wanted them is a short answer and is refused; a reply with names to a question that
        did not is read as the verdicts alone, because the caller said it would not look.
        """
        if not msg.get("success"):
            _raise_if_call_failed(msg)
            raise CuroboUnavailableError(
                f"the cuRobo sidecar answered explain_js with an unlabelled failure: {msg.get('reason')}"
            )
        collides, bounds = msg.get("self_collides"), msg.get("bound_ok")
        pairs, depths = msg.get("pairs"), msg.get("depths_mm")
        rows = (collides, bounds) + ((pairs, depths) if named else ())
        if not all(isinstance(row, list) and len(row) == sent for row in rows):
            raise CuroboUnavailableError(
                f"the cuRobo sidecar was asked about {sent} configuration(s) and answered for "
                f"{[len(row) if isinstance(row, list) else None for row in rows]}: nothing here judged them all"
            )
        assert isinstance(collides, list) and isinstance(bounds, list)  # noqa: S101 (narrowed by the check above)
        verdicts = {
            "self_collides": tuple(bool(value) for value in collides),
            "bound_ok": tuple(bool(value) for value in bounds),
        }
        if not named:
            return cls(pairs=UNSET, depths_mm=UNSET, **verdicts)
        assert isinstance(pairs, list) and isinstance(depths, list)  # noqa: S101
        return cls(
            pairs=tuple(
                (str(pair[0]), str(pair[1])) if isinstance(pair, (list, tuple)) and len(pair) == 2 else None
                for pair in pairs
            ),
            depths_mm=tuple(
                float(depth) if isinstance(depth, (int, float)) and math.isfinite(depth) else None
                for depth in depths
            ),
            **verdicts,
        )

    def render(self) -> str:
        """How many of them collide with the robot itself, and the deepest pair among those that do."""
        hits = sum(self.self_collides)
        if not chosen(self.pairs) or not chosen(self.depths_mm):
            named = ", which link pairs was not asked"
        else:
            deepest = max(
                ((depth, pair) for depth, pair in zip(self.depths_mm, self.pairs) if depth is not None),
                default=None,
            )
            named = (f", deepest {deepest[1][0]} and {deepest[1][1]} by {deepest[0]:.1f} mm"
                     if deepest and deepest[1] else "")
        outside = sum(not ok for ok in self.bound_ok)
        return (f"{hits} of {len(self)} configuration(s) collide with the robot itself{named}; "
                f"{outside} outside the planner's joint bounds")

    def to_dict(self) -> dict[str, Any]:
        """The same rows as plain data, ready for the evidence file."""
        named = self.pairs_named
        return {
            "self_collides": list(self.self_collides),
            "bound_ok": list(self.bound_ok),
            # `pairs_named` is what keeps a null here from reading as "nothing overlapped".
            "pairs_named": named,
            "pairs": ([list(pair) if pair is not None else None for pair in self.pairs]
                      if chosen(self.pairs) else None),
            "depths_mm": list(self.depths_mm) if chosen(self.depths_mm) else None,
        }


@dataclass(frozen=True, slots=True)
class SceneRegistration:
    """What the sidecar did with one scene request.

    ``world_set`` is how many boxes and meshes it holds now, 0 when it confirmed nothing.
    ``voxels_set`` is how many field values it took, or None when it took no field.
    ``reason`` is the sidecar's own sentence when it refused the field, word for word,
    because "started with no voxel storage" points an operator at the one setting that
    matters where "the planner refused it" points nowhere.
    """

    world_set: int
    voxels_set: int | None
    reason: str


def _verdict_from_reply(msg: dict, *, sent: int, clearance_m: float = 0.0) -> JointCheckVerdict:
    """Read a check_js reply as a verdict on ``sent`` samples, or raise: only a whole verdict passes.

    A failed call, an old sidecar and a reply that does not account for every sample sent
    all raise ``CuroboUnavailableError``, because each of them means nobody judged the path.
    So does a reply to a request that asked for a clearance and does not say it judged at
    that clearance: a sidecar from before the clearance judges at no penetration and passes
    a line that grazes the world, which is exactly the line the clearance is there to refuse.
    """
    if msg.get("success") and clearance_m > 0.0:
        judged_at = msg.get(CLEARANCE_KEY)
        if not isinstance(judged_at, (int, float)) or abs(float(judged_at) - clearance_m) > 1e-9:
            raise CuroboUnavailableError(
                f"the cuRobo sidecar was asked to judge at {clearance_m * 1000.0:g} mm clearance and says it judged at "
                f"{judged_at!r} m; it is not the sidecar in this tree, restart it"
            )
    if not msg.get("success"):
        reason = msg.get("reason")
        if _OLD_SIDECAR_MARK in str(reason):
            raise CuroboUnavailableError(
                "the sidecar does not know check_js; restart it from this tree "
                f"(it answered through its plan branch: {reason})"
            )
        _raise_if_call_failed(msg)
        raise CuroboUnavailableError(
            f"the cuRobo sidecar answered check_js with an unlabelled failure, so nothing was "
            f"judged: {reason}"
        )
    valid, first, checked = msg.get("valid"), msg.get("first_invalid"), msg.get("checked")
    counted = type(checked) is int and checked == sent
    if counted and valid is True and first is None:
        return JointCheckVerdict(
            valid=True, first_invalid=None, checked=sent,
            reason=f"all {sent} samples pass the cuRobo check",
        )
    if counted and valid is False and type(first) is int and 0 <= first < sent:
        refusal = _refusal_or_logged(
            msg.get("refusal"), what=f"the cuRobo sidecar refused sample {first} in terms this client cannot read",
        )
        return JointCheckVerdict(
            valid=False, first_invalid=first, checked=sent,
            reason=(f"the cuRobo check refuses sample {first} of {sent}, counted from 0: a joint "
                    f"limit, a self collision or the planner's world"),
            refusal=refusal,
        )
    raise CuroboUnavailableError(
        f"the cuRobo sidecar answered check_js on {sent} samples with a reply that is not a whole "
        f"verdict: {msg}"
    )


def _log_at_exit(emit: "Callable[..., None]", message: str, *args: object) -> None:
    """Log without raising while the interpreter is already tearing its logging streams down.

    `close()` is registered with atexit, so it runs after the host has closed the
    streams the handlers write to. A ValueError from a shutdown message is noise that
    reads like a real failure, arriving as an I/O-operation-on-closed-file traceback
    under a call stack pointing at this line.
    """
    try:
        emit(message, *args)
    except (ValueError, OSError):  # streams already closed at interpreter exit
        pass


_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


def _reported_sha256(value: object) -> Maybe[str]:
    return value if isinstance(value, str) and _SHA256_HEX.fullmatch(value) else UNSET


@dataclass(frozen=True)
class SidecarIdentity:
    """Who a started sidecar says it is, read from its ready line.

    ``provenance`` is the ``_provenance`` block of the descriptor, naming the arm, hand and
    plate it was built for, and ``None`` where the sidecar reported none. The hashes are the
    sidecar's own: ``arm_descriptor_sha256`` over the descriptor's ``robot_cfg`` and
    ``composed_sha256`` over the config the planner loaded without a wrist camera's body, both
    canonical JSON through ``_curobo_body_links.canonical_sha256``, and ``urdf_sha256`` over
    the URDF its kinematics resolved, line endings normalised. ``bodies`` names the body links
    the evidence config holds, ``wrist_bodies`` the wrist cameras loaded on top of it, and
    ``wrist_bodies_sha256`` the hash of those rows.

    Each is UNSET where the sidecar did not say it, never ``''`` or ``None``: a sidecar
    older than these fields says nothing, and an empty value would read as one. A sidecar
    handed no wrist body reports neither wrist field. The drivers refuse on the provenance
    and the hashes, and on the wrist bodies (``body_link.wrist_body_refusal``).
    """

    provenance: dict[str, object] | None = None
    arm_descriptor_sha256: Maybe[str] = UNSET
    urdf_sha256: Maybe[str] = UNSET
    composed_sha256: Maybe[str] = UNSET
    #: One row per body link the sidecar composed in, as ``_curobo_body_links.body_report``
    #: builds it: link, parent, fixed transform, sphere and slot counts, the spheres' hash,
    #: buffer and ignore list.
    bodies: Maybe[tuple[dict[str, object], ...]] = UNSET
    #: One row per wrist camera the sidecar loaded on top of the evidence config: the
    #: ``body_report`` row plus the rig, the camera model, the margin and the grown boxes it
    #: was sent with.
    wrist_bodies: Maybe[tuple[dict[str, object], ...]] = UNSET
    wrist_bodies_sha256: Maybe[str] = UNSET

    @property
    def body_names(self) -> tuple[str, ...]:
        """The links the sidecar added, in order; empty where it reported none or said nothing."""
        return tuple(str(row.get("link")) for row in self.bodies) if chosen(self.bodies) else ()

    @classmethod
    def from_ready(cls, ready: Mapping[str, object]) -> SidecarIdentity:
        """What a ready line says, field by field, and UNSET for every field it does not carry as a real value."""
        descriptor = ready.get("descriptor")
        bodies = ready.get("bodies")
        well_formed = isinstance(bodies, list) and all(
            isinstance(row, dict) and isinstance(row.get("link"), str) for row in bodies
        )
        wrist = ready.get("wrist_bodies")
        wrist_well_formed = isinstance(wrist, list) and all(
            isinstance(row, dict) and isinstance(row.get("link"), str) for row in wrist
        )
        return cls(
            provenance=dict(descriptor) if isinstance(descriptor, dict) else None,
            arm_descriptor_sha256=_reported_sha256(ready.get("arm_descriptor_sha256")),
            urdf_sha256=_reported_sha256(ready.get("urdf_sha256")),
            composed_sha256=_reported_sha256(ready.get("composed_sha256")),
            bodies=tuple(dict(row) for row in bodies) if well_formed and isinstance(bodies, list) else UNSET,
            wrist_bodies=(tuple(dict(row) for row in wrist) if wrist_well_formed and isinstance(wrist, list)
                          else UNSET),
            wrist_bodies_sha256=_reported_sha256(ready.get("wrist_bodies_sha256")),
        )

    @classmethod
    def from_client(cls, client: object) -> SidecarIdentity:
        """What any client says, an injected stub included: its identity, else its provenance alone, else nothing.

        Nothing is an identity with no provenance, which the descriptor refusal refuses, so
        a client that cannot say which descriptor it loaded never plans.
        """
        identity = getattr(client, "identity", None)
        if isinstance(identity, SidecarIdentity):
            return identity
        provenance = getattr(client, "descriptor_provenance", None)
        return cls(provenance=dict(provenance) if isinstance(provenance, dict) else None)

    def render(self) -> str:
        def said(value: object) -> str:
            return str(value) if chosen(value) else "not reported"

        if self.provenance is None:
            descriptor = "no _provenance reported"
        else:
            descriptor = ", ".join(f"{key} {self.provenance[key]}" for key in sorted(self.provenance))
        if chosen(self.bodies):
            bodies = "; ".join(
                f"{row.get('link')} under {row.get('parent')}, {row.get('spheres')} spheres" for row in self.bodies
            ) if self.bodies else "none"
        else:
            bodies = "not reported"
        lines = [
            "cuRobo sidecar identity",
            f"  descriptor: {descriptor}",
            f"  arm descriptor sha256: {said(self.arm_descriptor_sha256)}",
            f"  urdf sha256: {said(self.urdf_sha256)}",
            f"  composed sha256: {said(self.composed_sha256)}",
            f"  body links: {bodies}",
        ]
        if chosen(self.wrist_bodies):
            lines.append("  wrist cameras: " + ("; ".join(
                f"{row.get('link')} ({row.get('model')}), {row.get('spheres')} spheres" for row in self.wrist_bodies
            ) or "none"))
        return "\n".join(lines).encode("ascii", "backslashreplace").decode("ascii")

    def to_dict(self) -> dict[str, Any]:
        """The same fields as plain data, ``null`` for each the sidecar did not report."""
        return {
            "provenance": dict(self.provenance) if self.provenance is not None else None,
            "arm_descriptor_sha256": self.arm_descriptor_sha256 if chosen(self.arm_descriptor_sha256) else None,
            "urdf_sha256": self.urdf_sha256 if chosen(self.urdf_sha256) else None,
            "composed_sha256": self.composed_sha256 if chosen(self.composed_sha256) else None,
            "bodies": [dict(row) for row in self.bodies] if chosen(self.bodies) else None,
            "wrist_bodies": [dict(row) for row in self.wrist_bodies] if chosen(self.wrist_bodies) else None,
            "wrist_bodies_sha256": self.wrist_bodies_sha256 if chosen(self.wrist_bodies_sha256) else None,
        }


class CuroboPlanClient:
    """Spawns + drives the warm cuRobo planning server. ``plan`` returns a joint trajectory or ``None``."""

    def __init__(
        self,
        *,
        python_path: str | None = None,
        server_script: str | None = None,
        robot_config: str | None = None,
        scene_config: str | None = None,
        stderr_log: str | None = None,
        self_collision_margin_mm: float = 0.0,
        attach_spheres: int = 0,
        mesh_cache: int | None = None,
        voxel_grid: str | None = None,
        body_links: Sequence[Mapping[str, Any]] = (),
        wrist_body_links: Sequence[Mapping[str, Any]] = (),
        measure_only: bool = False,
        default_q: "Sequence[float] | None" = None,
    ) -> None:
        # Every path and knob resolves through safety.planning.environment, the single
        # anchor, so a caller overrides what it needs and the rest tracks the variables
        # documented there.
        self._python = python_path or curobo_python_path()
        self._script = server_script or _SERVER_SCRIPT
        self._robot = robot_config or curobo_robot_config()
        self._scene = scene_config or curobo_cuboid_cache()
        # The clearance the safety guard will demand of the final configuration of the
        # plan. It is handed to the sidecar so cuRobo plans with that margin instead of
        # returning paths the guard then refuses: measured, 9.44 to 9.47 mm plans against
        # a 10.000 mm guard margin. 0.0 leaves the config untouched.
        self._self_collision_margin_mm = float(self_collision_margin_mm)
        # Collision-sphere slots reserved for a carried payload. At 0 the sidecar robot
        # config is untouched and `attach_payload` refuses, which is the unchanged path.
        self._attach_spheres = max(0, int(attach_spheres))
        #: The bodies the sidecar adds to the arm's descriptor when it starts: the hand a
        #: cell names, placed by its declared tool frame. Empty adds nothing, and a driver
        #: then refuses the sidecar for having no hand.
        self._body_links = [dict(body) for body in body_links]
        #: A wrist camera's bodies, loaded on top of the config the evidence names. Sent in their
        #: own variable, so a cell without a camera sends the same bytes as a client that knows no
        #: wrist bodies.
        self._wrist_body_links = [dict(body) for body in wrist_body_links]
        # Slots for the two other kinds of obstacle. Both are reserved when the planner is
        # BUILT, so they are decided here and never again: a mesh or a voxel grid sent to a
        # planner that reserved none has nowhere to go. 0 and empty leave the sidecar as it was.
        self._mesh_cache = max(0, int(mesh_cache if mesh_cache is not None else curobo_mesh_cache()))
        self._voxel_grid = str(voxel_grid if voxel_grid is not None else curobo_voxel_grid()).strip()
        #: Where this client writes the live scene's field. The name carries the process and
        #: this client, because two arms in one process would otherwise write one file and
        #: each could register the other's scene.
        self._live_scene_path = str(
            Path(tempfile.gettempdir()) / f"willy_live_scene_{os.getpid()}_{id(self)}.npy"
        )
        #: True once a reservation decided the slots: from then on the config, not the shell,
        #: sizes the sidecar. A client nobody reserved for inherits the shell.
        self._reserved = False
        # The server stderr, carrying cuRobo warmup and plan diagnostics, goes to a log
        # file where one is requested through the parameter or WILLY_CUROBO_STDERR, and
        # is discarded otherwise. It is what makes the isolated server debuggable on-box.
        self._stderr_log = stderr_log or os.environ.get(ENV_CUROBO_STDERR)
        self._proc: subprocess.Popen[str] | None = None
        self._q: "queue.Queue[dict | None]" = queue.Queue()
        #: The monotonic request counter. Every request carries it and every reply echoes
        #: it, which is what lets a late answer be dropped instead of executed.
        self._next_id = 0
        #: False once the sidecar stdout hit EOF or a write to it failed. `self._proc`
        #: alone cannot say this, because only `close()` clears it, so a client whose
        #: child had died would still believe it was running.
        self._alive = True
        self._warned_unstamped = False
        self._reader: threading.Thread | None = None
        self.joint_names: list[str] = []
        self.dt: float = 0.0
        #: Who the sidecar said it is when it became ready: the provenance of the descriptor
        #: and the hashes of what it loaded. Nothing is reported until then. The drivers
        #: refuse on it; the client only keeps it.
        self.identity = SidecarIdentity()
        #: The retract this arm and hand were judged at, from the committed table. The
        #: descriptor carries one per arm and it can only be right about a bare arm: the
        #: ur3's own retract clears the Hand-E by 13.4 mm on the exact meshes and is a self
        #: collision in the planner's sphere model, so the sidecar never becomes ready. None
        #: leaves the descriptor's own pose, which is what a bare arm wants.
        self._default_q = [float(value) for value in default_q] if default_q is not None else None
        #: True for a caller that only wants to measure: it accepts a sidecar that reports a
        #: refused retract instead of exiting, and refuses to plan. Only the matrix gate asks
        #: for it, and a driver never does.
        self._measure_only = bool(measure_only)
        #: Why the sidecar refused the last thing it was asked, or None. It is set by
        #: start(), plan and plan_joint, and cleared by the next plan that succeeds, so it
        #: describes the last answer and not the history.
        self.last_refusal: StateRefusal | None = None

    @property
    def measure_only(self) -> bool:
        """True when this client asked for a sidecar that measures instead of planning."""
        return self._measure_only

    @property
    def descriptor_provenance(self) -> dict[str, object] | None:
        """The ``_provenance`` of the descriptor the sidecar loaded, a view of :attr:`identity`.

        It is ``None`` where the sidecar said none.
        """
        return self.identity.provenance

    @descriptor_provenance.setter
    def descriptor_provenance(self, provenance: dict[str, object] | None) -> None:
        self.identity = replace(self.identity, provenance=dict(provenance) if isinstance(provenance, dict) else None)

    @property
    def live_scene_path(self) -> str:
        """The file this client's live scene field is written to before it is registered."""
        return self._live_scene_path

    # --- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        """Spawn the server and block until it reports ``ready`` (raises CuroboUnavailableError otherwise)."""
        if self._proc is not None:
            return
        if not Path(self._python).exists():
            raise CuroboUnavailableError(f"cuRobo env python not found: {self._python}")
        logger.info(
            "spawning the cuRobo sidecar: python=%s robot=%s cuboid_cache=%s mesh_cache=%d voxel_grid=%s "
            "self_collision_margin=%.3f mm",
            self._python, self._robot, self._scene, self._mesh_cache, self._voxel_grid or "none",
            self._self_collision_margin_mm,
        )
        started = time.monotonic()
        stderr = open(self._stderr_log, "w") if self._stderr_log else subprocess.DEVNULL  # noqa: SIM115
        env = self._sidecar_env()
        self._alive = True
        self._proc = subprocess.Popen(
            [self._python, "-u", self._script, self._robot, self._scene],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr, text=True, bufsize=1,
            env=env,
        )
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()
        # Close on interpreter exit, so a runner that never calls disconnect() still
        # kills the server first. The Isaac picks rely on the Isaac hard teardown, and
        # without this the daemon _pump thread is left blocked on the server pipe when
        # Isaac tears the process down, which crashes the native teardown and produces a
        # 12 MB Kit dump. atexit runs before the deep teardown, and close() is
        # idempotent, so an earlier explicit disconnect is harmless.
        atexit.register(self.close)
        msg = self._recv(_READY_TIMEOUT_S)          # the handshake carries no id
        if not msg or msg.get("status") != "ready":
            reason = (msg or {}).get("reason", "no ready signal (timeout or server died)")
            # A refusal here is the sidecar judging its own retract and finding the arm
            # inside itself or inside its hand. It is typed, because the next step for an
            # operator is a geometry and not a restart.
            refused = _refusal_or_logged(
                (msg or {}).get("refusal"), what="the cuRobo sidecar refused to start in terms this client cannot read",
            )
            self.close()
            if refused is not None:
                raise CuroboNotReady(f"cuRobo server did not become ready: {refused.render()}", refusal=refused)
            raise CuroboUnavailableError(f"cuRobo server did not become ready: {reason}")
        if msg.get("measure_only") and not self._measure_only:
            # A measuring sidecar answers every plan with a refusal. Planning against one
            # would look like a cell whose every goal is blocked, which is exactly the
            # diagnosis it would send an operator chasing.
            self.close()
            raise CuroboUnavailableError(
                f"the cuRobo sidecar started to MEASURE, not to plan ({ENV_MEASURE_ONLY} is set in its environment), "
                "and this client asked for a planner"
            )
        self.last_refusal = _refusal_or_logged(
            msg.get("refusal"), what="the measuring sidecar reported a refusal this client cannot read",
        )
        self.joint_names = list(msg["joint_names"])
        self.dt = float(msg.get("dt", 0.0))
        self.identity = SidecarIdentity.from_ready(msg)
        # The warm-up cost is worth recording: about 25 s cold against about 8 s with a
        # warm kernel cache is the difference between a slow planner and a kernel cache
        # that was thrown away.
        logger.info(
            "cuRobo sidecar ready after %.1f s: %d joint(s), dt=%.4f s",
            time.monotonic() - started, len(self.joint_names), self.dt,
        )

    def _pump(self) -> None:
        """The reader thread. It pushes each JSON line from the server stdout onto the queue.

        It puts ``None`` on EOF. It captures the stream locally, because ``close()``
        nulls ``self._proc``, and swallows the read error that fires when ``close()``
        shuts the pipe under this thread during teardown, so a graceful close never
        raises here.
        """
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        try:
            for line in proc.stdout:
                line = line.strip()
                if line.startswith("{"):
                    try:
                        self._q.put(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        except (ValueError, OSError):  # stream closed under us during shutdown -> stop quietly
            pass
        # This is where the client learns its sidecar died. Only `close()` clears
        # `self._proc`, so without this a client whose child had exited keeps believing
        # it is running and the next `_send` writes into a dead pipe.
        self._alive = False
        self._q.put(None)  # EOF / server exit

    def _recv(self, timeout_s: float, *, want: "int | None" = None) -> "dict | None":
        """The reply to request ``want``, or ``None`` on a timeout or a server exit.

        A late reply is dropped here, and dropping it is the point. This is request and
        response over one queue: a call that times out stops waiting and leaves its
        request outstanding, so the late answer from the sidecar lands in the queue and
        would otherwise be handed to whatever asks next. Reproduced end to end with a
        stub sidecar, one plan timeout leaves the next motion holding the trajectory of
        the previous goal, and a real arm would drive it.

        The budget is not reset per message. Draining a backlog must not extend the
        caller timeout, or a sidecar emitting stale lines holds a motion open
        indefinitely.
        """
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return None
            try:
                msg = self._q.get(timeout=remaining)
            except queue.Empty:
                return None
            if msg is None:                       # EOF: the sidecar exited
                self._alive = False
                return None
            got = msg.get("id")
            if want is None or got is None or got == want:
                # `want is None` is the ready handshake, emitted before the server loop
                # and carrying no id. `got is None` is a reply from a server older than
                # this protocol, accepted rather than refused, because rejecting every
                # answer would be a worse failure than the one this guards, and said
                # once so it cannot pass unnoticed.
                if want is not None and got is None and not self._warned_unstamped:
                    self._warned_unstamped = True
                    logger.warning(
                        "the cuRobo sidecar answers without a request id, so a late reply cannot be "
                        "told apart from the right one; it is older than this client")
                return msg
            logger.warning(
                "dropped a late cuRobo reply for request %s while waiting for %s; it would have "
                "been used as the answer to this call", got, want)

    def _send(self, obj: dict) -> int:
        """Write one request and return the id its reply must carry.

        A dead sidecar fails typed here. Without `self._proc` being cleared and `poll()`
        consulted, the call after the sidecar dies writes to the stdin of a dead child
        and raises `OSError` straight out of the fail-closed motion path. Measured at
        0.0, 0.3 and 2.0 s after the death of the sidecar, only the 0.0 s case produces
        the typed error first, so in any realistic case the typed
        `CONTROLLER_REJECTED` never fires at all.
        """
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise CuroboUnavailableError("cuRobo server is not running")
        # The exit code is the half an operator can act on. `poll()` is consulted even
        # where `_alive` is already false, because a server that is not running and a
        # server that exited with 1 send someone to two different places, and only the
        # second names a child process.
        if not self._alive or proc.poll() is not None:
            self._alive = False
            code = proc.poll()
            raise CuroboUnavailableError(
                f"the cuRobo sidecar exited with code {code}; no plan can be trusted until it is "
                f"restarted" if code is not None else
                "the cuRobo sidecar's stream ended; no plan can be trusted until it is restarted")
        self._next_id += 1
        obj = {**obj, "id": self._next_id}
        try:
            proc.stdin.write(json.dumps(obj) + "\n")
            proc.stdin.flush()
        except OSError as exc:
            # The child can die between poll() and write(). A typed refusal keeps this
            # inside the vocabulary of the motion path instead of surfacing an OSError
            # from a pipe.
            self._alive = False
            raise CuroboUnavailableError(
                f"writing to the cuRobo sidecar failed ({type(exc).__name__}: {exc}); it is gone"
            ) from exc
        return self._next_id


    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                proc.stdin.flush()
            proc.wait(timeout=5)
            _log_at_exit(logger.info, "cuRobo sidecar shut down cleanly")
        except Exception as exc:  # noqa: BLE001 (shutdown is best-effort)
            _log_at_exit(logger.warning,
                         "cuRobo sidecar did not shut down cleanly (%s); killing it", exc)
            proc.kill()

    # --- planning ----------------------------------------------------------
    def plan(
        self,
        start_joints: list[float],
        goal_pos_m: list[float],
        goal_quat_wxyz: list[float],
    ) -> list[list[float]] | None:
        """Plan tool0 -> ``goal`` (metres, WXYZ, base frame) from ``start_joints`` (rad).

        It returns the interpolated joint trajectory ``[[6 rad], ...]`` in
        :attr:`joint_names` order, or ``None`` where cuRobo found no collision-free
        solution, and the caller then fails safe with no blind motion.
        """
        if self._proc is None:
            self.start()
        self.last_refusal = None
        started = time.monotonic()
        want = self._send({"start_joints": list(start_joints), "goal_pos_m": list(goal_pos_m),
                           "goal_quat_wxyz": list(goal_quat_wxyz)})
        msg = self._recv(_PLAN_TIMEOUT_S, want=want)
        if msg is None:
            raise CuroboUnavailableError("cuRobo server stopped responding (timeout / died)")
        if msg.get("success"):
            traj: list[list[float]] = msg["trajectory"]
            logger.debug(
                "planned to (%.3f, %.3f, %.3f) m in %.0f ms: %d waypoint(s)",
                goal_pos_m[0], goal_pos_m[1], goal_pos_m[2],
                (time.monotonic() - started) * 1000.0, len(traj),
            )
            return traj
        _raise_if_call_failed(msg)
        # A verdict rather than a failure, but the caller turns it into no motion, so the
        # reason a pick stopped is recoverable from here alone.
        self.last_refusal = _refusal_or_logged(
            msg.get("refusal"), what="the cuRobo sidecar refused this move in terms this client cannot read",
        )
        logger.warning(
            "cuRobo found NO collision-free plan to (%.3f, %.3f, %.3f) m after %.0f ms: %s",
            goal_pos_m[0], goal_pos_m[1], goal_pos_m[2],
            (time.monotonic() - started) * 1000.0,
            self.last_refusal.render() if self.last_refusal is not None else msg.get("reason", "no reason given"),
        )
        return None

    def plan_joint(
        self, start_joints: list[float], goal_joints: list[float]
    ) -> list[list[float]] | None:
        """Plan ``start_joints`` -> ``goal_joints`` (rad) collision-free, or ``None`` if it cannot.

        This is the joint-space twin of :meth:`plan`, for goals that are joint
        configurations, as park and home are stored. Going through FK and a Cartesian
        plan instead would let cuRobo satisfy the tool pose with any IK branch, and a
        flipped elbow branch is how 43 mm of hidden self-penetration gets into a pose
        that looks fine.

        ``None`` means there is no collision-free plan, and the caller fails safe rather
        than moving blindly, which is the reason to ask for a plan instead of
        interpolating.
        """
        if self._proc is None:
            self.start()
        self.last_refusal = None
        started = time.monotonic()
        want = self._send({"cmd": "plan_js", "start_joints": list(start_joints),
                           "goal_joints": list(goal_joints)})
        msg = self._recv(_PLAN_TIMEOUT_S, want=want)
        if msg is None:
            raise CuroboUnavailableError("cuRobo server stopped responding (timeout / died)")
        if msg.get("success"):
            traj: list[list[float]] = msg["trajectory"]
            logger.debug(
                "planned a joint move in %.0f ms: %d waypoint(s)",
                (time.monotonic() - started) * 1000.0, len(traj),
            )
            return traj
        _raise_if_call_failed(msg)
        self.last_refusal = _refusal_or_logged(
            msg.get("refusal"), what="the cuRobo sidecar refused this joint move in terms this client cannot read",
        )
        logger.warning(
            "cuRobo found NO collision-free joint plan after %.0f ms: %s",
            (time.monotonic() - started) * 1000.0,
            self.last_refusal.render() if self.last_refusal is not None else msg.get("reason", "no reason given"),
        )
        return None

    def check_joints(
        self, configs: "Sequence[Sequence[float]]", *, clearance_mm: float = 0.0,
    ) -> JointCheckVerdict:
        """Judge every configuration of a joint path in the world this sidecar plans in.

        ``configs`` are joint configurations in :attr:`joint_names` order, in radians and in
        path order. A path longer than :data:`MAX_CHECK_CONFIGURATIONS` is sent in that many
        at a time and the verdict names the first sample cuRobo refuses, counted in the whole
        path.

        ``clearance_mm`` above 0 refuses a sample closer than that to the planner's world, where
        0, the default and every planned path's check, refuses only a sample that penetrates it.
        The robot's own terms are judged as they always are. It is sent only when it is not 0, so
        every other request is the one it always was, and a reply that does not say it judged at
        the clearance asked is refused (``CuroboUnavailableError``).

        The cap is the size of a request and not a rule about moves. Refusing a longer path
        outright costs real picks: a cuRobo plan of 121 waypoints needs more than 1000
        samples, and that was about one pick in ten on the M2 Isaac gate. Splitting the
        request thins nothing and leaves no gap; it only takes longer.

        Refused with ``ValueError`` before anything is sent: no configuration, a value that
        is not a finite number, or a configuration whose length is not the sidecar's joint
        count. The first two are refused before the sidecar is started at all.

        Raises ``CuroboUnavailableError`` whenever no verdict came back: the sidecar could
        not start, did not answer in time, exited, failed inside the call, answered with
        something that is not a whole verdict, or is older than ``check_js``. None of these
        is ever returned as a verdict.
        """
        rows = [[float(v) for v in config] for config in configs]
        if not rows:
            raise ValueError(
                "check_joints was given no configuration, and an empty path is not a checked path"
            )
        clearance_m = float(clearance_mm) / 1000.0
        if not math.isfinite(clearance_m) or not 0.0 <= clearance_m <= MAX_CLEARANCE_M:
            raise ValueError(
                f"clearance_mm has to lie from 0 to {MAX_CLEARANCE_M * 1000.0:g} mm, got {clearance_mm!r}"
            )
        asked: dict[str, Any] = {CLEARANCE_KEY: clearance_m} if clearance_m > 0.0 else {}
        for index, row in enumerate(rows):
            if not all(math.isfinite(v) for v in row):
                raise ValueError(
                    f"sample {index} of the joint path holds a value that is not a finite number: {row}"
                )
        if self._proc is None:
            self.start()
        for index, row in enumerate(rows):
            if len(row) != len(self.joint_names):
                raise ValueError(
                    f"sample {index} of the joint path has {len(row)} joints and the sidecar plans "
                    f"{len(self.joint_names)}"
                )
        # perf_counter rather than monotonic: a check takes a few milliseconds, and monotonic
        # ticks in 15.6 ms steps on Windows, which logs most checks as 0 ms.
        started = time.perf_counter()
        checked = 0
        for offset in range(0, len(rows), MAX_CHECK_CONFIGURATIONS):
            batch = rows[offset : offset + MAX_CHECK_CONFIGURATIONS]
            want = self._send({"cmd": "check_js", "joints": batch, **asked})
            msg = self._recv(_PLAN_TIMEOUT_S, want=want)
            if msg is None:
                raise CuroboUnavailableError(
                    "the cuRobo sidecar gave no verdict on the joint path: it did not answer in time "
                    "or it exited"
                )
            verdict = _verdict_from_reply(msg, sent=len(batch), clearance_m=clearance_m)
            checked += verdict.checked
            if not verdict.valid:
                # Stop here: the arm is not taking this path, so the rest of it is not a
                # question. The index is restated over the whole path, because a sample
                # number counted from the start of a batch points at a configuration nobody
                # can find.
                first = None if verdict.first_invalid is None else verdict.first_invalid + offset
                took_ms = (time.perf_counter() - started) * 1000.0
                refused = JointCheckVerdict(
                    valid=False,
                    first_invalid=first,
                    checked=checked,
                    reason=(
                        f"the cuRobo check refuses sample {first} of {len(rows)}, counted from 0: "
                        f"a joint limit, a self collision or the planner's world"
                    ),
                    # Restated over the whole path, the verdict is a new object, so the typed
                    # reason is carried over by name: it is the only thing here that says
                    # which two links touched.
                    refusal=verdict.refusal,
                )
                # A verdict, and the caller turns it into no motion, so this is where the
                # reason stays.
                logger.warning(
                    "the cuRobo check refused a joint path after %.0f ms: %s%s", took_ms, refused.reason,
                    f". {refused.refusal.render()}" if refused.refusal is not None else "",
                )
                return refused
        took_ms = (time.perf_counter() - started) * 1000.0
        logger.debug("checked %d joint configuration(s) in %.0f ms: all pass", checked, took_ms)
        return JointCheckVerdict(
            valid=True, first_invalid=None, checked=checked,
            reason=f"all {checked} samples pass the cuRobo check",
        )

    def explain_joints(
        self, configs: "Sequence[Sequence[float]]", *, name_pairs: bool = True,
    ) -> SelfExplanation:
        """Ask the sidecar what it finds in each configuration, without asking it to plan.

        It is the same judgement as :meth:`check_joints`, reported per configuration instead
        of as one verdict, and with the pair of links named where the loaded descriptor says
        which link owns which sphere. It is batched at the cap of 1000, as a check is, and
        joined in the order sent, so the matrix gate reads the row for a pose by its index.

        ``name_pairs=False`` asks for the two verdicts alone. Naming a pair is pairwise
        arithmetic over every sphere of the robot, per configuration, on the CPU: 8.29 ms a
        pose on a ur5 with the EGU-50, at 590 spheres and 173,755 pairs, which is 51 minutes
        over one candidate family of the retract rule. That rule reads ``self_collides`` and
        ``bound_ok`` and nothing else. The answer then carries ``UNSET`` rather than a row of
        ``None``, because a ``None`` there is a real answer.

        Raises ``CuroboUnavailableError`` whenever any batch came back without judging every
        configuration in it.
        """
        rows = [[float(value) for value in row] for row in configs]
        if not rows:
            raise ValueError("explain_joints was given no configuration, and there is nothing to explain")
        for index, row in enumerate(rows):
            if not all(math.isfinite(value) for value in row):
                raise ValueError(f"configuration {index} holds a value that is not a finite number: {row}")
        if self._proc is None:
            self.start()
        for index, row in enumerate(rows):
            if len(row) != len(self.joint_names):
                raise ValueError(
                    f"configuration {index} has {len(row)} joints and the sidecar plans {len(self.joint_names)}"
                )
        empty: Any = () if name_pairs else UNSET
        explained = SelfExplanation(self_collides=(), bound_ok=(), pairs=empty, depths_mm=empty)
        for offset in range(0, len(rows), MAX_CHECK_CONFIGURATIONS):
            batch = rows[offset : offset + MAX_CHECK_CONFIGURATIONS]
            want = self._send({"cmd": "explain_js", "joints": batch, "name_pairs": name_pairs})
            msg = self._recv(_PLAN_TIMEOUT_S, want=want)
            if msg is None:
                raise CuroboUnavailableError(
                    "the cuRobo sidecar did not explain the configurations it was sent: it did not answer in time "
                    "or it exited"
                )
            explained = explained.joined(
                SelfExplanation.from_reply(msg, sent=len(batch), named=name_pairs))
        return explained

    def set_world(self, cuboids: list[dict], meshes: "list[dict] | None" = None) -> int:
        """Replace cuRobo's collision world with these obstacles (the scene->planner world-model).

        Each cuboid is ``{"name": str, "dims_m": [x,y,z], "pose": [px,py,pz,qw,qx,qy,qz]}``
        in the base frame, in metres, with a WXYZ quaternion. Each mesh is
        ``{"name": str, "file_path": str, "pose": [...]}`` with an optional
        ``"scale": [sx,sy,sz]``, and the sidecar reads the file itself: a tote is tens of
        thousands of triangles and this is a line-based JSON pipe.

        A mesh is how a container reaches the planner as the shape it is rather than as a
        solid block. Meshes need slots reserved at spawn through ``mesh_cache``; without
        them the sidecar has nowhere to put one and says so.

        It returns the count registered, or 0 on failure. It replaces rather than extends,
        so everything the planner must keep has to be in this one call.
        """
        if self._proc is None:
            self.start()
        request = {"cmd": "set_world", "cuboids": list(cuboids)}
        if meshes:
            request["meshes"] = list(meshes)
        want = self._send(request)
        msg = self._recv(_PLAN_TIMEOUT_S, want=want)
        if msg and msg.get("world_set") is not None:
            count = int(msg["world_set"])
            sent = len(cuboids) + len(meshes or ())
            # An obstacle the planner never received is one it will route straight
            # through, so the count registered rather than the count sent is what
            # matters.
            logger.info("collision world set: %d of %d obstacle(s) registered", count, sent)
            return count
        logger.error(
            "the cuRobo sidecar did not confirm the collision world (%d cuboid(s) sent); planning "
            "continues against the PREVIOUS world",
            len(cuboids),
        )
        return 0

    def set_voxels(
        self,
        path: "str | None",
        *,
        dims_m: "Sequence[float]" = (),
        voxel_size_m: float = 0.0,
        pose: "Sequence[float]" = (),
    ) -> "int | None":
        """Register the live scene as a distance field over a grid, or clear it with ``path=None``.

        This is the channel that carries a whole cell. The cuboid channel carries as many
        obstacles as there are slots and the caller then has to choose which ones matter; a
        grid carries everything the cameras saw, at the resolution it was cut to, and
        nothing is left out.

        ``path`` names a NumPy file holding the field as one flat array in the planner's
        own voxel order. It is a file rather than numbers in the request because a 30 mm
        grid over a 2 m cell is 179,560 values, which is not something to send down this
        pipe before every motion.

        The field is a signed distance in metres, negative inside an obstacle and positive
        in free space. Written positive inside, every voxel that is not an obstacle reads as
        inside one and the whole grid blocks, so a wall looks seen when the cell is, as
        ``scripts/curobo/probe_live_world.py`` measures.
        :mod:`src.robot.safety.planning.live_world` writes the file in this sign.

        It returns the number of values registered, or ``None`` where the sidecar refused,
        which is the state a caller must treat as no world at all.
        """
        if self._proc is None:
            self.start()
        request: dict = {"cmd": "set_voxels", "path": path}
        if path:
            request.update(
                {"dims_m": list(dims_m), "voxel_size_m": float(voxel_size_m), "pose": list(pose)}
            )
        want = self._send(request)
        msg = self._recv(_PLAN_TIMEOUT_S, want=want)
        if msg and msg.get("voxels_set") is not None:
            return int(msg["voxels_set"])
        reason = (msg or {}).get("reason", "no reply")
        logger.error(
            "the cuRobo sidecar did not register the live scene (%s); planning continues against "
            "the PREVIOUS world", reason,
        )
        return None

    def set_scene(
        self,
        cuboids: "Sequence[dict]",
        meshes: "Sequence[dict] | None",
        voxels: "dict | None",
    ) -> SceneRegistration:
        """Replace the collision world with boxes, meshes and a field in one request.

        ``voxels`` is ``{"path", "dims_m", "voxel_size_m", "pose"}`` for a field on disk, in
        the sign and unit :meth:`set_voxels` states, or None for a world with no field.

        It is one request because two are not atomic: the field request rebuilds the world
        from the boxes the sidecar remembered, so a declared mesh vanished every time a
        camera produced a field, and both requests reported success. A field the sidecar
        cannot hold comes back with its reason while the boxes and meshes stay registered.
        """
        if self._proc is None:
            self.start()
        request: dict[str, Any] = {
            "cmd": "set_world",
            "cuboids": list(cuboids),
            "meshes": list(meshes or ()),
            "voxels": None if voxels is None else dict(voxels),
        }
        want = self._send(request)
        msg = self._recv(_PLAN_TIMEOUT_S, want=want)
        if not msg or msg.get("world_set") is None:
            why = (msg or {}).get("reason") or "no reply"
            logger.error(
                "the cuRobo sidecar did not confirm the scene (%s); planning continues against the "
                "PREVIOUS world", why,
            )
            return SceneRegistration(
                world_set=0, voxels_set=None, reason=f"the sidecar did not confirm the scene ({why})"
            )
        count = int(msg["world_set"])
        taken = msg.get("voxels_set")
        voxels_set = None if taken is None else int(taken)
        reason = str(msg.get("reason") or "")
        if voxels is not None and voxels_set is None and not reason:
            reason = "the sidecar did not say whether it took the field"
        field_word = "none" if voxels is None else (
            "REFUSED" if voxels_set is None else f"{voxels_set} value(s)"
        )
        logger.info(
            "scene set: %d of %d obstacle(s) registered, field %s",
            count, len(cuboids) + len(meshes or ()), field_word,
        )
        return SceneRegistration(world_set=count, voxels_set=voxels_set, reason=reason)

    def reserve_world(self, reservation: "PlannerReservation") -> None:
        """Reserve what the planner allocates at start: box slots, mesh slots, grid, spheres.

        It takes effect only before :meth:`start`, because cuRobo allocates its collision
        storage once. The reservation is derived from the cell's config and wins over the
        three environment variables a shell can set, and a disagreeing one is named and
        ignored: a sidecar sized by a stale exported variable refuses the cell's own world
        with nothing in the config to explain why.
        """
        if self._proc is not None:
            logger.warning(
                "reserve_world after the sidecar started: its collision storage is already allocated, "
                "so this has no effect until the next start"
            )
            return
        wanted = {
            ENV_CUROBO_CUBOID_CACHE: str(reservation.cuboid_slots),
            ENV_CUROBO_MESH_CACHE: str(reservation.mesh_slots),
            ENV_CUROBO_VOXEL_GRID: reservation.voxel_grid,
        }
        for name, value in wanted.items():
            shell = (os.environ.get(name) or "").strip()
            if shell and shell != value:
                logger.warning(
                    "%s=%r disagrees with this cell's planner reservation (%r) and is ignored: the config "
                    "decides what the planner allocates", name, shell, value or "none",
                )
        self._scene = str(reservation.cuboid_slots)
        self._mesh_cache = int(reservation.mesh_slots)
        self._voxel_grid = reservation.voxel_grid
        if reservation.sphere_slots > 0:
            self._attach_spheres = int(reservation.sphere_slots)
        self._reserved = True

    def _sidecar_env(self) -> dict[str, str]:
        """The environment the sidecar is spawned with.

        The sidecar reads the mesh and grid variables itself. Once a reservation decided
        them, both are written here whatever the shell holds, a mesh count of 0 and the
        absence of a grid included, because an inherited stale variable would size the
        planner behind the config's back.
        """
        env = dict(os.environ)
        # The margin and the payload slots are written from this client whatever the shell
        # holds, and removed where it asked for neither: nothing in this repository sets
        # either variable, so a value in the shell is a leftover, and inheriting one would
        # give a cell a clearance or a payload its config never asked for.
        if self._self_collision_margin_mm > 0.0:
            env[ENV_SELF_COLLISION_MARGIN_MM] = repr(self._self_collision_margin_mm)
        else:
            env.pop(ENV_SELF_COLLISION_MARGIN_MM, None)
        if self._attach_spheres > 0:
            env[ENV_ATTACH_SPHERES] = str(self._attach_spheres)
        else:
            env.pop(ENV_ATTACH_SPHERES, None)
        if self._measure_only:
            env[ENV_MEASURE_ONLY] = "1"
        else:
            env.pop(ENV_MEASURE_ONLY, None)
        # Written from this client whatever the shell holds, for the same reason as the
        # bodies: an inherited pose would start the arm somewhere this cell never chose.
        if self._default_q is not None:
            env[ENV_DEFAULT_Q] = json.dumps(self._default_q)
        else:
            env.pop(ENV_DEFAULT_Q, None)
        # Written from this client whatever the shell holds: an inherited value would add a
        # hand nobody sent.
        if self._body_links:
            env[ENV_BODY_LINKS] = json.dumps(self._body_links, sort_keys=True)
        else:
            env.pop(ENV_BODY_LINKS, None)
        # The same for a wrist camera: an inherited value would load a housing this cell does not carry.
        if self._wrist_body_links:
            env[ENV_WRIST_BODY_LINKS] = json.dumps(self._wrist_body_links, sort_keys=True)
        else:
            env.pop(ENV_WRIST_BODY_LINKS, None)
        if self._reserved:
            env[ENV_CUROBO_MESH_CACHE] = str(self._mesh_cache)
            if self._voxel_grid:
                env[ENV_CUROBO_VOXEL_GRID] = self._voxel_grid
            else:
                env.pop(ENV_CUROBO_VOXEL_GRID, None)
            return env
        if self._mesh_cache > 0:
            env[ENV_CUROBO_MESH_CACHE] = str(self._mesh_cache)
        if self._voxel_grid:
            env[ENV_CUROBO_VOXEL_GRID] = self._voxel_grid
        return env

    def reserve_attach_spheres(self, slots: int) -> None:
        """Reserve payload collision spheres. Only takes effect before :meth:`start`."""
        if self._proc is not None:
            logger.warning(
                "reserve_attach_spheres(%d) after the sidecar started: the robot config is already "
                "built, so this has no effect until the next start", slots,
            )
            return
        self._attach_spheres = max(0, int(slots))

    def attach_payload(
        self,
        joints: list[float],
        dims_m: "Sequence[float]",
        pose: "Sequence[float]",
        *,
        name: str = "payload",
    ) -> bool:
        """Hang a box on the tool so later plans route the carried part around the world too.

        ``dims_m`` are full side lengths, and ``pose`` is ``[x, y, z, qw, qx, qy, qz]``
        in the tool frame. The hand's placement says which tool0 axis it approaches along,
        +Y on the Isaac cell and +Z on a UR declaring its own flange axis, and
        ``self_envelope.carried_part_box`` puts the part's centre past the fingertips on
        that axis. ``joints`` is the configuration the part was grasped in, which is where
        the attachment is fitted.

        It returns ``False`` where the sidecar could not attach, including the case of a
        sidecar started without a sphere budget, which therefore has no link to hang
        anything from. That is deliberately not an exception: a cell that cannot model
        its payload carries on planning without it and says so rather than stopping
        mid-pick.
        """
        if self._proc is None:
            self.start()
        want = self._send({"cmd": "attach", "joints": list(joints), "dims_m": list(dims_m),
                           "pose": list(pose), "name": name})
        msg = self._recv(_PLAN_TIMEOUT_S, want=want)
        if msg and msg.get("attached"):
            logger.info("payload attached to the planner model: dims_m=%s", list(dims_m))
            return True
        logger.error(
            "the cuRobo sidecar did not attach the payload (%s); it is planning as if the gripper "
            "were EMPTY", (msg or {}).get("reason", "no response"),
        )
        return False

    def detach_payload(self) -> bool:
        """Take the carried box off the planner's model. ``False`` if the sidecar did not confirm."""
        if self._proc is None:
            return True  # nothing was ever attached
        want = self._send({"cmd": "detach"})
        msg = self._recv(_PLAN_TIMEOUT_S, want=want)
        if msg and msg.get("detached"):
            return True
        logger.error(
            "the cuRobo sidecar did not detach the payload (%s); it will keep planning around a part "
            "the gripper no longer holds", (msg or {}).get("reason", "no response"),
        )
        return False

    def fk(self, joints: list[float]) -> tuple[list[float], list[float]] | None:
        """Tool0 FK of ``joints`` in radians, as (pos_m, quat_wxyz), or ``None`` on failure."""
        if self._proc is None:
            self.start()
        want = self._send({"cmd": "fk", "joints": list(joints)})
        msg = self._recv(_PLAN_TIMEOUT_S, want=want)
        if msg and msg.get("fk_pos_m") is not None:
            return msg["fk_pos_m"], msg["fk_quat_wxyz"]
        return None

    def __enter__(self) -> "CuroboPlanClient":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
