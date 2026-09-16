"""The cuRobo planning server, a process-isolated sidecar run by the cuRobo python 3.10 environment.

It is never imported by this package. A separate process is needed because cuRobo wants
warp 1.14 and Isaac ships warp 1.8.2, and one python process holds one ``warp``, so
cuRobo cannot run inside the Isaac python 3.11 process. The client
:class:`src.robot.safety.planning.curobo_client.CuroboPlanClient` spawns this script
with the python of the cuRobo environment and speaks newline-delimited JSON over stdio.
The planner, the kinematics and the JIT kernels load once during ``warmup`` and the
server then serves many plan requests, because a subprocess per call would re-warm and
be far too slow.

This file is python 3.10 cuRobo code and is never imported by the python 3.11 package,
so its cuRobo imports carry ``type: ignore`` for a mypy that has no cuRobo. Run it
through the client, or standalone:

    <cuRobo-env-python> -m ... curobo_planner_server.py [robot.yml] [cuboid slots]

Protocol, one JSON object per line:
  startup -> {"status":"ready","joint_names":[...],"default_q":[...],"dt":float,"start_pos_m":[...],
              "start_quat_wxyz":[...],"descriptor":{...}|null,"arm_descriptor_sha256":hex,
              "urdf_sha256":hex|null,"composed_sha256":hex,"bodies":[...],"measure_only":true?,"refusal":{...}?}
              |   {"status":"error","reason":str,"refusal":{...}?}

The robot config is loaded once and composed as one dict by ``_curobo_body_links``: the
descriptor named on the command line, the guard's margin, then a payload link. The two
sha256 fields over configs are canonical JSON, and the URDF one is over the file the
kinematics resolved with line endings normalised.
  request <- {"start_joints":[6 rad],"goal_pos_m":[x,y,z],"goal_quat_wxyz":[w,x,y,z]}
             {"cmd":"fk","joints":[6 rad]}   |   {"cmd":"shutdown"}
             {"cmd":"check_js","joints":[[6 rad],...]}   |   {"cmd":"explain_js","joints":[[6 rad],...]}
             {"cmd":"set_world","cuboids":[...],"meshes":[...],"voxels":{"path","dims_m","voxel_size_m","pose"}|null}
             {"cmd":"set_voxels","path":str|null,"dims_m":[...],"voxel_size_m":float,"pose":[...]}
  reply   -> {"success":bool,"trajectory":[[6 rad]...],"dt":float}  |  {"success":false,"reason":str}
             {"fk_pos_m":[...],"fk_quat_wxyz":[...]}
             check_js: {"success":true,"valid":bool,"first_invalid":int|null,"checked":int,"refusal":{...}?}
                       |  {"success":false,"planner_error":true,"reason":str}
             explain_js: {"success":true,"self_collides":[bool],"bound_ok":[bool],"pairs":[[a,b]|null],
                          "depths_mm":[float|null]}  |  {"success":false,"planner_error":true,"reason":str}
             set_world: {"world_set":int|null,"voxels_set":int|null,"reason":str}
             set_voxels: {"voxels_set":int|null,"reason":str}

A field is a signed distance in metres over the grid reserved at start, negative inside an
obstacle and positive in free space, in the planner's voxel order. Written the other way
round, every voxel that is not an obstacle reads as inside one and the whole grid blocks,
as ``scripts/curobo/probe_live_world.py`` measures.

A refusal block is {"where":"default_q"|"start"|"goal", "kind":"self_collision"|"joint_limit"|"world",
"joints":[6 rad], "pair":[link,link]?, "depth_mm":float?}. The pair is named from sphere
ownership in the config this sidecar loaded, through ``_curobo_pairs``; a descriptor that
resolves its spheres from a file carries no ownership, and the refusal is then unnamed
rather than absent. default_q is judged before the ready line and a refusal exits the
process, unless WILLY_CUROBO_MEASURE_ONLY is set, which keeps a sidecar up to be questioned
and refuses every plan.

check_js judges every configuration of a joint path, in joint_names order, against the
joint limits, the robot itself and the planner's world, on the planner's own collision
spheres so that an attached payload counts. A sample passes when those three terms sum to
exactly 0, and first_invalid is the 0-based index of the first sample that does not. A
sidecar older than check_js has no such branch: the request falls into the plan branch,
fails on the missing start_joints and answers planner_error, which the client reports as a
sidecar to restart.

The goal pose is the tool0 pose, which Lula calls the EE pose, in metres in the base
frame with a WXYZ quaternion. The caller maps its grasp TCP onto tool0 and converts
units and quaternion order before sending. The trajectory comes back in ``joint_names``
order.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from typing import Any

ROBOT = sys.argv[1] if len(sys.argv) > 1 else "ur5e.yml"
# Reliability for a constrained query, such as a tight bin. cuRobo plan_pose is
# stochastic, on random IK and trajopt seeds, and the default max_attempts=5 is flaky
# on a hard pick: a collision-free plan provably exists, since the same query measures
# as none and then as a plan across runs, and the seeds sometimes miss it. More
# attempts, each a fresh IK and trajopt seed batch, find it reliably. Keep
# enable_graph_attempt=1, the cuRobo default, so the first attempt is trajopt-only: the
# graph seeder can return no seed for a tight final approach and would otherwise skip
# every attempt. Both are overridable through the environment.
_PLAN_MAX_ATTEMPTS = int(os.environ.get("WILLY_CUROBO_MAX_ATTEMPTS", "16"))
_PLAN_GRAPH_FROM = int(os.environ.get("WILLY_CUROBO_GRAPH_FROM_ATTEMPT", "1"))
# How many obstacles of each kind this planner can ever hold. cuRobo allocates its
# collision storage once, at construction, and `collision_cache` is the supported way to
# say how much: the alternative this replaced was booting fifteen far-away placeholder
# cuboids purely to make the initial scene big enough, which reserved cuboids and nothing
# else.
#
# Reserving a kind is what makes it available at all. A mesh sent to a planner built with
# no mesh storage has nowhere to go, and a voxel grid likewise, so the two env vars below
# are the difference between a channel existing and not.
CUBOID_CACHE = int(sys.argv[2]) if len(sys.argv) > 2 else 16
MESH_CACHE = int(os.environ.get("WILLY_CUROBO_MESH_CACHE", "0") or 0)
#: `x,y,z,voxel` in metres: the size of the volume a live scene can occupy and how finely
#: it is cut. Unset leaves the planner with no voxel storage, which changes nothing.
VOXEL_GRID = os.environ.get("WILLY_CUROBO_VOXEL_GRID", "").strip()


#: The id of the request being answered. Without it a late reply is indistinguishable
#: from the right one, because this is request and response over one pipe: the client
#: sends, then reads the next line. When a client call times out it stops waiting while
#: the request is still outstanding, so the reply arrives later and is handed to
#: whoever reads next. On this seam that is a motion planned for a goal nobody asked
#: for, executed on a real arm.
#:
#: It is set once per request from `req["id"]`, stamped onto every reply by `_emit`
#: including the error paths, and left `None` for anything emitted outside the loop,
#: which is the `ready` handshake. One variable rather than a parameter on a dozen
#: `_emit` calls, because a reply that forgot to carry it would be exactly the
#: untraceable case this removes.
_REQUEST_ID: object = None

#: The cuboids and meshes most recently registered, re-sent underneath a voxel grid.
#:
#: Registration replaces the collision world rather than extending it, so a caller that
#: sends a live scene would otherwise delete the bench, the bin and every fixture it
#: declared, with nothing said about it anywhere. Without the meshes here a declared tote
#: vanishes the first time a camera produces a field, and both requests report success.
_LAST_CUBOIDS: dict = {}
_LAST_MESHES: dict = {}

#: The checker behind every judgement, built once at start, as _terms uses it, and kept for
#: the session. It stays None only while the sidecar is still loading: a build that fails is
#: a sidecar that never becomes ready, because the ready gate cannot judge the retract
#: without it.
_CHECKER: Any = None


#: What a measuring sidecar answers every command that would move something.
_MEASURE_ONLY_REASON = (
    "this cuRobo sidecar was started to MEASURE, not to plan: it reports what it finds in a configuration and "
    "refuses every command that would move anything"
)


def _first_refusal(judged: list, wheres: tuple) -> "dict | None":
    """The first refused configuration of a judged set, stamped with where it was, or None when all of them pass."""
    for where, block in zip(wheres, judged):
        if block is not None:
            return dict(block, where=where)
    return None


def _emit(obj: dict) -> None:
    if _REQUEST_ID is not None:
        obj = {**obj, "id": _REQUEST_ID}
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _grid_refusal(voxels: dict) -> str:
    """Why a field cannot go into the grid this planner reserved, or an empty string when it can.

    cuRobo allocates voxel storage once, for exactly the dims and voxel size it was started
    with. A field cut to any other grid registers without an error and puts its geometry
    somewhere the cell is not. The client rounds the reservation to four decimals, so the
    tolerance is 1e-4 m.
    """
    if _collision_cache.get("voxel") is None:
        return ("this planner was started with no voxel storage; set WILLY_CUROBO_VOXEL_GRID before "
                "it starts")
    dims = [float(v) for v in voxels.get("dims_m") or ()]
    size = float(voxels.get("voxel_size_m") or 0.0)
    want_dims = [float(v) for v in _collision_cache["voxel"]["dims"]]
    want_size = float(_collision_cache["voxel"]["voxel_size"])
    fits = len(dims) == 3 and all(abs(a - b) <= 1e-4 for a, b in zip(dims, want_dims))
    if not fits or abs(size - want_size) > 1e-4:
        return (f"this field is {dims} m at {size} m voxels and the planner reserved {want_dims} m at "
                f"{want_size} m voxels, so it cannot be registered; restart the planner with the grid "
                "the cell builds")
    return ""


def _field_block(voxels: dict) -> "tuple[dict[str, Any], int]":
    """The voxel entry of a scene for a field on disk, and how many values it holds."""
    import numpy as _np  # type: ignore[import-not-found]

    values = _np.load(voxels["path"])
    block = {
        "scene": {
            "dims": list(voxels["dims_m"]),
            "pose": list(voxels["pose"]),
            "voxel_size": float(voxels["voxel_size_m"]),
            "feature_tensor": torch.as_tensor(values, dtype=torch.float16, device="cuda"),
        }
    }
    return block, int(values.shape[0])


try:
    import torch  # type: ignore[import-not-found]
    from curobo._src.geom.types import SceneCfg  # type: ignore[import-not-found]
    from curobo.content import get_content_root as _content_root  # type: ignore[import-not-found]
    from curobo.kinematics import Kinematics, KinematicsCfg  # type: ignore[import-not-found]
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg  # type: ignore[import-not-found]
    from curobo.types import GoalToolPose, JointState  # type: ignore[import-not-found]

    # The boot world is a real table and nothing else. It is load-bearing: with no floor,
    # cuRobo plans a contorted path from park to pre-grasp that ends far from the goal.
    # The slots for everything a caller registers later are reserved through
    # `collision_cache` below rather than by filling this scene with placeholders.
    _world = {"table": {"dims": [1.6, 1.6, 0.05], "pose": [0.0, 0.0, -0.026, 1, 0, 0, 0]}}
    _collision_cache: dict = {"cuboid": max(1, CUBOID_CACHE)}
    if MESH_CACHE > 0:
        _collision_cache["mesh"] = MESH_CACHE
    if VOXEL_GRID:
        _dx, _dy, _dz, _vs = (float(v) for v in VOXEL_GRID.split(","))
        _collision_cache["voxel"] = {"layers": 1, "dims": [_dx, _dy, _dz], "voxel_size": _vs}
    from curobo._src.util.config_io import resolve_config  # type: ignore[import-not-found]

    # The composition lives in sibling modules, so it is importable from the python 3.11
    # package as well as from here, and this script's own directory is first on sys.path
    # because the client spawns it by path.
    from _curobo_attach import ATTACHED_LINK_NAME, ENV_ATTACH_SPHERES  # type: ignore[import-not-found]
    from _curobo_body_links import (  # type: ignore[import-not-found]
        ENV_BODY_LINKS,
        ENV_DEFAULT_Q,
        body_report,
        canonical_sha256,
        compose_with_counts,
    )
    from _curobo_margin import ENV_SELF_COLLISION_MARGIN_MM  # type: ignore[import-not-found]
    from _curobo_pairs import SphereLayout, deepest_pairs  # type: ignore[import-not-found]
    from _curobo_protocol import (  # type: ignore[import-not-found]
        ENV_MEASURE_ONLY,
        KIND_JOINT_LIMIT,
        KIND_SELF_COLLISION,
        KIND_WORLD,
        WHERE_DEFAULT_Q,
        WHERE_GOAL,
        WHERE_START,
    )

    # The one config this planner loads. The descriptor is read once, with cuRobo's own
    # loader, and composed in memory by _curobo_body_links: body links, then the guard's
    # margin, then a payload link. The planner, the Kinematics and the check_js checker are
    # each built from their own deep copy, because cuRobo's LinkParams.create rewrites the
    # dict it is handed. cuRobo's loader and yaml.safe_load read all 33 installed
    # descriptors alike.
    _desc_path = ROBOT if os.path.isabs(ROBOT) else os.path.join(str(_content_root()), "configs", "robot", ROBOT)
    _raw = resolve_config(_desc_path)
    if not isinstance(_raw, dict):
        raise ValueError(f"{_desc_path} holds no robot config")
    # Teach the planner the clearance the safety guard will demand, so it stops returning
    # paths the guard was always going to refuse: measured, 9.44 to 9.47 mm plans against a
    # 10.000 mm guard margin. Unset, or 0, leaves the config untouched.
    _margin_mm = float(os.environ.get(ENV_SELF_COLLISION_MARGIN_MM, "0") or 0.0)
    # A link to hang the grasped part from. Without it the planner model ends at the gripper
    # and every move after a successful close is planned as if the hand were empty.
    # `franka.yml` declares one and no UR config does. Unset, or 0, leaves the config
    # untouched, and the attach command then refuses rather than pretending.
    _attach_spheres = int(os.environ.get(ENV_ATTACH_SPHERES, "0") or 0)
    # The bodies the client adds: the hand a cell names, placed by its declared tool frame,
    # as a fixed link under tool0 (safety/planning/body_link.py). An empty list leaves the
    # descriptor as it is, and the client refuses a sidecar that then reports no hand.
    _bodies = json.loads(os.environ.get(ENV_BODY_LINKS) or "[]")
    if not isinstance(_bodies, list) or not all(isinstance(_body, dict) for _body in _bodies):
        raise ValueError(f"{ENV_BODY_LINKS} holds no JSON list of body links")
    # The retract this arm and hand were judged at. The descriptor is per arm, so the pose
    # in it can only be right about a bare arm; the pose for the pair comes from the client,
    # out of the committed table. Unset leaves the descriptor's own, which is what a sidecar
    # with no hand gets.
    _default_q_env = os.environ.get(ENV_DEFAULT_Q)
    _chosen_q = json.loads(_default_q_env) if _default_q_env else None
    if _chosen_q is not None and not isinstance(_chosen_q, list):
        raise ValueError(f"{ENV_DEFAULT_Q} holds no JSON list of joint values")
    _COMPOSED, _margin_links, _attach_added = compose_with_counts(
        _raw, bodies=_bodies, margin_mm=_margin_mm, attach_spheres=_attach_spheres, default_q=_chosen_q,
    )
    if _chosen_q is not None:
        print(f"[retract] {[round(float(v), 4) for v in _chosen_q]} from the cell, for this arm and "
              f"hand", file=sys.stderr, flush=True)
    _body_rows = body_report(_COMPOSED, [str(_body.get("link")) for _body in _bodies])
    for _row in _body_rows:
        print(f"[bodies] {_row['link']} under {_row['parent']}: {_row['spheres']} sphere(s) at "
              f"{_row['fixed_transform']}", file=sys.stderr, flush=True)
    if _margin_mm > 0.0:
        if _margin_links:
            print(f"[margin] +{_margin_mm:g} mm guard clearance on {_margin_links} links", file=sys.stderr, flush=True)
        else:
            # Said out loud, because the planner is about to run without the guard margin
            # and can still hand back configurations the guard refuses.
            print(f"[margin] !! {_desc_path} has no self_collision_buffer block; margin not applied",
                  file=sys.stderr, flush=True)
    if _attach_spheres > 0:
        if _attach_added:
            print(f"[attach] payload link with {_attach_spheres} sphere slot(s)", file=sys.stderr, flush=True)
        else:
            print(f"[attach] {_desc_path} already declares {ATTACHED_LINK_NAME}; using it as it is",
                  file=sys.stderr, flush=True)

    # Who this sidecar is, said once in the ready line. `_provenance` names the arm and hand
    # the descriptor was built for, as build_ur_config.py writes it, and the client refuses a
    # descriptor for another hand. The hashes name the bytes: the descriptor's robot_cfg and
    # the composed config as canonical JSON, and the URDF the kinematics resolved.
    _descriptor = _raw.get("_provenance")
    _arm_descriptor_sha256 = canonical_sha256(_raw.get("robot_cfg", _raw))
    _composed_sha256 = canonical_sha256(_COMPOSED)
    _planner = MotionPlanner(
        MotionPlannerCfg.create(
            robot=copy.deepcopy(_COMPOSED),
            scene_model={"cuboid": _world},
            collision_cache=_collision_cache,
        )
    )
    print(f"[cache] {_collision_cache}", file=sys.stderr, flush=True)
    _planner.warmup(enable_graph=True, num_warmup_iterations=5)
    _DT = float(_planner.trajopt_solver.config.interpolation_dt)
    _N = len(_planner.joint_names)
    _kin = Kinematics(KinematicsCfg.from_data_dict(copy.deepcopy(_COMPOSED)))
    _default_q = _planner.default_joint_state.position.squeeze().cpu().tolist()
    # Line endings normalised, because a URDF written in text mode on Windows carries CRLF and is the same robot.
    _urdf_path = getattr(_kin.config.generator_config, "urdf_path", None)
    _urdf_sha256 = None
    if isinstance(_urdf_path, str) and os.path.isfile(_urdf_path):
        with open(_urdf_path, "rb") as _urdf_file:
            _urdf_sha256 = hashlib.sha256(_urdf_file.read().replace(b"\r\n", b"\n")).hexdigest()

    def _fk(joints: list) -> tuple[list, list]:
        q = torch.tensor(joints, device="cuda", dtype=torch.float32).reshape(1, -1)
        st = _kin.compute_kinematics(JointState.from_position(q, joint_names=_kin.joint_names))
        p = st.tool_poses.get_link_pose(_kin.tool_frames[0])
        return p.position.squeeze().cpu().tolist(), p.quaternion.squeeze().cpu().tolist()

    # One judgement, asked three times: by the ready gate below, by the start and goal checks
    # in front of each plan, and by check_js. Three separate copies of it would be three
    # chances to disagree about what admissible means.
    #
    # The checker is built here rather than on the first check_js, because the ready gate
    # needs it: a sidecar that cannot judge its own retract must not report itself ready. It
    # costs start time and GPU memory. It is built on the planner's own
    # scene_collision_checker, the object update_world reloads, so it judges against
    # whatever world was registered last without being rebuilt.
    from curobo.collision_checking import (  # type: ignore[import-not-found]
        RobotCollisionChecker,
        RobotCollisionCheckerCfg,
    )

    _CHECKER = RobotCollisionChecker(
        RobotCollisionCheckerCfg.load_from_config(
            robot_config=copy.deepcopy(_COMPOSED),
            scene_collision_checker=_planner.scene_collision_checker,
            collision_activation_distance=0.0,
        )
    )
    # Who owns which collision sphere, so a refusal can name two links instead of a number.
    # None for a descriptor that resolves its spheres from a file: the refusal is then just
    # as real and the pair has no name.
    _LAYOUT = SphereLayout.from_robot_config(_COMPOSED)

    def _terms(rows: list) -> tuple:
        """The three costs cuRobo's own validation adds up, per configuration, plus the spheres they were read from.

        The spheres come from the planner's kinematics. A checker keeps a Kinematics of its
        own, and that copy never sees a payload attached to the planner: a part 50 mm inside
        a wall passes it. An activation distance of 0.0 makes a zero mean that the spheres do
        not penetrate, not that they keep any clearance.
        """
        cq = torch.tensor(rows, device="cuda", dtype=torch.float32)
        if cq.ndim != 2 or cq.shape[0] == 0 or cq.shape[1] != _N:
            raise ValueError(f"joints must be a non-empty list of {_N} joint configurations, "
                             f"got shape {tuple(cq.shape)}")
        cq = cq.unsqueeze(0)
        h = int(cq.shape[1])
        spheres = _planner.compute_kinematics(
            JointState.from_position(cq, joint_names=_planner.joint_names)
        ).robot_spheres.reshape(1, h, -1, 4)
        bound = _CHECKER.get_bound(cq).reshape(1, h, -1).sum(dim=-1).reshape(-1)
        self_hit = _CHECKER.get_self_collision(spheres).reshape(1, h, -1).sum(dim=-1).reshape(-1)
        world_hit = _CHECKER.get_collision_constraint(spheres).reshape(1, h, -1).sum(dim=-1).reshape(-1)
        return bound, self_hit, world_hit, spheres.reshape(h, -1, 4)

    def _named_pairs(spheres: Any, count: int) -> list:
        """The deepest overlapping link pair per configuration, or a row of None where nothing can name one."""
        if _LAYOUT is None:
            return [None] * count
        return list(deepest_pairs(spheres.detach().cpu().numpy().astype("float64"), _LAYOUT))

    def _judge_states(rows: list, *, world: bool) -> list:
        """One block per configuration saying what is wrong with it, or None where nothing is.

        The self collision is reported first and named, because it is the one an operator
        cannot see from the outside: it says this arm and this hand do not fit together in
        this pose. The joint bound comes second and the world last, and the world is left out
        where it is not a property of the robot, which is the ready gate.
        """
        bound, self_hit, world_hit, spheres = _terms(rows)
        pairs = _named_pairs(spheres, len(rows))
        found: list = []
        for index, row in enumerate(rows):
            joints = [float(value) for value in row]
            if float(self_hit[index]) > 0.0:
                block: dict = {"kind": KIND_SELF_COLLISION, "joints": joints}
                pair = pairs[index]
                if pair is not None:
                    # Two arithmetics over one set of spheres. Where cuRobo says a pose
                    # collides and this names no pair, the gate counts an attribution
                    # disagreement; the refusal stands either way.
                    block["pair"] = [pair.link_a, pair.link_b]
                    block["depth_mm"] = pair.depth_mm
                found.append(block)
            elif float(bound[index]) > 0.0:
                found.append({"kind": KIND_JOINT_LIMIT, "joints": joints})
            elif world and float(world_hit[index]) > 0.0:
                found.append({"kind": KIND_WORLD, "joints": joints, "depth_mm": float(world_hit[index]) * 1000.0})
            else:
                found.append(None)
        return found

    def _refusal_sentence(block: dict) -> str:
        """The refusal as one line for an operator: what was judged, what it found, and under which margin."""
        pair, depth = block.get("pair"), block.get("depth_mm")
        if block.get("kind") == KIND_SELF_COLLISION:
            reached = (
                f"{pair[0]} and {pair[1]} overlap by {depth:.1f} mm" if pair and depth is not None
                else "the robot's own collision spheres overlap, and this descriptor does not say which link owns "
                     "which sphere, so the pair has no name"
            )
        elif block.get("kind") == KIND_JOINT_LIMIT:
            reached = "a joint sits outside the limits this planner was built with"
        else:
            reached = f"it reaches {depth:.1f} mm into the world the planner holds" if depth is not None else (
                "it reaches into the world the planner holds")
        return (f"cuRobo refuses {block.get('where')}: {reached}. Descriptor {os.path.basename(_desc_path)}, "
                f"planner margin {_margin_mm:g} mm.")

except Exception as exc:  # noqa: BLE001 (any import/load/JIT failure -> a typed error line, then exit)
    _emit({"status": "error", "reason": f"{type(exc).__name__}: {exc}"})
    sys.exit(1)

_sp, _sq = _fk(_default_q)

# The ready gate. A planner whose own retract is inside the arm or inside the hand has
# nothing safe to say about any other pose, and the escape it would otherwise plan is the
# first motion of every pick. So it is judged here, and a sidecar that refuses it says which
# links touch and exits instead of becoming ready. The world is left out: the boot table is
# replaced by the cell's, and it is not a property of this arm and hand.
#
# Only the matrix gate sets MEASURE_ONLY, which keeps the sidecar up to be questioned and
# never to plan. Without it an arm and hand that refuse at exactly this line could not be
# measured at all.
_MEASURE_ONLY = bool(os.environ.get(ENV_MEASURE_ONLY))
try:
    _READY_REFUSAL = _judge_states([_default_q], world=False)[0]
except Exception as _exc:  # noqa: BLE001 (wrapped, because a traceback here is a client that says only "no signal")
    _emit({"status": "error", "reason": f"the sidecar could not judge its own retract, so it cannot say whether this "
                                        f"arm and hand fit together: {type(_exc).__name__}: {_exc}"})
    sys.exit(1)
if _READY_REFUSAL is not None:
    _READY_REFUSAL = dict(_READY_REFUSAL, where=WHERE_DEFAULT_Q)
    print(f"[ready] {_refusal_sentence(_READY_REFUSAL)}", file=sys.stderr, flush=True)
    if not _MEASURE_ONLY:
        _emit({"status": "error", "reason": _refusal_sentence(_READY_REFUSAL), "refusal": _READY_REFUSAL,
               "descriptor": _descriptor if isinstance(_descriptor, dict) else None,
               "arm_descriptor_sha256": _arm_descriptor_sha256, "composed_sha256": _composed_sha256})
        sys.exit(1)
#: Said in the ready line so a measuring client can read the refusal a driver would have
#: been stopped by, and so a client that wanted a planner refuses this sidecar rather than
#: reading every refusal as a blocked goal.
_MEASURING: dict = {"measure_only": True} if _MEASURE_ONLY else {}
if _MEASURE_ONLY and _READY_REFUSAL is not None:
    _MEASURING["refusal"] = _READY_REFUSAL

_emit({"status": "ready", "joint_names": list(_planner.joint_names), "default_q": _default_q,
       "dt": _DT, "start_pos_m": _sp, "start_quat_wxyz": _sq,
       "descriptor": _descriptor if isinstance(_descriptor, dict) else None,
       "arm_descriptor_sha256": _arm_descriptor_sha256, "urdf_sha256": _urdf_sha256,
       "composed_sha256": _composed_sha256, "bodies": _body_rows, **_MEASURING})

for _line in sys.stdin:
    _line = _line.strip()
    if not _line:
        continue
    try:
        req = json.loads(_line)
    except Exception as exc:  # noqa: BLE001
        _emit({"success": False, "planner_error": True, "reason": f"bad json: {exc}"})
        continue
    # Stamped onto every reply below, including the failure paths, so the client tells
    # an answer to this request from a late answer to the previous one.
    _REQUEST_ID = req.get("id")
    cmd = req.get("cmd")
    if cmd == "shutdown":
        break
    if cmd == "fk":
        try:
            p, q = _fk(req["joints"])
            _emit({"fk_pos_m": p, "fk_quat_wxyz": q})
        except Exception as exc:  # noqa: BLE001
            _emit({"fk_pos_m": None, "reason": f"{type(exc).__name__}: {exc}"})
        continue
    if cmd == "plan_js":
        # Plan to a joint configuration, collision-free, which cuRobo calls plan_cspace.
        #
        # It exists next to the Cartesian plan because some goals are joint
        # configurations: the park and home poses are stored that way. Reaching them
        # through FK and a Cartesian plan would let cuRobo pick any IK branch that hits
        # the same tool pose, and a flipped elbow branch has put 43 mm of
        # self-penetration into a pose that looked fine. Planning in joint space asks
        # for the configuration that was meant.
        if _MEASURE_ONLY:
            _emit({"success": False, "planner_error": False, "reason": _MEASURE_ONLY_REASON})
            continue
        try:
            # Judged before it is planned. A start the planner cannot leave and a goal it
            # cannot hold both come back as "no collision-free plan", which sends an
            # operator looking at the scene for a reason that is in the arm.
            _refused = _first_refusal(
                _judge_states([req["start_joints"], req["goal_joints"]], world=True), (WHERE_START, WHERE_GOAL),
            )
            if _refused is not None:
                print(f"[plan_js] {_refusal_sentence(_refused)}", file=sys.stderr, flush=True)
                _emit({"success": False, "planner_error": False,
                       "reason": _refusal_sentence(_refused), "refusal": _refused})
                continue
            q0 = torch.tensor(req["start_joints"], device="cuda", dtype=torch.float32).unsqueeze(0)
            qg = torch.tensor(req["goal_joints"], device="cuda", dtype=torch.float32).unsqueeze(0)
            start = JointState.from_position(q0, joint_names=_planner.joint_names)
            goal = JointState.from_position(qg, joint_names=_planner.joint_names)
            # Graph seeding, set explicitly, for the reason the Cartesian branch uses
            # it: a joint-space goal can be most of a turn away, since park is a pi
            # shoulder swing from anywhere in the workspace, and pure trajectory
            # optimisation from a straight-line seed has no way around an obstacle in
            # the middle. The graph search does.
            result = _planner.plan_cspace(
                goal, start, max_attempts=_PLAN_MAX_ATTEMPTS, enable_graph_attempt=_PLAN_GRAPH_FROM,
            )
            ok = result is not None and bool(result.success.any())
            print(f"[plan_js] goal={[round(x, 3) for x in req['goal_joints']]} -> "
                  f"{'OK' if ok else 'FAIL'}", file=sys.stderr, flush=True)
            if not ok:
                _emit({"success": False, "planner_error": False,
                       "reason": "no collision-free joint-space plan"})
                continue
            # reshape(-1, N) as the Cartesian branch does, because position carries a
            # leading batch dimension and tolist() without it emits [[[q...], [q...]]],
            # one waypoint that is itself the whole path. The caller then iterates once
            # over a list of lists. Measured: a home to park move came back as
            # 1 waypoint for a pi shoulder rotation.
            traj = result.get_interpolated_plan().position.reshape(-1, _N).cpu().tolist()
            _emit({"success": True, "trajectory": traj, "dt": _DT})
        except Exception as exc:  # noqa: BLE001 (report, never take the sidecar down mid-session)
            # planner_error=True means the call failed and the planner never rendered a
            # verdict. The distinction is not cosmetic, as the note on the client
            # plan_joint sets out: an AttributeError swallowed here, from a method name
            # that does not exist in this cuRobo build, reads at the caller as cuRobo
            # refusing the configuration.
            print(f"[plan_cspace] CALL FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            _emit({"success": False, "planner_error": True, "reason": f"{type(exc).__name__}: {exc}"})
        continue
    if cmd == "check_js":
        # Judge a whole joint path in one call. Every configuration, in joint_names order,
        # is held against the joint limits, the robot itself and the world this planner
        # holds, and the reply names the first one refused.
        #
        # The three terms are _terms, the one judgement this sidecar makes, so a path
        # checked here and a start judged before a plan cannot disagree. A sample passes
        # only when their sum is exactly 0: the checker carries the guard margin in its self
        # collision padding and an activation distance of 0.0, so a pass means cuRobo's
        # spheres do not penetrate, not that they keep any clearance.
        try:
            _bound, _self_hit, _world_hit, _spheres = _terms(req["joints"])
            _passes = ((_bound + _self_hit + _world_hit) == 0.0).reshape(-1).cpu().tolist()
            _first = next((i for i, ok in enumerate(_passes) if not ok), None)
            _word = "valid" if _first is None else f"invalid from sample {_first}"
            print(f"[check_js] {len(_passes)} sample(s): {_word}", file=sys.stderr, flush=True)
            _reply: dict = {"success": True, "valid": _first is None,
                            "first_invalid": _first, "checked": len(_passes)}
            if _first is not None:
                # The same judgement again, for the one sample the verdict names: the client
                # reports the pair beside the index, and an index alone points at a
                # configuration nobody can picture.
                _named = _judge_states([req["joints"][_first]], world=True)[0]
                if _named is not None:
                    _reply["refusal"] = dict(_named, where=WHERE_START)
            _emit(_reply)
        except Exception as exc:  # noqa: BLE001
            # The call failed and nothing was judged: labelled planner_error so the client
            # can never read it as a verdict about the path.
            print(f"[check_js] CALL FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            _emit({"success": False, "planner_error": True, "reason": f"{type(exc).__name__}: {exc}"})
        continue
    if cmd == "explain_js":
        # The matrix gate's question: judge these configurations and say what touched in
        # each, rather than answering with one verdict over the path. It exists so the
        # evidence is measured through the sidecar, loading the config exactly as a cell's
        # planner loads it, instead of through a second implementation beside it.
        try:
            _bound, _self_hit, _world_hit, _spheres = _terms(req["joints"])
            # Naming the pair is the expensive half, and not every caller reads it. `_terms`
            # runs on the GPU; `_named_pairs` is pairwise arithmetic over every sphere of the
            # robot, in numpy on the CPU, per configuration. On a ur5 with the EGU-50, at 590
            # spheres, that is 173,755 pairs and 8.29 ms a pose, which is 51 minutes over one
            # candidate family of the retract rule, whose judge reads the two verdicts and
            # throws the names away. So the question says whether it wants them, and the
            # default is yes for the gate that asked first.
            #
            # Where they are not wanted the reply carries a null rather than a row of nulls: a
            # null per entry means there is nothing to name there, which is a statement about
            # the robot, and this is not one.
            _name_pairs = bool(req.get("name_pairs", True))
            _pairs = _named_pairs(_spheres, len(req["joints"])) if _name_pairs else None
            _emit({
                "success": True,
                "self_collides": [float(value) > 0.0 for value in _self_hit],
                "bound_ok": [float(value) == 0.0 for value in _bound],
                "pairs_named": _name_pairs,
                "pairs": ([[pair.link_a, pair.link_b] if pair is not None else None for pair in _pairs]
                          if _pairs is not None else None),
                "depths_mm": ([pair.depth_mm if pair is not None else None for pair in _pairs]
                              if _pairs is not None else None),
            })
        except Exception as exc:  # noqa: BLE001
            print(f"[explain_js] CALL FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            _emit({"success": False, "planner_error": True, "reason": f"{type(exc).__name__}: {exc}"})
        continue
    if cmd == "attach":
        # Hang a box on the tool, so every later plan routes the carried part around
        # the world as well. dims_m are full side lengths, and pose is
        # [x, y, z, qw, qx, qy, qz] in the tool frame. A UR hand approaches along tool0
        # +Y, so the client sends the part's centre past the fingertips on +Y.
        #
        # It is refused while measuring: a payload changes the robot the gate is measuring,
        # and the evidence names none.
        if _MEASURE_ONLY:
            _emit({"success": False, "planner_error": False, "reason": _MEASURE_ONLY_REASON})
            continue
        try:
            if _attach_spheres <= 0:
                _emit({"attached": False,
                       "reason": f"{ENV_ATTACH_SPHERES} was unset, so this robot config has no "
                                 f"{ATTACHED_LINK_NAME} link to attach to"})
                continue
            from curobo._src.geom.types import Cuboid  # type: ignore[import-not-found]

            _aq = torch.tensor(req["joints"], device="cuda", dtype=torch.float32).unsqueeze(0)
            state = JointState.from_position(_aq, joint_names=_planner.joint_names)
            box = Cuboid(name=str(req.get("name", "payload")),
                         pose=list(req["pose"]), dims=list(req["dims_m"]))
            # num_spheres is pinned to the slots the link was derived with. The
            # automatic fit picks a count from the geometry, measured at 13 for an
            # 80x80x120 mm box, and raises rather than degrading where the link has
            # fewer, so the budget is not left to chance.
            _planner.attachment_manager.attach(
                state, [box], link_name=ATTACHED_LINK_NAME, num_spheres=_attach_spheres,
            )
            print(f"[attach] {req['dims_m']} m at {req['pose']}", file=sys.stderr, flush=True)
            _emit({"attached": True})
        except Exception as exc:  # noqa: BLE001 (report, never take the sidecar down mid-session)
            print(f"[attach] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            _emit({"attached": False, "reason": f"{type(exc).__name__}: {exc}"})
        continue
    if cmd == "detach":
        try:
            _planner.attachment_manager.detach(link_name=ATTACHED_LINK_NAME)
            _emit({"detached": True})
        except Exception as exc:  # noqa: BLE001
            print(f"[detach] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            _emit({"detached": False, "reason": f"{type(exc).__name__}: {exc}"})
        continue
    if cmd == "set_voxels":
        # Register the live scene as a distance field over a grid, which is the channel
        # that carries a whole cell rather than the eight boxes a slot budget leaves room
        # for.
        #
        # The sign and the unit decide everything. The field is a signed distance in
        # metres, negative inside an obstacle. Measured with probe_live_world: a uniform
        # field over the arm collides at -0.5 and is clear at +0.5, and a plane written
        # this way touches the arm exactly where the same plane as a box does. Written
        # positive inside, every voxel that is not an obstacle reads as inside one, so the
        # whole grid blocks and a wall looks seen when the cell is: it fails safe, and it
        # fails silently.
        #
        # The field arrives as a file path rather than as numbers in this request: a 30 mm
        # grid over a 2 m cell is 179,560 values, and that is not something to send down a
        # JSON pipe once per motion. Reading it here costs about 2 ms.
        try:
            if _collision_cache.get("voxel") is None:
                _emit({"voxels_set": None,
                       "reason": "this planner was started with no voxel storage; set "
                                 "WILLY_CUROBO_VOXEL_GRID before it starts"})
                continue
            # The declared world goes back underneath, meshes included, because
            # registration replaces.
            scene: dict[str, Any] = {"cuboid": _LAST_CUBOIDS or _world}
            if _LAST_MESHES:
                scene["mesh"] = _LAST_MESHES
            path = req.get("path")
            if not path:
                _planner.update_world(SceneCfg.create(scene))
                _emit({"voxels_set": 0})
                continue
            refusal = _grid_refusal(req)
            if refusal:
                _emit({"voxels_set": None, "reason": refusal})
                continue
            scene["voxel"], count = _field_block(req)
            _planner.update_world(SceneCfg.create(scene))
            _emit({"voxels_set": count})
        except Exception as exc:  # noqa: BLE001 (report, never take the sidecar down mid-session)
            print(f"[set_voxels] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            _emit({"voxels_set": None, "reason": f"{type(exc).__name__}: {exc}"})
        continue
    if cmd == "set_world":
        # Replace the cuRobo collision world with the caller obstacles, in the base frame
        # in metres, and with a field when the request carries one, in one update_world.
        #   cuboids: [{"name","dims_m":[x,y,z],"pose":[px,py,pz,qw,qx,qy,qz]}]
        #   meshes:  [{"name","file_path","pose":[...], "scale":[sx,sy,sz] optional}]
        #   voxels:  {"path","dims_m","voxel_size_m","pose"} or null, in the sign and unit
        #            set_voxels states
        #
        # A mesh is how a container reaches the planner as the shape it is. As a box a
        # tote is solid and the cell can never reach into it; as a mesh it keeps its
        # hollow. The file is read here rather than sent through the pipe, because a tote
        # is tens of thousands of triangles and this is a newline-delimited JSON protocol.
        #
        # One request rather than set_world then set_voxels, because the second rebuilds
        # the world from the cuboids it remembered and the meshes are lost in between. A
        # field this planner cannot hold is refused with its reason while the boxes and
        # meshes still go: they are true whatever the camera did, and the caller refuses
        # the motion on the reason.
        try:
            world = {c["name"]: {"dims": list(c["dims_m"]), "pose": list(c["pose"])} for c in req["cuboids"]}
            meshes = req.get("meshes") or []
            mesh_block = {
                m["name"]: {
                    "file_path": m["file_path"],
                    "pose": list(m["pose"]),
                    **({"scale": list(m["scale"])} if m.get("scale") else {}),
                }
                for m in meshes
            }
            world_scene: dict[str, Any] = {"cuboid": world}
            if mesh_block:
                world_scene["mesh"] = mesh_block
            reply: dict[str, Any] = {"world_set": len(world) + len(meshes)}
            voxels = req.get("voxels")
            if voxels is not None:
                refusal = _grid_refusal(voxels)
                if refusal:
                    reply.update({"voxels_set": None, "reason": refusal})
                else:
                    world_scene["voxel"], count = _field_block(voxels)
                    reply.update({"voxels_set": count, "reason": ""})
            _planner.update_world(SceneCfg.create(world_scene))
            # Remembered because registering voxels replaces the world too, and the
            # declared world has to go back underneath them or the bench disappears the
            # moment a camera speaks.
            _LAST_CUBOIDS = world
            _LAST_MESHES = mesh_block
            _emit(reply)
        except Exception as exc:  # noqa: BLE001
            _emit({"world_set": None, "reason": f"{type(exc).__name__}: {exc}"})
        continue
    if _MEASURE_ONLY:
        _emit({"success": False, "planner_error": False, "reason": _MEASURE_ONLY_REASON})
        continue
    try:
        # The start, judged before the plan. There is no goal configuration here to judge:
        # the goal is a tool pose, and which configuration would reach it is what the
        # planner is being asked.
        _refused = _first_refusal(_judge_states([req["start_joints"]], world=True), (WHERE_START,))
        if _refused is not None:
            print(f"[plan] {_refusal_sentence(_refused)}", file=sys.stderr, flush=True)
            _emit({"success": False, "planner_error": False,
                   "reason": _refusal_sentence(_refused), "refusal": _refused})
            continue
        q0 = torch.tensor(req["start_joints"], device="cuda", dtype=torch.float32).unsqueeze(0)
        q_start = JointState.from_position(q0, joint_names=_planner.joint_names)
        goal = GoalToolPose(
            tool_frames=_planner.tool_frames,
            position=torch.tensor([[[[req["goal_pos_m"]]]]], device="cuda", dtype=torch.float32),
            quaternion=torch.tensor([[[[req["goal_quat_wxyz"]]]]], device="cuda", dtype=torch.float32),
        )
        result = _planner.plan_pose(
            goal, q_start, max_attempts=_PLAN_MAX_ATTEMPTS, enable_graph_attempt=_PLAN_GRAPH_FROM,
        )
        ok = result is not None and bool(result.success.any())
        print(f"[plan] goal={[round(x,3) for x in req['goal_pos_m']]} -> "
              f"{'OK' if ok else 'NONE'}", file=sys.stderr, flush=True)
        if ok:
            traj = result.get_interpolated_plan().position.reshape(-1, _N).cpu().tolist()
            _emit({"success": True, "trajectory": traj, "dt": _DT})
        else:
            # result is None where the query is invalid, on unreachable IK or a
            # collision. It is fail-safe: the caller treats it as a timeout and there is
            # no blind motion.
            _emit({"success": False, "planner_error": False,
                   "reason": "plan_pose returned no solution (unreachable / in-collision)"})
    except Exception as exc:  # noqa: BLE001
        print(f"[plan] CALL FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        _emit({"success": False, "planner_error": True, "reason": f"{type(exc).__name__}: {exc}"})
