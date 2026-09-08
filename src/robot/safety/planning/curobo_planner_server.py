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

    <cuRobo-env-python> -m ... curobo_planner_server.py [robot.yml] [scene.yml]

Protocol, one JSON object per line:
  startup -> {"status":"ready","joint_names":[...],"default_q":[...],"dt":float,"start_pos_m":[...],
              "start_quat_wxyz":[...]}   |   {"status":"error","reason":str}
  request <- {"start_joints":[6 rad],"goal_pos_m":[x,y,z],"goal_quat_wxyz":[w,x,y,z]}
             {"cmd":"fk","joints":[6 rad]}   |   {"cmd":"shutdown"}
  reply   -> {"success":bool,"trajectory":[[6 rad]...],"dt":float}  |  {"success":false,"reason":str}
             {"fk_pos_m":[...],"fk_quat_wxyz":[...]}

The goal pose is the tool0 pose, which Lula calls the EE pose, in metres in the base
frame with a WXYZ quaternion. The caller maps its grasp TCP onto tool0 and converts
units and quaternion order before sending. The trajectory comes back in ``joint_names``
order.
"""
from __future__ import annotations

import json
import os
import sys

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

#: The cuboids most recently registered, re-sent underneath a voxel grid.
#:
#: Registration replaces the collision world rather than extending it, so a caller that
#: sends a live scene would otherwise delete the bench, the bin and every fixture it
#: declared, with nothing said about it anywhere.
_LAST_CUBOIDS: dict = {}


def _emit(obj: dict) -> None:
    if _REQUEST_ID is not None:
        obj = {**obj, "id": _REQUEST_ID}
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


try:
    import torch  # type: ignore[import-not-found]
    from curobo._src.geom.types import SceneCfg  # type: ignore[import-not-found]
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
    # Teach the planner the clearance the safety guard will demand, so it stops
    # returning paths the guard was always going to refuse: measured, 9.44 to 9.47 mm
    # plans against a 10.000 mm guard margin. The transform lives in a sibling module,
    # and this script own directory is first on sys.path because the client spawns it by
    # path. Unset, or 0, leaves the config untouched.
    from _curobo_attach import (  # type: ignore[import-not-found]
        ATTACHED_LINK_NAME,
        ENV_ATTACH_SPHERES,
        derive_attach_config_file,
    )
    from _curobo_margin import (  # type: ignore[import-not-found]
        ENV_SELF_COLLISION_MARGIN_MM,
        derive_margin_config_file,
    )

    _ROBOT_IN_USE = ROBOT
    _margin_mm = float(os.environ.get(ENV_SELF_COLLISION_MARGIN_MM, "0") or 0.0)
    if _margin_mm > 0.0:
        import tempfile

        from curobo.content import get_content_root  # type: ignore[import-not-found]

        _src = ROBOT if os.path.isabs(ROBOT) else os.path.join(
            str(get_content_root()), "configs", "robot", ROBOT
        )
        _dst = os.path.join(tempfile.gettempdir(), f"willy_guard{_margin_mm:g}mm_{os.path.basename(_src)}")
        _n = derive_margin_config_file(_src, _dst, _margin_mm)
        print(f"[margin] +{_margin_mm:g} mm guard clearance on {_n} links -> {_dst}", file=sys.stderr, flush=True)
        if _n:
            _ROBOT_IN_USE = _dst
        else:
            # Said out loud, because the planner is about to run without the guard
            # margin and can still hand back configurations the guard refuses.
            print(f"[margin] !! {_src} has no self_collision_buffer block; margin not applied",
                  file=sys.stderr, flush=True)
    # A link to hang the grasped part from. Without it the planner model ends at the
    # gripper and every move after a successful close is planned as if the hand were
    # empty. `franka.yml` declares one and no UR config does, so it is derived here and
    # chained onto whatever the margin step produced. Unset, or 0, leaves the config
    # untouched, and the attach command then refuses rather than pretending.
    _attach_spheres = int(os.environ.get(ENV_ATTACH_SPHERES, "0") or 0)
    if _attach_spheres > 0:
        import tempfile

        from curobo.content import get_content_root  # type: ignore[import-not-found]

        _asrc = _ROBOT_IN_USE if os.path.isabs(_ROBOT_IN_USE) else os.path.join(
            str(get_content_root()), "configs", "robot", _ROBOT_IN_USE
        )
        _adst = os.path.join(tempfile.gettempdir(),
                             f"willy_attach{_attach_spheres}_{os.path.basename(_asrc)}")
        if derive_attach_config_file(_asrc, _adst, spheres=_attach_spheres):
            print(f"[attach] payload link with {_attach_spheres} sphere slot(s) -> {_adst}",
                  file=sys.stderr, flush=True)
            _ROBOT_IN_USE = _adst
        else:
            print(f"[attach] {_asrc} already declares {ATTACHED_LINK_NAME}; using it as it is",
                  file=sys.stderr, flush=True)

    _planner = MotionPlanner(
        MotionPlannerCfg.create(
            robot=_ROBOT_IN_USE,
            scene_model={"cuboid": _world},
            collision_cache=_collision_cache,
        )
    )
    print(f"[cache] {_collision_cache}", file=sys.stderr, flush=True)
    _planner.warmup(enable_graph=True, num_warmup_iterations=5)
    _DT = float(_planner.trajopt_solver.config.interpolation_dt)
    _N = len(_planner.joint_names)
    _kin = Kinematics(KinematicsCfg.from_robot_yaml_file(_ROBOT_IN_USE))
    _default_q = _planner.default_joint_state.position.squeeze().cpu().tolist()

    def _fk(joints: list) -> tuple[list, list]:
        q = torch.tensor(joints, device="cuda", dtype=torch.float32).reshape(1, -1)
        st = _kin.compute_kinematics(JointState.from_position(q, joint_names=_kin.joint_names))
        p = st.tool_poses.get_link_pose(_kin.tool_frames[0])
        return p.position.squeeze().cpu().tolist(), p.quaternion.squeeze().cpu().tolist()

except Exception as exc:  # noqa: BLE001 (any import/load/JIT failure -> a typed error line, then exit)
    _emit({"status": "error", "reason": f"{type(exc).__name__}: {exc}"})
    sys.exit(1)

_sp, _sq = _fk(_default_q)
_emit({"status": "ready", "joint_names": list(_planner.joint_names), "default_q": _default_q,
       "dt": _DT, "start_pos_m": _sp, "start_quat_wxyz": _sq})

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
        try:
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
    if cmd == "attach":
        # Hang a box on the tool, so every later plan routes the carried part around
        # the world as well. dims_m are full side lengths, and pose is
        # [x, y, z, qw, qx, qy, qz] in the tool frame, so [0, 0, h/2, 1, 0, 0, 0] is a
        # part sitting h/2 beyond the flange.
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
        # ⛔ THE SIGN IS THE WHOLE THING, AND THE WRONG ONE FAILS SILENTLY. cuRobo reads a
        # value above minus half a voxel as occupied, so the field is positive INSIDE an
        # obstacle and negative in free space, which is the opposite of the
        # distance-to-obstacle a person would write. Measured both ways against the same
        # wall: positive-inside refused the path, positive-outside registered without an
        # error, reported success, and planned straight through it.
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
            path = req.get("path")
            if not path:
                _planner.update_world(SceneCfg.create({"cuboid": _LAST_CUBOIDS or _world}))
                _emit({"voxels_set": 0})
                continue
            import numpy as _np  # type: ignore[import-not-found]

            field = _np.load(path)
            grid = {
                "scene": {
                    "dims": list(req["dims_m"]),
                    "pose": list(req["pose"]),
                    "voxel_size": float(req["voxel_size_m"]),
                    "feature_tensor": torch.as_tensor(
                        field, dtype=torch.float16, device="cuda"
                    ),
                }
            }
            _planner.update_world(
                SceneCfg.create({"cuboid": _LAST_CUBOIDS or _world, "voxel": grid})
            )
            _emit({"voxels_set": int(field.shape[0])})
        except Exception as exc:  # noqa: BLE001 - report, never take the sidecar down mid-session
            print(f"[set_voxels] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            _emit({"voxels_set": None, "reason": f"{type(exc).__name__}: {exc}"})
        continue
    if cmd == "set_world":
        # Replace the cuRobo collision world with the caller obstacles, in the base frame
        # in metres.
        #   cuboids: [{"name","dims_m":[x,y,z],"pose":[px,py,pz,qw,qx,qy,qz]}]
        #   meshes:  [{"name","file_path","pose":[...], "scale":[sx,sy,sz] optional}]
        #
        # A mesh is how a container reaches the planner as the shape it is. As a box a
        # tote is solid and the cell can never reach into it; as a mesh it keeps its
        # hollow. The file is read here rather than sent through the pipe, because a tote
        # is tens of thousands of triangles and this is a newline-delimited JSON protocol.
        try:
            world = {c["name"]: {"dims": list(c["dims_m"]), "pose": list(c["pose"])} for c in req["cuboids"]}
            scene = {"cuboid": world}
            meshes = req.get("meshes") or []
            if meshes:
                scene["mesh"] = {
                    m["name"]: {
                        "file_path": m["file_path"],
                        "pose": list(m["pose"]),
                        **({"scale": list(m["scale"])} if m.get("scale") else {}),
                    }
                    for m in meshes
                }
            _planner.update_world(SceneCfg.create(scene))
            # Remembered because registering voxels replaces the world too, and the
            # declared boxes have to go back underneath them or the bench disappears the
            # moment a camera speaks.
            _LAST_CUBOIDS = world
            _emit({"world_set": len(world) + len(meshes)})
        except Exception as exc:  # noqa: BLE001
            _emit({"world_set": None, "reason": f"{type(exc).__name__}: {exc}"})
        continue
    try:
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
