"""The physics referee on MuJoCo, so a customer does not need a 47 GB NVIDIA-only simulator.

The shake answers the primary metric: does a grasp the geometry calls valid actually hold. Under
Isaac Sim that answer costs a Windows or Linux machine, an NVIDIA GPU and about 47 GB. Everything
else in this pipeline, rendering included, already runs without it, so the referee was the one step
that forced a workstation, and therefore the one step that decided whether "generate your own data"
is a claim a customer can act on. MuJoCo also runs beside other work on a machine Isaac would
occupy alone.

The interface is copied, the mechanics are not assumed. `PhysicsCell` here answers the same calls
`run_physics_sample` already makes, so nothing above it changes. What it must not do is reimplement
the Isaac cell's decisions from their shapes: each of them was a correction to a measured failure,
and a reimplementation that keeps the shape and loses the reason reintroduces the failure it was
written against. The three that matter are carried deliberately.

The hold test is gravity, not a lift. A gripper moved by a pose command transmits no tangential
force, so a lift scores every grasp as failed for a reason that has nothing to do with the grasp.
Taking the support away instead asks the same question with no gripper motion at all.

Squeeze, do not crush. Commanding full travel against a 40 mm object is 40 mm of over-travel into a
stiff drive, which fires the object sideways out from between the pads. The command stops a few
millimetres inside the object's own width.

The jaws are placed, not driven, between trials. A trial that blows up poisons every trial after it
when the jaw has to travel back from wherever the explosion left it, so the degrees of freedom are
written directly instead.

It must pass the same controls before any number it produces means anything. A harness that cannot
grip a block, that grips thin air, or that stops gripping once a second scene has been built
measures nothing, and all three have happened here. `run_controls` answers the same four questions
the Isaac cell does, and `run_physics_sample` refuses the run on the same conditions.

A second referee is not automatically the same referee. Passing the controls makes this a working
harness; it does not make its verdicts interchangeable with Isaac's. That measurement on the same
grasps does not exist yet, so a corpus shaken here has to say which engine shook it, and
`grasp_physics.jsonl` carries the engine for exactly that reason.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

__all__ = ["MUJOCO_ENGINE", "PhysicsCell"]

logger = logging.getLogger("datagen.physics.mujoco")

MUJOCO_ENGINE: Final[str] = "mujoco"

#: How far back along the approach the gripper starts, in millimetres. Far enough to begin clear of
#: the object; the Isaac cell keeps its own standoff in `datagen.grasps.physics`.
_STANDOFF_MM: Final[float] = 120.0

#: How far inside the object's own width the jaw is commanded. See the module docstring: full travel
#: against a 40 mm object fires the block sideways out from between the pads instead of gripping it.
_SQUEEZE_MM: Final[float] = 6.0

#: A fall of more than this during the hang counts as not held. Generous: the object may settle a
#: little in the pads without the grasp having failed, and a failed grasp falls much further.
_HELD_DROP_MM: Final[float] = 20.0

#: Simulation steps for each phase. The timestep is 2 ms, matching the render engine's settle, so a
#: hang of 400 steps is 0.8 s of gravity, which is several times what a slipping object needs.
_STEPS_APPROACH: Final[int] = 60
_STEPS_CLOSE: Final[int] = 150
_STEPS_HANG: Final[int] = 400
_TIMESTEP: Final[float] = 0.002

#: Jaw geometry, in millimetres, for the reference two-finger hand. Pads are boxes rather than a
#: scanned mesh: the referee's job is to say whether a parallel jaw of this aperture holds, and a
#: detailed pad shape would make the answer about a particular gripper's fillets.
_PAD_MM: Final[tuple[float, float, float]] = (8.0, 20.0, 30.0)
_PALM_BACK_MM: Final[float] = 50.0

#: The aperture of the modelled hand. Named here rather than written into `run_trial`, so the trial
#: and the jaw check read one number instead of two.
_JAW_APERTURE_MM: Final[float] = 85.0

#: How wide the jaw may read before the trial is refused. A jaw that ends far wider than it was
#: commanded was shoved apart by something the model does not contain, and the hold rate falls as
#: the reading grows: those are not grasps that failed, they are physics that left the model and
#: would otherwise be counted anyway.
#:
#: The fix is not a tighter joint range. The slide joints allow 110 mm per finger, well past this
#: hand, and that headroom is the detector: clamping the joints at the aperture would make a blown
#: jaw read exactly 85 mm and the check would go silent on the very trials it exists to catch. The
#: loose range stays; the threshold comes down to the hand.
#:
#: 10 % over the aperture. A position servo is back-driven by a stiff object, so a few millimetres
#: over is real and those trials do hold. The number is taken from the hand so it moves with the
#: hand, rather than fitted to the shape of one corpus.
_JAW_SANE_MM: Final[float] = _JAW_APERTURE_MM * 1.1

#: Millimetres to metres, once, because MuJoCo works in metres and everything above this file is in
#: millimetres. Every conversion in this file goes through it rather than through a literal 0.001.
_MM: Final[float] = 0.001

#: How far the palm may end from its commanded pose before the trial is refused. The weld is a
#: constraint rather than a teleport, so the palm can be shoved: that is what lets the pads hold, and
#: it also means a trial whose gripper was pushed aside measured the harness rather than the grasp.
_GRIPPER_OFFSET_REFUSE_MM: Final[float] = 25.0


@dataclass(frozen=True, slots=True)
class _Body:
    """One simulated solid: its MJCF name and the instance it belongs to."""

    name: str
    instance_id: int
    is_target_candidate: bool


def _quat(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """MJCF wants `w x y z`. Written out rather than imported so the convention is visible here."""
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 0.5 / np.sqrt(trace + 1.0)
        return (0.25 / scale,
                (matrix[2, 1] - matrix[1, 2]) * scale,
                (matrix[0, 2] - matrix[2, 0]) * scale,
                (matrix[1, 0] - matrix[0, 1]) * scale)
    index = int(np.argmax(np.diag(matrix)))
    other = [(1, 2), (2, 0), (0, 1)][index]
    scale = 2.0 * np.sqrt(1.0 + matrix[index, index]
                          - matrix[other[0], other[0]] - matrix[other[1], other[1]])
    values = [0.0, 0.0, 0.0]
    values[index] = 0.25 * scale
    values[other[0]] = (matrix[other[0], index] + matrix[index, other[0]]) / scale
    values[other[1]] = (matrix[other[1], index] + matrix[index, other[1]]) / scale
    w = (matrix[other[1], other[0]] - matrix[other[0], other[1]]) / scale
    return (w, values[0], values[1], values[2])


def _frame(approach: np.ndarray, closing: np.ndarray) -> np.ndarray:
    """A right-handed rotation whose +z is the approach and +x the closing axis.

    Gram-Schmidt against the approach, not the other way round. The approach is the direction the
    gripper travels and a millimetre of error there is a millimetre of depth; the closing axis is
    antipodal and its in-plane rotation is what the labeller sampled, so it is the one that gives.
    """
    z = np.asarray(approach, dtype=np.float64)
    z = z / max(float(np.linalg.norm(z)), 1e-9)
    x = np.asarray(closing, dtype=np.float64)
    x = x - float(x @ z) * z
    norm = float(np.linalg.norm(x))
    if norm < 1e-6:
        # A closing axis parallel to the approach is not a grasp; the caller refuses it, and this
        # branch only exists so the arithmetic cannot produce a NaN frame on the way there.
        x = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x = x - float(x @ z) * z
        norm = float(np.linalg.norm(x))
    x = x / max(norm, 1e-9)
    return np.stack([x, np.cross(z, x), z], axis=1)


class PhysicsCell:
    """One MuJoCo session, answering the same calls the Isaac cell does.

    The gripper is a welded body, not a mocap one, and that is the one place this deliberately
    differs from the Isaac cell's mechanics. A MuJoCo mocap body is teleported and transmits no
    tangential force, which is exactly the failure measured on Isaac. A free body welded to a mocap
    target follows the command through a constraint the solver can push back on, so the pads can
    actually hold. The hold test is still gravity rather than a lift, because that is the question
    the lift was a proxy for and it introduces no gripper motion at all.
    """

    def __init__(self, *, headless: bool = True, mesh_collision: str = "sdf") -> None:
        self._headless = headless
        self._mesh_collision = str(mesh_collision)
        self._mujoco: Any = None
        self._model: Any = None
        self._data: Any = None
        self._bodies: list[_Body] = []
        self._geometry: Any = None
        self._rest: dict[int, np.ndarray] = {}
        self._commanded: np.ndarray | None = None

    def __enter__(self) -> "PhysicsCell":
        import mujoco  # noqa: PLC0415 (the engine half, imported only when a run needs it)

        self._mujoco = mujoco
        return self

    def __exit__(self, *_exc: object) -> None:
        self._model = self._data = None

    # ---------------------------------------------------------------- building

    def _gripper_xml(self) -> str:
        """A parallel jaw: a mocap target, a welded palm, two sliding pads with position drives."""
        pad = [v * _MM for v in _PAD_MM]
        back = _PALM_BACK_MM * _MM
        return (
            '  <body name="target" mocap="true" pos="0 0 1">\n'
            '    <geom type="box" size="0.001 0.001 0.001" contype="0" conaffinity="0" '
            'rgba="0 0 0 0"/>\n'
            "  </body>\n"
            '  <body name="palm" pos="0 0 1">\n'
            '    <freejoint name="palm_free"/>\n'
            f'    <geom name="palm_geom" type="box" size="0.03 0.02 0.01" pos="0 0 {-back:.5f}" '
            'mass="0.5" rgba="0.3 0.3 0.35 1"/>\n'
            '    <body name="finger_l" pos="0 0 0">\n'
            '      <joint name="jaw_l" type="slide" axis="1 0 0" range="-0.11 0" damping="5"/>\n'
            f'      <geom name="pad_l" type="box" size="{pad[0] / 2:.5f} {pad[1] / 2:.5f} '
            f'{pad[2] / 2:.5f}" mass="0.05" friction="1.2 0.02 0.001" rgba="0.8 0.5 0.2 1"/>\n'
            "    </body>\n"
            '    <body name="finger_r" pos="0 0 0">\n'
            '      <joint name="jaw_r" type="slide" axis="-1 0 0" range="-0.11 0" damping="5"/>\n'
            f'      <geom name="pad_r" type="box" size="{pad[0] / 2:.5f} {pad[1] / 2:.5f} '
            f'{pad[2] / 2:.5f}" mass="0.05" friction="1.2 0.02 0.001" rgba="0.8 0.5 0.2 1"/>\n'
            "    </body>\n"
            "  </body>")

    def _solid_xml(self, name: str, solid: Any, *, kinematic: bool) -> str:
        """One object. Meshes go in as their own asset; primitives as MuJoCo's own shapes."""
        centre = np.asarray(solid.centre_mm, dtype=np.float64) * _MM
        w, x, y, z = _quat(np.asarray(solid.rotation, dtype=np.float64))
        half = np.asarray(solid.half_extent_mm, dtype=np.float64) * _MM
        joint = "" if kinematic else f'<freejoint name="{name}_free"/>'
        mass = max(float(getattr(solid, "mass_kg", 0.0)), 0.05)
        friction = 'friction="1.0 0.02 0.001"'
        if solid.kind == "mesh" and solid.mesh is not None:
            geom = (f'<geom name="{name}_geom" type="mesh" mesh="{name}_mesh" '
                    f'mass="{mass:.4f}" {friction}/>')
        elif solid.kind == "cylinder":
            geom = (f'<geom name="{name}_geom" type="cylinder" size="{half[0]:.5f} {half[2]:.5f}" '
                    f'mass="{mass:.4f}" {friction}/>')
        elif solid.kind == "sphere":
            geom = (f'<geom name="{name}_geom" type="sphere" size="{half[0]:.5f}" '
                    f'mass="{mass:.4f}" {friction}/>')
        else:
            geom = (f'<geom name="{name}_geom" type="box" '
                    f'size="{half[0]:.5f} {half[1]:.5f} {half[2]:.5f}" '
                    f'mass="{mass:.4f}" {friction}/>')
        return (f'  <body name="{name}" pos="{centre[0]:.5f} {centre[1]:.5f} {centre[2]:.5f}" '
                f'quat="{w:.6f} {x:.6f} {y:.6f} {z:.6f}">{joint}{geom}</body>')

    def _mesh_assets(self, geometry: Any) -> str:
        """Mesh assets for every solid that has one, keyed by the body name that uses it."""
        chunks: list[str] = []
        for name, solid in self._named_solids(geometry):
            if solid.kind != "mesh" or solid.mesh is None:
                continue
            vertices = np.asarray(solid.mesh.vertices_mm, dtype=np.float64) * _MM
            faces = np.asarray(solid.mesh.faces, dtype=np.int64)
            vertex_text = " ".join(f"{v:.6f}" for v in vertices.reshape(-1))
            face_text = " ".join(str(int(v)) for v in faces.reshape(-1))
            chunks.append(f'    <mesh name="{name}_mesh" vertex="{vertex_text}" '
                          f'face="{face_text}"/>')
        return "\n".join(chunks)

    @staticmethod
    def _named_solids(geometry: Any) -> list[tuple[str, Any]]:
        out = [(f"obj{instance}", solid) for instance, solid in sorted(geometry.objects.items())]
        for index, wall in enumerate(getattr(geometry, "walls", ()) or ()):
            out.append((f"wall{index}", wall))
        return out

    def build_scene(self, geometry: Any, *, kinematic_objects: bool = False) -> None:
        """Assemble the scene and the gripper into one model, and record the settled poses."""
        self._geometry = geometry
        walls = {f"wall{i}" for i in range(len(getattr(geometry, "walls", ()) or ()))}
        bodies = "\n".join(
            self._solid_xml(name, solid, kinematic=kinematic_objects or name in walls)
            for name, solid in self._named_solids(geometry))
        xml = (
            "<mujoco>\n"
            f'  <option timestep="{_TIMESTEP}" integrator="implicitfast" cone="elliptic"/>\n'
            "  <asset>\n" + self._mesh_assets(geometry) + "\n  </asset>\n"
            "  <worldbody>\n"
            '    <geom name="floor" type="plane" size="5 5 0.1" friction="1.0 0.02 0.001"/>\n'
            + bodies + "\n" + self._gripper_xml() + "\n"
            "  </worldbody>\n"
            "  <equality>\n"
            '    <weld name="hold" body1="target" body2="palm" solref="0.005 1" '
            'solimp="0.99 0.999 0.001"/>\n'
            "  </equality>\n"
            "  <actuator>\n"
            '    <position name="drive_l" joint="jaw_l" kp="400" ctrlrange="-0.11 0"/>\n'
            '    <position name="drive_r" joint="jaw_r" kp="400" ctrlrange="-0.11 0"/>\n'
            "  </actuator>\n"
            "</mujoco>")
        self._model = self._mujoco.MjModel.from_xml_string(xml)
        self._data = self._mujoco.MjData(self._model)
        self._mujoco.mj_forward(self._model, self._data)
        self._bodies = [_Body(name, int(name[3:]) if name.startswith("obj") else -1,
                              name.startswith("obj"))
                        for name, _ in self._named_solids(geometry)]
        self._rest = {body.instance_id: self._object_position(body.instance_id).copy()
                      for body in self._bodies if body.is_target_candidate}

    # ---------------------------------------------------------------- primitives

    def _step(self, count: int) -> None:
        for _ in range(count):
            self._mujoco.mj_step(self._model, self._data)

    def _id(self, kind: str, name: str) -> int:
        return int(self._mujoco.mj_name2id(
            self._model, getattr(self._mujoco.mjtObj, kind), name))

    def _object_position(self, instance_id: int) -> np.ndarray:
        """The body's world position in millimetres, which is what every caller above works in."""
        return np.asarray(self._data.xpos[self._id("mjOBJ_BODY", f"obj{instance_id}")],
                          dtype=np.float64) / _MM

    def _place(self, position_mm: np.ndarray, approach: np.ndarray, closing: np.ndarray) -> None:
        """Command the gripper pose by moving the weld target."""
        w, x, y, z = _quat(_frame(approach, closing))
        self._data.mocap_pos[0] = np.asarray(position_mm, dtype=np.float64) * _MM
        self._data.mocap_quat[0] = np.array([w, x, y, z])
        self._commanded = np.asarray(position_mm, dtype=np.float64).copy()

    def _drive_jaw(self, opening_mm: float, steps: int) -> None:
        travel = -max(0.0, opening_mm) * 0.5 * _MM
        self._data.ctrl[0] = self._data.ctrl[1] = travel
        self._step(steps)

    def _teleport_jaw(self, opening_mm: float) -> None:
        """Write the jaw degrees of freedom directly, rather than commanding them.

        A trial that blows up poisons every trial after it when the jaw has to travel back from
        wherever the explosion left it: the drive fights the broken joint and never gets it back
        into range.
        """
        travel = -max(0.0, opening_mm) * 0.5 * _MM
        for name in ("jaw_l", "jaw_r"):
            index = self._id("mjOBJ_JOINT", name)
            self._data.qpos[int(self._model.jnt_qposadr[index])] = travel
            self._data.qvel[int(self._model.jnt_dofadr[index])] = 0.0
        self._data.ctrl[0] = self._data.ctrl[1] = travel

    def _jaw_opening_mm(self) -> float:
        total = 0.0
        for name in ("jaw_l", "jaw_r"):
            index = self._id("mjOBJ_JOINT", name)
            total += abs(float(self._data.qpos[int(self._model.jnt_qposadr[index])]))
        return total / _MM

    def _set_contact(self, names: list[str], *, on: bool) -> None:
        """Turn a geom's collisions off, which is how the support is taken away."""
        for name in names:
            index = self._id("mjOBJ_GEOM", name)
            if index < 0:
                continue
            self._model.geom_contype[index] = 1 if on else 0
            self._model.geom_conaffinity[index] = 1 if on else 0

    # ---------------------------------------------------------------- the contract

    def jaw_fault(self) -> str:
        """Empty when the simulation is still physics, a reason when it has left it."""
        if self._data is None:
            return "no scene has been built"
        if not np.all(np.isfinite(self._data.qpos)) or not np.all(np.isfinite(self._data.qvel)):
            return "the simulation state went non-finite"
        opening = self._jaw_opening_mm()
        if not np.isfinite(opening) or opening > _JAW_SANE_MM:
            return (f"the jaw reads {opening:.1f} mm, past the {_JAW_APERTURE_MM:.0f} mm hand this "
                    f"cell models")
        return ""

    def gripper_offset_mm(self) -> float:
        """How far the palm ended from where it was commanded.

        The weld is a constraint, not a teleport, so the palm can be pushed off its command. That
        is the point, since a teleported gripper transmits no force. It also means a trial whose
        gripper was shoved aside measured the harness rather than the grasp, so the caller refuses it.
        """
        if self._commanded is None or self._data is None:
            return 0.0
        actual = np.asarray(self._data.xpos[self._id("mjOBJ_BODY", "palm")], dtype=np.float64) / _MM
        return float(np.linalg.norm(actual - self._commanded))

    def restore(self, geometry: Any) -> None:
        """Put the scene back where the labeller found it, before every trial."""
        self.build_scene(geometry)
        self._teleport_jaw(_JAW_APERTURE_MM)
        self._mujoco.mj_forward(self._model, self._data)

    def check_ready(self, geometry: Any, instance_id: int) -> str:
        """Empty when the restored scene is the labelled one, a reason when it is not."""
        if self._model is None:
            return "no scene has been built"
        if instance_id not in getattr(geometry, "objects", {}):
            return f"instance {instance_id} is not in this scene"
        if self._id("mjOBJ_BODY", f"obj{instance_id}") < 0:
            return f"instance {instance_id} has no body in the model"
        return ""

    def target_drift_mm(self, geometry: Any, instance_id: int) -> float:
        """How far the target sits from where the labeller recorded it."""
        solid = getattr(geometry, "objects", {}).get(instance_id)
        if solid is None:
            return float("inf")
        return float(np.linalg.norm(
            self._object_position(instance_id) - np.asarray(solid.centre_mm, dtype=np.float64)))

    # ---------------------------------------------------------------- the trial

    def run_trial(self, trial: Any, geometry: Any) -> Any:
        """Approach, close, take the support away, and see whether the object stays up.

        The caller restores the scene before each trial and this refuses itself if the restored scene
        is not the labelled one. A refusal is reported as such and kept out of every rate, because a
        grasp graded against a scene that drifted is not evidence about the grasp.
        """
        from datagen.grasps.physics import TrialOutcome  # noqa: PLC0415 (avoids a cycle)

        position = np.asarray(trial.position_mm, dtype=np.float64)
        approach = np.asarray(trial.approach, dtype=np.float64)
        closing = np.asarray(trial.closing_axis, dtype=np.float64)
        if float(np.linalg.norm(closing)) < 1e-6:
            return TrialOutcome(trial, False, 0.0, "refused: no closing axis recorded")
        if not (np.all(np.isfinite(position)) and np.all(np.isfinite(approach))):
            return TrialOutcome(trial, False, 0.0, "refused: the candidate pose is not finite")
        problem = self.jaw_fault() or self.check_ready(geometry, trial.instance_id)
        if problem:
            return TrialOutcome(trial, False, 0.0, f"refused: {problem}")

        # Measured before the approach. Afterwards the target has, with luck, moved, and a drift read
        # at the end would be reporting the grasp as a fault.
        drift_before = self.target_drift_mm(geometry, trial.instance_id)

        width = float(trial.width_mm) if float(trial.width_mm) > 1.0 else _JAW_APERTURE_MM
        self._teleport_jaw(min(_JAW_APERTURE_MM, width + 12.0))
        self._place(position - approach * _STANDOFF_MM, approach, closing)
        self._step(5)
        for step in range(_STEPS_APPROACH):
            fraction = (step + 1) / _STEPS_APPROACH
            self._place(position - approach * _STANDOFF_MM * (1.0 - fraction), approach, closing)
            self._step(1)

        commanded = max(4.0, width - _SQUEEZE_MM)
        self._drive_jaw(commanded, _STEPS_CLOSE)
        blown = self.jaw_fault()
        if blown:
            return TrialOutcome(trial, False, 0.0, f"refused: {blown}")
        reached = self._jaw_opening_mm()
        start_z = float(self._object_position(trial.instance_id)[2])

        # The hold test is gravity. See the module docstring: a commanded gripper transmits no
        # tangential force, so a lift scores every grasp as failed. Taking the support away asks the
        # same question with no gripper motion.
        support = ["floor"] + [f"obj{body.instance_id}_geom" for body in self._bodies
                               if body.is_target_candidate
                               and body.instance_id != trial.instance_id]
        support += [f"wall{index}_geom" for index in
                    range(len(getattr(geometry, "walls", ()) or ()))]
        self._set_contact(support, on=False)
        self._step(_STEPS_HANG)
        self._set_contact(support, on=True)

        end_z = float(self._object_position(trial.instance_id)[2])
        # Checked again after the hang, not only after the close. The hang is 0.8 s of gravity with
        # the support gone, which is the phase most able to blow the jaw open, and without this an
        # object can be reported as held by a pair of fingers far wider than the hand.
        blown = self.jaw_fault()
        if blown:
            return TrialOutcome(trial, False, 0.0, f"refused: {blown}")
        offset = self.gripper_offset_mm()
        if not (np.isfinite(start_z) and np.isfinite(end_z)):
            return TrialOutcome(trial, False, 0.0,
                                "refused: the object's pose went non-finite during the trial")
        if offset > _GRIPPER_OFFSET_REFUSE_MM:
            return TrialOutcome(trial, False, 0.0,
                                f"refused: the gripper ended {offset:.1f} mm off its commanded pose")
        drop = start_z - end_z
        return TrialOutcome(
            trial, bool(drop < _HELD_DROP_MM), -float(drop),
            f"drift {drift_before:.2f} mm; jaw asked {commanded:.1f} mm reached {reached:.1f} mm; "
            f"fell {drop:.1f} mm; engine {MUJOCO_ENGINE}")

    # ---------------------------------------------------------------- the controls

    def run_controls(self) -> dict:
        """The four questions that make every hold rate below them a statement about the grasp.

        A harness that cannot grip a block, that grips thin air, or that stops gripping once a second
        scene has been built measures nothing, and all three have happened here. A positive and a
        negative can both pass while the harness holds nothing, when the fault appears only on the
        second scene build, which is why the repeat is here and why it is the control that matters.
        """
        from datagen.grasps.labels import SceneGeometry  # noqa: PLC0415
        from datagen.grasps.shapes import Solid  # noqa: PLC0415

        def block(half_mm: tuple[float, float, float], centre_mm: tuple[float, float, float]):
            return SceneGeometry(
                scene_id="control", family="alone",
                objects={0: Solid(kind="box",
                                  half_extent_mm=np.asarray(half_mm, dtype=np.float64),
                                  rotation=np.eye(3),
                                  centre_mm=np.asarray(centre_mm, dtype=np.float64),
                                  instance_id=0, asset_id="control_block", mass_kg=0.2)},
                walls=())

        out: dict[str, Any] = {"engine": MUJOCO_ENGINE}

        # 1. Is the shut jaw solid? Without this every grasp reports "did not hold" for a reason
        #    that is not the grasp.
        #
        #    A differential, because an absolute rest height does not discriminate. A block dropped
        #    onto an open jaw also comes to rest above the pads: it falls between them and lands on
        #    the palm behind, which is correct physics and useless as a control. Worse, an absolute
        #    threshold passes when nothing is simulated at all, since a block that never fell is
        #    still above the line.
        #
        #    Shut against open, same pose, same block, is a question neither of those can answer by
        #    accident, and the two rest heights have to differ by more than 20 mm.
        shut = self._drop_onto_jaw(block, opening_mm=0.0)
        through = self._drop_onto_jaw(block, opening_mm=_JAW_APERTURE_MM)
        out["jaw_rest_shut_mm"] = round(shut, 1)
        out["jaw_rest_open_mm"] = round(through, 1)
        out["jaw_is_solid"] = bool(shut - through > 20.0)

        # 2. A positive: a 40 mm block, gripped across its width, must survive the support going.
        out["positive_held"] = self._control_grip(block, half=20.0)

        # 3. A negative: the same grasp commanded 60 mm wider than the block must not hold. A harness
        #    that reports a hold here is reporting something other than contact.
        out["negative_held"] = self._control_grip(block, half=20.0, opening_mm=100.0)

        # 4. The repeat, after an unrelated scene has come and gone. This is the one that catches a
        #    leaking scene lifecycle, where the positive and the negative both pass.
        self.build_scene(block((60.0, 60.0, 15.0), (200.0, 0.0, 15.0)))
        self._step(50)
        out["repeat_held"] = self._control_grip(block, half=20.0)
        return out

    def _drop_onto_jaw(self, block: Any, *, opening_mm: float) -> float:
        """Release a block above the pads and report where it comes to rest, in millimetres."""
        self.build_scene(block((20.0, 20.0, 20.0), (0.0, 0.0, 400.0)))
        self._teleport_jaw(opening_mm)
        self._place(np.array([0.0, 0.0, 300.0]), np.array([0.0, 0.0, 1.0]),
                    np.array([1.0, 0.0, 0.0]))
        self._step(400)
        return float(self._object_position(0)[2])

    def _control_grip(self, block: Any, *, half: float, opening_mm: float | None = None) -> bool:
        """Grip a block at the origin and report whether it survived the support going away."""
        geometry = block((half, half, half), (0.0, 0.0, half))
        self.build_scene(geometry)
        centre = np.array([0.0, 0.0, half])
        approach = np.array([0.0, 0.0, -1.0])
        closing = np.array([1.0, 0.0, 0.0])
        width = 2.0 * half if opening_mm is None else opening_mm
        self._teleport_jaw(min(_JAW_APERTURE_MM, width + 12.0))
        self._place(centre - approach * _STANDOFF_MM, approach, closing)
        self._step(5)
        for step in range(_STEPS_APPROACH):
            fraction = (step + 1) / _STEPS_APPROACH
            self._place(centre - approach * _STANDOFF_MM * (1.0 - fraction), approach, closing)
            self._step(1)
        self._drive_jaw(max(4.0, width - _SQUEEZE_MM), _STEPS_CLOSE)
        start_z = float(self._object_position(0)[2])
        self._set_contact(["floor"], on=False)
        self._step(_STEPS_HANG)
        self._set_contact(["floor"], on=True)
        end_z = float(self._object_position(0)[2])
        return bool(np.isfinite(end_z) and (start_z - end_z) < _HELD_DROP_MM)
