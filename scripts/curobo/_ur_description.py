"""Universal Robots' pinned description config, read without ROS: kinematics, joint limits and mesh owners per model.

The files are vendored under ``ur_ros2_description`` beside this module and held byte for byte by
``ur_ros2_description.sha256`` (``fetch_ur_meshes.py``). Standard library and PyYAML only, loaded by path like
``_description_check.py``, so the host venv, the cuRobo environment and Isaac read the same numbers.

Two things in UR's files a plain reader gets wrong:

* angles are written ``!degrees 180``, a tag PyYAML's safe loader refuses outright;
* the kinematics are URDF joint origins, not DH rows. ``dh_rows`` maps them (d1 = shoulder.z, a2 = forearm.x,
  a3 = wrist_1.x, d4 = wrist_1.z, d5 = -wrist_2.y, d6 = wrist_3.y; the twists from upper_arm.roll, wrist_2.roll and
  wrist_3's rpy) and refuses a model whose origins carry anything that mapping would drop, rather than dropping it.
"""

# No ``from __future__ import annotations``: this module is loaded by path, and a dataclass with string annotations
# looks its module up in sys.modules, where a by-path load is not registered.
import functools
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping

import yaml

__all__ = [
    "JOINTS", "JOINT_NAMES", "LINKS", "MANIFEST", "ROOT", "Description", "DescriptionError", "JointLimit", "LinkMeshes",
    "Pose", "dh_rows", "dh_rows_from", "load", "mesh_owner", "models", "parse_yaml", "read_yaml", "render_urdf",
    "render_urdf_from",
]

ROOT = Path(__file__).resolve().with_name("ur_ros2_description")
MANIFEST = Path(__file__).resolve().with_name("ur_ros2_description.sha256")

#: The kinematics blocks, parent to child, and the joints they place.
JOINTS = ("shoulder", "upper_arm", "forearm", "wrist_1", "wrist_2", "wrist_3")
JOINT_NAMES = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint",
               "wrist_3_joint")
#: The links that carry a mesh.
LINKS = ("base", "shoulder", "upper_arm", "forearm", "wrist_1", "wrist_2", "wrist_3")

#: A value the DH form requires to be zero may be float noise this large: UR writes wrist_2.z as -2.04e-11.
_ZERO = 1e-9


class DescriptionError(ValueError):
    """A vendored file that is missing, unreadable, or holds something this reader would otherwise have to guess."""


class _Loader(yaml.SafeLoader):
    """PyYAML's safe loader plus UR's ``!degrees`` tag. Any other tag still refuses."""


def _degrees(loader: yaml.SafeLoader, node: yaml.Node) -> float:
    return math.radians(float(loader.construct_scalar(node)))  # type: ignore[arg-type]


_Loader.add_constructor("!degrees", _degrees)


def parse_yaml(text: str) -> Any:
    return yaml.load(text, Loader=_Loader)  # noqa: S506 (a SafeLoader subclass with one extra scalar tag)


def read_yaml(path: Path) -> dict[str, Any]:
    path = Path(path)
    try:
        data = parse_yaml(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DescriptionError(f"{path} is not in the vendored description; run fetch_ur_meshes.py --dest "
                               f"{ROOT} --manifest {MANIFEST}") from exc
    except yaml.YAMLError as exc:
        raise DescriptionError(f"{path} cannot be read: {exc}") from exc
    if not isinstance(data, dict):
        raise DescriptionError(f"{path} holds {type(data).__name__}, not a mapping")
    return data


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float


@dataclass(frozen=True)
class JointLimit:
    """A joint's limits as UR declares them. A joint without position limits (``has_position_limits: false``, the
    continuous wrist_3 of the ur3 and ur3e) carries ``None`` for both ends rather than a number nobody wrote."""

    lower_rad: float | None
    upper_rad: float | None
    velocity_rad_s: float
    effort: float | None
    has_position_limits: bool


@dataclass(frozen=True)
class LinkMeshes:
    visual: str
    collision: str
    offset: Pose


@dataclass(frozen=True)
class Description:
    model: str
    kinematics: Mapping[str, Pose]
    joint_limits: Mapping[str, JointLimit]
    meshes: Mapping[str, LinkMeshes]
    physical: Mapping[str, Any]


def models() -> tuple[str, ...]:
    """Every model the vendored manifest pins a kinematics file for, sorted."""
    found = set()
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = PurePosixPath(line.partition("  ")[2].strip()).parts
        if len(parts) == 3 and parts[0] == "config" and parts[2] == "default_kinematics.yaml":
            found.add(parts[1])
    return tuple(sorted(found))


def _number(block: Mapping[str, Any], key: str, where: str) -> float:
    if key not in block:
        raise DescriptionError(f"{where} has no {key}")
    value = block[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DescriptionError(f"{where}.{key} is {value!r}, not a number")
    return float(value)


def _pose(block: Any, where: str) -> Pose:
    if not isinstance(block, Mapping):
        raise DescriptionError(f"{where} is not a mapping")
    return Pose(*(_number(block, key, where) for key in ("x", "y", "z", "roll", "pitch", "yaw")))


def _mesh_path(block: Any, where: str) -> str:
    try:
        path = block["mesh"]["path"]
    except (KeyError, TypeError) as exc:
        raise DescriptionError(f"{where} has no mesh.path") from exc
    if not isinstance(path, str) or not path:
        raise DescriptionError(f"{where}.mesh.path is {path!r}")
    return path


@functools.lru_cache(maxsize=None)
def load(model: str) -> Description:
    folder = ROOT / "config" / model
    if model not in models():
        raise DescriptionError(f"{model!r} is not a pinned model; the description pins {', '.join(models())}")

    kin_file = folder / "default_kinematics.yaml"
    kinematics_block = read_yaml(kin_file).get("kinematics")
    if not isinstance(kinematics_block, Mapping):
        raise DescriptionError(f"{kin_file} has no kinematics mapping")
    kinematics = {joint: _pose(kinematics_block.get(joint), f"{kin_file}:{joint}") for joint in JOINTS}

    lim_file = folder / "joint_limits.yaml"
    limits_block = read_yaml(lim_file).get("joint_limits")
    if not isinstance(limits_block, Mapping):
        raise DescriptionError(f"{lim_file} has no joint_limits mapping")
    limits = {}
    for name in JOINT_NAMES:
        block = limits_block.get(name)
        where = f"{lim_file}:{name}"
        if not isinstance(block, Mapping):
            raise DescriptionError(f"{where} is missing")
        has_effort = bool(block.get("has_effort_limits", False))
        if "has_position_limits" not in block:
            raise DescriptionError(f"{where} does not say whether it has position limits")
        has_position = bool(block["has_position_limits"])
        limits[name] = JointLimit(
            lower_rad=_number(block, "min_position", where) if has_position else None,
            upper_rad=_number(block, "max_position", where) if has_position else None,
            velocity_rad_s=_number(block, "max_velocity", where),
            effort=_number(block, "max_effort", where) if has_effort else None,
            has_position_limits=has_position,
        )

    vis_file = folder / "visual_parameters.yaml"
    mesh_block = read_yaml(vis_file).get("mesh_files")
    if not isinstance(mesh_block, Mapping):
        raise DescriptionError(f"{vis_file} has no mesh_files mapping")
    meshes = {}
    for link in LINKS:
        block = mesh_block.get(link)
        where = f"{vis_file}:{link}"
        if not isinstance(block, Mapping):
            raise DescriptionError(f"{where} is missing")
        meshes[link] = LinkMeshes(
            visual=_mesh_path(block.get("visual"), f"{where}.visual"),
            collision=_mesh_path(block.get("collision"), f"{where}.collision"),
            offset=_pose(block.get("mesh_offset"), f"{where}.mesh_offset"),
        )

    phys_file = folder / "physical_parameters.yaml"
    physical = read_yaml(phys_file).get("inertia_parameters")
    if not isinstance(physical, Mapping):
        raise DescriptionError(f"{phys_file} has no inertia_parameters mapping")

    return Description(model=model, kinematics=MappingProxyType(kinematics), joint_limits=MappingProxyType(limits),
                       meshes=MappingProxyType(meshes), physical=MappingProxyType(dict(physical)))


def dh_rows_from(kinematics: Mapping[str, Pose], *, where: str) -> tuple[tuple[float, float, float], ...]:
    """Six ``(a, d, alpha)`` rows in metres and radians from a kinematics block, or a refusal naming what does not fit."""
    missing = [joint for joint in JOINTS if joint not in kinematics]
    if missing:
        raise DescriptionError(f"{where} has no kinematics for {', '.join(missing)}")
    shoulder, upper_arm, forearm, wrist_1, wrist_2, wrist_3 = (kinematics[joint] for joint in JOINTS)
    must_be_zero = {
        "shoulder": (shoulder, ("x", "y", "roll", "pitch", "yaw")),
        "upper_arm": (upper_arm, ("x", "y", "z", "pitch", "yaw")),
        "forearm": (forearm, ("y", "z", "roll", "pitch", "yaw")),
        "wrist_1": (wrist_1, ("y", "roll", "pitch", "yaw")),
        "wrist_2": (wrist_2, ("x", "z", "pitch", "yaw")),
        "wrist_3": (wrist_3, ("x", "z")),
    }
    for joint, (pose, fields) in must_be_zero.items():
        for field in fields:
            value = getattr(pose, field)
            if abs(value) > _ZERO:
                raise DescriptionError(f"{where}: {joint}.{field} is {value!r}, which the UR DH form has no place for")
    for field in ("pitch", "yaw"):
        value = getattr(wrist_3, field)
        if abs(abs(value) - math.pi) > _ZERO:
            raise DescriptionError(f"{where}: wrist_3.{field} is {value!r}, not a half turn")
    # wrist_3's rpy (r, pi, pi) is Rz(pi) Ry(pi) Rx(r) = Rx(pi) Rx(r) = Rx(r + pi), wrapped into (-pi, pi].
    alpha_5 = math.remainder(wrist_3.roll + math.pi, 2.0 * math.pi)
    return (
        (0.0, shoulder.z, upper_arm.roll),
        (forearm.x, 0.0, 0.0),
        (wrist_1.x, 0.0, 0.0),
        (0.0, wrist_1.z, wrist_2.roll),
        (0.0, -wrist_2.y, alpha_5),
        (0.0, wrist_3.y, 0.0),
    )


def dh_rows(model: str) -> tuple[tuple[float, float, float], ...]:
    return dh_rows_from(load(model).kinematics, where=f"{ROOT / 'config' / model / 'default_kinematics.yaml'}")


def mesh_owner(model: str, link: str) -> str:
    """The arm whose mesh folder holds ``link``'s collision mesh for ``model``, read from the path UR declares."""
    description = load(model)
    if link not in description.meshes:
        raise DescriptionError(f"{link!r} is not a UR link; the links are {', '.join(LINKS)}")
    parts = PurePosixPath(description.meshes[link].collision).parts
    if len(parts) != 4 or parts[0] != "meshes" or parts[2] != "collision":
        raise DescriptionError(f"{model} {link}: collision mesh {description.meshes[link].collision!r} is not "
                               "meshes/<arm>/collision/<file>")
    return parts[1]


# ---------------------------------------------------------------------------------------------------------------------
# The URDF writer: ur_macro.xacro transcribed, without ROS
# ---------------------------------------------------------------------------------------------------------------------

def _pins() -> dict[str, str]:
    pins = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        digest, _, relative = line.partition("  ")
        pins[relative.strip()] = digest.strip()
    return pins


def _commit() -> str:
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") and " at commit " in line:
            return line.split(" at commit ", 1)[1].split()[0]
    raise DescriptionError(f"{MANIFEST} names no commit")


#: The macro's arm joints: name, the kinematics block that places it, parent link, child link.
_CHAIN = (
    ("shoulder_pan_joint", "shoulder", "base_link_inertia", "shoulder_link"),
    ("shoulder_lift_joint", "upper_arm", "shoulder_link", "upper_arm_link"),
    ("elbow_joint", "forearm", "upper_arm_link", "forearm_link"),
    ("wrist_1_joint", "wrist_1", "forearm_link", "wrist_1_link"),
    ("wrist_2_joint", "wrist_2", "wrist_1_link", "wrist_2_link"),
    ("wrist_3_joint", "wrist_3", "wrist_2_link", "wrist_3_link"),
)


def _n(value: float) -> str:
    return repr(float(value))


def _origin(pose: Pose) -> str:
    return (f'<origin xyz="{_n(pose.x)} {_n(pose.y)} {_n(pose.z)}" '
            f'rpy="{_n(pose.roll)} {_n(pose.pitch)} {_n(pose.yaw)}"/>')


def _physical(description: Description, *keys: str) -> float:
    node: Any = description.physical
    where = f"{description.model} inertia_parameters"
    for key in keys:
        if not isinstance(node, Mapping) or key not in node:
            raise DescriptionError(f"{where} has no {key}")
        node = node[key]
        where = f"{where}.{key}"
    if isinstance(node, bool) or not isinstance(node, (int, float)):
        raise DescriptionError(f"{where} is {node!r}, not a number")
    return float(node)


def _geometry(meshes: LinkMeshes) -> list[str]:
    out = []
    for kind, path in (("visual", meshes.visual), ("collision", meshes.collision)):
        out += [f"    <{kind}>", f"      {_origin(meshes.offset)}",
                f'      <geometry><mesh filename="{path}"/></geometry>', f"    </{kind}>"]
    return out


def render_urdf(model: str) -> str:
    """The URDF ``ur_macro.xacro`` would produce for ``model`` from its pinned config, as text.

    Deliberate differences from ``ur.urdf.xacro``: no world link and no base_joint (``base_link`` is the root, as in
    the Isaac descriptions the builder also reads), no ros2_control block, and no safety_controller (the macro emits
    one only with ``safety_limits``, which defaults to false). Mesh filenames are UR's ``path`` without the
    ``package://`` prefix the builder strips anyway. Numbers are ``repr(float)``; lines end in LF, so the hash of this
    text is the hash of the bytes a caller writes with ``newline="\\n"``.
    """
    return render_urdf_from(load(model))


def render_urdf_from(description: Description) -> str:
    model = description.model
    pins = _pins()
    sources = [f"config/{model}/{kind}.yaml" for kind in
               ("default_kinematics", "joint_limits", "physical_parameters", "visual_parameters")]
    sources += ["urdf/ur_macro.xacro", "urdf/inc/ur_common.xacro"]
    lines = [
        '<?xml version="1.0"?>',
        f"<!-- Written by scripts/curobo/_ur_description.py from Universal Robots' description at commit {_commit()}.",
        *(f"     {pins.get(source, 'unpinned')}  {source}" for source in sources),
        "-->",
        f'<robot name="{model}">',
        '  <link name="base_link"/>',
        '  <link name="base_link_inertia">',
        *_geometry(description.meshes["base"]),
    ]
    mass = _physical(description, "base_mass")
    radius = _physical(description, "links", "base", "radius")
    length = _physical(description, "links", "base", "length")
    side = 0.0833333 * mass * (3.0 * radius * radius + length * length)
    lines += [
        "    <inertial>",
        f'      <mass value="{_n(mass)}"/>',
        '      <origin xyz="0.0 0.0 0.0" rpy="0.0 0.0 0.0"/>',
        f'      <inertia ixx="{_n(side)}" ixy="0.0" ixz="0.0" iyy="{_n(side)}" iyz="0.0" '
        f'izz="{_n(0.5 * mass * radius * radius)}"/>',
        "    </inertial>",
        "  </link>",
    ]
    for _, block, _, child in _CHAIN:
        cog = [_physical(description, "center_of_mass", f"{block}_cog", axis) for axis in ("x", "y", "z")]
        rot = [_physical(description, "rotation", block, angle) for angle in ("roll", "pitch", "yaw")]
        tensor = {k: _physical(description, "tensor", block, k) for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")}
        lines += [
            f'  <link name="{child}">',
            *_geometry(description.meshes[block]),
            "    <inertial>",
            f'      <mass value="{_n(_physical(description, f"{block}_mass"))}"/>',
            f'      <origin xyz="{" ".join(_n(v) for v in cog)}" rpy="{" ".join(_n(v) for v in rot)}"/>',
            "      <inertia " + " ".join(f'{k}="{_n(v)}"' for k, v in tensor.items()) + "/>",
            "    </inertial>",
            "  </link>",
        ]
    lines += [
        '  <joint name="base_link-base_link_inertia" type="fixed">',
        '    <parent link="base_link"/>',
        '    <child link="base_link_inertia"/>',
        f'    <origin xyz="0.0 0.0 0.0" rpy="0.0 0.0 {_n(math.pi)}"/>',
        "  </joint>",
    ]
    for name, block, parent, child in _CHAIN:
        limit = description.joint_limits[name]
        if limit.effort is None:
            raise DescriptionError(f"{model} {name} declares no effort limit, which a URDF limit requires")
        if limit.has_position_limits:
            joint_type = "revolute"
            limit_line = (f'    <limit lower="{_n(limit.lower_rad)}" upper="{_n(limit.upper_rad)}" '
                          f'effort="{_n(limit.effort)}" velocity="{_n(limit.velocity_rad_s)}"/>')
        else:
            joint_type = "continuous"
            limit_line = f'    <limit effort="{_n(limit.effort)}" velocity="{_n(limit.velocity_rad_s)}"/>'
        lines += [
            f'  <joint name="{name}" type="{joint_type}">',
            f'    <parent link="{parent}"/>',
            f'    <child link="{child}"/>',
            f"    {_origin(description.kinematics[block])}",
            '    <axis xyz="0 0 1"/>',
            limit_line,
            '    <dynamics damping="0" friction="0"/>',
            "  </joint>",
        ]
    half = math.pi / 2.0
    for link, joint, parent, rpy in (
        ("ft_frame", "wrist_3_link-ft_frame", "wrist_3_link", (math.pi, 0.0, 0.0)),
        ("base", "base_link-base_fixed_joint", "base_link", (0.0, 0.0, math.pi)),
        ("flange", "wrist_3-flange", "wrist_3_link", (0.0, -half, -half)),
        ("tool0", "flange-tool0", "flange", (half, 0.0, half)),
    ):
        lines += [
            f'  <link name="{link}"/>',
            f'  <joint name="{joint}" type="fixed">',
            f'    <parent link="{parent}"/>',
            f'    <child link="{link}"/>',
            f'    <origin xyz="0.0 0.0 0.0" rpy="{" ".join(_n(v) for v in rpy)}"/>',
            "  </joint>",
        ]
    lines.append("</robot>")
    return "\n".join(lines) + "\n"
