"""The numbers a named registry hand supplies to its cell, and the refusal of a cell that says otherwise.

``robot.gripper.model`` names a hand, and the registry file of that hand holds its jaw. A profile that names a hand
and leaves ``grasping.gripper_geometry`` and the gripper widths unset would otherwise plan grasps against the schema
defaults, which are the 2F-85's fingers and stroke, whatever hand it names. So every one of the thirteen keys below
that the profile chain leaves unset is filled from the named hand before validation, and a stated number that
contradicts the hand is refused with both numbers and where each was written.

Two keys are policy rather than a measurement of the hand, and are read as ranges: a stated ``min_width_mm`` is
admitted between the hand's own floor and its aperture, and a stated ``finger_pad_overlap_mm`` at or above the hand's
value. Every other difference beyond the registry's written precision is a contradiction.

The registry is the repository's (``config/grippers``) whatever tree is loaded: the hand's body, sphere map, retract
rows and evidence are committed beside the code and written from it, so its numbers are the ones a cell runs. A
deployment tree may carry its own ``grippers/``; the validator and the desk refuse one that describes a named hand
differently.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

__all__ = ["HAND_KEYS", "HandNumbers", "apply_named_hand", "hand_source"]

#: The robot keys a registry hand determines, dotted below ``robot``, and the registry field each comes from. ``kind``
#: is the spec's own field, every other one a field of its jaw.
HAND_KEYS: Final = (
    ("gripper.max_width_mm", "aperture_mm"),
    ("gripper.min_width_mm", "min_width_mm"),
    ("gripper.closed_width_mm", "closed_width_mm"),
    ("grasping.gripper_geometry.kind", "kind"),
    ("grasping.gripper_geometry.parallel_jaw.finger_length_mm", "finger_length_mm"),
    ("grasping.gripper_geometry.parallel_jaw.finger_thickness_mm", "finger_thickness_mm"),
    ("grasping.gripper_geometry.parallel_jaw.finger_width_mm", "finger_width_mm"),
    ("grasping.gripper_geometry.parallel_jaw.finger_pad_overlap_mm", "finger_pad_overlap_mm"),
    ("grasping.gripper_geometry.parallel_jaw.fingertip_depth_mm", "finger_ahead_mm"),
    ("grasping.gripper_geometry.parallel_jaw.pad_length_mm", "pad_length_mm"),
    ("grasping.gripper_geometry.parallel_jaw.pad_ahead_mm", "pad_ahead_mm"),
    ("grasping.gripper_geometry.parallel_jaw.palm_depth_mm", "palm_depth_mm"),
    ("grasping.gripper_geometry.parallel_jaw.palm_width_mm", "palm_width_mm"),
)
#: The registry writes its numbers to a hundredth of a millimetre, and a restated decimal parses to the same float.
_TOLERANCE_MM: Final = 1e-6


def _get(tree: Any, dotted: str) -> Any:
    node = tree
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _set(tree: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = tree
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


@dataclass(frozen=True)
class HandNumbers:
    """What one registry hand says its cell's thirteen hand keys are."""

    model: str
    #: (robot key, registry field, value), in :data:`HAND_KEYS` order.
    values: tuple[tuple[str, str, Any], ...]
    #: Where those numbers were written, for a person: ``config/grippers/<model>.yaml``.
    registry_file: str
    #: The ``gripper.vendor`` drivers the registry file lists as able to actuate the hand, or ``None`` where it lists
    #: none.
    drivers: "tuple[str, ...] | None" = None

    @classmethod
    def from_registry(cls, *, model: str, data_dir: str | Path | None = None) -> "HandNumbers":
        """The hand ``model`` in the registry at ``data_dir``, the repository's when it is not given.

        The default tree is ``config/`` at the repository root, so the repository is its parent.
        """
        from .grippers import load_gripper
        from .loader import _DEFAULT_DATA_DIR

        spec = load_gripper(model, data_dir=data_dir, aliases=False)
        values = tuple(
            (key, field, str(spec.kind) if field == "kind" else float(getattr(spec.jaw, field)))
            for key, field in HAND_KEYS
        )
        directory = (Path(data_dir).resolve() if data_dir else _DEFAULT_DATA_DIR) / "grippers" / f"{spec.model}.yaml"
        repository = _DEFAULT_DATA_DIR.parent
        shown = directory.relative_to(repository).as_posix() if directory.is_relative_to(repository) else str(directory)
        return cls(model=spec.model, values=values, registry_file=shown, drivers=spec.drivers)

    def driver_refusal(self, robot: dict) -> "str | None":
        """Why the raw ``robot`` mapping may not drive this hand with its ``gripper.vendor``, or ``None``.

        Asked only of a real UR arm: a sim, a dummy or another vendor's arm reaches no Robotiq socket and no
        controller I/O through this config. ``none`` and ``dummy`` are always admitted. A vendor the schema refuses
        in its own words is not judged here.
        """
        if self.drivers is None:
            return None
        from src.robot.core.gripper_vendor import GripperVendor

        from .schema.robot import GripperConfig, RobotConfig

        arm = str(robot.get("vendor") or RobotConfig.model_fields["vendor"].default).strip().lower()
        if arm != "ur":
            return None
        stated = robot.get("gripper")
        gripper: dict = stated if isinstance(stated, dict) else {}
        written = gripper.get("vendor") or GripperConfig.model_fields["vendor"].default
        try:
            vendor = GripperVendor.from_string(str(written)).value
        except ValueError:
            return None
        if vendor in (GripperVendor.NONE.value, GripperVendor.DUMMY.value, *self.drivers):
            return None
        return (
            f"robot.gripper.model names {self.model} on a real UR arm, and robot.gripper.vendor is {vendor!r}: "
            f"{self.registry_file} lists the drivers that can actuate this hand, {list(self.drivers)}. Set "
            f"robot.gripper.vendor to one of them, or to 'none' while nothing on the flange is actuated; if {vendor!r} "
            f"does actuate this hand, add it to drivers in {self.registry_file}"
        )

    def applied(self, robot: dict, *, stated_in: "Callable[[], Mapping[str, str]] | None" = None) -> dict:
        """``robot`` with every unset hand key filled from this hand, or ``ConfigError`` naming every contradiction.

        ``stated_in`` returns a map of each dotted ``robot.`` key to where a layer wrote it, best effort, for the
        sentence. It is called only when there is a contradiction to name.
        """
        from .loader import ConfigError

        out = copy.deepcopy(robot)
        #: (key, stated, registry value, field, the admitted span or "")
        found: list[tuple[str, Any, Any, str, str]] = []
        for key, field, value in self.values:
            stated = _get(out, key)
            if stated is None:
                _set(out, key, value)
                continue
            if field == "kind":
                if str(stated) != value:
                    found.append((key, stated, value, field, ""))
                continue
            try:
                number = float(stated)
            except (TypeError, ValueError):
                continue  # the schema refuses a value that is not a number, in its own words
            if field == "min_width_mm":
                aperture = dict((f, v) for _, f, v in self.values)["aperture_mm"]
                if not float(value) - _TOLERANCE_MM <= number <= float(aperture) + _TOLERANCE_MM:
                    found.append((key, stated, value, field, f"from {value:g} up to the aperture {aperture:g}"))
            elif field == "finger_pad_overlap_mm":
                if number < float(value) - _TOLERANCE_MM:
                    found.append((key, stated, value, field, f"at or above {value:g}"))
            elif abs(number - float(value)) > _TOLERANCE_MM:
                found.append((key, stated, value, field, ""))
        if found:
            where = dict(stated_in()) if stated_in is not None else {}
            contradictions = [self._sentence(*row, where) for row in found]
            raise ConfigError(
                f"robot.gripper.model names {self.model}, and this config states numbers its registry file "
                f"{self.registry_file} contradicts:\n  " + "\n  ".join(contradictions) + "\n"
                f"A named hand supplies these itself: delete the stated values, or correct {self.registry_file} if the "
                f"hand was measured differently."
            )
        return out

    def _sentence(self, key: str, stated: Any, value: Any, field: str, allowed: str, where: Mapping[str, str]) -> str:
        location = where.get(f"robot.{key}", "")
        said = f" (written in {location})" if location else ""
        registry = f"{self.registry_file} jaw.{field}" if field != "kind" else f"{self.registry_file} kind"
        span = f", and a stated value is admitted {allowed}" if allowed else ""
        return f"robot.{key} is {stated}{said}, and {registry} is {value}{span}"

    def render(self) -> str:
        return f"{self.model} ({self.registry_file}): " + ", ".join(f"{key} {value}" for key, _, value in self.values)

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.model, "registry_file": self.registry_file,
                "drivers": list(self.drivers) if self.drivers is not None else None,
                "values": {key: {"field": field, "value": value} for key, field, value in self.values}}


def hand_source(cfg: Any, path: str) -> str:
    """Where the hand ``cfg`` names supplies the dotted key ``path``, for a person, or ``""`` when no hand supplies it.

    ``cfg`` is a loaded ``AppConfig`` or ``RobotConfig``. Only a caller that already knows no layer writes ``path``
    may read the answer as the key's provenance, because a stated number shadows the hand's.
    """
    fields = {f"robot.{key}": field for key, field in HAND_KEYS}
    if path not in fields:
        return ""
    robot = getattr(cfg, "robot", cfg)
    model = getattr(getattr(robot, "gripper", None), "model", None)
    if not isinstance(model, str) or not model:
        return ""
    try:
        registry_file = HandNumbers.from_registry(model=model).registry_file
    except Exception:  # noqa: BLE001 (provenance is for a person; a registry that cannot say leaves the key unnamed)
        return ""
    field = fields[path]
    return f"{registry_file} {'kind' if field == 'kind' else f'jaw.{field}'}"


def apply_named_hand(
    raw_robot: Any, *, stated_in: "Callable[[], Mapping[str, str]] | None" = None, data_dir: "Path | None" = None,
) -> Any:
    """The raw robot mapping with a named hand's numbers from the repository registry, or unchanged where it names none.

    A name that is not registry-shaped is passed through, so the schema refuses it in its own words. ``data_dir`` is
    the loaded tree, read only for the sentence: a hand that tree describes and the repository does not is refused
    with the reason a tree may only repeat the repository's hands.
    """
    from .grippers import MODEL_NAME_PATTERN, tree_hand_refusal
    from .loader import ConfigError

    if not isinstance(raw_robot, dict):
        return raw_robot
    gripper = raw_robot.get("gripper")
    model = gripper.get("model") if isinstance(gripper, dict) else None
    if not isinstance(model, str) or not re.fullmatch(MODEL_NAME_PATTERN, model):
        return raw_robot
    try:
        numbers = HandNumbers.from_registry(model=model)
    except ConfigError as exc:
        if data_dir is not None and (Path(data_dir) / "grippers" / f"{model}.yaml").is_file():
            refusal = tree_hand_refusal(model, data_dir=data_dir)
            if refusal is not None:
                raise ConfigError(refusal) from exc
        raise
    refusal = numbers.driver_refusal(raw_robot)
    if refusal is not None:
        raise ConfigError(refusal)
    return numbers.applied(raw_robot, stated_in=stated_in)
