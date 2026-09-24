"""Shared base classes and validators for every config schema.

Every configuration model inherits :class:`StrictModel`, which carries two
invariants across the whole tree:

* ``extra="forbid"``: a typo in a YAML file is refused at load time rather
  than ignored without comment.
* ``frozen=True``: a config object is immutable once constructed, so no
  component can mutate what another one reads. Configs are built once at
  startup and treated as values, not as state.

Use :class:`StrictModel` for every new config class, and :data:`ConfigPath` for every field that
names a file or a folder (:mod:`src.config.paths` states how such a path is read).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator

# The type of every field that names a file or a folder: a relative path in a tree is read against the
# tree's folder. Re-exported here so a schema module takes it from where it takes `StrictModel`.
from ..paths import ConfigPath as ConfigPath
from ..paths import anchor_outside_a_path, is_config_path_field


class StrictModel(BaseModel):
    """Base class for every configuration schema in this package.

    Carries the ``extra="forbid"`` and ``frozen=True`` invariants the module
    docstring describes. A subclass may opt out of either by overriding
    ``model_config``, and must state in a comment why.

    A third invariant holds for every subclass: ``${WILLY_PROJECT_ROOT}`` is read in a
    :data:`ConfigPath` field and refused in every other one (the validator below).
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    @field_validator("*", mode="after")
    @classmethod
    def _project_root_anchor_only_in_a_path(cls, value: Any, info: ValidationInfo) -> Any:
        """Refuse ``${WILLY_PROJECT_ROOT}`` in a field that is not typed :data:`ConfigPath`.

        The loader leaves the anchor in the text for the path rule, which only a path field applies, so
        in any other field it would reach the reader as the literal ``${WILLY_PROJECT_ROOT}/...``:
        measured 2026-09-24, ``robot.sim.assets_root: "${WILLY_PROJECT_ROOT}/isaac_assets"`` loaded
        and held that text, the asset root Isaac would be handed. Here, on the base, so every model is
        held to it whichever door built it, a tree load or a model built in code. A path field has
        already turned the anchor into the repository by the time this runs; it is skipped by type all
        the same, so the rule does not depend on the order pydantic runs the two in. The refusal is a
        ``ValueError``, which
        the loader reports with the key, its file and its line. :mod:`src.config.paths` says why the
        anchor is refused here rather than expanded.
        """
        refusal = anchor_outside_a_path(value)
        if refusal is None:
            return value
        field = cls.model_fields.get(info.field_name or "")
        if field is not None and is_config_path_field(field):
            return value
        raise ValueError(refusal)


# ---------------------------------------------------------------------------
# Reusable validators
# ---------------------------------------------------------------------------

# Known OpenCV ArUco dictionary names, held here rather than read from
# ``cv2.aruco``, so that validating a schema does not need OpenCV to be
# importable. Config introspection such as doc generation runs without it.
_ARUCO_DICT_NAMES: frozenset[str] = frozenset(
    {
        # Standard
        "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000",
        "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000",
        "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
        "DICT_7X7_50", "DICT_7X7_100", "DICT_7X7_250", "DICT_7X7_1000",
        # AprilTag-compatible
        "DICT_ARUCO_ORIGINAL",
        "DICT_APRILTAG_16h5", "DICT_APRILTAG_25h9",
        "DICT_APRILTAG_36h10", "DICT_APRILTAG_36h11",
        # MIP
        "DICT_ARUCO_MIP_36h12",
    }
)


def validate_aruco_dict_name(value: Any) -> str:
    """Validate that ``value`` names a known OpenCV ArUco dictionary.

    Used as an ``AfterValidator`` on the string fields that hold a dictionary
    name. Only the exact upper-case names recognised by
    ``cv2.aruco.getPredefinedDictionary`` in OpenCV >= 4.7 are accepted.
    """
    if not isinstance(value, str):
        raise TypeError(f"aruco_dict_name must be a string, got {type(value).__name__}")
    if value not in _ARUCO_DICT_NAMES:
        raise ValueError(
            f"unknown ArUco dictionary {value!r}; "
            f"must be one of: {sorted(_ARUCO_DICT_NAMES)}"
        )
    return value
