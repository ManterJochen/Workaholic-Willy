"""A hand in a vendor's frame and units becomes the bundle a bake writes.

numpy only, loadable by path. Every hand route that starts from geometry ends here: a vendor STL or OBJ
(``scripts/grippers/write_hand_from_mesh.py``) and one standalone hand USD (``scripts/grippers/bake_gripper_variant.py``).

The frame change runs in the order a bake does it: scale into millimetres, move the mounting face to zero along the
vendor axis the approach names, then rotate, ``bundle = vendor @ rotation.T``.

Axes are words, never a typed matrix. A customer states which vendor axis the closing, the approach and the binormal
of the hand lie along, each one of ``+X -X +Y -Y +Z -Z``. The rotation is derived from the words, and a spelling whose
determinant is -1 is refused: it is a mirror, and a mirrored symmetric gripper has identical extents, so no assertion
about sizes would ever catch it. The bundle frame is the one every hand model uses: closing X, approach +Y,
binormal Z.

No number that places a hand has a default: ``scale_to_mm``, ``mount_face_mm`` and ``origin`` are stated or refused.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping

import numpy as np

__all__ = ["AXIS_FLAGS", "AXIS_WORDS", "HandBundle", "HandBundleWritten", "VendorAxes", "hand_bundle_rules", "joined_axis_words"]

#: The words an axis is stated in.
AXIS_WORDS: Final = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")
#: The command line flags that take one.
AXIS_FLAGS: Final = ("--closing", "--approach", "--binormal")


def joined_axis_words(argv: list[str]) -> list[str]:
    """``--binormal -X`` as ``--binormal=-X``, because argparse reads a lone ``-X`` as an option.

    The natural spelling is the one a customer types.
    """
    out: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in AXIS_FLAGS and index + 1 < len(argv) and argv[index + 1] in AXIS_WORDS:
            out.append(f"{token}={argv[index + 1]}")
            index += 2
            continue
        out.append(token)
        index += 1
    return out


def hand_bundle_rules() -> Any:
    """``planning/_hand_bundle.py``: the one rule a bundle meets before it is written or composed.

    The package's own module where this one was imported as part of the package, so a refusal raised here is the
    class a caller catches; by path where a script loaded this file on its own.
    """
    if __name__.startswith("src."):
        from .. import _hand_bundle

        return _hand_bundle
    name = "willy_hand_bundle"
    if name not in sys.modules:
        path = Path(__file__).resolve().parents[1] / "_hand_bundle.py"
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _refused(sentence: str) -> Exception:
    return hand_bundle_rules().HandBundleRefused(sentence)


def _unit(word: str) -> tuple[float, float, float]:
    if word not in AXIS_WORDS:
        raise _refused(f"{word!r} is not an axis; an axis is one of {', '.join(AXIS_WORDS)}")
    out = [0.0, 0.0, 0.0]
    out["XYZ".index(word[1])] = 1.0 if word[0] == "+" else -1.0
    return (out[0], out[1], out[2])


@dataclass(frozen=True, slots=True)
class VendorAxes:
    """Which vendor axis the hand's closing, approach and binormal lie along."""

    closing: str
    approach: str
    binormal: str

    @classmethod
    def from_words(cls, *, closing: str, approach: str, binormal: str) -> "VendorAxes":
        axes = cls(closing=closing, approach=approach, binormal=binormal)
        rows = [_unit(closing), _unit(approach), _unit(binormal)]
        letters = [closing[1], approach[1], binormal[1]]
        if len(set(letters)) != 3:
            raise _refused(
                f"closing {closing}, approach {approach} and binormal {binormal} name the vendor axis "
                f"{max(set(letters), key=letters.count)} twice, and three directions of a hand are three different axes"
            )
        determinant = float(np.linalg.det(np.asarray(rows, dtype=np.float64)))
        if determinant < 0.0:
            raise _refused(
                f"closing {closing}, approach {approach} and binormal {binormal} have determinant {determinant:+.0f}: "
                f"that is a mirror, not a rotation, and a mirrored symmetric gripper has identical extents, so nothing "
                f"downstream would notice. Flip the sign of the binormal"
            )
        return axes

    @property
    def rotation(self) -> tuple[tuple[float, float, float], ...]:
        """Rows are the bundle's x, y and z in vendor axes: ``bundle = vendor @ rotation.T``."""
        return (_unit(self.closing), _unit(self.approach), _unit(self.binormal))

    @property
    def approach_index(self) -> int:
        """The vendor axis, as an index into (x, y, z), the mounting face is measured along."""
        return "XYZ".index(self.approach[1])

    def render(self) -> str:
        return f"closing {self.closing}, approach {self.approach}, binormal {self.binormal} in the vendor's axes"

    def to_dict(self) -> dict[str, Any]:
        return {"closing": self.closing, "approach": self.approach, "binormal": self.binormal,
                "rotation": [list(row) for row in self.rotation]}


@dataclass(frozen=True, slots=True)
class HandBundleWritten:
    """What one write put on disk."""

    path: Path
    origin: str
    #: part -> (vertex count, lower corner, upper corner), millimetres in the bundle frame.
    extents: tuple[tuple[str, int, tuple[float, float, float], tuple[float, float, float]], ...]

    def render(self) -> str:
        parts = "; ".join(
            f"{part} {count} vertices x [{lo[0]:.2f}, {hi[0]:.2f}] y [{lo[1]:.2f}, {hi[1]:.2f}] z [{lo[2]:.2f}, {hi[2]:.2f}]"
            for part, count, lo, hi in self.extents
        )
        return f"wrote {self.path.name}, numbers starting at the {self.origin}: {parts}"

    def to_dict(self) -> dict[str, Any]:
        return {"path": str(self.path), "origin": self.origin,
                "parts": {part: {"vertices": count, "lower_mm": list(lo), "upper_mm": list(hi)}
                          for part, count, lo, hi in self.extents}}


@dataclass(frozen=True, slots=True)
class HandBundle:
    """The three parts of one hand in the bundle frame, millimetres, and where their numbers start."""

    vertices: Mapping[str, Any]
    faces: Mapping[str, Any]
    origin: str

    @classmethod
    def from_parts(
        cls,
        *,
        parts: Mapping[str, tuple[Any, Any]],
        axes: VendorAxes,
        scale_to_mm: float,
        mount_face_mm: float,
        origin: str,
    ) -> "HandBundle":
        """``parts`` maps ``gripper``, ``lfinger`` and ``rfinger`` to (vertices, triangle faces) in the vendor's frame.

        ``scale_to_mm`` turns a vendor unit into millimetres (1000 for metres, 1 for millimetres, 25.4 for inches);
        ``mount_face_mm`` is the mounting face's coordinate along the approach axis, millimetres, in the vendor frame;
        ``origin`` says whether the numbers then start at the ``flange`` or at the hand's own ``mounting_face``.
        """
        rules = hand_bundle_rules()
        if origin not in rules.ORIGINS:
            raise _refused(f"origin {origin!r}: a hand's numbers start at the {' or the '.join(rules.ORIGINS)}")
        if not (np.isfinite(scale_to_mm) and scale_to_mm > 0.0):
            raise _refused(f"scale_to_mm {scale_to_mm!r}: one vendor unit is a positive number of millimetres")
        if not np.isfinite(mount_face_mm):
            raise _refused(f"mount_face_mm {mount_face_mm!r} is not a number")
        missing = [part for part in rules.HAND_PARTS if part not in parts]
        extra = sorted(set(parts) - set(rules.HAND_PARTS))
        if missing or extra:
            raise _refused(
                f"a hand is exactly {', '.join(rules.HAND_PARTS)}; these parts "
                f"{'lack ' + ', '.join(missing) if missing else ''}{' and ' if missing and extra else ''}"
                f"{'add ' + ', '.join(extra) if extra else ''}"
            )
        rotation = np.asarray(axes.rotation, dtype=np.float64)
        vertices: dict[str, np.ndarray] = {}
        faces: dict[str, np.ndarray] = {}
        for part in rules.HAND_PARTS:
            raw, tris = parts[part]
            points = np.asarray(raw, dtype=np.float64) * float(scale_to_mm)
            if points.ndim != 2 or points.shape[1:] != (3,):
                raise _refused(f"{part}: vertices of shape {points.shape}, and a mesh's vertices are (N, 3)")
            points[:, axes.approach_index] -= float(mount_face_mm)
            vertices[part] = points @ rotation.T
            faces[part] = np.asarray(tris, dtype=np.int32)
        return cls(vertices=vertices, faces=faces, origin=origin)

    @classmethod
    def from_mesh_files(
        cls,
        *,
        gripper: str | Path,
        lfinger: str | Path,
        rfinger: str | Path,
        axes: VendorAxes,
        scale_to_mm: float,
        mount_face_mm: float,
        origin: str,
    ) -> "HandBundle":
        """The three parts read from vendor mesh files (STL, OBJ, anything trimesh reads), then :meth:`from_parts`.

        Read with ``process=False``, so vertex order and faces survive as the file holds them; a file holding several
        bodies for one part is concatenated into that part.
        """
        import trimesh

        parts: dict[str, tuple[Any, Any]] = {}
        for part, path in (("gripper", gripper), ("lfinger", lfinger), ("rfinger", rfinger)):
            if not Path(path).is_file():
                raise _refused(f"{part}: no mesh file at {path}")
            mesh: Any = trimesh.load(str(path), force="mesh", process=False)
            if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
                raise _refused(f"{part}: {Path(path).name} holds no triangles")
            parts[part] = (np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64))
        return cls.from_parts(parts=parts, axes=axes, scale_to_mm=scale_to_mm, mount_face_mm=mount_face_mm,
                              origin=origin)

    def arrays(self, *, records: Mapping[str, Any] | None = None) -> dict[str, np.ndarray]:
        """The arrays a bake writes: float64 vertices, int32 faces, int64 frame 6 per part, and the origin."""
        rules = hand_bundle_rules()
        out: dict[str, np.ndarray] = {}
        for part in rules.HAND_PARTS:
            out[f"{part}__v"] = np.asarray(self.vertices[part], dtype=np.float64)
            out[f"{part}__f"] = np.asarray(self.faces[part], dtype=np.int32)
            out[f"{part}__frame"] = np.asarray([rules.HAND_FRAME], dtype=np.int64)
        out[rules.ORIGIN_KEY] = np.asarray([self.origin])
        for key, value in (records or {}).items():
            if not key.startswith(rules.RECORD_PREFIX):
                raise _refused(f"{key!r}: a record a hand bundle keeps about itself starts with {rules.RECORD_PREFIX}")
            out[key] = np.asarray([value])
        return out

    def write(self, path: str | Path, *, records: Mapping[str, Any] | None = None) -> HandBundleWritten:
        """Validate, refuse an existing file, and write ``path``: a scan describes a hand better than a second write."""
        rules = hand_bundle_rules()
        target = Path(path)
        arrays = self.arrays(records=records)
        refusal = rules.hand_bundle_refusal(arrays, name=target.name)
        if refusal is not None:
            raise _refused(refusal)
        if target.exists():
            raise _refused(f"{target} is already there, and a written hand bundle is never overwritten: delete it first")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = dict(arrays)
        np.savez_compressed(target, **payload)
        extents = tuple(
            (part, int(len(arrays[f"{part}__v"])),
             tuple(float(v) for v in arrays[f"{part}__v"].min(axis=0)),
             tuple(float(v) for v in arrays[f"{part}__v"].max(axis=0)))
            for part in rules.HAND_PARTS
        )
        return HandBundleWritten(path=target, origin=self.origin, extents=extents)  # type: ignore[arg-type]
