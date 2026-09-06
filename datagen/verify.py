"""Check a written dataset against itself: the gate that stops a labelling bug reaching a model.

Everything here reads only what is on disk: the instance PNGs, the scene JSONs, and the poses. No
Isaac, no GPU, so it runs in CI and on a laptop and can be pointed at a finished dataset months later.

The checks are ordered by how quietly they fail:

1. ``visible <= unoccluded``. Impossible by construction, therefore evidence that construction broke.
2. The PNG agrees with the JSON. The labels and the mask image are written from the same arrays; if
   they disagree, one of them was written from something else.
3. Every object's box is the one nearest to where that object actually is. This is the check the
   other two cannot make: a permuted mask set has perfect pixel counts, a perfect
   ``visible <= unoccluded`` and perfect PNG/JSON agreement, and is completely wrong. Only projecting
   each object's settled 3-D position through the recorded camera and asking which box it lands
   nearest notices the swap.
4. The colour image is an image. Checks 1 to 3 are all about geometry and labels, and every one of
   them passes on a scene whose RGB is entirely black: blank frames alongside correct depth, correct
   masks and correct boxes are written as ``ok`` unless the picture itself is checked.

Check 3 asks for the nearest box, not for containment, and that is a correction rather than a
weakening. A static occluder and the frame edge both move a centroid out of its own box without any
labelling error. Neither can move it closer to a different object's box, which is the thing actually
being tested.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.utility.log_cfg import create_logger

from datagen.constants import DATAGEN_LOG_DIR, VERIFY_LOG_FILE
from datagen.render.camera import intrinsics_matrix, look_at_camera_to_base, project_to_pixels
from datagen.render.labels import pixel_id
from datagen.render.quality import describe_rgb

__all__ = ["VerifyReport", "verify_dataset", "verify_scene"]

logger = create_logger("datagen.verify", VERIFY_LOG_FILE, log_dir=DATAGEN_LOG_DIR)

#: View outcomes that promise a colour image on disk. A skipped view legitimately has none.
_OUTCOMES_WITH_AN_IMAGE = frozenset({"rendered", "rendered_after_resample"})

#: An object below this visibility is mostly hidden, so where its centroid lands says little. Below
#: this the projection check is skipped rather than guessed at.
_PROJECTION_MIN_VISIBILITY = 0.6
#: How much closer another object's box must be before the pairing counts as wrong. Neighbouring boxes
#: in a packed bin are genuinely close together, and a few pixels of difference is not evidence.
#:
#: A floor, not the whole rule: see ``_projection_margin_px``. A fixed pixel threshold encodes "how
#: far is far" for one camera only. A wrist camera sits far closer to the objects than an oblique
#: one, so it sees them at several times the pixel size and looks at one face rather than a
#: silhouette, which puts a solid's centre of mass far from the pixels it occupies.
_PROJECTION_MARGIN_PX = 8.0
#: The scale-relative part: the mismatch must exceed this share of the object's own box diagonal.
#: "Wrong by more than the object's own size" means the same thing at every range, which is exactly
#: what a fixed pixel count does not. A genuine permutation hands an object a different object's mask
#: and so misses by far more than its own extent.
_PROJECTION_MARGIN_FRACTION = 0.6


@dataclass(frozen=True, slots=True)
class VerifyReport:
    """What the checks found. ``ok`` is the only thing a caller needs; the rest is why."""

    scenes: int
    views: int
    objects: int
    problems: tuple[str, ...]
    projection_checked: int
    projection_failed: int
    images_checked: int = 0
    images_blank: int = 0

    @property
    def ok(self) -> bool:
        return not self.problems

    def summary(self) -> str:
        head = (f"{self.scenes} scene(s), {self.views} view(s), {self.objects} object label(s); "
                f"{self.projection_checked} pose->box check(s), {self.projection_failed} failed; "
                f"{self.images_checked} image(s), {self.images_blank} blank")
        if self.ok:
            return f"OK: {head}"
        return f"FAILED: {head}\n  " + "\n  ".join(self.problems[:40])


def _read_png(path: Path) -> np.ndarray | None:
    """A written PNG exactly as it sits on disk: 16-bit stays 16-bit, channels stay channels."""
    import cv2  # type: ignore[import-not-found]  # noqa: PLC0415 (only the file checks need it)

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return None if image is None else np.asarray(image)


def _projection_margin_px(box: Sequence[float]) -> float:
    """How much closer a rival box must be to count as evidence, for this object at this range."""
    x1, y1, x2, y2 = (float(value) for value in box)
    diagonal = float(np.hypot(x2 - x1, y2 - y1))
    return max(_PROJECTION_MARGIN_PX, _PROJECTION_MARGIN_FRACTION * diagonal)


def _distance_to_box(point: np.ndarray, box: Sequence[float]) -> float:
    """Pixels from a point to a box; 0.0 inside it. The usual axis-aligned distance."""
    x1, y1, x2, y2 = (float(value) for value in box)
    dx = max(x1 - point[0], 0.0, point[0] - x2)
    dy = max(y1 - point[1], 0.0, point[1] - y2)
    return float(np.hypot(dx, dy))


def _camera_model(view: dict, camera: dict | None) -> tuple[np.ndarray, np.ndarray] | None:
    """The view's ``(camera_to_base, K)``: from the file when recorded, else from its pose.

    Preferring the recorded matrices is the point of recording them: the verifier then checks the
    labels against the same camera model the renderer used, instead of against its own idea of the
    convention.
    """
    recorded, intrinsics = view.get("camera_to_base_mm"), view.get("intrinsics")
    if recorded is not None and intrinsics is not None:
        return np.asarray(recorded, dtype=np.float64), np.asarray(intrinsics, dtype=np.float64)
    if camera is None:
        return None
    return (
        look_at_camera_to_base(view["camera_position_mm"], view["camera_look_at_mm"]),
        intrinsics_matrix(tuple(camera["resolution"]), float(camera["horizontal_fov_deg"])),
    )


def verify_scene(scene_dir: Path, *, expect_rgb: bool = True) -> VerifyReport:
    """Run every check on one written scene directory.

    ``expect_rgb`` is read from the dataset's own provenance stamp by `verify_dataset`, never
    inferred from which files happen to exist. That distinction is check 4: "there is no colour
    image" and "the colour image is black" are the same thing on disk and different facts about the
    run, so the check may not decide which one it is by looking for the file.
    """
    payload = json.loads((scene_dir / "scene.json").read_text(encoding="utf-8"))
    scene_id = scene_dir.name
    problems: list[str] = []
    views = objects = checked = failed = 0
    images = blank = 0

    # Objects the settle threw off the table. They must appear in nothing downstream; that is the
    # whole claim being made by removing them, so it is checked rather than trusted.

    dropped = {int(index) for index in payload.get("dropped_objects", [])}
    for index in sorted(dropped):
        if str(index) in payload.get("settled_poses_mm_xyzw", {}):
            problems.append(
                f"{scene_id}: object {index} is recorded as dropped but still has a settled pose"
            )

    fov_by_name = {camera["name"]: camera for camera in payload["spec"]["cameras"]}
    for view in payload["views"]:
        views += 1
        name = view["name"]
        mask_path = scene_dir / f"{name}_instances.png"
        instance_map = _read_png(mask_path) if mask_path.exists() else None
        camera = fov_by_name.get(name)

        if expect_rgb and str(view.get("outcome", "")) in _OUTCOMES_WITH_AN_IMAGE:
            # Driven by the recorded outcome, not by which files happen to exist. "Check it if it is
            # there" would pass a view that claims to have rendered and wrote nothing at all, which
            # is the same shape of hole this check closes.

            rgb_path = scene_dir / f"{name}_rgb.png"
            images += 1
            if not rgb_path.exists():
                blank += 1
                problems.append(
                    f"{scene_id}/{name}: outcome is '{view['outcome']}' but there is no "
                    f"{rgb_path.name}; the view claims an image it never wrote"
                )
            else:
                content = describe_rgb(_read_png(rgb_path))
                if not content.renderable:
                    blank += 1
                    problems.append(
                        f"{scene_id}/{name}: {rgb_path.name} is not an image; "
                        f"{content.why_not()}. Its labels may be perfect; the picture is not there"
                    )

        for label in view["objects"]:
            objects += 1
            index = int(label["instance_id"])
            visible_px, unoccluded_px = int(label["visible_px"]), int(label["unoccluded_px"])

            if index in dropped:
                problems.append(
                    f"{scene_id}/{name}: object {index} is recorded as dropped; it left the table "
                    f"and was removed; yet it has a label here"
                )

            if visible_px > unoccluded_px:
                problems.append(
                    f"{scene_id}/{name}: object {index} visible {visible_px} > unoccluded "
                    f"{unoccluded_px}; impossible by construction"
                )
            if instance_map is not None:
                painted = int(np.count_nonzero(np.asarray(instance_map) == pixel_id(index)))
                if painted != visible_px:
                    problems.append(
                        f"{scene_id}/{name}: object {index} has {painted} px in the mask image but "
                        f"{visible_px} in the labels; the PNG and the JSON disagree"
                    )
            model = _camera_model(view, camera)
            if model is None or label["bbox_xyxy"] is None:
                continue
            if float(label["visibility"]) < _PROJECTION_MIN_VISIBILITY:
                continue  # its centroid may honestly be behind something; see the module docstring

            pose = payload["settled_poses_mm_xyzw"].get(str(index))
            if pose is None:
                continue
            checked += 1
            camera_to_base, k = model
            uv = project_to_pixels(np.asarray([pose[0]], dtype=np.float64), camera_to_base, k)[0]
            if not np.all(np.isfinite(uv)):
                continue  # behind the camera; the box, if any, is not this object's to explain
            if camera is not None:
                width, height = (float(value) for value in camera["resolution"])
                if not (0.0 <= uv[0] < width and 0.0 <= uv[1] < height):
                    # The centre is not in the picture. "Which box does it land nearest" is then a
                    # malformed question, not a failed one: the object's visible sliver is clipped at
                    # the frame edge and its centre is somewhere off it, so the nearest box is
                    # whichever neighbour happens to sit inward.

                    continue
            own = _distance_to_box(uv, label["bbox_xyxy"])
            rivals = [
                (_distance_to_box(uv, other["bbox_xyxy"]), int(other["instance_id"]))
                for other in view["objects"]
                if other["bbox_xyxy"] is not None and int(other["instance_id"]) != index
            ]
            nearest, nearest_index = min(rivals, default=(float("inf"), -1))
            if nearest + _projection_margin_px(label["bbox_xyxy"]) < own:
                failed += 1
                problems.append(
                    f"{scene_id}/{name}: object {index} ({label['asset_id']}) sits at "
                    f"{tuple(round(value, 1) for value in pose[0])} mm, projects to "
                    f"({uv[0]:.0f}, {uv[1]:.0f}) px, which is {own:.0f} px from its OWN box "
                    f"{label['bbox_xyxy']} but only {nearest:.0f} px from object {nearest_index}'s; "
                    f"the masks look swapped"
                )
    return VerifyReport(
        scenes=1, views=views, objects=objects, problems=tuple(problems),
        projection_checked=checked, projection_failed=failed,
        images_checked=images, images_blank=blank,
    )


def _dataset_has_colour(root: Path) -> bool:
    """Did this dataset render colour at all? From its provenance stamp, not from the files.

    A missing or unreadable stamp returns ``True``, which keeps the check on. Fail-closed in the
    direction that matters: an unstamped dataset gets the stricter treatment, never the weaker one.
    """
    stamp_path = root / "provenance.json"
    if not stamp_path.is_file():
        return True
    try:
        stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
        return not bool(stamp.get("config", {}).get("render", {}).get("depth_only", False))
    except (OSError, ValueError):
        logger.warning("could not read %s; keeping the colour-image check on", stamp_path)
        return True


def verify_dataset(root: Path) -> VerifyReport:
    """Every scene under ``root/scenes`` that was actually written."""
    scenes_dir = Path(root) / "scenes"
    if not scenes_dir.is_dir():
        logger.error("no scenes directory under %s; nothing to verify", root)
        return VerifyReport(0, 0, 0, (f"no scenes directory under {root}",), 0, 0)
    scenes = views = objects = checked = failed = images = blank = 0
    problems: list[str] = []
    expect_rgb = _dataset_has_colour(Path(root))
    if not expect_rgb:
        # Said, not silently skipped. A depth-only dataset legitimately has no picture; a dataset
        # that lost its pictures looks identical from here, and the difference is only in the stamp.
        logger.info("%s was rendered depth-only; the colour-image check does not apply", root)
    for scene_dir in sorted(scenes_dir.iterdir()):
        if not (scene_dir / "scene.json").exists():
            continue
        report = verify_scene(scene_dir, expect_rgb=expect_rgb)
        scenes += 1
        views += report.views
        objects += report.objects
        checked += report.projection_checked
        failed += report.projection_failed
        images += report.images_checked
        blank += report.images_blank
        problems.extend(report.problems)
    # Per dataset, not per scene: a long run would drown the finding in its own progress. The failure
    # is an error because it is returned: a failed check never raises, so a caller that ignores `ok`
    # leaves no other trace that the labels disagreed with the pixels. A scene.json that will not
    # parse, or that has drifted from the schema, still raises out of `json.loads` and the payload
    # lookups.

    if problems:
        logger.error("%s FAILED: %d problem(s) over %d scene(s)/%d view(s); "
                     "%d/%d pose->box check(s) failed, %d/%d image(s) blank. First: %s",
                     root, len(problems), scenes, views, failed, checked, blank, images,
                     "; ".join(problems[:3]))
    else:
        logger.info("%s OK: %d scene(s), %d view(s), %d object label(s), "
                    "%d pose->box check(s), %d image(s) checked",
                    root, scenes, views, objects, checked, images)
    return VerifyReport(scenes, views, objects, tuple(problems), checked, failed, images, blank)
