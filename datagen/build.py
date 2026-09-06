"""The build loop: layout, render, label, write, with resume and a licence gate that runs first.

Deliberately thin. Everything it does is delegated: the scene comes from :mod:`datagen.scenes.layout`,
the images from :mod:`datagen.render.isaac`, the files from :mod:`datagen.render.writer`, because a
build loop that also decides things is a build loop nobody can test.

Two things it does decide, and both are about not wasting a night of GPU time:

* The licence audit runs before Isaac boots. A manifest problem discovered after 200 path-traced
  scenes costs 200 scenes; discovered here it costs a second. Every rendered image is a derivative
  work of the meshes in it, so an asset that cannot ship cannot be rendered.
* Resume is checked before anything is built. A re-run skips completed scenes, including rejected
  ones: re-rendering a scene that was rejected for instability would reject it again, same seed,
  same physics.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from src.utility.log_cfg import create_logger

from datagen.assets.manifest import AssetManifest
from datagen.config import DatagenConfig
from datagen.constants import BUILD_LOG_FILE, DATAGEN_LOG_DIR
from datagen.provenance import provenance_stamp
from datagen.render.writer import DatasetWriter, SceneRecord, encode_depth_png
from datagen.scenes.layout import (
    layout_scene,
    layout_scene_with_assets,
    plan_families,
    scene_seed,
)

__all__ = ["build_dataset", "build_manifest"]

logger = create_logger("datagen.build", BUILD_LOG_FILE, log_dir=DATAGEN_LOG_DIR)

#: Scenes that may fail in a row before the run gives up. One failure is an unlucky scene; this many
#: consecutive ones is a broken Isaac session, and grinding through 300 of them helps nobody.
_CONSECUTIVE_ERROR_LIMIT = 3
#: Attempts per scene before it is recorded as failed. See ``_render_with_retry``: an Isaac buffer
#: read fails intermittently, and one retry leaves a measurable share of scenes dying on it.
_RENDER_ATTEMPTS = 3


def build_manifest(config: DatagenConfig, scenes: int | None = None) -> AssetManifest:
    """Every asset the configured scenes will actually place, in one manifest the gate can audit.

    Takes the assets from the layout rather than re-drawing them from the same seed. Re-drawing
    depends on consuming the random stream in exactly the same order as the layout does, a contract
    no signature states and nothing checks. The manifest must be exactly the placed set: a superset
    audits clean while an unaudited asset reaches a pixel.
    """
    manifest = AssetManifest()
    seen: set[str] = set()
    families = plan_families(config)
    for index in range(scenes if scenes is not None else config.scenes):
        _spec, assets = layout_scene_with_assets(config, index, families[index])
        for asset in assets:
            if asset.asset_id in seen:
                continue
            seen.add(asset.asset_id)
            # The asset builds its own record. Assembling it field by field here drops whatever
            # the writer forgets: `parts` for composites, then `mesh_path` and `attribution` for a
            # scanned object, and rows missing their attribution are refused by the licence audit,
            # so a mesh dataset cannot be built at all. `SceneAsset.as_record` is the contract, and
            # a type that forgets a field fails mypy.
            manifest.add(asset.as_record())
    return manifest


def _render_with_retry(
    renderer: Any, spec: Any, manifest: AssetManifest, config: DatagenConfig, index: int,
) -> Any:
    """Render a scene, once more if the renderer refused it.

    Isaac's colour buffer occasionally comes up dead for an entire scene while its depth and
    segmentation stay correct, and the same scene then renders perfectly on a later attempt with no
    code change on the success path. The failure is intermittent, so the renderer's refusal, which
    is fail-closed and correct, turns each occurrence into a permanently missing scene unless
    something asks again.

    The retry rebuilds the scene from the same seed, so what is re-rendered is the same scene rather
    than a different one that happens to work: a retry that quietly changed the layout would bias the
    dataset towards whatever Isaac finds easy, which is the opposite of the point.

    One retry is not enough. Scenes still die on an Isaac read, on the depth buffer as well as the
    colour one, having already been re-rendered once. For an intermittent fault a further attempt is
    nearly free, because a failed read costs about a second while a lost scene costs the run a
    datapoint.

    Bounded at ``_RENDER_ATTEMPTS`` rather than looped: a scene that fails every attempt is evidence
    of something real, and grinding through retries would hide a genuine problem behind a longer
    runtime. The exception from the last attempt is the one that propagates, so the recorded reason is
    the one the scene actually died of.
    """
    last: Exception | None = None
    for attempt in range(_RENDER_ATTEMPTS):
        try:
            return renderer.render(
                spec, manifest, np.random.default_rng(scene_seed(config.seed, index)),
            )
        except Exception as exc:  # noqa: BLE001, PERF203 - the retry is the whole point
            last = exc
            if attempt + 1 < _RENDER_ATTEMPTS:
                # The retry rate is the measurement that set _RENDER_ATTEMPTS, and stdout is not
                # something an overnight run keeps. A degraded path that worked is still a degraded
                # path: warning.
                logger.warning("%s: %s on attempt %d/%d, re-rendering the same seed; %s",
                               spec.scene_id, type(exc).__name__, attempt + 1, _RENDER_ATTEMPTS,
                               str(exc).splitlines()[0][:110])
                print(f"[datagen] {spec.scene_id}: {type(exc).__name__}, attempt "
                      f"{attempt + 1}/{_RENDER_ATTEMPTS} ({str(exc).splitlines()[0][:110]})",
                      flush=True)
    assert last is not None  # noqa: S101 (the loop either returned or set it)
    raise last


def _write_scene(writer: DatasetWriter, spec: Any, result: Any, config: DatagenConfig) -> None:
    """Images and the scene JSON. One directory per scene, files a human can open."""
    scene_id = spec.scene_id
    for view in result.views:
        if view.rgb is not None:
            writer.write_png(scene_id, f"{view.name}_rgb.png", view.rgb.astype(np.uint8))
        if view.depth_mm is not None:
            writer.write_png(scene_id, f"{view.name}_depth.png", encode_depth_png(view.depth_mm))
        if view.depth_noisy_mm is not None:
            writer.write_png(
                scene_id, f"{view.name}_depth_noisy.png", encode_depth_png(view.depth_noisy_mm),
            )
        if view.instance_map is not None:
            writer.write_png(
                scene_id, f"{view.name}_instances.png", view.instance_map.astype(np.uint16),
            )
        if view.arm_mask is not None and view.arm_mask.any():
            # A separate file, not another id in `_instances.png`. That map's contract is
            # `value == instance_id + 1`, and the arm is not an instance: squeezing it in would
            # either take an object's id or add one nothing else knows how to skip.
            #
            # Only written when the arm is actually in frame, so a dataset rendered without an arm
            # has no arm files at all rather than a directory of empty ones. Consumers must treat a
            # missing file as "no arm pixels", which is what it means.
            writer.write_png(
                scene_id, f"{view.name}_arm.png", view.arm_mask.astype(np.uint8) * 255,
            )
    writer.write_json(scene_id, "scene.json", {
        "spec": asdict(spec),
        "status": result.status,
        # The one legitimate way spec.objects and the labels disagree: these left the table during the
        # settle and were removed before any camera saw them. Written explicitly, because a silent
        # disagreement between the plan and the labels is indistinguishable from a labelling bug.
        "dropped_objects": list(result.dropped),
        "settled_poses_mm_xyzw": {str(k): v for k, v in result.settled_poses.items()},
        "materials": {str(k): v.as_dict() for k, v in result.materials.items()},
        "arm": {
            "mode": config.render.arm.mode,
            "robot_model": config.render.arm.robot_model,
            "visible_in_views": config.render.arm.visible_in_views,
            # How the arm came to stand there. A parked arm occludes just like a posed one and an
            # image cannot tell them apart, so the difference is recorded rather than inferred.
            "origin": getattr(result.arm, "origin", None),
            "joints_rad": list(getattr(result.arm, "joints_rad", ()) or ()),
            "viewpoint_attempts": getattr(result.arm, "attempts", 0),
            "rejections": list(getattr(result.arm, "rejections", ()) or ()),
        },
        "arm_mode": config.render.arm.mode,
        "render_mode": config.render.mode,
        "views": [
            {
                "name": view.name,
                "outcome": str(view.outcome),
                "camera_position_mm": view.camera_position_mm,
                "camera_look_at_mm": view.camera_look_at_mm,
                # Written, not left to be re-derived. Position and look-at alone leave the handedness
                # to the reader, and a guess costs a 180 deg roll that looks exactly like a permuted
                # label set. These two matrices are the convention.
                "camera_to_base_mm": view.camera_to_base.tolist(),
                "intrinsics": view.intrinsics.tolist(),
                "objects": [asdict(label) for label in view.labels],
            }
            for view in result.views
        ],
    })


def build_dataset(
    config: DatagenConfig,
    *,
    name: str,
    out_root: str | Path | None = None,
    headless: bool = True,
    limit: int | None = None,
    preview: bool = False,
) -> dict[str, Any]:
    """Render the configured dataset. Returns the summary; every scene is on disk before it returns."""
    root = Path(out_root) if out_root is not None else Path(config.output.root)
    writer = DatasetWriter(root, name)

    manifest = build_manifest(config, scenes=limit or config.scenes)
    manifest.audit()  # before Isaac boots: a licence problem must not cost a night of rendering

    stamp = provenance_stamp(
        dataset_name=name, config=config,
        asset_rows=[record.as_row() for record in manifest],
        # The engine, not a hardcoded name: a corpus rendered with `engine: none` must not carry a
        # provenance claiming Isaac made it, which is the exact confusion the engine field exists to
        # prevent, and only the nested config block carries the truth.
        # The mode belongs to isaac and only isaac. `render.mode` selects path-tracing versus
        # rasterisation in the Isaac backend; the raster backends have no such choice, so stamping
        # "mujoco:pathtrace" tells a reader the scene was path-traced when nothing was even shaded.
        # The corpus stamp reads `split(":")[0]` and is unaffected: this is the human-readable half,
        # and a provenance line that reads plausibly and is wrong is worse than a terse one.
        renderer=(f"{config.render.engine}:{config.render.mode}"
                  if config.render.engine == "isaac" else f"{config.render.engine}:raster"),
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    writer.write_provenance(stamp)
    if config.output.attribution_file:
        writer.write_attribution(manifest.attribution_text(dataset_name=name))

    if preview:
        # Written up front, not with the previews: the note explains the files a human will open, and
        # a run that dies at scene 200 still leaves 199 scenes somebody has to make sense of.
        from datagen.preview import write_how_to_read

        write_how_to_read(root / name)

    completed = writer.completed()
    families = plan_families(config)
    total = limit if limit is not None else config.scenes
    pending = [index for index in range(total)
               if layout_scene(config, index, families[index]).scene_id not in completed]
    # The run header for a job measured in hours: what it is about to render, on which renderer, and
    # how much of it a resume already skipped. `completed` alone cannot say that: an index with 300
    # entries looks the same whether this run rendered them or found them.
    logger.info("build %s: %d scene(s) to render, %d already complete "
                "(render=%s, arm=%s, headless=%s, preview=%s)",
                name, len(pending), len(completed), config.render.mode, config.render.arm.mode,
                headless, preview)
    print(f"[datagen] {len(pending)} scene(s) to render, {len(completed)} already done", flush=True)
    if not pending:
        return writer.summary()

    # Through the factory, not `IsaacRenderer` by name. `render.engine` is a string selector, and a
    # fail-closed factory that every construction site bypasses by naming the class directly is
    # inert; the wiring guard cannot see that, because it enumerates boolean flags.
    # `robot.grasping.calculator` is the same shape.
    from datagen.render.engine import build_engine  # imported here: it may boot Isaac

    started = time.perf_counter()
    # Scenes that failed on an earlier run. Their failure is evidence about the scene and must not be
    # read as evidence that this session is broken: see `_CONSECUTIVE_ERROR_LIMIT`.
    known_bad = writer.previously_failed()
    if known_bad:
        logger.info("%d scene(s) failed on an earlier run and will not count towards the "
                    "consecutive-failure guard: %s", len(known_bad), ", ".join(sorted(known_bad)))
    with build_engine(config, headless=headless) as renderer:
        consecutive_errors = 0
        for position, index in enumerate(pending):
            spec = layout_scene(config, index, families[index])
            try:
                result = _render_with_retry(renderer, spec, manifest, config, index)
            except Exception as exc:  # noqa: BLE001 (one bad scene must not cost a night of GPU time)
                # Recorded, not raised. A 300-scene run that dies at scene 200 on one unlucky scene
                # has thrown away the other 199, and the index is the place a failure belongs anyway.
                traceback.print_exc()
                # A scene that already failed is not evidence that isaac is broken. The guard below
                # exists to stop a broken session grinding through 300 scenes; a backlog of known-bad
                # ones sitting at the head of the resume's pending list is the opposite situation,
                # and counting it aborts a restart before a single fresh scene is reached. A broken
                # session still trips the guard, because its fresh scenes fail too.
                if spec.scene_id not in known_bad:
                    consecutive_errors += 1
                # Recorded, not raised, so error, and only the first line: the traceback above is
                # the detail, and repeating it here would double every stack in the file.
                logger.error("%s render_error (%d in a row): %s: %s",
                             spec.scene_id, consecutive_errors, type(exc).__name__,
                             str(exc).splitlines()[0][:200])
                writer.append_index(SceneRecord(
                    scene_id=spec.scene_id, index=index, family=spec.family.value,
                    status="render_error", views=(), view_outcomes={},
                    objects=len(spec.objects), seconds=0.0,
                    note=f"{type(exc).__name__}: {exc}"[:400],
                ))
                print(f"[datagen] {position + 1}/{len(pending)}  {spec.scene_id}  render_error  "
                      f"({type(exc).__name__})", flush=True)
                if consecutive_errors >= _CONSECUTIVE_ERROR_LIMIT:
                    # A session that has failed this many times in a row is broken, not unlucky.
                    logger.error("stopping after %d consecutive failure(s): %d of %d scene(s) were "
                                 "never attempted", consecutive_errors,
                                 len(pending) - position - 1, len(pending))
                    print(f"[datagen] stopping: {consecutive_errors} scenes in a row failed",
                          flush=True)
                    break
                continue
            consecutive_errors = 0
            if result.status == "ok":
                _write_scene(writer, spec, result, config)
                if preview:
                    # Written as the run goes, not at the end: an overnight run that is quietly
                    # producing rubbish should be visible at scene 3, not at scene 300.
                    from datagen.preview import preview_scene

                    preview_scene(writer.scene_dir(spec.scene_id), root / name / "preview")
            writer.append_index(SceneRecord(
                scene_id=spec.scene_id, index=index, family=spec.family.value, status=result.status,
                views=tuple(view.name for view in result.views),
                view_outcomes={view.name: str(view.outcome) for view in result.views},
                objects=len(spec.objects), seconds=round(result.seconds, 3),
                note=(f"dropped {len(result.dropped)} escaped object(s): {list(result.dropped)}"
                      if result.dropped and result.status == "ok" else result.note),
            ))
            print(f"[datagen] {position + 1}/{len(pending)}  {spec.scene_id}  {result.status}  "
                  f"{result.seconds:.1f}s", flush=True)

    summary = writer.summary()
    # The wall clock is the one number nothing else keeps, and it is what the next run is planned
    # from: the status counts are already per-scene in dataset_writer.log.
    logger.info("build %s done in %.1f min: %s",
                name, (time.perf_counter() - started) / 60.0, summary["by_status"])
    print(f"[datagen] done: {summary}", flush=True)
    return summary
