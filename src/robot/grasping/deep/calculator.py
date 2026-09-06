"""The learned generator behind the calculator seam: perceive, propose, decode, never throw.

Selected by `robot.grasping.calculator`. Satisfies `deep.protocol.GraspCandidateGenerator` for the
runtime and `compute()` for the eval-grasps ladder, which are different methods and both required.

The sample is built with `corpus.sample.build_sample`, the same function the trainer uses, so
every convention the net was fitted under lives in exactly one place: metres, xy centred, z above
the support plane, the channel order, the three-way stratification. Train/serve skew cannot arise
here; reimplementing those conventions on this side is what would create it, and that is how a
model scores well offline and proposes nonsense in a cell.

The pick loop needs no change. `compute()` receives `depth_map` (the whole frame), `camera_to_base`,
`geometry_points_base_mm` (the target's fused cloud) and `scene_points_mm` (the neighbours'), and
the camera matrix is a constructor argument like the analytic calculator's. The full perceived cloud
is assembled here: fused objects where fusion supplied them, environment unprojected from the one
depth frame. The corpus is fused across three views and the runtime has one; that difference is
confined to the environment.

The proposals are on the target. Every training sample named a target and masked the loss to it, so
the graspability field this net produces is only meaningful there. Seeding anywhere else reads a
head outside the region it was graded on.

Never throws. The protocol says so and the reason is the pick: an optional generator that raises
takes a cell down. Everything is wrapped, and a failure returns a `GraspResult` with a typed reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover (typing only, so the module stays importable without torch)
    import torch

from src.robot.grasping.types.feedback import GraspFailureReason, GraspResult
from src.robot.grasping.collision.candidate_filter import rejection_reasons
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint
from src.utility.log_cfg import create_logger

__all__ = ["DeepGraspCalculator", "DeepCalculatorConfig"]

from src.robot.grasping.constants import DEEP_GENERATOR_LOG_FILE

logger = create_logger("DeepGraspCalculator", DEEP_GENERATOR_LOG_FILE)

#: The distance below which the retired binned decoder treated two proposals in the same approach
#: and rotation bin as one grasp: those bins are 17.8 deg and 15 deg apart, so sharing both already
#: means the two point the same way, and 10 mm is under half a 2F-85 pad. Nothing reads it, and the
#: set path de-duplicates nothing.
_DUPLICATE_MM = 10.0

#: The corpus's own shape, mirrored. `datagen.corpus.clouds` voxelises objects at 3 mm and the
#: environment at 6 mm, and crops the environment to the object's bounding box plus 150 mm. Serving
#: a net trained on a 3 mm voxel grid over a local neighbourhood a full-resolution pixel grid over
#: the whole frame instead costs 307,200 points and 14.0 s for one VGA call, against a production
#: dense-sampler budget of 150 ms.
#:
#: These are copies, and they have to be: `src` may never import `datagen`. They have to keep
#: matching the values in `datagen.corpus.clouds`; a drifting copy is a second, quieter train/serve
#: skew.
_VOXEL_MM = 3.0
_ENVIRONMENT_VOXEL_MM = 6.0
_ENVIRONMENT_MARGIN_MM = 150.0

#: An over-draw factor for a de-duplicating decode: enough seeds that collapsing duplicates still
#: leaves `max_candidates`. Nothing reads it. The set path draws exactly `max_candidates` seeds and
#: de-duplicates nothing.
_SEEDS_PER_CANDIDATE = 20

#: How many approach directions one seed may propose, when the artifact was trained multi-label.
#: Three, not "everything above a threshold": a threshold on an uncalibrated sigmoid is a number
#: nobody can defend, and a supervised point admits several approaches. Nothing reads it: the
#: fan-out per seed is the artifact's own `head.slots`, and no single-label artifact loads.
_APPROACHES_PER_SEED = 3


@dataclass(frozen=True, slots=True)
class DeepCalculatorConfig:
    """What the runtime needs to build this. Everything else lives in the artifact."""

    artifact_path: str
    #: Pinhole K, exactly as the analytic calculator takes it. Needed to unproject the depth frame
    #: into the environment cloud; without one this calculator refuses rather than guessing an FOV.
    camera_matrix: tuple[tuple[float, ...], ...] | None = None
    max_candidates: int = 12
    #: Which device to run on, or None to resolve it the way every other model wrapper does.
    #:
    #: Resolution goes through `utility.device.get_device`, so `WILLY_DEVICE` is honoured on this
    #: path as on every other, MPS is a candidate on a Mac, and landing on the CPU warns: a silent
    #: order-of-magnitude slowdown is exactly what a warning is for.
    #:
    #: A named device that is not there raises rather than falling back. `device: cuda` on a box
    #: with no CUDA would otherwise keep picking on the CPU, which files a CPU run's latency under a
    #: GPU run's name.
    device: str | None = None
    #: Points below this graspability probability are never seeded. 0.5 is the natural threshold for
    #: a calibrated head and this head is not calibrated, so the value is a knob, to be set once the
    #: score distribution of a given artifact is known.
    minimum_score: float = 0.5
    support_height_mm: float = 0.0

    #: The jaw. Zero means "not told", and then nothing is refused.
    #:
    #: Refused, not clamped. Clamping a too-wide prediction to the jaw silently rewrites the model's
    #: answer into one it did not make, and the contacts then sit inside the object. The analytic
    #: stack filters at generation for the same reason.
    min_grip_width_mm: float = 0.0
    max_grip_width_mm: float = 0.0


def _resolve_device(preference: str | None) -> "torch.device":
    """The device this calculator runs on, resolved the way every other model wrapper resolves it.

    `get_device` carries three behaviours a local resolver does not:

    * ``WILLY_DEVICE`` is honoured, so a box pinned to one device runs its detector, its segmenter
      and its grasp generator in the same place.
    * MPS is a candidate, so on a Mac the generator does not sit on the CPU while every other model
      uses the GPU.
    * Landing on the CPU warns: a silent order-of-magnitude slowdown is what a warning is for, and
      this calculator sits in the pick loop's inner path.

    A named device that is not available raises rather than falling back, so a config naming a
    device the machine does not have cannot file a CPU run's latency under a GPU run's name.

    An indexed device passes straight through. `get_device` accepts only auto/cuda/cpu/mps, so
    routing everything through it would make `device: "cuda:1"` raise on a multi-GPU box. An ordinal
    names one device explicitly, and torch fails loudly on its own when that device is not there.

    Torch is imported inside, so the module stays importable without it: the fence the module
    docstring describes.
    """
    import torch  # noqa: PLC0415 (the torch fence)

    from src.utility.device import get_device  # noqa: PLC0415 (the torch fence)

    if preference is not None and ":" in preference:
        return torch.device(preference)
    return get_device(preference)


class DeepGraspCalculator:
    """A learned 6-DoF grasp generator that satisfies the calculator seam."""

    def __init__(self, config: DeepCalculatorConfig | None = None, **overrides: Any) -> None:
        # `**overrides` swallows whatever a construction site passes for the analytic calculator.
        # The protocol requires an implementation to ignore arguments it does not understand rather
        # than crash, so a new config block cannot break a second implementation.
        known = {"artifact_path", "camera_matrix", "max_candidates", "device", "minimum_score",
                 "support_height_mm"}
        if config is None:
            config = DeepCalculatorConfig(**{k: v for k, v in overrides.items() if k in known})
        self.config = config
        self.render_debug_images: bool = False
        self.last_debug_image_png: bytes | None = None
        self.last_result: GraspResult = GraspResult()
        self.last_telemetry: dict[str, Any] = {}
        #: Candidates the jaw refused on the last decode. Reported rather than dropped silently:
        #: "proposed nothing" and "proposed nothing this gripper can hold" call for opposite
        #: repairs, and only the second one is fixed by a different gripper.
        self._refused_width = 0
        # An attribute, not only a config field, and the difference is a silent stand-down.
        # `pick_loop` reads `getattr(self.calculator, "camera_matrix", None)`; without the attribute
        # that lookup returns None, `_target_cloud_base_mm` bails, and `support.refine_from_target`
        # does nothing under a deep calculator while reporting nothing at all.
        self.camera_matrix = self.config.camera_matrix
        self._model: Any = None
        self._net_config: Any = None
        #: Which generator family the artifact belongs to: `"binned"` or `"set"`.
        #:
        #: `train-set` writes `kind="set_grasp_generator"`, and that is the only kind
        #: `_ensure_model` loads; the retired binned family has no loader, so the value stamped
        #: onto the telemetry is always `"set"`. One config key, one calculator.
        self._family: str = "binned"
        self._set_gripper: str = ""
        self._step: Any = None
        self._multilabel: bool = False
        self._sample_spec: dict[str, Any] = {}

    # ------------------------------------------------------------------ model

    def preload(self) -> None:
        """Load the artifact now and raise if it cannot be loaded. For callers that can still refuse.

        `_compute_result` catches everything, because the protocol forbids raising and a robot cell
        must not go down over one bad frame. That is right for the runtime and wrong for an
        evaluation: a configuration failure fails identically on every call, so an artifact that
        cannot load turns into "no candidates" for every object, and a ladder then reports a model
        that finds nothing instead of a model that was never there. Pointing a rung at a training
        checkpoint is the case that reaches this.

        Same rule, not a second one: this calls the very loader the runtime uses, so an artifact
        that passes here is an artifact the cell can load.
        """
        self._ensure_model()

    def _ensure_model(self) -> None:
        """Load the artifact on first use. Lazy, so importing this module costs no torch.

        One family: the retired binned trainer has no loader here, so nobody can read numbers off a
        system nobody trains.

        The loader's refusal is passed through rather than re-wrapped. Rebuilding a friendlier
        message means recovering the artifact's `kind` by splitting the text of the error being
        replaced; the refusal lives in `set_artifact.py`, next to the only line that knows the kind.
        """
        if self._model is not None:
            return
        self._load_set_generator(Path(self.config.artifact_path))

    def _load_set_generator(self, path: Path) -> None:
        """The K-slot family, loaded through its own reader rather than a second copy of one.

        Its own reader on purpose. `load_set_generator` already refuses four things this file would
        otherwise have to re-check: the wrong kind, a newer artifact version, a gripper this build
        cannot resolve, and a state dict that does not fit the config it travelled with. A second
        implementation of those checks is a second place for them to drift.
        """
        import dataclasses  # noqa: PLC0415

        from src.robot.grasping.deep.set_artifact import (  # noqa: PLC0415
            load_set_generator,
        )

        device = _resolve_device(self.config.device)
        loaded = load_set_generator(path, device=str(device))
        self._model = loaded.net
        self._net_config = loaded.net.config
        self._family = "set"
        # The artifact's stamp conditions the net, and the cell's hand cannot override it.
        # `DeepCalculatorConfig` has no `gripper` field and `build_calculator` passes none, so
        # `wanted` is always None, the refusal below is unreachable, and `loaded.gripper` is what
        # the net is conditioned on. `robot.grasping.deep_generator.gripper` is a schema key with
        # no reader. A model fitted across `narrow_55`, `slim_pad` and `wide_140` therefore
        # proposes for the stamped hand alone, and the difference is not cosmetic: the width it
        # proposes follows the hand it is conditioned on.
        #
        # The rule the refusal states: a hand the artifact never trained across is refused, not
        # served, because a conditioning vector the model has not seen produces grasps that read as
        # a bad model rather than as a wrong hand, which is the hardest failure in this stack to
        # attribute.
        wanted = getattr(self.config, "gripper", None)
        if wanted is not None and wanted not in loaded.grippers:
            raise ValueError(
                f"robot.grasping.deep.gripper is {wanted!r}, and {path.name} was trained across "
                f"{', '.join(loaded.grippers)}. Refusing to condition on a hand this model has never "
                f"seen: the grasps would read as a bad model rather than as a wrong gripper. Train "
                f"an artifact whose corpus carries {wanted!r}, or leave the field unset to use the "
                f"hand the artifact was stamped for ({loaded.gripper}).")
        self._set_gripper = wanted or loaded.gripper
        # The spec the artifact was fitted with. Serving a net a different sample than it was
        # trained on is skew that reads as a bad model.
        self._sample_spec = dataclasses.asdict(loaded.sample)
        # The trained seed mixture, kept, so this calculator can ask what the artifact was fitted
        # under instead of assuming `predicted=1.0`. That assumption draws every seed from one blob
        # of the model's own nearly-flat field, so the proposals collapse onto each other; the
        # artifact's own mixture keeps them distinct. `deep propose` and a cell running
        # `calculator: deep` have to read this the same way, or a ladder rung grades a system the
        # cell does not run.
        self._step = loaded.step
        self._multilabel = True         # a set head is multi-label by construction
        # `minimum_score` is consulted by nothing, and the log line below says so once at load so
        # that an operator does not tune a key that decides nothing. It gates the graspability
        # field of the retired binned decoder, which this build does not load; the set family ranks
        # by its own per-slot confidence.
        #
        # It is not wired across, deliberately. The set head's confidence is not calibrated, so its
        # range need not overlap `minimum_score`'s default of 0.5 at all: applying that default to
        # it can drop every candidate, and the cell then emits nothing while looking like a broken
        # model. The two numbers answer different questions.
        logger.info(
            "note: grasping.deep_generator.minimum_score (%.2f) is not consulted by the set family; "
            "it gates the binned family's graspability field. This family ranks by its own per-slot "
            "confidence, which is not calibrated and need not overlap that threshold",
            float(self.config.minimum_score))
        logger.info("deep set generator ready: %s, %d parameter(s), gripper %s, device %s",
                    path.name, sum(p.numel() for p in loaded.net.parameters()), loaded.gripper,
                    device)

    def _propose_set(self, sample: dict[str, Any]) -> list[GraspPoint]:
        """K ranked grasps per seed, decoded into the cell's frame.

        The order is the product and it is preserved. A cell takes the first candidate it can
        reach, so `decode_set_prediction` ranks by the head's own confidence and this keeps that
        order. Returning slots in emission order would hand the cell slot zero of seed zero whatever
        the head believed.

        Seeds come from the model's own graspability field and only from points on the target.
        Seeding elsewhere grades the field on points it did not choose, and every training sample
        masked the loss to the target, so a seed on the table reads a head outside where it was
        graded.
        """
        import dataclasses  # noqa: PLC0415

        import numpy as np  # noqa: PLC0415
        import torch  # noqa: PLC0415

        from src.robot.grasping.deep.net.gripper import (  # noqa: PLC0415
            JAW_GEOMETRY,
            gripper_vector,
        )
        from src.robot.grasping.deep.net.set_targets import (  # noqa: PLC0415
            sample_seeds,
        )
        from src.robot.grasping.deep.eval.propose import serving_shares  # noqa: PLC0415
        from src.robot.grasping.deep.set_decode import (  # noqa: PLC0415
            decode_set_prediction,
        )

        net = self._model
        device = next(net.parameters()).device
        cloud = torch.as_tensor(np.asarray(sample["points_m"])[None], dtype=torch.float32,
                                device=device)
        features = torch.as_tensor(np.asarray(sample["features"])[None], dtype=torch.float32,
                                   device=device)
        features = features[..., :net.backbone.config.in_features]
        gripper = gripper_vector(self._set_gripper).to(device)
        generator = torch.Generator().manual_seed(0)
        with torch.no_grad():
            encoded = net.encode(cloud, features)
            field = net.graspability_logit(encoded)[0]
            on_target = torch.as_tensor(
                np.asarray(sample["features"])[:, 3] > 0.5, dtype=torch.bool)
            if not bool(on_target.any()):
                # A runtime whose target flag is missing seeds from every point rather than being
                # refused.
                on_target = torch.ones(cloud.shape[1], dtype=torch.bool)
            # The artifact's own mixture, redistributed for serve time by `serving_shares`. See
            # `_load_set_generator`: an all-predicted draw concentrates on one blob of the model's own
            # nearly-flat field, whose points carry the same feature direction, and a head handed
            # one input returns one answer.
            draw = sample_seeds(count=int(self.config.max_candidates),
                                labelled=torch.zeros(cloud.shape[1], dtype=torch.bool),
                                score=field.detach().cpu(), candidate=on_target,
                                shares=serving_shares(getattr(self._step, "shares", None)),
                                generator=generator)
            picks = draw.point_index.to(device)
            # The cloud goes through, because a stage-3 net crops it. `propose` refuses
            # rather than silently skipping the crop, so this is what keeps the flag live
            # here as well as in training.
            prediction = net.propose(encoded, torch.zeros_like(picks), picks,
                                     gripper.expand(len(picks), gripper.shape[-1]),
                                     cloud)
        # The decode is a CPU function and it mixes the prediction with the cloud, so one tensor
        # left on the GPU is a RuntimeError on the first scene rather than a slow path.
        prediction = dataclasses.replace(prediction, **{
            f.name: getattr(prediction, f.name).cpu()
            for f in dataclasses.fields(prediction)
            if isinstance(getattr(prediction, f.name), torch.Tensor)})
        aperture = float(JAW_GEOMETRY[self._set_gripper]["aperture_mm"])
        grasps = decode_set_prediction(
            prediction, picks.cpu(), cloud[0].cpu(),
            centre_xy_mm=np.asarray(sample["centre_xy_mm"], dtype=np.float64),
            support_height_mm=float(np.asarray(sample["support_height_mm"]).reshape(-1)[0]),
            min_width_mm=1.0, max_width_mm=aperture,
            # The vocabulary comes from the artifact, so a role is named by the net that was
            # trained, never by whatever this build happens to think the roles are. Empty on a net
            # without an affordance head, and the decoder then reports no role rather than an
            # invented one.
            part_roles=tuple(self._model.config.head.part_roles))

        out: list[GraspPoint] = []
        refused_width = 0
        for grasp in grasps:
            width_mm = float(grasp.width_mm)
            if not self._fits_the_jaw(width_mm):
                # Counted rather than dropped silently; see `_refused_width`.
                refused_width += 1
                continue
            out.append(GraspPoint(
                position=np.asarray(grasp.position_mm, dtype=np.float64),
                approach=np.asarray(grasp.approach, dtype=np.float64),
                axis=np.asarray(grasp.axis, dtype=np.float64),
                grip_width_mm=width_mm,
                score=float(grasp.confidence),
                frame=GraspFrame.BASE,
                label="deep",
                metadata={
                    "generator": "deep",
                    "family": "set",
                    "slot": int(grasp.slot),
                    "seed_index": int(grasp.seed_index),
                    "seed_position_mm": tuple(float(v) for v in grasp.seed_position_mm),
                    # The head is not calibrated, and the key says so wherever this travels.
                    "graspability": float(grasp.confidence),
                    "score_is_calibrated": False,
                    # The role has to leave the calculator on the `GraspPoint`, or nothing outside
                    # this function can see it and the affordance head is unread. A source file
                    # holding the string `config.head.part_roles` is not the same thing as a value
                    # in this return.
                    #
                    # `None` on every net without the head, which is every artifact written so
                    # far, and a caller must read the absence as "this model cannot tell me" rather
                    # than as "no part". The two are different answers.
                    "part_role": grasp.part_role,
                    "part_score": grasp.part_score,
                },
            ))
            if len(out) >= self.config.max_candidates:
                break
        self._refused_width = refused_width
        return out

    # ------------------------------------------------------------------ cloud

    def _scene_dict(self, mask: np.ndarray, depth_map: np.ndarray, camera_to_base: np.ndarray,
                    target_points_base: np.ndarray | None,
                    neighbour_points_base: np.ndarray | None,
                    other_masks: list[np.ndarray]) -> dict[str, np.ndarray]:
        """A corpus-shaped scene from what the runtime has. Instance 0 is the target.

        The environment is whatever the depth frame shows that no segmentation claimed. Objects come
        from the fused clouds when they were supplied, because those see sides one camera cannot.
        They fall back to the single view otherwise.
        """
        from src.robot.grasping.geometry.normals import (  # noqa: PLC0415
            estimate_surface_normals,
        )
        # The fusion's own unprojection, not a second one written here and not datagen's: `src` may
        # never import `datagen`, and a runtime cloud built by different arithmetic than the corpus
        # is a frame defect. It is equal to `datagen.render.camera.unproject_to_base` on the same
        # input: identical pixel selection, a maximum difference of 3.05e-05 mm, which is float32
        # rounding rather than a convention.
        from src.robot.grasping.multiview.scene_geometry import to_base_mm  # noqa: PLC0415

        intrinsics = np.asarray(self.config.camera_matrix, dtype=np.float64)
        claimed = np.asarray(mask, dtype=bool).copy()
        for other in other_masks:
            claimed |= np.asarray(other, dtype=bool)
        environment = np.asarray(to_base_mm(~claimed, depth_map, intrinsics, camera_to_base),
                                 dtype=np.float64).reshape(-1, 3)

        target = (np.asarray(target_points_base, dtype=np.float64).reshape(-1, 3)
                  if target_points_base is not None and len(target_points_base)
                  else np.asarray(to_base_mm(mask, depth_map, intrinsics, camera_to_base),
                                  dtype=np.float64).reshape(-1, 3))
        # Whether the object clouds came from fusion, which decides how their normals are oriented.
        fused_objects = target_points_base is not None and len(target_points_base) > 0
        if neighbour_points_base is not None and len(neighbour_points_base):
            neighbours = np.asarray(neighbour_points_base, dtype=np.float64).reshape(-1, 3)
        elif other_masks:
            # The single-view fallback, and it is not optional. Neighbour pixels are always cut out
            # of the environment above, so without this they are cut and never put back and the net
            # sees a hole exactly where the clutter is: measured on a two-square frame, 64 neighbour
            # pixels become 0 neighbour points. The pick loop always passes `other_object_masks` and
            # only sometimes passes fused neighbour clouds, so this is the common runtime shape. The
            # target has the same fallback for the same reason.
            neighbours = np.vstack([
                np.asarray(to_base_mm(np.asarray(other, dtype=bool), depth_map, intrinsics,
                                      camera_to_base), dtype=np.float64).reshape(-1, 3)
                for other in other_masks])
        else:
            neighbours = np.zeros((0, 3))

        # Crop, then voxelise: the corpus's order, and the cheap one, because cropping first is what
        # makes the environment's voxelisation affordable at all.
        if len(target) and len(environment):
            low = target.min(axis=0) - _ENVIRONMENT_MARGIN_MM
            high = target.max(axis=0) + _ENVIRONMENT_MARGIN_MM
            near = ((environment >= low) & (environment <= high)).all(axis=1)
            # Not a hard filter: an environment cropped to nothing (a target seen against empty space,
            # or a margin that happens to exclude the plate) would leave the net with no support
            # geometry at all, which is worse than geometry at the wrong density.
            if near.any():
                environment = environment[near]
        target = _voxelise(target, _VOXEL_MM)
        neighbours = _voxelise(neighbours, _VOXEL_MM)
        environment = _voxelise(environment, _ENVIRONMENT_VOXEL_MM)

        points = np.vstack([target, neighbours, environment])
        instance = np.concatenate([
            np.zeros(len(target), dtype=np.int16),
            np.ones(len(neighbours), dtype=np.int16),
            np.full(len(environment), -1, dtype=np.int16),
        ])
        camera = np.asarray(camera_to_base, dtype=np.float64)[:3, 3]
        normals = np.zeros(points.shape, dtype=np.float32)
        valid = np.zeros(len(points), dtype=bool)
        for owner in np.unique(instance):
            rows = instance == owner
            subset = points[rows]
            if len(subset) < 3:
                continue
            estimated = estimate_surface_normals(subset)
            raw = np.asarray(estimated.normals, dtype=np.float64)
            if owner >= 0 and fused_objects:
                # A fused cloud spans sides one camera cannot see, so "point it at the camera" is
                # wrong for every point the other camera contributed. On a two-faced box the far
                # face is half the cloud, and every point of it ends up pointing into the object,
                # where the normal is three of the net's five input channels.
                #
                # Outward from the cloud's own centroid instead. Right for anything star-shaped,
                # which is most graspable geometry, and wrong on a concavity such as the inner wall
                # of a cup. The corpus does better: it orients each voxel at the mean of the cameras
                # that actually saw it, and matching that here needs the fusion to hand over its
                # per-camera map rather than one merged cloud.
                outward = subset - subset.mean(axis=0)
                length = np.linalg.norm(outward, axis=1, keepdims=True)
                outward = np.divide(outward, length, out=np.zeros_like(outward), where=length > 1e-9)
                raw[np.einsum("ij,ij->i", raw, outward) < 0.0] *= -1.0
            else:
                # Towards the camera, the corpus's rule for a single view: the observer is in free
                # space relative to a surface it can see, so a normal pointing at it points out.
                toward = camera[None, :] - subset
                raw[np.einsum("ij,ij->i", raw, toward) < 0.0] *= -1.0
            normals[rows] = raw.astype(np.float32)
            valid[rows] = np.asarray(estimated.valid_mask, dtype=bool)

        empty3 = np.zeros((0, 3), dtype=np.float32)
        return {
            "points_mm": points.astype(np.float32),
            "normals": normals,
            "normal_valid": valid,
            "view_count": np.ones(len(points), dtype=np.uint8),
            "instance_id": instance,
            "grasp_position_mm": empty3, "grasp_approach": empty3, "grasp_axis": empty3,
            "grasp_width_mm": np.zeros(0, dtype=np.float32),
            "grasp_instance": np.zeros(0, dtype=np.int16),
            "contact_points_mm": empty3,
            "contact_grasp_index": np.zeros(0, dtype=np.int32),
            "object_instance": np.asarray([0], dtype=np.int16),
            "object_asset_id": np.asarray(["runtime"], dtype="<U80"),
        }

    # ------------------------------------------------------------------ decode


    def _fits_the_jaw(self, width_mm: float) -> bool:
        """Whether this gripper can actually open to that span.

        Both bounds default to 0.0, meaning "not told"; a caller that passes neither gets no
        refusals.
        """
        low = float(self.config.min_grip_width_mm)
        high = float(self.config.max_grip_width_mm)
        if high > 0.0 and width_mm > high:
            return False
        return not (low > 0.0 and width_mm < low)
    # ------------------------------------------------------------------ seam

    def compute(self, segmentation: Any, depth_map: np.ndarray,
                T_cam_to_base: Any = None, *args: Any, **kwargs: Any) -> list[GraspPoint]:
        """Propose grasps. Returns the ranked candidates and never raises."""
        del args
        self.last_result = self._compute_result(segmentation, depth_map, T_cam_to_base, **kwargs)
        return list(self.last_result.candidates)

    def compute_result(self, *args: Any, **kwargs: Any) -> GraspResult:
        """The runtime's entry point. `compute()` is the ladder's; both are required."""
        self.compute(*args, **kwargs)
        return self.last_result

    def _compute_result(self, segmentation: Any, depth_map: np.ndarray,
                        T_cam_to_base: Any = None, **kwargs: Any) -> GraspResult:
        try:
            return self._propose(segmentation, depth_map, T_cam_to_base, **kwargs)
        except Exception:  # noqa: BLE001 (the protocol forbids raising; a cell must not go down)
            logger.exception("deep generator failed; returning no candidates")
            return GraspResult(reasons=(GraspFailureReason.NO_CANDIDATES_GENERATED,),
                               telemetry={"generator": "deep", "error": True})

    def _propose(self, segmentation: Any, depth_map: np.ndarray, T_cam_to_base: Any,
                 **kwargs: Any) -> GraspResult:

        import dataclasses  # noqa: PLC0415

        from src.robot.grasping.deep.corpus.sample import (  # noqa: PLC0415
            SampleSpec,
            build_sample,
        )

        if self.config.camera_matrix is None:
            return GraspResult(reasons=(GraspFailureReason.NO_CANDIDATES_GENERATED,),
                               telemetry={"generator": "deep", "reason": "no camera_matrix"})
        mask = _mask_of(segmentation)
        if mask is None or not mask.any():
            return GraspResult(reasons=(GraspFailureReason.EMPTY_MASK,),
                               telemetry={"generator": "deep"})
        transform = kwargs.get("camera_to_base")
        transform = transform if transform is not None else T_cam_to_base
        if transform is None:
            return GraspResult(reasons=(GraspFailureReason.NO_CANDIDATES_GENERATED,),
                               telemetry={"generator": "deep", "reason": "no camera_to_base"})
        matrix = _matrix_of(transform)

        self._ensure_model()
        scene = self._scene_dict(
            mask, np.asarray(depth_map, dtype=np.float64), matrix,
            kwargs.get("geometry_points_base_mm"),
            _to_base(kwargs.get("scene_points_mm"), matrix),
            [np.asarray(m, dtype=bool) for m in (kwargs.get("other_object_masks") or [])],
        )
        if not len(scene["points_mm"]):
            return GraspResult(reasons=(GraspFailureReason.NO_VALID_DEPTH,),
                               telemetry={"generator": "deep"})

        # The trained spec, with exactly two fields overridden and both for a reason:
        #   rotate_z:          augmentation is a training device; rotating a live scene would move
        #                      the grasp it returns.
        #   support_height_mm: the cell's surface, which no artifact can know.
        # Every other field (point budget, stratum fractions, graspability radius, bin counts) comes
        # from the artifact, because serving a net a different sample than it was fitted to is skew
        # that looks exactly like a bad model.
        known = {f.name for f in dataclasses.fields(SampleSpec)}
        spec = SampleSpec(**{**{k: v for k, v in self._sample_spec.items() if k in known},
                             "rotate_z": False,
                             "support_height_mm": self.config.support_height_mm})
        sample = build_sample(scene, np.random.default_rng(0), spec, target_instance=0)
        # One family: the set head, decoded by `_propose_set`.
        candidates = self._propose_set(sample)
        proposed = len(candidates)
        candidates, rejection = self._reject(candidates, scene, matrix, kwargs)
        telemetry = {
            "generator": "deep",
            # Which family answered. Families behind one config key report the same metrics, so a
            # rate that does not say which one produced it cannot be compared against anything.
            "family": self._family,
            "points": int(len(scene["points_mm"])),
            "sampled": int(len(sample["points_m"])),
            "target_points": int((sample["features"][:, 3] > 0.5).sum()),
            # Both denominators. `proposed` is what the net emitted, `candidates` what survived
            # geometry; a rate quoted over the wrong one is a misread. The rejection counters below
            # make the difference readable.
            "proposed": proposed,
            "candidates": len(candidates),
            "rejected_grip_width": int(self._refused_width),
            **rejection,
        }
        self.last_telemetry = telemetry
        if not candidates:
            # The reasons the recovery orchestrator routes on: ALL_COLLIDED agitates the container,
            # ALL_TABLE_CONFLICT moves to the next target and rescans. A cell that can only say
            # NO_VALID_GRASP reaches neither and retries the same thing instead of shaking the bin.
            # Same mapping the analytic path uses, from the same counters.
            return GraspResult(reasons=rejection_reasons(rejection), telemetry=telemetry)
        return GraspResult(candidates=tuple(candidates), telemetry=telemetry,
                           top_score=candidates[0].score,
                           metadata={"calculator": "deep"})

    def _reject(self, candidates: list[GraspPoint], scene: dict[str, Any],
                camera_to_base: np.ndarray,
                kwargs: dict[str, Any]) -> tuple[list[GraspPoint], dict[str, float]]:
        """Run the shared geometric rejection stage over BASE-frame candidates.

        `gripper_model`, `support_plane`, `min_table_clearance_mm` and `workspace` reach `_propose`
        through `**kwargs`, and without this stage a width comparison is the only filter. On the
        analytic stack the equivalent filter is what raises precision, and `SafetyPreflight`
        lives in the driver, so an unfiltered bad candidate does not fall through to the next one:
        the pick fails at the arm.

        Frames, which is the part that could go wrong. Verified against the producers:
          * candidates            BASE  (`_propose_set` stamps `frame=GraspFrame.BASE` on each one)
          * `support_plane`       BASE  (`resolve_support_plane` stamps `Frame.BASE` on every path)
          * `geometry_points_base_mm` BASE (the name says so, and the fused cloud is BASE throughout)
          * `scene_points_mm`     CAMERA, already converted by `_to_base` before it reaches `scene`
          * `rigid_obstacle_points_mm` CAMERA: the container walls start in BASE and the pick loop
            converts them to CAMERA for the analytic path, so they are converted back here
        Nothing in the collision stack compares frames, so a mistake does not raise: it returns a
        plausible wrong number. `filter_candidates` refuses a pose/plane frame mismatch for exactly
        that reason.
        """
        from src.robot.grasping.collision import filter_candidates, grasp_point_to_pose

        plane = kwargs.get("support_plane")
        workspace = kwargs.get("workspace")
        gripper_model = kwargs.get("gripper_model")
        if plane is None and workspace is None and gripper_model is None:
            # Nothing to filter with: a caller that supplies no geometry gets exactly the candidates
            # the net proposed.
            return (candidates, {})

        # Everything that is not the target, derived from the instance ids. There is no
        # `environment_mm` key: `scene.get("environment_mm", empty)` returns an empty cloud on every
        # call, the collision check then runs against nothing, finds nothing, and the stage looks
        # like it works while rejecting zero candidates. A default that silences a check is worse
        # than a KeyError.
        #
        # Instance 0 is the target (`_scene_dict`); -1 is the environment and any other id is a
        # neighbouring object. Both are obstacles for the gripper.
        points = np.asarray(scene["points_mm"], dtype=np.float64)
        instance = np.asarray(scene["instance_id"])
        obstacles = points[instance != 0] if len(points) == len(instance) else points
        walls_cam = kwargs.get("rigid_obstacle_points_mm")
        if walls_cam is not None and len(walls_cam):
            walls_base = _to_base(np.asarray(walls_cam, dtype=np.float64), camera_to_base)
            if walls_base is not None and len(walls_base):
                obstacles = (walls_base if not len(obstacles)
                             else np.vstack([obstacles, walls_base]))

        poses = []
        keep: list[int] = []
        for index, candidate in enumerate(candidates):
            try:
                poses.append(grasp_point_to_pose(candidate))
                keep.append(index)
            except ValueError:
                # A degenerate axis pair. Dropping it here rather than letting the validator raise:
                # one unusable candidate must not cost the whole attempt.
                continue

        outcome = filter_candidates(
            poses,
            scene_points_mm=obstacles if len(obstacles) else None,
            support_plane=plane,
            workspace=workspace,
            gripper_model=gripper_model,
            min_table_clearance_mm=float(kwargs.get("min_table_clearance_mm", 5.0)),
            collision_margin_mm=float(kwargs.get("collision_margin_mm", 0.0)),
        )
        return ([candidates[keep[i]] for i in outcome.kept], outcome.telemetry)


def _matrix_of(transform: Any) -> np.ndarray:
    """A 4x4 CAMERA to BASE matrix from whatever the caller had.

    `to_matrix()` first, then `.matrix` for anything that really carries one, then the array itself.

    The pick loop hands a typed `Transform`, which has `to_matrix()` and has no `matrix` attribute
    and no `__array__`. Reading `.matrix` first therefore falls through to the `Transform` itself,
    `np.asarray` raises `TypeError`, and the blanket `except Exception` in `_compute_result` turns
    that into `NO_CANDIDATES_GENERATED` with `error: True`: zero candidates on every attempt, under
    a refusal reason that reads like a hard scene.

    A caller that hands a raw `np.eye(4)`, as the ladder does, exercises none of that, so the array
    path working says nothing about the type the production caller passes.
    """
    to_matrix = getattr(transform, "to_matrix", None)
    if callable(to_matrix):
        return np.asarray(to_matrix(), dtype=np.float64)
    matrix = getattr(transform, "matrix", transform)
    array = np.asarray(matrix, dtype=np.float64)
    if array.shape != (4, 4):
        raise ValueError(
            f"camera_to_base must be a 4x4 matrix or carry to_matrix(); got shape {array.shape} "
            f"from {type(transform).__name__}")
    return array


def _voxelise(points: np.ndarray, edge_mm: float) -> np.ndarray:
    """One point per occupied voxel, at the centroid of what fell in it.

    The same reduction `datagen.corpus.clouds` applies (floor-divide into integer cells, group, take
    the mean), so a runtime cloud has the density the net was fitted to rather than the camera's.
    """
    if not len(points) or edge_mm <= 0.0:
        return points
    keys = np.floor(points / edge_mm).astype(np.int64)
    _unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    groups = int(inverse.max()) + 1
    counts = np.bincount(inverse, minlength=groups).astype(np.float64)
    return np.column_stack([
        np.bincount(inverse, weights=points[:, axis], minlength=groups) / counts
        for axis in range(3)])


def _mask_of(segmentation: Any) -> np.ndarray | None:
    """The boolean mask out of whatever shape a caller passes: an array or a segmentation object."""
    if segmentation is None:
        return None
    for attribute in ("mask", "binary_mask", "segmentation"):
        candidate = getattr(segmentation, attribute, None)
        if candidate is not None:
            return np.asarray(candidate, dtype=bool)
    return np.asarray(segmentation, dtype=bool)


def _to_base(points_cam: Any, camera_to_base: np.ndarray) -> np.ndarray | None:
    """CAMERA-frame points to BASE. `scene_points_mm` arrives in CAMERA and everything here is BASE.

    A named function rather than an inline matrix multiply, so that a quantity crossing frames is
    visible at every call site.
    """
    if points_cam is None:
        return None
    points = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
    if not len(points):
        return None
    return (camera_to_base[:3, :3] @ points.T).T + camera_to_base[:3, 3]
