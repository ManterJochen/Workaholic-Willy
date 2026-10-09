"""Live-camera perception: an RGB-D frame, GroundingDINO, SAM2, one :class:`PerceptionFrame`.

This is the real-hardware twin of ``willy_sim/perception/vision.py``'s
:class:`MultiObjectVisionPerceptionSource`, with the Isaac session-stepping removed and an RGB-D
streamer in place of an Isaac ``Camera``. The whole vision-driven pick path waits on it: the sim
feeds this stack, and a physical camera never has.

It lives under ``robot/`` and not ``camera/`` because the dependency stack only ever crosses
``robot -> camera`` (measured: ``grep "from src.robot" src/camera`` is empty). This adapter needs
both a camera streamer and ``robot.grasping``'s :class:`PerceptionFrame`, so it cannot sit in the
camera package without inverting that edge. It takes its streamer, detector and segmenter as
injected dependencies (duck-typed, exactly like the sim source), so this module imports with no
``pyrealsense2`` and no torch; the ``__main__`` exerciser is what builds the heavy pieces.

Bucket (3): the behaviour on real D435 depth, which returns holes (0) inside a mask and leaks at
mask edges, is unmeasured until a camera exists. The two robustness fixes carried over from the sim
source, top-referenced depth over holes and mask completion, are the right starting point and not a
proven answer.

Where a cell hides a task's own places from its detector (``robot.grasping.hide_own_places``, the
owner, 2026-10-08 night), a pick hands :meth:`RealSenseVisionPerceptionSource.acquire` a ``hide``:
the detector then reads a copy of the frame whose pixels standing in the regions the task keeps out
(its bin, the circles about its drops) are painted in the colour round them (:func:`kept_out_pixels`,
:func:`painted_copy`), so it boxes none of the parts already placed. The segmenter, the colour check,
the depth and the frame handed on are the real ones.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np

from src.robot.constants import create_robot_logger
from src.robot.core.errors import PerceptionFrameMoved
from src.robot.core.shutter_motion import PICK_FRAME_WARMUP_GRABS, ShutterStamp
from src.robot.grasping.types.perception import PerceptionFrame
from src.robot.perception.colour_check import (
    ColourCheck,
    ColourClipped,
    ask_colour,
    colour_named,
    judge_colour,
    refused_label,
)
from src.robot.perception.mask_completion import (
    DEFAULT_MASK_COMPLETION,
    MaskCompletion,
    complete_mask,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from src.geometry import Pose
    from src.robot.perception.kept_scene import Following

    #: What a pick hands :meth:`RealSenseVisionPerceptionSource.acquire` to follow the parts a task kept: called with
    #: the frame's measured depth (mm), its BGR image, its lens and the tool pose at its shutter, it answers the boxes
    #: to cut, or why the frame is grounded (``src/robot/perception/kept_scene.py``).
    Follow = Callable[[np.ndarray, np.ndarray, np.ndarray, "Pose | None"], Following]
    #: What a pick hands :meth:`RealSenseVisionPerceptionSource.acquire` to paint the regions its task keeps out over the
    #: detector's copy of the frame: called with the frame's measured depth (mm), its lens and the tool pose at its
    #: shutter, it answers the pixels to paint (:func:`kept_out_pixels`), ``None`` for none.
    Hide = Callable[[np.ndarray, np.ndarray, "Pose | None"], "np.ndarray | None"]

__all__ = ["RealSenseVisionPerceptionSource", "kept_out_pixels", "painted_copy"]

#: The source's own lines, to the robot log and ``perception_source.log``: what the colour check made of each part,
#: and what the detector called it.
_LOG: logging.Logger = create_robot_logger(__name__, "perception_source.log")

#: Two spellings of one word a detector and a phrase may disagree on.
_SPELLED: dict[str, str] = {"gray": "grey", "colour": "color"}

#: What stands more than this over the support, along its normal, millimetres, stands on it (a part, a bin's wall): it is
#: painted out of the detector's copy of a frame whole or not at all (:func:`kept_out_pixels`). The support's own band,
#: as the pick loop reads one where the live world says none.
RAISED_MM: float = 5.0
#: A pixel whose depth differs from a neighbour's by more than this, millimetres, lies on an edge: a flying pixel there
#: is placed anywhere between the two surfaces, so it never decides that what it belongs to reaches out of a region.
EDGE_MM: float = 10.0
#: How many pixels of something standing on the support, placed outside every region by their own depth, make it a part
#: that reaches out of them, which keeps every pixel: fewer are its edge's noise.
LEAST_OUTSIDE_PX: int = 25
#: How far round the pixels of a part that reaches out of a region nothing is painted either, pixels: its edge, where the
#: depth gives no sure place.
_KEPT_ROUND_PX: int = 3
#: How wide a ring round the painted pixels gives their colour, pixels: what the parts stand on, round the region.
_RING_PX: int = 6


def _singular(word: str) -> str:
    """A word's singular, by the plain English rules: "boxes" box, "cubes" cube, "glass" stays."""
    if len(word) > 3 and re.search(r"(s|x|z|ch|sh)es$", word):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _words(text: str) -> list[str]:
    """``text``'s words as a label is compared by them: lower case, grey for gray, each in the singular."""
    return [_singular(_SPELLED.get(word, word)) for word in re.findall(r"[a-z0-9]+", str(text).lower())]


class RealSenseVisionPerceptionSource:
    """Grab one RGB-D frame, ground + segment the prompted object(s), emit a :class:`PerceptionFrame`.

    Parameters
    ----------
    streamer
        Anything with ``grab() -> RGBDFrame`` (``.color`` BGR uint8, ``.depth`` uint16 millimetres)
        and ``get_intrinsics() -> 3x3 K | None``. In production this is
        ``camera.setup.image_taking.rgbd.RealSenseRGBDStreamer``; any object carrying those two
        methods satisfies it.
    backend
        A ready-made perception backend (``models.perception_spec.PerceptionSpec.build()``). When
        given, ``detector`` and ``segmenter`` are ignored; when omitted, both of them are required
        or ``__init__`` raises ``ValueError``. Both real construction sites in
        ``robot/execution/autonomous_grasp/cells.py`` pass ``backend``.
    detector, segmenter
        The alternative to ``backend``, composed here into a ``TwoStageBackend``:
        ``detector.detect_all(bgr, prompt) -> [Detection]`` and
        ``segmenter.segment_detection(bgr, det) -> SegmentationResult`` (``.mask`` HxW, ``.label`` str).
    prompt
        The GroundingDINO phrase(s). A multi-phrase prompt grounds every object (the neighbour clutter
        the dense sampler needs), not only the target. :meth:`set_prompt` changes it, together with
        ``object_labels``, between frames.
    object_labels
        Optional canonical labels (the known object names). When given, a detection's free-form
        label is mapped onto the one whose every word it holds (:meth:`_canonical_label`), so an
        exact ``seg.label == target`` match works downstream; a label of another colour is never
        mapped. When omitted, the detector's labels pass through unchanged.
    intrinsics
        Optional 3x3 K override. The D1 seam: leave ``None`` to use the streamer's factory K (the
        D435 ships calibrated), or pass a bench-calibrated K to override it. The source records
        which was used on the frame's provenance via ``intrinsics_source``.
    warmup_grabs
        Throwaway grabs before the real one, so a physical camera's auto-exposure / auto-white-balance
        has settled. A real RealSense needs a handful; a fake ignores it.
    mask_completion
        Which mask-completion policy to apply; the default is ``DEFAULT_MASK_COMPLETION``, which is
        ``NONE``. :mod:`mask_completion` carries the measurement behind that default.
    colour_check
        What a part mapped onto an object label whose words name a colour goes by when its pixels say
        another (:mod:`colour_check`, ``robot.grasping.colour_check``): ``on``, the default, gives it
        a label no object carries, so it stays a neighbour and is never a target; ``log`` judges and
        logs and changes nothing; ``off`` judges nothing.
    colour_check_clipped
        Whether the pixels a channel of which the camera clipped count (``robot.grasping.colour_check_clipped``):
        ``exclude``, the default (the owner, 2026-10-09), leaves them out of the colour's counts, so an orange part
        whose red clips is not read as yellow; ``keep`` counts them, as before.
    """

    def __init__(
        self,
        *,
        streamer: Any,
        detector: Any = None,
        segmenter: Any = None,
        backend: Any = None,
        prompt: str,
        object_labels: tuple[str, ...] = (),
        intrinsics: np.ndarray | None = None,
        warmup_grabs: int = PICK_FRAME_WARMUP_GRABS,
        mask_completion: MaskCompletion = DEFAULT_MASK_COMPLETION,
        colour_check: ColourCheck | str = ColourCheck.ON,
        colour_check_clipped: ColourClipped | str = ColourClipped.EXCLUDE,
    ) -> None:
        # Either a ready-made perception backend, or the detector plus segmenter pair, which is
        # composed into the same backend here rather than run as a two-stage chain by this source.
        # The pair stays a supported path because callers that hand-assemble models need it.
        if backend is None:
            if detector is None or segmenter is None:
                raise ValueError(
                    "RealSenseVisionPerceptionSource needs either backend=..., or both detector=... "
                    "and segmenter=..."
                )
            from src.models.perception_backend import TwoStageBackend

            backend = TwoStageBackend(detector=detector, segmenter=segmenter)
        self._streamer = streamer
        self._backend = backend
        self._prompt = prompt
        self._object_labels = tuple(object_labels)
        self._intrinsics_override = None if intrinsics is None else np.asarray(intrinsics, dtype=np.float64)
        self._warmup_grabs = max(0, int(warmup_grabs))
        #: What to do with a mask that underfills its detection box. `mask_completion.py` carries
        #: the measurement behind the default and the reason this is a lever.
        self._mask_completion = MaskCompletion(mask_completion)
        #: What a part whose pixels contradict its object label's colour goes by (:mod:`colour_check`).
        self._colour_check = ColourCheck(colour_check)
        #: Whether a channel the camera clipped leaves its pixel out of the colour's counts.
        self._colour_check_clipped = ColourClipped(colour_check_clipped)
        #: "override" if a calibrated K was supplied, else "factory". D1 provenance.
        self.intrinsics_source = "override" if intrinsics is not None else "factory"
        #: The stamp of a camera on the wrist (the arm's TCP reader, the rig's shutter tolerance, and
        #: how many more grabs a frame taken while the tool moved gets), `None` for a fixed camera.
        #: Bound late, through `stamp_tool_pose_with`, because the arm does not exist yet when this
        #: source is built.
        self._stamp: ShutterStamp | None = None
        #: How the last frame's parts were found where its acquire was handed a follow (:attr:`last_route`), `None`
        #: where it was not.
        self._last_route: tuple[str, str] | None = None
        #: How many pixels the last frame's detector copy was painted over (:attr:`last_hidden_px`), and why this source
        #: painted nothing though asked, each said once.
        self._hidden_px = 0
        self._hide_said: set[str] = set()

    # ------------------------------------------------------------------ the prompt
    @property
    def prompt(self) -> str:
        """The phrase this source grounds on every frame."""
        return self._prompt

    @property
    def object_labels(self) -> tuple[str, ...]:
        """The labels a detector's words are mapped onto. Empty passes the words through."""
        return self._object_labels

    @property
    def colour_check(self) -> ColourCheck:
        """What a part whose pixels contradict its object label's colour goes by: ``on``, ``log`` or ``off``."""
        return self._colour_check

    def set_prompt(self, prompt: str, *, object_labels: tuple[str, ...] = ()) -> None:
        """Ground ``prompt`` from the next frame on, and map the detector's words onto
        ``object_labels``.

        Nothing reopens and nothing reloads: the backend takes the phrase on every call, so a cell
        built for one object looks for another after this line. An empty phrase is refused here,
        before a frame is taken, because the detector refuses it on every frame.
        """
        phrase = str(prompt)
        if not phrase.strip():
            raise ValueError("a grounding prompt names what to find; the detector refuses an empty one")
        self._prompt = phrase
        self._object_labels = tuple(str(label) for label in object_labels)

    # ------------------------------------------------------------------ label canonicalisation
    def _canonical_label(self, gdino_label: str) -> str:
        """Map a detector's free-form label onto the object label whose every word it holds (identity if none given).

        Words are compared as :func:`_words` reads them: grey and gray are one word, and so are a word and its plural.
        Of the object labels a detector's label holds every word of, the most specific wins (the one of the most
        words); two equally specific ones map nothing, since the part is no more one than the other. A label that
        holds no object label whole maps onto the one object label where it is that label shortened ("cube" for
        "grey cube"), and onto none where there are several. Anything else passes through: "red cube" never becomes
        "grey cube" (it did when one shared word was enough, 2026-10-08), so a part of another colour stays an
        obstacle and is never a target.
        """
        if not self._object_labels:
            return gdino_label
        got = set(_words(gdino_label))
        fits = [label for label in self._object_labels if _words(label) and set(_words(label)) <= got]
        if fits:
            most = max(len(set(_words(label))) for label in fits)
            top = [label for label in fits if len(set(_words(label))) == most]
            return top[0] if len(top) == 1 else gdino_label
        if len(self._object_labels) == 1 and got and got <= set(_words(self._object_labels[0])):
            return self._object_labels[0]
        return gdino_label

    # ------------------------------------------------------------------ the colour a label names
    def _judged(self, bgr: np.ndarray, mask: np.ndarray, det: Any, label: str, said: str) -> str:
        """``label``, or a label no object carries where the part's pixels contradict the colour of the object label it
        was mapped onto (:func:`~src.robot.perception.colour_check.refused_label`).

        Only a part mapped onto an object label is judged, and only where that label's head names one colour
        (:func:`~src.robot.perception.colour_check.colour_named`), on its mask as segmented. An unsure verdict asks the
        backend for one colour word on a crop of the detection's box; no answer refuses the part. What it came to is
        logged with what the detector called the part, whatever the switch: ``log`` leaves the label as it is.
        """
        if self._colour_check is ColourCheck.OFF or label not in self._object_labels:
            return label
        wanted = colour_named(label)
        if wanted is None:
            return label
        verdict = judge_colour(bgr, mask, wanted, clipped=self._colour_check_clipped)
        if verdict.unsure:
            verdict = verdict.answered(ask_colour(self._backend, bgr, getattr(det, "box", None)))
        logged_only = not verdict.agrees and self._colour_check is ColourCheck.LOG
        _LOG.info("%s; the detector said %r%s", verdict.said(label), said,
                  "; logged only (colour_check: log), it stays a target" if logged_only else "")
        if verdict.agrees or logged_only:
            return label
        return refused_label(verdict, label)

    # ------------------------------------------------------------------ mask robustness
    def _maybe_fill_to_detection_box(self, mask: np.ndarray, det: Any) -> np.ndarray:
        """Apply the configured mask-completion policy.

        The rule lives in :mod:`mask_completion` so a real cell and the sim source cannot drift
        apart on a transform that decides the grasp's closing axis. The default is ``NONE``: over
        270 reference scenes the axis-aligned rule costs 7.5 percentage points of top-1 and loses
        five grasps for every one it wins. That module's docstring carries the numbers and the
        bound on the rescue it gives up.
        """
        return complete_mask(mask, det, policy=self._mask_completion)

    # ------------------------------------------------------------------ the frame
    #: These pixels come off a physical device, so the console may say "camera". See
    #: `perception/viewfinder.py` for why this is declared rather than inferred.
    colour_source_kind = "camera"
    #: This source's :meth:`acquire` takes a ``follow``: the parts a task kept, found again with no detector where every
    #: check passes (``src/robot/perception/kept_scene.py``). Declared, so a pick loop hands it only to a source that
    #: says so, and grounds every frame of any other.
    follows_parts = True
    #: This source's :meth:`acquire` takes a ``hide``: the regions a task keeps out, painted over the copy of the frame
    #: its detector reads (:func:`kept_out_pixels`). Declared, so a pick loop hands it only to a source that says so.
    hides_places = True

    @property
    def last_route(self) -> tuple[str, str] | None:
        """How the last frame's parts were found, where its :meth:`acquire` was handed a follow: ``("followed", what)``
        with no detector asked, or ``("grounded", "not followed: why")``; ``None`` for a frame grounded as every frame
        was, so a pick that follows nothing emits the events it always emitted."""
        return self._last_route

    @property
    def last_hidden_px(self) -> int:
        """How many pixels of the last frame were painted over in the copy its detector read (:meth:`acquire`'s
        ``hide``); 0 where none were, or the detector read the real frame."""
        return self._hidden_px

    def peek_color(self) -> np.ndarray | None:
        """One colour frame, with no detector, no segmenter and no depth work. BGR uint8.

        The :mod:`~src.robot.perception.viewfinder` capability, implemented here because this
        source owns an open streamer. It takes one frameset and no warm-up prefix: a viewer that
        discarded five frames per tick would pull the device at six times the rate it displays.

        Beside :meth:`acquire` it takes its turn: a cell hands this source a camera owner's handle,
        whose grabs run under the rig's lock. A raw streamer handed in directly has no such lock, and
        then the caller keeps the two apart; ``api/viewfinder.py`` stands down during a run either
        way.
        """
        grabbed = self._streamer.grab()
        colour = getattr(grabbed, "color", None)
        if colour is None:
            return None
        return np.ascontiguousarray(np.asarray(colour))

    @property
    def streamer(self) -> Any:
        """The camera this source reads, for a consumer that needs depth without the models.

        The planner world is that consumer. It refreshes before every motion, and a full
        perceive runs a detector and a segmenter for hundreds of milliseconds to answer a
        question it never asks: it wants to know where geometry is, not what it is called.

        Handed out rather than reached for. The composition root owns the rig and this
        source owns nothing it did not receive, so this is a view of the same handle rather
        than a second one.
        """
        return self._streamer

    @property
    def backend(self) -> Any:
        """The perception backend this source grounds its frames with: the one it was handed, or the detector and
        segmenter it composed. Read-only.

        Handed out, as :attr:`streamer` is, for a consumer that locates on this source's camera with this source's
        models: a task finds the bin it places into through ``Locator.from_parts(backend=source.backend)``
        (``robot.execution.place_target.locators_for_service``), so no second copy of the detector loads. The same
        object, never a copy, and nothing in it changes for being read: the backend is handed the phrase on every call.
        """
        return self._backend

    def close(self) -> None:
        """Release the camera. Idempotent, and it never raises.

        The counterpart to the ``provider.open_rig(...)`` that ``build_real_components`` performs;
        it gives the rig back through the handle's ``release()``, so the rig lifecycle stays with
        the ``FrameProvider`` and not with the streamer. A long-running process builds a cell more
        than once: a CLI runner opens a device and lets process exit close it, while a console
        builds, gets refused, has its config fixed and builds again. On real hardware a second
        ``pipeline.start()`` on a device the first build is still streaming fails, so without this
        the console cannot rebuild until it is restarted.

        Never raises, because this is teardown: a camera that cannot be closed must not stop the
        thing that was closing it.
        """
        release = getattr(self._streamer, "release", None)
        if callable(release):
            try:
                release()
            except Exception:  # noqa: BLE001 (teardown reports nothing it can fix)
                pass

    def stamp_tool_pose_with(
        self, reader: Callable[[], Pose], *, motion_tolerance: tuple[float, float], attempts: int,
    ) -> None:
        """Stamp every frame with the TCP at its shutter, and grab again while the tool moved.

        For a camera on the wrist, whose grasp is placed by where the tool was when the shutter
        opened. With an unstamped frame the eye-in-hand resolver reads the arm when the grasp is
        resolved, and every millimetre the tool travelled in between goes into the grasp.
        ``reader`` is the arm's own ``get_tcp_pose``, read before and after the real grab.
        ``motion_tolerance`` is the rig's ``(shutter_motion_tolerance_mm,
        shutter_motion_tolerance_deg)``, judged by the rule the live world judges it by
        (``ShutterMotion``). A frame taken while the tool moved beyond it is grabbed again, without
        the warm-ups, up to ``attempts`` more times, and then the acquire raises
        ``PerceptionFrameMoved``. The stamp is :class:`~src.robot.core.shutter_motion.ShutterStamp`,
        the one a hand finder over a wrist camera takes its frames through too.
        """
        self._stamp = ShutterStamp.of(reader, motion_tolerance=motion_tolerance, attempts=attempts)

    def _grab(self) -> "tuple[Any, Pose | None, float]":
        """The real grab: the frame, the tool pose at its shutter, and its capture time.

        The capture time is the one the camera owner read before its grab when the frame carries
        it, else a clock read here just before the grab, never after: age decides whether a frame
        may still be planned against.
        """
        stamp = self._stamp
        if stamp is None:
            before_grab = time.time()
            rgbd = self._streamer.grab()
            return rgbd, None, _captured_at(rgbd, before_grab)
        grabbed = stamp.grab(self._streamer.grab)
        if grabbed.tool_pose is None:
            raise PerceptionFrameMoved(
                camera=str(getattr(self._streamer, "rig_id", "") or "camera"), attempts=grabbed.grabs,
                moved_mm=grabbed.motion.moved_mm, turned_deg=grabbed.motion.turned_deg,
                tolerance_mm=stamp.tolerance_mm, tolerance_deg=stamp.tolerance_deg,
            )
        return grabbed.frame, grabbed.tool_pose, _captured_at(grabbed.frame, grabbed.before_grab_s)

    def acquire(self, follow: "Follow | None" = None, hide: "Hide | None" = None) -> PerceptionFrame:
        """One frame, its parts grounded by the backend; or, handed ``follow``, found again with no detector where the
        parts a task kept pass every check on this frame (:meth:`_followed`), and grounded as ever where any does not.

        One grab either way: following reads the frame the detector would have read. :attr:`last_route` says which.
        Handed ``hide``, a frame the detector grounds is read by it on a copy with the regions the task keeps out painted
        over (:meth:`_detector_copy`); the segmenter, the colour check, the depth and the frame this returns are the real
        ones, and :attr:`last_hidden_px` says how many pixels the copy painted.
        """
        self._last_route = None
        self._hidden_px = 0
        for _ in range(self._warmup_grabs):
            self._streamer.grab()  # discard: let auto-exposure / white-balance settle on real hardware

        # After the warm-ups, the real grab, with the tool pose read around it on a wrist camera.
        rgbd, tool_pose, captured_at_s = self._grab()
        bgr = np.ascontiguousarray(np.asarray(rgbd.color))          # detector/segmenter take OpenCV BGR
        depth_mm = np.asarray(rgbd.depth, dtype=np.float64)         # uint16 mm as float; 0 == hole
        # A copy, not the same array, and that is the point of keeping both fields. Their
        # contents are identical here; what separates them is what happens after, because a
        # noise harness replaces `depth_map` to stand in for a real sensor and leaves this one
        # measured. Aliasing would quietly hand the planner the noise as well.
        rendered_depth_mm = depth_mm.copy()

        if self._intrinsics_override is not None:
            intrinsics = self._intrinsics_override.copy()
        else:
            k = self._streamer.get_intrinsics()
            if k is None:
                raise RuntimeError(
                    "streamer.get_intrinsics() returned None: open() the streamer before acquire(), or "
                    "pass a calibrated `intrinsics=` (the D1 override)."
                )
            intrinsics = np.asarray(k, dtype=np.float64)

        segmentations = None
        if follow is not None:
            segmentations = self._followed(follow, bgr=bgr, depth_mm=rendered_depth_mm, intrinsics=intrinsics,
                                           tool_pose=tool_pose)
        if segmentations is None:
            # The detector/segmenter chain and its two failure rules live in the backend: a detector
            # error yields no objects (an honest no_valid_grasp downstream), and a single segmentation
            # error skips that object and keeps the rest. The detector may read a copy with the task's own
            # places painted over (the owner, 2026-10-08 night); the segmenter always reads the real frame.
            copy = self._detector_copy(hide, bgr=bgr, depth_mm=rendered_depth_mm, intrinsics=intrinsics,
                                       tool_pose=tool_pose) if hide is not None else None
            perceived = (self._backend.perceive(bgr, self._prompt) if copy is None
                         else self._backend.perceive(bgr, self._prompt, detect_on=copy))
            segmentations = self._segmented(bgr, perceived)

        rgb = bgr[..., ::-1]  # BGR -> RGB for any debugging consumer (no reader in robot/ today)
        # The depth this publishes is the depth the sensor measured, under a mask as everywhere
        # else. It used to replace every pixel under a mask with one number, the nearest surface
        # plus a few millimetres, because that is the plane a jaw is driven to. Six stages
        # downstream read the same array for the shape of what is there and got a flat sheet:
        # the antipodal search found no opposing normals, the dense sampler measured no
        # curvature, the support-plane refinement saw an extent of exactly zero, and the
        # multi-camera fusion fused sheets. Where a grasp is anchored inside an object is one
        # number about one candidate and it is decided by `grasping.geometry`, not by
        # overwriting the picture everything else reasons from.
        #
        # `surface_depth_map` stays, and now carries the same measurement without the sensor
        # noise a harness may add to `depth_map`: the planner's obstacle world is built from it,
        # and an obstacle that flickers frame to frame is worse than one that is slightly wrong.
        # The shutter time, on the `time.time` clock rather than a monotonic one, because a
        # consumer compares it against its own wall clock to decide whether the world is too old
        # to plan against.
        #
        # What the camera recorded for research with the frame goes along to the views, where its rig
        # records for research (`realsense.record_for_research`, the owner, 2026-10-09); `None` from every
        # other camera, and the frame is what it was.
        return PerceptionFrame(
            depth_map=depth_mm, intrinsics=intrinsics, segmentations=tuple(segmentations), rgb=rgb,
            timestamp=captured_at_s, surface_depth_map=rendered_depth_mm, tool_pose=tool_pose,
            research=_research_of(rgbd),
        )

    def _segmented(self, bgr: np.ndarray, perceived: Any) -> list[Any]:
        """The segmentations of ``perceived`` (``(detection, segmentation)`` pairs) as a frame carries them: each label
        mapped onto the object labels, its colour judged, its mask completed as the policy says."""
        segmentations: list[Any] = []
        for obj in perceived:
            det, seg = obj.detection, obj.segmentation
            said = getattr(seg, "label", "") or ""
            raw = np.asarray(seg.mask).astype(bool)
            # The colour the label names, judged on the mask as segmented, before any fusion reads it: a part of
            # another colour keeps its mask, its depth and its place in the planner world under a label no object
            # carries, so it stays a neighbour for every check and is never a target (the owner, 2026-10-08).
            label = self._judged(bgr, raw, det, self._canonical_label(said), said)
            mask = self._maybe_fill_to_detection_box(raw, det)
            seg = replace(seg, label=label, mask=mask.astype(np.uint8))
            segmentations.append(seg)
        return segmentations

    def _followed(
        self, follow: "Follow", *, bgr: np.ndarray, depth_mm: np.ndarray, intrinsics: np.ndarray,
        tool_pose: "Pose | None",
    ) -> "list[Any] | None":
        """This frame's segmentations as ``follow`` finds the parts a task kept again, or ``None`` where it does not and
        the frame is grounded by the detector, as every frame was (the owner, 2026-10-09: all or nothing per frame).

        ``follow`` reads the frame first (the scene check, each part where it stood) and answers the boxes to cut; the
        backend's segmenter cuts them (``segment_boxes``), no detector asked; each mask then goes through what every
        mask goes through here, the label mapping, the colour check and the mask completion, under its kept label; and
        ``follow`` accepts every mask, or none. Any doubt grounds the frame: a follow that cannot be judged or raises, a
        backend that cuts no boxes, a box not cut, a part the colour check refuses, a mask not accepted.
        """
        started = time.perf_counter()
        try:
            following = follow(depth_mm, bgr, intrinsics, tool_pose)
        except Exception as exc:  # noqa: BLE001 (a follow that cannot be judged grounds the frame, as every frame was)
            _LOG.exception("the parts kept could not be followed on this frame")
            return self._not_followed(f"the follow raised {type(exc).__name__}: {exc}")
        why = str(getattr(following, "why", "") or "") if following is not None else "nothing was kept to follow"
        if why:
            return self._not_followed(why)
        cut = getattr(self._backend, "segment_boxes", None)
        if not callable(cut):
            return self._not_followed("the perception backend cuts no boxes it is handed")
        boxes = tuple(following.boxes)
        try:
            segmented = cut(bgr, boxes)
        except Exception as exc:  # noqa: BLE001 (boxes that cannot be cut ground the frame)
            _LOG.exception("the boxes of the parts kept could not be cut")
            return self._not_followed(f"cutting the boxes raised {type(exc).__name__}: {exc}")
        objects = tuple(getattr(segmented, "objects", ()) or ())
        failed = tuple(getattr(segmented, "failed", ()) or ())
        if failed or len(objects) != len(boxes) or any(obj is None for obj in objects):
            return self._not_followed("the segmenter did not cut every box: " + ("; ".join(failed) or "no answer"))
        segmentations = self._segmented(bgr, objects)
        refused = [f"part {index} ({label!r}) became {seg.label!r}"
                   for index, (seg, (_, label)) in enumerate(zip(segmentations, boxes)) if seg.label != label]
        if refused:
            return self._not_followed("the colour check or the labels refused it: " + "; ".join(refused))
        accepted = following.accept(bgr, [np.asarray(seg.mask).astype(bool) for seg in segmentations])
        if accepted.why:
            return self._not_followed(accepted.why)
        followed = [replace(seg, mask=mask.astype(np.uint8)) for seg, mask in zip(segmentations, accepted.masks)]
        said = str(following.said or f"{len(followed)} part(s) followed")
        self._last_route = ("followed", said)
        _LOG.info("%s in %.0f ms: the detector was not asked", said, (time.perf_counter() - started) * 1000.0)
        return followed

    def _not_followed(self, why: str) -> "list[Any] | None":
        """A frame the parts kept were not followed on: said, and grounded by the detector, as every frame was; no
        segmentations, so :meth:`acquire` grounds it."""
        self._last_route = ("grounded", f"not followed: {why}")
        _LOG.info("the parts kept are not followed on this frame (%s): the detector grounds it", why)
        return None

    def _detector_copy(
        self, hide: "Hide", *, bgr: np.ndarray, depth_mm: np.ndarray, intrinsics: np.ndarray, tool_pose: "Pose | None",
    ) -> "np.ndarray | None":
        """A copy of ``bgr`` for the detector alone, the pixels ``hide`` answers painted over (:func:`painted_copy`);
        ``None`` where the detector reads the real frame: nothing to paint, a ``hide`` that raises or answers no mask of
        the frame's size, or a backend that does not say it hands its detector a copy (``detects_on_a_copy``), each
        said. ``hide`` reads the measured depth through a view it cannot write, so the frame keeps what was measured."""
        if getattr(self._backend, "detects_on_a_copy", False) is not True:
            why = (f"the perception backend ({type(self._backend).__name__}) hands its detector no copy of a frame, so "
                   "the regions the task keeps out are not hidden from it")
            if why not in self._hide_said:
                _LOG.warning("%s", why)
                self._hide_said.add(why)
            return None
        depth = depth_mm.view()
        depth.flags.writeable = False
        try:
            answer = hide(depth, intrinsics.copy(), tool_pose)
        except Exception:  # noqa: BLE001 (regions that cannot be placed hide nothing: the detector reads the frame)
            _LOG.exception("the regions the task keeps out could not be placed on this frame; the detector reads it whole")
            return None
        if answer is None:
            return None
        mask = np.asarray(answer)
        if mask.shape != bgr.shape[:2]:
            _LOG.warning("the regions the task keeps out came as a mask of %s pixels for a frame of %s; the detector "
                         "reads the frame whole", mask.shape, bgr.shape[:2])
            return None
        hidden = mask.astype(bool)
        count = int(np.count_nonzero(hidden))
        if count == 0:
            _LOG.info("nothing of the regions the task keeps out is painted on this frame (none of them in view, or "
                      "what stands in them reaches out of them): the detector reads it whole")
            return None
        self._hidden_px = count
        _LOG.info("%d pixel(s) of the regions the task keeps out (%.1f %% of the frame) are painted over in the "
                  "detector's copy; the segmenter reads the real frame", count, 100.0 * count / hidden.size)
        return painted_copy(bgr, hidden)


def _captured_at(frame: Any, before_grab: float) -> float:
    """The owner's capture time when the frame carries one, else the clock read just before the grab."""
    captured = getattr(frame, "captured_at_s", None)
    return float(captured) if isinstance(captured, (int, float)) else float(before_grab)


def _research_of(frame: Any) -> Any:
    """What the camera recorded for research with ``frame`` (``ResearchCapture``), else ``None``: a double that answers
    every attribute hands on nothing."""
    from src.camera.setup.image_taking.frames import ResearchCapture  # noqa: PLC0415 - the camera package, on demand

    research = getattr(frame, "research", None)
    return research if isinstance(research, ResearchCapture) else None


def _inside_region(region: Any, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Which BASE points (``x``, ``y``, arrays of one shape) stand in ``region`` (``ExclusionRegion``: a circle, or a
    turned rectangle), as ``ExclusionRegion.contains`` reads one point; a region of neither shape holds none."""
    centre = np.asarray(getattr(region, "centre_xy_mm", (np.nan, np.nan)), dtype=np.float64).reshape(2)
    dx, dy = x - centre[0], y - centre[1]
    radius = getattr(region, "radius_mm", None)
    if radius is not None:
        return np.asarray(dx * dx + dy * dy <= float(radius) ** 2)
    size = getattr(region, "size_mm", None)
    if size is None:
        return np.zeros(np.shape(x), dtype=bool)
    yaw = float(getattr(region, "yaw_rad", 0.0))
    cos, sin = float(np.cos(yaw)), float(np.sin(yaw))
    along, across = cos * dx + sin * dy, -sin * dx + cos * dy
    return np.asarray((np.abs(along) <= float(size[0]) / 2.0) & (np.abs(across) <= float(size[1]) / 2.0))


def kept_out_pixels(
    depth_mm: np.ndarray, intrinsics: np.ndarray, camera_to_base: np.ndarray, regions: "Sequence[Any]", *,
    support_mm: float, normal: "Sequence[float]" = (0.0, 0.0, 1.0),
) -> np.ndarray:
    """The pixels of a frame that show only what stands in a region a task keeps out (its bin's footprint, the circle
    about a drop: ``ExclusionRegion``), as a boolean mask the frame's size: what the detector's copy of it is painted
    over with (the owner, 2026-10-08 night: the task's own bins hidden from the detector).

    Every pixel is placed in BASE by its own depth, through ``intrinsics`` and ``camera_to_base``; one whose XY stands in
    a region is kept out. What stands more than :data:`RAISED_MM` over the support (``support_mm`` along ``normal``)
    is kept out whole or not at all: a connected patch of it with :data:`LEAST_OUTSIDE_PX` or more pixels placed
    outside every region, by depths that agree with their neighbours' (:data:`EDGE_MM`), is a part that reaches out of
    them, as a part beside a bin or reaching into a drop's circle does, and keeps every pixel, and those
    :data:`_KEPT_ROUND_PX` round it. A hole in the depth is kept out only where kept-out pixels close round it on every
    side. Nothing else is: a part outside every region keeps all of its pixels.
    """
    import cv2  # noqa: PLC0415 - deferred: OpenCV is no small import, and only a cell that hides its places needs it here

    depth = np.asarray(depth_mm, dtype=np.float64)
    rows, cols = depth.shape
    measured = np.isfinite(depth) & (depth > 0.0)
    kept = np.zeros((rows, cols), dtype=bool)
    if not regions or not measured.any():
        return kept
    lens = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
    placed = np.asarray(camera_to_base, dtype=np.float64).reshape(4, 4)
    # Every pixel placed in BASE on the image grid itself, in single precision (a micrometre at a metre): a whole
    # 1280 x 720 frame in tens of milliseconds. An unmeasured pixel is placed at the camera and counts for nothing.
    z = np.where(measured, depth, 0.0).astype(np.float32)
    u = np.arange(cols, dtype=np.float32)[None, :]
    v = np.arange(rows, dtype=np.float32)[:, None]
    xc = (u - np.float32(lens[0, 2])) * (z / np.float32(lens[0, 0]))
    yc = (v - np.float32(lens[1, 2])) * (z / np.float32(lens[1, 1]))
    rot, shift = placed[:3, :3].astype(np.float32), placed[:3, 3].astype(np.float32)
    base = [rot[i, 0] * xc + rot[i, 1] * yc + rot[i, 2] * z + shift[i] for i in range(3)]
    inside = np.zeros((rows, cols), dtype=bool)
    for region in regions:
        inside |= _inside_region(region, base[0], base[1])
    inside &= measured
    if not inside.any():
        return kept
    unit = np.asarray(normal, dtype=np.float64).reshape(3)
    unit = unit / max(float(np.linalg.norm(unit)), 1e-12)
    height = float(unit[0]) * base[0] + float(unit[1]) * base[1] + float(unit[2]) * base[2] - float(support_mm)
    raised = measured & (height > RAISED_MM)
    # Depths that agree with every measured neighbour's: no flying pixel at an edge among them.
    solid = measured.copy()
    for shifted, valid in _neighbours(z, measured):
        solid &= ~valid | (np.abs(z - shifted) <= EDGE_MM)
    count, labels = cv2.connectedComponents((raised & solid).astype(np.uint8), connectivity=8)
    reaching = np.zeros((rows, cols), dtype=bool)
    if count > 1:
        outside = np.bincount(labels[raised & solid & ~inside], minlength=count)
        reaches = outside >= LEAST_OUTSIDE_PX
        reaches[0] = False
        reaching = reaches[labels]
        if reaching.any():
            reaching = cv2.dilate(reaching.astype(np.uint8), np.ones((3, 3), np.uint8),
                                  iterations=_KEPT_ROUND_PX).astype(bool)
    kept = inside & ~reaching
    # A hole in the depth closed round on every side by kept-out pixels is kept out too; one that touches anything else
    # (a pixel not kept out, the frame's edge) is not.
    holes = ~measured
    if kept.any() and holes.any():
        count, labels = cv2.connectedComponents(holes.astype(np.uint8), connectivity=8)
        touching = cv2.dilate((measured & ~kept).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & holes
        open_holes = np.zeros(count, dtype=bool)
        open_holes[np.unique(labels[touching])] = True
        for border in (labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]):
            open_holes[np.unique(border)] = True
        open_holes[0] = True
        kept |= holes & ~open_holes[labels]
    return kept


def _neighbours(depth: np.ndarray, measured: np.ndarray) -> "list[tuple[np.ndarray, np.ndarray]]":
    """Each pixel's four neighbours' depths and whether each was measured, edge pixels facing nothing unmeasured."""
    found: list[tuple[np.ndarray, np.ndarray]] = []
    for axis, step in ((0, 1), (0, -1), (1, 1), (1, -1)):
        shifted = np.roll(depth, step, axis=axis)
        valid = np.roll(measured, step, axis=axis)
        edge = [slice(None), slice(None)]
        edge[axis] = slice(0, 1) if step == 1 else slice(-1, None)
        valid[tuple(edge)] = False
        found.append((shifted, valid))
    return found


def painted_copy(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """A copy of ``image_bgr`` with the pixels of ``mask`` painted in one colour: the median of the pixels in a ring
    :data:`_RING_PX` wide round them, what the parts stand on there, else of every pixel left, else mid grey. The image
    handed in is never changed."""
    import cv2  # noqa: PLC0415 - deferred, as in kept_out_pixels

    painted = np.array(image_bgr, copy=True)
    hide = np.asarray(mask, dtype=bool)
    if not hide.any():
        return painted
    ring = cv2.dilate(hide.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=_RING_PX).astype(bool) & ~hide
    around = painted[ring] if ring.any() else painted[~hide]
    colour = np.median(around.reshape(-1, painted.shape[-1]), axis=0) if around.size else np.full(painted.shape[-1], 128.0)
    painted[hide] = np.round(colour).astype(painted.dtype)
    return painted
