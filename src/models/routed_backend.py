"""Two perception backends behind one, chosen per prompt by a router.

    backend = RoutedPerceptionBackend(router=..., simple_factory=..., vlm_factory=...)
    objects = backend.perceive(image_bgr, "der kaputte Würfel")   # -> the VLM route
    backend.last_decision.describe()                              # 'vlm (non_english)'

Both routes are lazy. Building both eagerly would load GroundingDINO and Qwen at cell build, around
nine gigabytes of VRAM and ten seconds, on a cell that may only ever send simple prompts, or only
complex ones. Each route is built the first time it is chosen.

The simple route gets a normalised prompt; the VLM route does not. GroundingDINO's caption
convention, lowercase with one trailing period, is what its text encoder was trained on, while the
VLM is asked to reason about the operator's phrasing, so rewriting that would change the question.

The decision is recorded, not just acted on. ``last_decision`` and the ``on_decision`` callback are
what the console, the PERCEIVED event and ``GraspAttemptRecord.extra`` read, so an operator seeing a
slow pick can find out that their wording chose the expensive route, and which word did it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from src.models.constants import MODELS_LOG_DIR, PERCEPTION_ROUTING_LOG_FILE
from src.models.perception_backend import PerceivedObject, SegmentedBoxes, failures_of, last_failure_of
from src.models.routing import (
    PromptRouter,
    Route,
    RouteDecision,
    RuleBasedRouter,
    normalize_simple_prompt,
)
from src.utility.log_cfg import create_logger

__all__ = ["RoutedPerceptionBackend"]

#: One file for the routing verdicts, so a decision outlives ``last_decision``, which holds only the
#: most recent one. An injected ``logger=`` wins over this.
_LOG = create_logger(__name__, log_file=PERCEPTION_ROUTING_LOG_FILE, log_dir=MODELS_LOG_DIR)


class RoutedPerceptionBackend:
    """Pick a perception route per prompt and delegate to it.

    Satisfies :class:`~src.models.perception_backend.PerceptionBackend`, so any caller that already
    accepts a backend accepts this one unchanged.
    """

    def __init__(
        self,
        *,
        simple_factory: Callable[[], Any],
        vlm_factory: Callable[[], Any],
        router: PromptRouter | None = None,
        normalize: bool = True,
        on_decision: Callable[[RouteDecision], None] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._router: PromptRouter = router or RuleBasedRouter()
        self._factories: dict[Route, Callable[[], Any]] = {
            Route.SIMPLE: simple_factory,
            Route.VLM: vlm_factory,
        }
        self._built: dict[Route, Any] = {}
        self._normalize = normalize
        self._on_decision = on_decision
        self._log = logger or _LOG
        self._last_decision: RouteDecision | None = None
        self._last_failure = ""

    @property
    def last_decision(self) -> RouteDecision | None:
        """The most recent routing verdict, for telemetry. ``None`` before the first ``perceive``."""
        return self._last_decision

    @property
    def failures(self) -> int:
        """The built routes' counted failures, summed: a perceive that returned nothing because a model raised.

        A route not built yet has failed nothing. The sum only grows, as each route's count does.
        """
        return sum(failures_of(backend) for backend in tuple(self._built.values()))

    @property
    def last_failure(self) -> str:
        """The latest counted failure, from whichever route raised it, or ``""``."""
        return self._last_failure

    def built_routes(self) -> tuple[Route, ...]:
        """The routes constructed so far, which are the ones holding VRAM right now."""
        return tuple(self._built)

    def _backend_for(self, route: Route) -> Any:
        backend = self._built.get(route)
        if backend is None:
            self._log.info("building the %s perception route (first use)", route)
            backend = self._factories[route]()
            self._built[route] = backend
        return backend

    #: The chosen route's detector may read a copy of the image (``robot.grasping.hide_own_places``) where it takes
    #: one; :meth:`perceive` passes ``detect_on`` on.
    detects_on_a_copy = True

    def perceive(self, image_bgr: Any, prompt: str, *, detect_on: Any = None) -> tuple[PerceivedObject, ...]:
        decision = self._router.route(prompt)
        self._last_decision = decision
        if self._on_decision is not None:
            try:
                self._on_decision(decision)
            except Exception:  # noqa: BLE001 (telemetry must never break a pick)
                self._log.debug("route-decision callback raised; continuing", exc_info=True)

        self._log.info("prompt %r -> %s", prompt, decision.describe())
        text = prompt
        if decision.route is Route.SIMPLE and self._normalize:
            text = normalize_simple_prompt(prompt)

        # Not guarded: if the chosen route cannot run, enforcing that is the route's own contract.
        # ``GuardedVlmBackend`` decides refuse against degrade with context this class lacks, and
        # quietly substituting the other backend here would produce the confident wrong grasp that
        # routing exists to prevent.
        backend = self._backend_for(decision.route)
        before = failures_of(backend)
        try:
            if detect_on is not None and getattr(backend, "detects_on_a_copy", False) is True:
                return tuple(backend.perceive(image_bgr, text, detect_on=detect_on))
            return tuple(backend.perceive(image_bgr, text))
        finally:
            if failures_of(backend) > before:
                self._last_failure = last_failure_of(backend)

    def name_colour(self, image_bgr: Any, box: Any) -> str:
        """One colour word for the object in ``box``, asked of the VLM route: the one that can name a colour, built at
        its first use as a prompt that chooses it builds it. ``""`` where it answers none: a degraded route, or a route
        that names no colour.

        A question that raised is counted by that route (:attr:`failures` sums the built routes), and its sentence kept
        here as a perceive's is.
        """
        backend = self._backend_for(Route.VLM)
        name = getattr(backend, "name_colour", None)
        if not callable(name):
            return ""
        before = failures_of(backend)
        try:
            answer = name(image_bgr, box)
        finally:
            if failures_of(backend) > before:
                self._last_failure = last_failure_of(backend)
        return answer if isinstance(answer, str) else ""

    def segment_boxes(self, image_bgr: Any, boxes: Any) -> SegmentedBoxes:
        """Cut ``boxes`` with the route that grounded last, already built: never a route built for it, and so no model
        loaded. Both routes share one segmenter; a cell that has grounded nothing yet cuts none, and its frame is
        grounded."""
        decision = self._last_decision
        backend = self._built.get(decision.route) if decision is not None else None
        cut = getattr(backend, "segment_boxes", None)
        if not callable(cut):
            return SegmentedBoxes.refused(boxes, "no perception route that cuts boxes has grounded a frame yet")
        return cut(image_bgr, boxes)
