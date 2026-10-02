"""One VLM copy per process: the holder, the load lock, the text answer and the reader's availability rule.

No weights and no GPU. The grounder's heavy load is its ``_load`` seam, replaced here by a fake that counts its
calls, and the processor, model and torch it would return are small doubles that record what they were handed.

Pinned here:

* **One copy.** The perception build's grounder IS the holder's (an identity, not an equality), a rebuild gets the
  same object, and the command reader asks that same object.
* **One copy, whatever the config names.** A config that names other weights than the copy on the card (an edited
  YAML, no rebuild) gets a refusal from a command and from Laden, never a second load. Only a build switches the
  weights, and it unloads the copy it replaces.
* **One load.** Two threads that both need the weights load them once; a failed load is recorded and never reads
  as loaded; a caller that waited for a load that failed, and any command, does not try it again.
* **One inference at a time.** Reading a command and grounding a frame never run ``generate`` at once on the one
  model, and an unload waits for both.
* **The availability rule (owner decision Q8 A).** A command may load the VLM only on a cell whose detector it is;
  on any other cell a person loads it first ("Laden"), and until then the reader answers ``vlm_not_loaded``. The
  console's build of such a cell releases a copy nobody asked for.

Every test starts and ends with ``forget()`` on every holder it touched, the process-wide one included: a holder
that kept a grounder would leak it into `tests/test_perception_pipeline.py`, which asserts a build loads nothing.
"""

from __future__ import annotations

import contextlib
import threading
import time
import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np

from src.config.schema.models.models_schema import PipelineConfig, VlmConfig
from src.models.vlm import (
    CommandRefused,
    Qwen3VLGrounder,
    TextAsker,
    VlmHolder,
    VlmNotLoadedError,
    VlmUnavailableError,
    VlmWeights,
    read_command,
    reader_availability,
    shared_vlm,
    vlm_detects,
)

MODEL = "Qwen/Qwen3-VL-4B-Instruct"
#: Another checkpoint, as an edited models YAML names it during bring-up (4B against 8B).
OTHER = "Qwen/Qwen3-VL-8B-Instruct"
POSES = {"Ablage links": "ablage_links"}
_PROMPT_TOKENS = 7
_NEW_TOKENS = 4


# --- doubles ----------------------------------------------------------------------------------------------------


class _Batch(dict):  # type: ignore[type-arg]
    """What ``apply_chat_template(..., return_dict=True, return_tensors="pt")`` answers: a mapping that moves."""

    def to(self, device: Any) -> "_Batch":
        self.device = device
        return self


class _Processor:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.templates: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []
        self.decoded: list[tuple[np.ndarray, bool]] = []
        #: Raised by the next template call when set: a bug in the tokenising path.
        self.fail: BaseException | None = None

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> _Batch:
        if self.fail is not None:
            raise self.fail
        self.templates.append((messages, kwargs))
        return _Batch(input_ids=np.arange(_PROMPT_TOKENS).reshape(1, -1),
                      attention_mask=np.ones((1, _PROMPT_TOKENS)))

    def batch_decode(self, ids: Any, skip_special_tokens: bool = False) -> list[str]:
        self.decoded.append((np.asarray(ids).copy(), skip_special_tokens))
        return [self.answer]


class _Model:
    device = "cuda:0"

    def __init__(self, *, hold: threading.Event | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.hold = hold
        self.inside = 0
        self.most_inside = 0
        self.entered = threading.Event()
        self._count = threading.Lock()
        #: Raised by the next generate when set: the card ran out of memory mid-answer, say.
        self.fail: BaseException | None = None

    def generate(self, **kwargs: Any) -> np.ndarray:
        with self._count:
            self.inside += 1
            self.most_inside = max(self.most_inside, self.inside)
        self.entered.set()
        try:
            self.calls.append(kwargs)
            if self.hold is not None:
                self.hold.wait(5.0)
            if self.fail is not None:
                raise self.fail
            prompt = kwargs["input_ids"]
            return np.concatenate([prompt, np.full((1, _NEW_TOKENS), 99)], axis=1)
        finally:
            with self._count:
                self.inside -= 1


class _Cuda:
    def __init__(self) -> None:
        self.emptied = 0

    def empty_cache(self) -> None:
        self.emptied += 1


class _Torch:
    def __init__(self) -> None:
        #: Counts ``torch.cuda.empty_cache()``: an unloaded copy hands its memory back to the card.
        self.cuda = _Cuda()

    @staticmethod
    def inference_mode() -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()


class _CountingLock:
    """A lock that counts who asked for it, so a test lets the first caller go only once a second one waits."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._asked = 0
        self._changed = threading.Condition()

    def __enter__(self) -> "_CountingLock":
        with self._changed:
            self._asked += 1
            self._changed.notify_all()
        self._lock.acquire()
        return self

    def __exit__(self, *_: Any) -> None:
        self._lock.release()

    def asked(self, times: int, *, timeout: float = 5.0) -> bool:
        with self._changed:
            return self._changed.wait_for(lambda: self._asked >= times, timeout)


class _Loads:
    """A ``_load`` seam: counts its calls, may wait or fail, and hands back the doubles."""

    def __init__(self, *, answer: str = '"intent": "none"}', fail: BaseException | None = None,
                 wait: threading.Event | None = None, hold: threading.Event | None = None) -> None:
        self.calls = 0
        self.fail = fail
        self.wait = wait
        self.started = threading.Event()
        self.processor = _Processor(answer)
        self.model = _Model(hold=hold)
        self.torch = _Torch()
        self._lock = threading.Lock()

    def __call__(self, *_: Any) -> tuple[Any, Any, Any]:
        with self._lock:
            self.calls += 1
        self.started.set()
        if self.wait is not None:
            self.wait.wait(5.0)
        if self.fail is not None:
            raise self.fail
        return self.processor, self.model, self.torch


class _ByModel:
    """The class's ``_load`` seam for a whole build: records which checkpoint went on the card, in order."""

    def __init__(self) -> None:
        self.loaded: list[str] = []
        self.torch = _Torch()

    def install(self) -> Any:
        record = self

        def _load(grounder: Qwen3VLGrounder) -> tuple[Any, Any, Any]:
            record.loaded.append(grounder.model_id)
            return _Processor('"intent": "stop"}'), _Model(), record.torch

        return mock.patch.object(Qwen3VLGrounder, "_load", _load)


def _grounder(loads: _Loads, model_id: str = MODEL) -> Qwen3VLGrounder:
    grounder = Qwen3VLGrounder(model_id=model_id, local=True)
    grounder._load = loads  # type: ignore[method-assign]
    return grounder


def _holder(loads: _Loads) -> VlmHolder:
    return VlmHolder.from_parts(build=lambda weights: _grounder(loads, weights.model_id))


def _vlm(**overrides: Any) -> VlmConfig:
    return VlmConfig(**{"model_id": MODEL, "local": True, **overrides})


def _models(*, backend: str = "vlm", router: bool = False, kind: str = "zero_shot", **vlm: Any) -> Any:
    """The one field the availability rule reads, as a loaded config carries it."""
    pipeline = PipelineConfig(kind=kind, zero_shot={"backend": backend, "vlm": _vlm(**vlm)},
                              router={"enabled": router})
    return SimpleNamespace(pipeline=pipeline)


class _Clean(unittest.TestCase):
    def setUp(self) -> None:
        shared_vlm().forget()
        self.addCleanup(shared_vlm().forget)

    def _forgets(self, holder: VlmHolder) -> VlmHolder:
        self.addCleanup(holder.forget)
        return holder


# --- one copy ---------------------------------------------------------------------------------------------------


class TheHolderKeepsOneCopyTests(_Clean):
    def test_the_same_weights_get_the_same_grounder(self) -> None:
        holder = self._forgets(_holder(_Loads()))
        first = holder.grounder_for(_vlm())
        self.assertIs(holder.grounder_for(_vlm(preload=True, on_unavailable="degrade")), first,
                      "preload and on_unavailable name the same weights")

    def test_a_grounder_is_built_without_loading(self) -> None:
        loads = _Loads()
        holder = self._forgets(_holder(loads))
        grounder = holder.grounder_for(_vlm())
        self.assertFalse(grounder.loaded)
        self.assertEqual(loads.calls, 0)
        self.assertEqual(holder.status_for(_vlm()).state, "idle")

    def test_other_weights_replace_the_held_grounder(self) -> None:
        holder = self._forgets(_holder(_Loads()))
        first = holder.grounder_for(_vlm())
        other = holder.grounder_for(_vlm(model_id="Qwen/Qwen3-VL-8B-Instruct"))
        self.assertIsNot(other, first)
        self.assertIsNone(holder.held_for(_vlm()), "one copy at a time: the first is no longer held")
        self.assertIsNot(holder.grounder_for(_vlm()), first)

    def test_forget_drops_what_it_holds(self) -> None:
        holder = self._forgets(_holder(_Loads()))
        first = holder.grounder_for(_vlm())
        holder.forget()
        self.assertIsNone(holder.held_for(_vlm()))
        self.assertIsNot(holder.grounder_for(_vlm()), first)

    def test_status_builds_nothing(self) -> None:
        built: list[VlmWeights] = []
        holder = self._forgets(VlmHolder.from_parts(build=lambda w: built.append(w) or _grounder(_Loads())))
        self.assertEqual(holder.status_for(_vlm()).state, "idle")
        self.assertFalse(holder.loaded_for(_vlm()))
        self.assertEqual(built, [])

    def test_the_process_has_one_holder(self) -> None:
        self.assertIs(shared_vlm(), shared_vlm())

    def test_the_weights_are_what_decide_the_copy(self) -> None:
        self.assertEqual(VlmWeights.of(_vlm()), VlmWeights(model_id=MODEL, model_path=None, local=True))
        self.assertNotEqual(VlmWeights.of(_vlm()), VlmWeights.of(_vlm(local=False)))


class TheBuildsGrounderIsTheHoldersTests(_Clean):
    """The perception build takes its grounder from the shared holder: one copy reads commands and detects."""

    def _build(self, **pipeline: Any) -> tuple[Any, VlmConfig]:
        from src.config.loader import load_config
        from src.models import factory

        models = load_config().models.model_copy(update={"pipeline": PipelineConfig(**pipeline)})
        with mock.patch.object(factory, "_build_named_segmenter", return_value="SEG"):
            backend = factory.build_perception(models)
        assert models.pipeline is not None
        return backend, models.pipeline.zero_shot.vlm

    def test_the_vlm_stack_grounds_with_the_holders_copy(self) -> None:
        backend, vlm = self._build(zero_shot={"backend": "vlm"}, router={"enabled": False})
        self.assertIs(backend._vlm.detector, shared_vlm().grounder_for(vlm))  # noqa: SLF001

    def test_the_routed_stack_builds_its_vlm_route_on_the_holders_copy(self) -> None:
        from src.models.routing import Route

        backend, vlm = self._build(zero_shot={"backend": "vlm"}, router={"enabled": True})
        guarded = backend._backend_for(Route.VLM)  # noqa: SLF001 - the lazy route, built on demand
        self.assertIs(guarded._vlm.detector, shared_vlm().grounder_for(vlm))  # noqa: SLF001

    def test_two_builds_share_one_copy(self) -> None:
        first, _ = self._build(zero_shot={"backend": "vlm"}, router={"enabled": False})
        second, _ = self._build(zero_shot={"backend": "vlm"}, router={"enabled": False})
        self.assertIs(first._vlm.detector, second._vlm.detector)  # noqa: SLF001

    def test_a_preloaded_build_loads_once_and_a_rebuild_loads_nothing(self) -> None:
        """The owner's profile: backend vlm, router off, preload on."""
        loads = _Loads()
        with mock.patch.object(Qwen3VLGrounder, "_load", loads):
            first, vlm = self._build(zero_shot={"backend": "vlm", "vlm": {"preload": True}}, router={"enabled": False})
            second, _ = self._build(zero_shot={"backend": "vlm", "vlm": {"preload": True}}, router={"enabled": False})
        self.assertEqual(loads.calls, 1)
        self.assertTrue(first._vlm.detector.loaded)  # noqa: SLF001
        self.assertIs(first._vlm.detector, second._vlm.detector)  # noqa: SLF001
        self.assertTrue(shared_vlm().loaded_for(vlm))

    def test_the_reader_asks_the_copy_the_cell_detects_with(self) -> None:
        backend, vlm = self._build(zero_shot={"backend": "vlm"}, router={"enabled": False})
        asker = shared_vlm().asker_for(vlm, may_load=True)
        self.assertIs(asker.grounder, backend._vlm.detector)  # noqa: SLF001


class OneCopyWhateverTheConfigNamesTests(_Clean):
    """A cell built with one checkpoint and a config that now names another (an edited YAML, no rebuild): neither a
    command nor Laden puts the second on the card beside the first. Only a build switches the weights, and the build
    unloads the copy it replaces."""

    def _build(self, model_id: str, *, preload: bool = True, router: bool = False) -> Any:
        from src.config.loader import load_config
        from src.models import factory

        pipeline = PipelineConfig(zero_shot={"backend": "vlm",
                                             "vlm": {"model_id": model_id, "local": True, "preload": preload}},
                                  router={"enabled": router})
        models = load_config().models.model_copy(update={"pipeline": pipeline})
        with mock.patch.object(factory, "_build_named_segmenter", return_value="SEG"):
            return factory.build_perception(models)

    def test_a_command_naming_other_weights_is_refused_and_loads_nothing(self) -> None:
        loads = _ByModel()
        with loads.install():
            cell = self._build(MODEL)
            with self.assertRaises(CommandRefused) as caught:
                read_command("Nimm den Würfel", models=_models(model_id=OTHER), poses=POSES, weights_present=True)
        self.assertEqual(caught.exception.code, "vlm_not_loaded")
        self.assertIn("rebuild", caught.exception.cause.lower())
        self.assertEqual(loads.loaded, [MODEL], "a second checkpoint went on the card beside the cell's")
        grounder = cell._vlm.detector  # noqa: SLF001
        self.assertTrue(grounder.loaded)
        self.assertIs(shared_vlm().held_for(_vlm()), grounder, "the refused command took the cell's copy away")

    def test_laden_naming_other_weights_loads_nothing_and_says_why(self) -> None:
        loads = _ByModel()
        with loads.install():
            self._build(MODEL)
            status = shared_vlm().load_for(_vlm(model_id=OTHER))
        self.assertEqual(status.state, "idle")
        self.assertIn(MODEL, status.cause)
        self.assertIn(OTHER, status.cause)
        self.assertEqual(loads.loaded, [MODEL])

    def test_the_rule_refuses_reading_and_loading_while_other_weights_are_held(self) -> None:
        loads = _ByModel()
        with loads.install():
            self._build(MODEL)
        for backend in ("vlm", "grounded_sam"):
            with self.subTest(backend=backend):
                rule = reader_availability(_models(backend=backend, model_id=OTHER), weights_present=True)
                self.assertEqual((rule.refusal, rule.warmup_refusal, rule.may_load),
                                 ("vlm_not_loaded", "vlm_not_loaded", False))
                self.assertIn("rebuild", rule.cause.lower())
                self.assertIn(MODEL, rule.cause)

    def test_a_refused_command_takes_nothing_away(self) -> None:
        """A GroundingDINO cell with a copy a person loaded: a command naming other weights leaves it in place."""
        loads = _ByModel()
        with loads.install():
            shared_vlm().load_for(_vlm())
            held = shared_vlm().held_for(_vlm())
            with self.assertRaises(CommandRefused):
                read_command("Stopp", models=_models(backend="grounded_sam", model_id=OTHER), poses=POSES,
                             weights_present=True)
            with self.assertRaises(VlmNotLoadedError):
                shared_vlm().asker_for(_vlm(model_id=OTHER), may_load=False)
        self.assertIsNotNone(held)
        self.assertIs(shared_vlm().held_for(_vlm()), held)
        self.assertTrue(held is not None and held.loaded)
        self.assertEqual(loads.loaded, [MODEL])

    def test_a_rebuild_switches_the_weights_and_unloads_the_copy_it_replaces(self) -> None:
        loads = _ByModel()
        with loads.install():
            first = self._build(MODEL)
            second = self._build(OTHER)
        old, new = first._vlm.detector, second._vlm.detector  # noqa: SLF001
        self.assertFalse(old.loaded, "the replaced copy still holds the card")
        self.assertTrue(new.loaded)
        self.assertEqual(loads.loaded, [MODEL, OTHER])
        self.assertEqual(loads.torch.cuda.emptied, 1, "the replaced copy's memory went back to the card")

    def test_a_routed_rebuild_naming_other_weights_releases_the_old_copy(self) -> None:
        """A routed cell builds its VLM route lazily, so the build itself lets the old copy go."""
        loads = _ByModel()
        with loads.install():
            first = self._build(MODEL)
            routed = self._build(OTHER, preload=False, router=True)
            self.assertFalse(first._vlm.detector.loaded)  # noqa: SLF001
            self.assertIsNone(shared_vlm().held_for(_vlm()))
            self.assertEqual(routed.built_routes(), ())
            reading = read_command("Stopp", models=_models(router=True, model_id=OTHER), poses=POSES,
                                   weights_present=True)
        self.assertTrue(reading.loaded_now)
        self.assertEqual(loads.loaded, [MODEL, OTHER])

    def test_a_routed_rebuild_naming_the_same_weights_keeps_the_loaded_copy(self) -> None:
        loads = _ByModel()
        with loads.install():
            first = self._build(MODEL)
            self._build(MODEL, preload=False, router=True)
        self.assertTrue(first._vlm.detector.loaded)  # noqa: SLF001
        self.assertIs(shared_vlm().held_for(_vlm()), first._vlm.detector)  # noqa: SLF001
        self.assertEqual(loads.loaded, [MODEL])

    def test_a_cell_that_loads_its_replaced_copy_again_blocks_a_second_one(self) -> None:
        """Two cells in one program, built with different checkpoints: once the first one's look loads its copy
        again, a command naming the second checkpoint is refused rather than loading it beside."""
        loads = _ByModel()
        with loads.install():
            first = self._build(MODEL, preload=False)
            self._build(OTHER, preload=False)
            first._vlm.detector.ensure_loaded()  # noqa: SLF001 - the first cell's look
            with self.assertRaises(CommandRefused) as caught:
                read_command("Stopp", models=_models(model_id=OTHER), poses=POSES, weights_present=True)
        self.assertEqual(caught.exception.code, "vlm_not_loaded")
        self.assertIn("loaded in a cell built with it", caught.exception.cause)
        self.assertEqual(loads.loaded, [MODEL])


# --- one load ---------------------------------------------------------------------------------------------------


class TheLoadLockTests(_Clean):
    def test_two_threads_load_once(self) -> None:
        release = threading.Event()
        loads = _Loads(wait=release)
        grounder = _grounder(loads)
        answers: list[bool] = []
        threads = [threading.Thread(target=lambda: answers.append(grounder.ensure_loaded())) for _ in range(2)]
        for thread in threads:
            thread.start()
        self.assertTrue(loads.started.wait(5.0))
        time.sleep(0.05)  # the second thread is now waiting on the load lock, not loading
        release.set()
        for thread in threads:
            thread.join(5.0)
        self.assertEqual(loads.calls, 1, "the weights were loaded twice")
        self.assertEqual(sorted(answers), [False, True], "exactly one caller loaded them")
        self.assertTrue(grounder.loaded)

    def test_loading_is_visible_while_the_weights_load(self) -> None:
        release = threading.Event()
        loads = _Loads(wait=release)
        holder = self._forgets(_holder(loads))
        loader = threading.Thread(target=lambda: holder.load_for(_vlm()))
        loader.start()
        self.assertTrue(loads.started.wait(5.0))
        try:
            self.assertEqual(holder.status_for(_vlm()).state, "loading")
            self.assertFalse(holder.loaded_for(_vlm()))
        finally:
            release.set()
            loader.join(5.0)
        status = holder.status_for(_vlm())
        self.assertEqual(status.state, "ready")
        self.assertIsNotNone(status.load_s)

    def test_a_failed_load_is_recorded_and_never_reads_loaded(self) -> None:
        loads = _Loads(fail=OSError("no weights on disk"))
        grounder = _grounder(loads)
        with self.assertRaises(OSError):
            grounder.ensure_loaded()
        self.assertFalse(grounder.loaded)
        self.assertFalse(grounder.loading)
        self.assertIsInstance(grounder.load_error, OSError)

    def test_the_next_load_tries_again_and_clears_the_failure(self) -> None:
        loads = _Loads(fail=OSError("no weights on disk"))
        grounder = _grounder(loads)
        with self.assertRaises(OSError):
            grounder.ensure_loaded()
        loads.fail = None
        self.assertTrue(grounder.ensure_loaded())
        self.assertTrue(grounder.loaded)
        self.assertIsNone(grounder.load_error)
        self.assertEqual(loads.calls, 2)

    def test_a_loaded_grounder_loads_nothing_more(self) -> None:
        loads = _Loads()
        grounder = _grounder(loads)
        self.assertTrue(grounder.ensure_loaded())
        self.assertFalse(grounder.ensure_loaded())
        self.assertEqual(loads.calls, 1)

    def test_a_caller_that_waited_on_a_load_that_failed_does_not_load_again(self) -> None:
        """The failure it waited for is its answer too: a second multi-GB attempt on the heels of an OOM or a missing
        file helps nobody. The next caller that was not waiting tries again (a pick's next look, Laden)."""
        release = threading.Event()
        loads = _Loads(fail=OSError("no weights on disk"), wait=release)
        grounder = _grounder(loads)
        grounder._load_lock = lock = _CountingLock()  # type: ignore[assignment]
        failures: list[BaseException] = []

        def load() -> None:
            try:
                grounder.ensure_loaded()
            except OSError as exc:
                failures.append(exc)

        threads = [threading.Thread(target=load) for _ in range(2)]
        for thread in threads:
            thread.start()
        self.assertTrue(loads.started.wait(5.0))
        self.assertTrue(lock.asked(2), "the second caller never waited on the load")
        release.set()
        for thread in threads:
            thread.join(5.0)
        self.assertEqual(loads.calls, 1, "the caller that waited tried the failed load again")
        self.assertEqual(len(failures), 2)
        with self.assertRaises(OSError):
            grounder.ensure_loaded()
        self.assertEqual(loads.calls, 2, "a caller that did not wait tries again")


class TheExplicitLoadTests(_Clean):
    """"Laden" in the ready bar: load, or say why not; a failed load is tried again only here."""

    def test_load_for_loads_and_says_ready(self) -> None:
        loads = _Loads()
        holder = self._forgets(_holder(loads))
        status = holder.load_for(_vlm())
        self.assertEqual((status.state, status.model_id), ("ready", MODEL))
        self.assertEqual(loads.calls, 1)
        self.assertTrue(holder.loaded_for(_vlm()))

    def test_a_load_that_fails_is_reported_not_raised(self) -> None:
        holder = self._forgets(_holder(_Loads(fail=RuntimeError("CUDA out of memory"))))
        status = holder.load_for(_vlm())
        self.assertEqual(status.state, "failed")
        self.assertIn("CUDA out of memory", status.cause)
        self.assertIn("RuntimeError", status.cause)

    def test_load_for_tries_a_failed_load_again(self) -> None:
        loads = _Loads(fail=ImportError("no transformers"))
        holder = self._forgets(_holder(loads))
        self.assertEqual(holder.load_for(_vlm()).state, "failed")
        loads.fail = None
        self.assertEqual(holder.load_for(_vlm()).state, "ready")

    def test_a_bug_in_the_load_is_not_dressed_as_unavailable(self) -> None:
        holder = self._forgets(_holder(_Loads(fail=ValueError("a real bug"))))
        with self.assertRaises(ValueError):
            holder.load_for(_vlm())

    def test_status_renders_ascii_and_serialises(self) -> None:
        holder = self._forgets(_holder(_Loads(fail=OSError("kein Gewicht für Würfel"))))
        status = holder.load_for(_vlm())
        status.render().encode("ascii")
        self.assertEqual(str(status), status.render())
        self.assertEqual(status.to_dict()["state"], "failed")


# --- the text answer --------------------------------------------------------------------------------------------


class TheTextAnswerTests(_Clean):
    """``answer_text``: no image, the prefill continued, greedy, and the prompt never decoded back."""

    def _answered(self, *, prefill: str = "{", answer: str = '"intent": "none"}',
                  max_new_tokens: int = 96) -> tuple[str, _Loads]:
        loads = _Loads(answer=answer)
        grounder = _grounder(loads)
        text = grounder.answer_text("SYSTEM", "COMMAND: Stopp", prefill=prefill, max_new_tokens=max_new_tokens)
        return text, loads

    def test_the_answer_is_the_prefill_and_what_the_model_wrote(self) -> None:
        text, _ = self._answered()
        self.assertEqual(text, '{"intent": "none"}')

    def test_no_image_is_sent_and_the_prefill_is_continued(self) -> None:
        _, loads = self._answered()
        [(messages, kwargs)] = loads.processor.templates
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant"])
        self.assertEqual(messages[0]["content"], [{"type": "text", "text": "SYSTEM"}])
        self.assertEqual(messages[1]["content"], [{"type": "text", "text": "COMMAND: Stopp"}])
        self.assertEqual(messages[2]["content"], [{"type": "text", "text": "{"}])
        parts = [part for message in messages for part in message["content"]]
        self.assertTrue(all(part["type"] == "text" for part in parts), "a text answer sends no image")
        self.assertTrue(kwargs["continue_final_message"])
        self.assertFalse(kwargs["add_generation_prompt"])
        self.assertTrue(kwargs["tokenize"])
        self.assertTrue(kwargs["return_dict"])

    def test_without_a_prefill_the_model_starts_a_new_turn(self) -> None:
        text, loads = self._answered(prefill="", answer='{"intent": "none"}')
        [(messages, kwargs)] = loads.processor.templates
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertTrue(kwargs["add_generation_prompt"])
        self.assertFalse(kwargs["continue_final_message"])
        self.assertEqual(text, '{"intent": "none"}')

    def test_generation_is_greedy_and_bounded(self) -> None:
        _, loads = self._answered(max_new_tokens=160)
        [call] = loads.model.calls
        self.assertIs(call["do_sample"], False)
        self.assertEqual(call["max_new_tokens"], 160)

    def test_the_prompt_tokens_are_stripped_before_decoding(self) -> None:
        """Decoding the whole sequence would hand the instruction's own JSON example back as the answer."""
        _, loads = self._answered()
        [(ids, skip_special)] = loads.processor.decoded
        self.assertEqual(ids.shape, (1, _NEW_TOKENS))
        self.assertTrue((ids == 99).all())
        self.assertTrue(skip_special)

    def test_answer_text_loads_first_and_records_its_time(self) -> None:
        _, loads = self._answered()
        self.assertEqual(loads.calls, 1)

    def test_a_bound_below_one_token_is_refused(self) -> None:
        grounder = _grounder(_Loads())
        with self.assertRaises(ValueError):
            grounder.answer_text("S", "U", max_new_tokens=0)

    def test_the_last_answer_time_is_kept(self) -> None:
        loads = _Loads()
        grounder = _grounder(loads)
        self.assertIsNone(grounder.last_answer_ms)
        grounder.answer_text("S", "U")
        self.assertIsNotNone(grounder.last_answer_ms)


class OneInferenceAtATimeTests(_Clean):
    def test_reading_a_command_and_grounding_a_frame_never_generate_at_once(self) -> None:
        hold = threading.Event()
        loads = _Loads(answer='[{"bbox_2d": [100, 100, 200, 200], "label": "red cube"}]', hold=hold)
        grounder = _grounder(loads)
        grounder.ensure_loaded()
        reader = threading.Thread(target=lambda: grounder.answer_text("S", "U"))
        reader.start()
        self.assertTrue(loads.model.entered.wait(5.0))
        detections: list[Any] = []
        detector = threading.Thread(
            target=lambda: detections.extend(grounder.detect_all(np.zeros((480, 640, 3), np.uint8), "red cube")))
        detector.start()
        time.sleep(0.1)
        self.assertEqual(len(loads.model.calls), 1, "grounding started while a command was being read")
        hold.set()
        reader.join(5.0)
        detector.join(5.0)
        self.assertEqual(len(loads.model.calls), 2)
        self.assertEqual(loads.model.most_inside, 1)
        self.assertEqual(len(detections), 1, "the grounding answer still parses into a box")


class UnloadTests(_Clean):
    """``unload()``: the weights leave the card, never under an answer in flight; whoever needs them loads again."""

    def test_unload_drops_the_weights_and_hands_the_memory_back(self) -> None:
        loads = _Loads()
        grounder = _grounder(loads)
        grounder.ensure_loaded()
        self.assertTrue(grounder.unload())
        self.assertFalse(grounder.loaded)
        self.assertEqual(loads.torch.cuda.emptied, 1)
        self.assertFalse(grounder.unload(), "nothing left to drop")
        self.assertIsNone(grounder.load_s, "an unloaded copy reads as never loaded")

    def test_unload_waits_for_an_answer_in_flight(self) -> None:
        hold = threading.Event()
        loads = _Loads(hold=hold)
        grounder = _grounder(loads)
        grounder.ensure_loaded()
        answers: list[str] = []
        reader = threading.Thread(target=lambda: answers.append(grounder.answer_text("S", "U")))
        reader.start()
        self.assertTrue(loads.model.entered.wait(5.0))
        unloader = threading.Thread(target=grounder.unload)
        unloader.start()
        time.sleep(0.05)
        self.assertTrue(grounder.loaded, "the weights went while an answer was being written")
        hold.set()
        reader.join(5.0)
        unloader.join(5.0)
        self.assertEqual(answers, ['{"intent": "none"}'])
        self.assertFalse(grounder.loaded)

    def test_an_answer_that_may_not_load_refuses_weights_that_are_gone(self) -> None:
        loads = _Loads()
        grounder = _grounder(loads)
        with self.assertRaises(RuntimeError):
            grounder.answer_text("S", "U", load=False)
        self.assertEqual(loads.calls, 0)

    def test_a_look_after_an_unload_loads_again(self) -> None:
        loads = _Loads(answer="[]")
        grounder = _grounder(loads)
        grounder.ensure_loaded()
        grounder.unload()
        self.assertEqual(grounder.detect_all(np.zeros((48, 64, 3), np.uint8), "red cube"), [])
        self.assertEqual(loads.calls, 2)


# --- the asker --------------------------------------------------------------------------------------------------


class TheTextAskerTests(_Clean):
    def test_an_asker_that_may_load_loads_on_first_use_and_says_so(self) -> None:
        loads = _Loads()
        holder = self._forgets(_holder(loads))
        asker = holder.asker_for(_vlm(), may_load=True)
        self.assertIsInstance(asker, TextAsker)
        self.assertEqual(asker("S", "U"), '{"intent": "none"}')
        self.assertTrue(asker.loaded_now)
        self.assertEqual(asker.model_id, MODEL)
        again = holder.asker_for(_vlm(), may_load=True)
        again("S", "U")
        self.assertFalse(again.loaded_now, "the second reader found the weights loaded")
        self.assertEqual(loads.calls, 1)

    def test_an_asker_that_may_not_load_never_loads(self) -> None:
        loads = _Loads()
        holder = self._forgets(_holder(loads))
        with self.assertRaises(VlmNotLoadedError):
            holder.asker_for(_vlm(), may_load=False)("S", "U")
        self.assertEqual(loads.calls, 0)
        self.assertIsNone(holder.held_for(_vlm()), "a refusal built a grounder it could not use")

    def test_an_asker_that_may_not_load_refuses_a_held_copy_that_is_not_loaded(self) -> None:
        loads = _Loads()
        holder = self._forgets(_holder(loads))
        holder.grounder_for(_vlm())  # a cell built with preload off: held, never loaded
        with self.assertRaises(VlmNotLoadedError):
            holder.asker_for(_vlm(), may_load=False)("S", "U")
        self.assertEqual(loads.calls, 0)

    def test_a_load_failure_reaches_the_reader_as_unavailable(self) -> None:
        holder = self._forgets(_holder(_Loads(fail=OSError("no weights on disk"))))
        with self.assertRaises(VlmUnavailableError) as caught:
            holder.asker_for(_vlm(), may_load=True)("S", "U")
        self.assertIsInstance(caught.exception.cause, OSError)

    def test_a_bug_in_the_load_is_not_dressed_as_unavailable(self) -> None:
        holder = self._forgets(_holder(_Loads(fail=ValueError("a real bug"))))
        with self.assertRaises(ValueError):
            holder.asker_for(_vlm(), may_load=True)("S", "U")

    def test_a_command_never_retries_a_failed_load_it_finds(self) -> None:
        """Laden or a pick's look tries a failed load again; a command asks nothing of a copy whose load failed."""
        loads = _Loads(fail=OSError("no weights on disk"))
        holder = self._forgets(_holder(loads))
        self.assertEqual(holder.load_for(_vlm()).state, "failed")
        grounder = holder.held_for(_vlm())
        assert grounder is not None
        with self.assertRaises(VlmUnavailableError) as caught:
            TextAsker(grounder, may_load=True)("S", "U")
        self.assertIsInstance(caught.exception.cause, OSError)
        self.assertEqual(loads.calls, 1)

    def test_a_command_that_waited_on_a_failing_laden_does_not_load_again(self) -> None:
        """A command on a VLM cell arrives while Laden's load runs, waits for it, and the load fails."""
        release = threading.Event()
        loads = _Loads(fail=OSError("no weights on disk"), wait=release)
        holder = self._forgets(_holder(loads))
        grounder = holder.grounder_for(_vlm())
        grounder._load_lock = lock = _CountingLock()  # type: ignore[assignment]
        outcome: list[str] = []
        laden = threading.Thread(target=lambda: outcome.append(holder.load_for(_vlm()).state))
        laden.start()
        self.assertTrue(loads.started.wait(5.0))
        rule = reader_availability(_models(), weights_present=True, holder=holder)
        self.assertEqual((rule.state, rule.refusal, rule.may_load), ("loading", "", True))

        def command() -> None:
            try:
                read_command("Stopp", models=_models(), poses=POSES, weights_present=True, holder=holder)
            except CommandRefused as exc:
                outcome.append(exc.code)

        reader = threading.Thread(target=command)
        reader.start()
        self.assertTrue(lock.asked(2), "the command never waited on Laden's load")
        release.set()
        laden.join(5.0)
        reader.join(5.0)
        self.assertEqual(loads.calls, 1, "the command tried the failed load again")
        self.assertCountEqual(outcome, ["failed", "vlm_unavailable"])

    def test_a_command_waits_for_laden_trying_a_failed_load_again(self) -> None:
        """The last load failed and a person pressed Laden again: a command that arrives meanwhile waits for that
        load and is read with it, rather than being refused with the failure the load is about to clear."""
        loads = _Loads(answer=ReadCommandTests.ANSWER, fail=OSError("no weights on disk"))
        holder = self._forgets(_holder(loads))
        grounder = holder.grounder_for(_vlm())
        grounder._load_lock = lock = _CountingLock()  # type: ignore[assignment]
        self.assertEqual(holder.load_for(_vlm()).state, "failed")
        release = threading.Event()
        loads.fail, loads.wait, loads.started = None, release, threading.Event()
        laden = threading.Thread(target=lambda: holder.load_for(_vlm()))
        laden.start()
        self.assertTrue(loads.started.wait(5.0))
        readings: list[Any] = []
        reader = threading.Thread(target=lambda: readings.append(read_command(
            "Nimm alle Schrauben und leg sie auf Ablage links", models=_models(), poses=POSES, weights_present=True,
            holder=holder)))
        reader.start()
        try:
            self.assertTrue(lock.asked(3), "the command never waited on Laden's load")
        finally:
            release.set()
            laden.join(5.0)
            reader.join(5.0)
        self.assertEqual(loads.calls, 2, "the two Laden loads, and none of the command's own")
        [reading] = readings
        self.assertTrue(reading.understood)
        self.assertFalse(reading.loaded_now, "Laden loaded it, not the command")


class TheAnswerFailsTests(_Clean):
    """The weights are loaded, and the model fails while it answers: on a shared card that is mostly VRAM."""

    def test_a_model_that_fails_while_answering_is_unavailable(self) -> None:
        loads = _Loads()
        loads.model.fail = RuntimeError("CUDA out of memory")
        holder = self._forgets(_holder(loads))
        with self.assertRaises(CommandRefused) as caught:
            read_command("Stopp", models=_models(), poses=POSES, weights_present=True, holder=holder)
        self.assertEqual(caught.exception.code, "vlm_unavailable")
        self.assertIn("CUDA out of memory", caught.exception.cause)
        self.assertIn("answer", caught.exception.cause)

    def test_the_asker_says_unavailable_with_the_cause(self) -> None:
        loads = _Loads()
        loads.model.fail = RuntimeError("CUDA out of memory")
        holder = self._forgets(_holder(loads))
        with self.assertRaises(VlmUnavailableError) as caught:
            holder.asker_for(_vlm(), may_load=True)("S", "U")
        self.assertIsInstance(caught.exception.cause, RuntimeError)

    def test_a_bug_while_answering_is_not_dressed_as_unavailable(self) -> None:
        loads = _Loads()
        loads.processor.fail = ValueError("a real bug")
        holder = self._forgets(_holder(loads))
        with self.assertRaises(ValueError):
            read_command("Stopp", models=_models(), poses=POSES, weights_present=True, holder=holder)


# --- the availability rule (Q8 A) ---------------------------------------------------------------------------------


class TheReaderAvailabilityTests(_Clean):
    """Owner decision Q8 A: one shared copy; loaded at the first command where the VLM is the detector; on a
    GroundingDINO cell only after a person's "Laden"."""

    def _rule(self, models: Any, *, loads: _Loads | None = None, weights_present: bool | None = True,
              prepare: Any = None) -> Any:
        holder = self._forgets(_holder(loads or _Loads()))
        if prepare is not None:
            prepare(holder)
        return reader_availability(models, weights_present=weights_present, holder=holder)

    def test_a_vlm_cell_reads_its_first_command_by_loading_the_detectors_copy(self) -> None:
        rule = self._rule(_models(backend="vlm"))
        self.assertEqual((rule.state, rule.refusal, rule.may_load), ("idle", "", True))
        self.assertTrue(rule.shared_with_detection)
        self.assertTrue(rule.may_parse)

    def test_a_routed_vlm_cell_counts_as_a_vlm_cell(self) -> None:
        rule = self._rule(_models(backend="vlm", router=True))
        self.assertEqual((rule.refusal, rule.may_load, rule.shared_with_detection), ("", True, True))

    def test_a_grounding_dino_cell_refuses_until_a_person_loads_it(self) -> None:
        rule = self._rule(_models(backend="grounded_sam"))
        self.assertEqual((rule.state, rule.refusal, rule.may_load), ("idle", "vlm_not_loaded", False))
        self.assertEqual(rule.warmup_refusal, "", "Laden is how it gets loaded")
        self.assertFalse(rule.shared_with_detection)
        self.assertIn("Laden", rule.cause)

    def test_a_grounding_dino_cell_reads_once_it_was_loaded(self) -> None:
        rule = self._rule(_models(backend="grounded_sam"), prepare=lambda h: h.load_for(_vlm()))
        self.assertEqual((rule.state, rule.refusal), ("ready", ""))
        self.assertFalse(rule.shared_with_detection)

    def test_a_closed_set_cell_is_no_vlm_cell(self) -> None:
        rule = self._rule(_models(kind="closed_set", backend="grounded_sam"))
        self.assertEqual((rule.refusal, rule.may_load), ("vlm_not_loaded", False))

    def test_one_rule_says_which_cell_detects_with_the_vlm(self) -> None:
        """The console releases the copy when it builds a cell this rule says does not detect with it."""
        cases = ((_models(backend="vlm"), True), (_models(backend="vlm", router=True), True),
                 (_models(backend="grounded_sam"), False), (_models(kind="closed_set", backend="grounded_sam"), False),
                 (SimpleNamespace(pipeline=None), False))
        for models, detects in cases:
            with self.subTest(models=models):
                self.assertIs(vlm_detects(models), detects)
                if models.pipeline is not None:
                    self.assertIs(self._rule(models).shared_with_detection, detects)

    def test_a_cell_without_a_pipeline_block_names_no_vlm(self) -> None:
        rule = self._rule(SimpleNamespace(pipeline=None))
        self.assertEqual((rule.state, rule.refusal, rule.warmup_refusal), ("not_configured", "vlm_unavailable",
                                                                           "vlm_unavailable"))
        self.assertIsNone(rule.model_id)

    def test_missing_weights_refuse_both_reading_and_loading(self) -> None:
        rule = self._rule(_models(backend="vlm"), weights_present=False)
        self.assertEqual((rule.state, rule.refusal, rule.warmup_refusal), ("missing", "vlm_model_missing",
                                                                           "vlm_model_missing"))
        self.assertIn("fetch", rule.cause)

    def test_loaded_weights_need_no_files(self) -> None:
        rule = self._rule(_models(backend="vlm"), weights_present=False, prepare=lambda h: h.load_for(_vlm()))
        self.assertEqual((rule.state, rule.refusal), ("ready", ""))

    def test_a_failed_load_is_refused_until_a_person_loads_again(self) -> None:
        loads = _Loads(fail=OSError("no weights on disk"))
        rule = self._rule(_models(backend="vlm"), loads=loads, prepare=lambda h: h.load_for(_vlm()))
        self.assertEqual((rule.state, rule.refusal, rule.warmup_refusal), ("failed", "vlm_unavailable", ""))
        self.assertIn("no weights on disk", rule.cause)
        self.assertFalse(rule.may_load, "a command never retries a failed load by itself")

    def test_missing_weights_still_answer_missing_after_a_failed_load(self) -> None:
        """A preload or a pick's look failed for want of the weights: fetching them is the answer, not Laden."""
        loads = _Loads(fail=OSError("no weights on disk"))
        rule = self._rule(_models(backend="vlm"), loads=loads, weights_present=False,
                          prepare=lambda h: h.load_for(_vlm()))
        self.assertEqual((rule.state, rule.refusal, rule.warmup_refusal), ("missing", "vlm_model_missing",
                                                                           "vlm_model_missing"))

    def test_loading_on_a_vlm_cell_waits_for_the_load(self) -> None:
        release = threading.Event()
        loads = _Loads(wait=release)
        holder = self._forgets(_holder(loads))
        loader = threading.Thread(target=lambda: holder.load_for(_vlm()))
        loader.start()
        try:
            self.assertTrue(loads.started.wait(5.0))
            vlm_cell = reader_availability(_models(backend="vlm"), weights_present=True, holder=holder)
            dino_cell = reader_availability(_models(backend="grounded_sam"), weights_present=True, holder=holder)
        finally:
            release.set()
            loader.join(5.0)
        self.assertEqual((vlm_cell.state, vlm_cell.refusal), ("loading", ""))
        self.assertEqual((dino_cell.state, dino_cell.refusal), ("loading", "vlm_not_loaded"))

    def test_unknown_weights_do_not_refuse(self) -> None:
        rule = self._rule(_models(backend="vlm"), weights_present=None)
        self.assertEqual(rule.refusal, "")

    def test_the_rule_loads_and_builds_nothing(self) -> None:
        loads = _Loads()
        built: list[Any] = []
        holder = self._forgets(VlmHolder.from_parts(build=lambda w: built.append(w) or _grounder(loads)))
        for models in (_models(backend="vlm"), _models(backend="grounded_sam"), SimpleNamespace(pipeline=None)):
            reader_availability(models, weights_present=True, holder=holder)
        self.assertEqual((loads.calls, built), (0, []))

    def test_the_rule_reads_the_shared_holder_by_default(self) -> None:
        shared_vlm().grounder_for(_vlm())._load = _Loads()  # type: ignore[method-assign]
        shared_vlm().load_for(_vlm())
        self.assertEqual(reader_availability(_models(backend="grounded_sam"), weights_present=True).state, "ready")

    def test_the_answer_renders_ascii_and_serialises_without_the_block(self) -> None:
        rule = self._rule(_models(backend="grounded_sam"))
        rule.render().encode("ascii")
        payload = rule.to_dict()
        self.assertNotIn("vlm", payload)
        self.assertEqual(payload["refusal"], "vlm_not_loaded")


class ReleasingTheCopyTests(_Clean):
    """The console builds a cell whose detector is not the VLM: the copy goes unless a person loaded it (Q8 A: such
    a cell holds the VLM only after Laden) and the new config still names its weights."""

    def test_a_copy_nobody_asked_for_is_released_and_unloaded(self) -> None:
        loads = _Loads()
        holder = self._forgets(_holder(loads))
        grounder = holder.grounder_for(_vlm())
        grounder.ensure_loaded()  # a VLM cell's preload, or its first look
        self.assertTrue(holder.release_unless_requested(_vlm()))
        self.assertFalse(grounder.loaded)
        self.assertIsNone(holder.held_for(_vlm()))
        self.assertEqual(loads.torch.cuda.emptied, 1)

    def test_a_copy_a_person_loaded_stays_while_the_config_names_it(self) -> None:
        loads = _Loads()
        holder = self._forgets(_holder(loads))
        holder.load_for(_vlm())
        self.assertFalse(holder.release_unless_requested(_vlm()))
        self.assertTrue(holder.loaded_for(_vlm()))

    def test_a_copy_a_person_loaded_goes_when_the_config_names_other_weights(self) -> None:
        loads = _Loads()
        holder = self._forgets(_holder(loads))
        holder.load_for(_vlm())
        grounder = holder.held_for(_vlm())
        self.assertTrue(holder.release_unless_requested(_vlm(model_id=OTHER)))
        self.assertFalse(grounder is not None and grounder.loaded)
        self.assertIsNone(holder.held_for(_vlm()))

    def test_a_copy_a_person_loaded_goes_when_no_vlm_is_named(self) -> None:
        holder = self._forgets(_holder(_Loads()))
        holder.load_for(_vlm())
        self.assertTrue(holder.release_unless_requested(None))
        self.assertIsNone(holder.held_for(_vlm()))

    def test_nothing_held_is_nothing_released(self) -> None:
        holder = self._forgets(_holder(_Loads()))
        self.assertFalse(holder.release_unless_requested(_vlm()))

    def test_after_a_vlm_cell_a_grounding_dino_cell_waits_for_laden_again(self) -> None:
        """The probe that found it: a VLM cell's preload, then the YAML switched to grounded_sam and rebuilt."""
        loads = _Loads()
        holder = self._forgets(_holder(loads))
        holder.grounder_for(_vlm()).ensure_loaded()
        self.assertEqual(reader_availability(_models(backend="grounded_sam"), weights_present=True,
                                             holder=holder).state, "ready", "the precondition: still held")
        holder.release_unless_requested(_vlm())  # what the console's build does for a cell that is no VLM cell
        rule = reader_availability(_models(backend="grounded_sam"), weights_present=True, holder=holder)
        self.assertEqual((rule.state, rule.refusal, rule.warmup_refusal), ("idle", "vlm_not_loaded", ""))


class ReadCommandTests(_Clean):
    """The console's door: the rule first, then the reader, through the one copy."""

    ANSWER = '"intent": "task", "object": "screw", "object_said": "Schrauben", "place": null, ' \
             '"place_said": null, "place_pose": "ablage_links", "scope": "until_empty", "count": null, ' \
             '"return_to": null}'

    def test_a_grounding_dino_cell_refuses_and_loads_nothing(self) -> None:
        loads = _Loads(answer=self.ANSWER)
        holder = self._forgets(_holder(loads))
        with self.assertRaises(CommandRefused) as caught:
            read_command("Nimm alle Schrauben und leg sie auf Ablage links", models=_models(backend="grounded_sam"),
                         poses=POSES, weights_present=True, holder=holder)
        self.assertEqual(caught.exception.code, "vlm_not_loaded")
        self.assertEqual(loads.calls, 0)

    def test_a_vlm_cell_loads_at_its_first_command(self) -> None:
        loads = _Loads(answer=self.ANSWER)
        holder = self._forgets(_holder(loads))
        reading = read_command("Nimm alle Schrauben und leg sie auf Ablage links", models=_models(backend="vlm"),
                               poses=POSES, weights_present=True, holder=holder)
        self.assertTrue(reading.understood)
        self.assertTrue(reading.loaded_now)
        self.assertEqual(reading.model_id, MODEL)
        self.assertEqual((reading.place_pose, reading.scope), ("ablage_links", "until_empty"))
        again = read_command("Nimm alle Schrauben und leg sie auf Ablage links", models=_models(backend="vlm"),
                             poses=POSES, weights_present=True, holder=holder)
        self.assertFalse(again.loaded_now)
        self.assertEqual(loads.calls, 1)

    def test_missing_weights_refuse_before_any_load(self) -> None:
        loads = _Loads(answer=self.ANSWER)
        holder = self._forgets(_holder(loads))
        with self.assertRaises(CommandRefused) as caught:
            read_command("Stopp", models=_models(backend="vlm"), poses=POSES, weights_present=False, holder=holder)
        self.assertEqual(caught.exception.code, "vlm_model_missing")
        self.assertEqual(loads.calls, 0)

    def test_a_load_that_fails_at_the_first_command_is_unavailable_and_stays_failed(self) -> None:
        loads = _Loads(fail=OSError("no weights on disk"))
        holder = self._forgets(_holder(loads))
        with self.assertRaises(CommandRefused) as caught:
            read_command("Stopp", models=_models(backend="vlm"), poses=POSES, weights_present=True, holder=holder)
        self.assertEqual(caught.exception.code, "vlm_unavailable")
        self.assertIn("no weights on disk", caught.exception.cause)
        with self.assertRaises(CommandRefused) as again:
            read_command("Stopp", models=_models(backend="vlm"), poses=POSES, weights_present=True, holder=holder)
        self.assertEqual(again.exception.code, "vlm_unavailable")
        self.assertEqual(loads.calls, 1, "the second command did not try the load again")

    def test_an_empty_sentence_is_refused_before_the_rule(self) -> None:
        holder = self._forgets(_holder(_Loads()))
        with self.assertRaises(ValueError):
            read_command("  ", models=SimpleNamespace(pipeline=None), poses=POSES, holder=holder)


if __name__ == "__main__":
    unittest.main()
