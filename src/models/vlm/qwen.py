"""Qwen3-VL as a detector: ``detect_all(bgr, prompt) -> [Detection]``, the shape GroundingDINO has.

A grounding VLM produces boxes, and boxes are what
:class:`~src.models.perception_backend.TwoStageBackend` already hands to SAM2. So this is a drop-in
replacement for the detector stage rather than a second kind of pipeline, and the two-stage backend
composes it with the existing segmenter unchanged.

The same loaded model also answers text: :meth:`Qwen3VLGrounder.answer_text` asks it a question
with no image, which is how the console reads an operator's command
(:mod:`src.models.vlm.command`). One copy serves both, held once per process
(:mod:`src.models.vlm.holder`), so the load is guarded by a lock (two callers load it once, and a
caller that waited for a load that failed does not try again on its heels) and the two uses take
turns on one generate lock: the model runs one inference at a time. :meth:`Qwen3VLGrounder.unload`
hands the memory back when the holder retires the copy.

One call grounds several kinds of part (the sorting package, 2026-10-09): a class list,
``class_list_prompt(["green part", "red part"])`` or ``"green part | red part"``, is asked with
:data:`GROUNDING_CLASS_LIST_INSTRUCTION`, so each box comes back under the one description it
matches best, copied exactly, and :func:`classes_of` reads the descriptions back for whoever splits
the boxes by label. A box no one description claims is ``ambiguous`` (``parsing.AMBIGUOUS_LABEL``).
A prompt of one description is asked with :data:`GROUNDING_INSTRUCTION`, byte for byte as before.

No torch, transformers or weights are imported at module import. Everything heavy is loaded inside
:meth:`Qwen3VLGrounder._load`, so this module imports on CI, on macOS and on a box with no
GPU, and a cell that never sends a complex prompt never pays for the model.

Measured against the real 4B weights on the dev box's RTX 5080 (2026-10-01): the load takes about
6.3-6.5 s and holds 8.93 GB, a command's text answer peaks at 9.8-9.9 GB (a retried question is the
longest) and takes about 2.2 s, an unload hands all of it back (0.01 GB still allocated), and the
grounding tests pass (`tests/test_vlm_inference.py`, `tests/test_vlm_command_inference.py`). The
grounding quality on a real cell's scenes, the VRAM on any other card and the choice between the 4B
and 8B checkpoints are unmeasured, so the config default has to follow a measurement rather than the
other way round.

**What a grounding call costs is the answer's length** (measured 2026-10-08 on two of the cell's own
wrist frames with the 4B on that card). The image and the instruction are read in about 0.23 s; after
that every token of the answer is one pass of the model, about 32 ms there and about 140 ms on the
cell's 8B. Three cubes as today's instruction asks for them are 108 tokens. Three switches in the
``vlm`` block cut passes (:meth:`Qwen3VLGrounder.configure`):

* ``stop_at_answer_end`` (on): the answer ends where its JSON array closes, not one pass (the end
  token) or three (after a code fence) later. The tokens before are the ones greedy decoding writes
  anyway, and the parser reads the same boxes from them (:func:`answer_array_closed`). A command's
  reading ends where its JSON object closes, one pass before the end token (:func:`answer_object_closed`).
* ``prompt_lookup_tokens`` (0, off; 10 measured): the next tokens are proposed from the answer
  itself and checked in one pass (:class:`_AnswerLookup`): three cubes in 47 passes instead of 108,
  1.5 s instead of 3.5 s there. Not byte-identical: a pass that checks several tokens rounds
  differently, so a coordinate's last digit can come out one higher or lower, as on another card;
  over 90 calls on the cell's frames 42 answers came out identical, and in 2 the list held one box
  more or one fewer. Plain greedy decoding with a static KV cache, which only computes in another
  order, did the same: 41 identical, the same two lists.
* ``text_prompt_lookup_tokens`` (0, off; 10 measured): the same for :meth:`answer_text`, the
  command reader's answers, whose keys come from the instruction: ten commands in 128 passes
  instead of 617, every answer byte-identical, 3.2 times as fast. With the compact instruction of
  2026-10-08 (2026-10-09, 67 sentences): 1458 passes instead of 2539, and 3 readings came out
  otherwise ("Alle grauen Würfel von der schwarzen Matte ..." said "Alle grauen Würfel ..." where
  the plain answer said "graue Würfel ..."; "Leg den Becher in die Kiste neben Ablage links" placed
  in "bin next to Ablage links" where the plain answer said "bin").

**What a pass costs is the CPU's, not the card's** (measured 2026-10-09 with the 4B on the dev box, its
CPU busy with other work). A pass over one new token is about 2400 kernel launches, and the card runs
them faster than the CPU hands them over: it was busy 45 % of a plain pass there, and the cell's 8B
writes a token in about 140 ms where reading its weights takes about 20. ``decode_graphs`` (``auto``,
the default) hands each stretch between two layers' attentions to the card as one CUDA graph and leaves
the attention over the answer so far to the model's own cache and attention function
(:class:`_DecodeGraphs`), so a pass launches the plain pass's kernels, in its order, on its data. Over
the 90 calls on the cell's frames and 67 command sentences every answer came out byte-identical, with
and without the answer lookup, and the first pass of each shape is checked bit for bit before graphs
answer. A pass took 19.7 ms instead of 46.3, the card busy
83 % of it; with the lookup a token took 10.1 ms instead of 45.7. On the cell, whose CPU is three to
four times slower at this, a pass should cost about 33 to 53 ms (estimated from the CPU's share, not
measured there).

Measured and left out: a shorter answer format. "One line, no code fence" and a "one or two words"
label change nothing on the 4B, which already answers on one line and copies the description as
the label whatever it is shown; an instruction without a label drops it for one box only; forcing
the label out while decoding changed the box count in 16 of 90 calls on the cell's frames and broke
the JSON in some. A static KV cache rounds differently too and is no faster without a compiler:
it attends over the whole cache behind a mask, which this torch's math attention (Windows has no
flash attention) sums in another order: 4 in 10 of one layer's outputs moved in their last bit.
``torch.compile`` needs triton (absent here), its ``cudagraphs`` backend ran ten times slower, and
no Qwen3-VL smaller than the 4B is on the box to draft with.
"""

from __future__ import annotations

import contextlib
import functools
import json
import re
import sys
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from src.models.constants import MODELS_LOG_DIR, VLM_GROUNDER_LOG_FILE
from src.models.detection.types import Detection
from src.utility.log_cfg import create_logger

from .parsing import AMBIGUOUS_LABEL, CoordinateSpace, parse_grounding_response

__all__ = [
    "CLASS_LIST_SEPARATOR", "DECODE_GRAPHS", "GROUNDING_CLASS_LIST_INSTRUCTION", "GROUNDING_INSTRUCTION",
    "GROUNDING_MAX_NEW_TOKENS", "Qwen3VLGrounder", "answer_array_closed", "answer_object_closed", "class_list_prompt",
    "classes_of", "grounding_instruction",
]

#: A file sink: nothing in this repo configures the root logger, so a bare ``getLogger`` would
#: format the lines below and then discard them. The name stays ``__name__``, matching its siblings
#: in this package rather than the class-named loggers elsewhere in ``models``. ``create_logger`` is
#: stdlib-only, so the no-torch-at-import promise above holds.
_LOG = create_logger(__name__, log_file=VLM_GROUNDER_LOG_FILE, log_dir=MODELS_LOG_DIR)

#: How many tokens a grounding answer may run to. Qwen3-VL spends about 33 tokens on a box written compactly and
#: 47 on one fenced and indented, as it often answers: 512 tokens, the earlier cap, held 10 to 15 boxes, and a pile of
#: more parts on the mat came back cut off mid-list, which no JSON reading takes (measured on the checkpoint's own
#: tokenizer, 2026-10-06). 2048 holds about 40; greedy decoding stops at the answer's end, so a short answer costs
#: no more than before.
GROUNDING_MAX_NEW_TOKENS = 2048

#: The instruction wrapped around every operator prompt.
#:
#: It asks for Qwen's documented ``bbox_2d`` grounding format and for an empty list when the object
#: is absent. Instruct-tuned models are agreeable by default: without that sentence they invent a
#: plausible box rather than return nothing, and an invented box becomes a grasp at the wrong place.
GROUNDING_INSTRUCTION = (
    "Locate every object matching this description and return ONLY a JSON array, no prose:\n"
    '[{{"bbox_2d": [x0, y0, x1, y1], "label": "<short name>"}}]\n'
    "Coordinates are absolute pixels in this image. "
    "If no object matches, return an empty array []. Do not guess.\n\n"
    "Description: {prompt}"
)

#: What separates the descriptions of a class list (:func:`class_list_prompt`): ``"green part | red part"``. A bar,
#: which no description the command reader writes holds, with a space either side, so the prompt reads as words in a
#: log and on the camera's window.
CLASS_LIST_SEPARATOR = " | "

#: The instruction wrapped around a class list (:func:`class_list_prompt`): several kinds of part found in one call,
#: each box under the one description it matches best, copied exactly, so a box's label says which description
#: claims it. It is :data:`GROUNDING_INSTRUCTION` with a list of descriptions in place of one: the same answer format,
#: the same sentence on coordinates, the same empty array and the same "Do not guess". A box the model names by none,
#: or one object it boxes under two, is read as no description's (``parsing.AMBIGUOUS_LABEL``). How well the cell's 8B
#: keeps two colours apart in one call is not measured yet (``tests/test_vlm_class_list_inference.py``).
GROUNDING_CLASS_LIST_INSTRUCTION = (
    "Locate every object matching one of these descriptions and return ONLY a JSON array, no prose:\n"
    '[{{"bbox_2d": [x0, y0, x1, y1], "label": "<the description it matches, copied exactly>"}}]\n'
    "Coordinates are absolute pixels in this image. "
    "Give each object once, in a box of its own, labelled with the one description it matches best; "
    "leave out every object that matches none. "
    "If no object matches, return an empty array []. Do not guess.\n\n"
    "Descriptions: {descriptions}"
)


def _said_once(text: str) -> str:
    """A description as two of them are compared: case and spacing aside, as the parser compares labels."""
    return " ".join(text.casefold().split())


def class_list_prompt(phrases: Iterable[str]) -> str:
    """One grounding prompt for several descriptions, ``"green part | red part"``: every rule's kind of part, or every
    place a task still looks for, found in one call and told apart by label (:func:`classes_of` reads it back).

    Each phrase is stripped, and a phrase given twice (case and spacing aside) is one description, kept as it was first
    written. One description is no list: it comes back as it is and is grounded with today's instruction, byte for
    byte, so a task of one rule asks what it always asked. Raises ``ValueError`` for no phrase, an empty one, one that
    holds the separator's bar, and one that holds the word :data:`~src.models.vlm.parsing.AMBIGUOUS_LABEL`, the label
    of a box no description clearly claims, which would otherwise take that box for its own.
    """
    descriptions: list[str] = []
    for phrase in phrases:
        text = str(phrase).strip()
        if not text:
            raise ValueError("a class list holds descriptions, and one of them is empty")
        if CLASS_LIST_SEPARATOR.strip() in text:
            raise ValueError(f"the description {text!r} holds {CLASS_LIST_SEPARATOR.strip()!r}, which separates the "
                             "descriptions of a class list")
        if AMBIGUOUS_LABEL in re.findall(r"[a-z0-9]+", text.casefold()):
            raise ValueError(f"the description {text!r} holds the word {AMBIGUOUS_LABEL!r}, the label of a box no "
                             "description clearly claims")
        if _said_once(text) not in {_said_once(description) for description in descriptions}:
            descriptions.append(text)
    if not descriptions:
        raise ValueError("a class list needs at least one description")
    return CLASS_LIST_SEPARATOR.join(descriptions)


def classes_of(prompt: str) -> tuple[str, ...]:
    """The descriptions of a class list (:func:`class_list_prompt`), in its order; ``()`` for a prompt of one
    description, which is grounded as it always was.

    The prompt is split at the separator's bar, each piece stripped; an empty piece is left out and a piece said twice
    (case and spacing aside) is one description. Fewer than two descriptions left is no list. This is what tells the
    grounder to ask with :data:`GROUNDING_CLASS_LIST_INSTRUCTION` and the locator to map each box onto the description
    it names.
    """
    bar = CLASS_LIST_SEPARATOR.strip()
    if bar not in str(prompt):
        return ()
    descriptions: list[str] = []
    for piece in str(prompt).split(bar):
        text = piece.strip()
        if text and _said_once(text) not in {_said_once(description) for description in descriptions}:
            descriptions.append(text)
    return tuple(descriptions) if len(descriptions) >= 2 else ()


def grounding_instruction(prompt: str) -> str:
    """The text a grounding call asks beside its image: :data:`GROUNDING_INSTRUCTION` around a prompt of one
    description, byte for byte as ever, and :data:`GROUNDING_CLASS_LIST_INSTRUCTION` around the descriptions of a class
    list (:func:`classes_of`), each in double quotes as a JSON string writes it, so the label the model copies into its
    answer reads back as the description."""
    descriptions = classes_of(prompt)
    if not descriptions:
        return GROUNDING_INSTRUCTION.format(prompt=prompt)
    return GROUNDING_CLASS_LIST_INSTRUCTION.format(
        descriptions="; ".join(json.dumps(description, ensure_ascii=False) for description in descriptions))


def answer_array_closed(text: str) -> bool:
    """Whether a grounding answer has written one whole JSON array, so it may end here.

    Only an answer that opens with ``[`` (after a ```` ```json ```` line, if any) ends early, and only where its
    brackets and braces are both back to none outside strings: the parser then reads the same array from the
    answer cut here as from the whole one, whose tail is no more than a newline, the closing fence and the end
    token. Prose before the array, a cut-off list or unbalanced braces never end early.
    """
    body = text.lstrip()
    if body.startswith("```"):
        newline = body.find("\n")
        if newline < 0:
            return False
        body = body[newline + 1:].lstrip()
    if not body.startswith("["):
        return False
    brackets = braces = 0
    in_string = escaped = False
    for char in body:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "[":
            brackets += 1
        elif char == "]":
            brackets -= 1
            if brackets == 0 and braces == 0:
                return True
        elif char == "{":
            braces += 1
        elif char == "}":
            braces -= 1
    return False


def answer_object_closed(text: str) -> bool:
    """Whether a text answer has written one whole JSON object, so it may end here: the command reader's answer, which
    opens with the prefill ``{`` (:meth:`Qwen3VLGrounder.answer_text`).

    Only an answer that opens with ``{`` ends early, and only where its braces first close outside strings and all it
    holds there reads as one JSON object. The command reader takes an answer's first whole object
    (:func:`src.models.vlm.command.extract_command_object`), and what the model writes after one is the end token
    alone (every answer measured, 2026-10-09), so the cut answer reads as the whole one. An object that does not read
    where its braces close (a doubled brace, a missing comma) never ends early: the model ends it, and the reader looks
    further, as it always did.
    """
    body = text.lstrip()
    if not body.startswith("{"):
        return False
    depth = 0
    in_string = escaped = False
    for index, char in enumerate(body):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    value, end = json.JSONDecoder().raw_decode(body)
                except ValueError:
                    return False
                return isinstance(value, dict) and end == index + 1
    return False


class _EndOfAnswer:
    """A stopping criterion: the answer ends once it has written one whole JSON value, the array of a grounding answer
    (:func:`answer_array_closed`) or the object of a text answer (:func:`answer_object_closed`).

    The model's own end comes one pass later on a one-line answer and three later after a code fence (a newline,
    the fence, the end token). The answer is read whole only when one of its newest tokens holds the closing bracket,
    which each token is asked once; a text answer is read with its prefill in front, which the prompt holds. Duck-typed
    to transformers' ``StoppingCriteria``, so nothing here imports it.
    """

    #: How many of the newest tokens are looked at for a closing bracket: more than a pass ever adds.
    NEWEST = 32

    def __init__(self, processor: Any, torch: Any, prompt_length: int, *, bracket: str = "]",
                 closed: Callable[[str], bool] = answer_array_closed, prefix: str = "") -> None:
        self._processor = processor
        self._torch = torch
        self._start = int(prompt_length)
        self._bracket = bracket
        self._closed = closed
        self._prefix = prefix
        #: Whether a token's text holds the closing bracket, by token id.
        self._closes: dict[int, bool] = {}

    def __call__(self, input_ids: Any, scores: Any, **_: Any) -> Any:
        written = input_ids[0, self._start:]
        done = False
        if int(written.shape[0]) and self._closing(written[-self.NEWEST:].tolist()):
            whole = self._processor.batch_decode([written.tolist()], skip_special_tokens=True)[0]
            done = self._closed(self._prefix + str(whole))
        return self._torch.full((int(input_ids.shape[0]),), done, dtype=self._torch.bool, device=input_ids.device)

    def _closing(self, tokens: list[int]) -> bool:
        found = False
        for token in tokens:
            closes = self._closes.get(int(token))
            if closes is None:
                text = self._processor.batch_decode([[int(token)]], skip_special_tokens=True)[0]
                closes = self._closes[int(token)] = self._bracket in str(text)
            found = found or closes
        return found


def _token_id(tokenizer: Any, token: str) -> int | None:
    """The id of a one-token string, ``None`` where the tokenizer has no such token."""
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(convert):
        return None
    value = convert(token)
    return int(value) if isinstance(value, int) and value != getattr(tokenizer, "unk_token_id", None) else None


def _as_ids(value: Any) -> list[int]:
    """An ``eos_token_id`` as a list: transformers allows one id, a list of them, or ``None``."""
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    return [int(item) for item in value]


class _AnswerLookup:
    """Proposes an answer's next tokens from what followed the longest earlier match of its tail.

    transformers' assisted decoding checks the proposals in one pass of the model and keeps those the model itself
    picks (greedy) plus one token of its own, so a pass writes one token or several. Its own prompt lookup takes the
    first match from the left, which in a grounding call lies in the image's span (an image token with no features
    behind it fails the pass) or in the instruction's example. This one searches after the image only, takes the
    most recent of the longest matches, and treats every digit alike while matching: a box's tail finds the
    previous box's tail whatever its numbers, so the separators and the next box's opening come as proposals and
    mostly the coordinates' digits cost a pass each. Never proposes an image placeholder, nor an end token or
    anything after one (an accepted end in the middle of a pass would not stop the answer).

    The ``CandidateGenerator`` interface, duck-typed: ``get_candidates`` and ``update_candidate_strategy``.
    """

    #: The longest tail matched, in tokens.
    LONGEST = 16

    def __init__(self, torch: Any, *, tokens: int, digits: frozenset[int], never: frozenset[int],
                 ends: frozenset[int], after: int | None, max_length: int) -> None:
        self._torch = torch
        self._tokens = int(tokens)
        self._digits = digits
        self._never = never
        self._ends = ends
        self._after = after
        self._max_length = int(max_length)
        self._start: int | None = None
        self._ids: list[int] = []
        self._classes: list[int] = []

    def get_candidates(self, input_ids: Any) -> tuple[Any, None]:
        import numpy as np  # noqa: PLC0415 - numpy is no heavier than the model this runs beside

        length = int(input_ids.shape[1])
        if length < len(self._ids):  # not the answer grown since the last call: read it afresh
            self._ids, self._classes = [], []
        if length > len(self._ids):
            new = [int(token) for token in input_ids[0, len(self._ids):].tolist()]
            self._ids.extend(new)
            self._classes.extend(-1 if token in self._digits else token for token in new)
        if self._start is None:
            self._start = 0
            if self._after is not None and self._after in self._ids:
                self._start = len(self._ids) - self._ids[::-1].index(self._after)
        room = self._max_length - length - 1
        region = np.asarray(self._classes[self._start:], dtype=np.int64)
        size = int(region.shape[0])
        if room <= 0 or size < 2:
            return input_ids, None
        ends = np.arange(1, size)
        alive = np.ones(size - 1, dtype=bool)
        matched = np.zeros(size - 1, dtype=np.int64)
        for span in range(1, min(self.LONGEST, size - 1) + 1):
            before = ends - span
            valid = before >= 0
            same = np.zeros(size - 1, dtype=bool)
            same[valid] = region[before[valid]] == region[size - span]
            alive &= same
            if not alive.any():
                break
            matched[alive] = span
        longest = int(matched.max())
        if longest == 0:
            return input_ids, None
        at = self._start + int(ends[np.flatnonzero(matched == longest)[-1]])
        proposal: list[int] = []
        for token in self._ids[at:at + min(self._tokens, room)]:
            if token in self._never or token in self._ends:
                break
            proposal.append(token)
        if not proposal:
            return input_ids, None
        tail = self._torch.tensor([proposal], dtype=input_ids.dtype, device=input_ids.device)
        return self._torch.cat((input_ids, tail), dim=1), None

    def update_candidate_strategy(self, input_ids: Any, scores: Any, num_matches: int) -> None:
        """Nothing to learn: the proposals come from the answer itself."""


# --- the decode pass as CUDA graphs ---------------------------------------------------------------------------------

#: What ``decode_graphs`` may be set to (:meth:`Qwen3VLGrounder.configure`, the ``vlm`` block's key): ``auto`` decodes
#: with graphs where this copy's card can take them, ``on`` wherever they can be captured, ``off`` never.
DECODE_GRAPHS = ("auto", "on", "off")

#: The oldest card ``auto`` decodes with graphs on: compute capability 8.0 (Ampere), the first to compute bf16, the
#: model's own type, natively. Below it a pass waits on the card's own arithmetic rather than on the CPU, and graphs
#: save little; ``on`` captures there too.
GRAPHS_AUTO_CAPABILITY = (8, 0)

#: The most tokens one pass hands to graphs: a plain pass writes one, the answer lookup checks up to 64 (the bound of
#: ``prompt_lookup_tokens``) and the model's own one more. A longer pass runs plainly.
GRAPHS_MAX_TOKENS = 65

#: The one stream every capture in this process runs on, per device, and the lock captures take turns on it. cuBLAS
#: keeps a workspace for each stream it has computed on, for as long as the process runs: a stream of each loaded copy's
#: own left 8.5 MB on the card at every load and unload (measured), one stream leaves them once.
_CAPTURE_STREAMS: dict[str, Any] = {}
_CAPTURE_LOCK = threading.Lock()


def _graphs_unsupported(model: Any, torch: Any, *, auto: bool) -> str:
    """Why ``model`` decodes plainly, or ``""`` where its decode pass can be captured.

    Read off this copy, never off a list of cards: CUDA graphs in this torch, the model on a CUDA device (for ``auto``
    one of compute capability 8.0 or newer, :data:`GRAPHS_AUTO_CAPABILITY`), and a decoder laid out as Qwen3-VL's,
    whose layers :class:`_DecodeGraphs` splits around their attention. A double of the model or of torch is simply
    not supported.
    """
    cuda: Any = getattr(torch, "cuda", None)
    if not callable(getattr(cuda, "CUDAGraph", None)) or not callable(getattr(cuda, "is_available", None)) \
            or not cuda.is_available():
        return "this torch has no CUDA graphs"
    device = getattr(model, "device", None)
    if getattr(device, "type", None) != "cuda":
        return f"the model is on {device}, not on a CUDA device"
    if auto:
        capability = tuple(cuda.get_device_capability(device))
        if capability < GRAPHS_AUTO_CAPABILITY:
            return (f"{cuda.get_device_name(device)} computes at capability {capability[0]}.{capability[1]}, below "
                    f"{GRAPHS_AUTO_CAPABILITY[0]}.{GRAPHS_AUTO_CAPABILITY[1]} (decode_graphs: on captures there too)")
    text = getattr(getattr(model, "model", None), "language_model", None)
    modeling = sys.modules.get(type(text).__module__)
    parts = [getattr(text, name, None) for name in ("embed_tokens", "rotary_emb", "norm", "config")]
    parts += [getattr(model, "lm_head", None)]
    parts += [getattr(modeling, name, None) for name in ("apply_rotary_pos_emb", "create_causal_mask",
                                                         "eager_attention_forward")]
    layers = list(getattr(text, "layers", None) or ())
    for layer in layers:
        attention = getattr(layer, "self_attn", None)
        parts += [getattr(layer, name, None) for name in ("input_layernorm", "post_attention_layernorm", "mlp")]
        parts += [getattr(attention, name, None) for name in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm",
                                                              "k_norm", "head_dim", "scaling")]
    if not layers or any(part is None for part in parts):
        return f"{type(model).__name__} is not laid out as the Qwen3-VL decoder these graphs split"
    return ""


@dataclass
class _GraphSet:
    """The graphs of one pass shape, and the tensors they read and write.

    ``tokens`` and ``positions`` are the pass's inputs, copied in before a replay; ``attended`` holds each layer's
    attention output, copied in between two pieces; ``queries``, ``keys`` and ``values`` are what each piece leaves
    for the next layer's attention, and ``logits`` what the last one leaves. All of them stay where the capture put
    them, which is what a replay reads and writes.
    """

    tokens: Any
    positions: Any
    attended: list[Any]
    pieces: list[Any] = field(default_factory=list)
    pool: Any = None
    embeds: Any = None
    queries: list[Any] = field(default_factory=list)
    keys: list[Any] = field(default_factory=list)
    values: list[Any] = field(default_factory=list)
    logits: Any = None
    #: The rotary cosines and sines every layer reads: kept, since a replay reads them where the capture left them.
    rotary: Any = None
    capture_ms: float = 0.0


class _DecodeGraphs:
    """One loaded copy's decode passes as CUDA graphs, each layer's attention over the answer so far left to the model.

    A pass over one new token is about 2400 kernel launches, and the card runs them faster than the CPU hands them
    over: the cell's 8B writes a token in about 140 ms where reading its weights takes about 20. A CUDA graph hands a
    captured run of kernels over in one call. What cannot be captured is the attention over the answer so far: its
    kernels are shaped by the answer's length, which grows each pass, and this torch has no flash attention on Windows,
    so the model's SDPA runs its math kernels for each length. A static cache would keep the shapes fixed by attending
    over the whole cache with a mask over its unused tail, and that rounds differently (measured: 4 in 10 of one
    layer's outputs moved in their last bit, as the answer lookup's passes do). Graphs per length would capture it
    too, at about 1 MB of the card and 160 KB of the host for each layer and length (measured), thousands of them.

    So each pass is split around every layer's attention. The graphs run the rest, one per stretch between two
    attentions: the embedding and the rotary positions, the norms and projections, the MLPs, the head. They are
    captured once per pass shape (tokens a pass, logits kept) and call the model's own modules, so they launch the
    plain pass's kernels. Between two of them the layer's attention runs as the plain pass runs it, through the
    model's own cache (``DynamicCache.update``) and attention function, and its output is copied into the next
    graph's input. The first pass of each shape is checked against the plain pass bit for bit, logits and cache, and
    under ``auto`` graphs answer only where it matched.

    Installed as the model's ``forward`` for one ``generate`` at a time, under the generate lock: the prefill (the
    image and the instruction, an empty cache) and any call it does not know run the model's own ``forward``. A
    capture or a replay that raises turns graphs off for the loaded copy, the cache is cut back to before the pass,
    and the pass runs plainly: an answer never pays for the speed-up.
    """

    def __init__(self, model: Any, torch: Any, *, model_id: str) -> None:
        from transformers.cache_utils import DynamicCache, DynamicLayer  # noqa: PLC0415 - loaded with the model already
        from transformers.modeling_outputs import CausalLMOutputWithPast  # noqa: PLC0415
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS  # noqa: PLC0415

        self.model = model
        self._torch = torch
        self._model_id = model_id
        self._text = model.model.language_model
        self._layers = list(self._text.layers)
        modeling = sys.modules[type(self._text).__module__]
        self._rotate = modeling.apply_rotary_pos_emb
        self._causal_mask = modeling.create_causal_mask
        self._eager_attention = modeling.eager_attention_forward
        self._attention_functions = ALL_ATTENTION_FUNCTIONS
        self._cache_types = (DynamicCache, DynamicLayer)
        self._output = CausalLMOutputWithPast
        self._sets: dict[tuple[int, int], _GraphSet] = {}
        self._checked: set[tuple[int, int]] = set()
        self._warm = False
        #: ``auto`` or ``on``, as the answer in flight was asked with: ``on`` keeps a shape whose check differed.
        self.mode = "auto"
        #: Why graphs stopped on this copy, if they did: every pass runs plainly until a reload.
        self.failure: str | None = None

    def forward_for(self, original: Callable[..., Any]) -> Callable[..., Any]:
        """``original`` (the model's ``forward``), with the decode passes the graphs take run through them."""

        @functools.wraps(original)
        def forward(*args: Any, **kwargs: Any) -> Any:
            shape = None if args or self.failure is not None else self._shape_of(kwargs)
            if shape is None:
                return original(*args, **kwargs)
            cache = kwargs["past_key_values"]
            past = int(cache.get_seq_length())
            try:
                if shape not in self._checked:
                    return self._checked_pass(original, kwargs, shape, past)
                return self._output(logits=self._pass(kwargs, shape), past_key_values=cache)
            except Exception as exc:  # noqa: BLE001 - any failure of the speed-up falls back to the plain pass
                self._fail(f"{type(exc).__name__}: {exc}")
                cache.crop(past)
                return original(**kwargs)

        return forward

    def _shape_of(self, kwargs: dict[str, Any]) -> tuple[int, int] | None:
        """``(tokens a pass, logits kept)`` of a decode pass the graphs take; ``None`` for any other call: the prefill, an
        image, a batch, a cache of another kind, or outputs asked for that the graphs do not keep."""
        tokens, cache, positions = kwargs.get("input_ids"), kwargs.get("past_key_values"), kwargs.get("position_ids")
        if tokens is None or cache is None or positions is None or kwargs.get("use_cache") is False:
            return None
        if any(kwargs.get(name) is not None for name in ("pixel_values", "pixel_values_videos", "inputs_embeds",
                                                          "labels")):
            return None
        if kwargs.get("output_attentions") or kwargs.get("output_hidden_states"):
            return None
        whole, layer = self._cache_types
        layers = getattr(cache, "layers", None)
        if type(cache) is not whole or not layers or any(type(each) is not layer for each in layers):
            return None
        steps = int(tokens.shape[-1])
        past = int(cache.get_seq_length())
        if past == 0 or tokens.ndim != 2 or int(tokens.shape[0]) != 1 or not 1 <= steps <= GRAPHS_MAX_TOKENS:
            return None
        if tuple(positions.shape) != (4, 1, steps):
            return None
        mask = kwargs.get("attention_mask")
        if mask is not None and tuple(mask.shape) != (1, past + steps):
            return None
        keep = kwargs.get("logits_to_keep", 0)
        if isinstance(keep, bool) or not isinstance(keep, int) or not 0 <= keep <= steps:
            return None
        return steps, keep

    # --- one layer, split around its attention: the model's own modules, in the model's own order ----------------

    def _before_attention(self, layer: Any, hidden: Any, cos: Any, sin: Any) -> tuple[Any, Any, Any]:
        """A layer's norm, projections, query and key norms and rotary positions: what its attention reads."""
        attention = layer.self_attn
        normed = layer.input_layernorm(hidden)
        shape = (*normed.shape[:-1], -1, attention.head_dim)
        query = attention.q_norm(attention.q_proj(normed).view(shape)).transpose(1, 2)
        key = attention.k_norm(attention.k_proj(normed).view(shape)).transpose(1, 2)
        value = attention.v_proj(normed).view(shape).transpose(1, 2)
        query, key = self._rotate(query, key, cos, sin)
        return query, key, value

    @staticmethod
    def _after_attention(layer: Any, residual: Any, attended: Any) -> Any:
        """A layer from its attention's output on: the output projection, both residuals, the MLP."""
        out = layer.self_attn.o_proj(attended.reshape(*residual.shape[:-1], -1).contiguous())
        hidden = residual + out
        return hidden + layer.mlp(layer.post_attention_layernorm(hidden))

    def _run_pieces(self, graphs: _GraphSet, keep: int, piece: Callable[[int], Any],
                    between: Callable[[int], None] | None = None) -> None:
        """Every stretch of a pass between two attentions, in order, each inside ``piece(index)`` (a capture, or nothing
        for the warm-up), and ``between(index)`` after each but the last: nothing while capturing, since a replayed pass
        runs each layer's attention between two replays (:meth:`_pass`), and :meth:`_attend` for a pass run eagerly.
        ``graphs`` keeps what crosses from one stretch to the next."""
        text, layers = self._text, self._layers
        graphs.queries, graphs.keys, graphs.values = [], [], []
        hidden = None
        for index, layer in enumerate(layers):
            with piece(index):
                if index == 0:
                    hidden = text.embed_tokens(graphs.tokens)
                    cos, sin = text.rotary_emb(hidden, graphs.positions[1:])
                    graphs.embeds, graphs.rotary = hidden, (cos, sin)
                else:
                    hidden = self._after_attention(layers[index - 1], hidden, graphs.attended[index - 1])
                query, key, value = self._before_attention(layer, hidden, *graphs.rotary)
            graphs.queries.append(query)
            graphs.keys.append(key)
            graphs.values.append(value)
            if between is not None:
                between(index)
        with piece(len(layers)):
            hidden = text.norm(self._after_attention(layers[-1], hidden, graphs.attended[-1]))
            graphs.logits = self.model.lm_head(hidden[:, slice(-keep, None), :])

    def _attention_inputs(self, graphs: _GraphSet, kwargs: dict[str, Any]) -> tuple[Any, Any, Any]:
        """``(mask, attention function, text positions)`` for one pass, as the model's text forward makes them: the first
        row of the positions is the text's own, and the mask comes from the shapes and the cache, never from the
        embeddings' values."""
        config = self._text.config
        text_positions = kwargs["position_ids"][0]
        mask = self._causal_mask(config=config, inputs_embeds=graphs.embeds, attention_mask=kwargs.get("attention_mask"),
                                 past_key_values=kwargs["past_key_values"], position_ids=text_positions)
        attention = self._attention_functions.get_interface(config._attn_implementation, self._eager_attention)
        return mask, attention, text_positions

    def _attend(self, graphs: _GraphSet, index: int, cache: Any, inputs: tuple[Any, Any, Any]) -> None:
        """Layer ``index``'s attention as the plain pass runs it: the cache updated with what the stretch before left
        (``DynamicCache.update``), the model's attention function over all of it, and its output copied to where the
        next stretch reads it."""
        mask, attention, text_positions = inputs
        attention_module = self._layers[index].self_attn
        keys, values = cache.update(graphs.keys[index], graphs.values[index], index)
        attended, _ = attention(attention_module, graphs.queries[index], keys, values, mask, dropout=0.0,
                                scaling=attention_module.scaling, position_ids=text_positions, use_cache=False)
        graphs.attended[index].copy_(attended)

    def _capture(self, shape: tuple[int, int]) -> _GraphSet:
        """Capture the pieces of one pass shape, into a memory pool of their own, on a stream of this copy's."""
        torch = self._torch
        steps, keep = shape
        started = time.perf_counter()
        weight = self.model.lm_head.weight
        device, dtype = weight.device, weight.dtype
        heads = int(self._text.config.num_attention_heads)
        graphs = _GraphSet(
            tokens=torch.zeros((1, steps), dtype=torch.long, device=device),
            positions=torch.zeros((4, 1, steps), dtype=torch.long, device=device),
            attended=[torch.zeros((1, steps, heads, layer.self_attn.head_dim), dtype=dtype, device=device)
                      for layer in self._layers],
        )
        with _CAPTURE_LOCK:
            stream = _CAPTURE_STREAMS.get(str(device))
            if stream is None:
                stream = _CAPTURE_STREAMS[str(device)] = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                if not self._warm:
                    # Once a copy, eagerly, on the stream that captures: cuBLAS sets up its handle and workspace for a
                    # stream the first time it computes there, and a module may set itself up on its first call,
                    # neither of which a capture may do.
                    self._run_pieces(graphs, keep, lambda index: contextlib.nullcontext())
                    self._warm = True
                graphs.pool = torch.cuda.graph_pool_handle()
                graphs.pieces = [torch.cuda.CUDAGraph() for _ in range(len(self._layers) + 1)]

                @contextlib.contextmanager
                def piece(index: int) -> Iterator[None]:
                    graph = graphs.pieces[index]
                    # Thread-local: another thread's CUDA work (a SAM2 mask, a transcription) never spoils the capture.
                    graph.capture_begin(graphs.pool, capture_error_mode="thread_local")
                    try:
                        yield
                    finally:
                        graph.capture_end()

                self._run_pieces(graphs, keep, piece)
            torch.cuda.current_stream(device).wait_stream(stream)
        graphs.capture_ms = (time.perf_counter() - started) * 1000.0
        self._sets[shape] = graphs
        return graphs

    def _pass(self, kwargs: dict[str, Any], shape: tuple[int, int]) -> Any:
        """One decode pass through the graphs of ``shape``, the cache updated as the plain pass updates it: the logits."""
        graphs = self._sets.get(shape) or self._capture(shape)
        cache = kwargs["past_key_values"]
        graphs.tokens.copy_(kwargs["input_ids"])
        graphs.positions.copy_(kwargs["position_ids"])
        inputs = self._attention_inputs(graphs, kwargs)
        for index in range(len(self._layers)):
            graphs.pieces[index].replay()
            self._attend(graphs, index, cache, inputs)
        graphs.pieces[-1].replay()
        # A copy: the next replay writes the same memory, and generate may still hold these.
        return graphs.logits.clone()

    def _checked_pass(self, original: Callable[..., Any], kwargs: dict[str, Any], shape: tuple[int, int],
                      past: int) -> Any:
        """The first pass of ``shape``: through the graphs, then plainly over the same cache, and the two compared bit for
        bit, logits and what each wrote into the cache. Returns the plain pass's own output either way; a difference
        turns graphs off for the copy under ``auto`` and is only logged under ``on``."""
        torch = self._torch
        cache = kwargs["past_key_values"]
        graphs = self._sets.get(shape) or self._capture(shape)
        device = graphs.tokens.device
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        logits = self._pass(kwargs, shape)
        torch.cuda.synchronize(device)
        graph_ms = (time.perf_counter() - started) * 1000.0
        written = [(layer.keys[..., past:, :].clone(), layer.values[..., past:, :].clone()) for layer in cache.layers]
        cache.crop(past)
        started = time.perf_counter()
        output = original(**kwargs)
        torch.cuda.synchronize(device)
        plain_ms = (time.perf_counter() - started) * 1000.0
        same = bool(torch.equal(output.logits, logits)) and all(
            bool(torch.equal(layer.keys[..., past:, :], keys)) and bool(torch.equal(layer.values[..., past:, :], values))
            for layer, (keys, values) in zip(cache.layers, written, strict=True))
        self._checked.add(shape)
        steps, _ = shape
        if same:
            _LOG.info("VLM %s: decode graphs for %d token(s) a pass, %d piece(s) captured in %.0f ms; the first pass "
                      "wrote what the plain pass writes, bit for bit (plain %.1f ms, graphs %.1f ms)", self._model_id,
                      steps, len(graphs.pieces), graphs.capture_ms, plain_ms, graph_ms)
        elif self.mode == "on":
            _LOG.warning("VLM %s: decode graphs for %d token(s) a pass wrote other bits than the plain pass; "
                         "decode_graphs=on keeps them", self._model_id, steps)
        else:
            self._fail(f"the graphs for {steps} token(s) a pass wrote other bits than the plain pass")
        return output

    def _fail(self, cause: str) -> None:
        """Graphs off for this copy: what they captured goes, and every pass runs plainly until a reload."""
        self.failure = cause
        self._sets.clear()
        _LOG.warning("VLM %s: decode graphs off (%.300s); decoding plainly until the model is loaded again",
                     self._model_id, cause)


def _hand_back_memory(torch: Any) -> None:
    """Collect what a dropped model left behind and empty torch's CUDA cache, so the card itself has
    the memory back: the caching allocator would otherwise keep it reserved for this process, where
    the cuRobo sidecar or the next checkpoint cannot use it. Best effort: a failure is logged."""
    import gc  # noqa: PLC0415

    gc.collect()
    empty_cache = getattr(getattr(torch, "cuda", None), "empty_cache", None)
    if not callable(empty_cache):
        return
    try:
        empty_cache()
    except Exception as exc:  # noqa: BLE001 - emptying a cache must not fail the release it follows
        _LOG.warning("emptying the CUDA cache after an unload failed: %s: %s", type(exc).__name__, exc)


class Qwen3VLGrounder:
    """Ground a free-form prompt to boxes with a Qwen3-VL-Instruct checkpoint.

    Implements the ``detect_all`` detector contract, so ``TwoStageBackend`` takes it wherever a
    phrase detector goes.
    """

    def __init__(
        self,
        *,
        model_id: str,
        model_path: str | None = None,
        local: bool = False,
        device: str = "cuda",
        max_new_tokens: int = GROUNDING_MAX_NEW_TOKENS,
        preload: bool = False,
        coordinate_space: CoordinateSpace = CoordinateSpace.GRID_1000,
        prompt_lookup_tokens: int = 0,
        text_prompt_lookup_tokens: int = 0,
        stop_at_answer_end: bool = True,
        decode_graphs: str = "auto",
    ) -> None:
        #: What this checkpoint's box numbers mean, measured for Qwen3-VL-4B-Instruct. A parameter
        #: because a future checkpoint may differ and the failure is silent: grid values look like
        #: plausible pixels on a large frame. See :class:`CoordinateSpace`.
        self.coordinate_space = coordinate_space
        self.model_id = model_id
        self._source = model_path or model_id
        self._local = local
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None
        #: How the answers are written (:meth:`configure`), set below. Their own lock, not the generate
        #: lock: a build that sets them must not wait out a grounding in flight (15 s on the cell).
        self._settings_lock = threading.Lock()
        self._prompt_lookup_tokens = 0
        self._text_prompt_lookup_tokens = 0
        self._stop_at_answer_end = True
        self._decode_graphs = "auto"
        #: Why the answer lookup failed on this loaded copy, if it did: answers go on without it until a reload.
        self._lookup_failure: str | None = None
        #: The loaded copy's decode graphs (:class:`_DecodeGraphs`), made at its first answer with them on; and the
        #: last decision on whether it can take them, ``(model, mode, why not)``, logged once each. Both go at an unload.
        self._graphs: _DecodeGraphs | None = None
        self._graphs_decided: tuple[Any, str, str] | None = None
        #: Taken around the load: two callers that both need the weights (a pick's look and a
        #: command being read, or two requests) load them once.
        self._load_lock = threading.Lock()
        #: One inference at a time on the one model. Grounding a frame and answering a command take
        #: turns, and the span covers tokenising and decoding too: a fast tokenizer shared between
        #: threads raises "Already borrowed".
        self._generate_lock = threading.Lock()
        self._loading = False
        self._load_error: BaseException | None = None
        #: Load attempts that have ended, well or not. A caller that waited on the load lock compares
        #: it with what it read before waiting, so it knows the load it waited for has been tried.
        self._loads_ended = 0
        self._load_s: float | None = None
        self._last_answer_ms: float | None = None
        self.configure(prompt_lookup_tokens=prompt_lookup_tokens, text_prompt_lookup_tokens=text_prompt_lookup_tokens,
                       stop_at_answer_end=stop_at_answer_end, decode_graphs=decode_graphs)
        if preload:
            # Predictable latency and VRAM held from cell build, against nothing paid until the
            # first complex prompt arrives. Config chooses; both suit different cells.
            self._ensure_loaded()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def loading(self) -> bool:
        """True while a load is in progress on some thread."""
        return self._loading

    @property
    def load_error(self) -> BaseException | None:
        """What the last load raised, until a load succeeds; ``None`` otherwise."""
        return self._load_error

    @property
    def load_s(self) -> float | None:
        """How long the successful load took, in seconds; ``None`` before one."""
        return self._load_s

    @property
    def last_answer_ms(self) -> float | None:
        """How long the latest :meth:`answer_text` took, in milliseconds; ``None`` before one."""
        return self._last_answer_ms

    @property
    def prompt_lookup_tokens(self) -> int:
        """How many tokens the answer lookup proposes per pass of a grounding answer; 0 when it is off."""
        return self._prompt_lookup_tokens

    @property
    def text_prompt_lookup_tokens(self) -> int:
        """How many tokens the answer lookup proposes per pass of a text answer; 0 when it is off."""
        return self._text_prompt_lookup_tokens

    @property
    def stop_at_answer_end(self) -> bool:
        """Whether an answer ends where its JSON closes: a grounding answer's array, a text answer's object."""
        return self._stop_at_answer_end

    @property
    def decode_graphs(self) -> str:
        """``auto``, ``on`` or ``off``: whether the decode passes run as CUDA graphs (:class:`_DecodeGraphs`)."""
        return self._decode_graphs

    @property
    def lookup_failure(self) -> str | None:
        """Why the answer lookup failed on the loaded copy, which then answers without it; ``None`` otherwise."""
        return self._lookup_failure

    @property
    def graphs_failure(self) -> str | None:
        """Why the decode graphs stopped on the loaded copy, which then decodes plainly; ``None`` otherwise."""
        graphs = self._graphs
        return graphs.failure if graphs is not None else None

    def configure(
        self, *, prompt_lookup_tokens: int | None = None, text_prompt_lookup_tokens: int | None = None,
        stop_at_answer_end: bool | None = None, decode_graphs: str | None = None,
    ) -> None:
        """Set how this copy writes its answers; ``None`` keeps a setting. Loads and reloads nothing.

        The settings are the ``vlm`` block's (``prompt_lookup_tokens``, ``text_prompt_lookup_tokens``,
        ``stop_at_answer_end``, ``decode_graphs``), not the weights': a build hands them to the process's one copy,
        whatever a command or an earlier build left there, and the next inference uses them; one in flight keeps what
        it started with.
        """
        lookups = {"prompt_lookup_tokens": prompt_lookup_tokens, "text_prompt_lookup_tokens": text_prompt_lookup_tokens}
        for name, value in lookups.items():
            if value is not None and int(value) < 0:
                raise ValueError(f"{name} counts proposed tokens, 0 for none, not {value}")
        if decode_graphs is not None and decode_graphs not in DECODE_GRAPHS:
            raise ValueError(f"decode_graphs is one of {', '.join(DECODE_GRAPHS)}, not {decode_graphs!r}")
        with self._settings_lock:
            before = (self._prompt_lookup_tokens, self._text_prompt_lookup_tokens, self._stop_at_answer_end,
                      self._decode_graphs)
            if prompt_lookup_tokens is not None:
                self._prompt_lookup_tokens = int(prompt_lookup_tokens)
            if text_prompt_lookup_tokens is not None:
                self._text_prompt_lookup_tokens = int(text_prompt_lookup_tokens)
            if stop_at_answer_end is not None:
                self._stop_at_answer_end = bool(stop_at_answer_end)
            if decode_graphs is not None:
                self._decode_graphs = str(decode_graphs)
            after = (self._prompt_lookup_tokens, self._text_prompt_lookup_tokens, self._stop_at_answer_end,
                     self._decode_graphs)
        if after != before:
            _LOG.info("VLM %s answers with prompt_lookup_tokens=%d, text_prompt_lookup_tokens=%d, "
                      "stop_at_answer_end=%s, decode_graphs=%s", self.model_id, *after)

    def ensure_loaded(self) -> bool:
        """Load the weights unless they are loaded. ``True`` when this call loaded them.

        Raises what the load raised, unchanged, and records it in :attr:`load_error`; the next call
        tries again, except one that was already waiting for that load (see :meth:`_ensure_loaded`).
        Deciding what a failed load means is the caller's: ``GuardedVlmBackend`` for a pick, the
        holder's text asker for a command.
        """
        return self._ensure_loaded()

    def _ensure_loaded(self) -> bool:
        """Load weights on first use, once however many threads ask at the same time.

        Double-checked: the unlocked read is the fast path of every inference after the first, and
        the second read under the lock is what makes a caller that waited for another's load find
        the weights there instead of loading them again. A caller that waited for a load that
        FAILED takes that failure as its own answer rather than trying again on its heels: a second
        multi-GB attempt right after an out-of-memory or a missing file only repeats it, and a
        command that arrived during a failing Laden must not become the retry. The next caller that
        was not waiting tries again (a pick's next look, Laden). Raises the underlying error
        unchanged; ``GuardedVlmBackend`` decides what it means.
        """
        if self._model is not None:
            return False
        ended = self._loads_ended
        with self._load_lock:
            if self._model is not None:
                return False
            failure = self._load_error
            if failure is not None and self._loads_ended != ended:
                # Re-raised as it is, from every caller that waited for it, as a Future's result is.
                raise failure
            self._loading = True
            started = time.perf_counter()
            try:
                processor, model, torch = self._load()
            except BaseException as exc:
                self._load_error = exc
                _LOG.error("VLM %s failed to load: %s: %s", self.model_id, type(exc).__name__, exc)
                raise
            finally:
                self._loading = False
                self._loads_ended += 1
            self._processor = processor
            self._torch = torch
            self._load_error = None
            self._load_s = time.perf_counter() - started
            # Published last: `loaded` reads True only once the processor and torch are in place, so
            # the unlocked fast path above never meets a half-loaded model.
            self._model = model
            # The load cost is what `preload` trades against first-prompt latency.
            _LOG.info("VLM %s ready in %.1f s", self.model_id, self._load_s)
            with self._settings_lock:
                mode = self._decode_graphs
            if mode != "off":
                # Decided now, off the card this copy sits on, and logged; the graphs are captured at its first answer.
                self._graphs_decision(model, torch, mode)
            return True

    def _load(self) -> tuple[Any, Any, Any]:
        """The heavy part: ``(processor, model on its device, torch)``. Called under the load lock only."""
        # Imported here, not at module scope: this module must import with no torch and no GPU.
        import torch  # noqa: PLC0415
        from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: PLC0415

        _LOG.info("loading VLM %s (device=%s)", self._source, self._device)
        processor = AutoProcessor.from_pretrained(self._source, local_files_only=self._local)
        # Not `device_map=`: that routes through `accelerate`, which is absent from the validated
        # Isaac environment and fails at load with an error naming accelerate rather than the real
        # situation. `.to(device)` needs no extra dependency and is the right call for a single GPU;
        # device_map earns its keep only for multi-GPU sharding or CPU offload.
        # `model` is typed Any because transformers annotates `.to()` as taking a PreTrainedModel
        # rather than a device string, which mypy then rejects.
        model: Any = AutoModelForImageTextToText.from_pretrained(
            self._source,
            local_files_only=self._local,
            dtype="auto",
        )
        model = model.to(self._device)
        model.eval()
        return processor, model, torch

    def unload(self) -> bool:
        """Drop the loaded weights, so the card gets their memory back. ``True`` when there were some.

        Waits for a load or an inference in flight, so nothing loses its model halfway; an inference
        that starts afterwards loads the weights again (one that may not load refuses instead). The
        holder unloads a copy when it retires it: a rebuild that names other weights, or the
        console's release of a copy nobody asked for (:mod:`src.models.vlm.holder`).
        """
        with self._load_lock, self._generate_lock:
            model, torch = self._model, self._torch
            if model is None:
                return False
            # `loaded` reads False from here on, and an inference that takes the generate lock next
            # finds no model rather than half of one.
            self._model = None
            self._processor = None
            self._torch = None
            self._load_s = None
            self._last_answer_ms = None
            # A failed lookup belongs to the loaded copy: a reload tries it again.
            self._lookup_failure = None
            # So do the decode graphs: their memory goes with the weights, and a reload decides and captures anew.
            graphs, self._graphs = self._graphs, None
            with self._settings_lock:
                self._graphs_decided = None
        del model, graphs
        _hand_back_memory(torch)
        _LOG.info("VLM %s unloaded", self.model_id)
        return True

    def _in_hand(self) -> tuple[Any, Any, Any]:
        """``(processor, model, torch)``, read under the generate lock; refused when the weights are gone."""
        model = self._model
        if model is None:
            raise RuntimeError(
                f"the VLM {self.model_id!r} is not loaded: its weights were not there to answer with "
                f"(never loaded, or released since)"
            )
        return self._processor, model, self._torch

    def _generate(self, image_rgb: Any, prompt: str) -> tuple[str, int]:
        """``(answer, tokens written)`` for one grounding question, written as :meth:`configure` set: asked with today's
        instruction for one description and with the class-list instruction for several (:func:`grounding_instruction`);
        the answer's end, the lookup and the decode graphs are the same for both."""
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_rgb},
                {"type": "text", "text": grounding_instruction(prompt)},
            ],
        }]
        with self._generate_lock:
            with self._settings_lock:
                lookup_tokens, stop_at_end = self._prompt_lookup_tokens, self._stop_at_answer_end
                graphs_mode = self._decode_graphs
            processor, model, torch = self._in_hand()
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            ).to(model.device)
            prompt_length = int(inputs["input_ids"].shape[1])
            stops = [_EndOfAnswer(processor, torch, prompt_length)] if stop_at_end else []

            with torch.inference_mode():
                generated = self._greedy(
                    processor, model, torch, inputs, max_new_tokens=self._max_new_tokens,
                    lookup_tokens=lookup_tokens, stops=stops, graphs_mode=graphs_mode,
                )
            # Strip the prompt tokens: decoding the whole sequence would feed the instruction, which
            # contains a bbox_2d example, back into the parser and yield a fabricated detection.
            trimmed = generated[:, prompt_length:]
            return str(processor.batch_decode(trimmed, skip_special_tokens=True)[0]), int(trimmed.shape[1])

    def _greedy(
        self, processor: Any, model: Any, torch: Any, inputs: Any, *, max_new_tokens: int, lookup_tokens: int,
        stops: list[Any] | None = None, graphs_mode: str = "off",
    ) -> Any:
        """``model.generate``, greedy, with the answer lookup when ``lookup_tokens`` is above 0 and its decode passes
        as CUDA graphs where ``graphs_mode`` and the loaded copy take them (:meth:`_graphed`). Under the generate lock.

        A lookup that fails is logged, switched off for the loaded copy and the question asked again without
        it: a speed-up must never cost an answer. Running out of memory skips it for this question only. The graphs
        fall back pass by pass themselves (:class:`_DecodeGraphs`), so a failure of theirs never reaches the lookup.
        """
        options: dict[str, Any] = {"max_new_tokens": max_new_tokens, "do_sample": False}
        if stops:
            from transformers import StoppingCriteriaList  # noqa: PLC0415 - loaded with the model already

            options["stopping_criteria"] = StoppingCriteriaList(stops)
        tokens = int(lookup_tokens)
        with self._graphed(model, torch, graphs_mode):
            if tokens <= 0 or self._lookup_failure is not None:
                return model.generate(**inputs, **options)
            try:
                return self._generate_with_lookup(processor, model, torch, inputs, options, tokens)
            except Exception as exc:  # noqa: BLE001 - any failure of the speed-up falls back to the plain answer
                cause = f"{type(exc).__name__}: {exc}"
                if "OutOfMemory" not in type(exc).__name__:
                    self._lookup_failure = cause
                _LOG.warning("VLM %s: the answer lookup failed (%.300s); answering without it%s", self.model_id,
                             cause, "" if self._lookup_failure is None else " until the model is loaded again")
                return model.generate(**inputs, **options)

    @contextlib.contextmanager
    def _graphed(self, model: Any, torch: Any, mode: str) -> Iterator[None]:
        """The loaded copy's decode graphs installed as ``model``'s ``forward`` for the ``generate`` inside, where
        ``mode`` and the copy take them (:meth:`_graphs_for`); nothing changes otherwise. Under the generate lock: an
        attribute of the instance, so the class's own ``forward`` comes back afterwards and no other call sees it."""
        graphs = self._graphs_for(model, torch, mode)
        if graphs is None:
            yield
            return
        graphs.mode = mode
        installed = model.__dict__.get("forward")
        model.forward = graphs.forward_for(model.forward)
        try:
            yield
        finally:
            if installed is None:
                del model.forward
            else:
                model.forward = installed

    def _graphs_for(self, model: Any, torch: Any, mode: str) -> _DecodeGraphs | None:
        """The loaded copy's decode graphs where ``mode`` and its card take them, else ``None``: ``off``, a copy that
        cannot take them (:meth:`_graphs_decision`), or one whose graphs failed since its load. Under the generate lock.
        """
        if mode not in ("auto", "on") or self._graphs_decision(model, torch, mode):
            return None
        graphs = self._graphs
        if graphs is None or graphs.model is not model:
            try:
                graphs = _DecodeGraphs(model, torch, model_id=self.model_id)
            except Exception as exc:  # noqa: BLE001 - a transformers that moved its parts decodes plainly
                with self._settings_lock:
                    self._graphs_decided = (model, mode, f"{type(exc).__name__}: {exc}")
                _LOG.warning("VLM %s decodes plainly (decode_graphs=%s): %s: %s", self.model_id, mode,
                             type(exc).__name__, exc)
                return None
            self._graphs = graphs
        return None if graphs.failure is not None else graphs

    def _graphs_decision(self, model: Any, torch: Any, mode: str) -> str:
        """Why the loaded copy decodes plainly under ``mode``, ``""`` where it decodes with graphs: read off its card
        (:func:`_graphs_unsupported`) once per loaded copy and mode, and logged then."""
        with self._settings_lock:
            decided = self._graphs_decided
            if decided is not None and decided[0] is model and decided[1] == mode:
                return decided[2]
            try:
                reason = _graphs_unsupported(model, torch, auto=mode == "auto")
                card = "" if reason else "{}, compute capability {}.{}".format(
                    torch.cuda.get_device_name(model.device), *torch.cuda.get_device_capability(model.device))
            except Exception as exc:  # noqa: BLE001 - a card that cannot be read decodes plainly
                reason, card = f"the card could not be read: {type(exc).__name__}: {exc}", ""
            self._graphs_decided = (model, mode, reason)
        if reason:
            _LOG.info("VLM %s decodes plainly (decode_graphs=%s): %s", self.model_id, mode, reason)
        else:
            _LOG.info("VLM %s decodes with CUDA graphs (decode_graphs=%s, %s)", self.model_id, mode, card)
        return reason

    def _generate_with_lookup(
        self, processor: Any, model: Any, torch: Any, inputs: Any, options: dict[str, Any], tokens: int,
    ) -> Any:
        """``model.generate`` as transformers' assisted decoding, with :class:`_AnswerLookup` as its candidates.

        ``prompt_lookup_num_tokens`` selects assisted decoding; its candidate generator is this copy's, handed
        over through the model's ``_get_candidate_generator`` for this one call (an attribute of the instance, so
        the class's own comes back afterwards). Under the generate lock, so no other call sees it.
        """
        tokenizer = getattr(processor, "tokenizer", processor)
        config = getattr(model, "config", None)
        vision = [getattr(config, name, None) for name in (
            "image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id")]
        vision.append(_token_id(tokenizer, "<|vision_pad|>"))
        never = frozenset(int(token) for token in vision if isinstance(token, int))
        digits = frozenset(token for token in (_token_id(tokenizer, str(digit)) for digit in range(10))
                           if token is not None)
        ends = frozenset(_as_ids(getattr(getattr(model, "generation_config", None), "eos_token_id", None))
                         + _as_ids(getattr(tokenizer, "eos_token_id", None)))
        after = getattr(config, "vision_end_token_id", None)

        def candidates(**kwargs: Any) -> _AnswerLookup:
            return _AnswerLookup(
                torch, tokens=tokens, digits=digits, never=never, ends=ends,
                after=after if isinstance(after, int) else None,
                max_length=int(kwargs["generation_config"].max_length),
            )

        model._get_candidate_generator = candidates
        try:
            return model.generate(**inputs, **options, prompt_lookup_num_tokens=tokens)
        finally:
            del model._get_candidate_generator

    def answer_text(
        self, system: str, user: str, *, prefill: str = "{", max_new_tokens: int = 96,
        load: bool = True,
    ) -> str:
        """Ask the loaded model a text-only question: no image, greedy, at most ``max_new_tokens``.

        Returns ``prefill`` followed by what the model wrote. With a prefill, the answer is a
        continuation of an assistant turn that already starts with it (``continue_final_message``),
        which is how a JSON answer is made to open with ``{`` rather than with prose. Without one,
        the model starts a new turn. Loads the weights first if needed, and raises what the load
        raised; with ``load=False`` it never loads, and weights that are not there raise
        :class:`RuntimeError` (the holder's text asker decides loading itself).

        Every message is a list of typed text parts: the processor's tokenising path iterates the
        content looking for images, and a bare string is not iterable that way.
        """
        if max_new_tokens < 1:
            raise ValueError(
                f"an answer needs room for at least one token, not max_new_tokens={max_new_tokens}"
            )
        if load:
            self._ensure_loaded()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [{"type": "text", "text": user}]},
        ]
        if prefill:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": prefill}]})
        started = time.perf_counter()
        with self._generate_lock:
            with self._settings_lock:
                lookup_tokens, stop_at_end = self._text_prompt_lookup_tokens, self._stop_at_answer_end
                graphs_mode = self._decode_graphs
            processor, model, torch = self._in_hand()
            inputs = processor.apply_chat_template(
                messages, tokenize=True,
                add_generation_prompt=not prefill, continue_final_message=bool(prefill),
                return_dict=True, return_tensors="pt",
            ).to(model.device)
            prompt_length = int(inputs["input_ids"].shape[1])
            # An answer its prefill opens as a JSON object (the command reader's) ends where the object closes, one
            # pass before the model's own end token; the prefill is the prompt's, so the check reads it in front.
            stops = [_EndOfAnswer(processor, torch, prompt_length, bracket="}", closed=answer_object_closed,
                                  prefix=prefill)] if stop_at_end and prefill.lstrip().startswith("{") else []
            with torch.inference_mode():
                # The text answer has its own lookup switch: the instruction's own keys come back in the
                # answer, which is what it proposes.
                generated = self._greedy(processor, model, torch, inputs, max_new_tokens=max_new_tokens,
                                         lookup_tokens=lookup_tokens, stops=stops, graphs_mode=graphs_mode)
            # Stripped as in `_generate`: the instruction carries example answers, and decoding the
            # prompt back would hand one of them over as the model's own.
            trimmed = generated[:, prompt_length:]
            text = str(processor.batch_decode(trimmed, skip_special_tokens=True)[0])
            # Under the lock, so an unload that follows clears it rather than being followed by it.
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self._last_answer_ms = elapsed_ms
        _LOG.info("VLM %s answered %d character(s) of text in %.0f ms",
                  self.model_id, len(text), elapsed_ms)
        return prefill + text

    def detect_all(self, image_bgr: Any, prompt: str) -> list[Detection]:
        """Ground ``prompt``; ``[]`` when nothing matches. Same contract as the phrase detectors.

        A class list (:func:`class_list_prompt`) is grounded in this one call, each box under the description the model
        copied for it; a box it named by none, and every box of one object it boxed under two descriptions, comes back
        ``ambiguous`` (``parsing.AMBIGUOUS_LABEL``): an obstacle, never a target."""
        self._ensure_loaded()
        import numpy as np  # noqa: PLC0415

        array = np.asarray(image_bgr)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"expected an HxWx3 BGR image, got shape {array.shape}")
        height, width = int(array.shape[0]), int(array.shape[1])
        # The processor expects RGB; every caller here speaks BGR. `ascontiguousarray` is required:
        # a bare `[..., ::-1]` is a negative-stride view, and torch refuses those with "At least one
        # stride in the given numpy array is negative", raised deep inside the image processor and
        # several frames away from anything naming this file.
        image_rgb = np.ascontiguousarray(array[..., ::-1])

        # Timed around generate and parse together, because that span is what a caller waits for.
        started = time.perf_counter()
        answer, tokens = self._generate(image_rgb, prompt)
        detections = parse_grounding_response(
            answer, image_width=width, image_height=height, fallback_label=prompt,
            space=self.coordinate_space, classes=classes_of(prompt),
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if not detections:
            # "The model answered and nothing survived parsing" and "the model said the object is
            # absent" look identical downstream, and only this line distinguishes them.
            _LOG.info(
                "VLM grounded nothing for %r in %.0f ms, %d token(s) (raw answer: %.200s)",
                prompt, elapsed_ms, tokens, answer,
            )
        else:
            # The other half of the same event, so exactly one line per inference either way. The
            # token count is what the time is made of: one pass of the model each, unless looked up.
            # Every box with what the model called it: the only record of what it saw, since the camera
            # source maps a label onto the task's object or keeps it as an obstacle (2026-10-08).
            _LOG.info(
                "VLM grounded %d box(es) for %r in %.0f ms, %d token(s): %s", len(detections), prompt,
                elapsed_ms, tokens, "; ".join(
                    f"{detection.label!r} at {[round(float(v)) for v in detection.box]}" for detection in detections),
            )
        return detections

    #: What :meth:`name_colour` asks beside its crop: a colour word, never a yes or a no, which an instruct model gives
    #: too easily (the owner, 2026-10-08).
    COLOUR_QUESTION = "What colour is the object in the middle of this picture? Answer with one colour word."
    #: How far the crop reaches past the box, a share of the box's size on each side; its long side, pixels (about 50
    #: image tokens); and the answer's length, tokens.
    COLOUR_CROP_MARGIN = 0.075
    COLOUR_CROP_PX = 224
    COLOUR_MAX_NEW_TOKENS = 3

    def name_colour(self, image_bgr: Any, box: Any) -> str:
        """One colour word for the object in ``box`` (x0, y0, x1, y1, pixels) of ``image_bgr``: what a camera source
        asks where a part's pixels leave its colour unsure (``src.robot.perception.colour_check``).

        The model sees the box grown by 15 %, half of it on each side and cut at the image's edge, scaled so its long
        side is :attr:`COLOUR_CROP_PX` pixels, and is asked :attr:`COLOUR_QUESTION`; greedy, at most
        :attr:`COLOUR_MAX_NEW_TOKENS` new tokens, under the generate lock as every inference here. Loads the weights
        first if needed and raises what the load raised. Returns the model's text, stripped; the caller reads it
        through its colour words.
        """
        self._ensure_loaded()
        import cv2  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        array = np.asarray(image_bgr)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"expected an HxWx3 BGR image, got shape {array.shape}")
        x0, y0, x1, y1 = (float(value) for value in box)
        grow_x, grow_y = (x1 - x0) * self.COLOUR_CROP_MARGIN, (y1 - y0) * self.COLOUR_CROP_MARGIN
        height, width = int(array.shape[0]), int(array.shape[1])
        left, right = max(0, int(np.floor(x0 - grow_x))), min(width, int(np.ceil(x1 + grow_x)))
        top, bottom = max(0, int(np.floor(y0 - grow_y))), min(height, int(np.ceil(y1 + grow_y)))
        if right <= left or bottom <= top:
            raise ValueError(f"the box {list(box)} holds no pixel of a {width}x{height} image")
        crop = np.ascontiguousarray(array[top:bottom, left:right, ::-1])
        scale = self.COLOUR_CROP_PX / max(crop.shape[0], crop.shape[1])
        size = (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale)))
        crop = np.ascontiguousarray(cv2.resize(crop, size, interpolation=cv2.INTER_AREA if scale < 1.0
                                               else cv2.INTER_LINEAR))
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": crop},
                {"type": "text", "text": self.COLOUR_QUESTION},
            ],
        }]
        started = time.perf_counter()
        with self._generate_lock:
            with self._settings_lock:
                graphs_mode = self._decode_graphs
            processor, model, torch = self._in_hand()
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
            ).to(model.device)
            with torch.inference_mode():
                # Greedy and short, no lookup; its passes go through the decode graphs where the copy has them.
                generated = self._greedy(processor, model, torch, inputs, max_new_tokens=self.COLOUR_MAX_NEW_TOKENS,
                                         lookup_tokens=0, graphs_mode=graphs_mode)
            # Stripped of the question, as every answer here is.
            trimmed = generated[:, inputs["input_ids"].shape[1]:]
            answer = str(processor.batch_decode(trimmed, skip_special_tokens=True)[0]).strip()
        _LOG.info("VLM %s named the colour of the object in %s: %r, in %.0f ms", self.model_id,
                  [round(float(v)) for v in box], answer, (time.perf_counter() - started) * 1000.0)
        return answer
