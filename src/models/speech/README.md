# Speech: one engine, one voice detector, a microphone stream, `listen()`, push to talk and a confirmation

A recording uploaded to the console and an utterance cut from the cell PC's microphone go through one
`SpeechEngine` and come back as the same frozen `Transcript`: the text in the language it was spoken
in, which language that was, how long the recording was, and what the decode cost. Before Whisper is
asked what was said, Silero is asked whether anything was said at all, because Whisper answers silence
with a word. A transcript is a proposal. A human confirms it before anything acts, "Stopp" included, and
`Confirmation` is that answer as a report. At the cell PC a talk switch gates the microphone: a turn is
what is said while it is held.

## Contents

| File | Role |
| --- | --- |
| [`engine.py`](engine.py) | `SpeechEngine`, the seam every engine sits behind, and the three refusals: `SpeechStackUnavailable` (a package this host cannot import, or a DLL Windows refused), `SpeechModelMissing` (a model this box has not fetched), `RecordingTooLong` (past Whisper's 30 s window). |
| [`transcript.py`](transcript.py) | `Transcript`, `SpeechCheck` and `Proposal`, the frozen reports both paths return. stdlib only. |
| [`whisper_transformers.py`](whisper_transformers.py) | `WhisperTransformersEngine`: Whisper large-v3-turbo on transformers, the engine. |
| [`silero.py`](silero.py) | `SileroVoiceActivityDetector`: Silero VAD 6.2.1 as TorchScript on the CPU, on the torch this interpreter already has. |
| [`gate.py`](gate.py) | `SpeechGate.check()`: does this recording hold an utterance, by the microphone path's own rule. |
| [`holder.py`](holder.py) | `SpeechHolder` / `shared_speech()`, the process's one engine and gate, keyed on the `models.stt` section, and `HeldSpeech.propose()`, the upload path's verb. |
| [`capture.py`](capture.py) | `AudioSource`, `AudioBlock`, `MicrophoneSource` (a sounddevice `InputStream` copied into a ring), `MicrophoneUnavailable` (no input device the stream can open), `to_mono_float32` and `to_mono_at_rate`. |
| [`endpointing.py`](endpointing.py) | `VoiceActivityDetector`, the seam Silero fills, `Endpointing` and the `UtteranceCutter` state machine. |
| [`listener.py`](listener.py) | `Listener` and `listen()`, which returns an `Utterance` carrying the `Transcript`. |
| [`push_to_talk.py`](push_to_talk.py) | `TalkSwitch`, the seam a talk switch comes through; `TalkButton` and `shared_talk_button()`, the switch in software; `PushToTalkSource`, the microphone behind the switch, and `record()`, which returns a `TalkRecording` of one turn. |
| [`confirm.py`](confirm.py) | `Confirmation` (`from_proposal`, `from_utterance`), the `Confirmer` seam and `TerminalConfirmer`: a person says yes, types a correction, or nothing becomes a prompt. stdlib only at import. |

## Usage

```python
from src.config.loader import load_speech_section
from src.models.speech import Listener, shared_speech

config = load_speech_section()                  # models.stt alone: speech needs no camera and no robot
held = shared_speech().for_config(config=config)  # the process's engine and gate; loads nothing yet
proposal = held.propose(samples, samplerate=48000)  # loads each model on its first use, then decodes
print(proposal.render())

# The same engine drives the microphone. The listener builds a voice detector of its own, because a
# detector carries state from one window to the next and the gate's belongs to uploads.
listener = Listener.from_config(config=config, engine=held.engine)
with listener:                                  # opens the microphone
    heard = listener.listen(timeout_s=10.0)     # the first utterance that closes
print(heard.render())
```

Push to talk, and the confirmation before any words become a prompt:

```python
from src.models.speech import Confirmation, PushToTalkSource, TalkButton, TerminalConfirmer

switch = TalkButton.from_parts()               # pressed and released by whatever reads the switch; in the
                                               # console process, POST /v1/voice/talk presses shared_talk_button()
source = PushToTalkSource.from_config(config=config, switch=switch)
with source:                                    # opens the microphone; nothing is served before the press
    turn = source.record(timeout_s=10.0)        # everything said from the press to the release
print(turn.render())
if turn.ok:
    proposal = held.propose(turn.samples, samplerate=turn.samplerate)
    confirmation = Confirmation.from_proposal(proposal=proposal, confirmer=TerminalConfirmer.from_parts())
    print(confirmation.render())
    if confirmation.confirmed is not None:      # only an explicit yes or a typed correction
        prompt = confirmation.confirmed

# A Listener listens across a turn the same way; the release closes the utterance at the latest.
listener = Listener.from_config(config=config, engine=held.engine, source=source)
```

## What it decides

- **Transcribe, never translate.** `models.stt.task` accepts only `transcribe`, and a tree that says
  `translate` is refused at load with a sentence naming the key. `language: auto` picks German or
  English per recording; `german` or `english` forces that language token. `generate` is asked through
  `task` and `language`: transformers 5.5.4 deprecates `forced_decoder_ids` and ignores them once
  either is passed, and the engine clears a checkpoint's own forced tokens at load.
- **German or English, out of one encoder pass.** Under `auto` the encoder runs once and a single
  decoder step from the start token scores the language tokens; only `<|de|>` and `<|en|>` are compared,
  and the winner goes to `generate` with the same encoder output. Speech in a third language still comes
  out as one of the two, which is the decision, not a defect.
- **The language is read back from the decoder prompt.** With `return_dict_in_generate=True` a recording
  of up to 30 s comes back with the prompt at the front, so the report says the language token the
  decoder actually ran with, rather than the one that was asked for.
- **Silence is not transcribed.** Whisper answers a recording with no speech with a word, on this
  workstation as through the console. `SpeechGate` scores the recording with Silero first, through the
  same `UtteranceCutter` the microphone path cuts utterances with, and a recording in which no utterance
  closes comes back as an empty `Proposal` with the reason, with Whisper never asked.
- **One window.** A recording longer than 30 s is refused before either model runs. This engine decodes
  one Whisper window and does not stitch several together.
- **Loaded once.** `SpeechHolder` keeps one engine and one gate per `models.stt` section; an edited
  section builds a new pair. Building Whisper per request would put a whole model load inside the 2 s
  budget from the end of speaking.
- **One engine for both paths.** An upload and a `listen()` utterance return the same text for the same
  audio; nothing on either path lower-cases or rewrites it.
- **Every block scaled by its own format.** int16, int32 and float32 are each divided by their full scale
  and the channels are averaged. The callback only copies into the ring; a full ring keeps the newest
  audio and the next block reports the loss.
- **Bounded.** `listen()` returns `TIMED_OUT` at its bound (30 s unless chosen; `None` waits for the
  source). It drops audio captured before the call.
- Push to talk, and the release is the end. A turn is what the microphone catches while the talk
  switch is held. `PushToTalkSource` drops what was captured before the press, and at the release it
  serves what the ring still holds before it reports that it has ended; it never stops the microphone to
  end a turn, because `MicrophoneSource.stop()` drops the audio not yet read. A key that repeats while
  held is one press. `record()` bounds the wait for the press (NOT_PRESSED) and the turn itself at
  Whisper's 30 s window (HELD_TOO_LONG, not proposed: a command cut off at the limit is a different
  command). The console's `POST /v1/voice/listen` proposes a turn through `HeldSpeech.propose()`, so a
  turn and an upload go through one gate and one engine.
- A person confirms, and only two answers let words through. `Confirmation.confirmed` holds text for
  an explicit yes (y, yes, j, ja: the words as heard, SPOKEN) or a correction typed after e (TYPED), and
  None for every other answer (REFUSED), for a process with no terminal to ask (NO_PERSON) and for a
  proposal without words, which nobody is asked about (NOTHING_HEARD). A bare line of text is not a
  correction: a typo would otherwise become the grounding phrase of a pick.
- **Heavy imports are lazy.** Importing this package imports neither torch, transformers, scipy nor
  sounddevice. A missing package, a DLL the OS refuses, or a missing PortAudio is a
  `SpeechStackUnavailable`, which the console answers with 501 naming the requirements file that holds
  the package, or, for a refused DLL, the file Windows refused.

## What it runs, and what that cost

**Machinery, not accuracy.** No recording of a spoken command exists in this repository, so nothing here
says how well anything is heard. What was measured is what the machinery costs and what it does with
audio that carries no speech. RTX 5080, torch 2.7.1+cu128, `torch_dtype: auto` (float16 on CUDA),
weights from `assets/models/hf/speech/openai--whisper-large-v3-turbo` at the catalogue's pinned commit
`41f01f3`, synthetic audio at 16 kHz: 3 s of silence, 3 s and 8 s of a 440 Hz tone.

| What | Measured |
| --- | --- |
| `load()` of whisper-large-v3-turbo | **3.0 s** (weights already in the OS file cache; the first fetch of the day reads 1.62 GB from disk) |
| Weights on the GPU | **1543.7 MB** allocated by the load |
| Peak CUDA memory over the whole run | **1619.5 MB** allocated, 1768.0 MB reserved |
| First transcription after the load | **388 ms** (3 s of silence; it carries the CUDA warm-up) |
| Warm transcription, median of five | **60.1 ms** (3 s silence), **61.1 ms** (3 s tone), **61.6 ms** (8 s tone) |
| Silero `load()` | **0.046 s**, on the CPU |
| Silero per 512-sample window | **0.18 ms** warm (0.94 ms in the first stretch, which carries the warm-up) |

- **The decode does not get slower with a longer recording**, because Whisper pads every recording to its
  30 s window: 8 s of audio costs the same 61 ms as 3 s. The whole budget from the end of speaking is 2 s,
  and a warm decode spends 3 % of it.
- **Whisper answered 3 s of silence with `you`** and 3 s of a 440 Hz tone with `.`, both as `en`. That
  word would have landed in the prompt box. It is the whole reason `SpeechGate` runs first.
- **Silero refuses both**, far below the 0.5 onset: peak speech probability **0.0089** over 93 windows of
  silence and **0.0024** over the tone. Neither is speech, and the gate says so in 93 windows at 0.18 ms
  each.
- The first call after a load is six times a warm one. A cell that wants the first spoken command to be
  as quick as the tenth calls `engine.load()` at start-up and transcribes one short clip.

The measurements above need the real weights under this checkout's weights root; where they are absent,
the checks that reproduce them stand down with a sentence naming the fetch.

## Honesty

**Checked against fakes, plus the real models on this box.** The engine runs against fakes and against a
randomly initialised two-layer Whisper that exercises transformers' own `generate`; the microphone runs
against a fake `sounddevice` module; `listen()` runs against a scripted source, a loudness test double
and a recording engine. On top of that, the real whisper-large-v3-turbo weights and the real Silero
model run here, and what they measured is the table above.

**Not run here:** a physical microphone, any recording of a human voice, the 2 s budget end to end, and
therefore every accuracy question, German included. `Listener` has no caller: no console route and no
cell verb builds one, and there is no CLI twin. Every `Endpointing` default is a placeholder until it is
measured on recorded cell audio, and its trailing silence counts against that budget. Push to talk and
the console's `/v1/voice/listen` and `/v1/voice/talk` run against a ring double, a fake `sounddevice`
and a fake held speech; no physical microphone or switch has driven them, and no console screen calls
them yet (the console's talk button records in the browser and uploads).

**Not built:** faster-whisper on CTranslate2, which stays a later bake-off (CTranslate2 4.8.2 does not
load on this box: Smart App Control, event 3033, and its wheel carries no sm_120 kernels for the RTX
5080); a reader for a USB HID hand or foot switch at the cell PC, which waits for a licence check of
the HID library, while a foot switch that sends a key works in the browser; a wake word, for which no
licence-clean German model is known.

**Weights and licences.** `python scripts/model_weights/fetch.py whisper-turbo silero-vad` writes both,
each pinned: the checkpoint by hub commit, the Silero file by its wheel's sha256 and its own. Whisper's
weights are MIT and Silero's are MIT; Whisper's training data is undisclosed, which `NOTICE` records as
a known provenance gap rather than a proven violation.
