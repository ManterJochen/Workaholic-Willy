# Speech to a pick prompt (`src/models/speech`)

A recording becomes text a person confirms before anything uses it as a prompt. It comes from an
upload, or from the cell PC's microphone while a talk switch is held. One `SpeechEngine` (Whisper
large-v3-turbo) transcribes it, after Silero has checked that anything was said at all, because Whisper
answers silence with a word. Nothing here moves the arm; stopping stays with the emergency stop.

```python
from willy import (Confirmation, PushToTalkSource, TalkButton, TerminalConfirmer,
                   load_speech_section, shared_speech)

config = load_speech_section()                     # models.stt alone: no camera, no robot
held = shared_speech().for_config(config=config)   # the process's engine and gate; loads nothing yet
switch = TalkButton.from_parts()                   # press() and release() from whatever reads your switch

with PushToTalkSource.from_config(config=config, switch=switch) as microphone:
    turn = microphone.record(timeout_s=10.0)       # waits for the press, keeps what is said until release
print(turn)
if turn.ok:
    proposal = held.propose(turn.samples, samplerate=turn.samplerate)   # loads each model on first use
    confirmation = Confirmation.from_proposal(proposal=proposal, confirmer=TerminalConfirmer.from_parts())
    print(confirmation)                            # confirmation.confirmed is the prompt, or None
```

The whole chain to a pick is [12_speak_a_command.py](../../../examples/real_robot/12_speak_a_command.py).
The operator console serves the same engine: `POST /v1/voice/transcribe` proposes an uploaded recording,
`POST /v1/voice/talk` presses or releases the console's talk switch, and `POST /v1/voice/listen` records
one push-to-talk turn at the cell PC and proposes it ([api/README.md](../../../api/README.md)).

## The nouns

| Noun | Built by | Verb | Returns |
| --- | --- | --- | --- |
| `HeldSpeech` | `shared_speech().for_config(config=...)` | `propose(samples, samplerate=...)` | a `Proposal`: text, or the reason there is none |
| `PushToTalkSource` | `from_config(config=..., switch=...)` | `record(timeout_s=...)` | a `TalkRecording` of one turn |
| `TalkButton` | `from_parts()`, `shared_talk_button()` | `press()`, `release()` | the talk switch in software |
| `Listener` | `from_config(config=..., engine=held.engine)` | `listen(timeout_s=...)` | an `Utterance` cut by voice activity |
| `Confirmation` | `from_proposal(...)`, `from_utterance(...)` | read `confirmed` | the text a person let through, or `None` |

Every report prints as itself. `Listener` builds a voice detector of its own, because a detector carries
state from one window to the next; pass it `source=` a `PushToTalkSource` and the release closes the
utterance at the latest.

## What it refuses

| Refusal | When | What to do |
| --- | --- | --- |
| `SpeechStackUnavailable` | a package this host cannot import, a DLL Windows refused, or no PortAudio | install from `requirements.txt`; the console answers 501 naming the file |
| `SpeechModelMissing` | the Whisper or Silero weights are not on this machine | `python scripts/model_weights/fetch.py whisper-turbo silero-vad` |
| `RecordingTooLong` | a recording past Whisper's 30 s window, before either model runs | send one command per recording |
| `MicrophoneUnavailable` | no input device the stream can open | connect or select a microphone |
| an empty `Proposal` | Silero heard no utterance close, so Whisper is never asked | speak, or check the microphone level |
| a turn that is not `ok` | not pressed in time, nothing captured, or held past 30 s (`HELD_TOO_LONG`) | say it again: a cut-off command is a different command |
| `ConfigError` at load | `models.stt.task` other than `transcribe` | Whisper transcribes and never translates |

## What it decides

- **German or English, from one encoder pass.** `language: auto` scores only the `<|de|>` and `<|en|>`
  tokens and decodes with the winner; `german` or `english` forces one. A third language still comes
  out as one of the two. The report reads the language back from the decoder prompt it actually ran.
- **The text is never rewritten.** An upload and a microphone utterance return the same text for the
  same audio; nothing lower-cases or edits it.
- **Loaded once.** `shared_speech()` keeps one engine and one gate per `models.stt` section, because
  building Whisper per request would put a model load inside the 2 s budget from the end of speaking.
- **A turn is the time the switch is held.** Audio before the press is dropped. At the release the ring
  is drained before the turn ends, and a key that repeats while held is one press.
- **Only two answers let words through.** `Confirmation.confirmed` holds text for an explicit yes (y,
  yes, j, ja: the words as heard) or for a correction typed after `e`. Every other answer, a process with
  no terminal, and a proposal without words confirm nothing. A bare line of text is not a correction,
  because a typo would become the grounding phrase of a pick.
- **Heavy imports wait.** Importing this package loads neither torch, transformers, scipy nor
  sounddevice.

## What it costs

Measured on the workstation (RTX 5080, torch 2.7.1+cu128, `torch_dtype: auto`, float16 on CUDA), with
the weights at the catalogue's pinned commit and synthetic audio at 16 kHz: 3 s of silence, and 3 s and
8 s of a 440 Hz tone.

| What | Measured |
| --- | --- |
| `load()` of whisper-large-v3-turbo | 3.0 s with the weights in the OS file cache; a cold first read is 1.62 GB |
| Weights on the GPU | 1543.7 MB allocated by the load |
| Peak CUDA memory over the run | 1619.5 MB allocated, 1768.0 MB reserved |
| First transcription after the load | 388 ms (3 s of silence; it carries the CUDA warm-up) |
| Warm transcription, median of five | 60.1 ms (3 s silence), 61.1 ms (3 s tone), 61.6 ms (8 s tone) |
| Silero `load()` | 0.046 s, on the CPU |
| Silero per 512-sample window | 0.18 ms warm (0.94 ms while warming up) |

Whisper pads every recording to its 30 s window, so 8 s costs the same as 3 s, and a warm decode spends
3 % of the 2 s budget. Whisper answered 3 s of silence with `you` and the tone with `.`, both as `en`;
Silero scored both far below its 0.5 onset (peak 0.0089 on silence, 0.0024 on the tone). The first call
after a load is six times a warm one: a cell that wants its first command as quick as its tenth calls
`engine.load()` at start-up and transcribes one short clip. The checks that reproduce these numbers
stand down with a sentence naming the fetch when the weights are absent.

## Status

| Capability | Evidence |
| --- | --- |
| Whisper and Silero on synthetic audio, the costs above | measured in simulation (real weights, synthetic audio) |
| Push to talk, the console's voice routes, a physical microphone or switch, a human voice | never touched hardware |

Push to talk and the voice routes run against a fake `sounddevice` and a ring double. No recording of a
spoken command exists in this repository, so nothing here says how well anything is heard, German
included, and the 2 s budget has not been measured end to end. Every `Endpointing` default is a
placeholder until it is measured on cell audio. `Listener` has no caller yet: no console route and no
cell verb builds one. The console's own talk button records in the browser and uploads to
`/v1/voice/transcribe`.

Not built: faster-whisper on CTranslate2 (its wheel carries no kernels for the RTX 5080, and Windows
application control refused it on the workstation); a reader for a USB hand or foot switch, which waits
for a licence check of the HID library (a foot switch that types a key works in the browser); a wake
word, for which no licence-clean German model is known.

## Files

| File | Holds |
| --- | --- |
| [`engine.py`](engine.py) | the `SpeechEngine` seam and the three refusals |
| [`transcript.py`](transcript.py) | `Transcript`, `SpeechCheck` and `Proposal`; standard library only |
| [`whisper_transformers.py`](whisper_transformers.py) | `WhisperTransformersEngine`, Whisper large-v3-turbo on transformers |
| [`silero.py`](silero.py) | `SileroVoiceActivityDetector`, Silero VAD 6.2.1 as TorchScript on the CPU |
| [`gate.py`](gate.py) | `SpeechGate`: does this recording hold an utterance |
| [`holder.py`](holder.py) | `shared_speech()`, `SpeechHolder` and `HeldSpeech.propose()` |
| [`capture.py`](capture.py) | `MicrophoneSource`, `AudioBlock`, `MicrophoneUnavailable`, mono and rate conversion |
| [`endpointing.py`](endpointing.py) | `Endpointing` and the `UtteranceCutter` that closes an utterance |
| [`listener.py`](listener.py) | `Listener` and `listen()` |
| [`push_to_talk.py`](push_to_talk.py) | `TalkSwitch`, `TalkButton`, `PushToTalkSource` and `TalkRecording` |
| [`confirm.py`](confirm.py) | `Confirmation`, the `Confirmer` seam and `TerminalConfirmer` |

## Details

- Weights and licences: `python scripts/model_weights/fetch.py whisper-turbo silero-vad` writes both,
  pinned by hub commit and by sha256. Both are MIT; Whisper's training data is undisclosed, which
  `NOTICE` records as a known provenance gap.
- Guide: [models](../../../docs/guide/02-models.md); console routes: [api/README.md](../../../api/README.md)
- Tests: `tests/test_speech_engine.py`, `tests/test_speech_push_to_talk.py`, `tests/test_speech_confirm.py`
