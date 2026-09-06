/**
 * Record the microphone and hand back a 16-bit WAV.
 *
 * ⛔ WHY NOT `MediaRecorder`, WHICH IS THE OBVIOUS ANSWER. Because no browser records WAV. Chrome and
 * Firefox give `audio/webm;codecs=opus`, Safari gives `audio/mp4`, and the backend decodes WAV with
 * the standard library and everything else only with an optional extra. Recording what the platform
 * chose would make the console's own microphone the one client that needs a `pip install` to work --
 * on a machine where nothing else does. So the console encodes here, in about eighty lines, and the
 * extra exists for OTHER clients rather than for us.
 *
 * ⚠ SECURE CONTEXT ONLY. `getUserMedia` is undefined over plain HTTP to anything but `localhost`, so
 * a console reached at `http://192.168.1.50:8000` from another machine has no microphone at all --
 * the button must say that rather than fail on click. `micAvailability()` is that check, and it is
 * the reason the component can render a disabled button with a reason instead of a broken one.
 */

/** Why the microphone cannot be used here, or `null` if it can. */
export function micAvailability(): string | null {
  if (typeof navigator === 'undefined' || !navigator.mediaDevices?.getUserMedia) {
    // ⚠ The two causes are worth separating because only one is fixable by the operator. Over plain
    // HTTP to a remote host the API is simply absent -- no permission prompt, no error, just
    // `undefined` -- and an operator staring at a dead button would reasonably blame the microphone.
    return typeof window !== 'undefined' && !window.isSecureContext
      ? 'The microphone needs HTTPS or localhost. This page was loaded over plain HTTP from another machine, so the browser does not offer it at all.'
      : 'This browser does not expose a microphone to the page.'
  }
  if (typeof AudioContext === 'undefined') return 'This browser has no Web Audio support.'
  return null
}

export interface Recording {
  /** 16-bit mono PCM in a WAV container -- what `POST /v1/voice/transcribe` decodes for free. */
  readonly wav: Blob
  readonly seconds: number
  readonly sampleRate: number
}

/** A recording in progress. `stop()` resolves with the audio; `cancel()` throws it away. */
export interface ActiveRecording {
  stop: () => Promise<Recording>
  cancel: () => void
}

/** How long a single press may record. A stuck button must not fill memory with a live tap. */
const MAX_SECONDS = 60

export async function startRecording(): Promise<ActiveRecording> {
  const unavailable = micAvailability()
  if (unavailable) throw new Error(unavailable)

  const stream = await navigator.mediaDevices.getUserMedia({
    // Browser-side cleanup, on by default in every engine that has it. Left on deliberately: a
    // console sits next to a robot cell, and a cell is a loud room.
    audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  })
  const context = new AudioContext()
  const source = context.createMediaStreamSource(stream)
  // ⚠ `ScriptProcessorNode` IS DEPRECATED AND IS STILL THE RIGHT CALL HERE. Its replacement,
  // `AudioWorklet`, needs a separate module file fetched at runtime -- which the console cannot rely
  // on, because it is served as a built bundle and an extra fetch is an extra thing to misconfigure.
  // Every engine still implements this node, the recording is seconds long, and the alternative is a
  // deployment failure mode in exchange for a deprecation warning.
  const processor = context.createScriptProcessor(4096, 1, 1)
  const chunks: Float32Array[] = []
  let total = 0
  let stopped = false

  processor.onaudioprocess = (event) => {
    if (stopped) return
    // Copied, not referenced: the node reuses its buffer, so keeping the view would leave every
    // chunk pointing at whatever the last frame happened to contain.
    const frame = new Float32Array(event.inputBuffer.getChannelData(0))
    chunks.push(frame)
    total += frame.length
    if (total >= MAX_SECONDS * context.sampleRate) stopped = true
  }

  source.connect(processor)
  // Connected to the destination because several engines never run a processor that goes nowhere.
  // Gain zero so the console does not play the room back at itself through the speakers.
  const mute = context.createGain()
  mute.gain.value = 0
  processor.connect(mute)
  mute.connect(context.destination)

  const teardown = () => {
    stopped = true
    processor.onaudioprocess = null
    try {
      source.disconnect()
      processor.disconnect()
      mute.disconnect()
    } catch {
      /* already torn down */
    }
    // ⛔ THE TRACKS MUST BE STOPPED EXPLICITLY. Closing the context is not enough: the browser keeps
    // showing the recording indicator and holds the microphone open until every track ends. The
    // console had exactly this shape of leak once already, with cameras, and nobody noticed for
    // months because nothing in the UI said the device was still held.
    stream.getTracks().forEach((track) => track.stop())
    void context.close()
  }

  return {
    async stop(): Promise<Recording> {
      const sampleRate = context.sampleRate
      teardown()
      const samples = new Float32Array(total)
      let at = 0
      for (const chunk of chunks) {
        samples.set(chunk, at)
        at += chunk.length
      }
      return { wav: encodeWav(samples, sampleRate), seconds: total / sampleRate, sampleRate }
    },
    cancel: teardown,
  }
}

/** Float samples in [-1, 1] to a 16-bit mono WAV. */
export function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const bytes = new ArrayBuffer(44 + samples.length * 2)
  const view = new DataView(bytes)
  const ascii = (at: number, text: string) => {
    for (let i = 0; i < text.length; i++) view.setUint8(at + i, text.charCodeAt(i))
  }
  ascii(0, 'RIFF')
  view.setUint32(4, 36 + samples.length * 2, true)
  ascii(8, 'WAVE')
  ascii(12, 'fmt ')
  view.setUint32(16, 16, true) // PCM header length
  view.setUint16(20, 1, true) // format 1 = PCM
  view.setUint16(22, 1, true) // mono
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * 2, true) // byte rate
  view.setUint16(32, 2, true) // block align
  view.setUint16(34, 16, true) // bits per sample
  ascii(36, 'data')
  view.setUint32(40, samples.length * 2, true)
  for (let i = 0; i < samples.length; i++) {
    // Clamped before scaling. A sample above 1.0 -- which gain control can produce -- would wrap to
    // a large negative through the 16-bit conversion, turning a loud word into a click.
    const clamped = Math.max(-1, Math.min(1, samples[i]))
    view.setInt16(44 + i * 2, Math.round(clamped * 32767), true)
  }
  return new Blob([bytes], { type: 'audio/wav' })
}
