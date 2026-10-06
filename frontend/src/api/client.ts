/**
 * The only place in the console that talks to the backend.
 *
 * Every type here comes from `schema.d.ts`, which is GENERATED from the backend's own OpenAPI
 * document (`npm run api:types`). Nothing in this file re-describes a payload by hand: a console that
 * carries its own idea of what a `PreflightCheck` looks like is a console that disagrees with the cell
 * the first time the cell changes, and it disagrees silently.
 *
 * The error path is deliberately as typed as the success path. The backend answers every failure with
 * one `ErrorOut` envelope -- `{code, message, detail}` -- so there is exactly one shape to handle, and
 * `ApiError` below carries all three to the screen rather than collapsing them into a string. `detail`
 * is the half an operator can act on (which key, which validator, which run holds the lock).
 */

import type { components, paths } from './schema'

type Schemas = components['schemas']

export type PreflightOut = Schemas['PreflightOut']
export type PreflightCheckOut = Schemas['PreflightCheckOut']
export type CellOut = Schemas['CellOut']
export type ConnectPreviewOut = Schemas['ConnectPreviewOut']
export type TelemetryOut = Schemas['TelemetryOut']
export type DiagnosticsOut = Schemas['DiagnosticsOut']
export type PerceptionStackOut = Schemas['PerceptionStackOut']
export type RoutePreviewOut = Schemas['RoutePreviewOut']
export type RunOut = Schemas['RunOut']
export type RollupOut = Schemas['RollupOut']
export type RecordOut = Schemas['RecordOut']
export type ConfigValueOut = Schemas['ConfigValueOut']
export type WritableOut = Schemas['WritableOut']
export type TranscriptOut = Schemas['TranscriptOut']
export type SpeechCheckOut = Schemas['SpeechCheckOut']
export type ProposalOut = Schemas['ProposalOut']
export type ErrorOut = Schemas['ErrorOut']

// ── the contract of commit 2 (build plan section 1): tasks, the stop, the jaws, poses, commands ──────────────
export type CodesOut = Schemas['CodesOut']
export type TaskIn = Schemas['TaskIn']
export type TaskOptionsIn = Schemas['TaskOptionsIn']
export type TaskOptionsOut = Schemas['TaskOptionsOut']
export type TaskPlanOut = Schemas['TaskPlanOut']
export type PlanPlaceOut = Schemas['PlanPlaceOut']
export type PosePlaceIn = Schemas['PosePlaceIn']
export type CameraPlaceIn = Schemas['CameraPlaceIn']
export type CommandProvenanceIn = Schemas['CommandProvenanceIn']
export type CellFactsOut = Schemas['CellFactsOut']
export type PushFactsOut = Schemas['PushFactsOut']
export type BrakeFactsOut = Schemas['BrakeFactsOut']
export type PayloadFactsOut = Schemas['PayloadFactsOut']
export type RouteFactsOut = Schemas['RouteFactsOut']
export type DetectorFactsOut = Schemas['DetectorFactsOut']
export type RigOut = Schemas['RigOut']
export type ReadinessOut = Schemas['ReadinessOut']
export type LightOut = Schemas['LightOut']
export type BlockerOut = Schemas['BlockerOut']
export type BrakeOut = Schemas['BrakeOut']
export type HandOut = Schemas['HandOut']
export type HaltStateOut = Schemas['HaltStateOut']
export type PlannerOut = Schemas['PlannerOut']
export type RecoveryOut = Schemas['RecoveryOut']
export type JawsOut = Schemas['JawsOut']
export type JawQuestionOut = Schemas['JawQuestionOut']
export type LiveFrameOut = Schemas['LiveFrameOut']
export type PosesOut = Schemas['PosesOut']
export type PoseOut = Schemas['PoseOut']
export type PayloadOut = Schemas['PayloadOut']
export type TeachIn = Schemas['TeachIn']
export type TeachStartOut = Schemas['TeachStartOut']
export type TeachStateOut = Schemas['TeachStateOut']
export type CommandIn = Schemas['CommandIn']
export type CommandOut = Schemas['CommandOut']
export type CommandPhraseOut = Schemas['CommandPhraseOut']
export type CommandStatusOut = Schemas['CommandStatusOut']
export type MotionStackOut = Schemas['MotionStackOut']
export type VendorReadinessOut = Schemas['VendorReadinessOut']
export type ReachabilityOut = Schemas['ReachabilityOut']

/** The four verdicts a preflight row can carry. Pinned by the backend, mirrored here by generation. */
export type Status = Schemas['PreflightCheckOut']['status']

/**
 * A failure that arrived as the backend's own envelope.
 *
 * Thrown rather than returned so a screen cannot forget to check: an operator console that renders a
 * stale value because nobody looked at a return code is worse than one that shows an error.
 */
export class ApiError extends Error {
  readonly code: string
  readonly detail: Record<string, unknown>
  readonly httpStatus: number

  constructor(httpStatus: number, body: Partial<ErrorOut> | null, fallback: string) {
    super(body?.message || fallback)
    this.name = 'ApiError'
    this.httpStatus = httpStatus
    this.code = body?.code || `http_${httpStatus}`
    this.detail = (body?.detail as Record<string, unknown>) ?? {}
  }
}

/**
 * Where the API lives.
 *
 * Empty string in production: the SPA is served BY the backend, so every request is same-origin and
 * no configuration can point the console at a different cell than the one it is standing in front of.
 * In `npm run dev` Vite proxies `/v1` to the backend, so the same empty string is correct there too --
 * which is the point. One code path, both modes.
 */
const BASE = ''

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${BASE}${path}`, {
      method,
      headers: body === undefined ? {} : { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
    })
  } catch {
    // A dead socket is the single most common state during a bring-up (the server is not started
    // yet), and "Failed to fetch" tells an operator nothing. Name the actual situation.
    throw new ApiError(0, null, `No answer from the backend at ${window.location.origin}. Is it running?`)
  }
  if (!response.ok) {
    let payload: Partial<ErrorOut> | null = null
    try {
      payload = (await response.json()) as Partial<ErrorOut>
    } catch {
      payload = null
    }
    throw new ApiError(response.status, payload, `${method} ${path} failed with ${response.status}.`)
  }
  if (response.status === 204) return undefined as T
  const text = await response.text()
  if (!text) return undefined as T
  const contentType = response.headers.get('content-type') || ''
  return (contentType.includes('application/json') ? JSON.parse(text) : text) as T
}

/**
 * Upload a file. A sibling of `request` rather than a branch inside it, and deliberately so.
 *
 * `request` sets `Content-Type: application/json` and `JSON.stringify`s the body. Neither is right
 * here: the browser MUST set the content type for a multipart body, because only it knows the
 * boundary string it generated -- setting the header by hand produces a request the server cannot
 * parse, with an error that points at the payload rather than the header. Folding this into
 * `request` would mean one function with two mutually exclusive halves.
 *
 * The error path is the same `ApiError`, so a screen handles a failed upload exactly like any other.
 */
async function upload<T>(path: string, field: string, file: Blob, filename: string): Promise<T> {
  const form = new FormData()
  form.append(field, file, filename)
  let response: Response
  try {
    response = await fetch(`${BASE}${path}`, { method: 'POST', body: form })
  } catch {
    throw new ApiError(0, null, `No answer from the backend at ${window.location.origin}. Is it running?`)
  }
  if (!response.ok) {
    let payload: Partial<ErrorOut> | null = null
    try {
      payload = (await response.json()) as Partial<ErrorOut>
    } catch {
      payload = null
    }
    throw new ApiError(response.status, payload, `POST ${path} failed with ${response.status}.`)
  }
  return (await response.json()) as T
}

function query(params: Record<string, string | number | boolean | undefined>): string {
  const parts = Object.entries(params)
    .filter(([, v]) => v !== undefined && v !== '')
    .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`)
  return parts.length ? `?${parts.join('&')}` : ''
}

export const api = {
  health: () => request<{ status: string; version: string }>('GET', '/v1/health'),

  preflight: () => request<PreflightOut>('GET', '/v1/preflight'),

  diagnostics: () => request<DiagnosticsOut>('GET', '/v1/diagnostics'),

  /** Pure text analysis -- no GPU, no model, no image. Free to call on every keystroke. */
  routePreview: (prompt: string) =>
    request<RoutePreviewOut>('GET', `/v1/diagnostics/route${query({ prompt })}`),

  cell: () => request<CellOut>('GET', '/v1/cell'),

  /**
   * Assemble the cell. Touches no robot -- but on a REAL cell it opens the camera and loads two
   * models onto the GPU, which takes tens of seconds. `rehearse` substitutes a desk scene for both.
   */
  build: (rehearse: boolean) =>
    request<CellOut>('POST', `/v1/cell/build${query({ rehearse })}`),

  /** What connecting will do, plus the token that records having read it. */
  connectPreview: () => request<ConnectPreviewOut>('GET', '/v1/cell/connect-preview'),

  /** THIS MOVES. The token is the acknowledgement; there is deliberately no force flag. */
  connect: (token: string) => request<CellOut>('POST', '/v1/cell/connect', { token }),

  disconnect: () => request<CellOut>('POST', '/v1/cell/disconnect'),

  /**
   * A default tick is free. `includeControllerState` costs a dashboard socket round trip, which is
   * why it is a parameter and not always-on -- and why `TelemetryOut.controller_state_included` says
   * whether the mode/safety fields in the response were actually read or are simply absent.
   */
  status: (includeControllerState = false) =>
    request<TelemetryOut>(
      'GET',
      `/v1/cell/status${query({ include_controller_state: includeControllerState })}`,
    ),

  /**
   * Does not stop the arm. It declines to start the NEXT attempt -- the UI must say so. Naming the run makes the
   * server refuse a run that is not a pick run (`409 not_a_pick`) instead of stopping whatever is active. The console
   * starts no pick run: programs do, at `POST /v1/pick`; the console follows one and may stop it.
   */
  stopPick: (runId?: string) => request<RunOut>('POST', `/v1/pick/stop${query({ run_id: runId })}`),

  /**
   * Turn a recording into a text PROPOSAL. It starts nothing.
   *
   * ⛔ Deliberately not a shortcut to a pick, on both sides of the wire: the text lands in the
   * prompt box and a human presses the button, because a spoken command that went straight to motion
   * would mean a misheard word moves an arm.
   *
   * The answer carries `text` plus the evidence behind it: `speech`, what the voice detector found
   * before Whisper was asked, and `transcript`, Whisper's own report (the language it decoded, the
   * milliseconds it took). A recording the detector hears no speech in comes back with an empty
   * `text` and a `reason`, and Whisper is never asked, because Whisper answers silence with a word.
   *
   * ⚠ Send WAV. The console encodes it in the browser (`prompt/recordWav.ts`) because no browser
   * RECORDS WAV, and WAV is the only format the backend decodes: the second decoder is gone
   * (a GPL FFmpeg inside the wheel), and anything else is answered with 415.
   */
  transcribe: (wav: Blob) =>
    upload<ProposalOut>('/v1/voice/transcribe', 'audio', wav, 'prompt.wav'),

  runs: (limit?: number) => request<RunOut[]>('GET', `/v1/runs${query({ limit })}`),
  run: (runId: string) => request<RunOut>('GET', `/v1/runs/${encodeURIComponent(runId)}`),

  kpis: () => request<RollupOut>('GET', '/v1/history/kpis'),
  records: (limit = 100) => request<RecordOut[]>('GET', `/v1/history/records${query({ limit })}`),

  writable: () => request<WritableOut[]>('GET', '/v1/config/writable'),
  explain: (key: string) => request<ConfigValueOut>('GET', `/v1/config/explain${query({ key })}`),
  /** The body IS the mapping of key -> value; all of them land or the request fails and none do. */
  patch: (values: Record<string, unknown>) =>
    request<Schemas['ConfigPatchOut']>('PATCH', '/v1/config', values),

  // ── the contract of commit 2 (build plan 1.11) ─────────────────────────────────────────────────────────────
  //
  // Three of these move the arm: `task`, `restart` and `home`. Each is called from exactly one click, its button
  // says that the robot moves, and nothing in this file calls them on its own: no retry, no resend after a
  // reconnect. A refusal comes back as the typed `ApiError`, and the server enforces every gate itself.

  /** Every typed code (stop codes, events, refusals, lights). The console's unions are generated from the same. */
  codes: () => request<CodesOut>('GET', '/v1/codes'),

  /** THIS MOVES. One task: pick, place, return, look again. Start is the confirmation; 202, then events. */
  task: (body: TaskIn) => request<RunOut>('POST', '/v1/task', body),

  /** "Stop after this part": a held part is still placed and the arm returns. Not wired to the arm in flight. */
  stopTask: (runId: string) => request<RunOut>('POST', `/v1/task/stop${query({ run_id: runId })}`),

  /**
   * THIS MOVES. A NEW run for the console's recovery record; its first motion is the planned move to the return
   * pose. Refused until the cell is confirmed clear, the jaws are answered and no part is held.
   */
  restart: (runId: string) => request<RunOut>('POST', '/v1/task/restart', { run_id: runId }),

  /** What the cell is (cameras, looks, push, brake, payload, route). Read after build and after connect. */
  facts: () => request<CellFactsOut>('GET', '/v1/cell/facts'),

  /** The ready bar: lights and blockers. Always 200, moves nothing; Start only when `ready`. */
  readiness: () => request<ReadinessOut>('GET', '/v1/cell/readiness'),

  /**
   * A person's word that the cell is clear (`cell_clear` is the literal true), and, when ticked, that the jaws hold
   * nothing. It clears the console's latches; it never clears a protective stop, which is done at the pendant.
   */
  acknowledge: (jawsEmpty = false) =>
    request<CellOut>('POST', '/v1/cell/acknowledge', { cell_clear: true, jaws_empty: jawsEmpty }),

  /**
   * "Halt now": the run stops commanding and, where the arm latches, nothing after the move in flight is sent to it.
   * Only where the arm brakes a move in flight (`robot.ur.brake_on_halt`, off as shipped) is that move braked under
   * control. It is NOT an emergency stop. The red button at the cell is.
   */
  brake: () => request<BrakeOut>('POST', '/v1/cell/brake'),

  /** THIS MOVES. A planned move to Home or a taught pose; refused during a run, on a stopped controller, while held. */
  home: (to = 'home') => request<RunOut>('POST', '/v1/cell/home', { to }),

  /** Start cuRobo (about a minute). Moves nothing. */
  startPlanner: () => request<RunOut>('POST', '/v1/cell/planner'),

  /** THIS MOVES. Willy waves back at a greeting: two swings of the wrist, each judged; refused as a new task is. */
  wave: () => request<RunOut>('POST', '/v1/cell/wave'),

  /** The hand and the jaws question waiting for its answer, if any. Takes no session lock. */
  jaws: () => request<JawsOut>('GET', '/v1/cell/jaws'),

  /** One answer to the waiting question. `open_now` is ONE change of the output: the jaws move. */
  answerJaws: (questionId: string, choice: JawQuestionOut['choices'][number]) =>
    request<JawsOut>('POST', '/v1/cell/jaws/answer', { question_id: questionId, choice }),

  /** Ask where the jaws stand now (Restart, Setup, the ready bar). Blocks until answered; no answer is never open. */
  checkJaws: () => request<JawsOut>('POST', '/v1/cell/jaws/check'),

  /** One display frame through `Camera.peek`: never waits, never a measurement. `measuring` keeps the last frame. */
  live: (rig?: string | null, maxWidth = 960) =>
    request<LiveFrameOut>('GET', `/v1/camera/live${query({ rig: rig ?? undefined, max_width: maxWidth })}`),

  poses: () => request<PosesOut>('GET', '/v1/poses'),

  /** The default place pose, or `null` for none. Written to the cell profile's last layer. */
  setDefaultPlace: (name: string | null) => request<PosesOut>('PUT', '/v1/poses/default-place', { name }),

  /** The payload the controller compensates for, which the person confirms before the arm is freed. */
  teachPayload: () => request<PayloadOut>('GET', '/v1/teach/payload'),

  /** Frees the arm for a person to guide by hand. One pose per session; the answer carries the session's token. */
  teach: (body: TeachIn) => request<TeachStartOut>('POST', '/v1/teach', body),

  /** Every poll is the browser's heartbeat: 3 s without one and the arm is held once it stands still. */
  teachState: (runId: string, token: string) =>
    request<TeachStateOut>('GET', `/v1/teach/${encodeURIComponent(runId)}${query({ token })}`),

  /** Save: the arm is held, the pose screened, and written only when the screen clears it. */
  teachCapture: (runId: string, token: string) =>
    request<TeachStateOut>('POST', `/v1/teach/${encodeURIComponent(runId)}/capture${query({ token })}`),

  /** Cancel: the arm is held at once; nothing is saved. */
  teachCancel: (runId: string, token: string) =>
    request<TeachStateOut>('POST', `/v1/teach/${encodeURIComponent(runId)}/cancel${query({ token })}`),

  /** What a sentence asks for, read by the VLM. It creates no run and touches no cell; Start does, after a look. */
  parse: (body: CommandIn) => request<CommandOut>('POST', '/v1/commands/parse', body),

  commandStatus: () => request<CommandStatusOut>('GET', '/v1/commands/status'),

  /** Load the command reader's model. Refused during a run. */
  warmCommands: () => request<CommandStatusOut>('POST', '/v1/commands/warmup'),
}

export type Api = typeof api
export type { paths }

/** The n-th overlay a task run captured when its grasp was decided: its own image, never drawn over the live one. */
export function overlayUrl(runId: string, n: number): string {
  return `${BASE}/v1/runs/${encodeURIComponent(runId)}/overlays/${n}`
}

/** The target a camera place found, drawn over the frame it was found in. */
export function targetOverlayUrl(runId: string): string {
  return `${BASE}/v1/runs/${encodeURIComponent(runId)}/target/overlay`
}

/**
 * Cancel a teach session as the page goes away (`pagehide`): a beacon is the one request a closing tab still sends.
 * The server holds the arm at once on it. Returns whether the browser queued the beacon.
 */
export function sendTeachCancelBeacon(runId: string, token: string): boolean {
  const url = `${BASE}/v1/teach/${encodeURIComponent(runId)}/cancel${query({ token })}`
  try {
    return typeof navigator.sendBeacon === 'function' && navigator.sendBeacon(url)
  } catch {
    return false
  }
}
