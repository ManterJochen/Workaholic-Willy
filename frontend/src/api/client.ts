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
export type ViewfinderOut = Schemas['ViewfinderOut']
export type WritableOut = Schemas['WritableOut']
export type TranscriptOut = Schemas['TranscriptOut']
export type ErrorOut = Schemas['ErrorOut']

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

  /** THIS MOVES. Returns 202 immediately; everything after that arrives on the event stream. */
  pick: (prompt: string, picks: number) =>
    request<RunOut>('POST', '/v1/pick', { prompt, picks }),

  /** Does not stop the arm. It declines to start the NEXT attempt -- the UI must say so. */
  stopPick: () => request<RunOut>('POST', '/v1/pick/stop'),

  /**
   * Turn a recording into TEXT. It starts nothing.
   *
   * ⛔ Deliberately not a shortcut to a pick, on both sides of the wire: the text lands in the
   * prompt box and a human presses the button, because a spoken command that went straight to motion
   * would mean a misheard word moves an arm.
   *
   * ⚠ Send WAV. The console encodes it in the browser (`prompt/recordWav.ts`) because no browser
   * RECORDS WAV, and WAV is the one container the backend decodes without an optional extra. Other
   * formats are accepted where `requirements/voice.txt` is installed, and answered with a 501 naming
   * that file where it is not.
   */
  transcribe: (wav: Blob) =>
    upload<TranscriptOut>('/v1/voice/transcribe', 'audio', wav, 'prompt.wav'),

  runs: () => request<RunOut[]>('GET', '/v1/runs'),
  run: (runId: string) => request<RunOut>('GET', `/v1/runs/${encodeURIComponent(runId)}`),

  /**
   * What the cell is looking at, and which of two pictures it is.
   *
   * Cheap and safe to poll: the backend refuses to touch the camera while a pick owns it, and answers
   * 200 with `source: 'none'` for every no-picture state rather than an error -- a cell without a
   * camera is an ordinary cell, and a 404 would make it look broken.
   */
  camera: () => request<ViewfinderOut>('GET', '/v1/camera'),

  /**
   * Render the grasp overlay on every segmentation. NOT free -- it costs time inside the pick's own
   * latency budget -- which is why it is a switch an operator throws, not a default.
   */
  enableOverlay: (enabled: boolean) =>
    request<{ enabled: boolean }>('POST', `/v1/overlay/enable${query({ enabled })}`),

  kpis: () => request<RollupOut>('GET', '/v1/history/kpis'),
  records: (limit = 100) => request<RecordOut[]>('GET', `/v1/history/records${query({ limit })}`),
  historyRuns: () => request<RunOut[]>('GET', '/v1/history/runs'),

  writable: () => request<WritableOut[]>('GET', '/v1/config/writable'),
  explain: (key: string) => request<ConfigValueOut>('GET', `/v1/config/explain${query({ key })}`),
  /** The body IS the mapping of key -> value; all of them land or the request fails and none do. */
  patch: (values: Record<string, unknown>) =>
    request<Schemas['ConfigPatchOut']>('PATCH', '/v1/config', values),
}

export type Api = typeof api
export type { paths }
