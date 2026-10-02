/**
 * Where the cell stands on the way to its first task: Check -> Build -> Preview -> Connect -> Ready (build plan 4.3).
 *
 * Pure, from what the console reads (the cell's state, the checklist, whether the person read it, whether the connect
 * preview was read, the ready answer), so the stepper can never claim a step that was not reached. The checklist is
 * advice, never a gate: a blocking row is drawn as a warning on its step, because a cell with a blocking checklist can
 * still connect (and then refuse every motion), and the step that refuses a connect is the preview's own `blocking`
 * list. But it is never skipped: a cell that is down opens at the check until the person has read it and gone on
 * (`checkSeen`, kept for the browser session), and once passed the step wears a warning, never a tick, while the
 * checklist warns or blocks.
 */

export type StepId = 'check' | 'build' | 'preview' | 'connect' | 'ready'

export const STEP_ORDER: readonly StepId[] = ['check', 'build', 'preview', 'connect', 'ready']

/** `done` passed, `warn` passed with something to read, `current` the step to do now, `todo` not reached. */
export type StepMark = 'done' | 'warn' | 'current' | 'todo'

export interface SetupState {
  /** `CellOut.state`, or `null` before the first answer. */
  readonly cellState: string | null
  /** Blocking rows of the checklist, `null` while it was not read. */
  readonly preflightBlocking: number | null
  /** Warning rows of the checklist, `null` while it was not read. */
  readonly preflightWarnings: number | null
  /** The person read the checklist in this browser session and went on (or picked a later step). */
  readonly checkSeen: boolean
  /** A connect preview was read for the cell as it is built now (its token is held). */
  readonly previewRead: boolean
  /** `ReadinessOut.ready`, `null` when the server did not answer it. */
  readonly ready: boolean | null
}

/** The step to do now. */
export function currentStep(state: SetupState): StepId {
  switch (state.cellState) {
    case 'disconnected':
      return state.checkSeen ? 'build' : 'check'
    case 'built':
      return state.previewRead ? 'connect' : 'preview'
    case 'connecting':
      return 'connect'
    case 'connected':
      return 'ready'
    default:
      return 'check'
  }
}

/** Every step's mark, in order. The ready step is `done` only when the server says the cell is ready. */
export function stepMarks(state: SetupState): Record<StepId, StepMark> {
  const current = currentStep(state)
  const at = STEP_ORDER.indexOf(current)
  const marks = {} as Record<StepId, StepMark>
  STEP_ORDER.forEach((id, index) => {
    if (index < at) marks[id] = 'done'
    else if (index === at) marks[id] = 'current'
    else marks[id] = 'todo'
  })
  const notes = (state.preflightBlocking ?? 0) + (state.preflightWarnings ?? 0)
  if (notes > 0 && marks.check === 'done') marks.check = 'warn'
  if (current === 'ready' && state.ready === true) marks.ready = 'done'
  return marks
}

/** Where the browser session keeps that the person read the checklist (`sessionStorage`). */
export const CHECK_SEEN_KEY = 'willy.setup.checkSeen'

/** Whether the person read the checklist in this browser session. Never throws (a locked-down kiosk). */
export function readCheckSeen(): boolean {
  try {
    return sessionStorage.getItem(CHECK_SEEN_KEY) === '1'
  } catch {
    return false
  }
}

/** Remember for this browser session that the person read the checklist and went on. */
export function writeCheckSeen(): void {
  try {
    sessionStorage.setItem(CHECK_SEEN_KEY, '1')
  } catch {
    /* without storage the session still remembers it in its state */
  }
}
