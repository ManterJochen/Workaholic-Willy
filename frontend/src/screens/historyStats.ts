/**
 * A task's numbers, from the task's own events: the success rate and the median time per part.
 *
 * `GET /v1/history/runs` says how many parts a task placed and how many picks it made, but not which picks were empty
 * looks (an "until empty" task always ends on two, which would draw a perfect run as 50 %), and not how long each part
 * took. Both are in the run's event stream, which the server keeps with the run. So History replays each ended task
 * through the same reducer the cockpit draws with (`model/runModel`): the numbers on this page are the cockpit's
 * numbers, never a second formula that could disagree with them.
 *
 * Bounded: only the newest `REPLAY_TASKS` ended tasks are replayed, each stream is closed once its run's end has been
 * replayed, and one that does not end within `REPLAY_TIMEOUT_MS` is closed and counted as incomplete. A gap in a
 * replay (events the server no longer holds) marks the task's numbers incomplete; they are still shown, and said to be.
 *
 * The same streams say what a Home or teach run was asked to do (where it went, which pose it taught), which the run
 * record does not keep: only their first event is read (`useRunInputs`), for the newest `LABELLED_RUNS` of them.
 */

import { useEffect, useRef, useState } from 'react'

import type { RunOut } from '../api/client'
import { followRun } from '../api/events'
import { EMPTY_RUN, reduce, type RunView } from '../model/runModel'

/** How many of the newest ended tasks are replayed. */
export const REPLAY_TASKS = 12

/** A replay that has not reached its run's end by then is closed and counted incomplete. */
export const REPLAY_TIMEOUT_MS = 6000

export interface TaskStats {
  readonly runId: string
  /** Placed per fair pick, as the cockpit's model says it; `null` where no pick had anything to grip. */
  readonly successRate: number | null
  /** The median seconds of the parts placed, from the look to the return; `null` where none was placed. */
  readonly medianPartS: number | null
  readonly placed: number
  readonly picks: number
  readonly emptyLooks: number
  /**
   * The picks the rate is of, counted as the cockpit counts them (`StatsView.rated`): every pick but the empty looks,
   * the picks a person cut short (a halt, a cancel) and the parts a person halted in the jaws. A model that keeps no
   * such count leaves out what it does keep. The session's rate sums these, so it never disagrees with the bars beside
   * it or with the cockpit.
   */
  readonly fairPicks: number
  /** Every placed part's seconds, for the session's own median. */
  readonly partSeconds: readonly number[]
  /** A gap swallowed events, or the replay never reached the run's end. */
  readonly incomplete: boolean
}

/** The numbers of one replayed task. */
export function statsOf(runId: string, view: RunView, complete: boolean): TaskStats {
  const stats = view.stats
  // The model's own count of the picks its rate is of, where it keeps one; else the picks a person cut short, where it
  // counts them (they are no failed grasp), and the empty looks are left out.
  const rated = (stats as { readonly rated?: number }).rated
  const cut = (stats as { readonly cut?: number }).cut ?? 0
  return {
    runId,
    successRate: stats.successRate,
    medianPartS: stats.medianPartS,
    placed: stats.placed,
    picks: stats.picks,
    emptyLooks: stats.emptyLooks,
    fairPicks: typeof rated === 'number' ? Math.max(0, rated) : Math.max(0, stats.picks - stats.emptyLooks - cut),
    partSeconds: view.parts.filter((p) => p.placed === true && p.durationS !== null).map((p) => p.durationS as number),
    incomplete: !complete || view.gaps.count > 0,
  }
}

/** The tasks worth replaying: ended task runs, the newest first, at most `REPLAY_TASKS`. */
export function replayable(runs: readonly RunOut[]): RunOut[] {
  return runs.filter((run) => run.kind === 'task' && run.state !== 'running').slice(0, REPLAY_TASKS)
}

/** Each replayed task's numbers by run id; a task still being replayed is not in the map yet. */
export function useTaskStats(runs: readonly RunOut[]): ReadonlyMap<string, TaskStats> {
  const [stats, setStats] = useState<ReadonlyMap<string, TaskStats>>(() => new Map())
  /** Replays opened in this page's life: each task is replayed once. */
  const opened = useRef(new Set<string>())
  const wanted = replayable(runs)
  const key = wanted.map((run) => run.id).join(',')

  useEffect(() => {
    const stops: Array<() => void> = []
    for (const run of replayable(runs)) {
      if (opened.current.has(run.id)) continue
      opened.current.add(run.id)
      let view = reduce(EMPTY_RUN, { type: 'run.snapshot', run })
      let done = false
      let timer: ReturnType<typeof setTimeout> | undefined
      const finish = (complete: boolean) => {
        if (done) return
        done = true
        clearTimeout(timer)
        stop()
        setStats((previous) => new Map(previous).set(run.id, statsOf(run.id, view, complete)))
      }
      const stop = followRun(
        run.id,
        {
          onEvent: (event) => {
            if (done) return
            view = reduce(view, event)
            if (event.type === 'run_finished') finish(true)
          },
          onState: () => undefined,
        },
        { sinceSeq: 0 },
      )
      timer = setTimeout(() => finish(false), REPLAY_TIMEOUT_MS)
      stops.push(() => {
        if (done) return
        // The page went away before the replay ended: let a later mount replay it again.
        done = true
        clearTimeout(timer)
        stop()
        opened.current.delete(run.id)
      })
    }
    return () => {
      for (const stop of stops) stop()
    }
    // Keyed on the replayable ids: a fresh list with the same tasks opens nothing new.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key])

  return stats
}

/** How many of the newest Home and teach runs are asked where they went, or which pose they taught. */
export const LABELLED_RUNS = 20

/** What a Home or teach run was asked to do, as its `run_started` said it (the run record does not keep it). */
export interface RunInputs {
  /** A Home run's target: `home` or a pose name. */
  readonly to?: string
  /** A Home run's pose label, or a teach run's label. */
  readonly label?: string
  /** A teach run's pose name. */
  readonly name?: string
}

/** The Home and teach runs worth asking: the newest first, at most `LABELLED_RUNS`. */
export function labellable(runs: readonly RunOut[]): RunOut[] {
  return runs.filter((run) => run.kind === 'home' || run.kind === 'teach').slice(0, LABELLED_RUNS)
}

function text(value: unknown): string | undefined {
  return typeof value === 'string' && value !== '' ? value : undefined
}

/**
 * Where each Home run went and which pose each teach run taught, by run id: the run's first event, read from its
 * stream and then closed (one event per run, bounded like the task replays above). A run whose stream says nothing
 * within `REPLAY_TIMEOUT_MS` is left without words, never guessed.
 */
export function useRunInputs(runs: readonly RunOut[]): ReadonlyMap<string, RunInputs> {
  const [inputs, setInputs] = useState<ReadonlyMap<string, RunInputs>>(() => new Map())
  const opened = useRef(new Set<string>())
  const key = labellable(runs)
    .map((run) => run.id)
    .join(',')

  useEffect(() => {
    const stops: Array<() => void> = []
    for (const run of labellable(runs)) {
      if (opened.current.has(run.id)) continue
      opened.current.add(run.id)
      let done = false
      let timer: ReturnType<typeof setTimeout> | undefined
      const close = () => {
        if (done) return
        done = true
        clearTimeout(timer)
        stop()
      }
      const stop = followRun(
        run.id,
        {
          onEvent: (event) => {
            if (done || event.type !== 'run_started') return
            const data = (event.data ?? {}) as Record<string, unknown>
            const said: RunInputs = { to: text(data.to), label: text(data.label), name: text(data.name) }
            close()
            setInputs((previous) => new Map(previous).set(run.id, said))
          },
          onState: () => undefined,
        },
        { sinceSeq: 0 },
      )
      timer = setTimeout(close, REPLAY_TIMEOUT_MS)
      stops.push(() => {
        if (done) return
        close()
        // The page went away before the run said anything: a later mount asks again.
        opened.current.delete(run.id)
      })
    }
    return () => {
      for (const stop of stops) stop()
    }
    // Keyed on the runs worth asking: a fresh list with the same runs opens nothing new.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key])

  return inputs
}

/** The median of some numbers, `null` for none. */
export function median(values: readonly number[]): number | null {
  if (values.length === 0) return null
  const sorted = [...values].sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  return sorted.length % 2 === 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
}

/** A scale's top: the next round number (1, 2, 2.5 or 5 times a power of ten) at or above `value`. */
export function niceCeiling(value: number): number {
  if (!(value > 0)) return 1
  const power = 10 ** Math.floor(Math.log10(value))
  const step = [1, 2, 2.5, 5, 10].find((f) => f * power >= value - 1e-9) ?? 10
  return step * power
}
