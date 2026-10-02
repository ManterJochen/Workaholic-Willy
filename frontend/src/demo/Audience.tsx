/**
 * The audience window (`demo.html`, build plan 4.3, OD 4 and 21): a read-only mirror of the cell for the large screen.
 *
 * **Nothing here can move anything.** There is no button, no link, no input: the page reads the cell and the run and
 * draws them. The operator opens it from the console's top bar, drags it to the projector and presses F11.
 *
 * **What it shows**, in the logo's look (the black stage, the neon lime, the banner's HUD corner marks):
 * - the live image of the primary camera, full-bleed on the stage (`/v1/camera/live`, kept while the camera measures);
 * - a new grasp overlay pinned over the stage for 5 s, as its own picture, with the band that says it shows the grasp
 *   as decided and that a wrist camera's arm has moved since (build plan 1.7; never drawn over the moving image);
 * - the step in big words with the part and where it goes, the counters, and the steps as a strip (the one step the
 *   caption names is lit; the hands-off countdown before a motion the robot starts by itself is in the warning hue);
 * - a large read-only card when a run stopped on a problem or asks a question: what happened, and that a person at
 *   the console decides; once a person confirmed the cell is clear, that it waits for Restart or Home. While a card is
 *   up it is the one thing to read: the caption under it goes, and a problem stop has no joke beside it;
 * - the STILL GRINDING card: today's parts and working time, with no coffee breaks and no raise asked (the owner's
 *   brand, 2026-10-01; the one joke in the whole console lives here and nowhere else). Its counts never go down while
 *   the window is open: a run the server stops listing keeps counting.
 *
 * **What it follows.** The run the cell names (`active_run_id`, the shared poll every second in this window), through
 * the same run model the cockpit draws with; after a stop, the console's recovery record, so a stopped run's card is on
 * the projector even when the window was opened after the stop, until a Restart or Home ends the record. A run so short
 * that it began and ended between two reads of the cell (a once task, a Home) is caught from the session's runs, read
 * every few seconds: the newest one that began after the window opened is followed to its end. A taught pose is named
 * by its label (`GET /v1/poses`, read as runs start and end), never by its YAML name.
 */

import { useEffect, useRef, useState } from 'react'

import { api, type RunOut } from '../api/client'
import { isStopCode } from '../api/codes'
import { ProvenanceBadge } from '../components/ProvenanceBadge'
import { useT, type Msg } from '../i18n'
import { runKindMsg, stopMsg, stopSayMsg } from '../i18n/codes'
import { Icon, Logo } from '../icons'
import { usePoll } from '../lib/useAsync'
import {
  hasEnded,
  lookNote,
  phaseMsg,
  plannedLooks,
  STEPS,
  type OverlayPin,
  type RunView,
  type StepState,
} from '../model/runModel'
import { useCell } from '../model/useCell'
import { useRun } from '../model/useRun'
import { captionOf, grindOf, placeWords, type PoseLabels } from './caption'
import { DEMO, type DemoKey, type DemoT } from './i18n'
import { useLiveFrame } from './live'

/** How long a new overlay stays pinned over the stage (build plan 1.7). */
export const PIN_S = 5

/** The live picture on the projector: wide enough not to look soft on a 1080p screen. */
const LIVE_WIDTH = 1280

/** Four pictures a second; two while a person guides the arm in a teach (the teach watchdog's margin). */
const LIVE_MS = 250
const LIVE_TEACH_MS = 500

/** The session's runs, for today's counts and for a run too short for the cell's poll: read every 5 s, and whenever a
 * run starts or ends. */
export const RUNS_MS = 5000

/** Every run the server keeps (`api/runs.py` RETAINED_RUNS): today's counts are not cut at the default 50. */
const RUNS_KEPT = 200

function words(t: DemoT, value: string | Msg<string> | null): string {
  if (value === null) return ''
  return typeof value === 'string' ? value : t.msg(value)
}

/** The clock the pin and the counters read: seconds, ticking twice a second. */
function useNow(): number {
  const [now, setNow] = useState(() => Date.now() / 1000)
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now() / 1000), 500)
    return () => clearInterval(id)
  }, [])
  return now
}

/**
 * How far an overlay's capture may lie ahead of this window's clock and still be new: the clock here ticks twice a
 * second, so a grasp decided since the last tick reads as "in the future", and the server's clock may run a little
 * ahead of this screen's (another PC). Only a capture further ahead than that is not trusted as new.
 */
const AHEAD_S = 2

/** The newest overlay, while it is fresh: from the moment it arrives until `PIN_S` after it was captured. */
function freshPin(view: RunView, nowS: number): OverlayPin | null {
  const newest = view.overlays[view.overlays.length - 1]
  if (!newest) return null
  const age = nowS - newest.at
  return age > -AHEAD_S && age < PIN_S ? newest : null
}

function hours(t: DemoT, seconds: number): string {
  if (seconds < 60) return t('aud.grind.seconds', { s: Math.floor(seconds) })
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return t('aud.grind.minutes', { m: minutes })
  return t('aud.grind.hours', { h: Math.floor(minutes / 60), mm: String(minutes % 60).padStart(2, '0') })
}

function Grind({ t, runs, view, nowS }: { t: DemoT; runs: readonly RunOut[]; view: RunView; nowS: number }) {
  const grind = grindOf(runs, view, nowS)
  const row = (k: DemoKey, v: string) => (
    <div className="readout-row">
      <span className="k">{t(k)}</span>
      <span className="lead" aria-hidden="true" />
      <span className="v">{v}</span>
    </div>
  )
  return (
    <section className="readout aud-grind" aria-label={t('aud.grind.title')}>
      <div className="readout-title">{t('aud.grind.title')}</div>
      {row('aud.grind.parts', String(grind.partsToday))}
      {row('aud.grind.time', hours(t, grind.workS))}
      {row('aud.grind.coffee', '0')}
      {row('aud.grind.raises', '0')}
    </section>
  )
}

/** A stopped run's card, or a question's: read-only, big, and saying who decides. */
function Card({ t, view, cleared }: { t: DemoT; view: RunView; cleared: boolean }) {
  const stop = view.stopCard
  const ask = view.askCard
  const code = stop?.stopCode ?? ask?.stopCode ?? null
  if (!code || !isStopCode(code)) return null
  const part = stop?.part ?? ask?.part ?? null
  const step = stop?.step ?? null
  const next = !stop ? 'aud.card.ask' : cleared ? 'aud.card.cleared' : 'aud.card.problem'
  return (
    <section className={`aud-card hud-frame ${stop ? 'problem' : 'ask'}`} role="alert" aria-labelledby="aud-card-title">
      <div className="aud-card-head">
        <Icon name={stop ? 'alert' : 'info'} size={34} className="aud-card-icon" />
        <h2 id="aud-card-title">{t.msg(stopMsg(code))}</h2>
      </div>
      {part !== null && step && (
        <p className="aud-card-where">{t('aud.card.where', { part, step: t(`step.${step}`) })}</p>
      )}
      <p className="aud-card-said">{t.msg(stopSayMsg(code))}</p>
      <p className="aud-card-next">{t(next)}</p>
    </section>
  )
}

/**
 * How a step is drawn in the strip: the step the caption names is the one lit; another the model still marks active
 * (the looks detect while they look) is drawn done before it and not yet after it, so the strip never lights two.
 */
function stripState(view: RunView, id: (typeof STEPS)[number]): StepState {
  const state = view.timeline.find((step) => step.id === id)?.state ?? 'idle'
  if (state !== 'active') return state
  const active = STEPS.filter((step) => view.timeline.find((s) => s.id === step)?.state === 'active')
  const lit = view.current.step && active.includes(view.current.step) ? view.current.step : active[active.length - 1]
  if (id === lit) return 'active'
  return STEPS.indexOf(id) < STEPS.indexOf(lit) ? 'done' : 'idle'
}

export default function Audience() {
  const t = useT(DEMO)
  const { cell, facts, error } = useCell()
  const { view, follow } = useRun()
  const nowS = useNow()
  /** Every run seen since the window opened, the newest record of each: what the server stops listing keeps counting. */
  const [runs, setRuns] = useState<readonly RunOut[]>([])
  /** The newest run the server listed in this window's first read: anything older is not followed (`-Infinity`: none). */
  const [baseline, setBaseline] = useState<number | null>(null)
  /** The newest run of the last read, newest first as the server lists them. */
  const [newest, setNewest] = useState<RunOut | null>(null)
  const seen = useRef(new Map<string, RunOut>())

  const active = cell?.active_run_id ?? null
  const connected = cell?.state === 'connected'
  const teaching = Boolean(active) && view.kind === 'teach'
  const live = useLiveFrame({ maxWidth: LIVE_WIDTH, intervalMs: teaching ? LIVE_TEACH_MS : LIVE_MS, enabled: cell !== null })

  // After a stop: the recovery record's run, when no run is active and another is on screen.
  const record = cell?.recovery ?? null
  const recovery = record?.run_id ?? null
  useEffect(() => {
    if (!recovery || active || view.runId === recovery) return
    let cancelled = false
    api
      .run(recovery)
      .then((run) => {
        if (!cancelled) follow(run)
      })
      .catch(() => undefined)
    return () => {
      cancelled = true
    }
  }, [recovery, active, view.runId, follow])

  // The session's runs: today's counts, and the newest run.
  const readRuns = () => {
    api
      .runs(RUNS_KEPT)
      .then((answer) => {
        for (const run of answer) seen.current.set(run.id, run)
        setRuns([...seen.current.values()])
        setNewest(answer[0] ?? null)
        setBaseline((before) => before ?? (answer[0]?.started_at ?? Number.NEGATIVE_INFINITY))
      })
      .catch(() => undefined)
  }
  useEffect(readRuns, [active])
  usePoll(readRuns, RUNS_MS, true)

  // The taught poses' labels, read again whenever a run starts or ends (a teach may have added one): a Home run's
  // record names its pose only by its YAML name, and the projector says "Parkposition", never "park".
  const [labels, setLabels] = useState<PoseLabels>(() => new Map())
  useEffect(() => {
    let cancelled = false
    api
      .poses()
      .then((poses) => {
        if (cancelled) return
        setLabels(new Map((poses.poses ?? []).filter((pose) => pose.label).map((pose) => [pose.name, pose.label])))
      })
      .catch(() => undefined)
    return () => {
      cancelled = true
    }
  }, [active, connected])

  // A run too short for the cell's poll (it began and ended between two reads): the newest run, when it began after
  // the window opened and after the run on screen, while nothing runs and no stop record holds the projector.
  useEffect(() => {
    if (!newest || active || recovery || baseline === null) return
    if (newest.id === view.runId || newest.started_at <= baseline) return
    if (view.startedAt !== null && newest.started_at <= view.startedAt) return
    follow(newest)
  }, [newest, active, recovery, baseline, view.runId, view.startedAt, follow])

  useEffect(() => {
    document.title = `Workaholic-Willy · ${t('aud.title')}`
  }, [t])

  const caption = captionOf(view, connected, labels)
  const card = Boolean(view.stopCard || view.askCard)
  // A card is the one thing to read: no pinned picture beside it, no caption under it repeating its title, and no joke
  // beside a problem stop.
  const pin = card ? null : freshPin(view, nowS)
  const problem = Boolean(view.stopCard)
  // A person said the cell is clear since this stop: the card then says it waits for Restart or Home.
  const cleared =
    record !== null &&
    record.run_id === view.runId &&
    typeof record.cleared_at === 'number' &&
    typeof record.at === 'number' &&
    record.cleared_at >= record.at
  const wrist = facts?.wrist_camera === true
  const band = pin
    ? t(pin.kind === 'target' ? 'aud.pin.target' : wrist ? 'aud.pin.wrist' : 'aud.pin.fixed', { time: t.fmt.time(pin.at) })
    : ''
  const running = Boolean(view.runId) && !hasEnded(view.phase) && view.phase !== 'idle'
  const where = placeWords(view.plan)
  const look = lookNote(view, plannedLooks(view.plan, facts))

  const sub: string[] = []
  if (running && view.kind === 'task' && view.current.part !== null) {
    sub.push(
      view.current.of
        ? t('aud.sub.partOf', { part: view.current.part, of: view.current.of })
        : t('aud.sub.part', { part: view.current.part }),
    )
  }
  // While it runs the count is a counter; once ended the caption says it ("Fertig: 3 Teile abgelegt"), once.
  if (running && view.kind === 'task' && view.stats.placed > 0) sub.push(t('aud.sub.placed', { n: view.stats.placed }))
  if (running && view.current.attempt !== null && view.current.attemptTotal) {
    sub.push(t('aud.sub.attempt', { n: view.current.attempt, total: view.current.attemptTotal }))
  }
  if (running && view.current.step === 'look' && look) sub.push(t.msg(look))

  const target =
    view.kind === 'task' && where !== null
      ? view.plan?.place.kind === 'camera'
        ? t('aud.sub.camera', { where: words(t, where) })
        : t('aud.sub.pose', { where: words(t, where) })
      : null
  const scope =
    view.kind === 'task' && view.plan ? t(view.plan.scope === 'until_empty' ? 'aud.sub.untilEmpty' : 'aud.sub.once') : null

  const liveBadge = !live.src
    ? t('aud.live.none')
    : live.failed
      ? t('aud.live.stale')
      : live.frame?.reason === 'measuring'
        ? t('aud.live.measuring')
        : live.frame?.source === 'synthetic'
          ? t('aud.live.synthetic')
          : t('aud.live.camera')

  return (
    <div className={`aud stage tone-${caption.tone}`}>
      {/* `dimmed`, not `dim`: that is the console's muted-text utility, and it would grey the card's words too. */}
      <div className={`aud-stage hud-frame${card ? ' dimmed' : ''}`}>
        {live.src ? (
          <img className="aud-live" src={live.src} alt={t('aud.live.alt')} />
        ) : (
          <div className="aud-nolive">
            <Logo className="aud-nolive-mark" />
            <span>{t('aud.noPicture')}</span>
          </div>
        )}

        <header className="aud-head">
          <div className="aud-brand">
            <Logo className="aud-mark" />
            <span className="aud-wordmark">
              Workaholic <em>Willy</em>
            </span>
          </div>
          <div className="aud-chips">
            <span className={`aud-chip${live.src && !live.failed ? ' live' : ''}`}>
              <span className="aud-chip-dot" aria-hidden="true" />
              {liveBadge}
              {live.frame?.rig_id ? ` · ${live.frame.rig_id}` : ''}
            </span>
            {error && !cell ? (
              <span className="aud-chip warn">{t('aud.noServer')}</span>
            ) : !connected ? (
              <span className="aud-chip">{t('aud.notConnected')}</span>
            ) : (
              <ProvenanceBadgeOnStage />
            )}
            {view.runId && view.kind && (
              <span className="aud-chip">
                {t.msg(runKindMsg(view.kind))} · {view.stopCode ? t.msg(stopMsg(view.stopCode)) : t.msg(phaseMsg(view))}
              </span>
            )}
          </div>
        </header>

        {pin && (
          <figure className="aud-pin">
            <img src={pin.url} alt={band} />
            <figcaption>{band}</figcaption>
          </figure>
        )}

        <Card t={t} view={view} cleared={cleared} />

        <div className="aud-bottom">
          {card ? (
            <div className="aud-caption" />
          ) : (
            <div className="aud-caption">
              <h1 className="aud-verb">
                {caption.verb && t.msg(caption.verb)}
                {caption.verb && caption.what !== null && ' '}
                {caption.what !== null && <em>{words(t, caption.what)}</em>}
              </h1>
              {caption.sub && <p className="aud-sub-line">{t.msg(caption.sub)}</p>}
              {(sub.length > 0 || target || scope) && (
                <p className="aud-counters">
                  {sub.map((part) => (
                    <span key={part}>{part}</span>
                  ))}
                  {target && <span className="aud-target">{target}</span>}
                  {scope && <span>{scope}</span>}
                </p>
              )}
              {running && view.kind === 'task' && (
                <ol className="aud-steps" aria-hidden="true">
                  {STEPS.map((id) => (
                    <li key={id} className={`aud-step ${stripState(view, id)}`}>
                      {t(`step.${id}`)}
                    </li>
                  ))}
                </ol>
              )}
            </div>
          )}
          {!problem && <Grind t={t} runs={runs} view={view} nowS={nowS} />}
        </div>
      </div>
    </div>
  )
}

/** What is on the other end, on the stage: the provenance badge of the connected cell (a simulator says so). */
function ProvenanceBadgeOnStage() {
  const { telemetry } = useCell()
  return (
    <span className="aud-prov">
      <ProvenanceBadge telemetry={telemetry} />
    </span>
  )
}
