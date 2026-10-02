/**
 * The bar across the top of the cockpit (OD 16; build plan 1.5, 4.2).
 *
 * **Before a task: the ready bar.** One light per thing that must be true before Start (robot, cameras, planner,
 * gripper, carried part; commands for information only), each with the action that makes it true where a click can:
 * "Planer starten" (moves nothing, about a minute), "Backen prüfen" (the question opens in its own window, with no
 * default), "Zelle ist frei", "Laden" (the command reader). A stopped controller and an undeclared carried part say what
 * to do instead: those are fixed at the pendant and in the cell profile, never from here. "Alles bereit" only when the
 * server says ready; Start follows the same word. The verdict is written as words and drawn in the banner's capitals by
 * the style (`.ck-verdict`): the catalogs never shout. The commands light is named for what it is, the language model,
 * and where there is none the Understood card is filled in by hand ("Karte von Hand").
 *
 * **During a run: the run header**, what the task is, where it stands and how long it has run. **After a problem stop:
 * the stop** in red, where it stopped; the stop card in the chat says what to do, and a cell that is not connected
 * (the record survives a Disconnect and a rebuild) is said there too, with the way to Setup.
 *
 * An action is offered only where it can work: "Planer starten" needs a built cell (before that, Setup is the way), so
 * a cell that is not built offers none. A refusal of an action gets a row of its own under the lights.
 */

import { useState } from 'react'
import { Link } from 'react-router-dom'

import { ApiError, api, type CellOut, type LightOut, type PosesOut, type RunOut, type TaskPlanOut } from '../api/client'
import { ErrorBanner } from '../components/ui'
import { useT } from '../i18n'
import { blockerMsg, lightIdMsg, lightMsg, runKindMsg, stopMsg } from '../i18n/codes'
import { Icon } from '../icons'
import { phaseMsg, type RunView, type StepId } from '../model/runModel'
import { useCell } from '../model/useCell'
import { useRun } from '../model/useRun'
import { poseLabel } from './draft'
import { useNow } from './hooks'
import { COCKPIT } from './i18n'

type Action = 'planner' | 'jaws' | 'clear' | 'load'

export interface TopBarProps {
  readonly mode: 'ready' | 'running' | 'stopped'
  readonly view: RunView
  readonly poses: PosesOut | null
  /** The stopped run's record (stopped mode). */
  readonly stoppedRun: RunOut | null
  readonly stoppedAt: { part: number | null; step: StepId | null }
  readonly tech: boolean
  /** "Laden": the command reader, which says in the chat how the load went. */
  onLoad(): void
}

export default function ReadyBar(props: TopBarProps) {
  if (props.mode === 'running') return <RunHeader view={props.view} poses={props.poses} />
  if (props.mode === 'stopped') return <StoppedHeader run={props.stoppedRun} at={props.stoppedAt} />
  return <Lights tech={props.tech} onLoad={props.onLoad} />
}

/** A headline starts with a capital, whichever word leads it ("alles, was die Kamera sieht" in title position). */
function headline(text: string, lang: string): string {
  return text.length > 0 ? text.charAt(0).toLocaleUpperCase(lang) + text.slice(1) : text
}

/** The cell is not connected: nothing moves until it is, and Setup is where it is connected. */
function NotConnected({ halted }: { halted: boolean }) {
  const t = useT(COCKPIT)
  return (
    <>
      <Icon name="alert" size={18} />
      <span>{t(halted ? 'ck.ready.haltedBuilt' : 'ck.ready.notConnected')}</span>
      <Link to="/setup" className="ck-link">
        {t('ck.ready.toSetup')}
      </Link>
    </>
  )
}

function Lights({ tech, onLoad }: { tech: boolean; onLoad: () => void }) {
  const t = useT(COCKPIT)
  const { cell, error: cellError, readiness, refresh } = useCell()
  const { follow } = useRun()
  const [busy, setBusy] = useState<Action | null>(null)
  const [error, setError] = useState<ApiError | null>(null)
  const connected = cell?.state === 'connected'
  const noServer = !cell && cellError !== null

  const act = async (action: Action) => {
    setBusy(action)
    setError(null)
    try {
      if (action === 'planner') follow(await api.startPlanner())
      else if (action === 'jaws') await api.checkJaws()
      else if (action === 'clear') await api.acknowledge(false)
      else onLoad()
    } catch (err: unknown) {
      setError(err instanceof ApiError ? err : new ApiError(0, null, String(err)))
    } finally {
      setBusy(null)
      refresh()
    }
  }

  const lights = readiness?.lights ?? []
  const blockers = readiness?.blockers ?? []
  const verdict = readiness === null ? 'unknown' : readiness.ready ? 'all' : 'not'

  return (
    <section className="ck-top ck-ready" aria-label={t('ck.ready.label')}>
      {noServer && (
        <div className="ck-notice">
          <Icon name="unlink" size={18} />
          <span>{t('ck.ready.noServer')}</span>
        </div>
      )}
      {!noServer && !connected && (
        <div className="ck-notice">
          <NotConnected halted={Boolean(cell?.halted)} />
          {cell?.halted && !cell.active_run_id && (
            // Acknowledge is allowed while the cell is only built: Connect refuses a latched arm until then.
            <button type="button" className="primary ck-light-act" disabled={busy !== null} title={t('ck.act.clearTitle')} onClick={() => void act('clear')}>
              {busy === 'clear' ? t('ck.act.working') : t('ck.act.clear')}
            </button>
          )}
        </div>
      )}
      {lights.length > 0 && (
        <ul className="ck-lights">
          {lights.map((light) => (
            <li key={light.id} className={`ck-light ${light.state}`} title={tech ? light.message : undefined}>
              <span className="ck-dot" aria-hidden="true" />
              <span className="k">{t.msg(lightIdMsg(light.id))}</span>
              <span className="v">{t.msg(lightMsg(light.code))}</span>
              <LightAction light={light} cell={cell} busy={busy} act={act} />
            </li>
          ))}
        </ul>
      )}
      {blockers.length > 0 && (
        <ul className="ck-blockers">
          {blockers.map((blocker) => (
            <li key={blocker.code}>{t.msg(blockerMsg(blocker.code))}</li>
          ))}
        </ul>
      )}
      <span className={`ck-verdict ${verdict}`}>
        {t(verdict === 'all' ? 'ck.ready.all' : verdict === 'not' ? 'ck.ready.not' : 'ck.ready.unknown')}
      </span>
      {error && (
        <div className="ck-top-error">
          <ErrorBanner error={error} />
        </div>
      )}
    </section>
  )
}

/** The action a light offers, where a click can make it true; else what a person does instead, in words. */
function LightAction({
  light,
  cell,
  busy,
  act,
}: {
  light: LightOut
  cell: CellOut | null
  busy: Action | null
  act: (action: Action) => Promise<void>
}) {
  const t = useT(COCKPIT)
  const button = (action: Action, label: string, title: string) => (
    <button type="button" className="ck-light-act" disabled={busy !== null || Boolean(cell?.active_run_id)} title={title} onClick={() => void act(action)}>
      {busy === action ? t('ck.act.working') : label}
    </button>
  )
  const say = (text: string) => <span className="ck-light-say">{text}</span>
  // The planner starts on a built cell only (`POST /v1/cell/planner` refuses `not_built`): before that, Setup.
  const built = cell?.state === 'built' || cell?.state === 'connected'
  switch (`${light.id}:${light.code}`) {
    case 'robot:halted':
      return button('clear', t('ck.act.clear'), t('ck.act.clearTitle'))
    case 'robot:controller_stopped':
      return say(t('ck.act.pendant'))
    case 'planner:off':
    case 'planner:failed':
      return built ? button('planner', t('ck.act.planner'), t('ck.act.plannerTitle')) : null
    case 'gripper:jaws_unknown':
    case 'gripper:jaws_closed':
      return button('jaws', t('ck.act.jaws'), t('ck.act.jawsTitle'))
    case 'gripper:question_pending':
      return say(t('ck.act.answer'))
    case 'carried_part:not_modelled':
      return say(t('ck.act.payload'))
    case 'commands:idle':
    case 'commands:failed':
      return button('load', t('ck.act.load'), t('ck.act.loadTitle'))
    case 'commands:missing':
    case 'commands:not_configured':
      return say(t('ck.act.byHand'))
    default:
      return null
  }
}

/** What a task does, in the operator's own words where they were said: "alle grünen Würfel → Ablage links". */
function taskWords(plan: TaskPlanOut | null, anything: string, defaultPlace: string): { what: string; where: string } {
  const what = plan?.object_said || plan?.object || anything
  const place = plan?.place
  const where =
    place?.kind === 'camera' ? place.said || place.phrase || '—' : place?.kind === 'pose' ? place.pose_label || place.pose || defaultPlace : '—'
  return { what, where }
}

function RunHeader({ view, poses }: { view: RunView; poses: PosesOut | null }) {
  const t = useT(COCKPIT)
  const now = useNow(1000)
  const plan = view.plan
  let title: string
  let meta: string | null = null
  switch (view.kind) {
    case 'task': {
      const { what, where } = taskWords(plan, t('common.anything'), t('common.defaultPlace'))
      title = t('ck.run.task', { what, where })
      const back = plan?.return_to && plan.return_to !== 'home' ? plan.return_label || poseLabel(plan.return_to, poses) || plan.return_to : t('common.home')
      meta = t('ck.run.taskMeta', { scope: t(plan?.scope === 'until_empty' ? 'scope.until_empty' : 'scope.once'), back })
      break
    }
    case 'home':
      title = t('ck.run.home', { to: !view.to || view.to === 'home' ? t('common.home') : poseLabel(view.to, poses) || view.to })
      break
    case 'teach':
      title = t('ck.run.teach', { label: view.teach?.label || view.teach?.name || '—' })
      break
    case 'planner':
      title = t('ck.run.planner')
      break
    case 'pick':
      title = t('ck.run.pick', { what: view.prompt || t('common.anything') })
      break
    default:
      title = t('ck.chat.state.busy')
  }
  const elapsed = view.startedAt !== null ? t.fmt.duration(Math.max(0, now - view.startedAt)) : null
  return (
    <section className="ck-top ck-runhead" aria-label={t('chip.run')}>
      <strong className="ck-run-title">{headline(title, t.lang)}</strong>
      {view.current.part !== null && <span className="ck-run-part">{t('ck.run.part', { part: view.current.part })}</span>}
      {meta && <span className="ck-run-meta">{meta}</span>}
      <span className="ck-run-right mono">
        {view.kind && <span>{t.msg(runKindMsg(view.kind))}</span>}
        <span className="ck-run-phase">{t.msg(phaseMsg(view))}</span>
        {elapsed && <span>{elapsed}</span>}
      </span>
    </section>
  )
}

function StoppedHeader({ run, at }: { run: RunOut | null; at: { part: number | null; step: StepId | null } }) {
  const t = useT(COCKPIT)
  const { cell } = useCell()
  const record = cell?.recovery
  if (!record) return null
  const task = run?.plan ? taskWords(run.plan, t('common.anything'), t('common.defaultPlace')) : null
  return (
    <section className="ck-top ck-runhead stopped" aria-label={t('chip.run')}>
      <Icon name="alert" size={20} className="ck-alarm-icon" />
      <strong className="ck-run-title">{t.msg(stopMsg(record.stop_code))}</strong>
      {at.part !== null && at.step !== null && (
        <span className="ck-run-meta">{t('ck.run.stoppedAt', { part: at.part, at: t(`ck.at.${at.step}`) })}</span>
      )}
      {task && <span className="ck-run-meta">{headline(t('ck.run.task', task), t.lang)}</span>}
      <span className="ck-run-right mono">{t('ck.run.stoppedRun', { kind: runKindMsg(record.kind) })}</span>
      {cell.state !== 'connected' && (
        // The record outlives a Disconnect and a rebuild: the way back starts with connecting the cell again.
        <div className="ck-notice ck-top-row">
          <NotConnected halted={Boolean(cell.halted)} />
        </div>
      )}
    </section>
  )
}
