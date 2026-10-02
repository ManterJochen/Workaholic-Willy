/**
 * The Understood card (OD 7, 8, 22; build plan 4.2): what was understood, editable, and Start.
 *
 * **Greifen**: the English phrase the detector is given, the operator's own words under it, how the detector would
 * route it, and "bitte prüfen" where the reader could not check it against the sentence. An empty phrase asks what to
 * pick; on a real cell "alles, was die Kamera sieht, auch Kistenwände" only with the operator's tick (Q7 A+).
 * **Ablegen**: the default place, a taught pose by its label, or "Kamera sucht: …" (a target the camera finds).
 * **Umfang**: Einmal | Bis leer. **Danach**: Home or a taught pose. Then the Advanced drawer.
 *
 * **Start is the confirmation** (item 12): one click, no second dialog, and its label names the first motion, the
 * countdown when one is due, and for a camera place what happens if the target is not found. It is off, saying why,
 * until the cell is ready and the card is complete. Without a reader (501, or the model not loaded) the card opens by
 * hand with the reason, every field editable. A refused Start is said right above Start, where the person is looking,
 * and brought into view: under the card's foot the chat could cut it off, and the click would read as ignored.
 */

import { useId, useRef } from 'react'

import type { ApiError, CellFactsOut, PosesOut, RoutePreviewOut } from '../api/client'
import { ErrorBanner } from '../components/ui'
import { useT } from '../i18n'
import { noteMsg, refusalMsg } from '../i18n/codes'
import { Icon } from '../icons'
import AdvancedDrawer from './AdvancedDrawer'
import { RIM_AIR_MM, draftProblems, type Draft, type DraftField, type DraftProblem, type PlaceChoice } from './draft'
import { useBroughtIntoView } from './hooks'
import { COCKPIT } from './i18n'

export interface UnderstoodCardProps {
  readonly draft: Draft
  readonly poses: PosesOut | null
  readonly facts: CellFactsOut | null
  readonly tech: boolean
  /** The first motion, in words ("der Roboter fährt zu Blick 1"). */
  readonly motion: string
  /** A 3 s hands-off countdown comes first (`CellOut.countdown_due`). */
  readonly countdown: boolean
  /** Why Start is off for the cell's sake (readiness, a run, a stop record); empty when the cell allows it. */
  readonly cellOff: readonly string[]
  readonly busy: 'reading' | 'starting' | 'loading' | null
  readonly error: ApiError | null
  update(field: DraftField, change: Partial<Draft>): void
  start(): void
  discard(): void
  /** "Laden", where the reader is not loaded. */
  load(): void
  /** Read the sentence again, after a load. */
  retry(): void
}

const PROBLEM_KEY: Record<DraftProblem, 'ck.start.needObject' | 'ck.start.needPhrase' | 'ck.place.noDefault' | 'ck.start.unknownPose'> = {
  needObject: 'ck.start.needObject',
  needPhrase: 'ck.start.needPhrase',
  noDefault: 'ck.place.noDefault',
  unknownPose: 'ck.start.unknownPose',
}

function placeValue(place: PlaceChoice): string {
  if (place.kind === 'pose') return `pose:${place.pose}`
  return place.kind
}

export default function UnderstoodCard(props: UnderstoodCardProps) {
  const { draft, poses, facts, tech, motion, countdown, cellOff, busy, error, update, start, discard, load, retry } = props
  const t = useT(COCKPIT)
  const ids = useId()
  const rehearsal = facts?.rehearsal === true
  const problems = draftProblems(draft, { rehearsal, poses })
  const off = [...cellOff, ...problems.map((problem) => t(PROBLEM_KEY[problem]))]
  const canStart = off.length === 0 && busy === null
  const taught = poses?.poses ?? []
  const defaultLabel = poses?.default_place ? (taught.find((p) => p.name === poses.default_place)?.label || poses.default_place) : null
  const manual = draft.mode === 'manual'
  const title = t(manual ? 'ck.card.manual' : 'ck.card.understood')
  const camera = draft.place.kind === 'camera' ? draft.place : null
  // The title names the first motion; the lines under it what comes before it (the countdown) and what happens if a
  // camera's target is not found. Each its own line, so each reads as a sentence in both languages.
  const label = t('ck.start.label', { motion })
  const startRow = useRef<HTMLDivElement | null>(null)
  const refused = useBroughtIntoView<HTMLDivElement>(error)

  const setPlace = (value: string) => {
    if (value === 'default') update('place', { place: { kind: 'default' } })
    else if (value === 'camera') update('place', { place: { kind: 'camera', phrase: camera?.phrase ?? '', said: camera?.said ?? null } })
    else if (value.startsWith('pose:')) update('place', { place: { kind: 'pose', pose: value.slice(5) } })
  }

  return (
    <section className="ck-card ck-understood hud-frame" aria-label={title}>
      <header className="ck-card-head">
        <Icon name={manual ? 'info' : 'check'} size={20} className="ck-ok-icon" />
        <h3>{title}</h3>
        {draft.said.text && <span className="ck-card-meta">{t('ck.card.said', { text: draft.said.text })}</span>}
      </header>

      {manual && draft.manualCode && (
        <p className="ck-say warn">{t('ck.card.manualWhy', { why: refusalMsg(draft.manualCode) })}</p>
      )}
      {manual && tech && draft.manualMessage && <p className="ck-human">{draft.manualMessage}</p>}
      {manual && draft.manualCode === 'vlm_not_loaded' && (
        <div className="ck-card-actions">
          <button type="button" className="big" disabled={busy !== null} onClick={load} title={t('ck.act.loadTitle')}>
            {t('ck.act.load')}
          </button>
          <button type="button" className="ghost big" disabled={busy !== null} onClick={retry}>
            {t('ck.card.retry')}
          </button>
        </div>
      )}

      <dl className="ck-fields">
        <dt>
          <label htmlFor={`${ids}-pick`}>{t('ck.field.pick')}</label>
        </dt>
        <dd>
          <div className="ck-field-row">
            <input
              id={`${ids}-pick`}
              type="text"
              className="ck-input-phrase"
              value={draft.object}
              placeholder={t('ck.pick.placeholder')}
              onChange={(e) => update('object', { object: e.target.value, objectVerified: true })}
            />
            <RouteBadge route={draft.objectRoute} />
            {!manual && !draft.objectVerified && draft.object.trim() !== '' && <span className="ck-tag warn">{t('ck.pick.check')}</span>}
          </div>
          {draft.objectSaid && <small>{t('ck.pick.said', { said: draft.objectSaid })}</small>}
          {draft.object.trim() === '' && (
            <div className="ck-ask-pick">
              <p>{t('ck.pick.ask')}</p>
              {rehearsal ? (
                <small>{t('ck.pick.anythingProbe')}</small>
              ) : (
                <label className="ck-check-line">
                  <input
                    type="checkbox"
                    checked={draft.pickAnything}
                    onChange={(e) => update('pick_anything', { pickAnything: e.target.checked })}
                  />
                  {t('ck.pick.anything')}
                </label>
              )}
            </div>
          )}
        </dd>

        <dt>
          <label htmlFor={`${ids}-place`}>{t('ck.field.place')}</label>
        </dt>
        <dd>
          <select id={`${ids}-place`} value={placeValue(draft.place)} onChange={(e) => setPlace(e.target.value)}>
            <option value="default">{defaultLabel ? t('ck.place.default', { label: defaultLabel }) : t('ck.place.defaultNone')}</option>
            {taught.map((pose) => (
              <option key={pose.name} value={`pose:${pose.name}`}>
                {pose.label || pose.name}
              </option>
            ))}
            {draft.place.kind === 'pose' && !taught.some((p) => p.name === (draft.place as { pose: string }).pose) && (
              <option value={placeValue(draft.place)}>{draft.place.pose}</option>
            )}
            <option value="camera">{t('ck.place.camera')}</option>
          </select>
          {camera && (
            <>
              <div className="ck-field-row">
                <input
                  type="text"
                  className="ck-input-phrase"
                  aria-label={t('ck.place.target')}
                  value={camera.phrase}
                  placeholder={t('ck.place.phrase')}
                  onChange={(e) => update('place', { place: { kind: 'camera', phrase: e.target.value, said: camera.said }, placeVerified: true })}
                />
                <RouteBadge route={draft.placeRoute} />
                {!manual && !draft.placeVerified && <span className="ck-tag warn">{t('ck.pick.check')}</span>}
              </div>
              {camera.said && <small>{t('ck.pick.said', { said: camera.said })}</small>}
              <small>{t('ck.place.air', { mm: draft.options.rimAirMm ?? RIM_AIR_MM })}</small>
            </>
          )}
          {draft.place.kind !== 'camera' && <small>{t('ck.place.poseHint')}</small>}
        </dd>

        <dt id={`${ids}-scope`}>{t('ck.field.scope')}</dt>
        <dd>
          <div className="ck-seg" role="group" aria-labelledby={`${ids}-scope`}>
            {(['once', 'until_empty'] as const).map((scope) => (
              <button
                key={scope}
                type="button"
                aria-pressed={draft.scope === scope}
                onClick={() => draft.scope !== scope && update('scope', { scope })}
              >
                {t(`scope.${scope}`)}
              </button>
            ))}
          </div>
        </dd>

        <dt>
          <label htmlFor={`${ids}-after`}>{t('ck.field.after')}</label>
        </dt>
        <dd>
          <select id={`${ids}-after`} value={draft.returnTo} onChange={(e) => update('return_to', { returnTo: e.target.value })}>
            <option value="home">{t('common.home')}</option>
            {taught.map((pose) => (
              <option key={pose.name} value={pose.name}>
                {pose.label || pose.name}
              </option>
            ))}
            {draft.returnTo !== 'home' && !taught.some((p) => p.name === draft.returnTo) && (
              <option value={draft.returnTo}>{draft.returnTo}</option>
            )}
          </select>
        </dd>
      </dl>

      {draft.notes.length > 0 && (
        <ul className="ck-notes">
          {draft.notes.map((note) => (
            <li key={note}>{t.msg(noteMsg(note))}</li>
          ))}
        </ul>
      )}

      <AdvancedDrawer
        draft={draft}
        facts={facts}
        tech={tech}
        change={(options) => update('options', { options: { ...draft.options, ...options } })}
        // The drawer opens downwards: Start, under it, is brought back into the chat's view.
        opened={() => startRow.current?.scrollIntoView?.({ block: 'nearest', behavior: 'smooth' })}
      />

      {tech && draft.reading && (
        <div className="ck-human">
          {draft.reading.latencyMs !== null && draft.reading.attempts !== null
            ? t('ck.card.reading', { ms: Math.round(draft.reading.latencyMs), attempts: draft.reading.attempts, model: draft.reading.modelId ?? '—' })
            : null}
          {draft.reading.raw && (
            <details className="ck-data">
              <summary>{t('ck.card.raw')}</summary>
              <pre>{draft.reading.raw}</pre>
            </details>
          )}
        </div>
      )}

      {/* What the server refused, right above Start, and brought into view. */}
      {error && (
        <div className="ck-refusal" ref={refused}>
          <ErrorBanner error={error} />
        </div>
      )}

      <div className="ck-start" ref={startRow}>
        <button type="button" className="primary big ck-go ck-startbtn" disabled={!canStart} onClick={start}>
          <span className="ck-go-title">
            <Icon name="play" size={16} />
            {busy === 'starting' ? t('ck.start.starting') : label}
          </span>
          {countdown && <small>{t('ck.start.countdown')}</small>}
          {camera && camera.phrase.trim() !== '' && <small>{t('ck.start.camera')}</small>}
        </button>
        <button type="button" className="ghost big" onClick={discard} disabled={busy === 'starting'}>
          {t('ck.card.discard')}
        </button>
      </div>
      {off.length > 0 && (
        <div className="ck-off">
          <span>{t('ck.start.off')}</span>
          <ul>
            {off.map((why) => (
              <li key={why}>{why}</li>
            ))}
          </ul>
        </div>
      )}
    </section>
  )
}

/** How the detector would route a phrase: by the phrase grounder, or by the VLM; and whether it can run here. */
function RouteBadge({ route }: { route: RoutePreviewOut | null }) {
  const t = useT(COCKPIT)
  if (!route) return null
  const vlm = route.route === 'vlm'
  const blocked = !route.runnable
  return (
    <span className={`ck-tag ${blocked ? 'warn' : 'acc'}`} title={route.blocked_reason || route.description}>
      {t(vlm ? 'ck.route.vlm' : 'ck.route.simple')}
      {blocked && <span className="ck-tag-note"> · {t(vlm ? 'ck.route.needsVlm' : 'ck.route.blocked')}</span>}
    </span>
  )
}
