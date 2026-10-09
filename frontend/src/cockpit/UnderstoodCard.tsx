/**
 * The Understood card (OD 7, 8, 22; build plan 4.2): what was understood, editable, and Start.
 *
 * **Greifen**: the English phrase the detector is given, the operator's own words under it, how the detector would
 * route it, and "bitte prüfen" where the reader could not check it against the sentence. An empty phrase asks what to
 * pick; on a real cell "alles, was die Kamera sieht, auch Kistenwände" only with the operator's tick (Q7 A+).
 * **Welches**: the one part the sentence singled out ("the gray cube on top of the other one"), which the camera then
 * looks for alone; **Woher**: where the parts lie ("on the black mat"). Both English, both editable, both empty where
 * the sentence said nothing of it (2026-10-08).
 * **Ablegen**: the default place, a taught pose by its label, or "Kamera sucht: …" (a target the camera finds).
 * **Umfang**: Einmal | Bis leer. **Danach**: Home or a taught pose. Then the Advanced drawer.
 *
 * **Start is the confirmation** (item 12): one click, no second dialog, and its label names the first motion, the
 * countdown when one is due, and for a camera place what happens if the target is not found. It is off, saying why,
 * until the cell is ready and the card is complete. Without a reader (501, or the model not loaded) the card opens by
 * hand with the reason, every field editable. A refused Start is said right above Start, where the person is looking,
 * and brought into view: under the card's foot the chat could cut it off, and the click would read as ignored. Where
 * Enter starts a clean reading at once (the owner, 2026-10-08), this card is what Enter opens for every other one, and
 * for a start the server refused.
 *
 * The phrases take what the server takes, 200 characters each (`api/limits.ts`): a field stops typing there, and from
 * 80 % on it says how much it holds.
 *
 * **Regeln**, where the reading is a sort (the owner, 2026-10-09: "Grüne Teile in die gelbe Kiste, rote in die
 * blaue"): the first rule is Greifen and Ablegen above, each further one a row of its own, the kind of part → its
 * place (a target the camera finds or a taught pose, as Ablegen offers), with its "bitte prüfen" and a button that
 * removes it; "Regel hinzufügen" adds one, three at most beside the first. Every change of a row is recorded as
 * `rules`. A card of one rule looks as it always did.
 */

import { useId, useRef } from 'react'

import type { ApiError, CellFactsOut, PosesOut, RoutePreviewOut } from '../api/client'
import { MAX_FIELD_CHARS, MAX_MORE_RULES, showsCount } from '../api/limits'
import { ErrorBanner } from '../components/ui'
import { useT } from '../i18n'
import { noteMsg, refusalMsg } from '../i18n/codes'
import { Icon } from '../icons'
import AdvancedDrawer from './AdvancedDrawer'
import {
  PROBLEM_KEY,
  RIM_AIR_MM,
  draftProblems,
  newRule,
  placeChoiceWords,
  placesOf,
  ruleDoubt,
  type Draft,
  type DraftField,
  type DraftRule,
  type PlaceChoice,
} from './draft'
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

function placeValue(place: PlaceChoice): string {
  if (place.kind === 'pose') return `pose:${place.pose}`
  return place.kind
}

/** The place a select's value names; a target the camera finds keeps the phrase it had. */
function choiceOf(value: string, camera: { readonly phrase: string; readonly said: string | null } | null): PlaceChoice | null {
  if (value === 'default') return { kind: 'default' }
  if (value === 'camera') return { kind: 'camera', phrase: camera?.phrase ?? '', said: camera?.said ?? null }
  if (value.startsWith('pose:')) return { kind: 'pose', pose: value.slice(5) }
  return null
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
  const manual = draft.mode === 'manual'
  const title = t(manual ? 'ck.card.manual' : 'ck.card.understood')
  const camera = draft.place.kind === 'camera' ? draft.place : null
  // A sort's further rules (2026-10-09): the card shows them as rows of their own, and "anything" sorts nothing.
  const sort = draft.moreRules.length > 0
  // The put-back fallback is said wherever a rule puts its parts where the camera finds them.
  const finds = placesOf(draft).some((place) => place.kind === 'camera' && place.phrase.trim() !== '')
  // The card's notes: the reading's own and its first rule's, then each further rule's, said with its number.
  const notes = [
    ...draft.notes.map((note) => ({ id: note, text: t.msg(noteMsg(note)) })),
    ...draft.moreRules.flatMap((rule, index) =>
      rule.notes.map((note) => ({ id: `${index + 2}:${note}`, text: t.msg(ruleDoubt(index + 2, noteMsg(note))) })),
    ),
  ]
  // The title names the first motion; the lines under it what comes before it (the countdown) and what happens if a
  // camera's target is not found. Each its own line, so each reads as a sentence in both languages.
  const label = t('ck.start.label', { motion })
  const startRow = useRef<HTMLDivElement | null>(null)
  const refused = useBroughtIntoView<HTMLDivElement>(error)

  const setPlace = (value: string) => {
    const place = choiceOf(value, camera)
    if (place) update('place', { place })
  }
  const setRules = (moreRules: readonly DraftRule[]) => update('rules', { moreRules })
  const changeRule = (index: number, change: Partial<DraftRule>) =>
    setRules(draft.moreRules.map((rule, i) => (i === index ? { ...rule, ...change } : rule)))

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
              maxLength={MAX_FIELD_CHARS}
              placeholder={t('ck.pick.placeholder')}
              onChange={(e) => update('object', { object: e.target.value, objectVerified: true })}
            />
            <Count length={draft.object.length} />
            <RouteBadge route={draft.objectRoute} />
            {!manual && !draft.objectVerified && draft.object.trim() !== '' && <span className="ck-tag warn">{t('ck.pick.check')}</span>}
          </div>
          {draft.objectSaid && <small>{t('ck.pick.said', { said: draft.objectSaid })}</small>}
          {draft.object.trim() === '' && (
            <div className="ck-ask-pick">
              <p>{t('ck.pick.ask')}</p>
              {sort ? null : rehearsal ? (
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
          <label htmlFor={`${ids}-which`}>{t('ck.field.which')}</label>
        </dt>
        <dd>
          <div className="ck-field-row">
            <input
              id={`${ids}-which`}
              type="text"
              className="ck-input-phrase"
              value={draft.which}
              maxLength={MAX_FIELD_CHARS}
              placeholder={t('ck.which.placeholder')}
              onChange={(e) => update('which', { which: e.target.value })}
            />
            <Count length={draft.which.length} />
          </div>
          {draft.which.trim() !== '' && <small>{t('ck.which.hint')}</small>}
        </dd>

        <dt>
          <label htmlFor={`${ids}-source`}>{t('ck.field.source')}</label>
        </dt>
        <dd>
          <div className="ck-field-row">
            <input
              id={`${ids}-source`}
              type="text"
              className="ck-input-phrase"
              value={draft.source}
              maxLength={MAX_FIELD_CHARS}
              placeholder={t('ck.source.placeholder')}
              onChange={(e) => update('source', { source: e.target.value })}
            />
            <Count length={draft.source.length} />
          </div>
        </dd>

        <dt>
          <label htmlFor={`${ids}-place`}>{t('ck.field.place')}</label>
        </dt>
        <dd>
          <select id={`${ids}-place`} value={placeValue(draft.place)} onChange={(e) => setPlace(e.target.value)}>
            <PlaceOptions place={draft.place} poses={poses} />
          </select>
          {camera && (
            <>
              <div className="ck-field-row">
                <input
                  type="text"
                  className="ck-input-phrase"
                  aria-label={t('ck.place.target')}
                  value={camera.phrase}
                  maxLength={MAX_FIELD_CHARS}
                  placeholder={t('ck.place.phrase')}
                  onChange={(e) => update('place', { place: { kind: 'camera', phrase: e.target.value, said: camera.said }, placeVerified: true })}
                />
                <Count length={camera.phrase.length} />
                <RouteBadge route={draft.placeRoute} />
                {!manual && !draft.placeVerified && <span className="ck-tag warn">{t('ck.pick.check')}</span>}
              </div>
              {camera.said && <small>{t('ck.pick.said', { said: camera.said })}</small>}
              <small>{t('ck.place.air', { mm: draft.options.rimAirMm ?? RIM_AIR_MM })}</small>
            </>
          )}
          {draft.place.kind !== 'camera' && <small>{t('ck.place.poseHint')}</small>}
        </dd>

        {sort && (
          <>
            <dt id={`${ids}-rules`}>{t('ck.field.rules')}</dt>
            <dd>
              <ol className="ck-rules-edit" aria-labelledby={`${ids}-rules`}>
                <li className="ck-rule-row first">
                  <span className="ck-rule-n mono">1</span>
                  <span>
                    {t.msg({
                      key: 'list.rule',
                      params: { what: draft.objectSaid || draft.object.trim() || '—', where: placeChoiceWords(draft.place, poses) },
                    })}
                  </span>
                  <small>{t('ck.rules.first')}</small>
                </li>
                {draft.moreRules.map((rule, index) => (
                  <RuleRow
                    key={index}
                    rule={rule}
                    n={index + 2}
                    poses={poses}
                    manual={manual}
                    change={(change) => changeRule(index, change)}
                    remove={() => setRules(draft.moreRules.filter((_, i) => i !== index))}
                  />
                ))}
              </ol>
              <div className="ck-rules-foot">
                <button
                  type="button"
                  className="ghost ck-rule-add"
                  disabled={draft.moreRules.length >= MAX_MORE_RULES}
                  onClick={() => setRules([...draft.moreRules, newRule()])}
                >
                  <Icon name="plus" size={16} />
                  {t('ck.rules.add')}
                </button>
                {draft.moreRules.length >= MAX_MORE_RULES && <small>{t('ck.rules.max', { max: MAX_MORE_RULES })}</small>}
              </div>
            </dd>
          </>
        )}

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

      {notes.length > 0 && (
        <ul className="ck-notes">
          {notes.map((note) => (
            <li key={note.id}>{note.text}</li>
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
          {draft.reading.known
            ? t('ck.card.known')
            : draft.reading.latencyMs !== null && draft.reading.attempts !== null
              ? t(draft.reading.remembered ? 'ck.card.remembered' : 'ck.card.reading', {
                  ms: Math.round(draft.reading.latencyMs),
                  attempts: draft.reading.attempts,
                  model: draft.reading.modelId ?? '—',
                })
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
          {finds && <small>{t('ck.start.camera')}</small>}
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

/** The places a select offers: the default place, every taught pose by its label (a pose that is gone by its name),
 *  and a target the camera finds. */
function PlaceOptions({ place, poses }: { place: PlaceChoice; poses: PosesOut | null }) {
  const t = useT(COCKPIT)
  const taught = poses?.poses ?? []
  const defaultLabel = poses?.default_place ? (taught.find((p) => p.name === poses.default_place)?.label || poses.default_place) : null
  return (
    <>
      <option value="default">{defaultLabel ? t('ck.place.default', { label: defaultLabel }) : t('ck.place.defaultNone')}</option>
      {taught.map((pose) => (
        <option key={pose.name} value={`pose:${pose.name}`}>
          {pose.label || pose.name}
        </option>
      ))}
      {place.kind === 'pose' && !taught.some((p) => p.name === place.pose) && <option value={placeValue(place)}>{place.pose}</option>}
      <option value="camera">{t('ck.place.camera')}</option>
    </>
  )
}

/**
 * One further rule of a sort, compact: the kind of part, its place as Ablegen offers it (a target the camera finds
 * with its phrase), "bitte prüfen" where the reader did not find a word in the sentence, the operator's own words under
 * it, and the button that removes it.
 */
function RuleRow({
  rule,
  n,
  poses,
  manual,
  change,
  remove,
}: {
  rule: DraftRule
  /** The rule's number as the card says it: 2 for the first further rule. */
  n: number
  poses: PosesOut | null
  manual: boolean
  change(change: Partial<DraftRule>): void
  remove(): void
}) {
  const t = useT(COCKPIT)
  const camera = rule.place.kind === 'camera' ? rule.place : null
  const setPlace = (value: string) => {
    const place = choiceOf(value, camera)
    if (place) change({ place, placeRoute: null })
  }
  const said =
    rule.objectSaid && camera?.said
      ? t('ck.rules.saidBoth', { what: rule.objectSaid, where: camera.said })
      : rule.objectSaid || camera?.said
        ? t('ck.pick.said', { said: rule.objectSaid || camera?.said })
        : null
  return (
    <li className="ck-rule-row">
      <span className="ck-rule-n mono">{n}</span>
      <div className="ck-rule-body">
        <div className="ck-field-row">
          <input
            type="text"
            className="ck-input-phrase"
            aria-label={t('ck.rules.object', { n })}
            value={rule.object}
            maxLength={MAX_FIELD_CHARS}
            placeholder={t('ck.pick.placeholder')}
            onChange={(e) => change({ object: e.target.value, objectVerified: true, objectRoute: null })}
          />
          <Count length={rule.object.length} />
          <RouteBadge route={rule.objectRoute} />
          {!manual && !rule.objectVerified && rule.object.trim() !== '' && <span className="ck-tag warn">{t('ck.pick.check')}</span>}
        </div>
        <div className="ck-field-row">
          <span className="ck-rule-arrow" aria-hidden="true">
            →
          </span>
          <select aria-label={t('ck.rules.place', { n })} value={placeValue(rule.place)} onChange={(e) => setPlace(e.target.value)}>
            <PlaceOptions place={rule.place} poses={poses} />
          </select>
          {camera && (
            <>
              <input
                type="text"
                className="ck-input-phrase"
                aria-label={t('ck.rules.target', { n })}
                value={camera.phrase}
                maxLength={MAX_FIELD_CHARS}
                placeholder={t('ck.place.phrase')}
                onChange={(e) => change({ place: { kind: 'camera', phrase: e.target.value, said: camera.said }, placeVerified: true, placeRoute: null })}
              />
              <Count length={camera.phrase.length} />
              <RouteBadge route={rule.placeRoute} />
              {!manual && !rule.placeVerified && <span className="ck-tag warn">{t('ck.pick.check')}</span>}
            </>
          )}
        </div>
        {said && <small>{said}</small>}
      </div>
      <button type="button" className="ghost ck-rule-remove" aria-label={t('ck.rules.remove', { n })} title={t('ck.rules.remove', { n })} onClick={remove}>
        <Icon name="close" size={16} />
      </button>
    </li>
  )
}

/** How many of a phrase's characters are used, said once the field is 80 % full. */
function Count({ length }: { length: number }) {
  const t = useT(COCKPIT)
  if (!showsCount(length, MAX_FIELD_CHARS)) return null
  return <span className="ck-count">{t('ck.field.count', { n: length, max: MAX_FIELD_CHARS })}</span>
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
