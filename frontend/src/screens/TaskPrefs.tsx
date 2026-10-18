/**
 * How a typed or spoken task starts, for the cell (the owner, 2026-10-08): "Enter startet sofort", the mode set here
 * beforehand, what and where taken from the sentence, and these settings only where the sentence says nothing.
 *
 * They are the cell's, for every browser (the owner, 2026-10-16: `runtime.task` in `app/runtime.yaml`), read from the
 * server and written by each choice into the cell's own layer, the way a taught pose is written (2026-10-17;
 * `model/taskSettings.ts`). Six rows:
 *
 * * **Start**: "Enter startet sofort" (the default: a clean reading on a ready cell starts on Enter, any doubt opens
 *   the card) or "Erst die Karte" (Enter only reads; Start on the card starts, as before);
 * * **Modus**: Einmal or Bis leer, the mode of every task Enter starts; "alle" in the sentence does not change it;
 * * **Blicke**: how each pick looks (the owner, 2026-10-08 night: "Multi-View" and "Alle Posen" one choice): the first
 *   look only, when needed (the default) or every look; a card may change it for its task;
 * * **Ablegen ohne Angabe**: the cell's default place, a taught pose, or a target the camera finds, written once its
 *   phrase is typed and the field is left;
 * * **Kein Teil genannt**: the card asks, or comes ticked for anything the camera sees (Q7 A+); such a task never
 *   starts on Enter;
 * * **Kritische Teile**: the cell's switch the Advanced drawer starts from, read by the cell when it is built; where the
 *   built cell still runs the other value, the row says so.
 *
 * One change at a time: the rows wait for the server's answer, and a refusal (`run_active`, `no_layer`,
 * `invalid_value`, `unknown_pose`) is said under them in the reader's words, the settings as they were. Where the chain
 * has no layer of the cell's own, the rows are off and say why; the file a change goes to is said under them. The
 * cockpit says these under its command box, so the person who presses Enter sees what Enter does. Nothing here moves
 * anything or starts a task; the taught poses are only read, to offer them.
 */

import { useEffect, useId, useState } from 'react'

import { api, type ApiError, type PosesOut, type TaskSettingsIn } from '../api/client'
import { MAX_FIELD_CHARS } from '../api/limits'
import { ErrorBanner, Loading, Panel, Segmented } from '../components/ui'
import { useT } from '../i18n'
import { refusalMsg } from '../i18n/codes'
import { usePrefs } from '../model/prefs'
import { LOOKS_CHOICES, useTaskSettings } from '../model/taskSettings'
import { useCell } from '../model/useCell'
import { fileName } from '../setup/format'
import { SCREENS } from './i18n'

export default function TaskPrefs() {
  const t = useT(SCREENS)
  const ids = useId()
  const tech = usePrefs().view === 'tech'
  const { settings, error, change } = useTaskSettings()
  const { facts } = useCell()
  const [poses, setPoses] = useState<PosesOut | null>(null)
  const [busy, setBusy] = useState(false)
  const [refused, setRefused] = useState<ApiError | null>(null)
  /** The place as the select shows it where it is not the cell's yet: a camera place waits for its phrase. */
  const [choice, setChoice] = useState<string | null>(null)
  /** The phrase as typed, until it is written. */
  const [phrase, setPhrase] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    api
      .poses()
      .then((read) => {
        if (!cancelled) setPoses(read)
      })
      .catch(() => undefined)
    return () => {
      cancelled = true
    }
  }, [])

  if (!settings) {
    return (
      <Panel title={t('st.task.title')}>
        <p className="panel-lede">{t('st.task.lede')}</p>
        {error ? <ErrorBanner error={error} /> : <Loading what={t('st.task.what')} />}
      </Panel>
    )
  }

  /** One change, written into the cell's own layer; the rows wait for the answer. */
  const write = async (values: TaskSettingsIn): Promise<boolean> => {
    setBusy(true)
    setRefused(null)
    const failed = await change(values)
    setBusy(false)
    setRefused(failed)
    return failed === null
  }

  const place = settings.place
  const shown = choice ?? (place.kind === 'pose' ? `pose:${place.pose}` : place.kind)
  const typed = phrase ?? (place.kind === 'camera' ? place.phrase : '')
  const taught = poses?.poses ?? []
  const stored = place.kind === 'pose' ? place.pose : null

  const choose = (value: string) => {
    setChoice(value)
    setPhrase(null)
    if (value === 'camera') return // written once its phrase is typed and the field is left
    void write({ place: value.startsWith('pose:') ? { kind: 'pose', pose: value.slice(5) } : { kind: 'default' } }).then(
      () => setChoice(null),
    )
  }

  const keepPhrase = () => {
    const words = typed.trim()
    if (words === '' || (place.kind === 'camera' && words === place.phrase)) return
    void write({ place: { kind: 'camera', phrase: words } }).then((written) => {
      if (!written) return
      setChoice(null)
      setPhrase(null)
    })
  }

  const yesNo = (on: boolean) => t(on ? 'st.task.critical.yes' : 'st.task.critical.no')
  const built = facts?.push ?? null

  return (
    <Panel title={t('st.task.title')}>
      <p className="panel-lede">{t('st.task.lede')}</p>
      <fieldset className="st-rows" disabled={busy || settings.why_not !== ''}>
        <div className="st-row">
          <span className="st-name">{t('st.task.start')}</span>
          <Segmented
            label={t('st.task.start')}
            value={settings.start}
            options={[
              { value: 'enter', label: t('st.task.start.enter') },
              { value: 'card', label: t('st.task.start.card') },
            ]}
            onChange={(start) => void write({ start })}
          />
          <span className="st-hint">{t('st.task.start.hint')}</span>
        </div>

        <div className="st-row">
          <span className="st-name">{t('st.task.scope')}</span>
          <Segmented
            label={t('st.task.scope')}
            value={settings.scope}
            options={[
              { value: 'once', label: t('scope.once') },
              { value: 'until_empty', label: t('scope.until_empty') },
            ]}
            onChange={(scope) => void write({ scope })}
          />
          <span className="st-hint">{t('st.task.scope.hint')}</span>
        </div>

        <div className="st-row">
          <span className="st-name">{t('st.task.looks')}</span>
          <Segmented
            label={t('st.task.looks')}
            value={settings.looks}
            options={LOOKS_CHOICES.map((looks) => ({ value: looks, label: t(`looks.${looks}`) }))}
            onChange={(looks) => void write({ looks })}
          />
          <span className="st-hint">{t('st.task.looks.hint')}</span>
        </div>

        <div className="st-row">
          <label className="st-name" htmlFor={`${ids}-place`}>
            {t('st.task.place')}
          </label>
          <select id={`${ids}-place`} value={shown} onChange={(e) => choose(e.target.value)}>
            <option value="default">{t('st.task.place.default')}</option>
            {taught.map((pose) => (
              <option key={pose.name} value={`pose:${pose.name}`}>
                {pose.label || pose.name}
              </option>
            ))}
            {stored !== null && !taught.some((pose) => pose.name === stored) && <option value={`pose:${stored}`}>{stored}</option>}
            <option value="camera">{t('st.task.place.camera')}</option>
          </select>
          <span className="st-hint">{t('st.task.place.hint')}</span>
          {shown === 'camera' && (
            <input
              type="text"
              className="st-phrase"
              aria-label={t('st.task.place.phrase')}
              value={typed}
              maxLength={MAX_FIELD_CHARS}
              placeholder={t('st.task.place.placeholder')}
              onChange={(e) => setPhrase(e.target.value)}
              onBlur={keepPhrase}
              onKeyDown={(e) => {
                if (e.key === 'Enter') keepPhrase()
              }}
            />
          )}
        </div>

        <div className="st-row">
          <span className="st-name">{t('st.task.anything')}</span>
          <Segmented
            label={t('st.task.anything')}
            value={settings.anything ? 'anything' : 'ask'}
            options={[
              { value: 'ask', label: t('st.task.anything.ask') },
              { value: 'anything', label: t('st.task.anything.all') },
            ]}
            onChange={(value) => void write({ anything: value === 'anything' })}
          />
          <span className="st-hint">{t('st.task.anything.hint')}</span>
        </div>

        <div className="st-row">
          <span className="st-name">{t('st.task.critical')}</span>
          <Segmented
            label={t('st.task.critical')}
            value={settings.critical_parts ? 'yes' : 'no'}
            options={[
              { value: 'no', label: yesNo(false) },
              { value: 'yes', label: yesNo(true) },
            ]}
            onChange={(value) => void write({ critical_parts: value === 'yes' })}
          />
          <span className="st-hint">{t('st.task.critical.hint')}</span>
          {built !== null && built.critical_parts !== settings.critical_parts && (
            <span className="st-refused caution" role="status">
              {t('st.task.critical.built', { state: yesNo(built.critical_parts) })}
            </span>
          )}
        </div>
      </fieldset>

      {refused && <ErrorBanner error={refused} />}
      <p className="small dim" title={settings.target_file ?? undefined}>
        {settings.why_not
          ? t.msg(refusalMsg(settings.why_not_code || 'no_layer'))
          : settings.target_file
            ? t('st.task.target', { file: tech ? settings.target_file : fileName(settings.target_file) })
            : null}
      </p>
      {tech && settings.why_not && <p className="small mono dim">{settings.why_not}</p>}
    </Panel>
  )
}
