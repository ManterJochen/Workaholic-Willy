/**
 * How a typed or spoken task starts, for this browser (the owner, 2026-10-08): "Enter startet sofort", the mode set
 * here beforehand, what and where taken from the sentence, and these settings only where the sentence says nothing.
 *
 * Five rows, each written at once to the key the cockpit reads (`model/prefs.ts`):
 *
 * * **Start**: "Enter startet sofort" (the default: a clean reading on a ready cell starts on Enter, any doubt opens
 *   the card) or "Erst die Karte" (Enter only reads; Start on the card starts, as before);
 * * **Modus**: Einmal or Bis leer, the mode of every task Enter starts; "alle" in the sentence does not change it;
 * * **Blicke**: how each pick looks (the owner, 2026-10-08 night: "Multi-View" and "Alle Posen" one choice): the first
 *   look only, when needed (the default) or every look; a card may change it for its task;
 * * **Ablegen ohne Angabe**: the cell's default place, a taught pose, or a target the camera finds;
 * * **Kein Teil genannt**: the card asks, or comes ticked for anything the camera sees (Q7 A+); such a task never
 *   starts on Enter.
 *
 * The cockpit says these under its command box, so the person who presses Enter sees what Enter does. Nothing here
 * moves anything; the taught poses are only read, to offer them.
 */

import { useEffect, useId, useState } from 'react'

import { api, type PosesOut } from '../api/client'
import { MAX_FIELD_CHARS } from '../api/limits'
import { Panel, Segmented } from '../components/ui'
import { useT } from '../i18n'
import { LOOKS_CHOICES, usePrefs } from '../model/prefs'
import { SCREENS } from './i18n'

export default function TaskPrefs() {
  const t = useT(SCREENS)
  const ids = useId()
  const { task, setTask } = usePrefs()
  const [poses, setPoses] = useState<PosesOut | null>(null)
  /** The place's choice as the select shows it: a camera place waits for its phrase before it is one. */
  const [choice, setChoice] = useState(() => (task.place.kind === 'pose' ? `pose:${task.place.pose}` : task.place.kind))
  const [phrase, setPhrase] = useState(task.place.kind === 'camera' ? task.place.phrase : '')

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

  const taught = poses?.poses ?? []
  const stored = task.place.kind === 'pose' ? task.place.pose : null

  const choose = (value: string) => {
    setChoice(value)
    if (value === 'default') setTask({ place: { kind: 'default' } })
    else if (value.startsWith('pose:')) setTask({ place: { kind: 'pose', pose: value.slice(5) } })
    else if (value === 'camera' && phrase.trim() !== '') setTask({ place: { kind: 'camera', phrase: phrase.trim() } })
  }

  const type = (value: string) => {
    setPhrase(value)
    setTask({ place: value.trim() !== '' ? { kind: 'camera', phrase: value.trim() } : { kind: 'default' } })
  }

  return (
    <Panel title={t('st.task.title')}>
      <p className="panel-lede">{t('st.task.lede')}</p>
      <div className="st-rows">
        <div className="st-row">
          <span className="st-name">{t('st.task.start')}</span>
          <Segmented
            label={t('st.task.start')}
            value={task.start}
            options={[
              { value: 'enter', label: t('st.task.start.enter') },
              { value: 'card', label: t('st.task.start.card') },
            ]}
            onChange={(start) => setTask({ start })}
          />
          <span className="st-hint">{t('st.task.start.hint')}</span>
        </div>

        <div className="st-row">
          <span className="st-name">{t('st.task.scope')}</span>
          <Segmented
            label={t('st.task.scope')}
            value={task.scope}
            options={[
              { value: 'once', label: t('scope.once') },
              { value: 'until_empty', label: t('scope.until_empty') },
            ]}
            onChange={(scope) => setTask({ scope })}
          />
          <span className="st-hint">{t('st.task.scope.hint')}</span>
        </div>

        <div className="st-row">
          <span className="st-name">{t('st.task.looks')}</span>
          <Segmented
            label={t('st.task.looks')}
            value={task.looks}
            options={LOOKS_CHOICES.map((looks) => ({ value: looks, label: t(`looks.${looks}`) }))}
            onChange={(looks) => setTask({ looks })}
          />
          <span className="st-hint">{t('st.task.looks.hint')}</span>
        </div>

        <div className="st-row">
          <label className="st-name" htmlFor={`${ids}-place`}>
            {t('st.task.place')}
          </label>
          <select id={`${ids}-place`} value={choice} onChange={(e) => choose(e.target.value)}>
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
          {choice === 'camera' && (
            <input
              type="text"
              className="st-phrase"
              aria-label={t('st.task.place.phrase')}
              value={phrase}
              maxLength={MAX_FIELD_CHARS}
              placeholder={t('st.task.place.placeholder')}
              onChange={(e) => type(e.target.value)}
            />
          )}
        </div>

        <div className="st-row">
          <span className="st-name">{t('st.task.anything')}</span>
          <Segmented
            label={t('st.task.anything')}
            value={task.anything ? 'anything' : 'ask'}
            options={[
              { value: 'ask', label: t('st.task.anything.ask') },
              { value: 'anything', label: t('st.task.anything.all') },
            ]}
            onChange={(value) => setTask({ anything: value === 'anything' })}
          />
          <span className="st-hint">{t('st.task.anything.hint')}</span>
        </div>
      </div>
    </Panel>
  )
}
