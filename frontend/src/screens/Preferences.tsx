/**
 * The person's preferences, for this browser (build plan 4.3): language, theme, demo or tech view, voice output, and
 * the talk key a foot switch sends.
 *
 * Every switch writes the same key the rest of the console reads (`willy.lang`, `willy.theme`, `willy.view`,
 * `willy.voiceOut`, `willy.talkKey`), so the top bar, the cockpit and the audience window agree with this page. Voice
 * output is off unless switched on (OD 18); "Stimme testen" says one sentence in the reader's language so a person can
 * hear which voice the cell PC has. The talk key is only ever a key typing never uses (F1 to F24, Pause, Scroll Lock:
 * `talkKey.ts`); any other key pressed while it is being set is refused by its name, and capture goes on. Nothing here
 * moves anything.
 */

import { useEffect, useState } from 'react'

import { Panel, Segmented } from '../components/ui'
import { useLang, useT } from '../i18n'
import { usePrefs } from '../model/prefs'
import { SCREENS } from './i18n'
import { DEFAULT_TALK_KEY, isTalkKey, readTalkKey, writeTalkKey } from './talkKey'

function speech(): SpeechSynthesis | null {
  try {
    return typeof window !== 'undefined' && 'speechSynthesis' in window ? window.speechSynthesis : null
  } catch {
    return null
  }
}

export default function Preferences() {
  const t = useT(SCREENS)
  const { lang, setLang } = useLang()
  const prefs = usePrefs()
  const [talkKey, setTalkKey] = useState(readTalkKey)
  const [capturing, setCapturing] = useState(false)
  /** The last key refused while capturing, as `KeyboardEvent.key` gave it; `null` while none was. */
  const [refused, setRefused] = useState<string | null>(null)
  const synth = speech()
  const [voices, setVoices] = useState<number | null>(null)

  // The installed voices for the language: read now, and again when the browser finishes loading them.
  useEffect(() => {
    if (!synth) return
    const count = () => {
      try {
        setVoices(synth.getVoices().filter((voice) => voice.lang.toLowerCase().startsWith(lang)).length)
      } catch {
        setVoices(null)
      }
    }
    count()
    synth.addEventListener?.('voiceschanged', count)
    return () => synth.removeEventListener?.('voiceschanged', count)
  }, [synth, lang])

  // Capture the next key pressed anywhere: Escape cancels, a key typing uses is refused by its name and capture goes on.
  useEffect(() => {
    if (!capturing) return
    const down = (event: KeyboardEvent) => {
      event.preventDefault()
      if (event.key === 'Escape') {
        setCapturing(false)
        setRefused(null)
        return
      }
      if (!isTalkKey(event.key)) {
        setRefused(event.key)
        return
      }
      writeTalkKey(event.key)
      setTalkKey(event.key)
      setRefused(null)
      setCapturing(false)
    }
    window.addEventListener('keydown', down)
    return () => window.removeEventListener('keydown', down)
  }, [capturing])

  // A key refused now, or a stored one that may not be the talk key (stored by hand, or by an older console): said by
  // its name, the space bar by the word for it.
  const shownRefusal = refused ?? (isTalkKey(talkKey) ? null : talkKey)
  const refusedName =
    shownRefusal === null ? '' : shownRefusal === ' ' || shownRefusal === 'Spacebar' ? t('st.talk.space') : shownRefusal

  const testVoice = () => {
    if (!synth) return
    try {
      synth.cancel()
      const utterance = new SpeechSynthesisUtterance(t('st.voice.sample'))
      utterance.lang = lang === 'de' ? 'de-DE' : 'en-GB'
      synth.speak(utterance)
    } catch {
      /* no speech in this browser after all: the line below says so */
    }
  }

  const langName = lang === 'de' ? 'Deutsch' : 'English'

  return (
    <Panel title={t('st.title')}>
      <p className="panel-lede">{t('st.lede')}</p>
      <div className="st-rows">
        <div className="st-row">
          <span className="st-name">{t('lang.label')}</span>
          <Segmented
            label={t('lang.label')}
            value={lang}
            options={[
              { value: 'de', label: t('lang.de'), title: 'Deutsch' },
              { value: 'en', label: t('lang.en'), title: 'English' },
            ]}
            onChange={setLang}
          />
          <span className="st-hint">{t('st.lang.hint')}</span>
        </div>

        <div className="st-row">
          <span className="st-name">{t('theme.label')}</span>
          <Segmented
            label={t('theme.label')}
            value={prefs.theme}
            options={[
              { value: 'dark', label: t('theme.dark') },
              { value: 'light', label: t('theme.light') },
            ]}
            onChange={prefs.setTheme}
          />
          <span className="st-hint">{t('st.theme.hint')}</span>
        </div>

        <div className="st-row">
          <span className="st-name">{t('view.label')}</span>
          <Segmented
            label={t('view.label')}
            value={prefs.view}
            options={[
              { value: 'demo', label: t('view.demo') },
              { value: 'tech', label: t('view.tech') },
            ]}
            onChange={prefs.setView}
          />
          <span className="st-hint">{t('st.view.hint')}</span>
        </div>

        <div className="st-row">
          <span className="st-name">{t('st.voice')}</span>
          <Segmented
            label={t('st.voice')}
            value={prefs.voiceOut ? 'on' : 'off'}
            options={[
              { value: 'off', label: t('st.voice.off') },
              { value: 'on', label: t('st.voice.on') },
            ]}
            onChange={(value) => prefs.setVoiceOut(value === 'on')}
          />
          <span className="st-hint">
            {t('voice.hint')}{' '}
            {!synth
              ? t('st.voice.none')
              : voices === 0
                ? t('st.voice.noVoice', { lang: langName })
                : voices !== null
                  ? t('st.voice.voices', { n: voices, lang: langName })
                  : null}
          </span>
          <button type="button" className="ghost big st-test" disabled={!synth} onClick={testVoice}>
            {t('st.voice.test')}
          </button>
        </div>

        <div className="st-row">
          <span className="st-name">{t('st.talk')}</span>
          <span className="st-key mono">{t('st.talk.current', { key: talkKey })}</span>
          <span className="st-hint">
            {t('st.talk.hint')} {t('st.talk.applies')}
          </span>
          <div className="actions st-talk-actions">
            <button
              type="button"
              className={capturing ? 'primary big' : 'big'}
              aria-pressed={capturing}
              onClick={() => {
                setRefused(null)
                setCapturing(!capturing)
              }}
            >
              {capturing ? t('st.talk.press') : t('st.talk.set')}
            </button>
            {talkKey !== DEFAULT_TALK_KEY && (
              <button
                type="button"
                className="ghost big"
                onClick={() => {
                  writeTalkKey(null)
                  setTalkKey(DEFAULT_TALK_KEY)
                }}
              >
                {t('st.talk.reset', { key: DEFAULT_TALK_KEY })}
              </button>
            )}
          </div>
          {shownRefusal !== null && (
            <span className="st-refused caution" role="status">
              {t('st.talk.refused', { key: refusedName })}
            </span>
          )}
        </div>
      </div>
    </Panel>
  )
}
