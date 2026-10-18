/**
 * Settings (`/settings`, formerly `/config`): the person's preferences for this browser (language, theme, view, voice
 * output, the talk key), how a typed task starts for the whole cell ("Auftrag": Enter or the card, the mode, the
 * defaults, critical parts; the owner, 2026-10-08, the cell's own since 2026-10-17, written into its own layer), and the
 * config explained and its measured values written (the Config screen, unchanged in what it may write: only the keys the
 * server declares writable).
 */

import { ScreenHead } from '../components/ui'
import { useT } from '../i18n'
import Config from './Config'
import { SCREENS } from './i18n'
import Preferences from './Preferences'
import TaskPrefs from './TaskPrefs'
import './screens.css'

export default function Settings() {
  const t = useT(SCREENS)
  return (
    <div className="page">
      <ScreenHead title={t('nav.settings')} lede={t('st.page.lede')} />
      <Preferences />
      <TaskPrefs />
      <Config />
    </div>
  )
}
