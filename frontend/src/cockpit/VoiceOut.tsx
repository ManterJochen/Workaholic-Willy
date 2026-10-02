/**
 * Voice output for the run on screen (OD 18): renders nothing, says the key moments aloud while the person switched
 * voice output on (`announce.ts`). The cockpit mounts it, and only the cockpit: on the Setup page the teach dialog says
 * its own time warning (`setup/TeachDialog.tsx`), so a second mount in the shell would say that warning twice. Two
 * mounts of this one never say a line twice.
 */

import { useT } from '../i18n'
import { usePrefs } from '../model/prefs'
import { useRun } from '../model/useRun'
import { useAnnouncer } from './announce'
import { COCKPIT } from './i18n'

export default function VoiceOut() {
  const t = useT(COCKPIT)
  const { view } = useRun()
  const { voiceOut } = usePrefs()
  useAnnouncer(view.chat, voiceOut, t)
  return null
}
