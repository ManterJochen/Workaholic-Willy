/**
 * One line Willy says: the localised message, coloured by its tone. In the tech view the backend's own English
 * sentence sits under it, and the event's data can be opened (build plan 4.2: "In the tech view the backend `human`
 * sentence sits under each line, and `data` can be expanded").
 */

import { useT } from '../i18n'
import type { ChatLine } from '../model/runModel'
import { COCKPIT } from './i18n'

export default function ChatLineView({
  line,
  tech,
  step = false,
  willy = false,
  follows = false,
}: {
  line: ChatLine
  tech: boolean
  step?: boolean
  /** Drawn as Willy's own words (the run's start and end, his question), labelled, not as a step line. */
  willy?: boolean
  /** It follows another line of Willy's: his label is not said again. */
  follows?: boolean
}) {
  const t = useT(COCKPIT)
  const data = Object.keys(line.data).length > 0
  return (
    <div
      className={`${willy ? 'ck-bubble willy' : 'ck-line'} ${line.tone}${step ? ' step' : ''}${willy && follows ? ' follows' : ''}`}
      data-type={line.type || undefined}
    >
      {willy && !follows && <span className="ck-who">{t('ck.who.willy')}</span>}
      <span className="ck-line-text">{t.msg(line.msg)}</span>
      {tech && line.human && <span className="ck-human">{line.human}</span>}
      {tech && data && (
        <details className="ck-data">
          <summary>
            {line.type || t('ck.chat.data')}
            {line.seq ? <span className="ck-seq"> #{line.seq}</span> : null}
          </summary>
          <pre>{JSON.stringify(line.data, null, 2)}</pre>
        </details>
      )}
    </div>
  )
}
