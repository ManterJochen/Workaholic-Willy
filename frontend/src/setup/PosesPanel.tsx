/**
 * The poses (build plan 1.8, OD 15): Home, read-only from the config; every named pose by its label, with the verdict
 * of the screen it passed (an unscreened or refused pose is never written, so a taught pose is clear, or clear with a
 * short straight start); the default place, chosen with one radio; and "Pose einlernen", or the reason it cannot be
 * taught now ("Planer startet noch", "Dieser Arm lässt sich nicht von Hand führen.").
 *
 * The file a taught pose is written to is shown here, before anything is freed (Q10: the cell's own git-ignored layer).
 * Choosing the default place writes that layer through the pose door and moves nothing; teaching opens the dialog,
 * which frees the arm only at its own button.
 *
 * The list is read again whenever the cell changes in a way that decides whether a pose can be taught now (connected,
 * a run, the planner, a jaws question, a latch, a stop), and after every teach.
 */

import { useId, useState } from 'react'

import { ApiError, api, type PoseOut, type PosesOut } from '../api/client'
import { ErrorBanner, Loading } from '../components/ui'
import { useT } from '../i18n'
import { refusalMsg } from '../i18n/codes'
import { useAsync } from '../lib/useAsync'
import { usePrefs } from '../model/prefs'
import { useCell } from '../model/useCell'
import { fileName, jointsText } from './format'
import { SETUP, type SetupT } from './i18n'
import TeachDialog from './TeachDialog'
import VerdictChip from './VerdictChip'

function asApiError(err: unknown): ApiError {
  return err instanceof ApiError ? err : new ApiError(0, null, String(err))
}

function sourceWords(t: SetupT, pose: PoseOut): string {
  if (pose.source === 'taught' && pose.taught_at) {
    const at = new Date(pose.taught_at)
    const when = Number.isNaN(at.getTime()) ? pose.taught_at : at.toLocaleString(t.lang === 'de' ? 'de-DE' : 'en-GB')
    return t('ps.source.taught', { when })
  }
  return t('ps.source.config')
}

/** Why no pose can be taught now, in the reader's words. */
function whyNot(t: SetupT, poses: PosesOut): string {
  const code = poses.why_not_code
  if (code === 'planner_not_ready') return t('ps.why.planner_not_ready')
  if (code) return t.msg(refusalMsg(code))
  return poses.why_not
}

export default function PosesPanel() {
  const t = useT(SETUP)
  const titleId = useId()
  const { view } = usePrefs()
  const { cell } = useCell()
  const tech = view === 'tech'
  // What decides whether a pose can be taught now: read the list again whenever it changes.
  const key = [
    cell?.state,
    cell?.active_run_id,
    cell?.planner?.state,
    cell?.jaws_question,
    cell?.halted ? 'halted' : '',
    cell?.recovery?.run_id,
    cell?.recovery?.cleared_at,
    cell?.hand?.jaws,
  ].join('|')
  const poses = useAsync(() => api.poses(), [key])
  const [teaching, setTeaching] = useState(false)
  /** The default place being written: a name, `null` for none, `undefined` while nothing is written. */
  const [choosing, setChoosing] = useState<string | null | undefined>(undefined)
  const [error, setError] = useState<ApiError | null>(null)

  const choose = async (name: string | null) => {
    if (choosing !== undefined) return
    setChoosing(name)
    setError(null)
    try {
      await api.setDefaultPlace(name)
    } catch (err) {
      setError(asApiError(err))
    } finally {
      setChoosing(undefined)
      poses.reload()
    }
  }

  const data = poses.data
  const running = Boolean(cell?.active_run_id)
  const chosen = (name: string | null) => (choosing !== undefined ? choosing === name : (data?.default_place ?? null) === name)

  return (
    <section className="panel su-poses" aria-labelledby={titleId}>
      <div className="panel-head">
        <h3 id={titleId}>{t('ps.title')}</h3>
      </div>
      <p className="panel-lede">{t('ps.lede')}</p>
      {poses.error && <ErrorBanner error={poses.error} onRetry={poses.reload} />}
      {error && <ErrorBanner error={error} />}
      {poses.loading && !data && <Loading what={t('ps.what')} />}

      {data && (
        <>
          <div className="su-table">
            <table aria-labelledby={titleId}>
              <thead>
                <tr>
                  <th>{t('ps.col.pose')}</th>
                  <th>{t('ps.col.source')}</th>
                  <th>{t('ps.col.screen')}</th>
                  {tech && <th>{t('ps.col.joints')}</th>}
                  <th>{t('ps.col.default')}</th>
                </tr>
              </thead>
              <tbody>
                <tr className="su-home">
                  <td>
                    <strong>{data.home.label || 'Home'}</strong>
                    {tech && <span className="su-pose-name">{data.home.name}</span>}
                  </td>
                  <td className="prose">
                    {t('ps.source.config')} · {t('ps.home.readonly')}
                  </td>
                  <td>—</td>
                  {tech && <td>{jointsText(data.home.joints_deg, t.lang)}</td>}
                  <td>—</td>
                </tr>
                {(data.poses ?? []).map((pose) => (
                  <tr key={pose.name}>
                    <td>
                      <strong>{pose.label || pose.name}</strong>
                      <span className="su-pose-name">{pose.name}</span>
                    </td>
                    <td className="prose">{sourceWords(t, pose)}</td>
                    <td>
                      <VerdictChip verdict={pose.screen} />
                    </td>
                    {tech && <td>{jointsText(pose.joints_deg, t.lang)}</td>}
                    <td>
                      <input
                        type="radio"
                        name={`${titleId}-default`}
                        className="su-radio"
                        aria-label={t('ps.default.choose', { label: pose.label || pose.name })}
                        checked={chosen(pose.name)}
                        disabled={running || choosing !== undefined}
                        onChange={() => void choose(pose.name)}
                      />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {(data.poses ?? []).length === 0 && <p className="small faint su-poses-empty">{t('ps.empty')}</p>}

          <div className="su-poses-foot">
            <label className="check su-none">
              <input
                type="radio"
                name={`${titleId}-default`}
                checked={chosen(null)}
                disabled={running || choosing !== undefined || (data.poses ?? []).length === 0}
                onChange={() => void choose(null)}
              />
              {t('ps.default.none')}
            </label>
            <span className="small faint">{t('ps.default.help')}</span>
          </div>

          <div className="actions su-teach">
            <button type="button" className="primary big" disabled={!data.teachable} onClick={() => setTeaching(true)}>
              {t('ps.teach')}
            </button>
            {!data.teachable && <span className="su-why">{whyNot(t, data)}</span>}
            <span className="small dim su-target" title={data.target_file ?? undefined}>
              {data.target_file
                ? t('ps.target', { file: tech ? data.target_file : fileName(data.target_file) })
                : t('ps.target.none')}
            </span>
          </div>
          {tech && !data.teachable && data.why_not && <p className="small mono dim">{data.why_not}</p>}
        </>
      )}

      {teaching && data && (
        <TeachDialog
          poses={data}
          onClose={() => {
            setTeaching(false)
            poses.reload()
          }}
        />
      )}
    </section>
  )
}
