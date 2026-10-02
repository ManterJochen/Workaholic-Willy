/**
 * A config value, where it came from, and — for the short list the backend allows — how to change it.
 *
 * The project's own research concluded that this config tree's structure is state-of-the-art and its problem is that
 * it cannot EXPLAIN ITSELF. That finding is the entire brief for this screen, so it leads with provenance rather than
 * with values: with layered profiles, "what is the value" is only half a question, and the other half is which of the
 * files won.
 *
 * Two safety properties are the backend's, and this screen must not soften either:
 *
 * * only keys returned by `/v1/config/writable` are offered as fields. There is no free-text key box for writing,
 *   because a console that can write anywhere in the tree is a console that can quietly disable a guard;
 * * each writable key carries a `measure` string — what the operator PHYSICALLY DOES to obtain the value. A form that
 *   says "float, kg" has not told them anything, so the instruction is rendered as the field's own line, not a tooltip.
 *
 * **In the reader's words.** Every key this console knows is named in the reader's language with one line on how to
 * get its value (`cf.key.*`); the server's own English sentence, the full instruction, stays a detail: open in the tech
 * view, under "Details (Server)" in the demo view. A key the console has no words for keeps the server's, open. A
 * controller address is offered only for the vendor this cell is (a UR cell is not offered the KUKA address, and the
 * desk's dummy arm neither): the library offers each only for its vendor, and the server lists them all.
 *
 * The key's documentation, its measure and the files are the server's own words: data, shown as written.
 */

import { useCallback, useState } from 'react'

import { ApiError, api, type ConfigValueOut, type WritableOut } from '../api/client'
import { Caveat, ErrorBanner, Loading, Panel, StatusPill } from '../components/ui'
import { useT } from '../i18n'
import { useAsync } from '../lib/useAsync'
import { usePrefs } from '../model/prefs'
import { useCell } from '../model/useCell'
import { SCREENS } from './i18n'

/** The writable keys this console has words for (`cf.key.<id>.label`, `.hint`), by the server's key. */
const KEY_WORDS = {
  'robot.safety.payload.mass_kg': 'mass',
  'robot.safety.payload.cog_mm': 'cog',
  'robot.gripper.tool_frame.source': 'tfSource',
  'robot.gripper.tool_frame.offset_mm': 'tfOffset',
  'robot.gripper.tool_frame.rotation_quat_xyzw': 'tfRotation',
  'camera.cameras.rigs[*].serial_number': 'serial',
  'camera.cameras.primary_rig_id': 'primary',
  'camera.cameras.rigs[*].enabled': 'enabled',
  'robot.ur.ip': 'urIp',
  'robot.kuka.controller_ip': 'kukaIp',
} as const

type KeyWords = (typeof KEY_WORDS)[keyof typeof KEY_WORDS]

function wordsOf(key: string): KeyWords | null {
  return (KEY_WORDS as Record<string, KeyWords>)[key] ?? null
}

/** The robot vendors whose keys are their own (`robot.<vendor>.*`); every other key is every cell's. */
const VENDOR_KEYS = ['ur', 'kuka'] as const

/** Whether `field` is offered on a cell of `vendor`: a vendor's own key only on that vendor's cell. */
function offered(field: WritableOut, vendor: string | null): boolean {
  if (vendor === null) return true
  const owner = VENDOR_KEYS.find((name) => field.key.startsWith(`robot.${name}.`))
  return owner === undefined || owner === vendor
}

export default function Config() {
  const t = useT(SCREENS)
  const tech = usePrefs().view === 'tech'
  const vendor = useCell().cell?.vendor ?? null
  const writable = useAsync(() => api.writable())
  const fields = (writable.data ?? []).filter((field) => offered(field, vendor))
  const [key, setKey] = useState('')
  const [explained, setExplained] = useState<ConfigValueOut | null>(null)
  const [error, setError] = useState<ApiError | null>(null)
  const [draft, setDraft] = useState<Record<string, string>>({})
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState<string | null>(null)

  const explain = useCallback(async (which: string) => {
    if (!which.trim()) return
    setError(null)
    try {
      setExplained(await api.explain(which))
    } catch (err) {
      setExplained(null)
      setError(err instanceof ApiError ? err : new ApiError(0, null, String(err)))
    }
  }, [])

  const save = useCallback(async () => {
    const values = Object.fromEntries(Object.entries(draft).filter(([, v]) => v.trim() !== ''))
    if (Object.keys(values).length === 0) return
    setSaving(true)
    setError(null)
    setSaved(null)
    try {
      const result = await api.patch(values)
      setSaved(t('cf.write.saved', { files: result.files.join(', '), values: JSON.stringify(result.values) }))
      setDraft({})
    } catch (err) {
      setError(err instanceof ApiError ? err : new ApiError(0, null, String(err)))
    } finally {
      setSaving(false)
    }
  }, [draft, t])

  return (
    <>
      <h2 className="cf-title">{t('cf.title')}</h2>
      <p className="lede">{t('cf.lede')}</p>

      {error && <ErrorBanner error={error} />}

      <Panel title={t('cf.explain.title')}>
        <div className="actions">
          <input
            type="text"
            className="mono grow"
            value={key}
            placeholder="robot.safety.payload.mass_kg"
            aria-label={t('cf.explain.title')}
            onChange={(e) => setKey(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && void explain(key)}
          />
          <button className="primary" onClick={() => void explain(key)}>
            {t('cf.explain.go')}
          </button>
        </div>

        {explained && (
          <div className="stack">
            <div className="actions run-line">
              <span className="mono">{explained.key}</span>
              <StatusPill status={explained.writable ? 'ok' : 'idle'}>
                {explained.writable ? t('cf.writable') : t('cf.readonly')}
              </StatusPill>
              {explained.tier && <StatusPill status="info">{explained.tier}</StatusPill>}
            </div>

            <dl className="kv">
              <dt>{t('cf.k.value')}</dt>
              <dd>{JSON.stringify(explained.value)}</dd>
              <dt>{t('cf.k.type')}</dt>
              <dd>{explained.type ?? '—'}</dd>
              <dt>{t('cf.k.default')}</dt>
              <dd>{JSON.stringify(explained.default)}</dd>
              <dt>{t('cf.k.source')}</dt>
              <dd>{explained.source ?? '—'}</dd>
            </dl>

            {explained.doc && (
              <div className="stack-item">
                <h3>
                  {t('cf.means')}{' '}
                  {explained.doc_scope === 'block' && <span className="small faint">{t('cf.blockScope')}</span>}
                </h3>
                <div className="detail">{explained.doc}</div>
              </div>
            )}

            {explained.why && (
              <div className="stack-item">
                <h3>{t('cf.why')}</h3>
                <div className="fix">{explained.why}</div>
              </div>
            )}

            {(explained.layers ?? []).length > 0 && (
              <div className="stack-item">
                <h3>{t('cf.layers')}</h3>
                <table>
                  <thead>
                    <tr>
                      <th style={{ width: 90 }} />
                      <th>{t('cf.layer.location')}</th>
                      <th>{t('cf.layer.raw')}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(explained.layers ?? []).map((layer, i) => (
                      <tr key={`${layer.location}-${i}`}>
                        <td>
                          {layer.winner ? (
                            <StatusPill status="ok">{t('cf.winner')}</StatusPill>
                          ) : (
                            <span className="faint">{t('cf.overridden')}</span>
                          )}
                        </td>
                        <td>{layer.location}</td>
                        <td>{layer.raw}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <Caveat>{t('cf.layers.caveat')}</Caveat>
              </div>
            )}
          </div>
        )}
      </Panel>

      <Panel title={t('cf.write.title')}>
        {writable.error ? (
          <ErrorBanner error={writable.error} onRetry={writable.reload} />
        ) : writable.loading && !writable.data ? (
          <Loading what={t('cf.what.keys')} />
        ) : fields.length === 0 ? (
          <div className="empty">{t('cf.write.none')}</div>
        ) : (
          <>
            {fields.map((field) => {
              const words = wordsOf(field.key)
              const label = words ? t(`cf.key.${words}.label`) : field.label
              return (
                <div className="row" key={field.key}>
                  <div>
                    <span className="small mono faint">{field.unit || '—'}</span>
                  </div>
                  <div>
                    <div className="name">{label}</div>
                    <div className="small mono faint">{field.key}</div>
                    {words && <div className="fix">{t(`cf.key.${words}.hint`)}</div>}
                    {/* The server's own sentence: the only instruction for a key without words, else a detail. */}
                    {!words || tech ? (
                      <div className={words ? 'small dim' : 'fix'}>{field.measure}</div>
                    ) : (
                      <details className="payload-toggle pf-said">
                        <summary>{t('pf.details')}</summary>
                        {field.measure}
                      </details>
                    )}
                    <input
                      type="text"
                      className="mono draft-input"
                      value={draft[field.key] ?? ''}
                      placeholder={t('cf.write.keep')}
                      aria-label={label}
                      onChange={(e) => setDraft({ ...draft, [field.key]: e.target.value })}
                    />
                  </div>
                </div>
              )
            })}
            <div className="actions stack">
              <button
                className="primary"
                disabled={saving || Object.values(draft).every((v) => !v.trim())}
                onClick={() => void save()}
              >
                {saving ? t('cf.write.busy') : t('cf.write.go')}
              </button>
              <span className="small faint">{t('cf.write.allOrNone')}</span>
            </div>
            {saved && (
              <div className="banner ok stack-item">
                <div className="body small">{saved}</div>
              </div>
            )}
            <Caveat>{t('cf.write.caveat')}</Caveat>
          </>
        )}
      </Panel>
    </>
  )
}
