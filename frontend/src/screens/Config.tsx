/**
 * A config value, where it came from, and — for the short list the backend allows — how to change it.
 *
 * The project's own research concluded that this config tree's structure is state-of-the-art and its
 * problem is that it cannot EXPLAIN ITSELF. That finding is the entire brief for this screen, so it
 * leads with provenance rather than with values: with layered profiles, "what is the value" is only
 * half a question, and the other half is which of four files won.
 *
 * Two safety properties are the backend's, and this screen must not soften either:
 *
 * * only keys returned by `/v1/config/writable` are offered as fields. There is no free-text key box,
 *   because a console that can write anywhere in the tree is a console that can quietly disable a
 *   guard;
 * * each writable key carries a `measure` string — what the operator PHYSICALLY DOES to obtain the
 *   value. A form that says "float, kg" has not told them anything, so `measure` is rendered as the
 *   field's instruction, not as a tooltip.
 */

import { useCallback, useState } from 'react'

import { ApiError, api, type ConfigValueOut } from '../api/client'
import { Caveat, ErrorBanner, Loading, Panel, StatusPill } from '../components/ui'
import { useAsync } from '../lib/useAsync'

export default function Config() {
  const writable = useAsync(() => api.writable())
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
    const values = Object.fromEntries(
      Object.entries(draft).filter(([, v]) => v.trim() !== ''),
    )
    if (Object.keys(values).length === 0) return
    setSaving(true)
    setError(null)
    setSaved(null)
    try {
      const result = await api.patch(values)
      setSaved(
        `Written to ${result.files.join(', ')}. The tree now holds: ${JSON.stringify(result.values)}`,
      )
      setDraft({})
    } catch (err) {
      setError(err instanceof ApiError ? err : new ApiError(0, null, String(err)))
    } finally {
      setSaving(false)
    }
  }, [draft])

  return (
    <>
      <h2>Config</h2>
      <p className="lede">
        Any key, and which of the layered files actually set it. Editing is limited to the short list
        the backend declares writable.
      </p>

      {error && <ErrorBanner error={error} />}

      <Panel title="Explain a key">
        <div className="actions">
          <input
            type="text"
            className="mono grow"
            value={key}
            placeholder="robot.safety.payload.mass_kg"
            onChange={(e) => setKey(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && void explain(key)}
          />
          <button className="primary" onClick={() => void explain(key)}>
            Explain
          </button>
        </div>

        {explained && (
          <div className="stack">
            <div className="actions run-line">
              <span className="mono">{explained.key}</span>
              <StatusPill status={explained.writable ? 'ok' : 'idle'}>
                {explained.writable ? 'writable here' : 'read-only here'}
              </StatusPill>
              {explained.tier && <StatusPill status="info">{explained.tier}</StatusPill>}
            </div>

            <dl className="kv">
              <dt>value</dt>
              <dd>{JSON.stringify(explained.value)}</dd>
              <dt>type</dt>
              <dd>{explained.type ?? '—'}</dd>
              <dt>default</dt>
              <dd>{JSON.stringify(explained.default)}</dd>
              <dt>source</dt>
              <dd>{explained.source ?? '—'}</dd>
            </dl>

            {explained.doc && (
              <div className="stack-item">
                <h3>
                  What it means{' '}
                  {explained.doc_scope === 'block' && (
                    <span className="small faint">
                      (this prose describes the surrounding block, not this key alone)
                    </span>
                  )}
                </h3>
                <div className="detail">{explained.doc}</div>
              </div>
            )}

            {explained.why && (
              <div className="stack-item">
                <h3>Why this number</h3>
                <div className="fix">{explained.why}</div>
              </div>
            )}

            {(explained.layers ?? []).length > 0 && (
              <div className="stack-item">
                <h3>Which file won</h3>
                <table>
                  <thead>
                    <tr>
                      <th style={{ width: 70 }} />
                      <th>location</th>
                      <th>raw</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(explained.layers ?? []).map((layer, i) => (
                      <tr key={`${layer.location}-${i}`}>
                        <td>
                          {layer.winner ? (
                            <StatusPill status="ok">winner</StatusPill>
                          ) : (
                            <span className="faint">overridden</span>
                          )}
                        </td>
                        <td>{layer.location}</td>
                        <td>{layer.raw}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <Caveat>
                  The same tree yields different values under different profile chains. This table is
                  the answer for the chain this console was started with — nothing else.
                </Caveat>
              </div>
            )}
          </div>
        )}
      </Panel>

      <Panel title="Guided write">
        {writable.error ? (
          <ErrorBanner error={writable.error} onRetry={writable.reload} />
        ) : writable.loading && !writable.data ? (
          <Loading what="the writable keys" />
        ) : !writable.data || writable.data.length === 0 ? (
          <div className="empty">This cell offers no writable keys.</div>
        ) : (
          <>
            {writable.data.map((field) => (
              <div className="row" key={field.key}>
                <div>
                  <span className="small mono faint">{field.unit || '—'}</span>
                </div>
                <div>
                  <div className="name">{field.label}</div>
                  <div className="small mono faint">{field.key}</div>
                  <div className="fix">{field.measure}</div>
                  <input
                    type="text"
                    className="mono draft-input"
                    value={draft[field.key] ?? ''}
                    placeholder="leave blank to keep"
                    onChange={(e) => setDraft({ ...draft, [field.key]: e.target.value })}
                  />
                </div>
              </div>
            ))}
            <div className="actions stack">
              <button
                className="primary"
                disabled={saving || Object.values(draft).every((v) => !v.trim())}
                onClick={() => void save()}
              >
                {saving ? 'Writing…' : 'Write to the config tree'}
              </button>
              <span className="small faint">
                All of them land or the request fails and none do.
              </span>
            </div>
            {saved && (
              <div className="banner ok stack-item">
                <div className="body small">{saved}</div>
              </div>
            )}
            <Caveat>
              The values shown back are READ BACK from the tree, not echoed — what survives coercion
              and the validators is what the cell will run with.
            </Caveat>
          </>
        )}
      </Panel>
    </>
  )
}
