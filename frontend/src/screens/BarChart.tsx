/**
 * One measure per task, as thin columns: inline SVG, no chart library (build plan 4: no new runtime dependency).
 *
 * Drawn to the console's chart rules: one series in one hue (a lime step that sits inside the chart band of each
 * theme, `--hs-mark` in screens.css; the title names the series, so there is no legend box), columns at most 24 px
 * wide with a 4 px rounded tip and a square foot on one baseline, hairline solid gridlines at round ticks, and a value
 * written only at the newest column; every column carries its value on hover and on keyboard focus, and the runs table
 * below is the chart's table view. A task without a value draws no column and says "kein Wert": never a zero.
 */

import { useId, useState } from 'react'

export interface Bar {
  readonly key: string
  /** The tick under the column (the task's number in the session). */
  readonly tick: string
  /** `null`: no value for this task (said, never drawn as zero). */
  readonly value: number | null
  /** The value in words, for the tooltip and the label: "75 %", "24,6 s", "kein Wert". */
  readonly valueText: string
  /** A second line for the tooltip: what the value is out of. */
  readonly detail?: string
  /** The column's accessible name: "Auftrag 3, 12:31: 75 %". */
  readonly label: string
}

export interface BarChartProps {
  readonly title: string
  readonly note: string
  readonly bars: readonly Bar[]
  /** The top of the scale. */
  readonly max: number
  /** Round values the gridlines stand at, the baseline included. */
  readonly ticks: readonly number[]
  readonly tickText: (value: number) => string
  /** Said where there is no column at all. */
  readonly empty: string
}

const W = 560
const H = 200
const LEFT = 46
const RIGHT = 10
const TOP = 22
const BOTTOM = 26
const PLOT_W = W - LEFT - RIGHT
const PLOT_H = H - TOP - BOTTOM
const MAX_BAR = 24
const TIP_R = 4

/** A column from its foot up, square at the baseline and rounded at its tip. */
function column(x: number, top: number, width: number, height: number): string {
  const r = Math.min(TIP_R, width / 2, height)
  const foot = top + height
  return [
    `M${x},${foot}`,
    `L${x},${top + r}`,
    `Q${x},${top} ${x + r},${top}`,
    `L${x + width - r},${top}`,
    `Q${x + width},${top} ${x + width},${top + r}`,
    `L${x + width},${foot}`,
    'Z',
  ].join(' ')
}

export default function BarChart({ title, note, bars, max, ticks, tickText, empty }: BarChartProps) {
  const id = useId()
  const [hover, setHover] = useState<number | null>(null)
  const scale = max > 0 ? max : 1
  const y = (value: number) => TOP + PLOT_H - (Math.max(0, Math.min(value, scale)) / scale) * PLOT_H
  const n = bars.length
  const band = n > 0 ? PLOT_W / n : PLOT_W
  const width = Math.max(4, Math.min(MAX_BAR, band * 0.56))
  const newest = n - 1
  const shown = hover !== null ? bars[hover] : null

  return (
    <figure className="hs-chart" aria-labelledby={`${id}-title`}>
      <figcaption>
        <span className="hs-chart-title" id={`${id}-title`}>
          {title}
        </span>
        <span className="hs-chart-note">{note}</span>
      </figcaption>
      {n === 0 ? (
        <p className="empty">{empty}</p>
      ) : (
        <div className="hs-chart-plot">
          <svg viewBox={`0 0 ${W} ${H}`} role="group" aria-label={title} preserveAspectRatio="xMidYMid meet">
            {ticks.map((tick) => (
              <g key={tick} aria-hidden="true">
                <line className="hs-grid" x1={LEFT} x2={W - RIGHT} y1={y(tick)} y2={y(tick)} />
                <text className="hs-tick" x={LEFT - 8} y={y(tick)} dy="0.32em" textAnchor="end">
                  {tickText(tick)}
                </text>
              </g>
            ))}
            {bars.map((bar, index) => {
              const centre = LEFT + band * (index + 0.5)
              const top = bar.value !== null ? y(bar.value) : y(0)
              const height = y(0) - top
              return (
                <g
                  key={bar.key}
                  className={`hs-bar${hover === index ? ' hover' : ''}`}
                  role="img"
                  aria-label={bar.label}
                  tabIndex={0}
                  onPointerEnter={() => setHover(index)}
                  onPointerLeave={() => setHover((current) => (current === index ? null : current))}
                  onFocus={() => setHover(index)}
                  onBlur={() => setHover((current) => (current === index ? null : current))}
                >
                  {/* The hit area is the whole band, not the painted pixels. */}
                  <rect className="hs-hit" x={LEFT + band * index} y={TOP} width={band} height={PLOT_H} />
                  {bar.value !== null && height >= 1 && (
                    <path className="hs-mark" d={column(centre - width / 2, top, width, height)} />
                  )}
                  {bar.value !== null && height < 1 && (
                    <rect className="hs-mark" x={centre - width / 2} y={y(0) - 2} width={width} height={2} />
                  )}
                  {bar.value === null && (
                    <text className="hs-none" x={centre} y={y(0) - 6} textAnchor="middle" aria-hidden="true">
                      –
                    </text>
                  )}
                  {index === newest && bar.value !== null && (
                    <text className="hs-value" x={centre} y={top - 6} textAnchor="middle" aria-hidden="true">
                      {bar.valueText}
                    </text>
                  )}
                  <text className="hs-xtick" x={centre} y={H - 8} textAnchor="middle" aria-hidden="true">
                    {bar.tick}
                  </text>
                </g>
              )
            })}
          </svg>
          {shown && hover !== null && (
            <div
              className="hs-tip"
              role="presentation"
              style={{ left: `${((LEFT + band * (hover + 0.5)) / W) * 100}%` }}
            >
              <strong>{shown.valueText}</strong>
              {shown.detail && <span>{shown.detail}</span>}
            </div>
          )}
        </div>
      )}
    </figure>
  )
}
