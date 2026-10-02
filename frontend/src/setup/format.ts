/** Small helpers of Setup, kept out of the component files (Fast Refresh wants those to export components only). */

import type { CellFactsOut } from '../api/client'
import type { Lang } from '../i18n'

/** The last part of a path, whichever separator it uses: `D:\cell\robot.cell.yaml` -> `robot.cell.yaml`. */
export function fileName(path: string): string {
  const parts = path.split(/[\\/]/)
  return parts[parts.length - 1] || path
}

const oneDecimal = new Map<Lang, Intl.NumberFormat>()

/**
 * Numbers of a readout (joint angles in degrees, a TCP in mm) in the reader's numbers, always one decimal so the
 * columns line up: `-60,1  -95,2` in German, `-60.1  -95.2` in English, as the payload beside them reads (`1,5 kg`).
 */
export function readoutText(values: readonly number[], lang: Lang): string {
  let format = oneDecimal.get(lang)
  if (!format) {
    format = new Intl.NumberFormat(lang === 'de' ? 'de-DE' : 'en-GB', {
      minimumFractionDigits: 1,
      maximumFractionDigits: 1,
      useGrouping: false,
    })
    oneDecimal.set(lang, format)
  }
  // A plain hyphen for the sign, as the console's other readouts write it (Intl may use U+2212).
  return values.map((v) => format.format(v).replace('\u2212', '-')).join('  ')
}

/** Joint angles in degrees, one decimal, in the reader's numbers. */
export function jointsText(values: readonly number[], lang: Lang): string {
  return readoutText(values, lang)
}

/** A pose name the console may write: an ASCII identifier of at most 32 characters, never `home` (build plan 1.8). */
export function nameRefusal(name: string): boolean {
  return !/^[A-Za-z_][A-Za-z0-9_]{0,31}$/.test(name) || name.toLowerCase() === 'home'
}

/** A label the console may write: one line of 1 to 40 characters, without `" #"` (a YAML comment). */
export function labelRefusal(label: string): boolean {
  const text = label.trim()
  return text.length < 1 || text.length > 40 || /[\r\n]/.test(label) || label.includes(' #')
}

/** The rig a teach shows: the wrist camera where there is one (it sees where the person guides the tool), else the
 * server's primary rig (`null`). */
export function wristRig(facts: CellFactsOut | null): string | null {
  return facts?.cameras?.find((rig) => rig.mounting === 'wrist')?.rig_id ?? null
}
