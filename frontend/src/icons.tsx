/**
 * The console's icons: inline SVG, drawn for this console, no icon library.
 *
 * Every icon is a 24 x 24 outline in `currentColor`, so it takes the colour and the theme of the text beside it and
 * needs no stylesheet of its own. An icon never carries meaning alone: a button with an icon also has a label (or an
 * `aria-label`), and the icon is hidden from screen readers unless it is given a `title`.
 *
 * `Logo` is the Willy mark, modelled on the logo (docs/assets/willy_logo.png): the hexagonal gripper holding its lime
 * cube. Unlike the icons it keeps the logo's own colours in both themes.
 */

import type { ReactNode } from 'react'

export type IconName =
  | 'cockpit'
  | 'setup'
  | 'history'
  | 'settings'
  | 'sun'
  | 'moon'
  | 'speaker'
  | 'speakerOff'
  | 'monitor'
  | 'pulse'
  | 'globe'
  | 'eye'
  | 'play'
  | 'stop'
  | 'halt'
  | 'home'
  | 'restart'
  | 'check'
  | 'close'
  | 'plus'
  | 'alert'
  | 'info'
  | 'camera'
  | 'gripper'
  | 'controller'
  | 'link'
  | 'unlink'
  | 'bolt'
  | 'chevronRight'
  | 'chevronDown'
  | 'external'
  | 'mic'
  | 'chat'
  | 'cube'

/** Filled shapes are marked: everything else is a 1.8 px outline. */
const SHAPES: Record<IconName, ReactNode> = {
  cockpit: (
    <>
      <path d="M3.5 16a8.5 8.5 0 1 1 17 0" />
      <path d="M12 16l4.2-5.2" />
      <circle cx="12" cy="16" r="1.6" fill="currentColor" stroke="none" />
      <path d="M6.2 12.4l1.2.8M17.8 12.4l-1.2.8M12 7.5V9" />
    </>
  ),
  setup: (
    <>
      <path d="M4 7h9M17 7h3M4 17h3M11 17h9" />
      <circle cx="15" cy="7" r="2.2" />
      <circle cx="9" cy="17" r="2.2" />
      <path d="M4 12h5M13 12h7" />
      <circle cx="11" cy="12" r="2.2" />
    </>
  ),
  history: (
    <>
      <path d="M3.6 12a8.4 8.4 0 1 0 2.5-6" />
      <path d="M3.2 4.2v4.2h4.2" />
      <path d="M12 7.6V12l3.1 2" />
    </>
  ),
  settings: (
    <>
      <circle cx="12" cy="12" r="3.2" />
      <circle cx="12" cy="12" r="6.6" />
      <path d="M12 2.6v2.6M12 18.8v2.6M2.6 12h2.6M18.8 12h2.6M5.4 5.4l1.8 1.8M16.8 16.8l1.8 1.8M5.4 18.6l1.8-1.8M16.8 7.2l1.8-1.8" />
    </>
  ),
  sun: (
    <>
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2.8v2.2M12 19v2.2M2.8 12H5M19 12h2.2M5.5 5.5l1.6 1.6M16.9 16.9l1.6 1.6M5.5 18.5l1.6-1.6M16.9 7.1l1.6-1.6" />
    </>
  ),
  moon: <path d="M19.8 14.6A8.2 8.2 0 1 1 9.4 4.2a6.6 6.6 0 0 0 10.4 10.4z" />,
  speaker: (
    <>
      <path d="M4 9.5h3.4L12 5.6v12.8l-4.6-3.9H4z" />
      <path d="M15.4 9.2a4 4 0 0 1 0 5.6M17.9 6.7a7.5 7.5 0 0 1 0 10.6" />
    </>
  ),
  speakerOff: (
    <>
      <path d="M4 9.5h3.4L12 5.6v12.8l-4.6-3.9H4z" />
      <path d="M15.8 9.6l4.8 4.8M20.6 9.6l-4.8 4.8" />
    </>
  ),
  monitor: (
    <>
      <rect x="3" y="4" width="18" height="12" rx="2" />
      <path d="M9 20.5h6M12 16v4.5" />
    </>
  ),
  pulse: <path d="M3 12.5h4.2l2.4-6.2 4.2 12.4 2.4-6.2H21" />,
  globe: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="M3 12h18" />
      <path d="M12 3c2.4 2.5 3.7 5.5 3.7 9s-1.3 6.5-3.7 9c-2.4-2.5-3.7-5.5-3.7-9S9.6 5.5 12 3z" />
    </>
  ),
  eye: (
    <>
      <path d="M2.5 12S6 5.6 12 5.6 21.5 12 21.5 12 18 18.4 12 18.4 2.5 12 2.5 12z" />
      <circle cx="12" cy="12" r="3" />
    </>
  ),
  play: <path d="M8 5.2v13.6L18.8 12z" fill="currentColor" />,
  stop: <rect x="6" y="6" width="12" height="12" rx="1.6" />,
  halt: (
    <>
      <path d="M8 12.6V6.8a1.5 1.5 0 0 1 3 0v4.4" />
      <path d="M11 10.6V5.4a1.5 1.5 0 0 1 3 0v5.6" />
      <path d="M14 10.6V6.8a1.5 1.5 0 0 1 3 0v6.9a6.2 6.2 0 0 1-6.2 6.2h-.4a6 6 0 0 1-5-2.7l-1.6-2.5a1.5 1.5 0 0 1 2.5-1.7L8 14.6" />
    </>
  ),
  home: (
    <>
      <path d="M3.5 11.2L12 4.2l8.5 7" />
      <path d="M5.6 9.6V20h12.8V9.6" />
      <path d="M10 20v-5.4h4V20" />
    </>
  ),
  restart: (
    <>
      <path d="M20 12a8 8 0 1 1-2.4-5.7" />
      <path d="M20.2 3.8v4.6h-4.6" />
    </>
  ),
  check: <path d="M5 12.6l4.4 4.4L19.2 7.4" />,
  close: <path d="M6.2 6.2l11.6 11.6M17.8 6.2L6.2 17.8" />,
  plus: <path d="M12 5v14M5 12h14" />,
  alert: (
    <>
      <path d="M12 3.6l9.4 16.4H2.6z" />
      <path d="M12 9.8v4.6" />
      <circle cx="12" cy="17.2" r="1" fill="currentColor" stroke="none" />
    </>
  ),
  info: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 11v5.6" />
      <circle cx="12" cy="7.8" r="1" fill="currentColor" stroke="none" />
    </>
  ),
  camera: (
    <>
      <path d="M3.5 7.6h3.8l1.7-2.6h6l1.7 2.6h3.8v11.4h-17z" />
      <circle cx="12" cy="13" r="3.4" />
    </>
  ),
  gripper: (
    <>
      <path d="M7 3.5h10v4H7z" />
      <path d="M8 7.5v10h2.4v-10M13.6 7.5v10H16v-10" />
      <path d="M10.4 14.2h3.2v3.3h-3.2z" fill="currentColor" stroke="none" />
    </>
  ),
  controller: (
    <>
      <rect x="6" y="6" width="12" height="12" rx="2" />
      <rect x="9.5" y="9.5" width="5" height="5" rx="0.8" />
      <path d="M9 3.5V6M15 3.5V6M9 18v2.5M15 18v2.5M3.5 9H6M3.5 15H6M18 9h2.5M18 15h2.5" />
    </>
  ),
  link: (
    <>
      <path d="M10 14l4-4" />
      <path d="M8.6 10.4l-2 2a3.5 3.5 0 0 0 5 5l2-2" />
      <path d="M15.4 13.6l2-2a3.5 3.5 0 0 0-5-5l-2 2" />
    </>
  ),
  unlink: (
    <>
      <path d="M8.6 10.4l-2 2a3.5 3.5 0 0 0 5 5l1.2-1.2" />
      <path d="M15.4 13.6l2-2a3.5 3.5 0 0 0-5-5l-1.2 1.2" />
      <path d="M4 4l16 16" />
    </>
  ),
  bolt: <path d="M13 2.6L5.2 13.4h6l-1.2 8 7.8-10.8h-6z" />,
  chevronRight: <path d="M9.5 6l6 6-6 6" />,
  chevronDown: <path d="M6 9.5l6 6 6-6" />,
  external: (
    <>
      <path d="M14 4h6v6M20 4l-8.6 8.6" />
      <path d="M18 13.6V20H4V6h6.4" />
    </>
  ),
  mic: (
    <>
      <rect x="9" y="3" width="6" height="11" rx="3" />
      <path d="M5.6 11.4a6.4 6.4 0 0 0 12.8 0M12 17.8V21" />
    </>
  ),
  chat: <path d="M4 5h16v11H9.2L4 20.2z" />,
  cube: (
    <>
      <path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z" />
      <path d="M4 7.5l8 4.5 8-4.5M12 12v9" />
    </>
  ),
}

/**
 * One icon. `size` is in CSS pixels at the default root size, and is applied in `rem`, so an icon grows with the
 * console on a large screen (`styles.css` raises the root size there) instead of shrinking inside its button.
 */
export function Icon({ name, size = 18, title, className }: { name: IconName; size?: number; title?: string; className?: string }) {
  const rem = `${size / 16}rem`
  return (
    <svg
      className={className ? `icon ${className}` : 'icon'}
      width={size}
      height={size}
      style={{ width: rem, height: rem, flex: 'none' }}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.8}
      strokeLinecap="round"
      strokeLinejoin="round"
      role={title ? 'img' : undefined}
      aria-hidden={title ? undefined : true}
      aria-label={title}
      focusable="false"
    >
      {title ? <title>{title}</title> : null}
      {SHAPES[name]}
    </svg>
  )
}

/** The logo's colours, from the stylesheet's brand tokens, each with the logo's own value where no stylesheet is loaded. */
const NEON = 'var(--brand-neon, #bef702)'
const STEEL = 'var(--brand-steel, #121815)'
const STEEL_HI = 'var(--brand-steel-hi, #28322d)'
/** The cube's lit top and its shaded side: the neon lime, lighter and darker. */
const CUBE_TOP = '#e2ff8c'
const CUBE_SIDE = '#8fbf00'

/**
 * The Willy mark, modelled on the logo (docs/assets/willy_logo.png): the hexagonal gripper with the lime light in its
 * top bar, two struts down to its two fingers, and the lime cube held between their jaw pads.
 *
 * Drawn in the logo's own colours in both themes (dark steel, edged and lit in the neon lime: `--brand-steel`,
 * `--brand-steel-hi`, `--brand-neon`), as a logo is: on the light theme it reads as a dark badge, never as a pale
 * outline. Decorative unless given a `title`. Sized by its class (`.brand-mark` in the top bar).
 */
export function Logo({ className, title }: { className?: string; title?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 48 48"
      fill="none"
      role={title ? 'img' : undefined}
      aria-hidden={title ? undefined : true}
      aria-label={title}
      focusable="false"
    >
      {title ? <title>{title}</title> : null}
      {/* The hexagonal body. */}
      <path
        d="M24 3.4 34.9 9.7v12.6L24 28.6 13.1 22.3V9.7z"
        style={{ fill: STEEL, stroke: NEON }}
        strokeWidth="1.4"
        strokeLinejoin="round"
      />
      {/* The struts from the top bar down to the fingers, over the body: dark steel inside a lime edge. */}
      <path d="M19.8 12.6 13.8 18.6M28.2 12.6 34.2 18.6" style={{ stroke: NEON }} strokeWidth="3.4" strokeLinecap="round" />
      <path d="M19.8 12.6 13.8 18.6M28.2 12.6 34.2 18.6" style={{ stroke: STEEL_HI }} strokeWidth="1.8" strokeLinecap="round" />
      {/* The top bar and its light. */}
      <rect x="17.6" y="7.2" width="12.8" height="5.6" rx="1.6" style={{ fill: STEEL_HI, stroke: NEON }} strokeWidth="0.9" />
      <circle cx="24" cy="10" r="1.6" style={{ fill: NEON }} />
      {/* The two fingers, each with its lime slit, and the jaw pads that hold the cube. */}
      <rect x="9.2" y="16.6" width="5.4" height="24.8" rx="1.4" style={{ fill: STEEL_HI, stroke: NEON }} strokeWidth="0.9" />
      <rect x="33.4" y="16.6" width="5.4" height="24.8" rx="1.4" style={{ fill: STEEL_HI, stroke: NEON }} strokeWidth="0.9" />
      <path d="M11.9 20.4v8.4M36.1 20.4v8.4" style={{ stroke: NEON }} strokeWidth="1" strokeLinecap="round" />
      <rect x="14.6" y="33" width="2.8" height="5.2" rx="0.6" style={{ fill: STEEL, stroke: NEON }} strokeWidth="0.5" />
      <rect x="30.6" y="33" width="2.8" height="5.2" rx="0.6" style={{ fill: STEEL, stroke: NEON }} strokeWidth="0.5" />
      {/* The lime cube between the jaws: its lit top, its front in the neon, its shaded side. */}
      <g style={{ stroke: STEEL }} strokeWidth="0.6" strokeLinejoin="round">
        <path d="M24 29.4 30.4 33 24 36.6 17.6 33z" style={{ fill: CUBE_TOP }} />
        <path d="M17.6 33 24 36.6V44l-6.4-3.6z" style={{ fill: NEON }} />
        <path d="M24 36.6 30.4 33v7.4L24 44z" style={{ fill: CUBE_SIDE }} />
      </g>
    </svg>
  )
}
