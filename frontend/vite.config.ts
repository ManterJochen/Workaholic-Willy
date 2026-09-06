/**
 * Two pages, one build, and a dev proxy that makes both of them same-origin.
 *
 * **The proxy is a correctness device, not a convenience.** With it, `fetch('/v1/...')` and
 * `new WebSocket('ws://<host>/v1/events')` are the same code in development and in production, so the
 * client never carries a base URL, and no build flag can point the console at a different cell than
 * the one it is standing in front of. `ws: true` is load-bearing -- the run stream is a WebSocket, and
 * without it the Pick screen works in production and silently fails in `npm run dev`.
 *
 * **Two entries**, because the demo page is not a route of the console. It has no navigation, no
 * config, no write path, and a viewer must not be one mis-click from a Connect button. Keeping them
 * separate at the bundler level is what makes that structural rather than a matter of discipline.
 */

import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import tailwind from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
// From `vitest/config`, not `vite`: it is the same function widened with the `test` key, so the
// test settings below are type-checked rather than cast past the compiler.
import { defineConfig } from 'vitest/config'

/** Where the backend listens. Override for a cell on another box: `WILLY_API=http://10.0.0.5:8000`. */
const API = process.env.WILLY_API || 'http://127.0.0.1:8000'

// `__dirname` does not exist in an ESM config, and Vite's native config loader warns about the shim.
const here = dirname(fileURLToPath(import.meta.url))

export default defineConfig({
  // Tailwind v4 is CSS-first: the design tokens live in `src/styles.css` under `@theme`, not in a
  // JS config. That matters here -- the console's palette IS a contract (one status, one colour),
  // and a contract belongs in one file the stylesheet itself declares, not in a build config.
  plugins: [tailwind(), react()],
  build: {
    // Emitted straight into the directory the backend serves, so `npm run build` is the whole
    // deployment step and there is no copy to forget.
    outDir: resolve(here, '../api/static'),
    emptyOutDir: true,
    rollupOptions: {
      input: {
        console: resolve(here, 'index.html'),
        demo: resolve(here, 'demo.html'),
      },
    },
  },
  test: {
    // jsdom, because these tests render the screens rather than call functions: the bug class they
    // exist for ("optional field absent -> .map of undefined") only appears during a real render.
    environment: 'jsdom',
    globals: false,
  },
  server: {
    port: 5173,
    proxy: {
      '/v1': { target: API, changeOrigin: true, ws: true },
    },
  },
})
