/**
 * Start the console the smoke test drives: `python -m api --profile console_dummy` on a scratch copy of the config.
 *
 * The shipped desk profile as it is: its two poses ("Ablage links", the default place, and "Parkposition") are the
 * ones a task places at and returns to, so the browser proves the desk a person starts. The copy (under
 * `node_modules/.cache`, ignored by git) is the server's `--data`, so the repo's own config is never written and the
 * server keeps no stop file across runs (a tree named with `--data` keeps none unless `WILLY_CONSOLE_STOP_FILE` names
 * one, and this run clears that variable).
 *
 * The server serves the built bundle (`api/static`); without it this refuses to start, because a smoke test of a stale
 * or missing bundle proves nothing about the console in this tree.
 *
 *   node e2e/serve.mjs --port 8761          (Playwright's webServer runs exactly this)
 *
 * `WILLY_PYTHON` names the interpreter with the project's environment (default `python`).
 */

import { spawn } from 'node:child_process'
import { cpSync, existsSync, mkdirSync, rmSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const frontend = resolve(here, '..')
const root = resolve(frontend, '..')
const scratch = join(frontend, 'node_modules', '.cache', 'willy-e2e')
const config = join(scratch, 'config')

const portAt = process.argv.indexOf('--port')
const port = portAt >= 0 ? process.argv[portAt + 1] : '8761'
const python = process.env.WILLY_PYTHON || 'python'

if (!existsSync(join(root, 'api', 'static', 'index.html')) || !existsSync(join(root, 'api', 'static', 'demo.html'))) {
  console.error('serve.mjs: api/static holds no built bundle. Run `npm run build` first.')
  process.exit(1)
}

rmSync(config, { recursive: true, force: true })
mkdirSync(scratch, { recursive: true })
cpSync(join(root, 'config'), config, { recursive: true })

// No stop file: every run starts from a cell that never stopped, whatever a console before it left.
const env = { ...process.env, PYTHONPATH: root, PYTHONIOENCODING: 'utf-8' }
delete env.WILLY_CONSOLE_STOP_FILE

const child = spawn(python, ['-m', 'api', '--profile', 'console_dummy', '--data', config, '--port', String(port)], {
  cwd: root,
  env,
  stdio: 'inherit',
})

const stop = () => {
  if (!child.killed) child.kill()
}
process.on('SIGINT', stop)
process.on('SIGTERM', stop)
process.on('exit', stop)
child.on('exit', (code) => process.exit(code ?? 0))
