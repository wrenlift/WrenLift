# `wlift_wasm` — WrenLift in the browser

Wasm shim around the WrenLift runtime. Two flavours of the same
interpreter:

* **`wlift_wasm.wasm`** — `wasm-bindgen`-style cdylib for browser
  hosts. JS callers get `init()`, `version()`, and `run(source)`.
* **`smoke.wasm`** — a `wasm32-wasip1` binary that interprets a
  hard-coded Wren program. Used by `tests/wasm_smoke.rs` to drive
  the runtime under `wasmtime` in CI without needing a browser.

No JIT, no plugin loading, no fs / sockets / processes — those
are all gated to the host build under `feature = "host"`. The
runtime defaults to `ExecutionMode::Interpreter` on wasm.

## Browser build

```sh
# from the repo root
wasm-pack build wasm --target web --release --no-typescript
```

That writes `wasm/pkg/`:

```
pkg/
  wlift_wasm.js          # JS bindings (wasm-bindgen output)
  wlift_wasm_bg.wasm     # ~2.3 MB optimised cdylib
  package.json           # npm-publishable
```

Then serve `wasm/web/index.html` with any static file server
that loads ESM modules. Quick option:

```sh
cd wasm
python3 -m http.server 8080
# open http://127.0.0.1:8080/web/
```

The page is a two-pane REPL: textarea on the left, captured
`System.print` output on the right. Press ⌘↵ / Ctrl+↵ inside the
editor to re-run.

## CI smoke

```sh
cargo test --test wasm_smoke -- --nocapture
```

Builds `smoke.wasm` for `wasm32-wasip1`, runs it via the
embedded `wasmtime` crate, and asserts the captured stdout
matches the expected lines (`hello from wasm!`, fib output,
list ops, monotonic clock probe).

## What works today

* Parser + AST + MIR + bytecode interpreter
* GC (mark-sweep)
* Core classes — `Bool`, `List`, `Map`, `Range`, `String`,
  `Num`, `Fiber`, `Null`, `Object`, `Class`, `Sequence`,
  `TypedArray`, `Fn`, `MapEntry`
* Optional core modules — `random`, `time`, `regex`, `uuid`,
  `toml` (and `hatch` for module reload, gated host-only fs
  bits)
* `web-time`-backed clocks (`Time.unix`, `Time.mono`)

## What's gated out (Phase 1.1+)

* JIT / threaded interpreter
* `crypto` / `hash` / `socket` / `http` / `fs` / `os` / `proc` /
  `zip` / `regex` (regex stays portable; the others are real
  host-only)
* `foreign class` plugin loading
* `hatch.run` + everything that needs `tempfile`

## Roadmap (Phase 1.1+)

* `Promise` ↔ `Fiber.yield` bridge so `fetch` / `WebSocket` /
  `setTimeout` / `requestAnimationFrame` plug in via foreign
  methods.
* `core::browser` foreign-class registry (Web IDL → Wren `class`
  generator later; first cut is hand-written).
* Plugin-as-wasm pattern — start with `hatch-image` (pure-Rust
  `image` crate), then `hatch-gpu` (wgpu → WebGPU).
* Worker-driven REPL so long-running scripts don't freeze the
  page.
