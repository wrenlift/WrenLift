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
# from the repo root — baseline (no v128, runs on every browser)
wasm-pack build wasm --target web --release --no-typescript
```

That writes `wasm/pkg/`:

```
pkg/
  wlift_wasm.js          # JS bindings (wasm-bindgen output)
  wlift_wasm_bg.wasm     # ~2.3 MB optimised cdylib
  package.json           # npm-publishable
```

### SIMD-enabled build (optional)

The runtime's SIMD kernels (`Simd4f` / `Simd4i` ops) take an explicit
wasm `simd128` fast path through `core::arch::wasm32::*` — but the
intrinsics are gated on the build feature, so a default
`wasm-pack build` produces the scalar-fallback artifact. Build
the SIMD flavour by setting `RUSTFLAGS` and renaming the output:

```sh
RUSTFLAGS="-C target-feature=+simd128" \
  wasm-pack build wasm --target web --release --no-typescript \
                       --out-name wlift_wasm_bg.simd128
mv wasm/pkg/wlift_wasm_bg.simd128_bg.wasm \
   wasm/pkg/wlift_wasm_bg.simd128.wasm
```

Then ship both `wlift_wasm_bg.wasm` and `wlift_wasm_bg.simd128.wasm`
side-by-side and let `createWlift` feature-detect at load time:

```js
import { createWlift } from "./wlift.js";

const wlift = await createWlift({
  wasm: {
    url:        new URL("./wlift_wasm_bg.wasm",        import.meta.url),
    simd128Url: new URL("./wlift_wasm_bg.simd128.wasm", import.meta.url),
  },
});
```

`simd128Supported()` (also exported from `wlift.js`) returns the
boolean answer if you want to make the choice yourself. Pre-2021
Safari / older Edge builds without the SIMD proposal fall through
to the baseline URL.

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

Builds `smoke.wasm` for `wasm32-wasip1` twice — once with the
default flags (baseline) and once with `-C target-feature=+simd128`
— runs each under the embedded `wasmtime` runtime, and asserts
both produce identical output. The simd128 variant additionally
asserts wasm v128 opcodes are present (so a regression that
quietly drops the explicit intrinsic paths can't sneak through).

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
