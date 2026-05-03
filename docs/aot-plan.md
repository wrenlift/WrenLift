# AOT distribution

Compile a hatch project (or a standalone `.wren` file) to a
self-contained deployment artifact — native executable for
desktop / server / edge, or `.wasm` module for browser /
WASI / serverless. End user runs the artifact directly — no
`wlift` runtime install, no `.hatch` bundle, no interpreter
on the target machine.

## Goals

- `wlift build hello.wren -o hello` — single-file native
  standalone. Same parser + sema + MIR pipeline as the JIT,
  but Cranelift emits an object file instead of memory pages,
  and the linker pairs it with a statically-linked runtime.
- `wlift build hello.wren --target wasm32-unknown-unknown -o hello.wasm`
  — single-file `.wasm` module. Reuses the existing
  `codegen::wasm` MIR-to-wasm-encoder path the tier-up wasm
  JIT already drives, but emits the entire program upfront
  instead of one hot function at a time, and bundles a
  trimmed-down runtime (no JIT engine, no
  `wasm-encoder`-at-runtime machinery).
- `hatch build --aot` — for hatch projects: walk the
  manifest, lower every imported module + dep, link the
  whole graph into one artifact (native binary or `.wasm`
  depending on `--target`).
- Cross-compile out of the box for Linux / macOS / Windows ×
  x86_64 / aarch64, plus `wasm32-unknown-unknown` and
  `wasm32-wasip1`. Same target-triple shape as `cargo build
  --target …`, surfaced via `--target` on `wlift build` /
  `hatch build`.
- Ship the binaries small enough that "compile once, deploy"
  is realistic for CLI tools and game engines — single-file
  native binaries in the 8–15 MB range matching the size of
  the current `wlift` ELF, and `.wasm` modules in the
  300–800 KB range (already where the tier-up JIT's hot-
  function emits land for typical apps).

## Non-goals (initial release)

- AOT'd code can't tier-up at runtime — the JIT machinery is
  intentionally absent from the AOT binary. Hot loops live or
  die by what the AOT optimiser produced; no profile-guided
  re-jitting on the deployed machine.
- Plugin hot-reload. Plugins (today loaded via `libloading`
  at runtime) get statically linked at AOT time; swapping a
  plugin without rebuilding the host binary is out of scope.
- Incremental AOT compilation. Each `wlift build` is a clean
  build of the full module graph. Salsa-style memoisation
  lands later if compile times bite.
- Reflection / runtime class introspection beyond what the
  existing prelude offers. `Object.is(_)`, `instanceOf`, etc.
  keep working; runtime code-loading (eval-style) doesn't.

## Architecture

```
parse + sema + MIR  ──►  [AOT mode] ──┬─► native (Cranelift)
                                       │
                                       └─► wasm (wasm-encoder)

native:
  CLIF emit (object code path)
    ├── one Cranelift `Module` per Wren module
    ├── declared funcs link by mangled symbol
    └── output: `<crate>.o`
  runtime as `staticlib`
    ├── GC, fibers, value boxing
    ├── prelude (Object/Num/String/List/…)
    ├── interpreter (used for code that couldn't be AOT'd —
    │   closures over non-resolved imports, eval-shaped
    │   constructs, defensive fallback)
    └── plugins linked statically
  linker (lld / system ld)
    └── output: native executable

wasm:
  wasm-encoder emit (whole-program path)
    ├── one wasm `Module` per build invocation
    ├── inter-Wren-module calls resolve as direct func refs
    │   in the same `funcs` table the JIT path uses
    └── output: `<crate>.wasm` + (optional) `.js` glue
  runtime as wasm static
    ├── GC, fibers, value boxing — same Rust source,
    │   compiled with `--target wasm32-unknown-unknown` /
    │   `wasm32-wasip1`
    ├── prelude
    ├── interpreter (defensive fallback — same as native)
    └── browser bridges (Future / Browser / Dom / …) optional
        based on `--profile=browser` vs `--profile=wasi`
```

Reuses every layer above MIR unchanged. The split is at the
codegen boundary: Cranelift's `JITModule` becomes
`ObjectModule` for native, and the existing tier-up wasm
emit path (`codegen::wasm::emit_mir`) gets driven over the
whole module graph upfront for wasm. The runtime is already
a workspace member with `crate-type = ["lib", "cdylib",
"staticlib"]`, so the static-link side is mostly a packaging
exercise on native; on wasm it's a `wasm32-*` build of the
same crate stripped of the JIT/dynasm features.

## Compared to today

|                        | `wlift hello.wren`     | `hatch build` (`.hatch`) | native AOT (planned)        | wasm AOT (planned)              |
|------------------------|------------------------|--------------------------|-----------------------------|---------------------------------|
| Runtime install needed | yes, on every machine  | yes                      | **no** — self-contained     | **no** — wasm runtime ambient   |
| Output                 | n/a                    | `.hatch` bundle          | native ELF / Mach-O / PE    | `.wasm` (+ `.js` glue optional) |
| Deployment target      | desktop / server       | desktop / server         | desktop / server / edge     | browser / WASI / serverless     |
| Compile time           | none (parse on launch) | seconds                  | seconds                     | seconds                         |
| Startup time           | parser + sema cold     | bytecode load            | ~immediate                  | wasm instantiate                |
| Artifact size          | n/a                    | ~30–500 KB bundle        | 8–15 MB single executable   | 300–800 KB single `.wasm`       |
| Hot tier-up            | yes                    | yes                      | **no** (planned tradeoff)   | **no** (planned tradeoff)       |
| Plugins                | dlopen at runtime      | dlopen                   | static-linked at build      | static-linked at build (wasm-compatible only) |

## Phasing

### Phase 1 — Cranelift object emit

Switch the existing JIT backend behind a feature gate:
`feature = "aot"` selects `ObjectModule`, default stays
`JITModule`. Most of the codegen surface
(`codegen::cranelift_backend`) is shared — the difference is
which `Module` trait impl gets handed to the emitter.

Deliverable: a unit test that compiles a trivial Wren
function (e.g. `class App { static run() { return 42 } }`) to
a `.o` file and runs `nm` to assert the mangled symbol is
present.

### Phase 2 — Single-file driver

`wlift build <file.wren> -o <out>`:

1. Parse + sema the file.
2. Lower the resulting AST through MIR + the AOT codegen
   feature.
3. Drive the system linker (`cc`, falling back to `lld`) to
   pair the object with the runtime's `staticlib`.
4. Write the output binary; `chmod +x` on POSIX.

Deliverable: `wlift build examples/hello.wren -o /tmp/hello`
produces a binary that `/tmp/hello` runs and prints `Hello,
world!` without touching `~/.local/bin/wlift`.

### Phase 3 — Multi-module AOT

The single-file driver only sees one module. Real projects
import siblings + `@hatch:*` deps. Walk the import graph
from the entry, lower every reachable `.wren` source, and
hand the linker every resulting `.o`. Cross-module symbol
references go through Cranelift's external-symbol mechanism
(same shape the JIT already uses for runtime helper calls).

Hatch deps come pre-built as `.hatch` bundles today. AOT
needs the bundle's `Source` section (already present —
that's what the LSP reads for goto-def). Lower the source
into MIR at link time, link the resulting objects in.

Deliverable: `hatch build --aot` on a project that imports
`@hatch:fmt` produces a binary that runs the project's
entry without resolving deps at runtime.

### Phase 4 — Cross-compilation

Add a `--target <triple>` flag on `wlift build` /
`hatch build`. Cranelift already supports cross-target
codegen — the missing piece is ensuring the runtime
`staticlib` is available for the target triple. Lean on the
existing GitHub Actions release matrix
(linux-x86_64, linux-aarch64, macos-aarch64, macos-x86_64)
to publish a `wren_lift_runtime-<triple>.a` artifact
alongside each runtime release.

Deliverable: `wlift build hello.wren --target aarch64-unknown-linux-gnu`
on a macOS host produces a Linux ARM64 ELF that runs on
the target.

### Phase 5 — Plugins

Plugins (`@hatch:gpu`, `@hatch:sqlite`, etc.) currently load
via `libloading` at runtime against the host's plugin ABI.
For AOT we need the link step to bundle the plugin's
`staticlib` instead. Each plugin's `Cargo.toml` already
declares both `cdylib` (for runtime dlopen) and `rlib`/
`staticlib` artifacts; the AOT linker just needs to know
which to pick.

Risk: plugins that depend on system libraries (Vulkan, Metal,
audio frameworks) need the AOT linker to resolve those too.
Acceptable to push that onto the plugin's build script for
v1 — same model `cargo build` already uses.

Deliverable: an AOT'd hatch project that imports
`@hatch:sqlite` runs without `libwlift_sqlite.so` on the
target.

### Phase 6 — Code-size optimisation

AOT binary size starts ~15 MB (runtime + prelude + user
code + Cranelift output). Pass-level work to bring that
down:

- Dead-code elimination across the linked module graph.
  LLVM/lld already does function-level DCE; surface it via
  `--gc-sections` in the linker invocation.
- Strip unused prelude classes. Today every prelude class
  ships in every binary. A reachability analysis driven by
  the user's MIR can prune the prelude graph.
- `panic = "abort"` on the runtime's release profile
  (already on for the JIT path; matters more for AOT
  binary size).

Target: 8–10 MB for "hello world", scaling sub-linearly
with project size.

### Phase 7 — Wasm AOT

Reuses every phase 1–3 deliverable but swaps the codegen
backend. The tier-up wasm JIT (`tier_wasm`) already drives
`codegen::wasm::emit_mir` per hot function and instantiates
the result against the running module via `wasm-encoder`.
AOT mode walks the entire module graph, calls `emit_mir`
once per Wren function, and emits a single `.wasm` module
covering the whole program — no runtime `wasm-encoder`,
no tier-up, no parker / promotion broker.

```sh
$ wlift build hello.wren --target wasm32-unknown-unknown -o hello.wasm
$ wlift build hello.wren --target wasm32-wasip1 -o hello.wasm
$ hatch build --aot --target wasm32-wasip1
```

Two host profiles, picked via `--profile`:

- `--profile=browser` — links the browser bridges (Future /
  Browser / Dom / WebSocket / Storage / Canvas) the
  playground already uses; emits an ES-module-shaped
  `<out>.js` glue file alongside the `.wasm`. Drop-in for
  static hosting + serverless edge platforms that speak
  wasm-bindgen-style imports.
- `--profile=wasi` — drops the browser bridges, links a
  thin WASI shim for stdio / args / env, emits a single
  `.wasm` runnable under `wasmtime run`, `wasmer run`,
  Cloudflare Workers (WASI flavor), Deno Deploy, etc.

Differences from the native phases:

- **No external linker.** wasm-encoder produces the final
  module directly; there's no `ld` / `lld` step. Phase 1's
  object-emit gate doesn't apply — wasm has always emitted
  whole modules.
- **Plugins.** Wasm-compatible plugins (anything that
  builds for `wasm32-*` — math, image, sqlite-wasm-rs)
  link statically into the module. Native-only plugins
  (`@hatch:gpu`, `@hatch:window`) refuse with a clear error
  on wasm AOT; the playground already runs the
  wasm-compatible subset, so the boundary is well-trodden.
- **Cranelift JIT crate stays out of the binary.** The
  existing `wasm/Cargo.toml` already builds the runtime
  `--no-default-features` (no host JIT, no dynasm); AOT
  mode reuses that profile, just additionally strips the
  wasm-encoder runtime dep since modules are already final.
- **Memory model.** Linear memory layout matches what
  `wlift_wasm` lays down today (NaN-boxed values, GC
  arena, fiber stacks). The AOT'd module just doesn't need
  to grow the function table at runtime.

Deliverables, in order:

1. `wlift build --target wasm32-*` walks the module graph
   and feeds `emit_mir` whole-program. Output: a single
   `.wasm` that runs under `wasmtime run` and prints
   `Hello, world!` from a one-line Wren source.
2. Multi-module + hatch-dep wasm AOT (mirrors Phase 3 on
   the native side; same `Source`-section walking).
3. Browser profile: emits the `.js` glue, exercised by
   loading the AOT'd module from a static HTML page
   without the playground's tier-up JIT machinery active.
4. Wasm-aware code-size pass: drop tier-up bookkeeping
   tables, prune unreachable prelude, run `wasm-opt -Oz`
   as an optional post-step.

### Phase 8 — Component Model + edge runtime profiles

Once Phase 7 stabilises, opt-in `--profile=component` emits
a Wasm Component (not a core module) so Wren artifacts
plug into the wider WASI 0.2+ ecosystem (Spin, wasmCloud,
component-aware host runtimes). Same module emit + a
`witty`-style component wrapper. Planned, depends on
Component Model tooling maturing.

## Open questions

- Single-binary distribution vs. binary + sidecar plugins?
  Lean: single binary is the explicit goal of AOT; sidecar
  plugins are a separate "ship a hatch project as a
  directory tree of files" mode.
- AOT'd artifact picks up `wlift --version`-style metadata
  so the user can ask "which runtime built this?". Lean yes,
  print under `--wrenlift-version` on native binaries; embed
  in the wasm `producers` section on `.wasm` outputs so
  `wasm-tools` can read it.
- Source maps / debug info. AOT'd backtraces should still
  point at the user's `.wren` source. Cranelift can emit
  DWARF for native; wasm-encoder supports the wasm-source-
  map / `name` sections — thread the existing span info
  through both codegen paths.
- AOT-only diagnostics. Some MIR shapes (e.g. closures over
  unresolved imports) can't be statically lowered. Surface
  these as "this function will fall back to the interpreter
  at runtime" warnings rather than hard errors, so the user
  can decide whether to refactor or ship as-is.
- Wasm async story. The current `wlift_wasm` runtime drives
  async via the parked-fiber list + top-level `await`
  scheduler; AOT'd wasm modules need the same scheduler
  embedded. Decide whether to ship one scheduler statically
  or expose the loop as an imported host function so the
  edge runtime can multiplex.
- Browser-profile JS glue: hand-roll versus `wasm-bindgen`-
  driven? Today's playground takes the `wasm-bindgen` path
  for ergonomics; AOT may want the smaller hand-rolled
  shape. Lean: keep `wasm-bindgen` for the browser profile,
  hand-roll for `--profile=wasi`.

## Dependencies

- Cranelift backend (`feature = "cranelift"`) is the only
  codegen path the **native** AOT phases depend on. Phase 1
  lifts the AOT switch out of `JITModule` vs
  `ObjectModule` — same crate, same IR, same passes.
- Existing `codegen::wasm::emit_mir` is the only codegen
  path the **wasm** AOT phase depends on. The tier-up wasm
  JIT already drives it per hot function; Phase 7 drives
  it whole-program.
- Hatch dep bundle's `Source` section (already present, used
  by the LSP) is what Phase 3 + Phase 7's multi-module step
  consume for cross-module AOT.
- Runtime workspace `crate-type` already includes
  `staticlib`, so Phase 1's object-emit + static-link
  pipeline doesn't need a Cargo manifest change. The same
  workspace already builds for `wasm32-unknown-unknown` /
  `wasm32-wasip1` (the playground + the wasm smoke test
  prove this), so Phase 7's runtime side is also a
  packaging exercise rather than a port.
