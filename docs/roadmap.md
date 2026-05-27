# Roadmap

Top-level view of in-flight + near-term initiatives. Each
entry is self-contained; this is the public-facing surface.

## Active initiatives

### LSP server

Language server for Wren projects compiled through WrenLift,
with first-class hatch-package awareness. Diagnostics, hover,
goto-definition (locals + scoped imports + cross-file +
bundled `@hatch:*` source), member completion (literal +
sema-typed receivers), run-action codelens (file-level on
`main.wren` / `*.spec.wren`, per-block on
`Test.describe` / `Test.it`), and an Activity Bar sidebar
with project scaffolder + spec runner all ship today.
Pending: workspace-wide rename, references, document
symbols (outline view), inlay hints, and a manual
`reloadDeps` command.

### Docs generator

JSON doc model walked from `///` comments, consumed by the
LSP hover handler and shipped inside every `.hatch` bundle's
source section so downstream tools (LSP across dep
boundaries, future docs website renderer) see the same
shape. Today's coverage is the model + LSP integration; the
docs-website renderer is the next addition.

### AOT distribution

Compile a hatch project (or a standalone `.wren` file) to a
self-contained deployment artifact: native executable for
desktop / server / edge, or `.wasm` module for browser /
WASI / serverless. Reuses parser → sema → MIR; the only
structural change is at the codegen split — Cranelift's
object-file mode for native, the existing wasm codegen
emitter driven whole-program for wasm. Two host profiles on
the wasm side: `--profile=browser` (links the existing
browser bridges + emits ES-module JS glue) and
`--profile=wasi` (WASI shim for stdio / args / env, single
`.wasm` for wasmtime / wasmer / Cloudflare Workers / Deno
Deploy). Component Model output sits behind the wasi
profile once the ecosystem tooling stabilises.

### First-party `hatch test` integration

`hatch test` today walks `*.spec.wren` files and spawns
`wlift <spec>` per file, parsing the `ok: N/M passed` line
from stdout. The next pass wires `@hatch:test` into the CLI
as a built-in: `--filter "<group> > <name>"` runs only
matching cases, `--json` emits one structured event per
case-start / pass / fail / summary so editors can render
live results, and the VS Code spec-runner sidebar's per-row
▶ Run button drops the "runs the whole file" caveat.

### Idle-fiber yield should park, not spin

`Fiber.yield()` on a nested fiber (one that has a `caller`, i.e.
anything driven by `sched.tick` / `f.try()` / `f.call()`) takes
the stackless `pending_fiber_action::Yield` path: it sets the
action, returns through the dispatch chain, and the interpreter
processes the Yield arm via `handle_jit_fiber_action` —
[runtime_fns.rs:2710](src/codegen/runtime_fns.rs#L2710), which
just returns the yielded value's bits without actually
suspending the fiber. AOT-emitted bodies that wrap a yield in a
`while` loop therefore run the loop at full CPU speed: the
yield call returns and the next iteration fires immediately,
millions of times per second, until the wall-clock condition
breaks the loop.

This was the load-bearing OOM driver on hatch.wrenlift.com.
`Catalog.refreshLoop` slept between refresh cycles with `while
(Clock.mono < deadline) Fiber.yield()`. The intended cooperative
sleep was actually a hot spin at ~1000 Hz × the spin's
per-iteration dispatch allocation; on Linux glibc the resulting
allocator churn climbed the process from a clean 36 MB local
boot RSS to fly.io's 768 MB OOM ceiling in ~40–180 seconds of
boot work, every restart. The site's mitigation (replace the
pure-yield with chunked `Clock.sleepMs` + one yield per chunk —
see [hatch site Catalog.sleepYielding_](https://github.com/wrenlift/hatch/blob/main/site/lib/catalog.wren))
blocks the thread during each chunk, which papers over the
runtime gap but pauses every other fiber for the chunk window.

Two ways to fix this in the runtime:

1. **Park nested-fiber yields properly.** Today's `try_krio_yield`
   short-circuits when the current fiber has a caller —
   [core/fiber.rs:494–504](src/runtime/core/fiber.rs#L494). The
   fallback (`set_fiber_action_yield`) propagates a yield
   *value* but doesn't actually suspend the fiber's execution.
   The fix is to make the stackless yield path return control
   to the scheduler at the yield point: switch `vm.fiber` to
   `(*fiber).caller`, mark the yielding fiber `Suspended`, and
   resume the caller's frame. The next `f.try()` reinstates
   the suspended state and continues past the yield. That's
   what `handle_jit_fiber_action::Call` already does on entry;
   the symmetric path on Yield is missing.

2. **Scheduler-level idle parking.** When the scheduler's
   `tick` finds no runnable fiber (all in cooperative-wait
   states), block on a condvar / kqueue / epoll until the next
   timer expiry or socket event. The `@hatch:web` scheduler's
   `Clock.sleepMs(1)` backoff (web.wren:613) is the right
   shape but only fires when the accept queue is empty —
   doesn't help when a long-lived background fiber is in a
   cooperative-sleep loop. Wiring a proper "sleep until X"
   primitive (e.g. `Fiber.sleepUntil(deadline)`) that
   registers with the scheduler's idle-block list would let
   the runtime collapse multiple idle background fibers onto
   a single OS-level wait.

Either fix alone closes the cold-yield-allocation class; (1)
is the smaller change but (2) is the more general primitive.
The site's chunked-sleep workaround can stay until one of
these lands.

### Plugin ABI stability

Today's plugin loader statically links the full `wren_lift`
crate into every `cdylib` (`@hatch:sqlite`, `@hatch:gpu`,
`@hatch:image`, ...), so each plugin ships its own compiled
copy of `VM`, `GcImpl`, `Gc`, `ArenaGc`, `NativeContext`,
plus the field offsets and enum discriminants baked into
machine code. Any change to `VM`'s layout in a wren_lift
release (added field, reordered variant, host-gated dep
bump) causes a silent `EXC_BAD_ACCESS` on the first
foreign-method call — the plugin reads VM bytes at the wrong
offsets, `gc_dispatch!` picks the wrong enum arm, and the
mis-typed payload panics inside the wrong allocator. The
existing `WLIFT_PLUGIN_ABI_VERSION = 1` constant only fires
when a maintainer remembers to bump it; private VM fields
slip past unnoticed. Two-step fix:

1. **Build-time VM-layout fingerprint** (half a day). A
   `build.rs` emits a `const` derived from
   `size_of::<VM>() + size_of::<GcImpl>() + size_of::<Gc>()`
   (and the other plugin-reachable structs); plugins compare
   their compiled-in fingerprint against the host's at
   dlopen and abort cleanly with a "rebuild against
   wren_lift &lt;rev&gt;" message instead of SIGSEGV. Turns the
   undefined-behaviour failure mode into a refusal we can
   diagnose. No behaviour change for matching builds.
2. **Opaque-VM C ABI** (1–2 weeks). Plugins stop
   `use wren_lift::runtime::vm::VM` entirely and call only
   `#[no_mangle] extern "C"` functions the host exports
   from `capi.rs` — `wlift_vm_alloc_string`,
   `wlift_vm_alloc_list`, `wlift_vm_set_slot`, etc. The
   plugin's `wren_lift` Cargo dep shrinks to a header-only
   `wlift_abi` crate that holds `Value` bits + `ObjHeader`
   layout for plugin-owned objects + the function-pointer
   table type. No moving VM internals cross the boundary,
   so VM changes can't reach plugin code at all. The wasm
   static-link path that currently shares the same
   `lib.rs` files needs a split crate
   (`wlift_<name>_core` with Rust + `wlift_<name>_ffi` with
   the C entry points), same model `@hatch:gpu_web` /
   `@hatch:window_web` already follow.

#1 lands first as a forcing function — once any unrebuilt
plugin trips the fingerprint check, #2 stops being optional.
The Lua / Python / V8 / Node-native model is exactly #2:
extension authors only see a stable C header, never the
runtime's Rust (or C++) internals. We've ducked it so far
because hatch and wren_lift share a monorepo and we always
rebuild together — but the moment a third party ships a
plugin, the static-link contract becomes load-bearing for
everyone, not just hatch's own dylibs.

## Smaller open items

These don't carry a dedicated treatment but track real gaps.

- **`textDocument/references` + `documentSymbol` + the
  `wlift-lsp.reloadDeps` `executeCommand`** — the three
  remaining LSP handlers that haven't shipped yet.
- **Implicit-this typo detection** — bare identifier inside
  a method that's neither a local nor a known class member
  is rewritten as `this.<name>` rather than diagnosed. See
  [QUIRKS.md](../QUIRKS.md#implicit-this-swallows-typos-inside-method-bodies).
- **Resolver semver ranges** — the version field in
  hatchfiles is parsed as a literal pin today; needs a
  registry index file plus a real semver matcher. See
  [QUIRKS.md](../QUIRKS.md#resolver-doesnt-accept-version-ranges-or-wildcards).
- **Path drift between `~/.local/bin` and `~/.cargo/bin`** —
  the extension's binary resolver picks the older one when
  both exist. See [QUIRKS.md](../QUIRKS.md#cargo-install-and-installsh-ship-to-different-bin-paths-and-can-drift).

## Recently shipped

- **runtime v0.1.16 + vscode-wrenlift 0.1.5** (2026-05-03).
  Sema-on-partial-AST so undefined-name diagnostics fire
  even when the parser bailed elsewhere; literal + var-typed
  receiver completion (`(10).`, `"foo".`, `[1,2].`,
  `var x = 10; x.<...>`); per-block `▶ Run` codelenses on
  every `Test.describe` / `Test.it`; goto-def for bare
  locals, `for` bindings, method/closure params; `hatch
  init` scaffold with real `@hatch:test` + `@hatch:assert`
  deps; Activity Bar sidebar with project scaffolder + spec
  runner.
- **Editor section on wrenlift.com** — install + setup +
  marketplace / LSP-config tabbed surface.
- **Hatch site at hatch.wrenlift.com** — landing + package
  catalog + readme renderer + blog.
