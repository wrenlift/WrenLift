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

### SIMD for physics + game math

Rapier already exploits SIMD inside the plugin — glamx + parry compile
to f32x4 / SSE / NEON paths on x86_64 and aarch64, so the solver,
broad-phase BVH traversal, and contact manifold generation are
vectorized at the crate level. There's no win to chase by rewriting
`wlift_physics` internals. The latency that *is* visible to a Wren
game loop sits at three boundaries the runtime can move:

1. **Batched physics queries.** Per-call FFI dominates ray and shape
   sweeps at high cast counts (e.g. line-of-sight checks across a
   whole NPC roster, environment probes per particle). Add
   `World.castRaysBatched(origins, dirs, maxToi, outHits)` and
   `World.castShapesBatched(shape, origins, dirs, maxToi, outHits)`
   that take `Float32Array` inputs and write packed result records
   back into a caller-owned `Float32Array` / `Int32Array` pair — one
   FFI crossing per batch instead of one per cast. Mirrors the
   pattern `World.positionInto(out, offset)` already uses for
   transform readback.
2. **Rapier `parallel` feature.** Rapier ships a `rayon`-backed
   parallel island solver behind an optional feature; we don't
   enable it today. Body-heavy scenes (hundreds of dynamic
   colliders, ragdolls, debris) get a real multi-core win with a
   single `Cargo.toml` flip. Behind a `parallel` cfg on
   `wlift_physics` so single-core / wasm builds stay scalar.
3. **`@hatch:math` SIMD-typed surface.** `@hatch:math` already uses
   glamx Rust-side, but the Wren-facing `Vec3` / `Mat4` round-trip
   scalars through the FFI on every component access. Expose a
   `Simd4f`-backed `Vec3` / `Vec4` / `Mat4` so transform composition
   (`mat * mat`, `mat.transformPoint`, batched vertex skinning, the
   per-frame model-matrix array that feeds `Float32Array` into the
   GPU buffer) stays in vector registers. The Wren `Simd4f` /
   `Simd4i` built-ins already exist — this is wiring `@hatch:math`'s
   public surface to them, not new runtime work.

(1) is the cheapest concrete win and unblocks AI / VFX code that
currently amortizes FFI by caching results across frames. (2) is a
one-line Cargo change gated behind benchmarks on a body-heavy scene.
(3) is the largest mechanical change but the only one that helps
game code that isn't physics-heavy — UI layout math, particle
systems, animation blending.

### Particles — 3D path, GPU sim, blend modes

`@hatch:game/particles` ships CPU-driven 2D today: per-particle
position / velocity / lifetime / colour-over-life, output through
`Renderer2D.drawSpriteTinted`, slot-reuse pool keeps allocations
flat. Three deliberate gaps:

1. **3D particles.** The simulation half (`Particle_.step_`,
   `ParticleSystem.update`, the `Particles` registry) is
   dimension-agnostic — it integrates `vx/vy` today but extending
   to `vz` is a field-add, not a redesign. The blocker is on the
   draw side: `Renderer3D` has no `drawBillboard(tex, origin,
   width, height, tint)` entry yet. Land that first, then a
   sibling `ParticleSystem3D` (or a `kind: "3d"` switch on the
   existing class) feeds it. Shape stays config-driven and
   identical to 2D from the caller's view.
2. **GPU simulation + instanced draw.** Past ~5–10k live
   particles per frame the Wren-side update loop and the per-
   particle `drawSpriteTinted` calls dominate the frame. The
   replacement path is (a) store particle state in a storage
   buffer (`Float32Array` upload at spawn, compute shader steps
   each frame), (b) emit one `drawIndexed(6, instanceCount =
   liveCount)` per system with per-instance attributes for
   position / size / colour. `ParticleSystem.new({...})` config
   stays unchanged — the simulation backend is an internal swap.
   Gated behind benchmarks; the CPU path is fine for typical
   game scale (low thousands) and ships smaller.
3. **Blend modes on Renderer2D.** The `wlift_gpu` pipeline
   descriptor now accepts a `"blend"` Map (added for Bloom's
   upsample), so the GPU layer is unblocked. The remaining work
   is on Renderer2D's batcher: it ships a single pre-built
   pipeline today, so switching blend mode mid-frame means
   flushing the current batch and rebinding a parallel pipeline.
   Add `blend: "additive" | "alpha" | "premultiplied"` to
   `ParticleSystem` config (and any `Renderer2D.drawSprite*`
   variant that needs it), keyed off a small cache of
   pre-compiled pipelines per blend mode.

### HUD — retained widgets, bitmap-font import, gamepad nav

`@hatch:hud` ships an immediate-mode HUD overlay: `HUD.new(g)` then
`hud.label / hud.rect / hud.border / hud.button`. Renders through
the existing `Renderer2D` sprite batch (now with alpha blending, so
font glyphs + transparent UI panels composite correctly). Built-in
5×7 procedural font covers digits, uppercase, and 11 punctuation
characters — enough for the canonical HUD label set without
shipping a font asset. Hover / click state is tracked per widget
across frames via `id = "%(x):%(y):%(text)"`.

Three deliberate gaps:

1. **Retained widget tree.** Immediate mode handles HUDs and
   pause menus comfortably but breaks down for deep menu trees
   with focus management, layout, and animations. A separate
   package (`@hatch:menu`?) lands the declarative shape —
   widget hierarchy, layout solver (likely flexbox-shaped), and
   lifecycle hooks (`onMount`, `onUnmount`, `onUpdate`). HUD's
   immediate-mode surface stays unchanged.
2. **Bitmap-font import.** `BitmapFont.fromImage(img, {
   glyphWidth, glyphHeight, first, cols })` so games with brand
   typography swap the built-in 5×7 default. Trivial layer on
   top of Renderer2D's `drawSpriteUV` — each glyph is one UV
   slice into a font atlas image. Pairs with `@hatch:image` for
   loading.
3. **Gamepad navigation.** Plumb a focus pointer through
   registered widgets so D-pad / left-stick selection drives the
   same code that mouse clicks do. Needs button registration to
   record widget order per frame, plus a focus state on `HUD`
   (separate from the existing hover state). The window plugin
   already surfaces gamepad events; this is HUD-side wiring.

### Shadows — cascades, point/spot, depth-convention audit

`@hatch:gpu/Renderer3D` now ships directional-light shadow mapping:
opt-in `enableShadows({size, extent, near, far, bias, pcfRadius})`,
a depth-only `SHADOW_WGSL_` vertex pass, a fallback 1×1 depth
texture bound when shadows are disabled (the PBR shader gates
sampling on `counts.w == 0`), and 3×3 PCF in
`shadow_factor()` via the new `wlift_gpu` comparison-sampler
binding (`samplerType: "comparison"`). Only the *first*
shadow-casting `DirectionalLight` is honoured per frame
(`addDirectional(..., castsShadows: true)` inserts at slot 0).
Three deliberate gaps:

1. **Cascaded shadow maps.** Outdoor scenes covering thousands
   of metres need 3–4 cascade frusta layered along the view
   direction so shadow texels stay dense near the camera and
   sparse at distance. The single-map current path produces
   visible aliasing past ~50 m. Extension is well-trodden:
   replace the single depth texture with a texture array,
   compute per-cascade `light_vp` matrices from a sliced view
   frustum, pick the cascade in the fragment shader by sampling
   depth and looking up the slice.
2. **Point + spot shadows.** Point lights need a cube-map shadow
   (six face renders per light) or dual-paraboloid; spots use a
   perspective projection (one render). Both follow the same
   `texture_depth_*` + `textureSampleCompare` pattern. Per-light
   shadow toggle goes on `PointLight` / `SpotLight` symmetric
   with the existing `DirectionalLight.castsShadows`.
3. **Depth-convention audit.** `Mat4.ortho` / `Mat4.perspective`
   in `@hatch:math` emit the GL-style `[-1, 1]` z range, but
   WebGPU's clip-space z is `[0, 1]`. Existing pipelines work in
   practice because the renderer always renders both passes
   through the same projection (so the comparison stays
   self-consistent), but a Reverse-Z / depth-precision audit
   would close the issue formally. The fix is a `Mat4.orthoWebGPU`
   / `perspectiveWebGPU` variant that emits the `[0, 1]` mapping
   directly and a per-pipeline flag picking which one to use.

### Post-processing — DoF, motion-blur

`@hatch:game/chain` ships the orchestration primitive (`PostFX` +
`PostPass`); concrete effects live in `@hatch:postfx`. Six effects
shipped: `Tonemap`, `Vignette`, `FXAA`, `ColorGrade`,
`ChromaticAberration`, `Bloom` (mip-pyramid additive). `PostPass`
already exposes the hooks for the harder remaining ones —
`stepCount`, `requestTargets`, `wantsDepth`, `dispatchStep_` — and
`wlift_gpu` now parses pipeline `blend` descriptors
(`"alpha" | "additive" | "premultiplied" | { color: {...}, alpha: {...} }`)
so additive-accumulating chains build cleanly.

Two open gaps:

1. **Depth-of-field.** Needs the chain's `wantsDepth` hook (which
   exists), a CoC pre-pass, a separable blur (two-step), and a
   composite. ~3-pipeline effect; the composite step uses the
   same blend-descriptor wiring Bloom relies on so its soft-focus
   transition lands cleanly.
2. **Motion blur.** Needs per-frame previous-frame camera matrix
   threaded through `Game.run` so the velocity-buffer pass can
   reconstruct per-pixel motion vectors. Independent of the
   chain — its own piece of `Game.run` plumbing alongside `g.dt`
   / `g.elapsed`.

GPU integration tests stay limited to the Wren-side config
surface (parameter binding, uniform-write byte layout,
fragment-body sanity); the shader / pipeline / render path is
exercised end-to-end only by a running game. A headless wgpu test
fixture that compiles each WGSL shader without dispatching frames
is the realistic add — bigger than this initiative warrants on
its own.

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
