# Runtime Fix Plan

Phased plan to clear the runtime quirks blocking hatch development.
Phases are ordered by *leverage*: fix the foundational / upstream bugs
first because they often have downstream manifestations that disappear
for free. Each phase has a single theme so that a fix can land
independently.

The closure-mutated-upvalue fix (commit `b74c653`) was a good case
study: one MIR/CSE change retired a memory entry, three workarounds
in `@hatch:game`, and several ad-hoc patterns we'd been carrying
around in the framework code. Expect more of this as we work down.

---

## Phase 0 — Closure-mutated upvalue (DONE)

Status: **fixed (commit `b74c653`, 2026-04-26)**

CSE was merging two `subscript_get v2[0]` reads across an intervening
`Call`. Fix splits the seen-instructions cache into pure / memory
buckets; side-effecting instructions clear the memory bucket. See
[QUIRKS.md](../QUIRKS.md#closure-mutated-outer-locals-were-frozen-at-the-first-calls-value).

Downstream items potentially also resolved (need re-validation):

- `Fiber.try()` cross-fiber resume failures (memory:
  `project_nested_fiber_resume.md`).
- `for-in` body upvalue clobber that forced the `lastTime` /
  `startTime` workaround in `@hatch:game`.
- `@hatch:web` Stylesheet tier-up miscompile (memory:
  `project_web_jit_miscompile.md`) — same shape: closure-captured list
  mutated inside, read by an outer that gets stale CSE'd value.

Action: re-run each of these with the current CSE and update QUIRKS /
memory entries accordingly. If any persists, surface it as a new
phase rather than reopening this one.

---

## Phase 1 — `for-in` iterator family

Status: **fixed (commit `1441d38`, 2026-04-26)**

All three quirks shared one root cause: `lower_for` emitted the
`seq.iterate(iter_param)` advance only on the natural fall-through
path, and `continue` branched directly to `cond_bb` while shedding
the iterator-state arg. The latch block introduced in this commit
takes the same `[iter_phi, …vars_phi]` shape as `cond_bb` and
absorbs every `continue` jump; natural fall-through stays inline so
the hot path is unchanged. `bench/delta_blue.wren --mode tiered`
now runs without `COMPILE ERR FuncId(...)` fallback messages.

Outstanding tail: walk the `while (i < n)` workarounds in
`@hatch:fp::dropWhile`, `@hatch:json`, `@hatch:game`'s event drain,
and `@hatch:path`, restore the natural for-in/continue shape, and
keep the rewrites only where they're independently clearer.

Three open quirks all touch how the bytecode lowering of `for (e in
seq) { ... }` handles iterator state across `continue`, loop exit, and
the safepoint that fires on the back-edge:

1. **`for-in` with `continue` corrupts the next iteration's binding**
   (QUIRKS open #3). Workaround: rewrite as `while (i < xs.count) {
   var p = xs[i]; ... }`. We've taken this workaround across
   `@hatch:path`, `@hatch:json`, and most recently `@hatch:game`'s
   event drain. It's tax on every consumer.

2. **MIR `continue` inside a nested-if inside a `for-in` miscompiles**
   (memory: `project_mir_continue_in_nested_if.md`). Same root area;
   probably the same bug seen from the bytecode lowering side.

3. **`for-in` iterator variable leaks past the loop** — Cranelift
   verifier rejects 3 functions in `delta_blue` tiered mode (QUIRKS
   open #1). We pay interpreter dispatch on those fallbacks.

These are most likely one bug, one fix.

### Plan

1. Reproduce all three with a tight standalone repro for each, save
   under `tests/fixtures/for_in_*.wren`.
2. Compare lowered MIR vs intended SSA: the iterator state
   (`iter` + `e`) needs to be a block parameter on the loop header,
   reset by the `continue` edge to the latest `iter` value. Suspect
   the MIR builder is reusing a single `ValueId` for `iter` across
   `continue` edges, leaving the loop-header phi argument list
   inconsistent.
3. Land a regression test for each.
4. Re-validate the Cranelift verifier complaints against the fixed
   MIR — if they persist, dig into the legacy verifier dump under
   `WLIFT_TRACE_VERIFY=1`.

### Exit criteria

- All three QUIRKS open entries close.
- `bench/delta_blue.wren` runs in tiered mode without `COMPILE ERR`
  fallback messages, on x86_64 and aarch64.
- The `for-in` workaround comments scattered through `@hatch:*` come
  out (worth a second commit for hygiene).

---

## Phase 2 — FFI ABI safety hardening

Status: **fixed (commit `c22768c`, 2026-04-26)**

`WLIFT_PLUGIN_ABI_VERSION` lives in `runtime::foreign` and starts at
v1. Every cdylib plugin (`wlift_gpu`, `wlift_window`, `wlift_image`,
`wlift_audio`, `wlift_physics`, `wlift_sqlite`) exports
`wlift_plugin_abi_version()` → `u32`. The host calls it immediately
after `dlopen`; a mismatched (or missing — read as v0) value short-
circuits `load_library` with a clean `ForeignLoadError::AbiMismatch`
that names both versions in its message. Test coverage in
`runtime::foreign::tests` covers the message format and the `"self"`
sentinel exemption. Bump the constant whenever `NativeContext` or
the `ForeignCFn` shape changes; the docstring carries the bump
rationale.

Stale plugin dylibs SIGSEGV silently when the host runtime's vtable
shape moves underneath them. Diagnosed once by accident this session;
guaranteed to bite again unless we put a tripwire at the FFI
boundary.

### Plan

1. Mint a plugin ABI version constant in `wren_lift::runtime::object`
   (or a new `runtime::abi` module). Bump it every time
   `NativeContext` or related shared traits gain / lose a method.
2. Have every `cdylib` plugin export `wlift_plugin_abi_version()` →
   `u32`. Generate this via a `#[wlift_plugin]` proc-macro or a tiny
   declarative helper in the plugin crate.
3. The host calls `wlift_plugin_abi_version()` immediately after
   `dlopen()` and refuses to bind any symbols if the version doesn't
   match — surface as a runtime error, not a crash:

   > Plugin `wlift_gpu` was built against ABI v3, runtime expects
   > v5. Rebuild the plugin against the current host.

4. Optional follow-up: stamp a Cargo build-time hash of the
   `NativeContext` trait into the version constant so we can't forget
   to bump on a quiet ABI change.

### Exit criteria

- Mismatched dylib produces a clean runtime error with the version
  pair in the message.
- All in-tree plugins (`wlift_gpu`, `wlift_window`, `wlift_image`,
  `wlift_audio`, `wlift_physics`, `wlift_sqlite`) export the version
  symbol and load cleanly under the matching host.

---

## Phase 3 — Tiered / JIT correctness

Status: **mostly fixed downstream of Phase 0 / 1; one new sub-issue
deferred**

| Item | Status |
|---|---|
| `Null does not implement 'view'` in sprite-grid tiered | Fixed (downstream of for-in / CSE — sprite-grid runs cleanly under `--mode tiered` post-1441d38) |
| `@hatch:web` Stylesheet IC tier-up miscompile | Likely fixed (downstream of CSE memory-cache fix — chat builds + tiers without IC errors) |
| OSR entry "undefined value v53" | Not reproducing (30/30 clean runs of `template.spec.wren` in tiered) |
| Constructor JIT dispatch SIGSEGV under GC pressure | Still disabled (deferred) |
| **NEW: `@hatch:web` `App.listen` hangs in tiered mode** | Logged in QUIRKS as open — separate root cause, not downstream of Phase 0 / 1 |

Group of bugs that fire only under `--mode tiered`, where a hot
function gets compiled and the JIT dispatch path mis-handles
something the interpreter handled correctly.

1. **`Null does not implement 'view'` in `@hatch:game` tiered mode**
   (newly observed, sprite-grid). Surfaces at
   `surface.acquire().view`. Suspect: tiered IC for the `acquire`
   path is returning a `Null` frame, or the `view` lookup is hitting
   a stale IC entry.

2. **`@hatch:web` Stylesheet.add tier-up miscompile** (memory:
   `project_web_jit_miscompile.md`). Re-validate after the CSE fix —
   that's where I expect at least partial progress.

3. **OSR entry "undefined value v53"** (memory:
   `project_osr_entry_flake.md`). Inconsistent reachability in
   `osr_entry_layout`; needs a deterministic repro.

4. **Constructor JIT dispatch SIGSEGV under GC pressure** (memory:
   `project_cranelift_fixes.md`, currently disabled). Re-enable
   behind a feature flag so it can be tested without flipping the
   default.

### Plan

1. **Re-validate** post-CSE-fix: rerun the four scenarios. If any
   pass now, retire from this phase and update memory entries.
2. **Triage** the survivors: capture a deterministic repro and a MIR
   dump for each. The new ones (`view` null) should be smallest
   first.
3. **Common toolchain**: every fix in this phase ships with at least
   one e2e test under `tests/e2e.rs` parametrized over
   `ExecutionMode::Interpreter` *and* `ExecutionMode::Tiered`, so we
   can't regress the interpreter path while fixing tiered.
4. **Constructor JIT** is the last item — don't try to re-enable
   until 1-3 are clean, since a SIGSEGV crashes every other tier-up
   test running in parallel.

### Exit criteria

- Every game example runs cleanly under `--mode tiered` for at least
  60 frames.
- `@hatch:web` chat / counter / hello run in `--mode tiered` (without
  the `--mode interpreter` workaround that the chat example has
  documented).
- OSR entry test stops being flaky over 100 consecutive runs.
- Constructor JIT either re-enabled and passing GC-pressure tests, or
  the disable is documented as a permanent design choice.

---

## Phase 4 — Method dispatch edge cases

Status: **open**

Two bugs whose symptoms diverge but share the dispatch path:

1. **`Fiber.try` doesn't catch "does not implement" errors** (QUIRKS
   open #4). Process aborts; `fiber.error` not set. Native runtime
   errors and `Fiber.abort` are both caught, so the trap door is
   specifically the method-not-found path raising through
   `Fiber.try`.

2. **`obj.name` via `Meta.compile` dispatches to `Class.name`**
   (QUIRKS open #5). Closure compiled through `Meta.compile` and
   invoked from a class method's body resolves bare-identifier
   getters against the class metaobject rather than the instance.
   Renaming the getter sidesteps it; reproducible without `--no-opt`.
   Blocks `@hatch:json`'s `#json` attribute path.

### Plan

1. (1) — `Fiber.try` raise path. The runtime already routes
   `runtime_error` from native through the fiber's `try` slot; the
   method-not-found path takes a different exit (likely
   `process::exit` or a `Fiber.abort` that's fired *after* the fiber
   was popped). Trace the exact code path that raises "X does not
   implement Y" and route it through the same `set_fiber_error`
   sequence the native path uses.

2. (2) — `Meta.compile` resolution. The meta-compiled closure inherits
   a sema environment that's resolving bare `obj.name` against the
   *enclosing module's* class space, not the receiver. Investigate
   whether `Meta.compile` is producing the closure with the wrong
   `self` context, or if the IC for the closure's call site is
   caching a method from `Class` (the metaclass) instead of the
   instance class.

### Exit criteria

- `Fiber.try` catches `B.new().missing()` and surfaces it via
  `fiber.error`.
- `@hatch:json`'s `#json` attribute path runs end-to-end without
  rename gymnastics.

---

## Phase 5 — Niche / single-site regressions

Status: **open**

Lower-priority because each affects one library and has a narrow
trigger surface:

1. **`JSON.parse` fails on second HTTP response body in tiered mode**
   (QUIRKS open #2). 3/9 hatch-http e2e failures stem from the
   "second request" shape.

2. **Diamond dependency false-positive cycle** in `hatch build`
   (memory: `project_diamond_dep_cycle_false_positive.md`). Not a
   runtime bug, but the workaround (drop the redundant declaration)
   is annoying and surprises every example author. Should still be
   fixed.

### Plan

1. (1) Likely related to Phase 3 IC freshness or compiled-frame
   reuse. Defer until Phase 3 is clear; re-test, then dig if the bug
   survives.

2. (2) Tiny resolver fix in `src/hatch.rs:build_recursive` — the
   `visited` HashSet should short-circuit *successfully* on
   already-built paths, not error out. The current behaviour is the
   "encode: dependency cycle detected" message every example
   developer hits. Fix is one line plus test.

### Exit criteria

- `hatch build` accepts diamond deps without dropping declarations.
- `@hatch:http` e2e suite is fully green in tiered mode.

---

## Phase 6 — Optimizer infrastructure (deferred)

Status: **roadmap only** — see [QUIRKS.md → Roadmap](../QUIRKS.md#roadmap).

Per-function memory-effect summaries, propagated through the call
graph, would let CSE / LICM / DCE keep more loads hoisted past
known-pure callees. Significant pass; defer until a benchmark
surfaces real cost from the conservative bucket-flush we ship today.

---

## Working principles

- **One repro per fix.** Land a test before the fix, in the smallest
  Wren snippet that reproduces. Several quirks above are tracked
  in memory but lack a checked-in repro; convert as we touch them.
- **Validate both modes.** Every runtime test runs in
  `Interpreter` and `Tiered`, even if the bug only presents in one.
- **Memory entries close, never linger.** When a phase fix retires
  a memory entry, update the entry to status FIXED with the commit
  hash and link to the QUIRKS section. Don't delete; future grep
  for "why was X done this way" still wants the history.
- **Re-validate downstream after foundational fixes.** Every time a
  Phase 1 / Phase 2 / Phase 3 item lands, walk back through the
  later phases' suspects to see what fell out for free.
