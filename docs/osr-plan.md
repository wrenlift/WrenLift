# OSR Implementation Plan

This plan tracks the move from back-edge tier-up polling to real on-stack
replacement. The goal is to let hot loops transfer from bytecode interpretation
into native code without re-running function entry code.

## Phase 0: Safety Baseline

Status: complete for the initial safety baseline.

- Keep back-edge tier-up polling and compile-queue draining.
- Do not call the normal JIT function entry from a back-edge.
- Guard against regressions where OSR re-runs pre-loop module setup.

Implemented so far:

- Restart-from-entry OSR is disabled in the back-edge safepoint.
- `e2e_tiered_backedge_does_not_restart_module_entry` covers duplicated setup.

## Phase 1: OSR Metadata

Status: complete for unconditional bytecode back-edges.

- Record candidate OSR points while lowering bytecode.
- For each unconditional backward branch, preserve:
  - bytecode branch offset
  - bytecode target offset
  - target MIR block id
  - external live-in registers needed by the reachable loop/header region
  - target block-param destination registers
- Use this metadata later to map interpreter state to a compiled loop-header
  entry.

Implemented so far:

- `BytecodeFunction::osr_points` records candidate backward branches.
- `test_lower_records_backedge_osr_point` covers metadata generation.
- `test_lower_records_external_osr_live_in_before_block_params` covers the OSR
  ABI order: external live-ins first, then target block params.

## Phase 2: Native OSR Entry Artifacts

Status: complete for conservative value-typed top-level entries.

- Extend the native compile artifact with optional OSR entry pointers.
- Teach the Cranelift backend to compile an alternate entry for a target MIR
  block.
- The OSR entry should receive the live header params as normal NaN-boxed i64
  arguments, bind them to the target block params, and jump into the compiled
  loop/header block.
- Non-constant values used by the loop but defined outside the reachable loop
  region are passed as explicit OSR arguments. Constants are rematerialized in
  the OSR wrapper.
- Start with top-level/module functions and unconditional backward branches.

Implemented so far:

- Native artifacts and executable functions can carry OSR entry metadata.
- The execution engine has active, baseline, and optimized OSR-entry caches.
- Cranelift emits conservative loop-header OSR entries for unconditional
  backward branches.
- OSR wrappers can materialize external numeric/bool/null constants from loop
  preheaders before jumping to the target header.
- OSR wrappers receive value-typed external live-ins explicitly, avoiding
  re-reading mutable module state at the transfer point.

## Phase 3: Interpreter Transfer

Status: complete for top-level/module bytecode frames.

- At a back-edge safepoint, look up the matching `OsrPoint`.
- Poll/install compiled code as today.
- If a compatible OSR entry exists:
  - save the current frame state
  - collect live values from `param_regs`
  - set `JitContext` using the same save/root/restore discipline as native
    entry
  - call the OSR entry
  - restore JIT context, depth, and root snapshot
  - pop or resume the interpreter frame based on the native result
- Preserve closure and defining-class context before enabling method OSR.

Implemented so far:

- Bytecode back-edge safepoints look up matching `OsrPoint` metadata after
  compile-queue polling.
- Top-level/module frames can transfer into compiled loop-header OSR entries.
- The transfer path saves interpreter frame state, installs a JIT context,
  roots the active fiber and saved context pointers, restores JIT depth/context,
  then completes the current interpreter frame with the native result.
- OSR transfer now accepts object-valued live params for eligible top-level
  frames; the suspended MIR frame and rooted fiber keep those values visible to
  GC.
- OSR transfer is still gated to frames without closure/defining-class context,
  and when already inside native JIT depth; method/closure continuation handling
  needs a separate return/resume path.
- `e2e_tiered_backedge_enters_osr_entry` covers that an actual OSR entry is
  taken for a hot module loop.

## Phase 4: Eligibility Expansion

Status: in progress.

- Enable method/closure OSR once context preservation is proven.
- Add a continuation-safe method/closure OSR path for loops reached from native
  callers.
- Add GC stress coverage for OSR transfers.
- Add deopt or fallback rules for unsupported fiber actions.
- Consider conditional-back-edge OSR after unconditional loop-latch OSR works.

Implemented so far:

- `should_compile_osr_entries` now compiles loop-header OSR artifacts for any
  function with a backward branch, not just `<module>` top-level frames. The
  per-block `osr_entry_layout` analysis still filters unsupported live-in
  shapes.
- `try_enter_loop_osr` no longer rejects method/closure frames outright. The
  env var `WLIFT_DISABLE_METHOD_OSR` keeps a fast kill switch while we build
  out deopt/fallback coverage.
- The bytecode interpreter's inline frame push now keeps the local `func_id`
  in sync with the active callee frame. Without this, back-edge tier-up and
  OSR-entry lookups targeted the caller instead of the method running the
  loop.
- The back-edge safepoint reads closure / defining_class / module_name /
  return_dst from the current frame before calling `try_enter_loop_osr`, so
  inline-push call sites can't leave stale context for the transfer.
- `e2e_tiered_backedge_enters_osr_entry_in_method` covers a hot method loop
  taking an OSR entry.
- `e2e_tiered_backedge_osr_survives_gc_pressure` covers a method loop that
  allocates per iteration, exercising GC pressure while OSR is active.
- Threaded dispatch now preserves OSR safepoints conservatively: when tiered
  method OSR is enabled, functions with bytecode OSR points stay on the
  bytecode path instead of entering the threaded interpreter.
- The native JIT frame stack used by GC stack walking is thread-local, avoiding
  cross-test contamination when e2e tests run in parallel.
- Continuation-safe OSR: the `jit_depth > 0` hard reject is gone. OSR now
  transfers into native even when the interpreter is already nested inside a
  native caller (e.g. a hot inner method called from a natively-running
  module loop). The existing save/restore discipline — JIT context snapshot,
  closure/class rooting, fiber root, depth bump/restore — already covered
  nested re-entry; the gate was over-conservative. `WLIFT_DISABLE_NESTED_OSR`
  keeps a kill switch.
- `e2e_tiered_backedge_osr_nested_inside_native_caller` covers the nested
  case: a native module loop calling a hot method whose inner loop OSRs.

Known gaps:

- The threaded interpreter path does not yet poll back-edges for tier-up or
  OSR transfer. In tiered mode, threaded dispatch now falls back to bytecode
  for functions whose bytecode contains OSR points, preserving safepoints for
  hot method loops. The broader fix is still to fold back-edge polling directly
  into the threaded interpreter.
- Conditional-back-edge OSR, deopt paths, and explicit fallback rules for
  unsupported fiber actions remain pending.

## Phase 5: Performance Validation

Status: in progress.

- Compare `interpreter`, `tiered`, `jit`, and standard Wren for:
  - `bench/method_call.wren`
  - `bench/binary_trees.wren`
  - `bench/delta_blue.wren`
  - `fib`
- Keep correctness gates ahead of benchmark-driven changes.

Implemented so far:

- Cranelift OSR artifacts are emitted for functions with backward branches,
  with `osr_entry_layout` filtering unsupported live-in shapes.
- Cranelift method-call slow paths use plain method symbols again instead of
  packed call-site IC indexes. The packed path regressed `binary_trees` and made
  `delta_blue` crash or hang under broader tiered compilation.
- Current spot checks (2026-04-17, after Phase 4 nested-OSR):
  - `bench/fib.wren --mode tiered`: about 0.008s
  - `bench/delta_blue.wren --mode tiered`: `14065400`, about 0.25s
  - `bench/binary_trees.wren --mode tiered`: about 0.83s
  - `bench/method_call.wren --mode tiered`: about 0.30s
- Nested OSR does not move these benchmarks on its own: the inner methods in
  `delta_blue` back-edge too few times per call to reach the tier-up
  threshold, so opening the `jit_depth > 0` gate unlocks a capability without
  showing up in current bench output. It lets future work (in-threaded
  back-edge polling, lower per-method thresholds for OSR-eligible bodies)
  actually cash in on nested loops.
