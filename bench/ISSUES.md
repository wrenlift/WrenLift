# Benchmark Issues & Remediation Plan

Status snapshot. Re-run against standard Wren 0.4.0 and LuaJIT to keep
the head-to-head numbers honest.

## Current Results

### CI (ubuntu-latest x86_64, best of 10 runs — 2026-04-18)

| Benchmark      | WrenLift  | Wren 0.4 | LuaJIT   | LuaJIT (-joff) | Ratio vs Wren | Ratio vs LuaJIT |
|----------------|-----------|----------|----------|----------------|---------------|-----------------|
| Recursive Fib  | 0.0144s   | 0.208s   | 0.010s   | 0.068s         | 14.4x faster  | 1.4x slower     |
| Method Call    | 0.0550s   | 0.102s   | 0.007s   | 0.125s         | 1.9x faster   | 7.9x slower     |
| Binary Trees   | 0.533s    | 0.995s   | 1.045s   | 1.373s         | 1.9x faster   | **faster**      |
| DeltaBlue      | (on CI)   | (on CI)  | n/a      | n/a            | —             | —               |

### Local (Apple M3, release build, best of 3 runs — tiered mode)

| Benchmark      | WrenLift  |
|----------------|-----------|
| Recursive Fib  | 0.008s    |
| Method Call    | 0.089s    |
| Binary Trees   | 0.88s     |
| DeltaBlue      | 0.20s     |

History (aarch64 tiered):
- Oct 2025: HashMap registers + tree-walking interpreter — 3.5x / 4.1x slower than std Wren.
- Dec 2025: Vec registers + bytecode VM + method cache — caught up to std Wren.
- Mar 2026: Cranelift JIT backend + OSR — 14x faster than std Wren on fib.
- Apr 2026: Method OSR + nested-caller OSR + trivial setter / List fast paths + leaf IC
  context elision + Cranelift return-type coercion — current numbers above.

---

## Resolved (closed)

All four of the original P0–P2 issues from the Oct/Dec 2025 audit are fixed.

- **`HashMap<ValueId, InterpValue>` register file** → `Vec<Value>` with
  `UNDEF` sentinels, reused via a per-VM register pool.
- **String-allocating dispatch hot path** → SymbolId comparisons + pre-interned
  `static:` and `[_,...]=(_)` signatures.
- **Double method lookup per call** → single `find_method_with_class`.
- **`Vec` allocation per method call** → `SmallVec<[Value; 8]>`.
- **Block-param binding extra allocation** → direct pre-bind via
  `BytecodeFunction::param_offsets`.
- **SIGABRT under binary_trees GC pressure** → thread-local JIT root stack
  + shadow-frame rooting for live boxed values; covered by
  `e2e_tiered_nested_nonleaf_closure_call_survives_gc_pressure`,
  `e2e_bench_gc_pressure_10k`, and
  `e2e_tiered_backedge_osr_survives_gc_pressure`.
- **`super(x)` arity bug** — MIR builder now counts super-call args, not the
  enclosing constructor's params.
- **Implicit `this.member` resolution** — resolver rewrites bare
  identifiers as `this`-dispatched calls when no local/upvalue/module
  binding is found. DeltaBlue relies on this pattern throughout and runs.
- **Cranelift return-type mismatch on direct f64 returns** — the Return
  terminator now coerces the live Cranelift type to the function's
  declared return type. Silenced the three `COMPILE ERR FuncId(...)`
  lines on DeltaBlue and unblocked
  `jit_exec::test_jit_exec_f64_add` / `test_tier_up_compiles_function`.

---

## Open

### Issue A: Method dispatch overhead vs LuaJIT on `method_call`

**Severity:** Performance
**Gap:** ~8x slower than LuaJIT on CI; ~1.9x faster than standard Wren.

Cranelift already devirtualizes `kind=1` monomorphic IC sites via
`CallKnownFunc`, which emits a class-check + direct `call_indirect` with
zero FFI. The remaining cost is in the Call sites that were *not*
populated by the interpreter before the module tiered up — those fall
back to `wren_call_N`, which walks the method cache, saves/restores the
48-byte JIT context and bumps `jit_depth`.

Prior attempts:
- Inline IC class-check + direct call inside Cranelift for all call
  sites with a live IC pointer — crashed DeltaBlue with SIGSEGV under
  GC because the native fast path skipped `push_jit_frame`, leaving the
  caller invisible to the stack walker.
- Leaf fast path in `call_closure_jit_or_sync` — no effect on
  `method_call` because that path isn't hot for the module's own
  call sites.
- Leaf fast path in `dispatch_method` — broke DeltaBlue correctness
  (`14097848` vs `14065400`). Some call path reads `current_func_id`
  through a helper I didn't trace, despite `is_mir_inline_safe`
  claiming a leaf doesn't need it.

**Plan:**
1. Teach the native fast path to register a JIT frame. Easiest route is a
   `#[naked]` `wren_call_ic_N` wrapper that captures fp/ret_addr the same
   way `wren_call_N` does.
2. Re-attempt the inline Cranelift IC check on top of the naked wrapper
   so a class-match picks up the stable `jit_code[func_id]` slot and
   bypasses `wren_call_N` entirely.
3. Alternatively: after the module is installed, detect that the IC
   table has crossed a populated-ratio threshold and kick off a
   re-compile so the devirt pass catches the newly-hot call sites.

**Files:** `src/codegen/cranelift_backend.rs`, `src/codegen/runtime_fns.rs`.

### Issue B: `delta_blue` still 2x std Wren on aarch64

**Severity:** Performance
**Gap:** 0.20s vs ~0.09s.

`execute()` OSRs into native but its inner calls (`addPropagate`,
`chooseMethod`, `addConstraintsConsumingTo`) aren't hot enough
individually to hit the tier-up threshold per-call, so they execute
bytecode inside a native caller. The `jit_depth > 0` gate is already
lifted, so nested OSR is available — the bottleneck is that each
short method call pays `wren_call_N` overhead.

**Plan:** overlaps with Issue A. Specifically: a faster non-leaf
native-from-native dispatch (which tier-stats call `native2native`)
would let the hot protocol methods stay native once compiled instead
of stepping back into the interpreter.

### Issue C: `delta_blue` in the CI manifest

**Severity:** Tooling
**Status:** Landed in `f65e8ea` — CI now runs DeltaBlue alongside
`fib`, `method_call` and `binary_trees`. LuaJIT / Ruby / Python ports
don't exist yet, so only wrenlift and wren_cli numbers show up for
that row.

### Issue D: dominance-violation verifier errors on three DeltaBlue functions

**Severity:** Bug — performance-only (runtime stays correct via fallback)
**Status:** open

DeltaBlue prints three `COMPILE ERR FuncId(N): Compilation error: Verifier errors`
lines on startup. With `WLIFT_CL_VERIFY=1` the detailed messages are of the
form:

```
- inst94 (jump block17(..., v40, ...)): uses value v40 from non-dominating inst32
```

i.e. the Cranelift IR we emit contains a block argument that was defined in
a block that doesn't dominate the jump. The affected functions stay on the
bytecode interpreter, which costs a bit of perf but produces the correct
answer (`14065400`).

Previous related fix (commit in progress): the `Terminator::Return` emitter
now coerces the live Cranelift type to the function's declared return type,
which cleared the `result 0 has type f64, must match function signature of i64`
class of errors on `jit_exec::test_jit_exec_f64_add` and
`test_tier_up_compiles_function`. The dominance errors are a separate class —
likely in the inline IC / fast-path merge paths that introduce extra blocks.

**Reproduce:** `WLIFT_CL_VERIFY=1 ./target/release/wlift bench/delta_blue.wren --mode tiered`

**Next step:** dump the full IR for one failing function, identify which
emitter path creates the cross-block value use, and re-route the live-in
through a block param.

---

## Priority Order

| Priority | Issue | Why |
|----------|-------|-----|
| **P0**   | Issue A (inline IC with frame rooting) | Biggest headroom vs LuaJIT on method-heavy benches |
| **P1**   | Issue B (native2native dispatch)       | Unblocks DeltaBlue and any non-leaf protocol-heavy workload |
| **P2**   | Port `delta_blue.{py,rb,lua}`          | Makes Issue C a fair multi-language comparison |
