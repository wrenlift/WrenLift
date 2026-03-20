# Tiered Runtime Status

This note documents the tiered-runtime work completed so far, the current
verified state of the benchmark suite, and the remaining blockers.

## Starting point

The work started from `bench/delta_blue.wren` failing under `wlift` while
running correctly under `wren_cli`. Early symptoms included incorrect field
reads in `ScaleConstraint.execute()` and broader instability once tiered
promotion reached non-leaf functions.

That investigation grew into a larger runtime/JIT architecture pass because the
real issue was not just one field access. The tiered runtime did not yet have a
coherent contract for:

- non-leaf native execution
- GC-visible native roots
- safe fallback between compiled code and interpreter code
- profitable native-to-native call paths
- explicit telemetry for promotion and fallback behavior

## Major work completed

### 1. Native frame metadata and shadow-stack scaffolding

Added backend metadata describing boxed values and native safepoints, plus
runtime support for native shadow roots.

Relevant commits:

- `27f8c8c` `Add native frame metadata and shadow stack scaffolding`
- `aa59e02` `Gate non-leaf native tiering on spill-safe functions`
- `89c0031` `Continue tiered non-leaf GC debugging`
- `615bfda` `Fix generational tiered delta_blue correctness`
- `cc048fc` `Improve spill handling for safe non-leaf tiering`

What this accomplished:

- preserved native-frame/root metadata during tier-up
- added runtime shadow-root storage and GC scanning/write-back
- fenced non-leaf native execution to subsets that were at least spill-safe
- moved `delta_blue` failures from memory corruption toward specific semantic
  and runtime bugs

### 2. Correctness fixes for generational tiered `delta_blue`

Fixed several GC/runtime gaps that only appeared once non-leaf tiered code was
active.

Examples of fixes in this phase:

- rooted saved `JitContext` closure/defining-class state across nested calls
- refreshed module-var pointers after interpreter fallback
- rooted list/map construction inputs across allocation
- added missing GC reachability for helper/wrapper classes
- added JIT `SetField` write-barrier plumbing
- restricted non-leaf native admission to actually safe cases

Net result:

- `bench/delta_blue.wren --mode tiered` is now correct again under the default
  generational GC
- expected output: `14065400`

### 3. Tiered shutdown hang fix

Identified and fixed a separate shutdown bug where the program would print its
result and then appear to run forever.

Root cause:

- the script had already finished
- `ExecutionEngine::drop()` blocked waiting for a live background compile thread
  via `join()`

Fix:

- shutdown now joins a compile thread only if it has already finished
- otherwise it detaches the live thread instead of blocking process exit

Relevant commit:

- `0635eca` `Refactor tiered runtime and detach compile thread on exit`

### 4. Tier-state / telemetry refactor

Restructured tiering around explicit tiers and counters instead of relying only
on timing guesses and JIT logs.

What changed:

- separated baseline-native and optimized-native states
- added explicit tier statistics
- added `opt_threshold`
- kept bytecode available as fallback
- added tracing for tier-up queue/start/finish/install

This made it possible to see when hot functions were:

- interpreted
- baseline-native
- optimized-native
- falling back to interpreter
- entering native from native callers

### 5. Recursive static-self lowering

Added dedicated lowering support for recursive static-self calls so hot
recursive methods like `Fib.calc(_)` could use a direct compiled path instead of
always routing through generic runtime dispatch.

Relevant commit:

- `ab07302` `Add recursive static self-call lowering`

This was an important architectural step, but it did not fully solve recursive
tiered correctness on its own. A later bug was found in the machine-code call
targeting for recursive self-calls.

### 5b. Recursive `CallLocal` entry fix

Tiered `fib` was later found to be semantically wrong even though it looked very
fast. The root cause was a bad recursive-call target:

- recursive `CallLocal` jumped to a label after the function prologue
- recursive calls skipped frame allocation
- recursive activations reused the same native stack frame
- spill slots were corrupted across recursive calls

Fix:

- added a dedicated function-entry label before the prologue
- recursive static-self calls now target that real entry point
- each recursive call re-executes the prologue and gets its own frame

Result:

- tiered `fib` is now correct again
- the `fib` speedup is now a valid result instead of a bogus artifact

### 6. Native module-entry and hot-call performance work

Most recent performance work focused on the fact that a top-level script only
executes once, so ordinary entry-count-based tiering never promotes it. That
meant hot loops in top-level benchmark code stayed interpreted even when the
callees were compiled.

Relevant commit:

- `918a40a` `Speed tiered module entry and native hot calls`

What changed:

- eager baseline compilation for eligible module-entry functions
- native execution path for the root/top-level frame
- native-originated tier-up for hot closure calls
- direct leaf native-to-native call fast path from runtime helpers
- fixed JIT `ConstString` lowering
- fixed JIT `StringConcat` lowering for multi-part interpolated strings
- added a heuristic to avoid eager native entry for constructor-heavy module
  bodies, since those still route through a slow temp-fiber bridge

## Current verified behavior

### `delta_blue`

Verified correct in default tiered mode:

- command:
  `./target/release/wlift bench/delta_blue.wren --mode tiered`
- expected output:
  `14065400`

### `binary_trees`

Currently runs to completion again in tiered mode and prints structurally
correct check lines after the constructor-heavy entry heuristic was added.

### `method_call`

This benchmark materially improved after native-originated tier-up and the leaf
native-to-native fast path landed.

Observed change during this work:

- before the latest hot-call fix: about `0.94s`
- after the latest hot-call fix: about `0.23s`

It is still slower than standard Wren, but it is no longer spending nearly all
of its time bouncing through interpreter-originated dispatch.

### `fib`

Correct result for `Fib.calc(28)`:

- `317811`

Verified outputs now agree:

- `wlift --mode interpreter` -> `317811`
- `wlift --mode tiered` -> `317811`
- `wlift --mode jit` -> `317811`
- `wren_cli` -> `317811`

## Benchmark status snapshot

Current benchmark picture after the recursive-entry fix:

- `fib`: `0.029s` vs `0.176s`
- `method_call`: `0.208s` vs `0.083s`
- `binary_trees`: `6.024s` vs `0.966s`
- `delta_blue`: `1.141s` vs `0.093s`

So the current trustworthy performance picture is:

- `fib` is a real win, about `6x` faster than standard Wren
- `method_call` improved a lot, but is still about `2.5x` slower than Wren
- `binary_trees` is still much slower than Wren
- `delta_blue` is correct, but still much slower than Wren

## Why the remaining performance gap exists

The runtime is better than it was, but several major architectural gaps still
remain:

- many non-leaf calls still go through generic Rust runtime helpers instead of
  direct patchable compiled-call stubs
- hot method calls still pay a heavy path like:
  `blr -> wren_call_N -> dispatch_call -> method lookup -> call_closure_jit_or_sync`
- constructor-heavy native entry is still a poor fit because constructors route
  through a temp-fiber bridge
- native call-site IC behavior is still much weaker than the interpreter's
  hottest paths in several benchmark shapes
- optimize-tier profitability and safety are still constrained by the current
  fallback/rooting model

## Highest-value remaining work

The most important remaining architecture work is:

- inline caches in JIT code for call sites
- direct JIT-to-JIT call stubs that skip Rust FFI dispatch
- small method inlining for getters/setters and similar leaf bodies
- constructor fast path that avoids temp-fiber allocation

## Recommended next steps

1. Add monomorphic JIT call-site ICs for normal method calls.
2. Replace the Rust trampoline path with direct compiled-call stubs on IC hits.
3. Inline trivial getters/setters into callers.
4. Add a native constructor fast path.
5. Continue reducing fallback-heavy non-leaf dispatch in `delta_blue`.

## Useful repro commands

Check `fib` correctness:

```sh
./target/release/wlift /tmp/repro_fib_verify.wren --mode interpreter
./target/release/wlift /tmp/repro_fib_verify.wren --mode tiered
./target/release/wlift /tmp/repro_fib_verify.wren --mode jit
wren_cli /tmp/repro_fib_verify.wren
```

Check current benchmark snapshot:

```sh
BENCH_RUNS=1 ./bench/run.sh
```

Check `delta_blue` correctness:

```sh
./target/release/wlift bench/delta_blue.wren --mode tiered
```
