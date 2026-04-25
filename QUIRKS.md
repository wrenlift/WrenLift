# WrenLift runtime / compiler quirks

Running list of places where WrenLift diverges from stock Wren, with
status and references to the fixing commit. New entries land at the
top; once an item ships, it keeps its row so the git log retains
rationale for anyone who reads just this file.

## Open

### `@hatch:web` `App.listen` hangs in tiered mode

Status: **open (workaround: `--mode interpreter`)**

```
$ wlift --mode tiered web-hello.hatch &
@hatch:web listening on http://127.0.0.1:3000
$ curl http://127.0.0.1:3000/                       # never returns
```

Server binds, prints the listening message, never responds to
HTTP. The TCP handshake completes (curl gets ESTABLISHED), but
the request body hangs in the kernel — `tryAccept` either never
fires or `serve_` hangs once it does. Adding Wren-level
`System.print` traces into the listen loop produced no output
even with curl connected, so the loop body doesn't enter the
`conn != null` branch.

`--mode interpreter` runs the same code correctly. Reproduces
under `--opt-threshold 999999999` (no JIT compilation), so the
divergence is in the tiered execution mode's interpreter path,
not in JIT'd code. Leading suspect: the back-edge tier-up
machinery on a `while (true)` accept loop somehow elides the
`listener.tryAccept` foreign call or `sched.spawn` side effect
even though both have `has_side_effects` set.

Smaller fiber tests (Fiber.new + .try() in a tight loop) pass
in tiered mode, so the bug is specific to the listen-loop
shape — likely the combination of an unconditional infinite
loop, a foreign-fn call returning null, and the cooperative
fiber scheduler.

### `JSON.parse` fails on second HTTP response body in tiered mode

Status: **open**

Two `Http.get`/`Http.post` calls in sequence, where the second's
body is parsed as JSON, reproduce:

```
p1 ok                                 // first parse fine
Runtime error: JSON: expected string key at offset 1
```

Minimal repro requires a real network call; `JSON.parse` on the
same body string typed inline passes. Smells like a JIT state
leak around `String` operations that follow a large native-side
allocation (ureq's response body). `--opt-threshold 100000`
doesn't clear it, so it's not purely opt-tier.

Workaround: run affected scripts with `--mode interpreter`. All
@hatch:http spec cases pass cleanly there; tiered mode exposes
3 of 9 failures (all on the "second request" shape).

### `Fiber.try` doesn't catch "does not implement" method-dispatch errors

Status: **fixed (commit 260633a, 2026-04-26)**

`Fiber.try` now routes method-not-found errors through the same
fiber-error catch the native-side `ctx.runtime_error` path already
used. The dispatcher converts the error message to a String,
stores it on `fiber.error`, marks the fiber `Done`, and resumes
the caller — same shape `Fiber.abort` already followed.

```wren
class B { construct new() {} }
Fiber.new { B.new().missing() }.try()   // process aborts; fiber.error not set
```

Regular `Fiber.abort` is caught. `ctx.runtime_error` from native
code is caught. But Wren's own "class X does not implement Y"
method-not-found runtime error propagates straight through
`Fiber.try`. Blocks any code that wants to probe for a method's
existence by attempting the call — `@hatch:json` worked around
it by documenting `toJson()` as a required hook rather than
optionally-checked.

### `obj.name` compiled via `Meta.compile` dispatches to `Class.name`

Status: **not currently reproducing (likely fixed downstream of
Phase 0 / 1 — regression test landed in commit 260633a)**

The original symptom (a `Meta.compile("return Fn.new { |obj|
obj.name }").call()` invocation returning `"Class"` when called
from inside another class's static method) doesn't reproduce in
the current build. Locked in
`e2e_meta_compile_closure_dispatches_through_receiver` so a
re-occurrence flags loudly.

Original observation kept below for context — re-open this entry
if anyone hits the symptom again on real `@hatch:json` `#json`
attribute code.

A closure compiled through `Meta.compile` and invoked from
inside another module's class method returns the *class's*
`name` rather than the instance's `name` getter:

```wren
var acc = Meta.compile("return Fn.new { |obj| obj.name }\n").call()
acc.call(Shape.new("alpha"))    // direct, at module scope: "alpha"  ✓
Enc.invoke(acc, Shape.new("b")) // inside a class static method: "Class"  ✗
```

Renaming the getter (`displayName`, `label`, etc.) sidesteps the
issue — the bug is specific to a small set of names that clash
with methods on `Class` itself (`name`, `type`, `toString` all
candidate suspects). Reproduces even with `--no-opt`, so not MIR
optimizer. Method cache / dispatch path is the next place to
look.

Impact: blocked `@hatch:json`'s attribute-driven (`#json`)
auto-serialization path from landing; the library ships with
the `toJson()` hook alone for v0.1.

## Fixed

### `for-in` + `continue` infinite-looped / corrupted next binding / failed Cranelift verifier

Status: **fixed (commit 1441d38, 2026-04-26)**

Three quirks, one root cause. `lower_for` emitted the iterator
advance call (`seq.iterate(iter_param)`) only on the natural
fall-through path. `continue` branched straight to the cond
block with `[…vars]` while cond_bb's params were `[iter_phi,
…vars_phi]` — so the iter slot got reassigned to whatever value
zipped first, and the iter advance never ran. Manifested as:

1. `for (p in xs) { if (cond) continue; … }` infinite-loops
   (caught by the step-limit) — same element bound forever.
2. `continue` inside a nested-if inside for-in MIR-miscompiles
   (memory: `project_mir_continue_in_nested_if.md`).
3. Cranelift verifier rejects three functions in
   `bench/delta_blue.wren` tiered mode with "uses value
   from non-dominating inst" — the malformed back-edge produces
   an SSA shape Cranelift refuses; `delta_blue` falls back to
   the interpreter for those frames.

```wren
for (p in ["foo", "", "bar"]) {
  if (p == "") continue
  System.print(p)            // before: prints "foo" then loops
}                            // after:  prints "foo" then "bar"
```

Fix: introduce a `latch_bb` between the body and the cond block
shaped `[iter_phi, …vars_phi]` like cond_bb. `continue` jumps
to the latch with `[iter_param, …current_vars]`; the latch
advances the iterator and branches to cond_bb. Natural
fall-through stays inline (no extra block transition on the hot
path), so the no-`continue` case has identical bytecode density
to before — the latch is reachable only when at least one
`continue` lives in the loop body, and the optimizer prunes it
otherwise. `continue_targets` gained a "leading args" field for
the iter_param; while-loop callers pass an empty vec.

Workarounds elsewhere in the codebase (the `while (i < n)`
rewrites in `@hatch:path`, `@hatch:json`, `@hatch:game`'s event
drain) can come out at any time; we'll let them go in a hygiene
pass once the framework code is otherwise stable.

### Closure-mutated outer locals were frozen at the first call's value

Status: **fixed (commit b74c653, 2026-04-26)**

```wren
var n = 0
var bump = Fn.new { n = n + 1 }
bump.call();  System.print(n)   // 1   correct
bump.call();  System.print(n)   // 1   wrong — should be 2
bump.call();  System.print(n)   // 1   wrong — should be 3
```

Inner-scope reads inside the closure correctly observed each
prior write (`inside: n=1, 2, 3`), but the outer scope's reads
collapsed to the first post-call value. The bug bit any code
that captured + mutated an outer local — counters, accumulators,
event-pumping helpers — and surfaced indirectly in
`@hatch:game`'s loop locals, in the `for-in` body upvalue
clobber, and in `Fiber.try` nested-resume failures.

Root cause: `src/mir/opt/cse.rs` was a block-local CSE that
skipped *caching* side-effecting instructions but never
*invalidated* previously cached reads when one ran. Two
`subscript_get v2[0]` reads with a `call` between them merged
into a single read; the post-call value seen by the outer was
whatever the first read returned. The boxed-upvalue lowering
pattern (mutated locals captured by a closure get wrapped in a
1-element list, both inner and outer use `subscript_get` /
`subscript_set` against the same box) collapsed under that
merge.

Fix: split CSE's seen-instructions table into a `pure` cache
(arithmetic, constants, allocations — never invalidated) and a
`memory` cache (currently `SubscriptGet`). Side-effecting
instructions that may write the heap (`Call*`, `SubscriptSet`,
`SetField`, `SetUpvalue`, `SetStaticField`, `SetModuleVar`)
clear the memory cache; the pure cache is untouched. Nbody and
the rest of the e2e suite stay clean — pure-arithmetic CSE
inside hot loops is unaffected.

### Parser rejected method-call chains that wrapped across lines

Status: **fixed**

```wren
var m = foo()
  .bar()        // prior: "unexpected token '.'"
  .baz()        // prior: same
```

Stock Wren lets a method chain span lines — the newline between
`)` and `.` is swallowed because the `.` would be nonsense as the
start of a new statement. Our `postfix()` loop was calling
`match_token(&Token::Dot)` directly, which failed on any newline
in between and returned from the expression.

Fix: added `peek_past_newlines()` that reports the first
non-newline token; `postfix()` calls it before the dot match and
skips newlines only when a dot actually follows. Safe because `.`
can't legitimately begin a statement on its own.

### `str[a..b]` / `list[a..b]` threw "Subscript must be a number"

Status: **fixed**

Stock Wren accepts a `Range` as a subscript argument for strings
(returns substring) and lists (returns sublist). WrenLift's
`subscript` natives hard-required `Num`. Fix: added an
`ObjRange` path to both with `from..to` (inclusive) and
`from...to` (exclusive) handling, negative-index normalization,
and bounds checking.

### `list_iterator_value` panicked on a non-Num iterator

Status: **fixed**

`args[1].as_num().unwrap()` panicked when the iterator protocol
handed the native a non-Num value. Fix: treat "not a Num" as end
of iteration (return null) instead of aborting the process.
Doesn't address the *cause* of the bad iterator state (see the
open entry below), but removes the panic as a symptom.

### Classes invisible inside `Fiber.new { ... }` closures

Status: **fixed**

```wren
class Greet { static hi { "hello" } }
System.print(Fn.new    { Greet.hi }.call())    // "hello"  (stock + lift)
System.print(Fiber.new { Greet.hi }.call())    // "hello"  (stock)
                                               // throws   (pre-fix lift)
```

`setup_fiber_from_closure` in `src/runtime/core/fiber.rs` hardcoded
`module_name = "main"` when creating the fiber's initial frame. The
interpreter resolves module variables by looking up the frame's
module name in `engine.modules`, so a fiber spawned from any module
other than `main` failed to see its own classes / top-level vars.

Fix: `Fiber.new` inherits `module_name` from the caller's topmost
frame, matching what `call_closure_sync` already does.

### `[1, 2] == [3, 4]` throws `"List does not implement '==(_)'"`

Status: **fixed**

`ObjClass::new` clones the superclass's method table at class
creation time. During core-library bootstrap all classes are
created *before* any `*::bind()` runs — so when `List` inherits from
`Sequence` (which inherits from `Object`), the inherited methods
array is empty because `obj::bind` hasn't registered `==(_)` /
`toString` / `is(_)` yet.

Fix: added `propagate_inherited_methods(vm)` at the end of
`core::initialize`. For every core class, fills method-table slots
that are still empty from the superclass's (now-populated) table.
Existing overrides (like `Num`'s numeric `==`) are preserved because
propagation only writes to `None` slots.

Side benefit: `is`, `hashCode`, `type`, `toString`, `!=`, `!`, and
the static `Object.same(_,_)` now resolve on every core class.

### `[1, 2] is klass` returns `false` when `klass` is a variable

Status: **fixed**

```wren
test(klass) {
  return [1, 2] is klass     // pre-fix: false for ANY variable klass
}
```

The MIR lowering for `Expr::Is` special-cased `Expr::Ident` and used
the static `Instruction::IsType(val, class_sym)` opcode, which
compares the runtime value's class **name** (not pointer) to a
symbol id baked in at compile time. Literal idents like `Num` /
`List` matched because their names are interned. Local variables
and parameters — even when they held an actual class value — baked
the variable's identifier as the class symbol, which never matched
anything's class name.

Fix: lower `x is y` uniformly as `x.is(y)`, dispatching through the
existing `Object.is(_)` primitive which compares class pointers
while walking the superclass chain. A future devirt pass can fold
back to `IsType` when the RHS is statically a core class.

### `System.print(List)` / `"%(List)"` prints `"instance"` / `"instance of Class"`

Status: **fixed**

Two separate code paths (`format_object` in `core/system.rs` for
`System.print`, `value_to_string` in `vm_interp.rs` for string
interpolation) both had a catch-all `"instance"` / `"instance of
<class>"` fallback that hit for every `ObjType` they didn't
explicitly handle — including `Class`, `List`, `Map`, `Range`, `Fn`.

Fix: added explicit branches for `ObjType::Class` (resolve the
class name via the interner) and filled out the other core types in
`value_to_string` so interpolation matches `System.print`
formatting. Required threading `&dyn NativeContext` into
`format_object` so it can resolve symbols.

### Consecutive `startsWith` calls with different args aborted the second call

Status: **fixed (as a side-effect of the call-dispatch / closure-patching pass)**

```wren
var tok = "alice"
if (tok.startsWith("--")) { /* ... */ }
if (tok.startsWith("-")) { /* ... */ }    // used to abort: "Argument must be a string"
```

Previously reproduced when the two calls sat inside a hot method
body; the second call arrived with a stale arg register. After
the cross-module call-dispatch fix (frame binds to the callee's
defining module) and nested `MakeClosure` fn_id patching, the
repro no longer triggers in any mode. `@hatch:cli` reverted its
`tok[0] == "-"` workaround back to idiomatic `tok.startsWith(...)`
and its spec passes.

### `for` iteration over a freshly-built list handed `iteratorValue` a non-number

Status: **fixed (same class as `startsWith`)**

The `list[0]` dispatch receiving a non-`Num` iterator stemmed from
the same underlying call-dispatch bug. Once the call-frame module
binding and closure-patching fixes landed, iteration over a list
built in the same method body runs cleanly. The defensive null
fallback in the `list_iterator_value` native is left in place as
belt-and-braces.

### Fiber abort through an intermediate closure call corrupted caller state

Status: **fixed**

```wren
var b = Fiber.new {
  T.outer { T.inner("nope") }             // closure inside closure
}.try()
System.print(Expect.that("x"))            // used to print "?"  (UNDEFINED)
```

Root cause: `run_fiber_until_depth` (the native-to-Wren bridge used
by JIT trampolines and the constructor sync path) compared its
`stop_depth` against whatever fiber was currently active. When
`Fiber.try` / `Fiber.abort` switched `vm.fiber` from the try-fiber
back to main mid-run, the stop-depth check fired against main's
frame count — which matched coincidentally on the first nested
return after the abort — so the bridge returned early with the
wrong value, leaving the caller's register UNDEFINED.

Fix: the run loop captures `stop_fiber` at entry; both Op::Return
and Op::ReturnNull gate the `stop_depth` check on
`fiber == stop_fiber`, so a fiber switch mid-run can no longer
masquerade as the original completion.

Regression test:
`runtime::vm::tests::test_fiber_abort_through_closure_preserves_subsequent_calls`.

## Roadmap

Items that aren't bugs but are conservative-correct compromises we'd
like to revisit once they pay off in observed benchmark / framework
code.

### Per-function memory-effect summaries for CSE / LICM / DCE

Today the optimizer's memory model is binary: a `Call` instruction
clears the memory-read cache (currently `SubscriptGet`), forcing every
field/element read after a call to be re-issued. That's sound for
arbitrary code but wastes optimization opportunities for calls that
demonstrably can't write to the receiver in question — `Math.sin`,
`Num.+`, leaf accessor methods on immutable types, etc.

A real call-effect analysis would attach a per-function summary
(reads / writes / unknown) to each MIR function, propagate through
call edges, and use the summary at `Call` sites to invalidate only
the memory regions that callee can actually touch. The same
infrastructure unblocks tighter LICM (hoist a load across a call we
can prove is loop-invariant) and tighter DCE (a call whose summary
is "no observable effect" + unused result becomes dead).

Cost vs. payoff: significant — needs an effect lattice on MIR, a
worklist propagation pass over the call graph (with conservative
defaults at unknown call edges), and care around foreign / native
methods (which we have to assume `unknown` until annotated). Worth
doing once a benchmark surfaces a CSE flush that's clearly costing
us on a known-pure call; until then the bucket-flush captures the
correctness story without the bookkeeping.
