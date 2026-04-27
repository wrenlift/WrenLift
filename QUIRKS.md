# WrenLift runtime / compiler quirks

Running list of places where WrenLift diverges from stock Wren, with
status and references to the fixing commit. New entries land at the
top; once an item ships, it keeps its row so the git log retains
rationale for anyone who reads just this file.

## Open

### `@hatch:web` request handler returns null in tiered mode (counter / chat)

Status: **fixed (commit pending, 2026-04-27)**

Root cause: `dispatch_closure_bc`'s **threaded interpreter
fast path** built its `JitContext` with the *caller's* module
storage instead of the *callee's*. When `@hatch:web`'s listen
loop dispatched a user-registered route handler closure, the
closure's body inherited `@hatch:web`'s `module_vars`, so every
`GetModuleVar @N` resolved against the wrong slot table.

Concretely, `var page = Css.tw("...")` in `main.wren` would
read back as the `HxResponse` class object from inside the
handler closure (because slot N in `@hatch:web` happens to
hold `HxResponse`), and any `arg is String` / `value is
Response` check downstream raised "Right operand must be a
class" because the loaded "class" was a Style instance, a
Map, etc.

Fix: in the `has_tc` (threaded code installed) branch of
`dispatch_closure_bc_inner`, look up `module_vars` against
`vm.engine.func_module(target_func_id)` instead of the caller's
`module_name` parameter. Mirrors what the JIT-dispatch branch
above already does and what `dispatch_closure_bc`'s callee
context section does for the JitContext setup.

The non-threaded JIT branch and the bytecode-fallback branch
were already correct; this was a single missed swap on the
threaded path. Counter / chat / hello all serve the right HTML
in tiered mode after the fix; 845 lib + 108 e2e green.

Original investigation notes preserved:

`hatch run hatch/examples/web/counter` returns `204 No Content`
for `GET /` in tiered mode but the correct HTML in
`--mode interpreter`. Same shape on the chat example
(messages don't send). `web/hello` works in both modes.

Repro narrows to: when a user-defined closure
(`{|req| ...}`) registered via `app.get("/")` is called from
`@hatch:web`'s dispatch loop, instructions inside the closure
that touch a built-in class via `is` (e.g. `arg is String` in
css.wren's `applyToken_`, `value is Response` in web.wren's
`Response.coerce`) raise

```
Error: Right operand must be a class.
```

The error fires from `Object.is`'s native (`obj.rs:19`),
which checks `args[1].is_object()`. So `String` / `Response` /
`Style` etc. are loading as a non-object value. Most likely
the JIT's `wren_get_module_var` is reading from the wrong
module's variable storage — i.e. a missed cross-module
context swap somewhere in the dispatch chain that ends in the
user closure.

`call_jit_with_shadow` already does swap module_vars on
cross-module dispatch (see `runtime_fns.rs::call_jit_with_shadow`),
so the leak point is most likely either:
- closures dispatched via `wren_call_N` going through a path
  that bypasses the swap, or
- bytecode-interpreted paths with stale `JitContext` from a
  prior JIT call.

Local minimal repro:

```wren
import "@hatch:web" for App
var app = App.new()
app.get("/") {|req|
  var s = "hello"
  System.print(s is String)   // <-- in tiered: "Right operand must be a class"
  return "ok"
}
app.listen("127.0.0.1:3000")
```

Workaround: `wlift --mode interpreter <app.hatch>` until fixed.

### `wren_call_N` arg rooting SIGSEGVs on Linux x86_64

Status: **fixed (commit c11c3ff, 2026-04-26)**

Root cause: the `wren_call_3` and `wren_call_4` x86_64 wrappers
were regular Rust functions that read `[rbp]` and `[rbp+8]` via
inline asm to pull `jit_fp` (caller's frame pointer) and
`ret_addr` from the stack. The asm didn't declare rbp as an
input, so Rust's release optimiser treated rbp as a free
general-purpose register and clobbered it with a function arg
*before* the asm executed:

```
8374e0:  push   %rbp                  ; save caller's rbp (callee-saved)
...
8374f3:  mov    %rcx,%rbp             ; rbp = a1 (function arg)  ← clobber
...
837501:  mov    0x0(%rbp),%rcx        ; mov rcx, [rbp]           ← reads [a1]
```

`mov 0x0(%rbp), %rcx` then dereferences whatever Rust spilled
into rbp — in delta_blue, that's a NaN-boxed Value (e.g.
`0xfffc7fffd7c187a8`), a high-bit address. On Linux x86_64
user space ends at `0x7fff_ffff_ffff`, so reading from a 0xfffc
address faults — kernel-only memory. macOS (aarch64 native or
Rosetta x86_64) maps that range differently and the read
silently succeeds with garbage that doesn't crash.

The bug had been latent since the `wren_call_3/4` x86_64
wrappers were written; what changed in `a646e82` was the size
and shape of `wren_call_N_inner`, which made LLVM's register
allocator more likely to pick rbp for a spill in the wrapper.

Fix: rewrite `wren_call_3` and `wren_call_4` as
`#[unsafe(naked)]` (matching `wren_call_0/1/2`). Naked asm has
no Rust prologue, so `rbp` at function entry IS the caller's
value. The 7th and 8th args (jit_fp, ret_addr) are placed in
register r9 (wren_call_3) or pushed onto the stack (wren_call_4)
— SysV's 16-byte alignment is preserved with explicit `sub
rsp, 8` padding before the pair of pushes in wren_call_4.

Verified by reproducing the crash on Lima Ubuntu 24.04 x86_64
VM, applying the fix, and watching 50 stress runs of
`delta_blue` produce 14065400 with no SIGSEGV. CI bench (Linux
x86_64) green again with rooting in place; all four benchmarks
beat their historical targets:

| Bench         | WrenLift | Std Wren | Ratio |
|---------------|----------|----------|-------|
| fib           | 0.0122s  | 0.1774s  | 0.07x |
| method_call   | 0.0537s  | 0.0841s  | 0.64x |
| binary_trees  | 0.6613s  | 0.9426s  | 0.70x |
| delta_blue    | 0.1578s  | 0.0992s  | 1.59x |

### Tiered JIT-to-JIT method dispatch corrupts receiver in hot loops

Status: **fixed (commit e869735, 2026-04-26)**

Root cause: the JIT slow paths for `Mul` / `Sub` / `Div` / `Mod`
/ `<` / `>` / `<=` / `>=` were not dispatching to user-defined
operator overloads when the receiver was a non-Num heap object.
`wren_num_add` had the dispatch (`if !va.is_num()` → look up
`+(_)` and `dispatch_method`); the others bare-cast NaN-box bits
to f64 and ran `unbox(a) ⊕ unbox(b)`. For two object operands
the result was an implementation-defined NaN — on aarch64 the
hardware preserves the first operand's payload, so `bob * spin`
silently returned the bit pattern of `bob`. The downstream code
then "succeeded" with `bob.at(0,0) = 1` (translation matrix's
identity diagonal), making the cube collapse to a flat polygon
"after a few seconds" once the hot draw loop JIT-compiled.

Fix: factor the dispatch pattern into `wren_arith_dispatch` and
`wren_cmp_dispatch`, and route every helper through it.

Local repro keeps the prior 60-line Mat4 test as the lock-in
case; full-stack `cube-3d` runs cleanly for 20s+ in tiered mode
post-fix.

Below is the original bisect notes — the previous "JIT-frame
stack maps required" hypothesis turned out to be wrong; the
real bug was a missing dispatch path, not a register-staling
issue.

Smallest reproducer (no plugin / no GPU / no game machinery):

```wren
class Mat4 {
  construct new() { _m = List.filled(16, 0) }
  static identity {
    var m = Mat4.new()
    m.set(0, 0, 1)
    m.set(1, 1, 1)
    m.set(2, 2, 1)
    m.set(3, 3, 1)
    return m
  }
  static rotationY(angle) {
    var c = angle.cos
    var s = angle.sin
    var m = Mat4.identity
    m.set(0, 0, c)
    m.set(0, 2, s)
    m.set(2, 0, -s)
    m.set(2, 2, c)
    return m
  }
  static translation(x, y, z) {
    var m = Mat4.identity
    m.set(0, 3, x)
    m.set(1, 3, y)
    m.set(2, 3, z)
    return m
  }
  set(r, c, v) { _m[r * 4 + c] = v }
  at(r, c) { _m[r * 4 + c] }
  *(o) {
    var r = Mat4.new()
    var i = 0
    while (i < 4) {
      var j = 0
      while (j < 4) {
        var s = 0
        var k = 0
        while (k < 4) {
          s = s + at(i, k) * o.at(k, j)
          k = k + 1
        }
        r.set(i, j, s)
        j = j + 1
      }
      i = i + 1
    }
    return r
  }
}

var spin = Mat4.rotationY(3.808)
var bob  = Mat4.translation(0, 0.5, 0)
var i = 0
while (i < 500) {
  var m = bob * spin
  if ((m.at(0,0) - 3.808.cos).abs > 0.01) {
    System.print("DIVERGE i=%(i) m[0,0]=%(m.at(0,0))")
    break
  }
  i = i + 1
}
```

`bob` and `spin` are pinned outside the loop; they NEVER change.
After ~250 iterations under `--mode tiered`, `bob * spin` returns
the identity matrix instead of the correct product (m[0,0] = 1
instead of cos(3.808) ≈ -0.78). Same shape under generational
and mark-sweep GC; `--gc arena` (no collection) and
`--mode interpreter` both run clean to 500 iterations.

What we know:

- JIT-only: `WLIFT_SKIP_JIT='*'` keeps the loop correct for 500+
  iterations.
- Pinpointed to the `*(_) ↔ at(_,_)` JIT-to-JIT dispatch:
  skipping JIT for *either* `*(_)` or `at(_,_)` alone clears it.
  Both must be JIT-compiled for the corruption to surface.
- Bytecode interpreter is fine: `--mode interpreter` runs the
  full 500.
- Optimized tier is not the trigger:
  `--opt-threshold 999999999` (baseline-only) still reproduces.
- IC fast path is not the trigger: `WLIFT_DISABLE_IC_JIT=1`
  still reproduces; the path through `wren_call_2` is enough.
- Threaded interpreter is not the trigger:
  `WLIFT_DISABLE_THREADED=1` still reproduces.

Surface symptoms in real apps: cube-3d's spinning cube collapses
to a flat polygon after a few seconds (`bob * spin` model matrix
goes to identity-shaped garbage); web chat's HTTP `tryAccept`
loop stops processing requests once the inner dispatch is
JIT-compiled.

Suspected root cause: register-held receiver / arg pointers in
the JIT'd caller go stale across an internal allocator
(`Mat4.new()` at the top of `*(_)`). The `wren_call_N` rooting
in c11c3ff covers the dispatch boundary but doesn't help the
JIT'd body itself, which loads `this` / `o` into a register
once and uses the register copy on every iteration of the
inner loops. Mark-sweep also reproduces because object
identity is preserved but the JIT register's pointer is
suspect for an unrelated reason — perhaps the dispatch result
slot. Needs deeper investigation.

Workarounds:
- `WLIFT_SKIP_JIT='*(_),at(_,_)'` (or any one of them).
- `--mode interpreter` for game / web examples until fixed.
- Hot-shape rewrites that move the inner dispatch out of the
  JIT'd body (defeats the purpose, but unblocks specific apps).

The IC inline JIT-leaf dispatch passes recv + args to the compiled
callee in registers. Even after pushing them as JIT roots before
the call (commit fddbd6c), the JIT'd body itself reads those args
back from registers, not the roots Vec — and any GC fired anywhere
in the surrounding interpreter loop after this path completes
(next op's MakeList / MakeMap / native helper / etc.) ages those
register-bound pointers out of any GC root set. Even when the
JIT'd callee itself is alloc-free, the result and other live
values straddle the boundary; a chain of fast IC dispatches with
one intervening allocator triggers the corruption.

Until JIT-frame stack maps land, the IC kind=1 path is opt-in via
`WLIFT_ENABLE_IC_JIT=1`. Game examples (sprite-grid, cube-3d,
ecs, bouncing-ball) run cleanly under `--mode tiered` with the
default off. method_call benchmark regresses ~45% (90ms → 131ms);
fib still 21× faster than std Wren.

Audit also surfaced and fixed: `Op::MakeList`, `Op::MakeMap`,
`Op::StringConcat` now publish the caller's register file to
`frame.values` before invoking the allocator. The four
`wren_call_N` runtime helpers root their receiver + args before
dispatch.

Two JIT-leaf inline dispatch paths in `vm_interp` (slow-path
Op::Call after IC miss; IC fast path on subsequent hits) used to
invoke a JIT'd callee while the caller's register file lived only
in the local Rust `values: Vec<Value>` — never published to
`frame.values`. A GC fired by the callee (or by a runtime helper
the callee dispatches through) traced the fiber's `mir_frames`
and missed every still-live pointer in the local Vec; generational
promote moved those objects without updating the caller's view.
Subsequent ops dereferenced freed memory and surfaced as either
misdispatched ops ("subscript get with arity 1" on what should
have been a method call) or "Null does not implement X" on a
wrapper that's still allocated elsewhere on the heap.

The fix saves `values` into `frame.values` before each JIT-leaf
inline call and reloads after. Bouncing-ball's failure shifts
from line 127 to line 128 (one op later); sprite-grid /
cube-3d / ecs still fail at the same line. At least one more
unsafely-rooted-locals path remains — likely involving the
`@hatch:gpu` `Surface.acquire` foreign-method chain (where a Wren
method wraps a foreign call and constructs a wrapper object).

Reproduces under `--opt-threshold 999999999`, so the residual
divergence is in the tiered-mode interpreter path itself, not in
JIT'd code. Workaround: `--mode interpreter`.

Three game examples reach a runtime error in `--mode tiered` after
several frames. Each surfaces inside `@hatch:game`'s render-pass
construction at:

```
var frame = surface.acquire()
var encoder = device.createCommandEncoder()
var passDesc = {
  "colorAttachments": [{
    "view": frame.view,    // ← Null does not implement 'view'
    ...
```

`frame` was a valid `SurfaceFrame` two lines earlier (a Wren-level
`System.print("DBG ... frame=%(frame)")` between the assignment and
the Map literal makes the bug *disappear* — adding a probe forces an
extra register usage that prevents whatever miscompile is happening).
Reproduces under `--opt-threshold 999999999` so JIT'd code is not
involved; the bug is in the tiered-mode interpreter path itself.

Bouncing-ball trips a sibling form: `_world.position(b.body)` raises
"unsupported: subscript get with arity 1" — the receiver register has
been replaced by something the wlift_physics native dispatcher doesn't
recognize as a `World2D` instance.

Both shapes:

* Foreign-method call returns a Wren-level wrapper object.
* Surrounding Wren code does another foreign call (createCommandEncoder
  / step).
* Subsequent member access on the original wrapper sees null /
  wrong-type.

Adding any read of the wrapper between the foreign calls papers over
the bug, suggesting a register / SSA value-tracking issue specific to
the tiered-interpreter dispatch where the result slot for one Op::Call
or the bound-receiver slot of another gets mis-routed when the call
chain looks "foreign → Wren method → foreign". 5+ iterations always
succeed; failure fires after that, indicating tier-up profiling state
matters even when no JIT compile actually happens.

`@hatch:web App.listen` hang (entry below) is likely the same root
cause — different surface symptom because there's no synchronous
return chain to surface the corrupted register, the loop just stops
making progress.

### `@hatch:web` `App.listen` hangs in tiered mode

Status: **fixed downstream of the IC-fast-path gating (commit
a646e82, 2026-04-26)**

Same root cause as the game-example register-corruption above:
the listen loop's `tryAccept` foreign call returned a wrapper
that subsequent ops dereferenced via stale register pointers.
With `WLIFT_ENABLE_IC_JIT` off by default, `wlift --mode tiered
web-hello.hatch` serves both `/` and `/hi/:who` correctly.

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

Status: **not currently reproducing (likely fixed downstream of
Phase 0 / 1 / 4 — `@hatch:http http.spec.wren --mode tiered` runs
25/25 clean as of 2026-04-26, was 6/9 previously)**

Re-validate with the original repro if it surfaces again on real
network code.

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

### Constructor JIT SIGSEGV under GC pressure

Status: **fixed (2026-04-26)**

`call_constructor_sync_impl` invokes the JIT'd constructor body
with `(instance, ...ctor_args)` loaded directly into argument
registers. The instance was being rooted before dispatch, but
the user-supplied args were not — so any GC fired inside the
constructor body (allocations, foreign calls, member assignment
that grows a class shape table, etc.) staled the register-bound
arg pointers without updating them.

```wren
class Pair {
  construct new(left, right) {
    _left = left            // ← any allocation here can promote
    _right = right          //   `left` / `right`; register copies
  }                         //   stay pointing to old addresses
}
var prev = null
for (i in 0...2000) prev = Pair.new(prev, i)
```

Fix: push every ctor arg as a JIT root before dispatching the
JIT'd body, then re-read each arg from the roots Vec into the
`jit_args[..]` array. The instance was already rooted by
`call_constructor_sync`; this extends the same shape to the
remaining args. Locked in by
`e2e_gc_pressure_constructor_with_object_args` which chains
`Pair.new(prev, i)` for 2000 iterations under both Interpreter
and Tiered modes and walks the chain back to dereference every
still-live pointer the constructor stashed.

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

Status: **seed landed** (commit ad6b4ed) — `pure_call: bool` flag
on `Instruction::Call`, set at MIR-build time for known-pure
builtins (Num arithmetic + comparisons, Math intrinsics, bitwise).
CSE keeps its memory-read cache valid across pure calls, so `xs[0]
+ 1` followed by `xs[0] + 2` no longer re-emits the second read.

The full version walks the call graph, propagates a per-function
effect summary (reads / writes / unknown) through call edges, and
lets CSE keep the cache across user-defined leaf methods,
LICM hoist a load past a provably-pure call, DCE drop a call whose
summary is "no observable effect" with unused result.

Cost vs. payoff: substantive — needs an effect lattice on MIR, a
worklist propagation pass over the call graph (with conservative
defaults at unknown call edges), and care around foreign / native
methods (assume `unknown` until annotated). The current `pure_call`
seed gets us the highest-frequency pure call sites (operators) for
free; expand the analysis when a benchmark shows CSE losing on a
specific user-defined leaf.
