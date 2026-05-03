# WrenLift runtime and tooling quirks

User-facing divergence from stock Wren in the runtime (interpreter,
JIT, GC, native bindings, fibers, parser surface) and the tooling
around it (hatch CLI, LSP, playground). Open items track current
divergence; Fixed keeps a one-paragraph record of past bugs so the
git log retains the rationale. Forward-looking optimisation work
lives elsewhere — this doc is for observed quirks only.

## Open

### Resolver doesn't accept version ranges or wildcards

`Dependency::Version(String)` is parsed as a literal version that
the resolver feeds straight into `release_url(name, version)`.
There's no `"^0.2.0"` / `">=0.2.5,<0.3"` / `"*"` syntax, so a
patch-level publish downstream forces a manual cascade of bumps:
gpu 0.2.6 → 0.2.7 forces game to republish 0.2.9 just to track
the gpu pin.

The registry has no list-versions API. `release_url` constructs a
fixed URL per `(name, version)` pair, and there's no manifest to
ask "which versions of `@hatch:foo` exist?"

Fix sketch: publish an index file at the registry root listing
every `(name, version)` pair, route `Dependency::Version` through
Cargo's `semver` crate, and have the resolver pick the highest
matching version per entry.

### Playground autocomplete pops inside comments

`wrenCompletionSource` in `wasm/web/index.html` matches `\w*`
against the line and forwards the result to `completeWren`. It
doesn't classify the cursor context — string and comment runs
look like ordinary text to the regex.

Fix sketch: expose a wasm helper that returns whether a byte is
inside a string or comment lex token (the lexer already knows);
`wrenCompletionSource` bails when it is. Eventually distinguish
`///` doc-comments so future cross-ref completions
(`[ClassName]`, `[Class.method]`) can fire there.

### Hover doesn't propagate superclass `///` docs to subclass overrides

A subclass's overriding method shows the bare signature when the
override has no `///` of its own, even though the parent class
documents it. `docs::collect::collect_module` builds one
`ClassDoc` per `class` AST node and walks only that class's
member list; the hover lookup takes the first member-name match
without following the superclass chain.

Fix sketch: `ClassDoc` learns a `superclass: Option<String>`
field; the hover member-match loop substitutes the first non-empty
doc found along the chain and labels the source class
(`*inherited from Game*`).

### Hover on a method parameter shows `local <name>` instead of the inferred class

`TypeInferrer` records var, field, and method-return types but
not parameter types — params start as `Any` and stay there.
Hover on a bare `g` falls through to the param-line fallback,
which only knows the method's signature line. `g.dt` resolves
correctly because the receiver-type lookup goes through the
expression-type table.

Fix sketch: per-`(class, method, param_index)` type table fed by
JSDoc-style `@param {Type}` annotations from the docs collector,
plus a hover post-pass that copies framework-declared param types
onto user spans when a signature matches a known override.

## Fixed

Each entry is a single-paragraph record of what was wrong. Commit
references stay so the git log retains the full investigation.

### Constructor JIT SIGSEGV under GC pressure (2026-04-26)

`call_constructor_sync_impl` rooted the instance before
dispatching the JIT'd ctor body but not the user-supplied args.
Any GC fired inside the body (allocations, foreign calls, member
assignment that grew a class shape table) staled the
register-bound arg pointers. Fix: push every ctor arg as a JIT
root and reload from the roots Vec into `jit_args[..]` before
dispatch. Locked in by
`e2e_gc_pressure_constructor_with_object_args`.

### `wren_call_N` arg rooting SIGSEGV on Linux x86_64 (commit c11c3ff, 2026-04-26)

The `wren_call_3` and `wren_call_4` x86_64 wrappers read `[rbp]`
and `[rbp+8]` via inline asm to fetch `jit_fp` / `ret_addr`, but
didn't declare rbp as an asm input. Rust's release optimiser
treated rbp as a free GP register and clobbered it with a
function arg before the asm executed; the subsequent
`mov 0x0(%rbp), %rcx` then dereferenced a NaN-boxed Value. macOS
maps the high-bit range differently and silently succeeded with
garbage; Linux faulted because the address sat in kernel-only
memory. Fix: rewrite both wrappers as `#[unsafe(naked)]` matching
`wren_call_0/1/2`, with explicit SysV alignment.

### Tiered JIT-to-JIT method dispatch corrupted receiver in hot loops (commit e869735, 2026-04-26)

The JIT slow paths for `Mul` / `Sub` / `Div` / `Mod` / `<` / `>`
/ `<=` / `>=` bare-cast NaN-box bits to f64 and ran `unbox(a) ⊕
unbox(b)` instead of dispatching to user-defined operator
overloads when the receiver was a non-Num heap object. For two
object operands the result was an implementation-defined NaN —
on aarch64 the hardware preserves the first operand's payload,
so `bob * spin` silently returned `bob`'s bits. Fix: factor
dispatch into `wren_arith_dispatch` / `wren_cmp_dispatch` and
route every helper through it.

### `@hatch:web` request handler returned null in tiered mode (2026-04-27)

`dispatch_closure_bc`'s threaded-interpreter fast path built its
`JitContext` with the *caller's* module storage instead of the
*callee's*. Route handler closures inherited `@hatch:web`'s
`module_vars`, so every `GetModuleVar @N` resolved against the
wrong slot table. Fix: in the `has_tc` branch of
`dispatch_closure_bc_inner`, look up `module_vars` against
`vm.engine.func_module(target_func_id)`. Mirrors the JIT-dispatch
branch above and the callee-context section in
`dispatch_closure_bc`.

### `@hatch:web` `App.listen` hung in tiered mode (commit a646e82, 2026-04-26)

Same root cause as the foreign-call register-corruption family:
`tryAccept` returned a wrapper that subsequent ops dereferenced
via stale register pointers. Disabling the IC fast-path
inline-JIT-leaf dispatch by default (opt-in via
`WLIFT_ENABLE_IC_JIT=1`) cleared this and the related game-example
crashes; proper fix waits on JIT-frame stack maps.

### `Fiber.try` didn't catch "does not implement" errors (commit 260633a, 2026-04-26)

Method-not-found errors propagated past `Fiber.try` even though
`Fiber.abort` and `ctx.runtime_error` were caught. The dispatcher
now converts the error message to a String, stores it on
`fiber.error`, marks the fiber `Done`, and resumes the caller —
same shape `Fiber.abort` already followed.

### `for-in` + `continue` infinite-looped / corrupted next binding / failed Cranelift verifier (commit 1441d38, 2026-04-26)

`lower_for` emitted the iterator advance call only on the natural
fall-through path. `continue` branched straight to the cond block
with `[…vars]` while cond_bb's params were `[iter_phi,
…vars_phi]` — so the iter slot got reassigned to whatever value
zipped first, and the iter advance never ran. Fix: introduce a
`latch_bb` shaped `[iter_phi, …vars_phi]` like cond_bb;
`continue` jumps to the latch with `[iter_param, …current_vars]`,
the latch advances the iterator and branches to cond_bb. Natural
fall-through stays inline so the no-`continue` case keeps its
bytecode density.

### Closure-mutated outer locals were frozen at the first call's value (commit b74c653, 2026-04-26)

`src/mir/opt/cse.rs` was a block-local CSE that skipped *caching*
side-effecting instructions but never *invalidated* previously
cached reads when one ran. Two `subscript_get v2[0]` reads with
a `call` between them merged into a single read; the boxed-upvalue
lowering pattern collapsed under that merge. Fix: split CSE's
seen-instructions table into a `pure` cache (arithmetic,
constants, allocations) and a `memory` cache (currently
`SubscriptGet`); side-effecting instructions clear the memory
cache while the pure cache stays live.

### Parser rejected method-call chains that wrapped across lines

`postfix()` called `match_token(&Token::Dot)` directly, which
failed on any newline between `)` and `.`. Fix: added
`peek_past_newlines()`; `postfix()` skips newlines only when a
dot actually follows. Safe because `.` can't legitimately begin
a statement.

### `str[a..b]` / `list[a..b]` threw "Subscript must be a number"

The `subscript` natives hard-required `Num`. Fix: added an
`ObjRange` path to both with `from..to` (inclusive) and
`from...to` (exclusive) handling, negative-index normalization,
and bounds checking.

### `list_iterator_value` panicked on a non-Num iterator

`args[1].as_num().unwrap()` panicked when the iterator protocol
handed the native a non-Num value. Fix: treat "not a Num" as end
of iteration (return null) instead of aborting the process.

### Classes invisible inside `Fiber.new { ... }` closures

`setup_fiber_from_closure` hardcoded `module_name = "main"` when
creating the fiber's initial frame; fibers spawned from any other
module couldn't see their own classes. Fix: inherit `module_name`
from the caller's topmost frame, matching `call_closure_sync`.

### `[1, 2] == [3, 4]` threw `"List does not implement '==(_)'"`

`ObjClass::new` clones the superclass's method table at class
creation time. During core-library bootstrap all classes were
created before any `*::bind()` ran, so inherited method slots
were empty. Fix: `propagate_inherited_methods(vm)` at the end of
`core::initialize` fills slots that are still `None` from the
superclass's now-populated table; existing overrides are
preserved.

### `[1, 2] is klass` returned `false` when `klass` was a variable

The MIR lowering for `Expr::Is` special-cased `Expr::Ident` and
used `Instruction::IsType(val, class_sym)`, comparing the runtime
value's class *name* to a symbol id baked in at compile time.
Local variables and parameters baked the variable's identifier
as the class symbol, which never matched anything. Fix: lower
`x is y` uniformly as `x.is(y)`, dispatching through the existing
`Object.is(_)` primitive that compares class pointers along the
superclass chain.

### `System.print(List)` / `"%(List)"` printed `"instance"`

Two formatter paths (`format_object` for `System.print`,
`value_to_string` for interpolation) had a catch-all `"instance"`
fallback for `ObjType` variants they didn't explicitly handle.
Fix: added explicit branches for `ObjType::Class` and the rest of
the core types so interpolation matches `System.print` formatting.

### Consecutive `startsWith` / freshly-built-list `for` iteration mis-dispatched

The same stale-arg-register family. After the cross-module
call-dispatch fix (frame binds to the callee's defining module)
and nested `MakeClosure` fn_id patching, the repros no longer
trigger. The defensive null fallback in `list_iterator_value`
stays as belt-and-braces.

### Fiber abort through an intermediate closure call corrupted caller state

`run_fiber_until_depth` (the native-to-Wren bridge used by JIT
trampolines and the constructor sync path) compared its
`stop_depth` against whatever fiber was currently active. When
`Fiber.try` / `Fiber.abort` switched `vm.fiber` from the try-fiber
back to main mid-run, the stop-depth check fired against main's
frame count and the bridge returned early with the wrong value.
Fix: capture `stop_fiber` at run-loop entry; both `Op::Return`
and `Op::ReturnNull` gate the depth check on
`fiber == stop_fiber`. Regression test:
`runtime::vm::tests::test_fiber_abort_through_closure_preserves_subsequent_calls`.
