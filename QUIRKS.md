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

### LSP completions don't classify cursor context

The runtime's `complete_wren` returns matches for any byte
offset the caller passes in — it doesn't classify whether the
cursor sits inside code, a string literal, or a `//` / `///`
comment. The LSP currently ships diagnostic-only, so this isn't
visible yet, but the playground (which already calls the
function from `wasm/web/index.html`) demonstrates the symptom:
every word-shape prefix triggers a completion, including inside
comments and strings.

Fix sketch: add a wasm helper that returns the lex-token kind
covering a byte (string / line-comment / doc-comment / code) so
both the LSP and the playground can bail uniformly. `///` is
the doc-comment context — it should stay open as a future
hook for cross-ref completions (`[ClassName]`,
`[Class.method]`) without leaking through to plain `//`.

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

### Implicit-this swallows typos inside method bodies

A bare identifier inside a method that doesn't resolve as a
local, parameter, field, or module-level name is rewritten to
`this.<name>` rather than diagnosed. That's correct Wren
semantics — dynamic dispatch through inheritance can't be
proven at compile time — but it means common typos like `sed`
inside `static run() { ... sed ... }` don't surface as
"undefined variable" the way the same identifier at module
scope does.

Fix sketch: track each class's declared methods / getters /
setters during sema's class walk; on `ImplicitThis(name)`
lookup, check the enclosing class's member set (and walk the
superclass chain when resolvable), and emit a soft "unknown
member on `<class>`" diagnostic when the name isn't there.
Be lenient when the parent class lives behind an unresolvable
import so legitimate framework-method-override patterns don't
false-positive.

### Spec runner's per-test "Run" button executes the whole file

The VS Code extension's spec runner sidebar (and the per-block
codelenses on `Test.describe(...)` / `Test.it(...)`) all
dispatch `wrenlift.runFile <uri>`, which spawns `wlift <spec>`
and runs every registered case. There's no per-case filter at
runtime: `@hatch:test` doesn't expose a setter, and Wren has no
env access without a plugin so the extension can't slip a
filter in via `WLIFT_TEST_FILTER` or similar.

Fix sketch: add a static `Test.filter=` setter to
`@hatch:test`'s `test.wren`; have `Test.run()` skip cases whose
`<group> > <name>` label doesn't contain the filter substring.
The extension writes a tmp wrapper file (`/tmp/__wlift_runner_<uuid>.wren`)
that imports `@hatch:test`, sets `Test.filter`, then imports
the spec by absolute path (Wren's loader accepts absolute
paths), and runs `wlift <wrapper>`. The codelens command
dispatch already takes the case label as an argument; the
upgrade is local once the runtime setter ships.

### `cargo install` and `install.sh` ship to different `~/bin` paths and can drift

`install.sh` writes to `~/.local/bin/{wlift,hatch,wlift-lsp}`;
`cargo install` writes to `~/.cargo/bin/...`. The VS Code
extension's `resolveBinary` probes `~/.local/bin` first when
the configured `serverPath` is the default, so a developer who
rebuilds via `cargo install` updates `~/.cargo/bin/wlift-lsp`
but the extension keeps spawning the older
`~/.local/bin/wlift-lsp`. Symptom: new LSP features (codelens,
goto-def expansions, completion improvements) don't appear in
the editor even though the binary on `$PATH` is up to date.

Fix sketch: `resolveBinary` could compare mtimes between the
two paths and prefer the newer one, OR the extension's "Show
Toolchain Versions" command could surface the drift as a
warning when the two binaries differ in version. For now: copy
manually after rebuild, or set `wrenlift.serverPath` to the
absolute path you want.

## Fixed

Each entry is a single-paragraph record of what was wrong. Commit
references stay so the git log retains the full investigation.

### LSP rejected `#!native` / `#!wasm` cfg attributes on imports (commit 5e15aa4, 2026-05-03)

Cross-target imports — the canonical shape for any package
that ships separate native / wasm backends — surfaced as
"attributes cannot attach to import statements" in the editor.
Root cause: `parse::cfg::apply` strips bare cfg lines as a
pre-parse pass, but the LSP feeds raw source straight to
`parser`, so the `#!native` token reached `parse_decl_or_stmt`
and `reject_attributes_on` flagged it. Fix: filter the two
known cfg names (`wasm`, `native`) out of
`reject_attributes_on`'s rejection loop, scoped to bare
hashbang flags. Other `#!foo` attrs still flag so unknown
gates don't sneak through silently.

### `cargo install` on macOS produced SIGKILL'd binaries (commit 54aa607, 2026-05-03)

Cargo strips Rust binaries by default in some toolchain
versions; the strip invalidates the linker-generated adhoc
codesignature, and macOS's kernel SIGKILLs binaries whose
embedded signature no longer matches the file. Symptom in VS
Code: `spawn ENOENT`-style failures even though the binary is
on disk. Symptom in a terminal: exit 137 with no output. Fix:
pin `strip = "none"` in `[profile.release]` so cargo install
respects the unstripped layout. Workaround for already-broken
installs: `codesign --force --sign - ~/.cargo/bin/{wlift,hatch,wlift-lsp}`.

### `cargo install --git` skipped `wlift_lsp` (commit 1e1d66b, 2026-05-03)

The site's "build from source" chip ran
`cargo install --git github.com/wrenlift/WrenLift`, which only
installs the root crate's bins (`wlift` + `hatch`). `wlift_lsp`
lives in `crates/wlift_lsp/` as a separate workspace member,
so users following the build-from-source path got a runtime +
hatch CLI but no language server. Fix: chip now passes both
package names — `cargo install --git URL wren_lift wlift_lsp`.

### vscode-wrenlift 0.1.2 shipped without `node_modules` (commit 106d599, 2026-05-03)

Marketplace install activated the extension but the LSP client
never started, the status bar didn't show, and the install
dialog never fired. Only the static contributions (grammar,
icons) worked. Root cause: `.vscodeignore` excluded
`node_modules/**`, so `vscode-languageclient` was missing from
the published `.vsix` and `require("vscode-languageclient/node")`
threw at activate time. Fix: drop the exclusion and add a
`npm prune --omit=dev` step before `vsce publish` in CI so
only production deps ship. Removed the `vscode:prepublish`
hook so vsce doesn't try to re-tsc against the pruned tree.

### vscode-wrenlift restart hung after the LSP crashed 5+ times (commit f336c83, 2026-05-03)

vscode-languageclient gives up auto-restart after 5 crashes in
3 minutes. The user-driven Restart command then hung because
`await client.stop()` waited indefinitely for a shutdown ack
from a process that no longer existed. Fix: cap `client.stop()`
at 2s and skip it entirely when the client is in
Stopped/Crashed state — restart now reliably tears the dead
client down and spawns a fresh one. Drop the redundant
`stopServer`-then-`startServer` sequence in `restartServer`;
`startServer`'s prelude does the same teardown without
double-stopping.

### Auto-install poll spawned the LSP before `chmod +x` finished (commit 808c9d2, 2026-05-03)

`install.sh` runs `mv` then `chmod +x` as separate steps; the
extension's post-install poll watched `existsSync` and triggered
the spawn between the two, catching EACCES that surfaced as a
silent client.start() rejection. Fix: gate on
`accessSync(X_OK)` instead, add a 250 ms settle delay before
the first attempt, and keep polling until the client reaches
Running or the 60 s window expires. Add a `starting` re-entrancy
guard so the poll can't race a still-pending start triggered
from the dialog path.

### `install.sh` picked `vscode-v0.1.1` as the latest runtime release (commit 253d4f3, 2026-05-03)

`resolve_latest` followed the `releases/latest` redirect, which
GitHub computes across the entire tag namespace. The VS Code
extension is versioned independently with `vscode-v*` tags, so
a freshly-cut extension tag was preferred over the most recent
runtime tag, and the installer downloaded a tarball that
didn't exist for `wlift`. Fix: `resolve_latest` now hits
`/releases` (plural), filters `tag_name` to `^v[0-9]`, and
picks the first match. The release workflow's `v*` glob also
tightened to `v[0-9]*` so it doesn't fire on the extension
namespace.

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
