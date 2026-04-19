# WrenLift runtime / compiler quirks

Running list of places where WrenLift diverges from stock Wren, with
status and references to the fixing commit. New entries land at the
top; once an item ships, it keeps its row so the git log retains
rationale for anyone who reads just this file.

## Fixed

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

## Open

### Consecutive `startsWith` calls with different args return stale/wrong arg

Status: **open**

```wren
var tok = "alice"
if (tok.startsWith("--")) { /* ... */ }
if (tok.startsWith("-")) { /* ... */ }    // aborts: "Argument must be a string"
```

The first call succeeds (both receiver and arg are strings). The
second call, with a different literal arg, aborts inside the
`starts_with` native with "Argument must be a string" — as if
`args[1]` is not the string literal we passed.

Reproduces only when surrounded by enough code (empty standalone
file doesn't trigger). Suspect an IC / inline-cache bug where the
argument register is read from a stale slot on the second
dispatch.

Worked around in `@hatch:cli` by collapsing both branches into a
single index-based check (`tok[0] == "-"`).

### `for` iteration over a list that has just been built leaves `iteratorValue` passed a non-number

Status: **open (partial mitigation)**

A `for (a in list) { ... }` loop inside a method body, where the
list was populated earlier in the same method, sometimes hands
`iteratorValue` a non-numeric iterator. Prior to the mitigation,
this panicked via `unwrap()`. The list native now tolerates the
bad input (returns null), so the process no longer aborts — but
the loop terminates early, which usually surfaces as silent
misbehavior rather than an error.

Same class of bug as the `startsWith` one above — likely an IC /
register-file quirk across method calls. Needs proper
investigation in `vm_interp.rs`.

### Fiber abort through an intermediate closure call corrupts caller state

Status: **open**

```wren
class T {
  static outer(body) { body.call() }
  static inner(arg) {
    if (!(arg is Fn)) Fiber.abort("bad arg")
  }
}

// Direct abort in the fiber's entry closure — works:
var a = Fiber.new { T.inner("nope") }.try()
System.print(Expect.that("x"))            // "instance of Assertion"  ✓

// Abort one frame deeper (through an intermediate body.call()):
var b = Fiber.new {
  T.outer { T.inner("nope") }             // closure inside closure
}.try()
System.print(Expect.that("x"))            // "?"  ✗  UNDEFINED returned
```

After the nested abort, the next call in the outer fiber returns
`UNDEFINED` (hence `"?"` via interpolation). Suggests the fiber
unwind isn't fully restoring the caller's register file or call
frame when the abort crosses a `body.call()`-style re-entry.

The simpler single-frame case (fiber → native that aborts) works
fine, as does the single-frame case where the fiber body itself
calls `Fiber.abort` directly. Only the *two-frame-deep* pattern
(fiber → Wren closure → native abort) triggers it.

Impact on the ecosystem: blocks `@hatch:test` from running a spec
that tests a non-Fn `Test.it` argument inside a `Test.describe`
block — we work around by asserting the abort against a bare
`Test.it` call. Also blocks any user-level "try this block, catch
the error" pattern that has nested closures.

Next step: instrument the fiber-unwind path in
`src/runtime/vm_interp.rs` (look at `resume_caller` /
`try_run_root_frame` / the abort propagation loop) to see what
state survives vs. what gets clobbered.
