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
