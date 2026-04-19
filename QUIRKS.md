# WrenLift runtime / compiler quirks

Running list of places where WrenLift diverges from stock Wren, with
status and references to the fixing commit. New entries land at the
top; once an item ships, it keeps its row so the git log retains
rationale for anyone who reads just this file.

## Fixed

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

### (none currently tracked)

Add entries as they surface; ideal shape is a minimal repro +
expected-vs-actual + where the offending code lives.
