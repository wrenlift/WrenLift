# Semantic Analysis

This module performs two post-parse, pre-codegen passes over the Wren AST:
**name resolution** and **speculative type inference**.

---

## Name & Scope Resolution

Walks the AST and binds every identifier to a `ResolvedName`:

| Variant | Meaning |
|---|---|
| `Local(u16)` | Local variable in the current scope |
| `Upvalue(u16)` | Captured variable from an enclosing scope |
| `ModuleVar(u16)` | Top-level module variable |

### Scope stack

A `Vec<Scope>` acts as the scope stack. Each `Scope` tracks its `ScopeKind`
(Module, Class, Method, Block, Closure), a `Vec<Local>` of declared locals, a
depth counter, and a `Vec<UpvalueInfo>` for captured variables.

Resolution walks the stack from innermost to outermost:
1. Current scope locals.
2. Enclosing scope locals (creating an upvalue chain if found).
3. Module-level variables.

Module resolution runs two passes: a first pass registers all top-level `class`,
`var`, and `import` names (enabling forward references), then a second pass
resolves all bodies.

### Upvalue chain tracking

When a name is found in a non-adjacent enclosing scope, `capture_upvalue`
threads an `UpvalueInfo` through every intermediate scope from source to target.
Each `UpvalueInfo` records an `index` and `is_local` flag -- `is_local` is true
only for the first hop (capturing directly from the declaring scope), and false
for subsequent hops that capture an already-captured upvalue. Deduplication
happens via `Scope::add_upvalue`. On scope pop, non-empty upvalue vectors are
stored in `ResolveResult::upvalues` keyed by `scope_id`.

### Error detection

| Error | Trigger |
|---|---|
| Undefined variable | Identifier not found in any scope or module vars |
| Duplicate module variable | `var`/`class`/`import` alias collides at module level |
| Duplicate local declaration | Same name declared twice at the same block depth |
| `this` outside method | `Expr::This` encountered with `in_method == false` |
| `super` outside method | `Expr::SuperCall` encountered with `in_method == false` |
| Fields outside method | `Expr::Field` / `Expr::StaticField` with `in_method == false` |
| `break` outside loop | `Stmt::Break` with `loop_depth == 0` |
| `continue` outside loop | `Stmt::Continue` with `loop_depth == 0` |

`loop_depth` is incremented around `while` and `for` bodies.
`in_class` / `in_method` are saved and restored around class and method bodies.

---

## Speculative Type Inference

Forward-flow lattice analysis that assigns an `InferredType` to every expression
and variable. This is *speculative* -- it guides optimization (unboxed `f64`
arithmetic in numeric loops) rather than enforcing correctness.

### Type lattice

Concrete types: `Num`, `Bool`, `Null`, `String`, `List`, `Map`, `Range`, `Fn`.
Named class: `Class(SymbolId)`.
Top element: `Any` (unknown / too complex).

The lattice is flat: `join(T, T) = T`, `join(T, U) = Any` for `T != U`.
There is no explicit Union variant; divergent branches collapse to `Any`.

### Propagation rules

| Expression | Inferred type |
|---|---|
| Numeric literal | `Num` |
| String literal / interpolation | `String` |
| Bool literal | `Bool` |
| `null` | `Null` |
| `[...]` | `List` |
| `{k: v, ...}` | `Map` |
| `a..b` / `a...b` | `Range` |
| Closure (`{ ... }` / `Fn.new { }`) | `Fn` |
| `-x`, `~x` where `x: Num` | `Num` |
| `!x` | `Bool` |
| `+`, `-`, `*`, `/`, `%` with `Num` operands | `Num` |
| `+` with any `String` operand | `String` |
| Bitwise ops with `Num` operands | `Num` |
| Comparison / equality | `Bool` |
| `is` expression | `Bool` |
| Logical `&&` / `\|\|` | `join(left, right)` |
| Conditional `? :` | `join(then, else)` |
| Method call / super call / subscript read | `Any` |
| `var x = init` | type of `init` (or `Null` if uninitialized) |
| `for (i in range)` where range is `Range` | loop var is `Num` |

### Widening on reassignment

`TypeEnv::widen_var` joins the current variable type with the new assigned type.
If a variable typed `Num` is reassigned a `String`, it widens to `Any`.

### is-narrowing

`Expr::Is { value, type_name }` always produces `Bool`. The inference records
this but does not currently narrow the variable's type in subsequent code --
narrowing is left to downstream passes.

### TypeEnv

`TypeEnv` stores two maps (keyed by AST span start position):
- `vars`: variable declaration types.
- `exprs`: expression result types.

Both default to `Any` for unknown keys.
