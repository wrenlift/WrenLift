# Mid-level Intermediate Representation

SSA-based intermediate representation for WrenLift. The MIR uses **block
parameters** (not phi nodes), following the Cranelift/MLIR convention. Each
`MirFunction` is a control-flow graph of `BasicBlock`s, where every block
contains a linear sequence of `Instruction`s and ends with a `Terminator`.
SSA values are referenced by `ValueId`; each instruction produces at most one.

## Core IR Data Structures

- **`ValueId(u32)`** / **`BlockId(u32)`** -- Thin newtypes for type-safe SSA
  value and block references. Display as `v0`, `bb0`, etc.
- **`MirType`** -- Type tag for SSA values: `Value` (NaN-boxed 64-bit Wren
  value), `F64`, `Bool`, `I64`, `Void`.
- **`Instruction`** -- Enum with ~50 variants grouped into:
  - Constants (`ConstNum`, `ConstBool`, `ConstNull`, `ConstString`, `ConstF64`,
    `ConstI64`)
  - Boxed arithmetic/comparison (`Add`, `Sub`, `CmpLt`, ...)
  - Unboxed f64 arithmetic/comparison (`AddF64`, `CmpLtF64`, ...)
  - Logical/bitwise (`Not`, `BitAnd`, `Shl`, ...)
  - Type guards (`GuardNum`, `GuardBool`, `GuardClass`) for speculative
    optimization
  - Box/Unbox conversion
  - Object access (`GetField`, `SetField`, `GetModuleVar`, `SetModuleVar`)
  - Calls (`Call`, `SuperCall`), closures, collections, string ops
  - `Move` (SSA copy) and `BlockParam` (block parameter receiver)
  - Provides `has_side_effects()` (used by DCE) and `operands()` (used by
    analysis and rewriting).
- **`Terminator`** -- `Return(ValueId)`, `ReturnNull`, `Branch` (unconditional
  jump with block args), `CondBranch` (conditional with separate target args),
  `Unreachable`. Provides `successors()` and `operands()`.
- **`BasicBlock`** -- `id`, `params: Vec<(ValueId, MirType)>` (block
  parameters), `instructions`, `terminator`, `predecessors`.
- **`MirFunction`** -- `name`, `arity`, `blocks` (entry is always `blocks[0]`),
  `strings` (constant table), value/block ID counters.
  `compute_predecessors()` populates predecessor lists from terminator edges.
- **Pretty printer** -- `MirFunction::pretty_print()` emits a CLIF-style text
  format (`function %name(arity) { bb0(...): ... }`).

## AST to MIR Lowering

`MirBuilder` walks the Wren AST and emits MIR instructions. Tracks a
`current_block`, a `variables: HashMap<SymbolId, ValueId>` map for local
bindings, and `break_targets`/`continue_targets` stacks for loop control flow.

Key lowering patterns:

- **Expressions** lower to one or more instructions, returning a `ValueId`.
  Literals become `Const*` instructions; binary/unary ops lower to the
  corresponding boxed instruction; method calls become `Call`.
- **`if`** -- Emits `CondBranch` to `then_bb`/`else_bb`, both branching to a
  `merge_bb`.
- **`while`** -- Creates `cond_bb` (loop header), `body_bb`, `exit_bb`.
  The body back-edges to `cond_bb`.
- **`for`** -- Desugars Wren's iterator protocol: calls `iterate(_)` and
  `iteratorValue(_)` on the receiver. The iterator state threads through
  `cond_bb` as a block parameter.
- **Logical `&&`/`||`** -- Short-circuit via `CondBranch`; the result merges
  through a block parameter on `merge_bb`.
- **Ternary `? :`** -- Like `if` but both arms produce a value merged through a
  block parameter.
- **`break`/`continue`** -- Branch to the enclosing loop's exit/header block.
  A dead block is created after the branch so subsequent code has somewhere to
  go.

Entry points: `lower_module()` (top-level statements) and `build_body()`
(function body with parameters).

## MIR Interpreter

An interpreter that executes `MirFunction` by walking blocks, evaluating
instructions, and following terminators. Its purpose is **testing optimization
correctness**: the test pattern is `eval(f) == eval(optimize(f))`.

- `InterpValue` tracks boxed (`Value`) vs. unboxed (`F64`, `I64`, `Bool`)
  representations so `Box`/`Unbox` instructions work correctly.
- Supports constants, all arithmetic (boxed + unboxed), comparisons,
  logical/bitwise, guards, box/unbox, module variables, moves, block
  parameters, and control flow.
- Returns `InterpError::Unsupported` for operations that need a full VM
  (calls, closures, field access, collections).
- Execution steps through blocks up to a configurable limit (default 10000)
  to catch infinite loops.

## opt/ -- Optimization passes

### Infrastructure

- **`trait MirPass`** -- `name() -> &str`, `run(&self, func) -> bool` (returns
  whether the function was modified).
- **`run_pipeline(func, passes)`** -- Runs each pass once in order.
- **`run_to_fixpoint(func, passes, max_iters)`** -- Iterates the pipeline
  until no pass reports changes or the iteration limit is reached.
- **`replace_uses_in_func(func, map)`** -- Rewrites all `ValueId` references
  according to a replacement map, following chains (v1->v2->v3 resolves to v3).
  Used by CSE and other passes.

### ConstFold

Constant folding and propagation. Walks instructions linearly, maintaining a
`HashMap<ValueId, ConstVal>` of known constants.

- Folds arithmetic on constant operands: `ConstNum(2) + ConstNum(3)` becomes
  `ConstNum(5)`. Handles boxed and unboxed (f64) variants, comparisons,
  bitwise, logical not.
- Propagates through `Move`, `Unbox` (Num -> F64), `Box` (F64 -> Num).
- Folds `CondBranch` on constant condition to unconditional `Branch`
  (branch elimination).

### DCE

Dead code elimination, two phases:

1. **Unreachable block removal** -- BFS from the entry block; clears blocks
   with no path from entry.
2. **Dead instruction removal** -- Computes the set of used values by seeding
   from terminators and side-effecting instructions, then propagating
   transitively. Removes instructions whose result is unused and that have no
   side effects.

### CSE

Common subexpression elimination via value numbering. For each pure
instruction, builds a `CseKey` from the instruction discriminant, constant
payload, and resolved operand IDs. If an identical key was already seen,
records a replacement mapping. After the scan, applies `replace_uses_in_func`
to redirect all uses of the duplicate to the original. The now-dead duplicate
instruction is cleaned up by a subsequent DCE pass.

Skips side-effecting instructions, `BlockParam`, `GetModuleVar`, and
`GetUpvalue` (mutable reads).

### TypeSpecialize

Type specialization (devirtualization). When both operands of a boxed
arithmetic or comparison instruction are known-Num (via `ConstNum`, `GuardNum`,
or `Box`), replaces the single boxed op with an Unbox->unboxed op->Box
sequence:

```
v2 = add v0, v1          v3 = unbox v0
           =>             v4 = unbox v1
                          v5 = fadd v3, v4
                          v2 = box v5
```

Handles `Add`/`Sub`/`Mul`/`Div`/`Mod`/`Neg` and `CmpLt`/`CmpGt`/`CmpLe`/`CmpGe`.
The introduced `Unbox`/`Box` pairs are then candidates for constant folding
and CSE elimination.

### Escape Analysis

Determines which heap allocations (`MakeList`, `MakeMap`, `MakeRange`) do NOT
escape the function. A value escapes if it is:

- Returned
- Passed as a branch argument
- Stored to a field, module variable, or upvalue
- Passed as a call/super argument or closure capture
- Nested inside another collection

Returns `HashSet<ValueId>` of non-escaping allocations. Used by SRA.

### SRA

Scalar replacement of aggregates. Eliminates non-escaping `MakeList`
allocations when all `SubscriptGet` accesses use constant indices. Replaces
each `SubscriptGet` with a `Move` of the corresponding list element value,
turning the list into individual SSA scalars. The now-unused `MakeList`
instruction is cleaned up by DCE.

Bails out if any access uses a dynamic (non-constant) index.

## Design decisions

- **Block parameters, not phi nodes.** Values flowing into a block are passed
  as explicit arguments on branch instructions and received as block
  parameters. This follows Cranelift and MLIR rather than LLVM-style phis.
  It makes CFG transformations simpler because there is no separate phi node
  to keep synchronized with predecessor order.
- **Boxed + unboxed value representations.** The IR has dual instruction sets
  (e.g. `Add` vs `AddF64`) and explicit `Box`/`Unbox`. Type specialization
  lowers boxed ops to unboxed when types are known, and constant folding
  propagates through the box/unbox boundary.
- **Side-effect tracking on instructions.** `has_side_effects()` is the
  single predicate that DCE and CSE use to decide what is removable or
  mergeable. Only stores, calls, and upvalue writes are marked effectful.
- **Interpreter-driven testing.** Every optimization pass is tested by
  comparing `eval(f)` before and after the transform, providing a
  correctness oracle without needing the full Wren VM.
- **Single-function scope.** Each `MirFunction` is optimized independently.
  Cross-function analysis (inlining, interprocedural escape) is not yet
  implemented.
