# Benchmark Issues & Remediation Plan

Discovered during benchmark comparison against standard Wren 0.4.0.

## Current Results (Apple M3, release build, best of 5 runs)

| Benchmark       | WrenLift   | Wren 0.4 | Ratio  | Status               |
|-----------------|------------|----------|--------|----------------------|
| Recursive Fib   | 0.62s      | 0.17s    | 3.5x   | Runs, improving      |
| Method Call     | 0.36s      | 0.09s    | 4.1x   | Runs, improving      |
| Binary Trees    | CRASH      | 0.97s    | —      | SIGABRT at depth ~10 |
| DeltaBlue       | COMPILE_ERR| —        | —      | Blocked by Issue 4   |

*History: original 27.2x/29.8x → P2 hot-path fixes 24.0x/25.4x → Vec registers 12.4x/13.5x → bytecode VM 5.4x/5.5x → method cache 3.5x/4.1x*

---

## Issue 1: MIR interpreter is ~13x slower than Wren's bytecode VM

**Severity:** Performance
**Benchmarks affected:** All

### 1a. HashMap register file (highest impact)

**File:** `src/runtime/object.rs:721` — `MirCallFrame.values: HashMap<ValueId, InterpValue>`
**Hot path:** `vm_interp.rs:752` `values.insert()` per instruction, `vm_interp.rs:928` `values.get()` per operand.

Every instruction does 2-4 HashMap lookups (get operands + insert result). ValueId is a dense u32.

**Fix:** Replace `HashMap<ValueId, InterpValue>` with `Vec<InterpValue>` indexed by `ValueId.0 as usize`. O(1) array index vs O(1)-amortized hash probe (~3-4x speedup on value access alone).

### 1b. String allocations in dispatch hot path

Every method call allocates strings that could be avoided:

| Location | Pattern | Why unnecessary |
|----------|---------|-----------------|
| `vm_interp.rs:222` | `vm.interner.resolve(*method).to_string()` for `starts_with("call")` check | Compare against interned "call" SymbolId instead |
| `vm_interp.rs:254` | `format!("static:{}", method_name_str)` | Pre-intern static prefixed symbols at class bind time |
| `vm_interp.rs:450` | `format!("[{}]", vec!["_"; n].join(","))` per subscript | Cache the 5 common signatures `[_]`..`[_,_,_,_,_]` |
| `vm_interp.rs:489` | `format!("[{}]=(_)", ...)` per subscript set | Same — pre-intern |
| `vm_interp.rs:569` | `to_string()` per class in IsType hierarchy walk | Compare `SymbolId` (u32 ==) instead of String == |

### 1c. Double method lookup per call

`vm_interp.rs:245-248` — calls `find_method()` then `find_method_class()`, both walking the full superclass chain with HashMap lookups.

**Fix:** Single `find_method_with_class()` that returns `(Method, *mut ObjClass)` in one pass.

### 1d. Vec allocation per method call

`vm_interp.rs:216-219` — `vec![recv_val]` + push args for every call. `vm_interp.rs:328` — `arg_vals.clone()` for constructors.

**Fix:** Use a reusable `SmallVec<[Value; 8]>` or a thread-local scratch buffer. Most Wren methods have ≤4 args.

### 1e. Block param binding allocations

`vm_interp.rs:936` — `collect()` args into intermediate Vec, then insert each into HashMap.

**Fix:** Direct single-pass: read each arg from values and insert the target param in one loop.

### Expected impact

| Fix | Estimated speedup | Effort |
|-----|-------------------|--------|
| 1a (Vec registers) | 3-5x | Medium — touch MirCallFrame + all access sites |
| 1b (kill string allocs) | 1.5-2x | Small — targeted changes |
| 1c (single method walk) | 1.1-1.3x | Small — refactor one function |
| 1d (SmallVec args) | 1.1-1.2x | Small — swap Vec for SmallVec |
| 1e (block param pass) | 1.05x | Trivial |
| **Combined** | **~5-10x** | |

Steps 1a+1b alone should bring ratio from 30x → ~5-8x. JIT tier-up would eliminate the remaining gap.

---

## Issue 2: super() constructor dispatches with wrong arity

**Severity:** Bug — correctness
**Benchmarks affected:** method_call (blocked NthToggle inheritance pattern)

**Repro:**
```wren
class Base {
  construct new(x) { _x = x }
}
class Child is Base {
  construct new(x, y) {
    super(x)  // ERROR: looks for Base.new(_,_) instead of Base.new(_)
    _y = y
  }
}
```

**Root cause:** When lowering `super(args...)` inside a constructor, the MIR builder constructs the method signature using the *enclosing* constructor's parameter count (2 params → `new(_,_)`) instead of counting the actual `super(...)` call arguments (1 arg → `new(_)`).

**Fix:** In the MIR builder's super-call handling, build the signature from `super_call.args.len()`, not from the enclosing method's parameter list.

**Files:** `src/mir/builder.rs` — super call lowering in constructor context.

---

## Issue 3: SIGABRT under heavy GC pressure (binary_trees)

**Severity:** Crash — correctness
**Benchmarks affected:** binary_trees

**Repro:** `bench/binary_trees.wren` at `maxDepth = 14` — processes stretch tree and first few iterations of depth-4 and depth-6 loops, then crashes with SIGABRT during depth-10 iteration (~500K+ cumulative allocations).

**Root cause candidates (most to least likely):**
1. **Missing GC roots** — temporary `Value` objects in MIR interpreter frames (`HashMap<ValueId, InterpValue>`) aren't registered as GC roots during collection. When GC fires mid-iteration, live references get collected.
2. **Dangling pointer after collection** — an `ObjInstance` field or constructor return value points to memory that was freed during a GC sweep.
3. **Nursery overflow** — objects allocated faster than promotion can handle, leading to memory corruption in the bump allocator.

**Fix plan:**
1. Audit `vm_interp.rs` to ensure all live `InterpValue::Boxed(Value)` in every active `MirCallFrame` are traced as GC roots.
2. Add a stress-test GC mode (`GC_STRESS=1`) that collects on every allocation to surface dangling pointers deterministically.
3. Run binary_trees under AddressSanitizer: `RUSTFLAGS="-Zsanitizer=address" cargo run bench/binary_trees.wren`

**Files:** `src/runtime/gc.rs`, `src/runtime/vm_interp.rs`

---

## Issue 4: Implicit `this` method calls not resolved

**Severity:** Bug — correctness
**Benchmarks affected:** delta_blue (and any real-world Wren code using this pattern)

**Repro:**
```wren
class Foo {
  construct new() { _val = 42 }
  value { _val }
  test() {
    return value  // ERROR: "undefined variable 'value'"
    // Should resolve to: return this.value
  }
}
```

In standard Wren, bare identifiers inside a method body that don't match any local/module variable are resolved as method calls on `this`. This is extremely common in real Wren code — nearly every class with getters/methods relies on it.

**Root cause:** The semantic resolver (`src/sema/resolve.rs`) only checks local variables, upvalues, and module-level variables. It doesn't fall back to checking whether the identifier is a method/getter on the current class (or its superclass chain).

**Fix plan:**
1. In `resolve.rs`, when a bare identifier lookup fails all scopes and we're inside a class method body, emit it as an implicit `this.identifier` call instead of an error.
2. In Wren, this applies to: getters (`value`), methods with args (`doSomething(x)`), and the setter pattern (`value = x` → `this.value = x`).
3. The resolver doesn't need to verify the method exists at compile time — that's a runtime dispatch. It just needs to rewrite bare identifiers as `this`-dispatched calls when no variable binding is found.

**Files:** `src/sema/resolve.rs`, possibly `src/mir/builder.rs`

---

## Priority Order

| Priority | Issue | Why |
|----------|-------|-----|
| **P0** | Issue 4 (implicit this) | Blocks most real-world Wren programs |
| **P0** | Issue 2 (super arity) | Blocks inheritance, a core language feature |
| **P1** | Issue 3 (GC crash) | Blocks allocation-heavy workloads |
| **P2** | Issue 1b (string allocs) | Low-hanging fruit, ~2x improvement |
| **P2** | Issue 1c-1e (dispatch overhead) | Small targeted fixes, ~1.3x combined |
| **P3** | Issue 1a (Vec registers) | Biggest perf win but most invasive change |
