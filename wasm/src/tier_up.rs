//! Tier-up to wasm — Phase 1 scaffolding.
//!
//! Goal: hot Wren methods get compiled to native wasm functions
//! at runtime via `wren_lift::codegen::wasm::emit_mir`,
//! instantiated by the JS host, and dispatched through a shared
//! funcref table. Closes the ~2 µs/call BC-interpreter floor for
//! compute-heavy code (e.g. `fib(20)`: 41 ms → expected ~2-5 ms).
//!
//! Phased plan in
//! [`memory/project_wasm_tier_up.md`](../../../.claude/...). This
//! module is **Phase 1 plumbing** only:
//!
//!   * Export a `jit_test_emit()` that builds a tiny hand-MIR
//!     (`fn() -> 42`) and returns wasm bytes. Lets the JS host
//!     verify the codegen path is wired in the `wlift_wasm`
//!     crate (no `feature = "host"` weirdness, MIR types
//!     reachable, `emit_mir` produces a validatable module).
//!
//!   * Export `jit_emit_from_source(source, fn_name)` so the
//!     host can compile a real Wren function and inspect bytes.
//!
//! Phase 2 wires:
//!   * the `wren_*` runtime imports the emitted module needs,
//!   * a JS shim (`_wlift_jit_instantiate(bytes)`) that does
//!     `new WebAssembly.Module + Instance` with the imports +
//!     wlift's shared memory + a shared funcref table, and
//!     returns a slot index,
//!   * a dispatch hook in `dispatch_closure_bc_inner` that
//!     `call_indirect`s through the slot before falling through
//!     to the bytecode interpreter.
//!
//! Phase 3+ are: tier-up trigger integration, inter-fn calls
//! (so JIT'd fib calls JIT'd fib without going back through JS),
//! GC root visibility, and async fallback for large modules.
//!
//! Each phase is a single focused commit; this file grows in
//! place rather than getting renamed.

use wasm_bindgen::prelude::*;
use wren_lift::runtime::value::Value;

/// Phase 1 smoke test — emit a hand-built MIR function that
/// returns `42` as a NaN-boxed `Value` and hand back the wasm
/// bytes. JS-side test:
///
/// ```js
/// const bytes = wlift_wasm.jit_test_emit();
/// const mod   = new WebAssembly.Module(bytes);
/// const inst  = new WebAssembly.Instance(mod, { wren: {} });
/// const r     = inst.exports.fn_0();
/// // r is a BigInt — 42.0's NaN-boxed bits = 0x4045000000000000
/// console.log(r === BigInt.asIntN(64, 0x4045000000000000n));
/// ```
///
/// `fn_0` is the export name `emit_mir` uses for this MIR's
/// `name.index() == 0`.
#[wasm_bindgen]
pub fn jit_test_emit() -> Vec<u8> {
    use wren_lift::codegen::wasm::emit_mir;
    use wren_lift::intern::Interner;
    use wren_lift::mir::{Instruction, MirFunction, Terminator};

    let mut interner = Interner::new();
    let name = interner.intern("jit_test_const");
    let mut mir = MirFunction::new(name, 0);

    let bb = mir.new_block();
    let v0 = mir.new_value();
    mir.block_mut(bb)
        .instructions
        .push((v0, Instruction::ConstNum(42.0)));
    mir.block_mut(bb).terminator = Terminator::Return(v0);

    match emit_mir(&mir) {
        Ok(module) => module.bytes,
        Err(_) => Vec::new(),
    }
}

/// Phase 1 — emit wasm bytes for the *first* MIR function
/// produced by compiling `source`. Lets the host smoke-test
/// codegen on real Wren code (e.g. a simple math function),
/// not just hand-built MIR.
///
/// Returns an empty Vec on any compile / emit failure — the
/// host can `bytes.length === 0` to detect.
#[wasm_bindgen]
pub fn jit_emit_from_source(source: &str) -> Vec<u8> {
    use wren_lift::codegen::wasm::emit_mir;
    use wren_lift::runtime::engine::{ExecutionMode, FuncId, InterpretResult};
    use wren_lift::runtime::vm::{VMConfig, VM};

    let mut vm = VM::new(VMConfig {
        execution_mode: ExecutionMode::Interpreter,
        ..VMConfig::default()
    });
    if !matches!(vm.interpret("__jit_test", source), InterpretResult::Success) {
        return Vec::new();
    }

    // Search for the first user-defined function in the engine.
    // FuncId(0) is typically the module top-level; subsequent
    // ids are class methods / closures declared in source.
    for fid in 1u32..(vm.engine.functions.len() as u32) {
        if let Some(mir) = vm.engine.get_mir(FuncId(fid)) {
            // Skip generated-name functions (e.g. closures) so the
            // first hit is the user's first named function.
            let name_str = vm.interner.resolve(mir.name);
            if name_str.starts_with('<') {
                continue;
            }
            return match emit_mir(&mir) {
                Ok(module) => module.bytes,
                Err(_) => Vec::new(),
            };
        }
    }
    Vec::new()
}

// ---------------------------------------------------------------------------
// Phase 2a — runtime helpers that compiled wasm modules import.
// ---------------------------------------------------------------------------
//
// `emit_mir` produces wasm modules that import a `wren` namespace
// — `wren_num_add(i64, i64) -> i64`, `wren_cmp_lt`, etc. To
// instantiate one of those modules, the JS shim needs concrete
// implementations to bind. We export them here, wired to the
// existing `Value` semantics. Each takes/returns a NaN-boxed
// `u64` (which wasm marshals as i64 — same bit pattern).
//
// The implementations match Wren's core arithmetic / comparison
// behaviour: type-check operands, fall back to a safe default
// (currently `null`) on mismatch. Phase 4 (inter-fn calls + GC
// integration) replaces the fallbacks with proper runtime errors
// once the dispatch path is wired.
//
// `wren_call` (general method dispatch) is **not** here — that
// needs access to the live VM and lands in Phase 4 alongside the
// dispatch hook. Compiled functions that only do arithmetic +
// comparisons (the most fib-shaped workloads) work with just
// these helpers.

fn binop_num<F: Fn(f64, f64) -> f64>(a: u64, b: u64, op: F) -> u64 {
    let av = Value::from_bits(a);
    let bv = Value::from_bits(b);
    match (av.as_num(), bv.as_num()) {
        (Some(an), Some(bn)) => Value::num(op(an, bn)).to_bits(),
        _ => Value::null().to_bits(),
    }
}

fn cmp_num<F: Fn(f64, f64) -> bool>(a: u64, b: u64, op: F) -> u64 {
    let av = Value::from_bits(a);
    let bv = Value::from_bits(b);
    match (av.as_num(), bv.as_num()) {
        (Some(an), Some(bn)) => Value::bool(op(an, bn)).to_bits(),
        _ => Value::null().to_bits(),
    }
}

#[wasm_bindgen]
pub fn wren_num_add(a: u64, b: u64) -> u64 {
    binop_num(a, b, |x, y| x + y)
}
#[wasm_bindgen]
pub fn wren_num_sub(a: u64, b: u64) -> u64 {
    binop_num(a, b, |x, y| x - y)
}
#[wasm_bindgen]
pub fn wren_num_mul(a: u64, b: u64) -> u64 {
    binop_num(a, b, |x, y| x * y)
}
#[wasm_bindgen]
pub fn wren_num_div(a: u64, b: u64) -> u64 {
    binop_num(a, b, |x, y| x / y)
}
#[wasm_bindgen]
pub fn wren_num_mod(a: u64, b: u64) -> u64 {
    binop_num(a, b, |x, y| x % y)
}
#[wasm_bindgen]
pub fn wren_num_neg(a: u64) -> u64 {
    let av = Value::from_bits(a);
    match av.as_num() {
        Some(n) => Value::num(-n).to_bits(),
        None => Value::null().to_bits(),
    }
}

#[wasm_bindgen]
pub fn wren_cmp_lt(a: u64, b: u64) -> u64 {
    cmp_num(a, b, |x, y| x < y)
}
#[wasm_bindgen]
pub fn wren_cmp_gt(a: u64, b: u64) -> u64 {
    cmp_num(a, b, |x, y| x > y)
}
#[wasm_bindgen]
pub fn wren_cmp_le(a: u64, b: u64) -> u64 {
    cmp_num(a, b, |x, y| x <= y)
}
#[wasm_bindgen]
pub fn wren_cmp_ge(a: u64, b: u64) -> u64 {
    cmp_num(a, b, |x, y| x >= y)
}

#[wasm_bindgen]
pub fn wren_cmp_eq(a: u64, b: u64) -> u64 {
    let av = Value::from_bits(a);
    let bv = Value::from_bits(b);
    Value::bool(av.equals(bv)).to_bits()
}
#[wasm_bindgen]
pub fn wren_cmp_ne(a: u64, b: u64) -> u64 {
    let av = Value::from_bits(a);
    let bv = Value::from_bits(b);
    Value::bool(!av.equals(bv)).to_bits()
}

#[wasm_bindgen]
pub fn wren_not(a: u64) -> u64 {
    let av = Value::from_bits(a);
    Value::bool(!av.is_truthy_wren()).to_bits()
}
