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
