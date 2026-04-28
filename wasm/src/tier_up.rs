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

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use wasm_bindgen::prelude::*;
use wren_lift::runtime::value::Value;

// ---------------------------------------------------------------------------
// Diagnostic counters
// ---------------------------------------------------------------------------
//
// Lets the host page check whether tier-up is actually firing.
// All four are bumped on the dispatch hot path; cost is one
// relaxed atomic per event, negligible vs. the call itself.

static COMPILE_COUNT: AtomicU32 = AtomicU32::new(0);
static COMPILE_REJECT_COUNT: AtomicU32 = AtomicU32::new(0);
static DISPATCH_FROM_BC_COUNT: AtomicU64 = AtomicU64::new(0);
static DISPATCH_FAST_PATH_COUNT: AtomicU64 = AtomicU64::new(0);

/// Number of MIR functions successfully compiled to wasm.
#[wasm_bindgen]
pub fn jit_compile_count() -> u32 {
    COMPILE_COUNT.load(Ordering::Relaxed)
}

/// Number of MIR functions rejected by the helper-set gate.
#[wasm_bindgen]
pub fn jit_compile_reject_count() -> u32 {
    COMPILE_REJECT_COUNT.load(Ordering::Relaxed)
}

/// Number of dispatches via the BC interp's hook (a normal
/// `Op::Call` in interpreted code that found a JIT'd slot).
#[wasm_bindgen]
pub fn jit_dispatch_from_bc_count() -> u64 {
    DISPATCH_FROM_BC_COUNT.load(Ordering::Relaxed)
}

/// Number of dispatches via `wren_call_1`'s short-circuit
/// (JIT'd code calling another JIT'd function directly,
/// without re-entering the BC interp). High value = recursion
/// stays in JIT; low value with high BC count = the alternating
/// JIT/BC pattern we're trying to avoid.
#[wasm_bindgen]
pub fn jit_dispatch_fast_path_count() -> u64 {
    DISPATCH_FAST_PATH_COUNT.load(Ordering::Relaxed)
}

/// Reset all counters — handy for per-run measurements.
/// Doesn't touch the runtime's `dispatch_hook_hits` counter
/// (it's `pub fn` in tier_wasm.rs without a reset hook); read
/// the delta yourself if you need a per-run number.
#[wasm_bindgen]
pub fn jit_counters_reset() {
    COMPILE_COUNT.store(0, Ordering::Relaxed);
    COMPILE_REJECT_COUNT.store(0, Ordering::Relaxed);
    DISPATCH_FROM_BC_COUNT.store(0, Ordering::Relaxed);
    DISPATCH_FAST_PATH_COUNT.store(0, Ordering::Relaxed);
}

/// Total times the BC interpreter's wasm dispatch hook ran —
/// i.e. how many Wren closure-method calls reached
/// `dispatch_closure_bc_inner`'s wasm-only block. If this stays
/// 0 while a script runs, closure dispatch is going through a
/// path that bypasses the hook (rare but possible — e.g. if a
/// call is intercepted by an earlier match arm in `Op::Call`).
#[wasm_bindgen]
pub fn jit_dispatch_hook_hits() -> u64 {
    wren_lift::runtime::tier::dispatch_hook_hits()
}

// Phase 2b — extern bindings for the JS-side instantiate + call
// shims. Installed by `wlift.js` (main mode) and `worker.js`
// (worker mode) right after `await init()`. The shim takes wasm
// bytes, builds a `WebAssembly.Module` + `Instance` with the
// `wren_*` runtime helpers as imports, stores the instance in a
// per-page table, and returns a slot index. Subsequent
// `jit_call_N(slot, fn_idx, args...)` calls dispatch through
// the slot.
//
// Future-proof signatures: the Phase 4 inter-fn-call work
// replaces these per-arity exports with `call_indirect` through
// a shared funcref table; the per-arity shims here are
// transitional.
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = globalThis, js_name = _wlift_jit_instantiate)]
    fn js_jit_instantiate(bytes: &[u8]) -> u32;
    /// Each slot holds a direct reference to the module's
    /// `fn_*` export — the JS shim resolves it once at
    /// instantiate time, so the per-arity call shims don't pay
    /// name-lookup cost on the dispatch hot path.
    #[wasm_bindgen(js_namespace = globalThis, js_name = _wlift_jit_call_0)]
    fn js_jit_call_0(slot: u32) -> u64;
    #[wasm_bindgen(js_namespace = globalThis, js_name = _wlift_jit_call_1)]
    fn js_jit_call_1(slot: u32, a: u64) -> u64;
    #[wasm_bindgen(js_namespace = globalThis, js_name = _wlift_jit_call_2)]
    fn js_jit_call_2(slot: u32, a: u64, b: u64) -> u64;
}

// ---------------------------------------------------------------------------
// Phase 3a — runtime callback bridge
// ---------------------------------------------------------------------------
//
// The runtime crate (`wren_lift`) decides *when* to tier up;
// this crate (`wlift_wasm`) handles *how* via wasm-bindgen +
// JS instantiation. The bridge: at module init,
// `register_tier_up_callbacks()` plugs Rust function pointers
// into `wren_lift::runtime::tier::set_*_callback`. The runtime
// crate stays wasm-bindgen-free.

/// Compile a MIR function via `emit_mir` and instantiate via the
/// JS shim. Returns the slot index in the page's JIT instance
/// table. `None` on emit / instantiate failure or if the MIR
/// uses instructions whose runtime helpers we haven't bound yet
/// (would LinkError at instantiation).
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
fn compile_callback(mir: &wren_lift::mir::MirFunction) -> Option<u32> {
    if mir_needs_unsupported_helpers(mir) {
        // Phase 4 lifts this — wires `wren_call`, `wren_make_*`,
        // `wren_to_string`, etc. through a thread-local VM
        // pointer so JIT'd code can call back into the runtime.
        COMPILE_REJECT_COUNT.fetch_add(1, Ordering::Relaxed);
        return None;
    }
    let module = match wren_lift::codegen::wasm::emit_mir(mir) {
        Ok(m) => m,
        Err(_) => {
            COMPILE_REJECT_COUNT.fetch_add(1, Ordering::Relaxed);
            return None;
        }
    };
    let bytes = module.bytes;
    if bytes.is_empty() {
        COMPILE_REJECT_COUNT.fetch_add(1, Ordering::Relaxed);
        return None;
    }
    let slot = js_jit_instantiate(&bytes);
    COMPILE_COUNT.fetch_add(1, Ordering::Relaxed);
    Some(slot)
}

/// Walk MIR for any instruction whose `wren_*` runtime helper
/// isn't bound yet. Phase 3 supports the leaf-arithmetic subset:
/// ConstNum / Box / Unbox / Add / Sub / Mul / Div / Mod / Neg /
/// Cmp* / Not / IsTruthy / Branch / CondBranch / Return /
/// ReturnNull. Anything that would require a `wren_call`,
/// `wren_make_*`, GC alloc, or upvalue access gets rejected so
/// instantiation never trips LinkError.
fn mir_needs_unsupported_helpers(mir: &wren_lift::mir::MirFunction) -> bool {
    use wren_lift::mir::Instruction;
    for block in &mir.blocks {
        for (_, inst) in &block.instructions {
            match inst {
                // `Call` is supported for arity 1 (Phase 4 step 2's
                // `wren_call_1` helper). Other arities still reject
                // until those helpers land.
                Instruction::Call { args, .. } if args.len() != 1 => return true,
                Instruction::CallStaticSelf { .. }
                | Instruction::SuperCall { .. }
                | Instruction::MakeClosure { .. }
                | Instruction::GetUpvalue(_)
                | Instruction::SetUpvalue(..)
                | Instruction::MakeList(_)
                | Instruction::MakeMap(_)
                | Instruction::MakeRange(..)
                | Instruction::StringConcat(_)
                | Instruction::ToString(_)
                | Instruction::SubscriptGet { .. }
                | Instruction::SubscriptSet { .. }
                | Instruction::GetField(..)
                | Instruction::SetField(..)
                | Instruction::GetStaticField(..)
                | Instruction::SetStaticField(..)
                | Instruction::GetModuleVar(_)
                | Instruction::SetModuleVar(..)
                | Instruction::IsType(..)
                | Instruction::GuardClass(..) => return true,
                _ => {}
            }
        }
    }
    false
}

/// Invoke a JIT'd slot via the JS shim. Args are NaN-boxed
/// `u64`s (wasm-bindgen marshals as BigInt). The slot already
/// holds a direct function reference, so this is a single
/// JS hop into a wasm-to-wasm cross-module call.
///
/// `_fn_export_name` is unused — the JS shim resolved the
/// `fn_*` export once at instantiate time. Kept in the signature
/// because the runtime crate's `DispatchCallback` type carries
/// it for future expansion (e.g. tables exporting multiple
/// functions per module in Phase 4).
///
/// Currently capped at 2 args (fib's signature). Phase 4 swaps
/// per-arity shims for `call_indirect` through a shared funcref
/// table.
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
fn dispatch_callback(slot: u32, _fn_export_name: &str, args: &[u64]) -> u64 {
    DISPATCH_FROM_BC_COUNT.fetch_add(1, Ordering::Relaxed);
    match args.len() {
        0 => js_jit_call_0(slot),
        1 => js_jit_call_1(slot, args[0]),
        2 => js_jit_call_2(slot, args[0], args[1]),
        // Phase 4 generalises this — return 0 (null bits) so the
        // caller can detect the dispatch failed and fall back.
        _ => 0,
    }
}

/// Plug the compile + dispatch hooks into the runtime's
/// tier-up broker. Called from `_wasm_init` so callbacks are
/// armed before any user code runs.
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
pub fn register_tier_up_callbacks() {
    wren_lift::runtime::tier::set_compile_callback(compile_callback);
    wren_lift::runtime::tier::set_dispatch_callback(dispatch_callback);
}

/// Host stub so non-wasm builds still link cleanly.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub fn register_tier_up_callbacks() {}

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

/// Truthiness probe — emit_mir uses this for `if`/`while` tests
/// on boxed values. Returns `1` for truthy, `0` for falsy. Note
/// the result type is i32 (a wasm bool) not i64 — the codegen
/// wires it directly into a `br_if` so an i64 boxed bool would
/// need an extra unboxing step.
#[wasm_bindgen]
pub fn wren_is_truthy(a: u64) -> u32 {
    let av = Value::from_bits(a);
    if av.is_truthy_wren() {
        1
    } else {
        0
    }
}

/// One-arg method call — `receiver.method(arg)`.
///
/// Phase 4 step 2: emit_mir lowers `Call` instructions to
/// `wren_call_<argc>` imports. The 1-arg variant is enough to
/// unlock self-recursive numeric code (`fib.call(n)`,
/// `factorial.call(n)`, etc.). Higher arities follow the same
/// template; the runtime's `call_method_on` handles any arity.
///
/// Method dispatch:
///   * `method_id` is the `SymbolId` index emit_mir baked in
///     when compiling — `mir.name.index() as i64`. Resolves
///     against the live VM's interner (must be the same VM
///     that compiled the MIR; we can't migrate slots across
///     VMs).
///   * `call_method_on` short-circuits to `call_closure_sync`
///     when the receiver is a Closure and the method name
///     starts with `call`. That's the hot path for
///     `fib.call(n)`-style recursion.
///
/// **GC SAFETY (Phase 4 step 4 territory):** the JIT'd caller
/// holds NaN-boxed `Value`s in wasm locals while this runs.
/// The wlift GC has no visibility into wasm locals — if a
/// callee allocates and triggers a GC pass, those locals can
/// dangle. fib + similar pure-arithmetic recursive code
/// doesn't allocate and is safe. **Do not tier up code that
/// allocates inside its hot loop until step 4 lands.** The
/// MIR reject list catches most allocating instructions
/// (MakeList / MakeMap / StringConcat / ToString); the gap is
/// callees reached via `Call` that themselves allocate.
#[wasm_bindgen]
pub fn wren_call_1(receiver_bits: u64, method_id: u64, arg_bits: u64) -> u64 {
    let vm_ptr = wren_lift::runtime::tier::current_vm();
    if vm_ptr.is_null() {
        // No VM in scope — must be called from outside any
        // `run_fiber` window, which shouldn't happen since
        // JIT'd code only runs through a dispatch from
        // `dispatch_closure_bc_inner`. Be defensive.
        return Value::null().to_bits();
    }
    let vm = unsafe { &mut *vm_ptr };
    let receiver = Value::from_bits(receiver_bits);
    let arg = Value::from_bits(arg_bits);

    // FAST PATH — receiver is a Closure whose function already
    // has a wasm JIT slot installed. Dispatch directly through
    // the slot, skipping `call_method_on` →
    // `call_closure_sync` → BC frame setup. Without this,
    // recursion alternates JIT→BC→JIT→BC every level (each
    // BC frame setup costs more than the BC interp saved by
    // tier-up), making JIT'd self-recursive code *slower* than
    // pure BC.
    //
    // Eligibility: receiver must be a Closure object whose
    // FuncId has a slot in `wasm_jit_slots`. Skips for non-
    // closure receivers, generic method dispatch, or
    // un-tier-up'd functions.
    //
    // Gated to `target_os = "unknown"` (the browser-bindgen
    // target) because `js_jit_call_1` is a wasm-bindgen
    // import that only exists there. The wasi smoke build
    // (`wasm32-wasip1`) doesn't have JIT'd modules to dispatch
    // to and falls through to the slow path below.
    #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
    if receiver.is_object() {
        let ptr = receiver.as_object().unwrap();
        let header = ptr as *const wren_lift::runtime::object::ObjHeader;
        let is_closure =
            unsafe { (*header).obj_type == wren_lift::runtime::object::ObjType::Closure };
        if is_closure {
            let closure_ptr = ptr as *const wren_lift::runtime::object::ObjClosure;
            let fn_ptr = unsafe { (*closure_ptr).function };
            let target_fn = wren_lift::runtime::engine::FuncId(unsafe { (*fn_ptr).fn_id });
            if let Some(slot) = vm.engine.wasm_jit_slot(target_fn) {
                DISPATCH_FAST_PATH_COUNT.fetch_add(1, Ordering::Relaxed);
                return js_jit_call_1(slot, arg_bits);
            }
        }
    }

    // SLOW PATH — full method dispatch. `method_id` was baked
    // in at compile time as a `SymbolId` index. Resolve back
    // to a method-name string and let the runtime's existing
    // closure-fast-path / find_method machinery handle it.
    // Cloning the &str because `call_method_on` borrows
    // `&mut self`, conflicting with the &str borrow against
    // `vm.interner`.
    let sym = wren_lift::intern::SymbolId::from_raw(method_id as u32);
    let method_name = vm.interner.resolve(sym).to_string();

    use wren_lift::runtime::object::NativeContext;
    match vm.call_method_on(receiver, &method_name, &[arg]) {
        Some(v) => v.to_bits(),
        None => Value::null().to_bits(),
    }
}

// ---------------------------------------------------------------------------
// Phase 2b — end-to-end smoke test
// ---------------------------------------------------------------------------
//
// Drives `emit_mir` → `js_jit_instantiate` → `js_jit_call_0` so
// callers can verify the full tier-up round-trip works without
// hand-marshalling bytes from JS. Returns the raw u64 the
// compiled function produced — for `jit_test_emit` this is the
// NaN-boxed bits of `42.0` (= `0x4045_0000_0000_0000`).
//
// JS-side check after init():
//
//   const r = wlift_wasm.jit_smoke_run_const();
//   console.log(r === 0x4045000000000000n); // true → end-to-end works

/// Phase 2b smoke — emit the const-42 module, hand bytes to the
/// JS instantiate shim, call the resulting function via the
/// JS call-0 shim, return the raw u64. Returns `0` on any
/// failure (cleanest signal for the JS test code).
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[wasm_bindgen]
pub fn jit_smoke_run_const() -> u64 {
    let bytes = jit_test_emit();
    if bytes.is_empty() {
        return 0;
    }
    let slot = js_jit_instantiate(&bytes);
    js_jit_call_0(slot)
}
