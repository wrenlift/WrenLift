//! Tier-up to wasm.
//!
//! Hot Wren methods get compiled to native wasm functions at runtime
//! via `wren_lift::codegen::wasm::emit_mir`, instantiated by the JS
//! host, and dispatched through a shared funcref table. Closes the
//! ~2 µs/call BC-interpreter floor for compute-heavy code.
//!
//! Pieces:
//!   * `jit_test_emit()` / `jit_emit_from_source` — codegen smoke
//!     hooks for the JS host.
//!   * `wren_*` extern helpers imported by the emitted modules
//!     (arithmetic, comparison, method dispatch, module-var
//!     access, JIT root tracking).
//!   * `register_tier_up_callbacks` plugs compile + dispatch
//!     hooks into the runtime's tier-up broker.

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
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[wasm_bindgen]
pub fn jit_dispatch_hook_hits() -> u64 {
    wren_lift::runtime::tier::dispatch_hook_hits()
}

// Extern bindings for the JS-side instantiate + call shims.
// Installed by `wlift.js` (main mode) and `worker.js` (worker
// mode) right after `await init()`. The shim takes wasm bytes,
// builds a `WebAssembly.Module` + `Instance` with the `wren_*`
// runtime helpers as imports, stores the instance in a per-page
// table, and returns a slot index. Subsequent `jit_call_N(slot,
// fn_idx, args...)` calls dispatch through the slot.
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
    #[wasm_bindgen(js_namespace = globalThis, js_name = _wlift_jit_call_3)]
    fn js_jit_call_3(slot: u32, a: u64, b: u64, c: u64) -> u64;
    #[wasm_bindgen(js_namespace = globalThis, js_name = _wlift_jit_call_4)]
    fn js_jit_call_4(slot: u32, a: u64, b: u64, c: u64, d: u64) -> u64;
    /// JS-provided shim that resets the host's JIT-instance
    /// pool (the array `__wliftJitInstances` and the funcref
    /// table `__wliftJitTable`). Called from
    /// `reset_jit_runtime_caches` at the start of each `run()`
    /// so accumulated entries from previous VMs don't leak as
    /// the page is held open. Safe because each `run()`
    /// instantiates a fresh VM whose `wasm_jit_slots` is empty
    /// — no in-flight reference to a prior slot can survive
    /// across the boundary.
    #[wasm_bindgen(js_namespace = globalThis, js_name = _wlift_jit_reset)]
    fn js_jit_reset();
}

// ---------------------------------------------------------------------------
// Runtime callback bridge
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
    // The wasm SIMD inliner needs the global interner so
    // `emit_mir_with_interner` can resolve a Call's method
    // SymbolId to its textual selector ("bitmask" / "+(_)" /
    // etc.) and decide whether to route through a dedicated
    // intrinsic helper. Pull it from the live VM — `current_vm`
    // is set on every `interpret` / `run_fiber` boundary, so
    // we're inside one whenever tier-up fires.
    let vm_ptr = wren_lift::runtime::tier::current_vm();
    if mir_needs_unsupported_helpers(mir, vm_ptr) {
        COMPILE_REJECT_COUNT.fetch_add(1, Ordering::Relaxed);
        return None;
    }
    let module_result = if vm_ptr.is_null() {
        wren_lift::codegen::wasm::emit_mir(mir)
    } else {
        let interner = unsafe { &(*vm_ptr).interner };
        // Bake the cdylib's `current_module_vars_cell::CURRENT`
        // address into the JIT'd module so `GetModuleVar` lowers
        // to inline `i32.load + i64.load` rather than crossing
        // the wasm-bindgen boundary on every read. The
        // dispatcher updates the cell on each BC→JIT entry.
        let module_vars_ptr_addr = wren_lift::runtime::tier::current_module_vars_addr();
        wren_lift::codegen::wasm::emit_mir_with_runtime_addrs(mir, interner, module_vars_ptr_addr)
    };
    let module = match module_result {
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
/// isn't bound yet. Currently supports the leaf-arithmetic subset
/// (ConstNum / Box / Unbox / arithmetic / comparison / branches /
/// Return) plus arity-1 `Call`, the SIMD intrinsic Calls
/// (`bitmask` / `allTrue` / `anyTrue`, regardless of arity), and
/// `GetModuleVar`. Anything that would require an unbound helper
/// (`wren_make_*`, GC alloc, upvalue access, etc.) gets rejected
/// so instantiation never trips LinkError.
///
/// `vm_ptr` lets the gate resolve a Call's method `SymbolId` to
/// its textual selector — needed for the SIMD-intrinsic exception.
/// Pass null when the interner isn't available (e.g. wasi smoke);
/// the gate falls back to the strict arity-1 rule.
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
fn mir_needs_unsupported_helpers(
    mir: &wren_lift::mir::MirFunction,
    vm_ptr: *mut wren_lift::runtime::vm::VM,
) -> bool {
    use wren_lift::mir::Instruction;
    let interner = if vm_ptr.is_null() {
        None
    } else {
        Some(unsafe { &(*vm_ptr).interner })
    };
    let is_simd_intrinsic_call = |method: wren_lift::intern::SymbolId, arity: usize| -> bool {
        let Some(interner) = interner else {
            return false;
        };
        matches!(
            (interner.resolve(method), arity),
            ("bitmask", 0) | ("allTrue", 0) | ("anyTrue", 0)
        )
    };
    for block in &mir.blocks {
        for (_, inst) in &block.instructions {
            match inst {
                // `Call` supports arities 1..=4 via the
                // `wren_call_<N>_slow` ladder (slow path) plus the
                // per-arity `call_indirect` fast path. SIMD
                // intrinsics route through dedicated helpers
                // regardless of arity. Higher arities still
                // reject — the slow-path ladder caps at 4.
                Instruction::Call { method, args, .. }
                    if !is_simd_intrinsic_call(*method, args.len())
                        && wren_lift::codegen::wasm::call_slow_helper_name(args.len())
                            .is_none() =>
                {
                    return true
                }
                // MakeClosure — per-arity helper
                // (`wren_make_closure_<N>`) for capture counts
                // 0..=4. Higher counts reject.
                Instruction::MakeClosure { upvalues, .. } if upvalues.len() > 4 => return true,
                // GetUpvalue / SetUpvalue currently lower to an
                // external `wren_get_upvalue` / `wren_set_upvalue`
                // call per access — each one a wasm-bindgen
                // boundary round-trip back into JS land. For a
                // capturing closure on a hot path (e.g. fib's
                // recursive self-reference, the SIMD batch's
                // `STEP` capture) the per-access cost dominates
                // the BC interpreter's local frame load, making
                // tier-up a net loss. Reject capturing-closure
                // bodies until the wasm emitter can inline
                // upvalue loads directly against the
                // `current_closure` cell + the closure's
                // upvalue array. MakeClosure stays accepted so
                // closure *construction* still tier-ups; only
                // bodies that actually *read* an upvalue stay
                // BC.
                Instruction::GetUpvalue(_) | Instruction::SetUpvalue(..) => return true,
                // CallStaticSelf — per-arity ladder
                // (`wren_call_static_self_0` … `_4`). Higher
                // arities reject through the gate.
                Instruction::CallStaticSelf { args } if args.len() > 4 => return true,
                // SuperCall — per-arity ladder
                // (`wren_super_call_1` … `_4`). MIR always
                // prepends `this` so args.len() >= 1; cap at 4
                // (this + 3 positional).
                Instruction::SuperCall { args, .. } if !(1..=4).contains(&args.len()) => {
                    return true
                }
                // Subscript get/set support arity-1 (`list[i]`,
                // `map[k]`, `arr[i] = v`) — the host helper is
                // single-index. Multi-arg subscripts reject.
                Instruction::SubscriptGet { args, .. }
                    if wren_lift::codegen::wasm::subscript_get_helper_name(args.len())
                        .is_none() =>
                {
                    return true
                }
                Instruction::SubscriptSet { args, .. }
                    if wren_lift::codegen::wasm::subscript_set_helper_name(args.len())
                        .is_none() =>
                {
                    return true
                }
                // MakeList / MakeMap / StringConcat now have
                // per-arity helpers (`wren_make_list_<N>` etc.)
                // for `N` up to the cap that matches the host
                // ladder. Larger arities still reject — they'd
                // need an additional ladder rung or a buffered
                // shape, which lands separately.
                Instruction::MakeList(elems)
                    if wren_lift::codegen::wasm::make_list_helper_name(elems.len()).is_none() =>
                {
                    return true
                }
                Instruction::MakeMap(pairs)
                    if wren_lift::codegen::wasm::make_map_helper_name(pairs.len()).is_none() =>
                {
                    return true
                }
                Instruction::StringConcat(parts)
                    if wren_lift::codegen::wasm::string_concat_helper_name(parts.len())
                        .is_none() =>
                {
                    return true
                }
                // Below the cap, MakeList / MakeMap /
                // StringConcat / ToString / MakeRange /
                // GetModuleVar / GetField / SetField are all
                // accepted — their `wren_*` helpers are exported
                // by the cdylib and resolve through the wasm-
                // side shim.
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
/// functions per module).
///
/// Currently capped at 2 args; higher arities will move to
/// `call_indirect` through a shared funcref table.
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
fn dispatch_callback(slot: u32, _fn_export_name: &str, args: &[u64]) -> u64 {
    DISPATCH_FROM_BC_COUNT.fetch_add(1, Ordering::Relaxed);
    match args.len() {
        0 => js_jit_call_0(slot),
        1 => js_jit_call_1(slot, args[0]),
        2 => js_jit_call_2(slot, args[0], args[1]),
        // Higher arities not wired yet — return 0 (null bits)
        // so the caller can detect the dispatch failed and fall
        // back.
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

/// Smoke test — emit a hand-built MIR function that returns `42`
/// as a NaN-boxed `Value` and hand back the wasm bytes. JS-side
/// test:
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

/// Emit wasm bytes for the *first* MIR function produced by
/// compiling `source`. Lets the host smoke-test codegen on real
/// Wren code (e.g. a simple math function), not just hand-built
/// MIR.
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
// Runtime helpers that compiled wasm modules import.
// ---------------------------------------------------------------------------
//
// `emit_mir` produces wasm modules that import a `wren` namespace
// — `wren_num_add(i64, i64) -> i64`, `wren_cmp_lt`, etc. The JS
// shim needs concrete implementations to bind; they're exported
// here, wired to the existing `Value` semantics. Each
// takes/returns a NaN-boxed `u64` (which wasm marshals as i64 —
// same bit pattern).
//
// Implementations match Wren's core arithmetic / comparison:
// type-check operands, fall back to `null` on mismatch.

fn binop_num<F: Fn(f64, f64) -> f64>(a: u64, b: u64, sig: &str, op: F) -> u64 {
    let av = Value::from_bits(a);
    if let Some(an) = av.as_num() {
        if let Some(bn) = Value::from_bits(b).as_num() {
            return Value::num(op(an, bn)).to_bits();
        }
    }
    // Type mismatch — try SIMD specializations, then fall back
    // to full method dispatch. Mirrors the host's
    // `wren_arith_dispatch` so observable behaviour matches the
    // BC interpreter for any non-Num receiver.
    if let Some(result) = try_simd_arith_fast_path(a, b, sig) {
        return result;
    }
    method_dispatch_fallback(a, sig, &[Value::from_bits(b)])
}

fn cmp_num<F: Fn(f64, f64) -> bool>(a: u64, b: u64, sig: &str, op: F) -> u64 {
    let av = Value::from_bits(a);
    if let Some(an) = av.as_num() {
        if let Some(bn) = Value::from_bits(b).as_num() {
            return Value::bool(op(an, bn)).to_bits();
        }
    }
    if let Some(result) = try_simd_arith_fast_path(a, b, sig) {
        return result;
    }
    method_dispatch_fallback(a, sig, &[Value::from_bits(b)])
}

/// Try a `Simd4f` / `Simd4i`-specialized fast path for an
/// arithmetic / comparison helper. Cranelift native inlines the
/// equivalent guard + kernel call directly into JIT'd machine
/// code; on wasm we route through the helper instead because
/// the JIT'd module reaches the runtime via wasm imports — but
/// matching here once still skips the cost of
/// `interner.lookup → find_method_with_class → dispatch_method`,
/// which is the next-biggest line item after the f64 fast path.
///
/// Returns `None` on a non-Simd receiver so the caller can fall
/// through to full method dispatch.
fn try_simd_arith_fast_path(a: u64, b: u64, sig: &str) -> Option<u64> {
    use wren_lift::runtime::object::{ObjHeader, ObjSimd, ObjType, SimdKind};
    let av = Value::from_bits(a);
    if !av.is_object() {
        return None;
    }
    let ptr = av.as_object()?;
    if ptr.is_null() {
        return None;
    }
    let header = ptr as *const ObjHeader;
    if unsafe { (*header).obj_type } != ObjType::Simd {
        return None;
    }
    let kind = unsafe { (*(ptr as *const ObjSimd)).kind_tag() };
    match (kind, sig) {
        // f32x4 binops
        (SimdKind::F32x4, "+(_)") => Some(wren_simd4f_add(a, b)),
        (SimdKind::F32x4, "-(_)") => Some(wren_simd4f_sub(a, b)),
        (SimdKind::F32x4, "*(_)") => Some(wren_simd4f_mul(a, b)),
        (SimdKind::F32x4, "/(_)") => Some(wren_simd4f_div(a, b)),
        // f32x4 comparisons (return Simd4i lane masks)
        (SimdKind::F32x4, "==(_)") => Some(wren_simd4f_eq(a, b)),
        (SimdKind::F32x4, "!=(_)") => Some(wren_simd4f_ne(a, b)),
        (SimdKind::F32x4, "<(_)") => Some(wren_simd4f_lt(a, b)),
        (SimdKind::F32x4, "<=(_)") => Some(wren_simd4f_le(a, b)),
        (SimdKind::F32x4, ">(_)") => Some(wren_simd4f_gt(a, b)),
        (SimdKind::F32x4, ">=(_)") => Some(wren_simd4f_ge(a, b)),
        // i32x4 binops + bitwise
        (SimdKind::I32x4, "+(_)") => Some(wren_simd4i_add(a, b)),
        (SimdKind::I32x4, "-(_)") => Some(wren_simd4i_sub(a, b)),
        (SimdKind::I32x4, "*(_)") => Some(wren_simd4i_mul(a, b)),
        (SimdKind::I32x4, "&(_)") => Some(wren_simd4i_and(a, b)),
        (SimdKind::I32x4, "|(_)") => Some(wren_simd4i_or(a, b)),
        (SimdKind::I32x4, "^(_)") => Some(wren_simd4i_xor(a, b)),
        // i32x4 comparisons
        (SimdKind::I32x4, "==(_)") => Some(wren_simd4i_eq(a, b)),
        (SimdKind::I32x4, "!=(_)") => Some(wren_simd4i_ne(a, b)),
        (SimdKind::I32x4, "<(_)") => Some(wren_simd4i_lt(a, b)),
        (SimdKind::I32x4, "<=(_)") => Some(wren_simd4i_le(a, b)),
        (SimdKind::I32x4, ">(_)") => Some(wren_simd4i_gt(a, b)),
        (SimdKind::I32x4, ">=(_)") => Some(wren_simd4i_ge(a, b)),
        _ => None,
    }
}

/// Full method dispatch through the live VM. `sig` is the Wren
/// selector ("+(_)" / "<(_)" / etc.); we hand it straight to
/// `call_method_on`, which handles class lookup, method-table
/// walk, and result boxing. Returns NaN-boxed null when no VM
/// is in scope (defensive — the JIT'd code only runs inside
/// `vm.interpret`, but a stray top-level call shouldn't panic).
fn method_dispatch_fallback(recv_bits: u64, sig: &str, args: &[Value]) -> u64 {
    let vm_ptr = wren_lift::runtime::tier::current_vm();
    if vm_ptr.is_null() {
        return Value::null().to_bits();
    }
    let vm = unsafe { &mut *vm_ptr };
    use wren_lift::runtime::object::NativeContext;
    match vm.call_method_on(Value::from_bits(recv_bits), sig, args) {
        Some(v) => v.to_bits(),
        None => Value::null().to_bits(),
    }
}

// ---------------------------------------------------------------------------
// JIT root tracking
// ---------------------------------------------------------------------------
//
// JIT'd wasm code holds NaN-boxed `Value`s in wasm locals. Those
// locals aren't visible to the GC's normal root scan — if a slow
// path crosses back into the runtime and triggers GC mid-call,
// any object Value still in flight would be reclaimed.
//
// The runtime crate already maintains a process-wide
// `JIT_ROOTS_STORE: Vec<Value>` for the Cranelift JIT (see
// `codegen::runtime_fns::push_jit_root` etc.); the GC drains it
// in `take_jit_roots` and reinstalls forwarded entries via
// `set_jit_roots`. We just expose the same surface to the wasm
// JIT'd module via three thin shims, and emit_mir wraps each
// Call site with snapshot / push args / restore_len so live
// values survive a runtime entry.
//
// `snapshot_len` returns the current depth as `u32`; `push`
// appends a NaN-boxed value; `restore_len` truncates back to a
// snapshot. Single-threaded wasm so no atomic — the runtime's
// `UnsafeCell` does the heavy lifting.

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn wren_jit_root_push(bits: u64) {
    wren_lift::codegen::runtime_fns::push_jit_root(Value::from_bits(bits));
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn wren_jit_roots_snapshot_len() -> u32 {
    wren_lift::codegen::runtime_fns::jit_roots_snapshot_len() as u32
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn wren_jit_roots_restore_len(len: u32) {
    wren_lift::codegen::runtime_fns::jit_roots_restore_len(len as usize);
}

#[wasm_bindgen]
pub fn wren_num_add(a: u64, b: u64) -> u64 {
    binop_num(a, b, "+(_)", |x, y| x + y)
}
#[wasm_bindgen]
pub fn wren_num_sub(a: u64, b: u64) -> u64 {
    binop_num(a, b, "-(_)", |x, y| x - y)
}
#[wasm_bindgen]
pub fn wren_num_mul(a: u64, b: u64) -> u64 {
    binop_num(a, b, "*(_)", |x, y| x * y)
}
#[wasm_bindgen]
pub fn wren_num_div(a: u64, b: u64) -> u64 {
    binop_num(a, b, "/(_)", |x, y| x / y)
}
#[wasm_bindgen]
pub fn wren_num_mod(a: u64, b: u64) -> u64 {
    binop_num(a, b, "%(_)", |x, y| x % y)
}
#[wasm_bindgen]
pub fn wren_num_neg(a: u64) -> u64 {
    let av = Value::from_bits(a);
    if let Some(n) = av.as_num() {
        return Value::num(-n).to_bits();
    }
    // Non-Num: dispatch as the prefix `-` method on the receiver.
    method_dispatch_fallback(a, "-", &[])
}

#[wasm_bindgen]
pub fn wren_cmp_lt(a: u64, b: u64) -> u64 {
    cmp_num(a, b, "<(_)", |x, y| x < y)
}
#[wasm_bindgen]
pub fn wren_cmp_gt(a: u64, b: u64) -> u64 {
    cmp_num(a, b, ">(_)", |x, y| x > y)
}
#[wasm_bindgen]
pub fn wren_cmp_le(a: u64, b: u64) -> u64 {
    cmp_num(a, b, "<=(_)", |x, y| x <= y)
}
#[wasm_bindgen]
pub fn wren_cmp_ge(a: u64, b: u64) -> u64 {
    cmp_num(a, b, ">=(_)", |x, y| x >= y)
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

// ---------------------------------------------------------------------------
// SIMD intrinsic helpers — wasm tier-up fast path.
// ---------------------------------------------------------------------------
//
// The Cranelift backend inlines SIMD ops directly into emitted
// machine code (`try_lower_simd_intrinsic_call`). The wasm
// emitter can't read ObjSimd lanes inline because the JIT'd
// wasm module has its own linear memory, separate from the
// cdylib's heap. Instead, the wasm emitter calls one of these
// per-intrinsic helpers — each takes NaN-boxed `Value`s, type-
// checks the receiver / arg, applies the kernel from
// `simd_kernels`, and allocates the result. The kernel itself
// runs at v128 speed when the cdylib was built with
// `+simd128`, falling through to a scalar loop otherwise.
//
// On a type mismatch we delegate to `wren_call_1_slow` (binops /
// cmps) or `wren_call_0_slow` semantics (reductions) so the
// observable result matches what the slow-path JIT would
// produce. Method names are the standard Wren selectors —
// "+(_)", "==(_)", "bitmask", etc.

#[inline(always)]
fn simd_lanes_if_kind(
    bits: u64,
    expected: wren_lift::runtime::object::SimdKind,
) -> Option<[u32; 4]> {
    use wren_lift::runtime::object::{ObjHeader, ObjSimd, ObjType};
    let v = Value::from_bits(bits);
    if !v.is_object() {
        return None;
    }
    let ptr = v.as_object()?;
    if ptr.is_null() {
        return None;
    }
    let header = ptr as *const ObjHeader;
    if unsafe { (*header).obj_type } != ObjType::Simd {
        return None;
    }
    let simd = ptr as *const ObjSimd;
    if unsafe { (*simd).kind_tag() } != expected {
        return None;
    }
    Some(unsafe { (*simd).lanes })
}

#[inline(always)]
fn make_simd4f(lanes: [u32; 4]) -> u64 {
    let vm_ptr = wren_lift::runtime::tier::current_vm();
    if vm_ptr.is_null() {
        return Value::null().to_bits();
    }
    unsafe {
        (*vm_ptr)
            .new_simd(wren_lift::runtime::object::SimdKind::F32x4, lanes)
            .to_bits()
    }
}

#[inline(always)]
fn make_simd4i(lanes: [u32; 4]) -> u64 {
    let vm_ptr = wren_lift::runtime::tier::current_vm();
    if vm_ptr.is_null() {
        return Value::null().to_bits();
    }
    unsafe {
        (*vm_ptr)
            .new_simd(wren_lift::runtime::object::SimdKind::I32x4, lanes)
            .to_bits()
    }
}

/// Type-mismatch fallback for arity-1 SIMD intrinsics — defers
/// to the same dispatch path `wren_call_1` uses so the result
/// matches the slow-path JIT and the BC interpreter exactly.
#[inline(always)]
fn simd_intrinsic_fallback_1(recv_bits: u64, sig: &str, arg_bits: u64) -> u64 {
    let vm_ptr = wren_lift::runtime::tier::current_vm();
    if vm_ptr.is_null() {
        return Value::null().to_bits();
    }
    let vm = unsafe { &mut *vm_ptr };
    let recv = Value::from_bits(recv_bits);
    let arg = Value::from_bits(arg_bits);
    use wren_lift::runtime::object::NativeContext;
    match vm.call_method_on(recv, sig, &[arg]) {
        Some(v) => v.to_bits(),
        None => Value::null().to_bits(),
    }
}

#[inline(always)]
fn simd_intrinsic_fallback_0(recv_bits: u64, sig: &str) -> u64 {
    let vm_ptr = wren_lift::runtime::tier::current_vm();
    if vm_ptr.is_null() {
        return Value::null().to_bits();
    }
    let vm = unsafe { &mut *vm_ptr };
    let recv = Value::from_bits(recv_bits);
    use wren_lift::runtime::object::NativeContext;
    match vm.call_method_on(recv, sig, &[]) {
        Some(v) => v.to_bits(),
        None => Value::null().to_bits(),
    }
}

macro_rules! simd4f_binop {
    ($name:ident, $sig:literal, $kernel:ident) => {
        #[wasm_bindgen]
        pub fn $name(a: u64, b: u64) -> u64 {
            use wren_lift::runtime::core::simd_kernels as k;
            use wren_lift::runtime::object::SimdKind;
            let Some(la) = simd_lanes_if_kind(a, SimdKind::F32x4) else {
                return simd_intrinsic_fallback_1(a, $sig, b);
            };
            let Some(lb) = simd_lanes_if_kind(b, SimdKind::F32x4) else {
                return simd_intrinsic_fallback_1(a, $sig, b);
            };
            make_simd4f(k::$kernel(la, lb))
        }
    };
}
simd4f_binop!(wren_simd4f_add, "+(_)", f32x4_add);
simd4f_binop!(wren_simd4f_sub, "-(_)", f32x4_sub);
simd4f_binop!(wren_simd4f_mul, "*(_)", f32x4_mul);
simd4f_binop!(wren_simd4f_div, "/(_)", f32x4_div);
simd4f_binop!(wren_simd4f_min, "min(_)", f32x4_min);
simd4f_binop!(wren_simd4f_max, "max(_)", f32x4_max);

macro_rules! simd4f_cmp {
    ($name:ident, $sig:literal, $kernel:ident) => {
        #[wasm_bindgen]
        pub fn $name(a: u64, b: u64) -> u64 {
            use wren_lift::runtime::core::simd_kernels as k;
            use wren_lift::runtime::object::SimdKind;
            let Some(la) = simd_lanes_if_kind(a, SimdKind::F32x4) else {
                return simd_intrinsic_fallback_1(a, $sig, b);
            };
            let Some(lb) = simd_lanes_if_kind(b, SimdKind::F32x4) else {
                return simd_intrinsic_fallback_1(a, $sig, b);
            };
            make_simd4i(k::$kernel(la, lb))
        }
    };
}
simd4f_cmp!(wren_simd4f_eq, "==(_)", f32x4_eq);
simd4f_cmp!(wren_simd4f_ne, "!=(_)", f32x4_ne);
simd4f_cmp!(wren_simd4f_lt, "<(_)", f32x4_lt);
simd4f_cmp!(wren_simd4f_le, "<=(_)", f32x4_le);
simd4f_cmp!(wren_simd4f_gt, ">(_)", f32x4_gt);
simd4f_cmp!(wren_simd4f_ge, ">=(_)", f32x4_ge);

macro_rules! simd4i_binop {
    ($name:ident, $sig:literal, $kernel:ident) => {
        #[wasm_bindgen]
        pub fn $name(a: u64, b: u64) -> u64 {
            use wren_lift::runtime::core::simd_kernels as k;
            use wren_lift::runtime::object::SimdKind;
            let Some(la) = simd_lanes_if_kind(a, SimdKind::I32x4) else {
                return simd_intrinsic_fallback_1(a, $sig, b);
            };
            let Some(lb) = simd_lanes_if_kind(b, SimdKind::I32x4) else {
                return simd_intrinsic_fallback_1(a, $sig, b);
            };
            make_simd4i(k::$kernel(la, lb))
        }
    };
}
simd4i_binop!(wren_simd4i_add, "+(_)", i32x4_add);
simd4i_binop!(wren_simd4i_sub, "-(_)", i32x4_sub);
simd4i_binop!(wren_simd4i_mul, "*(_)", i32x4_mul);
simd4i_binop!(wren_simd4i_min, "min(_)", i32x4_min);
simd4i_binop!(wren_simd4i_max, "max(_)", i32x4_max);
simd4i_binop!(wren_simd4i_and, "&(_)", i32x4_and);
simd4i_binop!(wren_simd4i_or, "|(_)", i32x4_or);
simd4i_binop!(wren_simd4i_xor, "^(_)", i32x4_xor);

macro_rules! simd4i_cmp {
    ($name:ident, $sig:literal, $kernel:ident) => {
        #[wasm_bindgen]
        pub fn $name(a: u64, b: u64) -> u64 {
            use wren_lift::runtime::core::simd_kernels as k;
            use wren_lift::runtime::object::SimdKind;
            let Some(la) = simd_lanes_if_kind(a, SimdKind::I32x4) else {
                return simd_intrinsic_fallback_1(a, $sig, b);
            };
            let Some(lb) = simd_lanes_if_kind(b, SimdKind::I32x4) else {
                return simd_intrinsic_fallback_1(a, $sig, b);
            };
            make_simd4i(k::$kernel(la, lb))
        }
    };
}
simd4i_cmp!(wren_simd4i_eq, "==(_)", i32x4_eq);
simd4i_cmp!(wren_simd4i_ne, "!=(_)", i32x4_ne);
simd4i_cmp!(wren_simd4i_lt, "<(_)", i32x4_lt);
simd4i_cmp!(wren_simd4i_le, "<=(_)", i32x4_le);
simd4i_cmp!(wren_simd4i_gt, ">(_)", i32x4_gt);
simd4i_cmp!(wren_simd4i_ge, ">=(_)", i32x4_ge);

#[wasm_bindgen]
pub fn wren_simd4i_bitmask(recv: u64) -> u64 {
    use wren_lift::runtime::core::simd_kernels as k;
    use wren_lift::runtime::object::SimdKind;
    let Some(lanes) = simd_lanes_if_kind(recv, SimdKind::I32x4) else {
        return simd_intrinsic_fallback_0(recv, "bitmask");
    };
    Value::num(k::i32x4_bitmask(lanes) as f64).to_bits()
}

#[wasm_bindgen]
pub fn wren_simd4i_all_true(recv: u64) -> u64 {
    use wren_lift::runtime::core::simd_kernels as k;
    use wren_lift::runtime::object::SimdKind;
    let Some(lanes) = simd_lanes_if_kind(recv, SimdKind::I32x4) else {
        return simd_intrinsic_fallback_0(recv, "allTrue");
    };
    Value::bool(k::i32x4_all_true(lanes)).to_bits()
}

#[wasm_bindgen]
pub fn wren_simd4i_any_true(recv: u64) -> u64 {
    use wren_lift::runtime::core::simd_kernels as k;
    use wren_lift::runtime::object::SimdKind;
    let Some(lanes) = simd_lanes_if_kind(recv, SimdKind::I32x4) else {
        return simd_intrinsic_fallback_0(recv, "anyTrue");
    };
    Value::bool(k::i32x4_any_true(lanes)).to_bits()
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

/// Look up the JIT slot for a closure receiver, returning
/// `slot + 1` (so `0` means "no JIT, take the slow path"). Used
/// by emit_mir's `Call` lowering to decide between
/// `call_indirect` (fast) and `wren_call_1_slow` (slow). Single
/// wasm-to-wasm cross-module call per Call site — no JS hop.
///
/// The `+ 1` encoding lets the caller emit a single
/// `i32.eqz`-based branch rather than a sentinel comparison
/// against `-1` or similar.
///
/// GC safety note for the slow-path companion `wren_call_1`:
/// JIT'd callers hold NaN-boxed `Value`s in wasm locals, which
/// the GC's root scan can't see. The slow path roots receiver +
/// arg via `JIT_ROOTS_STORE` before any path that might trigger
/// a GC. Pure-arithmetic recursive code doesn't allocate and is
/// safe; do not tier up code that allocates inside its hot loop
/// without similar rooting.
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[wasm_bindgen]
pub fn wren_jit_slot_plus_one(receiver_bits: u64) -> u32 {
    let vm_ptr = wren_lift::runtime::tier::current_vm();
    if vm_ptr.is_null() {
        return 0;
    }
    let vm = unsafe { &*vm_ptr };
    let receiver = Value::from_bits(receiver_bits);
    if !receiver.is_object() {
        return 0;
    }
    let ptr = receiver.as_object().unwrap();
    let header = ptr as *const wren_lift::runtime::object::ObjHeader;
    let is_closure = unsafe { (*header).obj_type == wren_lift::runtime::object::ObjType::Closure };
    if !is_closure {
        return 0;
    }
    let closure_ptr = ptr as *const wren_lift::runtime::object::ObjClosure;
    let fn_ptr = unsafe { (*closure_ptr).function };
    let target_fn = wren_lift::runtime::engine::FuncId(unsafe { (*fn_ptr).fn_id });
    match vm.engine.wasm_jit_slot(target_fn) {
        Some(slot) => {
            DISPATCH_FAST_PATH_COUNT.fetch_add(1, Ordering::Relaxed);
            slot.saturating_add(1)
        }
        None => 0,
    }
}

/// `wren_call_1_slow` — fallback for `wren_jit_slot_plus_one`'s
/// `0` case. Aliased to `wren_call_1` so emit_mir can pick
/// between fast (call_indirect) and slow (this) without name
/// collision; modules importing `wren_call_1` still link.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn wren_call_1_slow(receiver_bits: u64, method_id: u64, arg_bits: u64) -> u64 {
    wren_call_1(receiver_bits, method_id, arg_bits)
}

#[cfg(target_arch = "wasm32")]
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
    // Push receiver + arg as JIT roots before any path that might
    // trigger GC. The wasm JIT'd caller's `receiver_bits` /
    // `arg_bits` are primitive u64s on the wasm stack — invisible
    // to the GC's root scan. Copying them into `JIT_ROOTS_STORE`
    // means a GC inside `call_method_on` walks them, and we
    // re-read post-GC via `jit_root_at` so a moved object's
    // updated bits propagate back to the locals here.
    let root_base = wren_lift::codegen::runtime_fns::jit_roots_snapshot_len();
    wren_lift::codegen::runtime_fns::push_jit_root(Value::from_bits(receiver_bits));
    wren_lift::codegen::runtime_fns::push_jit_root(Value::from_bits(arg_bits));
    let result = wren_call_1_inner(vm, root_base, method_id);
    wren_lift::codegen::runtime_fns::jit_roots_restore_len(root_base);
    result
}

#[cfg(target_arch = "wasm32")]
fn wren_call_1_inner(vm: &mut wren_lift::runtime::vm::VM, root_base: usize, method_id: u64) -> u64 {
    use wren_lift::codegen::runtime_fns::jit_root_at;
    // Re-read receiver / arg through the root store so a GC
    // mid-call sees forwarded pointers reflected in the locals
    // we use here.
    let receiver = jit_root_at(root_base);
    let arg = jit_root_at(root_base + 1);

    // FAST PATH — receiver is a Closure whose function already
    // has a wasm JIT slot installed. Dispatch directly through
    // the slot, skipping `call_method_on` → `call_closure_sync`
    // → BC frame setup. Without this, recursion alternates
    // JIT→BC→JIT→BC every level — each BC frame setup costs
    // more than the BC interp saved by tier-up, making JIT'd
    // self-recursive code *slower* than pure BC.
    //
    // Gated to `target_os = "unknown"` because `js_jit_call_1`
    // is a wasm-bindgen import that only exists on the
    // browser-bindgen target. The wasi smoke build
    // (`wasm32-wasip1`) has no JIT'd modules to dispatch to and
    // falls through to the slow path below.
    #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
    if receiver.is_object() {
        let ptr = receiver.as_object().unwrap();
        let header = ptr as *const wren_lift::runtime::object::ObjHeader;
        let is_closure =
            unsafe { (*header).obj_type == wren_lift::runtime::object::ObjType::Closure };
        if is_closure {
            let closure_ptr = ptr as *mut wren_lift::runtime::object::ObjClosure;
            let fn_ptr = unsafe { (*closure_ptr).function };
            let target_fn = wren_lift::runtime::engine::FuncId(unsafe { (*fn_ptr).fn_id });
            if let Some(slot) = vm.engine.wasm_jit_slot(target_fn) {
                DISPATCH_FAST_PATH_COUNT.fetch_add(1, Ordering::Relaxed);
                // Install the receiver closure so the
                // tier-up'd target's `wren_get_upvalue`
                // resolves the right upvalue array, restore
                // afterwards so a deeper nested call sees its
                // caller's closure. Same shape for the module-
                // vars cell — the callee may live in a different
                // module than the caller, and its inline
                // `GetModuleVar` lowerings read from the cell.
                let prev_closure = wren_lift::runtime::tier::enter_closure(closure_ptr);
                let new_vars = unsafe {
                    wren_lift::runtime::tier::module_vars_ptr_for_closure(vm, closure_ptr)
                };
                let prev_vars = wren_lift::runtime::tier::enter_module_vars(new_vars);
                let result = js_jit_call_1(slot, arg.to_bits());
                wren_lift::runtime::tier::restore_module_vars(prev_vars);
                wren_lift::runtime::tier::restore_closure(prev_closure);
                return result;
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
// Higher-arity Call helpers (2..=4 args).
//
// Same shape as `wren_call_1`: stage receiver + args into the
// JIT root store before any path that might trigger GC, attempt
// the JIT-slot fast path when the receiver is a closure with an
// installed slot, fall through to full method dispatch on miss.
// `*_slow` aliases let the wasm emitter pick between the fast
// (call_indirect) and slow paths without two import names —
// modules that import either link to the right symbol.

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn wren_call_2_slow(receiver_bits: u64, method_id: u64, a0: u64, a1: u64) -> u64 {
    wren_call_2(receiver_bits, method_id, a0, a1)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn wren_call_2(receiver_bits: u64, method_id: u64, a0: u64, a1: u64) -> u64 {
    let vm_ptr = wren_lift::runtime::tier::current_vm();
    if vm_ptr.is_null() {
        return Value::null().to_bits();
    }
    let vm = unsafe { &mut *vm_ptr };
    let root_base = wren_lift::codegen::runtime_fns::jit_roots_snapshot_len();
    wren_lift::codegen::runtime_fns::push_jit_root(Value::from_bits(receiver_bits));
    wren_lift::codegen::runtime_fns::push_jit_root(Value::from_bits(a0));
    wren_lift::codegen::runtime_fns::push_jit_root(Value::from_bits(a1));
    let result = wren_call_n_inner(vm, root_base, method_id, 2);
    wren_lift::codegen::runtime_fns::jit_roots_restore_len(root_base);
    result
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn wren_call_3_slow(receiver_bits: u64, method_id: u64, a0: u64, a1: u64, a2: u64) -> u64 {
    wren_call_3(receiver_bits, method_id, a0, a1, a2)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn wren_call_3(receiver_bits: u64, method_id: u64, a0: u64, a1: u64, a2: u64) -> u64 {
    let vm_ptr = wren_lift::runtime::tier::current_vm();
    if vm_ptr.is_null() {
        return Value::null().to_bits();
    }
    let vm = unsafe { &mut *vm_ptr };
    let root_base = wren_lift::codegen::runtime_fns::jit_roots_snapshot_len();
    wren_lift::codegen::runtime_fns::push_jit_root(Value::from_bits(receiver_bits));
    wren_lift::codegen::runtime_fns::push_jit_root(Value::from_bits(a0));
    wren_lift::codegen::runtime_fns::push_jit_root(Value::from_bits(a1));
    wren_lift::codegen::runtime_fns::push_jit_root(Value::from_bits(a2));
    let result = wren_call_n_inner(vm, root_base, method_id, 3);
    wren_lift::codegen::runtime_fns::jit_roots_restore_len(root_base);
    result
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn wren_call_4_slow(
    receiver_bits: u64,
    method_id: u64,
    a0: u64,
    a1: u64,
    a2: u64,
    a3: u64,
) -> u64 {
    wren_call_4(receiver_bits, method_id, a0, a1, a2, a3)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn wren_call_4(receiver_bits: u64, method_id: u64, a0: u64, a1: u64, a2: u64, a3: u64) -> u64 {
    let vm_ptr = wren_lift::runtime::tier::current_vm();
    if vm_ptr.is_null() {
        return Value::null().to_bits();
    }
    let vm = unsafe { &mut *vm_ptr };
    let root_base = wren_lift::codegen::runtime_fns::jit_roots_snapshot_len();
    wren_lift::codegen::runtime_fns::push_jit_root(Value::from_bits(receiver_bits));
    wren_lift::codegen::runtime_fns::push_jit_root(Value::from_bits(a0));
    wren_lift::codegen::runtime_fns::push_jit_root(Value::from_bits(a1));
    wren_lift::codegen::runtime_fns::push_jit_root(Value::from_bits(a2));
    wren_lift::codegen::runtime_fns::push_jit_root(Value::from_bits(a3));
    let result = wren_call_n_inner(vm, root_base, method_id, 4);
    wren_lift::codegen::runtime_fns::jit_roots_restore_len(root_base);
    result
}

#[cfg(target_arch = "wasm32")]
fn wren_call_n_inner(
    vm: &mut wren_lift::runtime::vm::VM,
    root_base: usize,
    method_id: u64,
    arity: usize,
) -> u64 {
    use wren_lift::codegen::runtime_fns::jit_root_at;
    // Re-read receiver / args through the root store so a GC
    // mid-call sees forwarded pointers reflected in the locals
    // we use here.
    let receiver = jit_root_at(root_base);

    // FAST PATH — receiver is a Closure with a JIT slot. Same
    // rationale as `wren_call_1_inner`; without it, a recursive
    // arity-N method alternates JIT→BC→JIT every level. Per-
    // arity JS shim because each call site has a distinct
    // wasm-bindgen-bound JS name.
    #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
    if receiver.is_object() {
        let ptr = receiver.as_object().unwrap();
        let header = ptr as *const wren_lift::runtime::object::ObjHeader;
        let is_closure =
            unsafe { (*header).obj_type == wren_lift::runtime::object::ObjType::Closure };
        if is_closure {
            let closure_ptr = ptr as *mut wren_lift::runtime::object::ObjClosure;
            let fn_ptr = unsafe { (*closure_ptr).function };
            let target_fn = wren_lift::runtime::engine::FuncId(unsafe { (*fn_ptr).fn_id });
            if let Some(slot) = vm.engine.wasm_jit_slot(target_fn) {
                DISPATCH_FAST_PATH_COUNT.fetch_add(1, Ordering::Relaxed);
                let prev_closure = wren_lift::runtime::tier::enter_closure(closure_ptr);
                let new_vars = unsafe {
                    wren_lift::runtime::tier::module_vars_ptr_for_closure(vm, closure_ptr)
                };
                let prev_vars = wren_lift::runtime::tier::enter_module_vars(new_vars);
                let result = match arity {
                    2 => js_jit_call_2(
                        slot,
                        jit_root_at(root_base + 1).to_bits(),
                        jit_root_at(root_base + 2).to_bits(),
                    ),
                    3 => js_jit_call_3(
                        slot,
                        jit_root_at(root_base + 1).to_bits(),
                        jit_root_at(root_base + 2).to_bits(),
                        jit_root_at(root_base + 3).to_bits(),
                    ),
                    4 => js_jit_call_4(
                        slot,
                        jit_root_at(root_base + 1).to_bits(),
                        jit_root_at(root_base + 2).to_bits(),
                        jit_root_at(root_base + 3).to_bits(),
                        jit_root_at(root_base + 4).to_bits(),
                    ),
                    _ => Value::null().to_bits(),
                };
                wren_lift::runtime::tier::restore_module_vars(prev_vars);
                wren_lift::runtime::tier::restore_closure(prev_closure);
                return result;
            }
        }
    }

    // SLOW PATH — full method dispatch via call_method_on.
    let sym = wren_lift::intern::SymbolId::from_raw(method_id as u32);
    let method_name = vm.interner.resolve(sym).to_string();
    let mut args: Vec<Value> = Vec::with_capacity(arity);
    for i in 0..arity {
        args.push(jit_root_at(root_base + 1 + i));
    }
    use wren_lift::runtime::object::NativeContext;
    match vm.call_method_on(receiver, &method_name, &args) {
        Some(v) => v.to_bits(),
        None => Value::null().to_bits(),
    }
}

/// `wren_get_module_var(slot_idx)` — read a module-level
/// variable. The JIT'd code emits this for any reference to a
/// `var name = …` declared at module scope (closure-recursive
/// `fib` → `fib.call(n-1)` lowers to `GetModuleVar(fib_slot)`).
///
/// **Hardcoded module: `"main"`.** Only user-source vars are
/// reachable; the prelude module isn't. Acceptable because the
/// prelude classes always reject for other reasons (Call arity >
/// 1, MakeList, etc.) so their MIR never compiles to wasm. User-
/// source MIR is always in module `"main"` (see
/// `vm.interpret("main", &combined)` in `lib.rs::run`).
// Cache of the "main" module pointer keyed by VM pointer so the
// per-call HashMap<String,_>::get fallback falls away. Each
// `run()` instantiates a fresh VM, so we key on `vm_ptr` and
// refresh on mismatch — once per run, not per recursion. Only
// the wasm tier hits this hot loop (BC interp resolves module
// vars through different paths), so the cache is gated to
// wasm32-unknown. Single-threaded wasm makes the `Sync` newtype
// pattern (cf. `tier_wasm::current_vm_cell::VmCell`) safe.
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
mod main_module_cache {
    use std::cell::UnsafeCell;
    use wren_lift::runtime::engine::ModuleEntry;
    use wren_lift::runtime::vm::VM;

    pub(super) struct Cell {
        pub(super) vm: UnsafeCell<*mut VM>,
        pub(super) module: UnsafeCell<*const ModuleEntry>,
    }
    unsafe impl Sync for Cell {}

    pub(super) static CACHE: Cell = Cell {
        vm: UnsafeCell::new(std::ptr::null_mut()),
        module: UnsafeCell::new(std::ptr::null()),
    };
}

/// Invalidate per-run JIT caches. Called at the start of every
/// `run()` because the VM is freshly `Box`-allocated and the
/// allocator may reuse a prior run's address — without this
/// reset the next prologue would deref a freed `ModuleEntry`
/// and trap. Also clears the JS-side `__wliftJitInstances` and
/// `__wliftJitTable` so accumulated wasm modules from previous
/// runs become garbage-collectable; otherwise the page would
/// grow without bound on every Run-button click.
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
pub fn reset_jit_runtime_caches() {
    unsafe {
        *main_module_cache::CACHE.vm.get() = std::ptr::null_mut();
        *main_module_cache::CACHE.module.get() = std::ptr::null();
    }
    js_jit_reset();
}

/// Stub for non-wasm builds so `lib.rs::run` can call it
/// unconditionally.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub fn reset_jit_runtime_caches() {}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
fn cached_main_module(
    vm: &wren_lift::runtime::vm::VM,
    vm_ptr: *mut wren_lift::runtime::vm::VM,
) -> *const wren_lift::runtime::engine::ModuleEntry {
    // SAFETY: wasm32 single-threaded; UnsafeCell access is the
    // standard pattern for these runtime caches.
    let cached_vm = unsafe { *main_module_cache::CACHE.vm.get() };
    if cached_vm == vm_ptr {
        let cached_module = unsafe { *main_module_cache::CACHE.module.get() };
        if !cached_module.is_null() {
            return cached_module;
        }
    }
    let module_ptr: *const wren_lift::runtime::engine::ModuleEntry = vm
        .engine
        .modules
        .get("main")
        .map_or(std::ptr::null(), |m| m);
    unsafe {
        *main_module_cache::CACHE.vm.get() = vm_ptr;
        *main_module_cache::CACHE.module.get() = module_ptr;
    }
    module_ptr
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[wasm_bindgen]
pub fn wren_get_module_var(slot_idx: u64) -> u64 {
    let vm_ptr = wren_lift::runtime::tier::current_vm();
    if vm_ptr.is_null() {
        return Value::null().to_bits();
    }
    let vm = unsafe { &*vm_ptr };
    let module_ptr = cached_main_module(vm, vm_ptr);
    if module_ptr.is_null() {
        return Value::null().to_bits();
    }
    let idx = slot_idx as usize;
    // SAFETY: module_ptr is a non-null pointer into
    // `vm.engine.modules` whose backing storage is owned by `vm`
    // for the duration of `run()`. The Vec's stable address is
    // implied by the fact that we don't mutate
    // `vm.engine.modules` between fetches in normal scripts.
    let module = unsafe { &*module_ptr };
    module
        .vars
        .get(idx)
        .copied()
        .unwrap_or(Value::null())
        .to_bits()
}

/// Read field `field_idx` of the receiver instance. The wasm
/// emitter generates this for every `Instruction::GetField`,
/// which is what `obj._foo` lowers to inside the class's own
/// methods. Returns NaN-boxed null on a non-Instance receiver
/// or out-of-range index — matches the BC interpreter's
/// fallback rather than aborting, since the JIT'd path is
/// expected to be hit only after the type-checking front end
/// already validated the access.
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[wasm_bindgen]
pub fn wren_get_field(recv_bits: u64, field_idx: u64) -> u64 {
    use wren_lift::runtime::object::{ObjHeader, ObjInstance, ObjType};
    let v = Value::from_bits(recv_bits);
    if !v.is_object() {
        return Value::null().to_bits();
    }
    let Some(ptr) = v.as_object() else {
        return Value::null().to_bits();
    };
    let header = ptr as *const ObjHeader;
    if unsafe { (*header).obj_type } != ObjType::Instance {
        return Value::null().to_bits();
    }
    let inst = unsafe { &*(ptr as *const ObjInstance) };
    inst.get_field(field_idx as usize)
        .unwrap_or(Value::null())
        .to_bits()
}

/// Companion to `wren_get_field` — write `val_bits` into field
/// `field_idx` of the receiver. Returns the written value (so
/// the wasm emitter can plumb the result of `obj._foo = v`
/// through the next instruction's local-set without a separate
/// load), or null on a non-Instance / out-of-range write.
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[wasm_bindgen]
pub fn wren_set_field(recv_bits: u64, field_idx: u64, val_bits: u64) -> u64 {
    use wren_lift::runtime::object::{ObjHeader, ObjInstance, ObjType};
    let v = Value::from_bits(recv_bits);
    if !v.is_object() {
        return Value::null().to_bits();
    }
    let Some(ptr) = v.as_object() else {
        return Value::null().to_bits();
    };
    let header = ptr as *const ObjHeader;
    if unsafe { (*header).obj_type } != ObjType::Instance {
        return Value::null().to_bits();
    }
    let inst = unsafe { &mut *(ptr as *mut ObjInstance) };
    inst.set_field(field_idx as usize, Value::from_bits(val_bits));
    val_bits
}

// ---------------------------------------------------------------------------
// MakeList / MakeMap / StringConcat / ToString / MakeRange.
//
// Per-arity helpers because wasm imports are typed by signature
// — a single `wren_make_list` import with variable arity would
// collide with other call sites of the same name. The wasm
// emitter picks `wren_make_list_<N>` based on the literal's
// element count; arities beyond the cap reject through
// `mir_needs_unsupported_helpers` and stay BC-interpreted.
//
// Each helper allocates through the live VM (acquired via
// `current_vm()`), pushes inputs into the JIT root store before
// the GC-triggering allocation so they survive a collection
// inside `alloc_*`, then restores the root cursor before
// returning.

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[inline(always)]
fn current_vm_or_null() -> Option<&'static mut wren_lift::runtime::vm::VM> {
    let p = wren_lift::runtime::tier::current_vm();
    if p.is_null() {
        None
    } else {
        Some(unsafe { &mut *p })
    }
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
fn make_list_wasm(elements: &[u64]) -> u64 {
    use wren_lift::codegen::runtime_fns::{
        finish_alloc, jit_root_at, jit_roots_restore_len, jit_roots_snapshot_len, push_jit_root,
    };
    let Some(vm) = current_vm_or_null() else {
        return Value::null().to_bits();
    };
    let root_len_before = jit_roots_snapshot_len();
    for &elem in elements {
        push_jit_root(Value::from_bits(elem));
    }
    let list_ptr = vm.gc.alloc_list();
    let list_val = Value::object(list_ptr as *mut u8);
    push_jit_root(list_val);
    unsafe {
        (*list_ptr).header.class = vm.list_class;
        for idx in 0..elements.len() {
            let elem = jit_root_at(root_len_before + idx);
            (*list_ptr).add(elem);
        }
    }
    let val = jit_root_at(root_len_before + elements.len());
    jit_roots_restore_len(root_len_before);
    unsafe { finish_alloc(vm, val) }
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[wasm_bindgen]
pub fn wren_make_list_0() -> u64 {
    make_list_wasm(&[])
}
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[wasm_bindgen]
pub fn wren_make_list_1(a: u64) -> u64 {
    make_list_wasm(&[a])
}
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[wasm_bindgen]
pub fn wren_make_list_2(a: u64, b: u64) -> u64 {
    make_list_wasm(&[a, b])
}
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[wasm_bindgen]
pub fn wren_make_list_3(a: u64, b: u64, c: u64) -> u64 {
    make_list_wasm(&[a, b, c])
}
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[wasm_bindgen]
pub fn wren_make_list_4(a: u64, b: u64, c: u64, d: u64) -> u64 {
    make_list_wasm(&[a, b, c, d])
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
fn make_map_wasm(pairs: &[(u64, u64)]) -> u64 {
    use wren_lift::codegen::runtime_fns::{
        finish_alloc, jit_root_at, jit_roots_restore_len, jit_roots_snapshot_len, push_jit_root,
    };
    let Some(vm) = current_vm_or_null() else {
        return Value::null().to_bits();
    };
    let root_len_before = jit_roots_snapshot_len();
    for &(k, v) in pairs {
        push_jit_root(Value::from_bits(k));
        push_jit_root(Value::from_bits(v));
    }
    let map_ptr = vm.gc.alloc_map();
    let map_val = Value::object(map_ptr as *mut u8);
    push_jit_root(map_val);
    unsafe {
        (*map_ptr).header.class = vm.map_class;
        for idx in 0..pairs.len() {
            let k = jit_root_at(root_len_before + idx * 2);
            let v = jit_root_at(root_len_before + idx * 2 + 1);
            (*map_ptr).set(k, v);
        }
    }
    let val = jit_root_at(root_len_before + pairs.len() * 2);
    jit_roots_restore_len(root_len_before);
    unsafe { finish_alloc(vm, val) }
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[wasm_bindgen]
pub fn wren_make_map_0() -> u64 {
    make_map_wasm(&[])
}
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[wasm_bindgen]
pub fn wren_make_map_1(k0: u64, v0: u64) -> u64 {
    make_map_wasm(&[(k0, v0)])
}
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[wasm_bindgen]
pub fn wren_make_map_2(k0: u64, v0: u64, k1: u64, v1: u64) -> u64 {
    make_map_wasm(&[(k0, v0), (k1, v1)])
}
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[wasm_bindgen]
pub fn wren_make_map_3(k0: u64, v0: u64, k1: u64, v1: u64, k2: u64, v2: u64) -> u64 {
    make_map_wasm(&[(k0, v0), (k1, v1), (k2, v2)])
}
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn wren_make_map_4(
    k0: u64,
    v0: u64,
    k1: u64,
    v1: u64,
    k2: u64,
    v2: u64,
    k3: u64,
    v3: u64,
) -> u64 {
    make_map_wasm(&[(k0, v0), (k1, v1), (k2, v2), (k3, v3)])
}

/// Build an inclusive / exclusive `Range` from `from..to`. The
/// MIR builder lowers `a..b` and `a...b` into `MakeRange(a, b,
/// inclusive_flag)`; the helper returns the boxed range and
/// stays a single cross-instance call regardless of inputs.
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[wasm_bindgen]
pub fn wren_make_range(from: u64, to: u64, inclusive: u64) -> u64 {
    let Some(vm) = current_vm_or_null() else {
        return Value::null().to_bits();
    };
    let from_val = f64::from_bits(from);
    let to_val = f64::from_bits(to);
    let is_inclusive = inclusive != 0;
    let range_ptr = vm.gc.alloc_range(from_val, to_val, is_inclusive);
    unsafe {
        (*range_ptr).header.class = vm.range_class;
    }
    let val = Value::object(range_ptr as *mut u8);
    unsafe { wren_lift::codegen::runtime_fns::finish_alloc(vm, val) }
}

/// Convert any value to its string form. Used by string
/// interpolation (`"%(x)"`) — every part lowers to a `ToString`
/// before the `StringConcat` joins them, so this is a hot path
/// for any tier-up'd code that builds strings dynamically.
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[wasm_bindgen]
pub fn wren_to_string(val: u64) -> u64 {
    let Some(vm) = current_vm_or_null() else {
        return Value::null().to_bits();
    };
    let v = Value::from_bits(val);
    let s = wren_lift::runtime::vm_interp::value_to_string(vm, v);
    let val = vm.new_string(s);
    unsafe { wren_lift::codegen::runtime_fns::finish_alloc(vm, val) }
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
fn string_concat_wasm(parts: &[u64]) -> u64 {
    let Some(vm) = current_vm_or_null() else {
        return Value::null().to_bits();
    };
    let mut buf = String::new();
    for &p in parts {
        let s = wren_lift::runtime::vm_interp::value_to_string(vm, Value::from_bits(p));
        buf.push_str(&s);
    }
    let val = vm.new_string(buf);
    unsafe { wren_lift::codegen::runtime_fns::finish_alloc(vm, val) }
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[wasm_bindgen]
pub fn wren_string_concat_2(a: u64, b: u64) -> u64 {
    string_concat_wasm(&[a, b])
}
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[wasm_bindgen]
pub fn wren_string_concat_3(a: u64, b: u64, c: u64) -> u64 {
    string_concat_wasm(&[a, b, c])
}
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[wasm_bindgen]
pub fn wren_string_concat_4(a: u64, b: u64, c: u64, d: u64) -> u64 {
    string_concat_wasm(&[a, b, c, d])
}

/// Subscript get — `recv[idx]` for `List` / `Map` / `String` /
/// `TypedArray`, falling through to `[_]` method dispatch for
/// user classes. Forwards to the host's `wren_subscript_get`,
/// which since the unified `vm_ref` fallback uses
/// `current_vm()` on wasm.
///
/// Capped at single-index access (the only arity the host
/// helper supports). Multi-arg subscripts (`m[r, c]`-style) are
/// rejected by the wasm tier-up gate; they're rare in Wren and
/// every concrete user is matrix indexing on `Mat4`, which uses
/// `m.at(r, c)` instead.
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[wasm_bindgen]
pub fn wren_subscript_get_1(receiver: u64, index: u64) -> u64 {
    wren_lift::codegen::runtime_fns::wren_subscript_get(receiver, index)
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[wasm_bindgen]
pub fn wren_subscript_set_1(receiver: u64, index: u64, value: u64) -> u64 {
    wren_lift::codegen::runtime_fns::wren_subscript_set(receiver, index, value)
}

// ---------------------------------------------------------------------------
// Same-class / super dispatch + class-level state + type checks.
//
// Thin `#[wasm_bindgen]` forwarders to the host helpers in
// `runtime_fns`. The host impls all use `vm_ref` which since
// today's earlier change falls back to `current_vm()` on wasm,
// so the same logic covers both targets — these wrappers exist
// purely to surface the symbols through wasm-bindgen's JS-
// boundary export table (raw `#[no_mangle]` symbols on a
// dependent crate aren't necessarily passed through by
// `wasm-pack`).

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[wasm_bindgen]
pub fn wren_set_module_var(slot: u64, value: u64) -> u64 {
    // The host impl reads `JitContext.module_vars` — a pointer
    // populated by the Cranelift entry shim, but null on wasm.
    // Mirror `wren_get_module_var`'s pattern instead: walk the
    // `current_vm()` and mutate the module's `vars` vector
    // directly. The cdylib's `module_vars: NonNull<u64>` cache
    // points at this same Vec's backing storage, so updates here
    // are visible to subsequent `wren_get_module_var` calls.
    let vm_ptr = wren_lift::runtime::tier::current_vm();
    if vm_ptr.is_null() {
        return value;
    }
    let vm = unsafe { &*vm_ptr };
    let module_ptr = cached_main_module(vm, vm_ptr);
    if module_ptr.is_null() {
        return value;
    }
    // SAFETY: cached_main_module returns a stable pointer into
    // `vm.engine.modules`. We need `&mut` for the assignment;
    // wasm32 is single-threaded so the aliasing rules hold.
    let module = unsafe { &mut *(module_ptr as *mut wren_lift::runtime::engine::ModuleEntry) };
    let idx = slot as usize;
    if let Some(slot_ref) = module.vars.get_mut(idx) {
        *slot_ref = Value::from_bits(value);
    }
    value
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[wasm_bindgen]
pub fn wren_call_static_self_0() -> u64 {
    wren_lift::codegen::runtime_fns::wren_call_static_self_0()
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[wasm_bindgen]
pub fn wren_call_static_self_1(a0: u64) -> u64 {
    wren_lift::codegen::runtime_fns::wren_call_static_self_1(a0)
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[wasm_bindgen]
pub fn wren_call_static_self_2(a0: u64, a1: u64) -> u64 {
    wren_lift::codegen::runtime_fns::wren_call_static_self_2(a0, a1)
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[wasm_bindgen]
pub fn wren_call_static_self_3(a0: u64, a1: u64, a2: u64) -> u64 {
    wren_lift::codegen::runtime_fns::wren_call_static_self_3(a0, a1, a2)
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[wasm_bindgen]
pub fn wren_call_static_self_4(a0: u64, a1: u64, a2: u64, a3: u64) -> u64 {
    wren_lift::codegen::runtime_fns::wren_call_static_self_4(a0, a1, a2, a3)
}

/// Super-method dispatch — `super.foo(args)`. The MIR builder
/// always includes `this` as the first arg, so the per-arity
/// ladder starts at 1. `_<N>` takes `(method, this, a0..a_{N-2})`.
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[wasm_bindgen]
pub fn wren_super_call_1(method: u64, this: u64) -> u64 {
    wren_lift::codegen::runtime_fns::wren_super_call_1(method, this)
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[wasm_bindgen]
pub fn wren_super_call_2(method: u64, this: u64, a0: u64) -> u64 {
    wren_lift::codegen::runtime_fns::wren_super_call_2(method, this, a0)
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[wasm_bindgen]
pub fn wren_super_call_3(method: u64, this: u64, a0: u64, a1: u64) -> u64 {
    wren_lift::codegen::runtime_fns::wren_super_call_3(method, this, a0, a1)
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[wasm_bindgen]
pub fn wren_super_call_4(method: u64, this: u64, a0: u64, a1: u64, a2: u64) -> u64 {
    wren_lift::codegen::runtime_fns::wren_super_call_4(method, this, a0, a1, a2)
}

/// Type check — `x is Klass`. Returns NaN-boxed `Bool`.
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[wasm_bindgen]
pub fn wren_is_type(val: u64, class_sym: u64) -> u64 {
    wren_lift::codegen::runtime_fns::wren_is_type(val, class_sym)
}

/// Class assertion — passes the value through if it matches
/// the cached class, else `runtime_error`s. Used by MIR's
/// type-narrowing prologue when the front end deduces a
/// receiver class.
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[wasm_bindgen]
pub fn wren_guard_class(value: u64, class: u64) -> u64 {
    wren_lift::codegen::runtime_fns::wren_guard_class(value, class)
}

/// Read a class-level static field by symbol id. Static fields
/// belong to the enclosing class object; the symbol id is
/// baked at MIR build time.
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[wasm_bindgen]
pub fn wren_get_static_field(field_sym: u64) -> u64 {
    wren_lift::codegen::runtime_fns::wren_get_static_field(field_sym)
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[wasm_bindgen]
pub fn wren_set_static_field(field_sym: u64, value: u64) -> u64 {
    wren_lift::codegen::runtime_fns::wren_set_static_field(field_sym, value)
}

// ---------------------------------------------------------------------------
// Closure capture: MakeClosure / GetUpvalue / SetUpvalue.
//
// `wren_make_closure_<N>` allocates an `ObjClosure` whose
// upvalue array carries the N captured values. The MIR builder
// emits the captures in the order the resolver assigned them
// (locals lifted from the enclosing scope; box once on first
// capture, share through the rest of the chain).
//
// `wren_get_upvalue` / `wren_set_upvalue` operate on the
// closure currently executing the JIT'd function — that
// receiver isn't in the wasm function's parameter list (the
// function takes only the user-visible args), so we read it
// from the `runtime::tier::current_closure` cell. The BC
// interpreter's tier-up dispatch and the JIT-to-JIT recursive
// fast path in `wren_call_<N>_inner` both set the cell via
// `enter_closure` / `restore_closure` around the
// cross-instance call.

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[wasm_bindgen]
pub fn wren_make_closure_0(fn_id: u64) -> u64 {
    wren_lift::codegen::runtime_fns::wren_make_closure_0(fn_id)
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[wasm_bindgen]
pub fn wren_make_closure_1(fn_id: u64, uv0: u64) -> u64 {
    wren_lift::codegen::runtime_fns::wren_make_closure_1(fn_id, uv0)
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[wasm_bindgen]
pub fn wren_make_closure_2(fn_id: u64, uv0: u64, uv1: u64) -> u64 {
    wren_lift::codegen::runtime_fns::wren_make_closure_2(fn_id, uv0, uv1)
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[wasm_bindgen]
pub fn wren_make_closure_3(fn_id: u64, uv0: u64, uv1: u64, uv2: u64) -> u64 {
    wren_lift::codegen::runtime_fns::wren_make_closure_3(fn_id, uv0, uv1, uv2)
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[wasm_bindgen]
pub fn wren_make_closure_4(fn_id: u64, uv0: u64, uv1: u64, uv2: u64, uv3: u64) -> u64 {
    wren_lift::codegen::runtime_fns::wren_make_closure_4(fn_id, uv0, uv1, uv2, uv3)
}

/// Read upvalue `index` of the JIT'd-currently-executing
/// closure. Falls back to NaN-boxed null when the cell isn't
/// set (defensive — JIT'd code only runs through a tier-up
/// dispatch that installs it).
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[wasm_bindgen]
pub fn wren_get_upvalue(index: u64) -> u64 {
    let closure = wren_lift::runtime::tier::current_closure();
    if closure.is_null() {
        return Value::null().to_bits();
    }
    let idx = index as usize;
    unsafe {
        let upvalues = &(*closure).upvalues;
        if idx < upvalues.len() {
            let uv = upvalues[idx];
            (*uv).get().to_bits()
        } else {
            Value::null().to_bits()
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[wasm_bindgen]
pub fn wren_set_upvalue(index: u64, value: u64) -> u64 {
    let closure = wren_lift::runtime::tier::current_closure();
    if closure.is_null() {
        return value;
    }
    let idx = index as usize;
    let v = Value::from_bits(value);
    unsafe {
        let upvalues = &(*closure).upvalues;
        if idx < upvalues.len() {
            let uv = upvalues[idx];
            (*uv).set(v);
            // Inter-generational write barrier — captured
            // upvalues live on the closure's upvalue chain.
            let vm_ptr = wren_lift::runtime::tier::current_vm();
            if !vm_ptr.is_null() {
                let vm = &mut *vm_ptr;
                vm.gc
                    .write_barrier(uv as *mut wren_lift::runtime::object::ObjHeader, v);
            }
        }
    }
    value
}

/// Combined helper: load the closure stored in
/// `module_vars["main"][idx]` *and* return its JIT slot+1 in a
/// single cross-instance call. Halves the prologue's
/// cross-instance overhead vs. calling `wren_get_module_var` and
/// `wren_jit_slot_plus_one` separately, which matters because
/// the prologue runs once per outer call.
///
/// Returns 0 if the module / var / closure isn't JIT'd, mirroring
/// the `slot + 1` encoding of `wren_jit_slot_plus_one`.
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[wasm_bindgen]
pub fn wren_jit_slot_for_module_var(slot_idx: u64) -> u32 {
    let vm_ptr = wren_lift::runtime::tier::current_vm();
    if vm_ptr.is_null() {
        return 0;
    }
    let vm = unsafe { &*vm_ptr };
    let module_ptr = cached_main_module(vm, vm_ptr);
    if module_ptr.is_null() {
        return 0;
    }
    let module = unsafe { &*module_ptr };
    let idx = slot_idx as usize;
    let receiver = match module.vars.get(idx).copied() {
        Some(v) => v,
        None => return 0,
    };
    if !receiver.is_object() {
        return 0;
    }
    let ptr = receiver.as_object().unwrap();
    let header = ptr as *const wren_lift::runtime::object::ObjHeader;
    let is_closure = unsafe { (*header).obj_type == wren_lift::runtime::object::ObjType::Closure };
    if !is_closure {
        return 0;
    }
    let closure_ptr = ptr as *const wren_lift::runtime::object::ObjClosure;
    let fn_ptr = unsafe { (*closure_ptr).function };
    let target_fn = wren_lift::runtime::engine::FuncId(unsafe { (*fn_ptr).fn_id });
    match vm.engine.wasm_jit_slot(target_fn) {
        Some(slot) => {
            DISPATCH_FAST_PATH_COUNT.fetch_add(1, Ordering::Relaxed);
            slot.saturating_add(1)
        }
        None => 0,
    }
}

// ---------------------------------------------------------------------------
// End-to-end smoke test
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

/// Emit the const-42 module, hand bytes to the JS instantiate
/// shim, call the resulting function via the JS call-0 shim,
/// return the raw u64. Returns `0` on any failure.
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
