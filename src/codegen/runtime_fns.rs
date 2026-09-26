//! Runtime functions callable from JIT-compiled code.
//!
//! These are `extern "C"` functions that the JIT emits `CallRuntime` for.
//! The `resolve` function maps a runtime function name to its address so
//! the codegen can patch `CallRuntime` → `CallInd` with the actual pointer.
//!
//! # Convention
//! - All values are passed/returned as NaN-boxed `u64`
//! - VM context is accessed via thread-local `JIT_CONTEXT`
//! - x86_64: System V ABI (RDI, RSI, RDX, RCX, R8, R9 → RAX)
//! - aarch64: AAPCS64 (X0-X7 → X0)
//!
//! # Safety
//!
//! Every `pub unsafe extern "C"` entry below is a JIT-ABI runtime entry
//! point called only from compiled code; the safety contract is the
//! module-wide convention above — callers are the JIT emitter, which
//! respects the ABI it was emitted against. The per-item
//! `missing_safety_doc` lint is silenced so each arch-gated `#[cfg]`
//! twin doesn't need its own duplicated `# Safety` block.

#![allow(clippy::missing_safety_doc)]

use crate::runtime::gc_trait::GcAllocator;
use crate::runtime::object::{
    MapKey, Method, NativeContext, ObjClass, ObjClosure, ObjHeader, ObjInstance, ObjList, ObjMap,
    ObjSimd, ObjString, ObjType, SimdKind,
};
use crate::runtime::value::Value;
use std::sync::OnceLock;

/// Flat shadow root storage. All shadow frames share a single contiguous
/// `Vec<Value>`. Push/pop is offset arithmetic — zero heap allocation after
/// the Vec capacity stabilizes (typically after the first few calls).
#[derive(Default)]
struct FlatShadowStack {
    /// All shadow root values, contiguous. Frames are stacked sequentially.
    roots: Vec<Value>,
    /// Stack of frame start offsets. boundaries[i] is the start of frame i.
    boundaries: Vec<u16>,
}

/// Shadow roots pointer — written by Rust, read by JIT-generated code via raw ldr.
/// Uses volatile read/write to prevent compiler reordering across JIT calls.
static mut CURRENT_NATIVE_SHADOW_ROOTS: *mut Value = std::ptr::null_mut();

// Stack of active JIT frames for GC stack walking.
// Each entry: (frame_pointer, func_id, return_address).
// The return_address identifies the active safepoint for precise root scanning.
/// Push a JIT frame for GC visibility.
#[inline(always)]
pub fn push_jit_frame(fp: usize, func_id: u32, ret_addr: usize) {
    push_frame_on(jit_state(), fp, func_id, ret_addr);
}

#[inline(always)]
fn push_frame_on(j: *mut JitThread, fp: usize, func_id: u32, ret_addr: usize) {
    let frames = unsafe { &mut (*j).frames };
    if frames.len() < 64 {
        frames.push((fp, func_id, ret_addr));
    }
}

/// Pop a JIT frame.
#[inline(always)]
pub fn pop_jit_frame() {
    unsafe { (*jit_state()).frames.pop() };
}

/// Get all active JIT frames (fp, func_id, ret_addr) for GC stack walking.
pub fn jit_frame_entries() -> Vec<(usize, u32, usize)> {
    unsafe { (*jit_state()).frames.clone() }
}

/// Trace switches read once: these gates sit on every dispatch.
fn env_flag(cell: &'static std::sync::OnceLock<bool>, name: &str) -> bool {
    *cell.get_or_init(|| std::env::var_os(name).is_some())
}

#[inline(always)]
fn trace_jit_ic(msg: impl FnOnce() -> String) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if env_flag(&ON, "WLIFT_TRACE_JIT_IC") {
        eprintln!("{}", msg());
    }
}

/// The method symbol in a call's packed word, and the call site's
/// inline cache when the word names one: its index in the calling
/// function's table, with that function when the word names it too
/// (the thread's current function otherwise).
#[inline(always)]
fn decode_method_and_ic(packed: u64) -> (crate::intern::SymbolId, Option<(usize, Option<u32>)>) {
    let method = crate::intern::SymbolId::from_raw(packed as u32);
    let ic_tag = ((packed >> 32) & 0xffff) as u32;
    let func_tag = (packed >> 48) as u32;
    let ic = if ic_tag == 0 {
        None
    } else {
        Some(((ic_tag - 1) as usize, (func_tag != 0).then(|| func_tag - 1)))
    };
    (method, ic)
}

/// A call's packed method word: the symbol, and the site's inline
/// cache index and function when the compile asks for the cache to
/// be filled. Sites past the packing's range go without.
pub fn pack_method_word(method: u32, site: Option<(usize, u32)>) -> u64 {
    let mut bits = method as u64;
    if let Some((i, func_id)) = site
        && i < 0xffff
        && func_id < 0xffff
    {
        bits |= ((i as u64) + 1) << 32;
        bits |= ((func_id as u64) + 1) << 48;
    }
    bits
}

fn with_rooted_args<T>(args: &[Value], f: impl FnOnce(&[Value]) -> T) -> T {
    let root_len_before = jit_roots_snapshot_len();
    for &arg in args {
        push_jit_root(arg);
    }
    debug_assert!(
        args.len() <= 8,
        "compiled runtime helpers only support small fixed arg lists"
    );
    let mut rooted_args = [Value::null(); 8];
    for (idx, slot) in rooted_args.iter_mut().take(args.len()).enumerate() {
        *slot = jit_root_at(root_len_before + idx);
    }
    let result = f(&rooted_args[..args.len()]);
    jit_roots_restore_len(root_len_before);
    result
}

#[inline(always)]
fn cache_key_class(
    vm: &crate::runtime::vm::VM,
    recv: Value,
    class: *mut ObjClass,
) -> *mut ObjClass {
    if class == vm.class_class {
        recv.as_object().unwrap_or(std::ptr::null_mut()) as *mut ObjClass
    } else {
        class
    }
}

#[inline(always)]
fn fast_list_index(value: Value, count: usize) -> Option<usize> {
    let raw_num = value.as_num()?;
    if raw_num != raw_num.trunc() || raw_num.is_infinite() {
        return None;
    }
    let raw = raw_num as i64;
    let index = if raw < 0 { raw + count as i64 } else { raw };
    if index < 0 || index as usize >= count {
        return None;
    }
    Some(index as usize)
}

#[inline(always)]
fn note_list_native_fastpath(vm: &mut crate::runtime::vm::VM) {
    vm.engine
        .note_runtime_call_stats(|s| s.dispatch_call_list_native_fastpath += 1);
}

#[inline(always)]
fn try_dispatch_list_native_fastpath(
    vm: &mut crate::runtime::vm::VM,
    recv: Value,
    method_sym: crate::intern::SymbolId,
    args: &[Value],
) -> Option<u64> {
    let syms = vm.hot_method_symbols;
    let list_ptr = recv.as_object()? as *mut ObjList;
    debug_assert_eq!(
        unsafe { (*(list_ptr as *const ObjHeader)).obj_type },
        ObjType::List
    );

    if Some(method_sym) == syms.list_count && args.len() == 1 {
        note_list_native_fastpath(vm);
        return Some(Value::num(unsafe { (*list_ptr).count } as f64).to_bits());
    }

    if Some(method_sym) == syms.list_add && args.len() == 2 {
        let value = args[1];
        unsafe {
            (*list_ptr).add(value);
        }
        note_list_native_fastpath(vm);
        return Some(args[0].to_bits());
    }

    if Some(method_sym) == syms.list_subscript && args.len() == 2 {
        let count = unsafe { (*list_ptr).count as usize };
        let index = fast_list_index(args[1], count)?;
        note_list_native_fastpath(vm);
        return Some(unsafe { (*list_ptr).get(index).unwrap_or(Value::null()).to_bits() });
    }

    if Some(method_sym) == syms.list_subscript_set && args.len() == 3 {
        let count = unsafe { (*list_ptr).count as usize };
        let index = fast_list_index(args[1], count)?;
        let value = args[2];
        unsafe {
            (*list_ptr).set(index, value);
        }
        note_list_native_fastpath(vm);
        return Some(value.to_bits());
    }

    if Some(method_sym) == syms.list_remove_at && args.len() == 2 {
        let count = unsafe { (*list_ptr).count as usize };
        let index = fast_list_index(args[1], count)?;
        let removed = unsafe { (*list_ptr).remove(index).unwrap_or(Value::null()) };
        note_list_native_fastpath(vm);
        return Some(removed.to_bits());
    }

    if Some(method_sym) == syms.list_iterate && args.len() == 2 {
        let count = unsafe { (*list_ptr).count as usize };
        let iterator = args[1];
        note_list_native_fastpath(vm);
        if iterator.is_null() {
            if count == 0 {
                return Some(Value::bool(false).to_bits());
            }
            return Some(Value::num(0.0).to_bits());
        }
        if !iterator.is_num() {
            return Some(Value::bool(false).to_bits());
        }
        let next = iterator.as_num().unwrap() + 1.0;
        if next >= count as f64 {
            return Some(Value::bool(false).to_bits());
        }
        return Some(Value::num(next).to_bits());
    }

    if Some(method_sym) == syms.list_iterator_value && args.len() == 2 {
        let index = args[1].as_num()? as usize;
        note_list_native_fastpath(vm);
        return Some(unsafe { (*list_ptr).get(index).unwrap_or(Value::null()).to_bits() });
    }

    None
}

#[inline(always)]
fn try_dispatch_trivial_accessor_fastpath(
    vm: &mut crate::runtime::vm::VM,
    method: Method,
    args: &[Value],
) -> Option<u64> {
    let Method::Closure(cp) = method else {
        return None;
    };
    let fn_ref = unsafe { &*(*cp).function };
    if fn_ref.trivial_getter_field != u16::MAX && args.len() == 1 {
        let receiver_obj = args[0].as_object().unwrap_or(std::ptr::null_mut());
        if !receiver_obj.is_null()
            && unsafe { (*(receiver_obj as *const ObjHeader)).obj_type } == ObjType::Instance
        {
            let instance = receiver_obj as *mut ObjInstance;
            let fields = unsafe { (*instance).fields };
            if !fields.is_null() {
                vm.engine.note_runtime_call_stats(|s| {
                    s.dispatch_method_closure += 1;
                    s.dispatch_method_trivial_getter += 1;
                });
                return Some(unsafe {
                    (*fields.add(fn_ref.trivial_getter_field as usize)).to_bits()
                });
            }
        }
    }
    if fn_ref.trivial_setter_field != u16::MAX && args.len() == 2 {
        let receiver_obj = args[0].as_object().unwrap_or(std::ptr::null_mut());
        if !receiver_obj.is_null()
            && unsafe { (*(receiver_obj as *const ObjHeader)).obj_type } == ObjType::Instance
        {
            let instance = receiver_obj as *mut ObjInstance;
            let fields = unsafe { (*instance).fields };
            if !fields.is_null() {
                let value = args[1];
                unsafe {
                    (*instance).set_field_unchecked(fn_ref.trivial_setter_field as usize, value);
                }
                vm.engine.note_runtime_call_stats(|s| {
                    s.dispatch_method_closure += 1;
                    s.dispatch_method_trivial_setter += 1;
                });
                return Some(value.to_bits());
            }
        }
    }

    None
}

/// What the frameless pass over a send found.
enum Frameless {
    Done(u64),
    /// The method cache resolved a host method; it runs under a frame.
    Host(crate::runtime::object::HostFn, usize),
    Miss,
}

/// A trivial accessor answered the site without a frame: record it in
/// the site's cache as the full dispatch would, so a compile of the
/// caller loads or stores the field inline. A site that already holds
/// this class, or has gone polymorphic, is left as it is.
#[inline(always)]
fn note_accessor_fast_path_ic(
    vm: &mut crate::runtime::vm::VM,
    j: *mut JitThread,
    ic_idx: Option<(usize, Option<u32>)>,
    cache_key_class: *mut ObjClass,
    method: Method,
    defining_class: *mut ObjClass,
) {
    let Some((idx, func)) = ic_idx else {
        return;
    };
    let Some(ic_ptr) = current_jit_callsite_ic(vm, j, idx, func) else {
        return;
    };
    let known = unsafe { (*ic_ptr).snapshot() }.is_some_and(|ic| {
        ic.kind & crate::mir::bytecode::IC_POLYMORPHIC != 0
            || (ic.kind != 0 && ic.class == cache_key_class as usize)
    });
    if !known {
        populate_callsite_ic(vm, ic_ptr, cache_key_class, method, defining_class);
    }
}

/// A List fast path answered the site: record the receiver class in
/// the site's cache (kind 9) so a compile of the caller knows it.
#[inline(always)]
fn note_list_fast_path_ic(
    vm: &mut crate::runtime::vm::VM,
    j: *mut JitThread,
    ic_idx: Option<(usize, Option<u32>)>,
) {
    let Some((idx, func)) = ic_idx else {
        return;
    };
    let Some(ic_ptr) = current_jit_callsite_ic(vm, j, idx, func) else {
        return;
    };
    let empty = unsafe { (*ic_ptr).snapshot() }.is_none_or(|ic| ic.kind == 0);
    if empty {
        unsafe {
            (*ic_ptr).store(crate::mir::bytecode::CallSiteIC {
                class: vm.list_class as usize,
                jit_ptr: std::ptr::null(),
                closure: std::ptr::null(),
                func_id: 0,
                kind: 9,
            })
        };
    }
}

#[inline(always)]
fn try_dispatch_call_noframe_fast(
    vm: &mut crate::runtime::vm::VM,
    j: *mut JitThread,
    recv: Value,
    method_packed: u64,
    args: &[Value],
) -> Frameless {
    // Same has_error short-circuit as `dispatch_call_rooted`. The
    // no-frame fast path is the first stop on every JIT-issued
    // Call; without the guard a chained call after a Fiber.abort
    // would still hit a list/method-cache fast path and overwrite
    // the in-flight error with whatever the second call returns.
    if vm.has_error {
        return Frameless::Done(Value::null().to_bits());
    }
    let (method_sym, ic_idx) = decode_method_and_ic(method_packed);
    let class = vm.class_of(recv);

    if class == vm.list_class
        && let Some(result) = try_dispatch_list_native_fastpath(vm, recv, method_sym, args)
    {
        vm.engine.note_runtime_call_stats(|s| {
            s.wren_call_noframe_fastpath += 1;
            s.dispatch_call_entries += 1;
        });
        note_list_fast_path_ic(vm, j, ic_idx);
        return Frameless::Done(result);
    }

    let cache_key_class = cache_key_class(vm, recv, class);
    if let Some((method, defining_class)) = vm.method_cache.lookup(cache_key_class, method_sym) {
        if let Some(result) = try_dispatch_trivial_accessor_fastpath(vm, method, args) {
            note_accessor_fast_path_ic(vm, j, ic_idx, cache_key_class, method, defining_class);
            vm.engine.note_runtime_call_stats(|s| {
                s.wren_call_noframe_fastpath += 1;
                s.dispatch_call_entries += 1;
                s.dispatch_call_method_cache_hits += 1;
            });
            return Frameless::Done(result);
        }
        if let Method::Host(host_fn, context) = method {
            vm.engine.note_runtime_call_stats(|s| {
                s.dispatch_call_entries += 1;
                s.dispatch_call_method_cache_hits += 1;
            });
            return Frameless::Host(host_fn, context);
        }
    }

    Frameless::Miss
}

#[inline(always)]
fn current_jit_callsite_ic(
    vm: &mut crate::runtime::vm::VM,
    j: *mut JitThread,
    ic_idx: usize,
    func: Option<u32>,
) -> Option<*mut crate::mir::bytecode::CallSiteIC> {
    let ctx = unsafe { &(*j).ctx };
    let func_id = if let Some(f) = func {
        crate::runtime::engine::FuncId(f)
    } else if ctx.current_func_id != u32::MAX as u64 {
        crate::runtime::engine::FuncId(ctx.current_func_id as u32)
    } else {
        let closure_ptr = ctx.closure as *mut ObjClosure;
        if closure_ptr.is_null() {
            return None;
        }
        crate::runtime::engine::FuncId(unsafe { (*(*closure_ptr).function).fn_id })
    };
    let bc_ptr = vm
        .engine
        .bc_cache
        .get(func_id.0 as usize)
        .copied()
        .filter(|p| !p.is_null())
        .or_else(|| vm.engine.ensure_bytecode(func_id))?;
    let bc = unsafe { &mut *(bc_ptr as *mut crate::mir::bytecode::BytecodeFunction) };
    let ic_table = unsafe { &mut *bc.ic_table.get() };
    trace_jit_ic(|| {
        format!(
            "jit-ic: func={} ic_idx={} table_len={} closure_null={}",
            func_id.0,
            ic_idx,
            ic_table.len(),
            ctx.closure.is_null()
        )
    });
    ic_table.get_mut(ic_idx).map(|ic| ic as *mut _)
}

/// Unified JIT dispatch. Shadow stores replaced by stack map GC root scanning.
///
/// # Safety
/// `fn_ptr` must point to a valid JIT-compiled function whose ABI matches the
/// number of arguments in `args`. The JIT code must be safe to call from the
/// current execution context.
#[inline(always)]
pub unsafe fn call_jit_with_shadow(
    vm: &crate::runtime::vm::VM,
    fn_ptr: *const u8,
    func_id: crate::runtime::engine::FuncId,
    args: &[Value],
) -> u64 {
    unsafe { call_jit_with_shadow_st(jit_state(), vm, fn_ptr, func_id, args) }
}

/// `call_jit_with_shadow` over an already fetched thread state.
///
/// Swaps in the callee's module context for the duration of the call
/// so its `GetModuleVar` reads its own slots; a callee in the caller's
/// module needs no swap, which is the common case.
#[inline(always)]
pub unsafe fn call_jit_with_shadow_st(
    j: *mut JitThread,
    vm: &crate::runtime::vm::VM,
    fn_ptr: *const u8,
    func_id: crate::runtime::engine::FuncId,
    args: &[Value],
) -> u64 {
    let ctx = unsafe { &mut (*j).ctx };
    let callee_module = vm.engine.func_module(func_id);
    let (mv_ptr, mv_count) = vm.engine.module_vars_for(func_id);
    let same_module = match callee_module {
        Some(_) => mv_ptr.is_null() || mv_ptr == ctx.module_vars,
        None => true,
    };
    if same_module {
        return unsafe { call_jit_cached_st(ctx, fn_ptr, args) };
    }
    // Cross-module call: swap context.
    let mod_name = callee_module.unwrap();
    let saved_ctx = *ctx;
    let bytes = mod_name.as_bytes();
    ctx.module_vars = mv_ptr;
    ctx.module_var_count = mv_count;
    ctx.module_name = bytes.as_ptr();
    ctx.module_name_len = bytes.len() as u32;
    ctx.current_func_id = func_id.0 as u64;
    let result = unsafe { call_jit_cached_st(ctx, fn_ptr, args) };
    *unsafe { &mut (*j).ctx } = saved_ctx;
    result
}

#[inline(always)]
unsafe fn call_jit_cached(fn_ptr: *const u8, args: &[Value]) -> u64 {
    unsafe { call_jit_cached_st(&mut (*jit_state()).ctx, fn_ptr, args) }
}

/// Call compiled code with the context of the JIT state `j`, which the
/// caller fetched once with `jit_state`.
///
/// # Safety
/// As `call_jit_cached_st`: `fn_ptr` takes `args.len()` word arguments.
pub unsafe fn call_jit_at(j: *mut JitThread, fn_ptr: *const u8, args: &[Value]) -> u64 {
    unsafe { call_jit_cached_st(&mut (*j).ctx, fn_ptr, args) }
}

#[inline(always)]
unsafe fn call_jit_cached_st(ctx: *mut JitContext, fn_ptr: *const u8, args: &[Value]) -> u64 {
    let j = jit_state();
    let mut link = EntryLink::new();
    unsafe { enter_link(j, &mut link) };
    let result = unsafe { call_jit_cached_inner(ctx, fn_ptr, args) };
    unsafe { leave_link(j, &link) };
    result
}

#[inline(always)]
unsafe fn call_jit_cached_inner(ctx: *mut JitContext, fn_ptr: *const u8, args: &[Value]) -> u64 {
    unsafe {
        #[cfg(not(target_arch = "aarch64"))]
        let _ = ctx;
        // Ensure x20 holds the JitContext pointer for the JIT code.
        #[cfg(target_arch = "aarch64")]
        {
            let ctx_ptr = ctx as u64;
            core::arch::asm!(
                "mov x20, {ctx}",
                ctx = in(reg) ctx_ptr,
                lateout("x20") _,
                options(nostack, nomem),
            );
        }
        match args.len() {
            0 => {
                let f: extern "C" fn() -> u64 = std::mem::transmute(fn_ptr);
                f()
            }
            1 => {
                let f: extern "C" fn(u64) -> u64 = std::mem::transmute(fn_ptr);
                f(args[0].to_bits())
            }
            2 => {
                let f: extern "C" fn(u64, u64) -> u64 = std::mem::transmute(fn_ptr);
                f(args[0].to_bits(), args[1].to_bits())
            }
            3 => {
                let f: extern "C" fn(u64, u64, u64) -> u64 = std::mem::transmute(fn_ptr);
                f(args[0].to_bits(), args[1].to_bits(), args[2].to_bits())
            }
            4 => {
                let f: extern "C" fn(u64, u64, u64, u64) -> u64 = std::mem::transmute(fn_ptr);
                f(
                    args[0].to_bits(),
                    args[1].to_bits(),
                    args[2].to_bits(),
                    args[3].to_bits(),
                )
            }
            5 => {
                let f: extern "C" fn(u64, u64, u64, u64, u64) -> u64 = std::mem::transmute(fn_ptr);
                f(
                    args[0].to_bits(),
                    args[1].to_bits(),
                    args[2].to_bits(),
                    args[3].to_bits(),
                    args[4].to_bits(),
                )
            }
            6 => {
                let f: extern "C" fn(u64, u64, u64, u64, u64, u64) -> u64 =
                    std::mem::transmute(fn_ptr);
                f(
                    args[0].to_bits(),
                    args[1].to_bits(),
                    args[2].to_bits(),
                    args[3].to_bits(),
                    args[4].to_bits(),
                    args[5].to_bits(),
                )
            }
            7 => {
                let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64) -> u64 =
                    std::mem::transmute(fn_ptr);
                f(
                    args[0].to_bits(),
                    args[1].to_bits(),
                    args[2].to_bits(),
                    args[3].to_bits(),
                    args[4].to_bits(),
                    args[5].to_bits(),
                    args[6].to_bits(),
                )
            }
            // 8+ args. AOT-emitted method bodies declare their
            // physical arity to match the call site. Each explicit
            // arm below transmutes to an extern "C" fn with the
            // matching signature so cranelift's calling-convention
            // register-passing lines up — a `_` arm that drops to a
            // smaller signature silently truncates higher arg slots
            // and leaves them reading garbage in the JIT'd frame
            // (the previous version capped at 8, so Renderer2D's
            // `drawSprite_(texture, x, y, w, h, u0, v0, u1, v1, r, g,
            // b, a)` saw garbage for `v1` through `a`, surfacing as
            // `Float32Array[_]=: value must be a number` across
            // every sprite-batch flush).
            8 => {
                let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64) -> u64 =
                    std::mem::transmute(fn_ptr);
                f(
                    args[0].to_bits(),
                    args[1].to_bits(),
                    args[2].to_bits(),
                    args[3].to_bits(),
                    args[4].to_bits(),
                    args[5].to_bits(),
                    args[6].to_bits(),
                    args[7].to_bits(),
                )
            }
            9 => {
                let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64) -> u64 =
                    std::mem::transmute(fn_ptr);
                f(
                    args[0].to_bits(),
                    args[1].to_bits(),
                    args[2].to_bits(),
                    args[3].to_bits(),
                    args[4].to_bits(),
                    args[5].to_bits(),
                    args[6].to_bits(),
                    args[7].to_bits(),
                    args[8].to_bits(),
                )
            }
            10 => {
                let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64) -> u64 =
                    std::mem::transmute(fn_ptr);
                f(
                    args[0].to_bits(),
                    args[1].to_bits(),
                    args[2].to_bits(),
                    args[3].to_bits(),
                    args[4].to_bits(),
                    args[5].to_bits(),
                    args[6].to_bits(),
                    args[7].to_bits(),
                    args[8].to_bits(),
                    args[9].to_bits(),
                )
            }
            11 => {
                let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64) -> u64 =
                    std::mem::transmute(fn_ptr);
                f(
                    args[0].to_bits(),
                    args[1].to_bits(),
                    args[2].to_bits(),
                    args[3].to_bits(),
                    args[4].to_bits(),
                    args[5].to_bits(),
                    args[6].to_bits(),
                    args[7].to_bits(),
                    args[8].to_bits(),
                    args[9].to_bits(),
                    args[10].to_bits(),
                )
            }
            12 => {
                let f: extern "C" fn(
                    u64,
                    u64,
                    u64,
                    u64,
                    u64,
                    u64,
                    u64,
                    u64,
                    u64,
                    u64,
                    u64,
                    u64,
                ) -> u64 = std::mem::transmute(fn_ptr);
                f(
                    args[0].to_bits(),
                    args[1].to_bits(),
                    args[2].to_bits(),
                    args[3].to_bits(),
                    args[4].to_bits(),
                    args[5].to_bits(),
                    args[6].to_bits(),
                    args[7].to_bits(),
                    args[8].to_bits(),
                    args[9].to_bits(),
                    args[10].to_bits(),
                    args[11].to_bits(),
                )
            }
            13 => {
                let f: extern "C" fn(
                    u64,
                    u64,
                    u64,
                    u64,
                    u64,
                    u64,
                    u64,
                    u64,
                    u64,
                    u64,
                    u64,
                    u64,
                    u64,
                ) -> u64 = std::mem::transmute(fn_ptr);
                f(
                    args[0].to_bits(),
                    args[1].to_bits(),
                    args[2].to_bits(),
                    args[3].to_bits(),
                    args[4].to_bits(),
                    args[5].to_bits(),
                    args[6].to_bits(),
                    args[7].to_bits(),
                    args[8].to_bits(),
                    args[9].to_bits(),
                    args[10].to_bits(),
                    args[11].to_bits(),
                    args[12].to_bits(),
                )
            }
            14 => {
                let f: extern "C" fn(
                    u64,
                    u64,
                    u64,
                    u64,
                    u64,
                    u64,
                    u64,
                    u64,
                    u64,
                    u64,
                    u64,
                    u64,
                    u64,
                    u64,
                ) -> u64 = std::mem::transmute(fn_ptr);
                f(
                    args[0].to_bits(),
                    args[1].to_bits(),
                    args[2].to_bits(),
                    args[3].to_bits(),
                    args[4].to_bits(),
                    args[5].to_bits(),
                    args[6].to_bits(),
                    args[7].to_bits(),
                    args[8].to_bits(),
                    args[9].to_bits(),
                    args[10].to_bits(),
                    args[11].to_bits(),
                    args[12].to_bits(),
                    args[13].to_bits(),
                )
            }
            15 => {
                let f: extern "C" fn(
                    u64,
                    u64,
                    u64,
                    u64,
                    u64,
                    u64,
                    u64,
                    u64,
                    u64,
                    u64,
                    u64,
                    u64,
                    u64,
                    u64,
                    u64,
                ) -> u64 = std::mem::transmute(fn_ptr);
                f(
                    args[0].to_bits(),
                    args[1].to_bits(),
                    args[2].to_bits(),
                    args[3].to_bits(),
                    args[4].to_bits(),
                    args[5].to_bits(),
                    args[6].to_bits(),
                    args[7].to_bits(),
                    args[8].to_bits(),
                    args[9].to_bits(),
                    args[10].to_bits(),
                    args[11].to_bits(),
                    args[12].to_bits(),
                    args[13].to_bits(),
                    args[14].to_bits(),
                )
            }
            16 => {
                let f: extern "C" fn(
                    u64,
                    u64,
                    u64,
                    u64,
                    u64,
                    u64,
                    u64,
                    u64,
                    u64,
                    u64,
                    u64,
                    u64,
                    u64,
                    u64,
                    u64,
                    u64,
                ) -> u64 = std::mem::transmute(fn_ptr);
                f(
                    args[0].to_bits(),
                    args[1].to_bits(),
                    args[2].to_bits(),
                    args[3].to_bits(),
                    args[4].to_bits(),
                    args[5].to_bits(),
                    args[6].to_bits(),
                    args[7].to_bits(),
                    args[8].to_bits(),
                    args[9].to_bits(),
                    args[10].to_bits(),
                    args[11].to_bits(),
                    args[12].to_bits(),
                    args[13].to_bits(),
                    args[14].to_bits(),
                    args[15].to_bits(),
                )
            }
            n => panic!(
                "call_jit_cached: arity {} not in explicit table; extend the match arms up to {} \
             (truncating to a smaller signature silently drops higher args)",
                n, n
            ),
        }
    }
}

#[inline(always)]
fn trivial_getter_field_index(
    vm: &crate::runtime::vm::VM,
    closure_ptr: *mut ObjClosure,
) -> Option<u16> {
    let func_id = crate::runtime::engine::FuncId(unsafe { (*(*closure_ptr).function).fn_id });
    if let Some(cached) = vm
        .engine
        .trivial_getter_fields
        .get(func_id.0 as usize)
        .copied()
    {
        return cached;
    }

    // Fallback for functions registered before the cache was populated.
    use crate::mir::{Instruction, Terminator};

    let mir = vm.engine.get_mir(func_id)?;
    if mir.blocks.len() != 1 {
        return None;
    }
    let block = &mir.blocks[0];
    let mut self_param = None;
    let mut getter = None;
    for (vid, inst) in &block.instructions {
        match inst {
            Instruction::BlockParam(0) if self_param.is_none() => self_param = Some(*vid),
            Instruction::GetField(recv, idx) if getter.is_none() && Some(*recv) == self_param => {
                getter = Some((*vid, *idx));
            }
            _ => return None,
        }
    }
    match (getter, &block.terminator) {
        (Some((ret_vid, field_idx)), Terminator::Return(v)) if *v == ret_vid => Some(field_idx),
        _ => None,
    }
}

#[inline(always)]
fn populate_callsite_ic(
    vm: &crate::runtime::vm::VM,
    ic_ptr: *mut crate::mir::bytecode::CallSiteIC,
    cache_key_class: *mut ObjClass,
    method: Method,
    defining_class: *mut ObjClass,
) {
    let entry = match method {
        Method::Closure(closure_ptr) => {
            let fn_idx = unsafe { (*(*closure_ptr).function).fn_id } as usize;
            let jit_ptr = vm
                .engine
                .jit_code
                .get(fn_idx)
                .copied()
                .unwrap_or(std::ptr::null());
            if let Some(field_idx) = trivial_getter_field_index(vm, closure_ptr) {
                crate::mir::bytecode::CallSiteIC {
                    class: cache_key_class as usize,
                    jit_ptr: std::ptr::null(),
                    closure: closure_ptr as *const u8,
                    func_id: field_idx as u64,
                    kind: 5,
                }
            } else if !jit_ptr.is_null()
                && !jit_disabled()
                && ic_jit_kind1_enabled()
                && vm.engine.jit_leaf.get(fn_idx).copied().unwrap_or(false)
            {
                crate::mir::bytecode::CallSiteIC {
                    class: cache_key_class as usize,
                    jit_ptr,
                    closure: closure_ptr as *const u8,
                    func_id: fn_idx as u64,
                    kind: 1,
                }
            } else if !jit_ptr.is_null() && !jit_disabled() {
                // Kind=6: non-leaf direct JIT dispatch (bypasses call_closure_jit_or_sync)
                crate::mir::bytecode::CallSiteIC {
                    class: cache_key_class as usize,
                    jit_ptr,
                    closure: closure_ptr as *const u8,
                    func_id: fn_idx as u64,
                    kind: 6,
                }
            } else {
                crate::mir::bytecode::CallSiteIC {
                    class: cache_key_class as usize,
                    jit_ptr: defining_class as *const u8,
                    closure: closure_ptr as *const u8,
                    func_id: fn_idx as u64,
                    kind: 2,
                }
            }
        }
        Method::Constructor(closure_ptr) => {
            let fn_idx = unsafe { (*(*closure_ptr).function).fn_id } as usize;
            let ctor_jit_ptr = vm
                .engine
                .jit_code
                .get(fn_idx)
                .copied()
                .unwrap_or(std::ptr::null());
            crate::mir::bytecode::CallSiteIC {
                class: cache_key_class as usize,
                jit_ptr: ctor_jit_ptr,
                closure: closure_ptr as *const u8,
                func_id: fn_idx as u64,
                kind: 3,
            }
        }
        Method::Native(native_fn) => crate::mir::bytecode::CallSiteIC {
            class: cache_key_class as usize,
            jit_ptr: std::ptr::null(),
            closure: native_fn as *const () as *const u8,
            func_id: 0,
            kind: 4,
        },
        // ForeignC and ForeignCDynamic methods aren't cached in the
        // IC — each call falls back to full method-table lookup.
        // For ForeignC the dispatch is already dominated by the
        // dlsym'd plugin call; for ForeignCDynamic it's dominated
        // by the JS-bridge round-trip. Re-resolution overhead is
        // negligible against either.
        Method::ForeignC(_) | Method::ForeignCDynamic(_) => {
            crate::mir::bytecode::CallSiteIC::default()
        }
        Method::Host(host_fn, context) => crate::mir::bytecode::CallSiteIC {
            class: cache_key_class as usize,
            jit_ptr: std::ptr::null(),
            closure: host_fn as *const () as *const u8,
            func_id: context as u64,
            kind: 8,
        },
    };

    unsafe { (*ic_ptr).store_seen(entry) };
    trace_jit_ic(|| {
        format!(
            "jit-ic: populate kind={} ic_ptr=0x{:x}",
            entry.kind, ic_ptr as usize
        )
    });
}

#[inline(always)]
fn maybe_upgrade_closure_ic_to_leaf(
    vm: &mut crate::runtime::vm::VM,
    ic_ptr: *mut crate::mir::bytecode::CallSiteIC,
    cache_key_class: *mut ObjClass,
    closure_ptr: *mut ObjClosure,
    args_len: usize,
) -> bool {
    if args_len > 4
        || jit_disabled()
        || vm.engine.mode == crate::runtime::engine::ExecutionMode::Interpreter
    {
        return false;
    }

    let func_id = crate::runtime::engine::FuncId(unsafe { (*(*closure_ptr).function).fn_id });
    let fn_idx = func_id.0 as usize;

    let mut jit_ptr = vm
        .engine
        .jit_code
        .get(fn_idx)
        .copied()
        .unwrap_or(std::ptr::null());
    if jit_ptr.is_null() {
        let should_tier_up = vm.engine.record_call(func_id);
        if should_tier_up {
            vm.request_tier_up(func_id);
        }
        if vm.engine.has_pending_compilations() {
            vm.engine.poll_compilations();
        }
        jit_ptr = vm
            .engine
            .jit_code
            .get(fn_idx)
            .copied()
            .unwrap_or(std::ptr::null());
    }

    if jit_ptr.is_null()
        || !ic_jit_kind1_enabled()
        || !vm.engine.jit_leaf.get(fn_idx).copied().unwrap_or(false)
    {
        return false;
    }

    unsafe {
        (*ic_ptr).store_seen(crate::mir::bytecode::CallSiteIC {
            class: cache_key_class as usize,
            jit_ptr,
            closure: closure_ptr as *const u8,
            func_id: func_id.0 as u64,
            kind: 1,
        });
    }
    trace_jit_ic(|| format!("jit-ic: upgrade kind=1 func={}", func_id.0));
    true
}

#[inline(always)]
fn maybe_request_next_tier(
    vm: &mut crate::runtime::vm::VM,
    func_id: crate::runtime::engine::FuncId,
) {
    if vm.engine.mode != crate::runtime::engine::ExecutionMode::Tiered {
        return;
    }
    let should_tier_up = vm.engine.record_call(func_id);
    if should_tier_up {
        vm.request_tier_up(func_id);
    }
    // NOTE: poll_compilations is NOT called here. Installing a new compiled
    // version during an IC hit path would invalidate the IC entry we just
    // matched (jit_ptr changes → IC cleared). poll_compilations is called
    // at safepoints in dispatch_closure_bc and call_closure_jit_or_sync.
}

#[inline(always)]
fn try_dispatch_callsite_ic(
    vm: &mut crate::runtime::vm::VM,
    ic_ptr: *mut crate::mir::bytecode::CallSiteIC,
    recv: Value,
    args: &[Value],
    cache_key_class: *mut ObjClass,
) -> Option<u64> {
    let ic = unsafe { (*ic_ptr).snapshot()? };
    if ic.class != cache_key_class as usize || ic.class == 0 {
        vm.engine
            .note_runtime_call_stats(|s| s.dispatch_call_ic_class_misses += 1);
        if ic.func_id != 0 {
            vm.engine
                .note_ic_miss(crate::runtime::engine::FuncId(ic.func_id as u32));
        }
        return None;
    }

    match ic.kind {
        1 => {
            if args.len() > 4 {
                vm.engine
                    .note_runtime_call_stats(|s| s.ic_invalidations += 1);
                unsafe { (*ic_ptr).clear() };
                return None;
            }
            let fn_idx = ic.func_id as usize;
            let func_id = crate::runtime::engine::FuncId(ic.func_id as u32);
            maybe_request_next_tier(vm, func_id);
            let live_ptr = vm
                .engine
                .jit_code
                .get(fn_idx)
                .copied()
                .unwrap_or(std::ptr::null());
            if live_ptr.is_null()
                || live_ptr != ic.jit_ptr
                || !vm.engine.jit_leaf.get(fn_idx).copied().unwrap_or(false)
                || jit_disabled()
            {
                vm.engine
                    .note_runtime_call_stats(|s| s.ic_invalidations += 1);
                unsafe { (*ic_ptr).clear() };
                return None;
            }
            vm.engine.note_ic_hit(func_id);
            vm.engine.note_runtime_call_stats(|s| s.ic_kind1_hits += 1);
            trace_jit_ic(|| format!("jit-ic: hit kind=1 func={}", ic.func_id));
            Some(unsafe { call_jit_with_shadow(vm, live_ptr, func_id, args) })
        }
        2 => {
            let closure_ptr = ic.closure as *mut ObjClosure;
            if closure_ptr.is_null() {
                vm.engine
                    .note_runtime_call_stats(|s| s.ic_invalidations += 1);
                unsafe { (*ic_ptr).clear() };
                return None;
            }
            let func_id = crate::runtime::engine::FuncId(ic.func_id as u32);
            maybe_request_next_tier(vm, func_id);
            if maybe_upgrade_closure_ic_to_leaf(
                vm,
                ic_ptr,
                cache_key_class,
                closure_ptr,
                args.len(),
            ) {
                let up = unsafe { (*ic_ptr).snapshot() }?;
                let live_ptr = up.jit_ptr;
                let fid = crate::runtime::engine::FuncId(up.func_id as u32);
                vm.engine.note_ic_hit(fid);
                vm.engine.note_runtime_call_stats(|s| s.ic_kind1_hits += 1);
                trace_jit_ic(|| format!("jit-ic: hit upgraded kind=1 func={}", up.func_id));
                return Some(unsafe { call_jit_with_shadow(vm, live_ptr, fid, args) });
            }
            vm.engine.note_ic_hit(func_id);
            vm.engine.note_runtime_call_stats(|s| s.ic_kind2_hits += 1);
            trace_jit_ic(|| format!("jit-ic: hit kind=2 func={}", ic.func_id));
            let defining_class = ic.jit_ptr as *mut ObjClass;
            Some(call_closure_jit_or_sync(
                vm,
                closure_ptr,
                args,
                if defining_class.is_null() {
                    None
                } else {
                    Some(defining_class)
                },
            ))
        }
        3 => {
            let closure_ptr = ic.closure as *mut ObjClosure;
            let class_ptr = recv.as_object().unwrap_or(std::ptr::null_mut()) as *mut ObjClass;
            if closure_ptr.is_null() || class_ptr.is_null() || args.is_empty() {
                vm.engine
                    .note_runtime_call_stats(|s| s.ic_invalidations += 1);
                unsafe { (*ic_ptr).clear() };
                return Some(Value::null().to_bits());
            }
            let func_id = crate::runtime::engine::FuncId(ic.func_id as u32);
            maybe_request_next_tier(vm, func_id);
            if vm.engine.has_pending_compilations() {
                vm.engine.poll_compilations();
            }
            vm.engine.note_ic_hit(func_id);
            vm.engine.note_runtime_call_stats(|s| s.ic_kind3_hits += 1);
            trace_jit_ic(|| format!("jit-ic: hit kind=3 func={}", ic.func_id));
            Some(
                vm.call_constructor_sync(class_ptr, closure_ptr, &args[1..])
                    .to_bits(),
            )
        }
        4 => {
            vm.engine.note_runtime_call_stats(|s| s.ic_kind4_hits += 1);
            trace_jit_ic(|| "jit-ic: hit kind=4".to_string());
            let native_fn: crate::runtime::object::NativeFn =
                unsafe { std::mem::transmute(ic.closure) };
            Some(native_fn(vm, args).to_bits())
        }
        8 => {
            vm.engine.note_runtime_call_stats(|s| s.ic_kind8_hits += 1);
            trace_jit_ic(|| "jit-ic: hit kind=8".to_string());
            let host_fn: crate::runtime::object::HostFn =
                unsafe { std::mem::transmute(ic.closure) };
            let context = ic.func_id as usize;
            let result = host_fn(vm, context, args).to_bits();
            if let Some(action) = vm.pending_fiber_action.take() {
                return Some(handle_jit_fiber_action(vm, action));
            }
            Some(result)
        }
        5 => {
            let instance = recv
                .as_object()
                .map(|p| p as *mut ObjInstance)
                .unwrap_or(std::ptr::null_mut());
            if instance.is_null() {
                vm.engine
                    .note_runtime_call_stats(|s| s.ic_invalidations += 1);
                unsafe { (*ic_ptr).clear() };
                return None;
            }
            if !ic.closure.is_null() {
                let func_id = crate::runtime::engine::FuncId(unsafe {
                    (*(*(ic.closure as *mut ObjClosure)).function).fn_id
                });
                vm.engine.note_ic_hit(func_id);
            }
            vm.engine.note_runtime_call_stats(|s| s.ic_kind5_hits += 1);
            trace_jit_ic(|| format!("jit-ic: hit kind=5 field={}", ic.func_id));
            let fields_ptr = unsafe { (*instance).fields };
            if fields_ptr.is_null() {
                vm.engine
                    .note_runtime_call_stats(|s| s.ic_invalidations += 1);
                unsafe { (*ic_ptr).clear() };
                return None;
            }
            Some(unsafe { (*fields_ptr.add(ic.func_id as usize)).to_bits() })
        }
        6 => {
            // Kind=6: non-leaf direct JIT dispatch — bypasses call_closure_jit_or_sync.
            let depth = jit_depth();
            if args.len() > 4 || depth >= 32 {
                return None; // fall through to slow path, keep IC valid
            }
            let fn_idx = ic.func_id as usize;
            let func_id = crate::runtime::engine::FuncId(ic.func_id as u32);
            maybe_request_next_tier(vm, func_id);
            let live_ptr = vm
                .engine
                .jit_code
                .get(fn_idx)
                .copied()
                .unwrap_or(std::ptr::null());
            if live_ptr.is_null() || live_ptr != ic.jit_ptr || jit_disabled() {
                vm.engine
                    .note_runtime_call_stats(|s| s.ic_invalidations += 1);
                unsafe { (*ic_ptr).clear() };
                return None;
            }
            vm.engine.note_ic_hit(func_id);
            vm.engine.note_runtime_call_stats(|s| s.ic_kind6_hits += 1);
            // Direct field access via UnsafeCell — no 48-byte copy.
            let saved_func_id = unsafe {
                let ctx = &mut (*jit_state()).ctx;
                let old = ctx.current_func_id;
                ctx.current_func_id = fn_idx as u64;
                ctx.closure = ic.closure as *mut u8;
                old
            };
            unsafe { (*jit_state()).depth = depth + 1 };
            let result = unsafe { call_jit_cached(live_ptr, args) };
            unsafe { (*jit_state()).depth = depth };
            unsafe {
                (*jit_state()).ctx.current_func_id = saved_func_id;
            }
            Some(result)
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// JIT context — thread-local VM state for runtime functions
// ---------------------------------------------------------------------------

/// Context passed to JIT-compiled code via thread-local storage.
/// Set by the interpreter before calling native code, read by runtime functions.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct JitContext {
    /// Pointer to module variable storage (`Vec<Value>` data pointer).
    pub module_vars: *mut u64,
    /// Number of module variables.
    pub module_var_count: u32,
    /// Raw pointer to the VM (for method dispatch, allocation, etc.).
    pub vm: *mut u8,
    /// Module name pointer (C string, for engine lookups).
    pub module_name: *const u8,
    /// Module name length.
    pub module_name_len: u32,
    /// Currently executing function id, used to find per-function call-site ICs.
    /// Widened to u64 for clean 8-byte aligned access from JIT code via x19.
    pub current_func_id: u64,
    /// Current closure pointer (for upvalue access from JIT-compiled closure bodies).
    pub closure: *mut u8,
    /// The class that defines the current method (for static field access).
    pub defining_class: *mut u8,
    /// Base pointer to engine.jit_code array (Vec<*const u8> data pointer).
    /// Used by CallKnownFunc to load callee JIT pointers at runtime.
    pub jit_code_base: *const *const u8,
    /// Number of entries in jit_code array.
    pub jit_code_len: u32,
}

unsafe impl Send for JitContext {}

impl Default for JitContext {
    fn default() -> Self {
        Self {
            module_vars: std::ptr::null_mut(),
            module_var_count: 0,
            vm: std::ptr::null_mut(),
            module_name: std::ptr::null(),
            module_name_len: 0,
            current_func_id: u32::MAX as u64,
            closure: std::ptr::null_mut(),
            defining_class: std::ptr::null_mut(),
            jit_code_base: std::ptr::null(),
            jit_code_len: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// JIT context and root set — using Cell/UnsafeCell for zero-overhead access.
// thread_local! ensures test parallelism safety; Cell/UnsafeCell eliminates
// RefCell borrow-checking overhead on the hot path.
// ---------------------------------------------------------------------------

/// Everything a JIT dispatch helper touches per call, behind one
/// thread-local: the context the compiled code reads, the root set and
/// the native recursion depth. A helper fetches the address once with
/// `jit_state` and works through it, because each thread-local access is
/// an out-of-line lookup on macOS.
pub struct JitThread {
    pub ctx: JitContext,
    pub roots: Vec<Value>,
    /// Compiled frames the collector walks: (fp, func_id, return address).
    pub frames: Vec<(usize, u32, usize)>,
    pub depth: u32,
    /// All JIT dispatch off, for a shadow check against the interpreter.
    pub disabled: bool,
    /// The frame a loop entry stub posted for the body it is about to
    /// call; 0 between stubs.
    pub osr_frame: u64,
    /// Where compiled code is, for a trace (see [`EntryLink`]): the
    /// frame pointer and site key of the compiled frame that last
    /// called a helper that can raise, and the innermost link of the
    /// entries into compiled code. Compiled code stores the pair
    /// through one address ([`jit_cur_cell`]), so they sit together.
    pub cur: CurFrame,
    pub entry_top: u64,
}

/// The compiled frame a raise would be inside: its frame pointer and
/// key (function id in the low word, site in the high word). Written
/// by compiled code before each call into the runtime that can raise
/// or run code, read by a trace, 0 when no compiled frame has.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CurFrame {
    pub fp: u64,
    pub key: u64,
}

/// What a trace needs of compiled frames, and what it costs them:
/// nothing on entry, return or a direct call. Compiled bodies keep
/// frame pointers, so from a frame every caller's frame and return
/// address follow, and a return address inside compiled code names its
/// site through the code's own table. Only where compiled code calls
/// the runtime is that chain unreadable (runtime frames keep no frame
/// pointer), so such a call stores its frame and site first
/// ([`CurFrame`]), and each entry from the runtime into compiled code
/// links the pair it displaces here, on the entering frame's stack, for
/// the walk to continue below the entry. The interpreter notes the
/// innermost link in each frame it pushes, so a trace can interleave
/// the two stacks.
#[repr(C)]
pub struct EntryLink {
    pub prev: u64,
    pub fp: u64,
    pub key: u64,
}

/// Bit of a key's site word for a frame the trace leaves out: the
/// activation went on elsewhere, in a higher tier's loop entry or in
/// the interpreter after a deopt, and that frame stands for it.
pub const KEY_SHADOWED: u64 = 1 << 63;

/// Bit of a key's site word for a frame handed out as a record rather
/// than a frame pointer (see `llvm_backend::frame_records`).
pub const KEY_RECORD: u64 = 1 << 62;

impl EntryLink {
    pub const fn new() -> Self {
        EntryLink {
            prev: 0,
            fp: 0,
            key: 0,
        }
    }
}

impl Default for EntryLink {
    fn default() -> Self {
        Self::new()
    }
}

/// Entering compiled code from the runtime: the link takes the current
/// frame pair, and a fresh segment starts.
///
/// # Safety
/// `j` is this thread's state; `link` outlives the call and is passed
/// to [`leave_link`] after it.
#[inline(always)]
pub unsafe fn enter_link(j: *mut JitThread, link: &mut EntryLink) {
    unsafe {
        let j = &mut *j;
        link.prev = j.entry_top;
        link.fp = j.cur.fp;
        link.key = j.cur.key;
        j.entry_top = link as *mut EntryLink as u64;
        j.cur.fp = 0;
    }
}

/// Back from compiled code: the pair the link held is current again.
///
/// # Safety
/// As [`enter_link`], with the same link.
#[inline(always)]
pub unsafe fn leave_link(j: *mut JitThread, link: &EntryLink) {
    unsafe {
        let j = &mut *j;
        j.cur.fp = link.fp;
        j.cur.key = link.key;
        j.entry_top = link.prev;
    }
}

/// The innermost entry link, for an interpreter frame to note.
#[inline(always)]
pub fn entry_top() -> u64 {
    unsafe { (*jit_state()).entry_top }
}

/// The frame pair and entry link of this thread, for a fiber switch to
/// carry with the stack they describe.
pub fn native_frames_state() -> (CurFrame, u64) {
    let j = unsafe { &*jit_state() };
    (j.cur, j.entry_top)
}

pub fn set_native_frames_state(state: (CurFrame, u64)) {
    let j = unsafe { &mut *jit_state() };
    j.cur = state.0;
    j.entry_top = state.1;
}

/// Address of this thread's current frame pair, for a body compiled on
/// any thread to run on this one.
pub fn jit_cur_cell() -> usize {
    unsafe { std::ptr::addr_of_mut!((*jit_state()).cur) as usize }
}

/// The current compiled frame's activation goes on in the interpreter
/// (a deopt): a trace leaves the frame out and continues from it.
pub fn shadow_current() {
    unsafe { (*jit_state()).cur.key |= KEY_SHADOWED };
}

/// The activation the innermost entry displaced goes on in the body
/// entered (a loop entry into a higher tier): a trace leaves its frame
/// out and continues from it.
pub fn shadow_entry() {
    let j = unsafe { &mut *jit_state() };
    if j.entry_top != 0 {
        let link = unsafe { &mut *(j.entry_top as *mut EntryLink) };
        link.key |= KEY_SHADOWED;
    }
}

thread_local! {
    static JIT: std::cell::UnsafeCell<JitThread> = const { std::cell::UnsafeCell::new(JitThread {
        ctx: JitContext {
            module_vars: std::ptr::null_mut(),
            module_var_count: 0,
            vm: std::ptr::null_mut(),
            module_name: std::ptr::null(),
            module_name_len: 0,
            current_func_id: u32::MAX as u64,
            closure: std::ptr::null_mut(),
            defining_class: std::ptr::null_mut(),
            jit_code_base: std::ptr::null(),
            jit_code_len: 0,
        },
        roots: Vec::new(),
        frames: Vec::new(),
        depth: 0,
        disabled: false,
        osr_frame: 0,
        cur: CurFrame { fp: 0, key: 0 },
        entry_top: 0,
    }) };

    /// Flat shadow root stack — zero-alloc push/pop after warmup.
    static FLAT_SHADOW: std::cell::UnsafeCell<FlatShadowStack> =
        const { std::cell::UnsafeCell::new(FlatShadowStack {
            roots: Vec::new(),
            boundaries: Vec::new(),
        }) };

}

/// This thread's JIT state. Valid for the thread's lifetime; only this
/// thread may touch it.
#[inline(always)]
pub fn jit_state() -> *mut JitThread {
    JIT.with(|j| j.get())
}

/// A loop entry stub about to call its body: `frame` points at the
/// live-ins, with the entry index in the word before them. The frame
/// is kept for this thread; `pending` is the body's count of posted
/// frames, which its entry checks before looking for one.
///
/// # Safety
/// `pending` is the body's request word; the stub calls the body next
/// on this thread.
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub unsafe extern "C" fn wren_osr_post(pending: *mut u64, frame: u64) -> u64 {
    unsafe { (*jit_state()).osr_frame = frame };
    unsafe { std::sync::atomic::AtomicU64::from_ptr(pending) }
        .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    0
}

/// A body's entry taking the frame a stub posted for this thread, or 0
/// when another thread's stub raised the count.
///
/// # Safety
/// `pending` is the body's request word.
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub unsafe extern "C" fn wren_osr_take(pending: *mut u64) -> u64 {
    let frame = unsafe { std::mem::replace(&mut (*jit_state()).osr_frame, 0) };
    if frame != 0 {
        unsafe { std::sync::atomic::AtomicU64::from_ptr(pending) }
            .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    }
    frame
}

/// A loop the top tier compiled before it ran has now run a few
/// hundred iterations through generic calls, filling their caches:
/// compile the function again from them.
#[cfg(feature = "host")]
#[cfg_attr(not(target_arch = "wasm32"), unsafe(no_mangle))]
pub extern "C" fn wren_cold_loop_hot(func_id: u64) -> u64 {
    if let Some(vm) = unsafe { vm_ref() } {
        vm.cold_loop_hot(crate::runtime::engine::FuncId(func_id as u32));
    }
    0
}

/// What a baseline body's re-tier poll gets back when the top tier
/// has no entry for it yet: the loop goes on in the baseline. Any
/// other value, the internal undefined sentinel included, is the
/// body's result to return.
pub const RETIER_DECLINED: u64 = crate::runtime::value::TAG_UNDEFINED ^ (1 << 32);

/// Baseline code whose tier countdown reached zero: proposes the top
/// tier when the engine's policy says so and reloads the countdown.
#[cfg(feature = "host")]
#[cfg_attr(not(target_arch = "wasm32"), unsafe(no_mangle))]
pub extern "C" fn wren_tier_tick(func_id: u64) -> u64 {
    let vm = read_jit_ctx().vm as *mut crate::runtime::vm::VM;
    if vm.is_null() {
        return 0;
    }
    // SAFETY: the context's vm pointer is the running VM; compiled code
    // only runs while it is alive.
    let vm = unsafe { &mut *vm };
    vm.native_tick(crate::runtime::engine::FuncId(func_id as u32));
    0
}

/// Baseline code at a loop header whose function has top-tier code:
/// `buf` holds `n` (register, value) pairs of the header's live-ins.
/// Runs the top tier's OSR entry for that header to completion and
/// returns its result, or the internal undefined sentinel when no
/// entry takes these values, in which case the poll is switched off
/// for the function.
///
/// # Safety
/// `buf` must point at `2 * n` readable u64s; compiled code passes its
/// own stack buffer.
#[cfg(feature = "host")]
#[cfg_attr(not(target_arch = "wasm32"), unsafe(no_mangle))]
pub unsafe extern "C" fn wren_retier(func_id: u64, header: u64, buf: *const u64, n: u64) -> u64 {
    let decline = RETIER_DECLINED;
    let vm = read_jit_ctx().vm as *mut crate::runtime::vm::VM;
    if vm.is_null() {
        return decline;
    }
    let func_id = func_id as u32 as u64;
    let vm = unsafe { &mut *vm };
    let id = crate::runtime::engine::FuncId(func_id as u32);
    // The caller's generation rides above the header id.
    let caller_gen = (header >> 32) as u32;
    let header = header as u32;
    let Some(entry) = vm
        .engine
        .top_tier_osr_entry(id, crate::mir::BlockId(header), caller_gen)
    else {
        vm.retier_declined(id, crate::mir::BlockId(header), caller_gen);
        return decline;
    };
    let pairs: Vec<(u32, Value)> = (0..n as usize)
        .map(|i| unsafe {
            (
                *buf.add(2 * i) as u32,
                Value::from_bits(*buf.add(2 * i + 1)),
            )
        })
        .collect();
    let integral = |v: Value| {
        v.as_num()
            .map(|n| n == n.trunc() && n.abs() <= 9007199254740992.0)
            .unwrap_or(false)
    };
    // The word before the live-ins is the entry stub's.
    let mut args: Vec<Value> = Vec::with_capacity(entry.live_in_regs.len() + 1);
    args.push(Value::null());
    for (i, reg) in entry.live_in_regs.iter().enumerate() {
        let needs_field = entry.live_in_field.get(i).copied().flatten().is_some();
        let needs_num = entry.live_in_num.get(i).copied().unwrap_or(false);
        let needs_int = entry.live_in_int.get(i).copied().unwrap_or(false);
        // A module variable the loop carries in a parameter: the body
        // polling here stored it back first.
        let value = match entry.live_in_modvar.get(i).copied().flatten() {
            Some(slot) => vm.engine.module_var(id, slot),
            None => pairs.iter().find(|(r, _)| r == reg).map(|(_, v)| *v),
        };
        match value {
            Some(v)
                if !needs_field
                    && !v.is_undefined()
                    && !(needs_num && !v.is_num())
                    && !(needs_int && !integral(v)) =>
            {
                args.push(v)
            }
            _ => {
                if env_flag(&OSR_TRACE, "WLIFT_OSR_TRACE") {
                    eprintln!(
                        "osr-trace: retier decline FuncId({}) bb{} v{}",
                        func_id, header, reg
                    );
                }
                vm.engine.stop_retier(id, caller_gen);
                return decline;
            }
        }
    }
    let depth = jit_depth();
    if depth >= MAX_JIT_DEPTH {
        return decline;
    }
    if env_flag(&OSR_TRACE, "WLIFT_OSR_TRACE") {
        eprintln!(
            "osr-trace: [{:.2}ms] retier FuncId({}) bb{} argc={}",
            crate::runtime::engine::trace_clock_ms(),
            func_id,
            header,
            args.len() - 1
        );
    }
    vm.engine.note_osr_entry(id);
    let saved_func_id = unsafe { (*jit_state()).ctx.current_func_id };
    unsafe { (*jit_state()).ctx.current_func_id = func_id };
    set_jit_depth(depth + 1);
    let f: extern "C" fn(*const u64) -> u64 = unsafe { std::mem::transmute(entry.ptr) };
    // The caller's activation goes on in the body entered.
    let j = jit_state();
    let mut link = EntryLink::new();
    unsafe { enter_link(j, &mut link) };
    shadow_entry();
    let result = f(args[1..].as_ptr() as *const u64);
    unsafe { leave_link(j, &link) };
    set_jit_depth(depth);
    unsafe { (*jit_state()).ctx.current_func_id = saved_func_id };
    result
}

/// Readable zero words compiled code reads in place of an object
/// header when the receiver is not an object: its class is null, so a
/// class check on it always misses without a branch before the load.
pub static JIT_NULL_OBJECT: [u64; 8] = [0; 8];

static OSR_TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
static TIER_TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

#[inline(always)]
fn set_current_native_shadow_roots_ptr(ptr: *mut Value) {
    unsafe {
        std::ptr::write_volatile(std::ptr::addr_of_mut!(CURRENT_NATIVE_SHADOW_ROOTS), ptr);
    }
    // Full compiler fence: ensures the store is visible before any
    // subsequent JIT function call reads the global via raw ldr.
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
}

#[inline(always)]
pub fn current_native_shadow_roots_ptr() -> *mut Value {
    unsafe { std::ptr::read_volatile(std::ptr::addr_of!(CURRENT_NATIVE_SHADOW_ROOTS)) }
}

#[inline(always)]
fn sync_flat_shadow_ptr(stack: &mut FlatShadowStack) {
    let ptr = if let Some(&start) = stack.boundaries.last() {
        if (start as usize) < stack.roots.len() {
            unsafe { stack.roots.as_mut_ptr().add(start as usize) }
        } else {
            std::ptr::null_mut()
        }
    } else {
        std::ptr::null_mut()
    };
    set_current_native_shadow_roots_ptr(ptr);
}

/// Maximum native JIT recursion depth before falling back to interpreter.
/// Each JIT call chain level uses ~1-2KB of native stack.
/// 256 levels ≈ 256-512KB, well within the default 8MB stack.
pub const MAX_JIT_DEPTH: u32 = 256;

/// Native frames entered through direct calls that are still active;
/// compiled code counts them itself and takes the helper path past
/// `MAX_JIT_DEPTH`, whose fallback runs the callee in the interpreter.
/// Process-wide: a second VM thread only makes the bound stricter.
pub static JIT_DIRECT_DEPTH: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Read the current JIT native recursion depth.
#[inline(always)]
pub fn jit_depth() -> u32 {
    unsafe { (*jit_state()).depth }
}

/// Set the JIT native recursion depth.
#[inline(always)]
pub fn set_jit_depth(depth: u32) {
    unsafe { (*jit_state()).depth = depth };
}

/// Check if JIT dispatch is disabled (for shadow check mode).
#[inline(always)]
pub fn jit_disabled() -> bool {
    unsafe { (*jit_state()).disabled }
}

/// Whether the IC kind=1 inline-JIT fast path is enabled. Default ON.
///
/// `WLIFT_DISABLE_IC_JIT=1` turns it off — useful when chasing a
/// PAC-fault / stale-arg miscompile (the kind=1 path transmutes
/// `jit_ptr` back into a function pointer; without JIT stack maps
/// register-passed args can be stale across GC).
#[inline(always)]
pub fn ic_jit_kind1_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("WLIFT_DISABLE_IC_JIT").is_none())
}

#[inline(always)]
pub fn shadow_nonleaf_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("WLIFT_SHADOW_NONLEAF").is_some())
}

/// Set up the JIT context before calling compiled code.
#[inline(always)]
pub fn set_jit_context(ctx: JitContext) {
    unsafe { (*jit_state()).ctx = ctx };
}

/// Read the current JIT context (zero-copy reference via UnsafeCell).
#[inline(always)]
pub fn read_jit_ctx() -> JitContext {
    unsafe { (*jit_state()).ctx }
}

/// Fast null check for ctx.vm without copying the full context.
#[inline(always)]
pub fn jit_ctx_vm_is_null() -> bool {
    unsafe { (*jit_state()).ctx.vm.is_null() }
}

/// Mutate the JIT context in place (no copy — direct field access).
#[inline(always)]
pub fn mutate_jit_ctx(f: impl FnOnce(&mut JitContext)) {
    unsafe { f(&mut (*jit_state()).ctx) };
}

/// The calling thread's JIT context, for a loop that writes it on every
/// iteration without paying the thread-local lookup each time. Valid for
/// the thread's lifetime and only on this thread.
#[inline(always)]
pub fn jit_ctx_raw() -> *mut JitContext {
    unsafe { &mut (*jit_state()).ctx as *mut JitContext }
}

// ---------------------------------------------------------------------------
// JIT root set — GC-visible heap pointers held by native frames
// ---------------------------------------------------------------------------

/// Push a value into the JIT root set so GC can see it.
#[inline(always)]
pub fn push_jit_root(v: Value) {
    unsafe { (*jit_state()).roots.push(v) };
}

thread_local! {
    /// Depth of native callbacks in progress. While non-zero the
    /// conservative collector does not collect from allocation
    /// helpers: a native may hold values in a Rust `Vec` across the
    /// callback, which no scan reaches.
    static COLLECT_SUPPRESS: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// Suppresses allocation-time collection until dropped.
pub struct CollectSuppressGuard(());

impl CollectSuppressGuard {
    pub fn new() -> Self {
        COLLECT_SUPPRESS.with(|c| c.set(c.get() + 1));
        CollectSuppressGuard(())
    }
}

impl Default for CollectSuppressGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for CollectSuppressGuard {
    fn drop(&mut self) {
        COLLECT_SUPPRESS.with(|c| c.set(c.get() - 1));
    }
}

#[inline(always)]
fn collect_suppressed() -> bool {
    COLLECT_SUPPRESS.with(|c| c.get() != 0)
}

/// The end of an allocation helper called from compiled code, which
/// is a safepoint in every tier: native frames are scanned, not
/// mapped, so the value needs no root entry, and it is pinned in this
/// frame across the collection.
///
/// wasm32 JIT frames keep values in locals no scan reaches, so
/// allocation stays a non-safepoint there until the shadow stack
/// lands.
///
/// # Safety
/// `vm` must be the current thread's running VM.
#[inline]
pub unsafe fn finish_alloc(vm: &mut crate::runtime::vm::VM, val: Value) -> u64 {
    #[cfg(not(target_arch = "wasm32"))]
    if vm.safepoint_due() && !collect_suppressed() {
        let pinned = std::hint::black_box(val);
        vm.safepoint_work(false);
        if vm.gc.take_freed_code_objects() {
            vm.method_cache.invalidate();
            vm.engine.invalidate_inline_caches();
        }
        return std::hint::black_box(&pinned).to_bits();
    }
    #[cfg(target_arch = "wasm32")]
    let _ = vm;
    val.to_bits()
}

/// Snapshot the current JIT roots length. Called at AOT function
/// entry; paired with `wren_jit_roots_restore_len` at exit so any
/// roots leaked into `JIT_ROOTS_STORE` by the function's
/// allocations get released at the function boundary.
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_jit_roots_snapshot() -> u64 {
    jit_roots_snapshot_len() as u64
}

/// Restore JIT roots to a previous snapshot length. Called at AOT
/// function exit (or any other scope boundary the lowering wants
/// to release roots at).
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_jit_roots_restore(len: u64) {
    jit_roots_restore_len(len as usize);
}

/// A copy of this thread's JIT roots, for publishing to a collector
/// on another thread.
pub fn jit_roots_snapshot() -> Vec<Value> {
    unsafe { (*jit_state()).roots.clone() }
}

/// Take this thread's JIT roots, for a fiber switch that keeps its own.
pub fn take_jit_roots() -> Vec<Value> {
    unsafe { std::mem::take(&mut (*jit_state()).roots) }
}

/// Put a fiber's JIT roots back.
pub fn set_jit_roots(roots: Vec<Value>) {
    unsafe { (*jit_state()).roots = roots };
}

/// Pop and return the last JIT root, if any.
#[inline(always)]
pub fn pop_jit_root() -> Option<Value> {
    unsafe { (*jit_state()).roots.pop() }
}

/// Clear JIT roots (called after native code returns to interpreter).
#[inline(always)]
pub fn clear_jit_roots() {
    unsafe {
        let v = &mut (*jit_state()).roots;
        v.clear();
        // Prevent capacity from growing unboundedly after root spikes.
        if v.capacity() > 256 {
            v.shrink_to(256);
        }
    }
}

/// Get current JIT roots count (for save/restore around re-entrant calls).
#[inline(always)]
#[allow(dead_code)]
fn jit_roots_len() -> usize {
    unsafe { (*jit_state()).roots.len() }
}

/// Truncate JIT roots back to a saved length (pop roots added by this frame).
#[inline(always)]
#[allow(dead_code)]
fn jit_roots_truncate(len: usize) {
    unsafe { (*jit_state()).roots.truncate(len) };
}

/// Get the current JIT roots length for save/restore by external callers.
#[inline(always)]
pub fn jit_roots_snapshot_len() -> usize {
    unsafe { (*jit_state()).roots.len() }
}

/// Read the JIT root at the given index (for reading GC-forwarded pointers).
#[inline(always)]
pub fn jit_root_at(idx: usize) -> crate::runtime::value::Value {
    unsafe { (&(*jit_state()).roots)[idx] }
}

/// Truncate JIT roots to the given length (public version for external callers).
#[inline(always)]
pub fn jit_roots_restore_len(len: usize) {
    unsafe { (*jit_state()).roots.truncate(len) };
}

/// Read JitContext's GC-managed pointers as Values for root scanning.
/// Returns (closure_val, defining_class_val) — null if not set.
pub fn jit_context_roots() -> (Value, Value) {
    let ctx = read_jit_ctx();
    let closure = if ctx.closure.is_null() {
        Value::null()
    } else {
        Value::object(ctx.closure)
    };
    let defining_class = if ctx.defining_class.is_null() {
        Value::null()
    } else {
        Value::object(ctx.defining_class)
    };
    (closure, defining_class)
}

/// Push the GC-managed pointers of `ctx` (closure + defining_class)
/// onto the JIT roots stack so they survive any GC fired by code
/// running after this call. Returns the snapshot length to pass to
/// [`restore_rooted_jit_context`] for the matching restore.
#[inline(always)]
pub(crate) fn root_saved_jit_context(ctx: JitContext) -> usize {
    let root_len_before = jit_roots_snapshot_len();
    push_jit_root(if ctx.closure.is_null() {
        Value::null()
    } else {
        Value::object(ctx.closure)
    });
    push_jit_root(if ctx.defining_class.is_null() {
        Value::null()
    } else {
        Value::object(ctx.defining_class)
    });
    root_len_before
}

/// Read the (possibly GC-updated) closure + defining_class back
/// from the JIT roots stack, overlay them onto `saved_ctx`, then
/// restore the resulting context as the live JitContext. Pair with
/// [`root_saved_jit_context`] across any callsite that can trigger
/// GC and clobber the snapshot's pointers.
#[inline(always)]
pub(crate) fn restore_rooted_jit_context(mut saved_ctx: JitContext, root_len_before: usize) {
    saved_ctx.closure = jit_root_at(root_len_before)
        .as_object()
        .unwrap_or(std::ptr::null_mut());
    saved_ctx.defining_class = jit_root_at(root_len_before + 1)
        .as_object()
        .unwrap_or(std::ptr::null_mut());
    jit_roots_restore_len(root_len_before);
    set_jit_context(saved_ctx);
    refresh_module_vars();
}

/// Access the JIT context. Returns None if not set (vm is null).
#[inline(always)]
fn with_context<T>(f: impl FnOnce(&JitContext) -> T) -> Option<T> {
    let ctx = read_jit_ctx();
    if ctx.vm.is_null() {
        None
    } else {
        Some(f(&ctx))
    }
}

pub fn push_native_shadow_frame(slot_count: usize) {
    FLAT_SHADOW.with(|s| unsafe {
        let stack = &mut *s.get();
        let start = stack.roots.len();
        stack.boundaries.push(start as u16);
        // resize reuses existing capacity after warmup — zero alloc.
        stack.roots.resize(start + slot_count, Value::null());
        sync_flat_shadow_ptr(stack);
    });
}

pub fn pop_native_shadow_frame() {
    FLAT_SHADOW.with(|s| unsafe {
        let stack = &mut *s.get();
        if let Some(start) = stack.boundaries.pop() {
            stack.roots.truncate(start as usize);
        }
        sync_flat_shadow_ptr(stack);
    });
}

/// A copy of this thread's native shadow roots.
pub fn native_shadow_roots_snapshot() -> Vec<Value> {
    FLAT_SHADOW.with(|s| unsafe { (*s.get()).roots.clone() })
}

#[inline(always)]
fn trace_native_entry(
    vm: &crate::runtime::vm::VM,
    func_id: crate::runtime::engine::FuncId,
    kind: &str,
) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !env_flag(&ON, "WLIFT_TRACE_NATIVE_ENTRY") {
        return;
    }
    let name = vm
        .engine
        .get_mir(func_id)
        .map(|mir| vm.interner.resolve(mir.name).to_string())
        .unwrap_or_else(|| "<unknown>".to_string());
    eprintln!("native-entry: {kind} FuncId({}) {}", func_id.0, name);
}

#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_shadow_store(slot: u64, value: u64) -> u64 {
    let slot = slot as usize;
    let value = Value::from_bits(value);
    FLAT_SHADOW.with(|s| unsafe {
        let stack = &mut *s.get();
        if let Some(&start) = stack.boundaries.last() {
            let idx = start as usize + slot;
            if idx < stack.roots.len() {
                stack.roots[idx] = value;
            }
        }
    });
    value.to_bits()
}

#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_shadow_load(slot: u64) -> u64 {
    let slot = slot as usize;
    FLAT_SHADOW.with(|s| unsafe {
        let stack = &*s.get();
        stack
            .boundaries
            .last()
            .and_then(|&start| stack.roots.get(start as usize + slot).copied())
            .unwrap_or(Value::null())
            .to_bits()
    })
}

/// Callee-managed shadow frame: push in JIT prologue.
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_enter_shadow_frame(slot_count: u64) {
    let count = slot_count as usize;
    if count > 0 {
        push_native_shadow_frame(count);
    }
}

/// Callee-managed shadow frame: pop in JIT epilogue.
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_exit_shadow_frame() {
    pop_native_shadow_frame();
}

/// Get the VM pointer from the JIT context, falling back to the
/// wasm-side `runtime::tier::current_vm()` when the JIT context
/// isn't set. The native JIT installs `read_jit_ctx().vm` on
/// every dispatch boundary, so the fallback never fires there;
/// the wasm tier-up never installs that thread-local but does
/// set `current_vm` on `interpret` / `run_fiber` entry, so this
/// keeps every `wren_*` helper that uses `vm_ref` working on
/// both targets without a parallel wasm copy.
///
/// # Safety
/// Caller must ensure the VM pointer (whichever path is active)
/// is valid for the duration of the borrow.
#[inline(always)]
unsafe fn vm_ref() -> Option<&'static mut crate::runtime::vm::VM> {
    unsafe { vm_at(jit_state()) }
}

/// The VM the given JIT state runs on.
#[inline(always)]
unsafe fn vm_at(j: *mut JitThread) -> Option<&'static mut crate::runtime::vm::VM> {
    let vm = unsafe { (*j).ctx.vm };
    if !vm.is_null() {
        return Some(unsafe { &mut *(vm as *mut crate::runtime::vm::VM) });
    }
    #[cfg(all(target_arch = "wasm32", not(feature = "host")))]
    {
        let p = crate::runtime::tier::current_vm();
        if !p.is_null() {
            return Some(unsafe { &mut *p });
        }
    }
    None
}

/// Refresh JitContext's module_vars pointer from the VM's engine.
/// Must be called after any operation that might reallocate the module's
/// variable Vec (e.g., call_closure_sync which can import modules).
#[allow(dead_code)]
fn refresh_module_vars() {
    let ctx = read_jit_ctx();
    if ctx.module_name.is_null() {
        return;
    }

    let vm = unsafe { vm_ref() };
    if let Some(vm) = vm {
        let name = unsafe {
            std::str::from_utf8_unchecked(std::slice::from_raw_parts(
                ctx.module_name,
                ctx.module_name_len as usize,
            ))
        };
        if let Some(m) = vm.engine.modules.get(name) {
            mutate_jit_ctx(|c| {
                c.module_vars = m.vars.as_ptr() as *mut u64;
                c.module_var_count = m.vars.len() as u32;
            });
        }
    }
}

/// Get the module name from the JIT context.
pub fn module_name() -> String {
    let ctx = read_jit_ctx();
    if ctx.module_name.is_null() || ctx.module_name_len == 0 {
        String::new()
    } else {
        unsafe {
            let slice = std::slice::from_raw_parts(ctx.module_name, ctx.module_name_len as usize);
            String::from_utf8_lossy(slice).into_owned()
        }
    }
}

// ---------------------------------------------------------------------------
// Runtime function implementations
// ---------------------------------------------------------------------------

/// Get a module variable by slot index.
#[cfg(any(feature = "host", feature = "aot_runtime"))]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_get_module_var(slot: u64) -> u64 {
    with_context(|ctx| {
        let idx = slot as usize;
        if idx < ctx.module_var_count as usize {
            unsafe { *ctx.module_vars.add(idx) }
        } else {
            Value::null().to_bits()
        }
    })
    .unwrap_or(Value::null().to_bits())
}

/// Read the current `JitContext.closure` raw pointer. AOT bodies
/// call this once at function entry (when the body has any
/// upvalue access) and stash the result in a function-scoped
/// local — every subsequent `Instruction::GetUpvalue` /
/// `SetUpvalue` then lowers to inline pointer chasing against
/// that local instead of routing through `wren_get/set_upvalue`,
/// which would re-read TLS and pay a helper-call's worth of
/// overhead on every access.
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_load_jit_closure() -> u64 {
    let ctx = read_jit_ctx();
    ctx.closure as u64
}

/// Returns 1 when the VM has an in-flight runtime error, 0 otherwise.
/// AOT bodies poll this at every basic-block entry — without it,
/// straight-line Cranelift code keeps running after `Fiber.abort`
/// (the BC interp's per-opcode `has_error` check has no analogue),
/// turning a single abort inside a `while (true) { … }` loop into
/// an infinite stream of repeat-aborts.
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_aot_check_error() -> u64 {
    match unsafe { vm_ref() } {
        Some(vm) if vm.has_error => 1,
        _ => 0,
    }
}

/// A word on a cache line of its own: compiled code reads it after
/// every call, and a line that any hot counter is written to would
/// stall those reads.
#[repr(align(128))]
pub struct PendingWord(pub std::sync::atomic::AtomicU32);

/// Set while any VM has an error to unwind: compiled code polls this
/// word after every call that can raise and asks `wren_aot_check_error`
/// only when it is set, so the poll costs a load on the way through
/// and the helper answers for the VM in hand. Raised with every error,
/// lowered when one is drained; a stale raise costs a call, a missed
/// one is a poll that goes on.
pub static ERROR_PENDING: PendingWord = PendingWord(std::sync::atomic::AtomicU32::new(0));

#[inline]
pub fn note_error_pending() {
    ERROR_PENDING
        .0
        .store(1, std::sync::atomic::Ordering::Release);
}

#[inline]
pub fn clear_error_pending() {
    ERROR_PENDING
        .0
        .store(0, std::sync::atomic::Ordering::Release);
}

/// Whether the helper `name` can leave an error pending: it dispatches
/// a method, runs a compiled callee, or raises on its operand. A body
/// polls [`ERROR_PENDING`] after every call to one.
pub fn helper_can_raise(name: &str) -> bool {
    const PREFIXES: [&str; 11] = [
        "wren_call_",
        "wren_known_call_",
        "wren_ic_call_",
        "wren_ic_host_",
        "wren_ic_native_",
        "wren_ic_ctor_",
        "wren_super_call",
        "wren_construct_",
        "wren_num_",
        "wren_bit_",
        "wren_cmp_",
    ];
    PREFIXES.iter().any(|p| name.starts_with(p))
        || matches!(
            name,
            "wren_to_string" | "wren_subscript_get" | "wren_subscript_set"
        )
}

/// Load the JIT code pointer for a given function ID.
/// Returns the function pointer as u64 (0 if not compiled).
/// Used by CallKnownFunc to do direct JIT-to-JIT calls.
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_load_jit_ptr(func_id: u64) -> u64 {
    let ctx = read_jit_ctx();
    let idx = func_id as usize;
    if idx < ctx.jit_code_len as usize && !ctx.jit_code_base.is_null() {
        unsafe { *ctx.jit_code_base.add(idx) as u64 }
    } else {
        0
    }
}

/// Set a module variable by slot index.
#[cfg(any(feature = "host", feature = "aot_runtime"))]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_set_module_var(slot: u64, value: u64) -> u64 {
    with_context(|ctx| {
        let idx = slot as usize;
        if idx < ctx.module_var_count as usize {
            unsafe {
                *ctx.module_vars.add(idx) = value;
            }
        }
    });
    value
}

/// Try to call a closure via native code if it's compiled; fall back to interpreter.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn call_closure_jit_or_sync(
    vm: &mut crate::runtime::vm::VM,
    closure_ptr: *mut ObjClosure,
    args: &[Value],
    defining_class: Option<*mut crate::runtime::object::ObjClass>,
) -> u64 {
    vm.engine
        .note_runtime_call_stats(|s| s.call_closure_entries += 1);
    // Save the caller's JIT context so we can restore it after the nested call.
    // Without this, interpreter fallback paths (call_closure_sync → run_fiber)
    // can overwrite the context via dispatch_closure_bc, corrupting the caller's
    // module_vars / closure / defining_class when control returns.
    let saved_ctx = read_jit_ctx();
    let saved_ctx_root_len = root_saved_jit_context(saved_ctx);
    vm.engine
        .note_runtime_call_stats(|s| s.jit_context_save_restore_pairs += 1);

    // Update context for the callee: set closure, defining_class, and
    // ensure vm is set (may be null if we're called from a fresh thread
    // or after a context restore). Swap `module_vars` to the closure's
    // defining module so its `GetModuleVar(slot)` ops resolve against
    // the right slot table — without this, a closure defined in main
    // and invoked from inside `@hatch:template`'s body (e.g. site's
    // `FnLoader.new(Fn.new {|name| Fs.readText("./views/" + name) })`
    // called from `TemplateRegistry.get`) reads the caller's slot N
    // and `Fs` comes back null. Same fast/slow split
    // `call_jit_with_shadow` uses: skip the swap for intra-module
    // calls.
    let callee_func_id =
        crate::runtime::engine::FuncId(unsafe { (*(*closure_ptr).function).fn_id });
    let callee_module = vm.engine.func_module(callee_func_id).cloned();
    let defining_class = defining_class.or_else(|| unsafe { (*closure_ptr).defining_class_opt() });
    mutate_jit_ctx(|ctx| {
        if ctx.vm.is_null() {
            ctx.vm = vm as *mut _ as *mut u8;
        }
        ctx.current_func_id = callee_func_id.0 as u64;
        ctx.closure = closure_ptr as *mut u8;
        ctx.defining_class = defining_class
            .map(|p| p as *mut u8)
            .unwrap_or(std::ptr::null_mut());
    });
    if let Some(mn) = callee_module.as_ref() {
        let (mv_ptr, mv_count) = vm.engine.module_vars_for(callee_func_id);
        if !mv_ptr.is_null() {
            let bytes = mn.as_bytes();
            mutate_jit_ctx(|ctx| {
                ctx.module_vars = mv_ptr;
                ctx.module_var_count = mv_count;
                ctx.module_name = bytes.as_ptr();
                ctx.module_name_len = bytes.len() as u32;
            });
        }
    }

    // Install completed compilations so we can dispatch natively.
    if vm.engine.has_pending_compilations() {
        vm.engine.poll_compilations();
    }
    let mut had_native_candidate = false;
    if args.len() <= 8 {
        let func_id = crate::runtime::engine::FuncId(unsafe { (*(*closure_ptr).function).fn_id });

        let native_fn_ptr: Option<*const u8> = if jit_disabled() {
            None
        } else {
            vm.engine
                .jit_code
                .get(func_id.0 as usize)
                .copied()
                .filter(|p| !p.is_null())
        };

        if let Some(fn_ptr) = native_fn_ptr {
            had_native_candidate = true;
            vm.engine
                .note_runtime_call_stats(|s| s.call_closure_native_candidates += 1);
            // Guard: check native recursion depth to prevent stack overflow.
            // Each JIT call level uses ~1-2KB native stack; limit to 256 levels.
            let depth = unsafe { (*jit_state()).depth };
            let is_leaf = vm
                .engine
                .jit_leaf
                .get(func_id.0 as usize)
                .copied()
                .unwrap_or(false);
            if depth < MAX_JIT_DEPTH {
                trace_native_entry(
                    vm,
                    func_id,
                    if is_leaf {
                        "leaf-nested"
                    } else {
                        "nonleaf-nested"
                    },
                );
                vm.engine.note_native_entry(func_id);
                vm.engine.note_native_to_native_call(func_id);
                vm.engine
                    .note_runtime_call_stats(|s| s.call_closure_native_entries += 1);
                unsafe { (*jit_state()).depth = depth + 1 };
                let root_len_before = jit_roots_snapshot_len();
                // Shadow frame push/pop handled by call_jit_with_shadow.
                let result = unsafe { call_jit_with_shadow(vm, fn_ptr, func_id, args) };
                unsafe { (*jit_state()).depth = depth };

                jit_roots_restore_len(root_len_before);
                // Restore caller's JIT context.
                restore_rooted_jit_context(saved_ctx, saved_ctx_root_len);
                return result;
            }
            // depth >= MAX_JIT_DEPTH: fall through to interpreter path.
        }
    }

    // Fall back to interpreter.
    if had_native_candidate {
        let func_id = crate::runtime::engine::FuncId(unsafe { (*(*closure_ptr).function).fn_id });
        vm.engine.note_fallback_to_interpreter(func_id);
        vm.engine
            .note_runtime_call_stats(|s| s.call_closure_interpreter_fallbacks += 1);
    }
    let result = vm
        .call_closure_sync(closure_ptr, args, defining_class)
        .map(|v| v.to_bits())
        .unwrap_or(Value::null().to_bits());
    // Restore caller's JIT context.
    restore_rooted_jit_context(saved_ctx, saved_ctx_root_len);
    result
}

/// Walk the class hierarchy to find the method and its defining class.
/// Unlike `find_method()` which returns from the flat copied table,
/// this walks superclass pointers to find the original defining class —
/// needed for correct super() dispatch and static field access.
unsafe fn find_method_with_class(
    cls: *mut ObjClass,
    method: crate::intern::SymbolId,
) -> Option<(Method, *mut ObjClass)> {
    unsafe {
        let idx = method.index() as usize;
        let mut c = cls;
        while !c.is_null() {
            let cls_ref = &*c;
            if idx < cls_ref.methods.len()
                && let Some(m) = &cls_ref.methods[idx]
            {
                return Some((*m, c));
            }
            c = (*c).superclass;
        }
        None
    }
}

/// Internal: dispatch a method call with a pre-built args slice.
fn dispatch_call(recv: Value, method_packed: u64, args: &[Value]) -> u64 {
    let j = jit_state();
    match unsafe { vm_at(j) } {
        Some(vm) => dispatch_call_rooted(vm, j, recv, method_packed, args),
        None => Value::null().to_bits(),
    }
}

fn dispatch_call_rooted(
    vm: &mut crate::runtime::vm::VM,
    j: *mut JitThread,
    recv: Value,
    method_packed: u64,
    args: &[Value],
) -> u64 {
    // If we already have a runtime error in flight, short-circuit
    // every subsequent call within the AOT body to a null return.
    // The block-entry `wren_aot_check_error` poll catches the
    // error at the next branch boundary and unwinds, but a
    // straight-line block can hold multiple Call sites — without
    // this guard, an aborted `checkAlive_()` is followed by the
    // foreign `SqliteCore.execute(_id, ...)` call, which validates
    // _id and overwrites the original "use after close" error
    // with its own "id must be a non-negative integer". Mirrors
    // the BC interpreter's per-opcode `has_error` short-circuit.
    if vm.has_error {
        return Value::null().to_bits();
    }
    vm.engine
        .note_runtime_call_stats(|s| s.dispatch_call_entries += 1);
    let (method_sym, ic_idx) = decode_method_and_ic(method_packed);

    // Match the interpreter's closure call fast path. Fn.call(...) is a stub
    // on the Fn class; actual closure invocation must pass only the user args.
    if vm.is_call_sym(method_sym)
        && recv.is_object()
        && let Some(ptr) = recv.as_object()
    {
        let header = ptr as *const ObjHeader;
        if unsafe { (*header).obj_type } == ObjType::Closure {
            let closure_ptr = ptr as *mut ObjClosure;
            vm.engine
                .note_runtime_call_stats(|s| s.dispatch_call_fn_fastpath += 1);
            return call_closure_jit_or_sync(vm, closure_ptr, &args[1..], None);
        }
    }

    let class = vm.class_of(recv);
    let cache_key_class = cache_key_class(vm, recv, class);
    if class == vm.list_class
        && let Some(result) = try_dispatch_list_native_fastpath(vm, recv, method_sym, args)
    {
        note_list_fast_path_ic(vm, j, ic_idx);
        return result;
    }
    let ic_ptr = ic_idx.and_then(|(idx, func)| current_jit_callsite_ic(vm, j, idx, func));

    if let Some(ic_ptr) = ic_ptr {
        vm.engine
            .note_runtime_call_stats(|s| s.dispatch_call_ic_attempts += 1);
        if let Some(result) = try_dispatch_callsite_ic(vm, ic_ptr, recv, args, cache_key_class) {
            return result;
        }
    }

    // Check method cache first (avoids find_method + static symbol resolution).
    if let Some((m, dc)) = vm.method_cache.lookup(cache_key_class, method_sym) {
        vm.engine
            .note_runtime_call_stats(|s| s.dispatch_call_method_cache_hits += 1);
        if let Some(ic_ptr) = ic_ptr {
            populate_callsite_ic(vm, ic_ptr, cache_key_class, m, dc);
        }
        return dispatch_method(vm, m, args, Some(dc));
    }

    // Cache miss: full hierarchy lookup to get the correct defining class.
    vm.engine
        .note_runtime_call_stats(|s| s.dispatch_call_method_cache_misses += 1);
    let lookup = unsafe { find_method_with_class(class, method_sym) };

    match lookup {
        Some((m, dc)) => {
            vm.method_cache.insert(cache_key_class, method_sym, m, dc);
            if let Some(ic_ptr) = ic_ptr {
                populate_callsite_ic(vm, ic_ptr, cache_key_class, m, dc);
            }
            dispatch_method(vm, m, args, Some(dc))
        }
        None => {
            // Try static method dispatch (receiver IS a class object)
            if class == vm.class_class && !cache_key_class.is_null() {
                // Stack-allocated buffer to build "static:method" without heap alloc.
                let method_str = vm.interner.resolve(method_sym);
                let prefix = b"static:";
                let total = prefix.len() + method_str.len();
                let mut buf = [0u8; 128];
                let found = if total <= buf.len() {
                    buf[..prefix.len()].copy_from_slice(prefix);
                    buf[prefix.len()..total].copy_from_slice(method_str.as_bytes());
                    let static_str = unsafe { std::str::from_utf8_unchecked(&buf[..total]) };
                    vm.interner
                        .lookup(static_str)
                        .and_then(|sym| unsafe { find_method_with_class(cache_key_class, sym) })
                } else {
                    None
                };
                if let Some((m, dc)) = found {
                    vm.method_cache.insert(cache_key_class, method_sym, m, dc);
                    if let Some(ic_ptr) = ic_ptr {
                        populate_callsite_ic(vm, ic_ptr, cache_key_class, m, dc);
                    }
                    dispatch_method(vm, m, args, Some(dc))
                } else {
                    let name = vm.interner.resolve(method_sym).to_string();
                    raise_method_not_found(vm, recv, &name)
                }
            } else {
                let name = vm.interner.resolve(method_sym).to_string();
                raise_method_not_found(vm, recv, &name)
            }
        }
    }
}

/// Process a pending fiber action set by a native method, synchronously from
/// within the JIT dispatch path.  Returns the value that the caller should see
/// as the result of the `fiber.call()` / `Fiber.yield()` etc. invocation.
///
/// The key design: when the JIT calls `fiber.call()`, we run the target fiber
/// synchronously via the interpreter.  We temporarily remove the caller's
/// mir_frames so the interpreter does not re-execute the caller's module body
/// (which is being run by JIT code) when the child fiber returns.
fn handle_jit_fiber_action(
    vm: &mut crate::runtime::vm::VM,
    action: crate::runtime::vm::FiberAction,
) -> u64 {
    use crate::runtime::object::FiberState;
    use crate::runtime::vm::FiberAction;

    // Determine Call vs Transfer before destructuring moves the enum.
    let is_call = matches!(&action, FiberAction::Call { .. });

    // Save the caller's JIT context — `run_fiber` below drives the
    // bytecode interpreter on the target fiber, which resets the
    // JitContext on every loop iteration to point at the active
    // frame's module. When the child eventually yields/completes
    // and we return to the caller's JIT'd code, that JIT'd code
    // expects its own module's `module_vars` / `module_name` /
    // `closure` / `defining_class` / `current_func_id` —
    // otherwise downstream `wren_get_module_var` reads the wrong
    // slot table (e.g. an `@hatch:web` closure dispatched
    // mid-listen would inherit the SSE writer's module after a
    // yield, and `listener.tryAccept` would resolve `listener`
    // against the wrong module).
    let saved_jit_ctx = read_jit_ctx();
    let saved_jit_depth = jit_depth();

    match action {
        FiberAction::Call { target, value } | FiberAction::Transfer { target, value } => {
            let caller = vm.fiber;
            unsafe {
                if !caller.is_null() {
                    (*caller).state = FiberState::Suspended;
                }
            }

            // krio fast path. AOT-compiled `fiber.try() / .call()`
            // sites bypass the `fiber_try_*` / `fiber_call_*` foreign
            // methods entirely and emit a direct call to this helper —
            // so the krio routing that lives in those foreign methods
            // never runs for AOT'd bodies. Routing through krio here
            // lets the body run on its own stack so yield can switch
            // back.
            #[cfg(feature = "host")]
            {
                if is_call
                    && unsafe { (*target).krio_fiber.is_some() }
                    && crate::runtime::core::fiber::current_vm_krio_active(vm)
                    && let Some(v) = crate::runtime::core::fiber::try_krio_call_pub(target, value)
                {
                    set_jit_context(saved_jit_ctx);
                    set_jit_depth(saved_jit_depth);
                    return v.to_bits();
                }
            }
            // A krio-backed target returned to the caller through its own
            // stack above; only the shared-loop paths below hand the
            // caller to the target. A krio loop that found a caller
            // would run the caller's frame on the fiber's own stack.
            unsafe {
                if is_call {
                    (*target).caller = caller;
                }
            }

            let target_state = unsafe { (*target).state };
            if target_state == FiberState::Suspended {
                // Resuming a suspended fiber: deliver the value.
                unsafe {
                    if let Some(dst) = (*target).resume_value_dst.take()
                        && let Some(frame) = (*target).mir_frames.last_mut()
                    {
                        let i = dst.0 as usize;
                        if i < frame.values.len() {
                            frame.values[i] = value;
                        } else {
                            frame.values.resize(i + 1, Value::null());
                            frame.values[i] = value;
                        }
                    }
                }
            }
            // Switch to the target fiber and run it synchronously.
            //
            // Problem: when the child fiber completes/yields, the interpreter's
            // fiber_loop resumes the caller via resume_caller() which sets
            // vm.fiber = caller and continues the fiber_loop.  That would
            // re-run the caller's frames (which are being executed by JIT),
            // causing double execution.
            //
            // Solution: temporarily take the caller's mir_frames so the
            // interpreter sees frame_count == 0 on the caller fiber and
            // returns instead of re-executing the module body.
            let saved_caller_frames = if !caller.is_null() {
                unsafe { std::mem::take(&mut (*caller).mir_frames) }
            } else {
                Vec::new()
            };
            // Root target / caller across run_fiber the same way the
            // SM-poll branch above does: run_fiber drives the BC
            // interpreter, which can collect.
            let target_root_idx = jit_roots_snapshot_len();
            push_jit_root(Value::object(target as *mut u8));
            let caller_root_idx = if !caller.is_null() {
                let idx = jit_roots_snapshot_len();
                push_jit_root(Value::object(caller as *mut u8));
                Some(idx)
            } else {
                None
            };
            vm.fiber = target;
            let result = crate::runtime::vm_interp::run_fiber(vm);
            // Refresh forwarded fiber pointers from JIT_ROOTS_STORE
            // before any dereferences. Pop in reverse-push order.
            let target = jit_root_at(target_root_idx)
                .as_object()
                .map(|p| p as *mut crate::runtime::object::ObjFiber)
                .unwrap_or(target);
            let caller = caller_root_idx
                .map(|idx| {
                    jit_root_at(idx)
                        .as_object()
                        .map(|p| p as *mut crate::runtime::object::ObjFiber)
                        .unwrap_or(caller)
                })
                .unwrap_or(caller);
            if caller_root_idx.is_some() {
                let _ = pop_jit_root();
            }
            let _ = pop_jit_root();
            // Restore the caller's frames and re-activate it.
            if !caller.is_null() {
                // Check if the child fiber yielded a value back to us.
                // resume_caller() stores it in jit_resume_value when frames
                // are empty (our JIT barrier).
                let yield_val = unsafe { (*caller).jit_resume_value.take() };
                unsafe {
                    (*caller).mir_frames = saved_caller_frames;
                    (*caller).state = FiberState::Running;
                }
                vm.fiber = caller;
                set_jit_context(saved_jit_ctx);
                set_jit_depth(saved_jit_depth);
                if let Some(v) = yield_val {
                    return v.to_bits();
                }
            } else {
                set_jit_context(saved_jit_ctx);
                set_jit_depth(saved_jit_depth);
                // No outer caller fiber (top-level AOT body
                // running directly from `wlift_aot_main`).
                // Without this branch the abort message stays
                // on `target.error` but never reaches the AOT
                // call site of `fiber.try()`, which then sees
                // `null` instead of the error string. Surface
                // it directly so `var e = Fiber.new { Fiber.abort(
                // "boom") }.try()` returns `"boom"` whether the
                // call is happening inside a fiber or at the
                // script's top level.
                let target_state = unsafe { (*target).state };
                if matches!(target_state, FiberState::Done) {
                    let err = unsafe { (*target).error };
                    if !err.is_null() {
                        return err.to_bits();
                    }
                }
            }
            match result {
                Ok(v) => v.to_bits(),
                Err(e) => {
                    // Propagate the runtime error from the child fiber.
                    vm.has_error = true;
                    note_error_pending();
                    vm.note_raise();
                    vm.last_error = Some(e.to_string());
                    Value::null().to_bits()
                }
            }
        }
        FiberAction::Yield { value } => {
            // Yield from JIT context — the fiber should return the value.
            value.to_bits()
        }
        FiberAction::Suspend => {
            // Suspend from JIT context — just return null.
            Value::null().to_bits()
        }
    }
}

/// Call `closure`'s body on `args`, receiver first, with `defining_class`
/// as the context's class: what `dispatch_method` does for a closure, for
/// a caller that has found the method itself and calls it again and
/// again. The compiled body when there is one, through the thread's JIT
/// state read once; else the interpreter, with the tier ticked so the
/// body compiles.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn call_found_closure(
    vm: &mut crate::runtime::vm::VM,
    closure: *mut ObjClosure,
    args: &[Value],
    defining_class: *mut crate::runtime::object::ObjClass,
) -> u64 {
    if let Some(result) = try_dispatch_trivial_accessor_fastpath(vm, Method::Closure(closure), args)
    {
        return result;
    }
    let func_id = crate::runtime::engine::FuncId(unsafe { (*(*closure).function).fn_id });
    let fn_idx = func_id.0 as usize;
    let fn_ptr = vm
        .engine
        .jit_code
        .get(fn_idx)
        .copied()
        .unwrap_or(std::ptr::null());
    #[cfg(feature = "cranelift")]
    let compiled = !fn_ptr.is_null();
    #[cfg(not(feature = "cranelift"))]
    let compiled = {
        let is_leaf = vm.engine.jit_leaf.get(fn_idx).copied().unwrap_or(false);
        !fn_ptr.is_null() && is_leaf
    };
    let j = jit_state();
    let state = unsafe { &mut *j };
    if !compiled || args.len() > 8 || state.disabled {
        if fn_ptr.is_null() {
            vm.engine.note_interpreted_entry(func_id);
            if vm.engine.mode != crate::runtime::engine::ExecutionMode::Interpreter
                && vm.engine.record_call(func_id)
            {
                vm.request_tier_up(func_id);
            }
        }
        return call_closure_jit_or_sync(vm, closure, args, Some(defining_class));
    }
    if state.depth >= MAX_JIT_DEPTH {
        return call_closure_jit_or_sync(vm, closure, args, Some(defining_class));
    }
    // Only the fields written here are saved; a cross-module call
    // restores the rest itself.
    let saved_vm = state.ctx.vm;
    let saved_func_id = state.ctx.current_func_id;
    let saved_closure = state.ctx.closure;
    let saved_defining_class = state.ctx.defining_class;
    let saved_depth = state.depth;
    if saved_vm.is_null() {
        state.ctx.vm = vm as *mut _ as *mut u8;
    }
    state.ctx.current_func_id = func_id.0 as u64;
    state.ctx.closure = closure as *mut u8;
    state.ctx.defining_class = defining_class as *mut u8;
    state.depth = saved_depth + 1;
    vm.engine.note_native_entry(func_id);
    let result = unsafe { call_jit_with_shadow_st(j, vm, fn_ptr, func_id, args) };
    let state = unsafe { &mut *j };
    state.depth = saved_depth;
    state.ctx.vm = saved_vm;
    state.ctx.current_func_id = saved_func_id;
    state.ctx.closure = saved_closure;
    state.ctx.defining_class = saved_defining_class;
    result
}

/// Dispatch a resolved method entry.
#[inline(always)]
pub fn dispatch_method_pub(
    vm: &mut crate::runtime::vm::VM,
    method: Method,
    args: &[Value],
    defining_class: Option<*mut crate::runtime::object::ObjClass>,
) -> u64 {
    dispatch_method(vm, method, args, defining_class)
}

fn dispatch_method(
    vm: &mut crate::runtime::vm::VM,
    method: Method,
    args: &[Value],
    defining_class: Option<*mut crate::runtime::object::ObjClass>,
) -> u64 {
    match method {
        Method::Native(native_fn) => {
            vm.engine
                .note_runtime_call_stats(|s| s.dispatch_method_native += 1);
            let result = native_fn(vm, args).to_bits();
            if let Some(action) = vm.pending_fiber_action.take() {
                return handle_jit_fiber_action(vm, action);
            }
            result
        }
        Method::Host(host_fn, context) => {
            let result = host_fn(vm, context, args).to_bits();
            if let Some(action) = vm.pending_fiber_action.take() {
                return handle_jit_fiber_action(vm, action);
            }
            result
        }
        Method::ForeignC(foreign_fn) => {
            vm.engine
                .note_runtime_call_stats(|s| s.dispatch_method_native += 1);
            let result =
                crate::runtime::foreign::dispatch_foreign_c(vm, foreign_fn, args).to_bits();
            if let Some(action) = vm.pending_fiber_action.take() {
                return handle_jit_fiber_action(vm, action);
            }
            result
        }
        Method::ForeignCDynamic(idx) => {
            vm.engine
                .note_runtime_call_stats(|s| s.dispatch_method_native += 1);
            let result = crate::runtime::foreign::dispatch_dynamic(vm, idx, args).to_bits();
            if let Some(action) = vm.pending_fiber_action.take() {
                return handle_jit_fiber_action(vm, action);
            }
            result
        }
        Method::Closure(cp) => {
            if let Some(result) =
                try_dispatch_trivial_accessor_fastpath(vm, Method::Closure(cp), args)
            {
                return result;
            }
            vm.engine
                .note_runtime_call_stats(|s| s.dispatch_method_closure += 1);
            let func_id = crate::runtime::engine::FuncId(unsafe { (*(*cp).function).fn_id });
            let fn_idx = func_id.0 as usize;
            if vm.engine.mode != crate::runtime::engine::ExecutionMode::Interpreter {
                let needs_tier_up = vm
                    .engine
                    .jit_code
                    .get(fn_idx)
                    .copied()
                    .unwrap_or(std::ptr::null())
                    .is_null();
                if needs_tier_up {
                    let should_tier_up = vm.engine.record_call(func_id);
                    if should_tier_up {
                        vm.request_tier_up(func_id);
                    }
                }
            }

            if args.len() <= 8 && !jit_disabled() {
                let fn_ptr = vm
                    .engine
                    .jit_code
                    .get(fn_idx)
                    .copied()
                    .unwrap_or(std::ptr::null());
                // With Cranelift, allow non-leaf JIT dispatch — Cranelift
                // handles register allocation and call conventions correctly.
                // The non-cranelift fallback still gates on is_leaf to avoid
                // spill-slot bugs.
                #[cfg(feature = "cranelift")]
                let allow_jit = !fn_ptr.is_null();
                #[cfg(not(feature = "cranelift"))]
                let allow_jit = {
                    let is_leaf = vm.engine.jit_leaf.get(fn_idx).copied().unwrap_or(false);
                    !fn_ptr.is_null() && is_leaf
                };
                if allow_jit {
                    let saved_ctx = read_jit_ctx();
                    vm.engine
                        .note_runtime_call_stats(|s| s.jit_context_save_restore_pairs += 1);
                    let depth = jit_depth();
                    if depth < MAX_JIT_DEPTH {
                        mutate_jit_ctx(|ctx| {
                            // Ensure vm is set — may have been cleared by
                            // a prior context restore if this is a nested call.
                            if ctx.vm.is_null() {
                                ctx.vm = vm as *mut _ as *mut u8;
                            }
                            ctx.current_func_id = unsafe { (*(*cp).function).fn_id } as u64;
                            ctx.closure = cp as *mut u8;
                            ctx.defining_class = defining_class
                                .map(|p| p as *mut u8)
                                .unwrap_or(std::ptr::null_mut());
                        });
                        vm.engine.note_native_entry(func_id);
                        vm.engine.note_native_to_native_call(func_id);
                        set_jit_depth(depth + 1);
                        let result = unsafe { call_jit_with_shadow(vm, fn_ptr, func_id, args) };
                        set_jit_depth(depth);
                        set_jit_context(saved_ctx);
                        return result;
                    }
                }
            }
            call_closure_jit_or_sync(vm, cp, args, defining_class)
        }
        Method::Constructor(cp) => {
            vm.engine
                .note_runtime_call_stats(|s| s.dispatch_method_constructor += 1);
            let class_ptr = args
                .first()
                .and_then(|v| v.as_object())
                .map(|p| p as *mut crate::runtime::object::ObjClass)
                .unwrap_or(std::ptr::null_mut());
            if class_ptr.is_null() {
                return Value::null().to_bits();
            }
            // args[1..] are the user-visible constructor arguments (args[0] is the class)
            vm.call_constructor_sync(class_ptr, cp, &args[1..])
                .to_bits()
        }
    }
}

/// Call a method with 0 extra args. Codegen: `[receiver, method_sym]`
/// On aarch64, wren_call_N uses `#[naked]` wrappers to capture the caller's
/// frame pointer (x29) at zero cost to JIT code. The FP is passed as the
/// LAST argument to the inner function, which pushes it to the thread's
/// JIT frames for GC stack walking.
///
/// # Safety
/// Called only from JIT-compiled code via `CallRuntime`. The receiver and
/// method arguments are NaN-boxed values produced by the JIT.
#[cfg(all(target_arch = "aarch64", feature = "host"))]
#[unsafe(naked)]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub unsafe extern "C" fn wren_call_0(_receiver: u64, _method: u64) -> u64 {
    core::arch::naked_asm!(
        "mov x2, x29",       // pass JIT FP as 3rd arg
        "mov x3, x30",       // pass return address as 4th arg
        "b {inner}",
        inner = sym wren_call_0_inner,
    );
}
#[cfg(all(target_arch = "x86_64", not(windows)))]
#[unsafe(naked)]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub unsafe extern "C" fn wren_call_0(_receiver: u64, _method: u64) -> u64 {
    // SysV: rdi=receiver, rsi=method → inner gets rdx=jit_fp, rcx=ret_addr
    core::arch::naked_asm!(
        "mov rdx, rbp",       // pass JIT FP (caller's RBP) as 3rd arg
        "mov rcx, [rsp]",     // pass return address as 4th arg
        "jmp {inner}",
        inner = sym wren_call_0_inner,
    );
}
#[cfg(all(
    any(
        not(any(target_arch = "aarch64", target_arch = "x86_64")),
        all(target_arch = "x86_64", windows)
    ),
    any(feature = "host", feature = "aot_runtime")
))]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub unsafe extern "C" fn wren_call_0(receiver: u64, method: u64) -> u64 {
    wren_call_0_inner(receiver, method, 0, 0)
}
#[inline(always)]
fn root_at(j: *mut JitThread, idx: usize) -> Value {
    unsafe { (&(*j).roots)[idx] }
}

/// One send issued by compiled code. The receiver and arguments come
/// in registers, invisible to the collector, so they are rooted first and
/// read back from the roots around anything that may allocate. The
/// thread's JIT state is fetched once; the whole send works through it.
#[inline(always)]
fn wren_call_inner<const M: usize>(
    method: u64,
    words: [u64; M],
    jit_fp: u64,
    ret_addr: u64,
) -> u64 {
    let j = jit_state();
    let root_base = unsafe {
        let roots = &mut (*j).roots;
        let base = roots.len();
        roots.extend(words.iter().map(|&w| Value::from_bits(w)));
        base
    };
    let result = match unsafe { vm_at(j) } {
        Some(vm) => {
            vm.engine
                .note_runtime_call_stats(|s| s.wren_call_entries += 1);
            let args: [Value; M] = std::array::from_fn(|i| root_at(j, root_base + i));
            match try_dispatch_call_noframe_fast(vm, j, args[0], method, &args) {
                Frameless::Done(result) => result,
                Frameless::Host(host_fn, context) => {
                    let func_id = unsafe { (*j).ctx.current_func_id } as u32;
                    push_frame_on(j, jit_fp as usize, func_id, ret_addr as usize);
                    let result = host_fn(vm, context, &args).to_bits();
                    let result = match vm.pending_fiber_action.take() {
                        Some(action) => handle_jit_fiber_action(vm, action),
                        None => result,
                    };
                    unsafe { (*j).frames.pop() };
                    result
                }
                Frameless::Miss => {
                    let func_id = unsafe { (*j).ctx.current_func_id } as u32;
                    push_frame_on(j, jit_fp as usize, func_id, ret_addr as usize);
                    let args: [Value; M] = std::array::from_fn(|i| root_at(j, root_base + i));
                    let result = dispatch_call_rooted(vm, j, args[0], method, &args);
                    unsafe { (*j).frames.pop() };
                    result
                }
            }
        }
        None => Value::null().to_bits(),
    };
    unsafe { (*j).roots.truncate(root_base) };
    result
}

extern "C" fn wren_call_0_inner(receiver: u64, method: u64, jit_fp: u64, ret_addr: u64) -> u64 {
    wren_call_inner(method, [receiver], jit_fp, ret_addr)
}

/// # Safety
/// Called only from JIT-compiled code via `CallRuntime`.
#[cfg(all(target_arch = "aarch64", feature = "host"))]
#[unsafe(naked)]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub unsafe extern "C" fn wren_call_1(_receiver: u64, _method: u64, _a0: u64) -> u64 {
    core::arch::naked_asm!(
        "mov x3, x29",
        "mov x4, x30",
        "b {inner}",
        inner = sym wren_call_1_inner,
    );
}
#[cfg(all(target_arch = "x86_64", not(windows)))]
#[unsafe(naked)]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub unsafe extern "C" fn wren_call_1(_receiver: u64, _method: u64, _a0: u64) -> u64 {
    // SysV: rdi=receiver, rsi=method, rdx=a0 → inner gets rcx=jit_fp, r8=ret_addr
    core::arch::naked_asm!(
        "mov rcx, rbp",
        "mov r8, [rsp]",
        "jmp {inner}",
        inner = sym wren_call_1_inner,
    );
}
#[cfg(all(
    any(
        not(any(target_arch = "aarch64", target_arch = "x86_64")),
        all(target_arch = "x86_64", windows)
    ),
    any(feature = "host", feature = "aot_runtime")
))]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub unsafe extern "C" fn wren_call_1(receiver: u64, method: u64, a0: u64) -> u64 {
    wren_call_1_inner(receiver, method, a0, 0, 0)
}
extern "C" fn wren_call_1_inner(
    receiver: u64,
    method: u64,
    a0: u64,
    jit_fp: u64,
    ret_addr: u64,
) -> u64 {
    wren_call_inner(method, [receiver, a0], jit_fp, ret_addr)
}

/// # Safety
/// Called only from JIT-compiled code via `CallRuntime`.
#[cfg(all(target_arch = "aarch64", feature = "host"))]
#[unsafe(naked)]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub unsafe extern "C" fn wren_call_2(_receiver: u64, _method: u64, _a0: u64, _a1: u64) -> u64 {
    core::arch::naked_asm!(
        "mov x4, x29",
        "mov x5, x30",
        "b {inner}",
        inner = sym wren_call_2_inner,
    );
}
#[cfg(all(target_arch = "x86_64", not(windows)))]
#[unsafe(naked)]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub unsafe extern "C" fn wren_call_2(_receiver: u64, _method: u64, _a0: u64, _a1: u64) -> u64 {
    // SysV: rdi=receiver, rsi=method, rdx=a0, rcx=a1 → inner gets r8=jit_fp, r9=ret_addr
    core::arch::naked_asm!(
        "mov r8, rbp",
        "mov r9, [rsp]",
        "jmp {inner}",
        inner = sym wren_call_2_inner,
    );
}
#[cfg(all(
    any(
        not(any(target_arch = "aarch64", target_arch = "x86_64")),
        all(target_arch = "x86_64", windows)
    ),
    any(feature = "host", feature = "aot_runtime")
))]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub unsafe extern "C" fn wren_call_2(receiver: u64, method: u64, a0: u64, a1: u64) -> u64 {
    wren_call_2_inner(receiver, method, a0, a1, 0, 0)
}
extern "C" fn wren_call_2_inner(
    receiver: u64,
    method: u64,
    a0: u64,
    a1: u64,
    jit_fp: u64,
    ret_addr: u64,
) -> u64 {
    wren_call_inner(method, [receiver, a0, a1], jit_fp, ret_addr)
}

/// # Safety
/// Called only from JIT-compiled code via `CallRuntime`.
#[cfg(all(target_arch = "aarch64", feature = "host"))]
#[unsafe(naked)]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub unsafe extern "C" fn wren_call_3(
    _receiver: u64,
    _method: u64,
    _a0: u64,
    _a1: u64,
    _a2: u64,
) -> u64 {
    core::arch::naked_asm!(
        "mov x5, x29",
        "mov x6, x30",
        "b {inner}",
        inner = sym wren_call_3_inner,
    );
}
#[cfg(all(target_arch = "x86_64", not(windows)))]
#[unsafe(naked)]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub unsafe extern "C" fn wren_call_3(
    _receiver: u64,
    _method: u64,
    _a0: u64,
    _a1: u64,
    _a2: u64,
) -> u64 {
    // SysV: rdi=recv, rsi=method, rdx=a0, rcx=a1, r8=a2. The inner
    // also wants jit_fp (arg 6) and ret_addr (arg 7); we place them
    // in r9 and on the stack respectively.
    //
    // Naked is required: a regular Rust wrapper that reads `[rbp]`
    // via inline asm is unsafe under `preserve_frame_pointers=false`
    // Cranelift codegen — the compiler doesn't know the asm needs
    // rbp to hold the caller's frame value, so it uses rbp as a
    // general-purpose register to spill an arg, and the asm reads
    // garbage. The result on Linux x86_64 is a SIGSEGV inside
    // wren_call_3 the first time JIT'd code dispatches a 3-arg
    // method whose receiver register happened to be a NaN-boxed
    // Value (a high-bit address that's kernel-only on Linux).
    //
    // Stack on entry: [rsp]=original ret_addr, rsp is 8-aligned
    // (16-aligned just before JIT's `call wren_call_3`, then
    // `call` pushed 8 bytes).
    //
    // Layout after `push rax` + `call inner`:
    //   [rsp+0]  = post_call_ret_addr (auto-pushed by `call`)
    //   [rsp+8]  = original ret_addr (arg 7, from our push)
    // — matches SysV stack-arg conventions for a 7-arg fn.
    core::arch::naked_asm!(
        "mov r9, rbp",          // arg 6: jit_fp = caller's rbp
        "mov rax, [rsp]",       // rax = original ret addr
        "push rax",             // arg 7 on stack; also aligns rsp 8 → 16
        "call {inner}",
        "add rsp, 8",           // pop the pushed ret_addr
        "ret",                  // return using original ret_addr at [rsp]
        inner = sym wren_call_3_inner,
    );
}
#[cfg(all(
    any(
        not(any(target_arch = "aarch64", target_arch = "x86_64")),
        all(target_arch = "x86_64", windows)
    ),
    any(feature = "host", feature = "aot_runtime")
))]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub unsafe extern "C" fn wren_call_3(receiver: u64, method: u64, a0: u64, a1: u64, a2: u64) -> u64 {
    wren_call_3_inner(receiver, method, a0, a1, a2, 0, 0)
}
extern "C" fn wren_call_3_inner(
    receiver: u64,
    method: u64,
    a0: u64,
    a1: u64,
    a2: u64,
    jit_fp: u64,
    ret_addr: u64,
) -> u64 {
    wren_call_inner(method, [receiver, a0, a1, a2], jit_fp, ret_addr)
}

/// # Safety
/// Called only from JIT-compiled code via `CallRuntime`.
#[cfg(all(target_arch = "aarch64", feature = "host"))]
#[unsafe(naked)]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub unsafe extern "C" fn wren_call_4(
    _receiver: u64,
    _method: u64,
    _a0: u64,
    _a1: u64,
    _a2: u64,
    _a3: u64,
) -> u64 {
    core::arch::naked_asm!(
        "mov x6, x29",
        "mov x7, x30",
        "b {inner}",
        inner = sym wren_call_4_inner,
    );
}
#[cfg(all(target_arch = "x86_64", not(windows)))]
#[unsafe(naked)]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub unsafe extern "C" fn wren_call_4(
    _receiver: u64,
    _method: u64,
    _a0: u64,
    _a1: u64,
    _a2: u64,
    _a3: u64,
) -> u64 {
    // Same naked-asm fix as wren_call_3 — see that function for
    // the rationale. Here all 6 SysV register slots are used by
    // (recv, method, a0, a1, a2, a3), so jit_fp and ret_addr both
    // go on the stack as args 7 and 8.
    //
    // Stack-arg ordering: caller pushes higher-index args first.
    // Layout post-`call inner`:
    //   [rsp+0]  = post_call_ret_addr
    //   [rsp+8]  = arg 7 = jit_fp
    //   [rsp+16] = arg 8 = ret_addr
    //   [rsp+24] = padding (preserves 16-byte alignment)
    core::arch::naked_asm!(
        "mov r10, rbp",         // r10 = jit_fp (will become arg 7)
        "mov rax, [rsp]",       // rax = original ret addr (arg 8)
        "sub rsp, 8",           // align: 8-aligned → 0-aligned, 2 pushes will land 0-aligned
        "push rax",             // arg 8 (ret_addr)
        "push r10",             // arg 7 (jit_fp)
        "call {inner}",
        "add rsp, 24",          // 8 pad + 8 + 8 args
        "ret",
        inner = sym wren_call_4_inner,
    );
}
#[cfg(all(
    any(
        not(any(target_arch = "aarch64", target_arch = "x86_64")),
        all(target_arch = "x86_64", windows)
    ),
    any(feature = "host", feature = "aot_runtime")
))]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub unsafe extern "C" fn wren_call_4(
    receiver: u64,
    method: u64,
    a0: u64,
    a1: u64,
    a2: u64,
    a3: u64,
) -> u64 {
    wren_call_4_inner(receiver, method, a0, a1, a2, a3, 0, 0)
}
/// Call helper for arity 5..8. Same shape as wren_call_4_inner —
/// roots every arg, dispatches via the standard helper, pops
/// frame on the way back. Skipping the naked-asm jit_fp/ret_addr
/// capture used by call_0..4: with all 8 aarch64 register slots
/// already filled by user args + recv + method + jit_fp, the
/// 9th arg (ret_addr) would have to spill — and stack maps are
/// gated off by default anyway. Pass 0/0 here, matching the
/// non-mainstream-arch fallback the lower-arity helpers already
/// take.
fn wren_call_n_inner(receiver: u64, method: u64, args_in: &[u64]) -> u64 {
    let j = jit_state();
    let root_base = unsafe {
        let roots = &mut (*j).roots;
        let base = roots.len();
        roots.push(Value::from_bits(receiver));
        roots.extend(args_in.iter().map(|&w| Value::from_bits(w)));
        base
    };
    let n = args_in.len() + 1;
    let read = |i: usize| root_at(j, root_base + i);
    let result = match unsafe { vm_at(j) } {
        Some(vm) => {
            vm.engine
                .note_runtime_call_stats(|s| s.wren_call_entries += 1);
            let args: Vec<Value> = (0..n).map(read).collect();
            match try_dispatch_call_noframe_fast(vm, j, args[0], method, &args) {
                Frameless::Done(result) => result,
                Frameless::Host(host_fn, context) => {
                    let func_id = unsafe { (*j).ctx.current_func_id } as u32;
                    push_frame_on(j, 0, func_id, 0);
                    let result = host_fn(vm, context, &args).to_bits();
                    let result = match vm.pending_fiber_action.take() {
                        Some(action) => handle_jit_fiber_action(vm, action),
                        None => result,
                    };
                    unsafe { (*j).frames.pop() };
                    result
                }
                Frameless::Miss => {
                    let func_id = unsafe { (*j).ctx.current_func_id } as u32;
                    push_frame_on(j, 0, func_id, 0);
                    let args: Vec<Value> = (0..n).map(read).collect();
                    let result = dispatch_call_rooted(vm, j, args[0], method, &args);
                    unsafe { (*j).frames.pop() };
                    result
                }
            }
        }
        None => Value::null().to_bits(),
    };
    unsafe { (*j).roots.truncate(root_base) };
    result
}

#[cfg(any(feature = "host", feature = "aot_runtime"))]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_call_5(
    receiver: u64,
    method: u64,
    a0: u64,
    a1: u64,
    a2: u64,
    a3: u64,
    a4: u64,
) -> u64 {
    wren_call_n_inner(receiver, method, &[a0, a1, a2, a3, a4])
}

#[cfg(any(feature = "host", feature = "aot_runtime"))]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_call_6(
    receiver: u64,
    method: u64,
    a0: u64,
    a1: u64,
    a2: u64,
    a3: u64,
    a4: u64,
    a5: u64,
) -> u64 {
    wren_call_n_inner(receiver, method, &[a0, a1, a2, a3, a4, a5])
}

#[cfg(any(feature = "host", feature = "aot_runtime"))]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_call_7(
    receiver: u64,
    method: u64,
    a0: u64,
    a1: u64,
    a2: u64,
    a3: u64,
    a4: u64,
    a5: u64,
    a6: u64,
) -> u64 {
    wren_call_n_inner(receiver, method, &[a0, a1, a2, a3, a4, a5, a6])
}

#[cfg(any(feature = "host", feature = "aot_runtime"))]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_call_8(
    receiver: u64,
    method: u64,
    a0: u64,
    a1: u64,
    a2: u64,
    a3: u64,
    a4: u64,
    a5: u64,
    a6: u64,
    a7: u64,
) -> u64 {
    wren_call_n_inner(receiver, method, &[a0, a1, a2, a3, a4, a5, a6, a7])
}

/// Variadic dispatch helper for `> 8`-arg call sites. Cranelift
/// emits a `[u64; n]` stack buffer at the call site, fills it
/// with arg bits, and routes through here so codegen doesn't have
/// to mint a fresh `wren_call_N` per arity. AOT bodies with deep
/// receiver-heavy method calls (`@hatch:gpu`'s pipeline /
/// render-pass setup goes up to 12 + receiver) used to abort the
/// build with "Call with arity N not supported by JIT (max 8)".
///
/// # Safety
/// `args_ptr` must point to `count` `u64`s.
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub unsafe extern "C" fn wren_call_dynamic(
    receiver: u64,
    method: u64,
    count: u64,
    args_ptr: *const u64,
) -> u64 {
    if args_ptr.is_null() {
        return wren_call_n_inner(receiver, method, &[]);
    }
    let slice = unsafe { std::slice::from_raw_parts(args_ptr, count as usize) };
    wren_call_n_inner(receiver, method, slice)
}

extern "C" fn wren_call_4_inner(
    receiver: u64,
    method: u64,
    a0: u64,
    a1: u64,
    a2: u64,
    a3: u64,
    jit_fp: u64,
    ret_addr: u64,
) -> u64 {
    wren_call_inner(method, [receiver, a0, a1, a2, a3], jit_fp, ret_addr)
}

// ---------------------------------------------------------------------------
// Known-function dispatch: skip method lookup, call by FuncId directly.
// Used by CallKnownFunc for devirtualized call sites. ~5x faster than
// wren_call_N because we skip: method_and_ic decode, IC check, method
// cache lookup, class hierarchy walk. Just: jit_code[func_id] → call.
// ---------------------------------------------------------------------------

/// `packed` = func_id (lower 32) | method_sym (upper 32)
fn wren_known_call_inner(packed: u64, args: &[Value]) -> u64 {
    let func_id = (packed & 0xFFFF_FFFF) as u32;
    let method_raw = (packed >> 32) as u32;
    let vm = unsafe { vm_ref() };
    let vm = match vm {
        Some(v) => v,
        None => return Value::null().to_bits(),
    };
    let fid = func_id as usize;
    let method_sym = crate::intern::SymbolId::from_raw(method_raw);
    // Same staleness window as `wren_ic_call_inner`: the JIT'd
    // caller's args arrive in CPU registers and are invisible to
    // the GC. Any allocation between here and the eventual
    // `call_jit_with_shadow` (method_cache miss path can allocate;
    // the callee freely allocates) can promote the underlying
    // objects out from under us. Push every arg as a root, read
    // them back fresh at each dispatch site.
    let root_base = jit_roots_snapshot_len();
    for &v in args {
        push_jit_root(v);
    }
    let arg_count = args.len();
    let load_args = || -> smallvec::SmallVec<[Value; 5]> {
        (0..arg_count).map(|i| jit_root_at(root_base + i)).collect()
    };
    let recv0 = || jit_root_at(root_base);

    let result = (|| {
        // Verify the receiver's actual method is FuncId(func_id).
        // Devirtualization is speculative — for polymorphic call
        // sites, the receiver class may differ from the one
        // observed at compile time, so we must check before
        // calling the pre-selected FuncId.
        let class = vm.class_of(recv0());
        let actual_method = vm
            .method_cache
            .lookup(class, method_sym)
            .or_else(|| unsafe { find_method_with_class(class, method_sym) });

        let is_match = match actual_method {
            Some((crate::runtime::object::Method::Closure(cp), _)) => unsafe {
                (*(*cp).function).fn_id == func_id
            },
            _ => false,
        };

        if !is_match {
            // Polymorphic miss — fall back to full dispatch.
            let collected = load_args();
            return dispatch_call(recv0(), method_sym.index() as u64, &collected);
        }

        let fid_obj = crate::runtime::engine::FuncId(func_id);
        let jit_ptr = vm
            .engine
            .jit_code
            .get(fid)
            .copied()
            .unwrap_or(std::ptr::null());

        if !jit_ptr.is_null() && arg_count <= 4 {
            let saved_ctx = read_jit_ctx();
            mutate_jit_ctx(|ctx| {
                ctx.current_func_id = func_id as u64;
            });
            let depth = jit_depth();
            if depth < MAX_JIT_DEPTH {
                set_jit_depth(depth + 1);
                let collected = load_args();
                let result = unsafe { call_jit_with_shadow(vm, jit_ptr, fid_obj, &collected) };
                set_jit_depth(depth);
                set_jit_context(saved_ctx);
                return result;
            }
            set_jit_context(saved_ctx);
        }

        // Callee not JIT'd yet — full dispatch via method symbol.
        let collected = load_args();
        dispatch_call(recv0(), method_sym.index() as u64, &collected)
    })();
    jit_roots_restore_len(root_base);
    result
}

/// Known call with 0 extra args: (func_id, recv) -> result
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_known_call_0(func_id: u64, recv: u64) -> u64 {
    wren_known_call_inner(func_id, &[Value::from_bits(recv)])
}

/// Fast variant: caller (Cranelift) has already verified the class matches.
/// Skips class_of + method_cache lookup. Just loads jit_code[func_id] and
/// does the context swap + call.
#[inline]
fn wren_known_call_nocheck_inner(packed: u64, args: &[Value]) -> u64 {
    let func_id = (packed & 0xFFFF_FFFF) as u32;
    let method_raw = (packed >> 32) as u32;
    // One thread-local fetch for the whole call.
    let j = jit_state();
    let vm_ptr = unsafe { (*j).ctx.vm } as *mut crate::runtime::vm::VM;
    if vm_ptr.is_null() {
        return Value::null().to_bits();
    }
    let vm = unsafe { &mut *vm_ptr };
    let fid = func_id as usize;
    let fid_obj = crate::runtime::engine::FuncId(func_id);
    let jit_ptr = vm
        .engine
        .jit_code
        .get(fid)
        .copied()
        .unwrap_or(std::ptr::null());
    // The collector scans this frame, so the arguments need no root
    // entries.
    if !jit_ptr.is_null() && args.len() <= 4 {
        let depth = unsafe { (*j).depth };
        if depth < MAX_JIT_DEPTH {
            // A leaf callee never looks up an IC table, so its
            // current_func_id need not be set.
            let is_leaf = vm.engine.jit_leaf.get(fid).copied().unwrap_or(false);
            let saved_func_id = unsafe { (*j).ctx.current_func_id };
            if !is_leaf {
                unsafe { (*j).ctx.current_func_id = func_id as u64 };
            }
            unsafe { (*j).depth = depth + 1 };
            let result = unsafe { call_jit_with_shadow_st(j, vm, jit_ptr, fid_obj, args) };
            unsafe {
                (*j).depth = depth;
                if !is_leaf {
                    (*j).ctx.current_func_id = saved_func_id;
                }
            }
            return result;
        }
    }

    // Callee not compiled yet → fall back to full dispatch.
    let recv = args.first().copied().unwrap_or(Value::null());
    let method_sym = crate::intern::SymbolId::from_raw(method_raw);
    dispatch_call(recv, method_sym.index() as u64, args)
}

/// `Class.new(args)` from a call site whose cache resolved the class:
/// allocate the instance and run the compiled initialiser on it, the
/// way a method is reached through `wren_known_call_N_nocheck`. Falls
/// back to full dispatch on the class when the initialiser is not
/// compiled. `packed` is the initialiser's function id in the low word
/// and the call's method symbol in the high word.
fn wren_construct_inner(packed: u64, class_bits: u64, args: &[u64]) -> u64 {
    let func_id = (packed & 0xFFFF_FFFF) as u32;
    let method_raw = (packed >> 32) as u32;
    let j = jit_state();
    let vm_ptr = unsafe { (*j).ctx.vm } as *mut crate::runtime::vm::VM;
    if vm_ptr.is_null() {
        return Value::null().to_bits();
    }
    let vm = unsafe { &mut *vm_ptr };
    let fid = func_id as usize;
    let fid_obj = crate::runtime::engine::FuncId(func_id);
    let jit_ptr = vm
        .engine
        .jit_code
        .get(fid)
        .copied()
        .unwrap_or(std::ptr::null());
    let depth = unsafe { (*j).depth };
    let class_val = Value::from_bits(class_bits);
    let class_ptr = class_val
        .as_object()
        .map(|p| p as *mut ObjClass)
        .unwrap_or(std::ptr::null_mut());
    if jit_ptr.is_null() || class_ptr.is_null() || args.len() > 3 || depth >= MAX_JIT_DEPTH {
        let mut all: smallvec::SmallVec<[Value; 5]> = smallvec::SmallVec::new();
        all.push(class_val);
        all.extend(args.iter().map(|a| Value::from_bits(*a)));
        return dispatch_call(class_val, method_raw as u64, &all);
    }
    // The conservative collector scans this frame; the initialiser
    // runs with the instance and arguments pinned here.
    let inst = vm.gc.alloc_instance(class_ptr);
    let inst_bits = unsafe { finish_alloc(vm, Value::object(inst as *mut u8)) };
    let mut call_args: smallvec::SmallVec<[Value; 5]> = smallvec::SmallVec::new();
    call_args.push(Value::from_bits(inst_bits));
    call_args.extend(args.iter().map(|a| Value::from_bits(*a)));
    let saved_func_id = unsafe { (*j).ctx.current_func_id };
    let saved_class = unsafe { (*j).ctx.defining_class };
    unsafe {
        (*j).ctx.current_func_id = func_id as u64;
        (*j).ctx.defining_class = class_ptr as *mut u8;
        (*j).depth = depth + 1;
    }
    let _ = unsafe { call_jit_with_shadow_st(j, vm, jit_ptr, fid_obj, &call_args) };
    unsafe {
        (*j).depth = depth;
        (*j).ctx.current_func_id = saved_func_id;
        (*j).ctx.defining_class = saved_class;
    }
    inst_bits
}

#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_construct_0(packed: u64, class: u64) -> u64 {
    wren_construct_inner(packed, class, &[])
}
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_construct_1(packed: u64, class: u64, a0: u64) -> u64 {
    wren_construct_inner(packed, class, &[a0])
}
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_construct_2(packed: u64, class: u64, a0: u64, a1: u64) -> u64 {
    wren_construct_inner(packed, class, &[a0, a1])
}
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_construct_3(packed: u64, class: u64, a0: u64, a1: u64, a2: u64) -> u64 {
    wren_construct_inner(packed, class, &[a0, a1, a2])
}

#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_known_call_0_nocheck(packed: u64, recv: u64) -> u64 {
    wren_known_call_nocheck_inner(packed, &[Value::from_bits(recv)])
}
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_known_call_1_nocheck(packed: u64, recv: u64, a0: u64) -> u64 {
    wren_known_call_nocheck_inner(packed, &[Value::from_bits(recv), Value::from_bits(a0)])
}
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_known_call_2_nocheck(packed: u64, recv: u64, a0: u64, a1: u64) -> u64 {
    wren_known_call_nocheck_inner(
        packed,
        &[
            Value::from_bits(recv),
            Value::from_bits(a0),
            Value::from_bits(a1),
        ],
    )
}
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_known_call_3_nocheck(
    packed: u64,
    recv: u64,
    a0: u64,
    a1: u64,
    a2: u64,
) -> u64 {
    wren_known_call_nocheck_inner(
        packed,
        &[
            Value::from_bits(recv),
            Value::from_bits(a0),
            Value::from_bits(a1),
            Value::from_bits(a2),
        ],
    )
}

/// Known call with 1 extra arg: (func_id, recv, a0) -> result
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_known_call_1(func_id: u64, recv: u64, a0: u64) -> u64 {
    wren_known_call_inner(func_id, &[Value::from_bits(recv), Value::from_bits(a0)])
}

/// Known call with 2 extra args: (func_id, recv, a0, a1) -> result
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_known_call_2(func_id: u64, recv: u64, a0: u64, a1: u64) -> u64 {
    wren_known_call_inner(
        func_id,
        &[
            Value::from_bits(recv),
            Value::from_bits(a0),
            Value::from_bits(a1),
        ],
    )
}

/// Known call with 3 extra args
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_known_call_3(func_id: u64, recv: u64, a0: u64, a1: u64, a2: u64) -> u64 {
    wren_known_call_inner(
        func_id,
        &[
            Value::from_bits(recv),
            Value::from_bits(a0),
            Value::from_bits(a1),
            Value::from_bits(a2),
        ],
    )
}

// ---------------------------------------------------------------------------
// Indirect IC dispatch: lightweight JIT-to-JIT calls via IC entry.
// These are ~10x faster than wren_call_N because they skip method lookup.
// Called from Cranelift JIT code when the indirect IC class check passes.
// The IC entry provides the jit_ptr; these functions set up minimal context
// (func_id + closure) and do a direct call.
// ---------------------------------------------------------------------------

/// IC call with 0 extra args. Signature: (ic_ptr, recv) -> result
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_ic_call_0(ic_ptr: u64, recv: u64) -> u64 {
    wren_ic_call_inner(ic_ptr, &[recv])
}

/// IC call with 1 extra arg. Signature: (ic_ptr, recv, a0) -> result
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_ic_call_1(ic_ptr: u64, recv: u64, a0: u64) -> u64 {
    wren_ic_call_inner(ic_ptr, &[recv, a0])
}

/// IC call with 2 extra args. Signature: (ic_ptr, recv, a0, a1) -> result
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_ic_call_2(ic_ptr: u64, recv: u64, a0: u64, a1: u64) -> u64 {
    wren_ic_call_inner(ic_ptr, &[recv, a0, a1])
}

/// IC call with 3 extra args.
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_ic_call_3(ic_ptr: u64, recv: u64, a0: u64, a1: u64, a2: u64) -> u64 {
    wren_ic_call_inner(ic_ptr, &[recv, a0, a1, a2])
}

fn wren_ic_call_inner(ic_ptr_raw: u64, args: &[u64]) -> u64 {
    let ic_ptr = ic_ptr_raw as *mut crate::mir::bytecode::CallSiteIC;
    let ic = unsafe { (*ic_ptr).snapshot() }.unwrap_or_default();
    let jit_ptr = ic.jit_ptr;
    // Root every inbound arg before any dispatch. Args arrive in
    // CPU registers from the JIT'd caller, which means the GC has
    // no way to see them — if the callee allocates and triggers a
    // collection, the underlying objects get freed / relocated and
    // we hand the callee a stale Value. `wren_call_N_inner` already
    // does this for the non-IC path; mirror it here.
    let root_base = jit_roots_snapshot_len();
    for &raw in args {
        push_jit_root(Value::from_bits(raw));
    }
    // Helper: snapshot the current root-window's bits. Used at
    // every "we're about to dispatch" point so callers see the
    // GC-forwarded values, not the stale `args` slice we were
    // handed by the JIT'd caller.
    let collect_args = || -> smallvec::SmallVec<[Value; 5]> {
        (0..args.len())
            .map(|i| jit_root_at(root_base + i))
            .collect()
    };
    let result = (|| {
        if jit_ptr.is_null() {
            // The JIT'd call site reached us with an unbound IC. This
            // shouldn't happen if the codegen pre-checks `ic.class ==
            // recv.class` before dispatching, but a previous call into
            // this IC may have cleared it (e.g. live==null branch
            // below) and the codegen doesn't always re-read the class
            // word between callsites. Returning `Value::null()` here
            // would silently corrupt the JIT'd caller's stack — so
            // route through the cached closure if one's still around,
            // or abort the fiber with a clear message rather than
            // letting the caller crash on a null receiver downstream.
            let closure = ic.closure as *mut ObjClosure;
            if !closure.is_null()
                && let Some(vm) = unsafe { vm_ref() }
            {
                return call_closure_jit_or_sync(vm, closure, &collect_args(), None);
            }
            // No usable dispatch info — surface to stderr so the
            // failure is at least visible. (`runtime_error` lives on
            // VM but isn't reachable from this calling convention
            // without re-borrowing.)
            eprintln!(
                "wren_ic_call: empty IC (kind={}, class=0x{:x}) — call site \
             lost dispatch info between codegen and runtime",
                ic.kind, ic.class
            );
            return Value::null().to_bits();
        }
        // Leaf (kind=1) fast path: `is_mir_inline_safe` guarantees the callee
        // makes no outbound calls, touches no upvalues and never reads
        // current_func_id / closure / defining_class. The caller's ctx.vm /
        // module_vars / jit_code_base are already correct, so we can skip the
        // full save/restore and jit_depth bump — for tight dispatch loops this
        // removes ~6 TLS roundtrips and a 48-byte copy per call.
        //
        // Revalidate `jit_ptr` against the live `engine.jit_code` slot
        // before the transmute — a tier-up since this IC was installed
        // may have freed / relocated the code blob, and reading the
        // stale pointer PAC-faults on arm64.
        //
        // On a refresh hit (live exists but moved), update the IC's
        // pointer in-place and call the new address — the IC's other
        // fields (class, func_id, kind) are still valid because it's
        // the same function, just relocated. Returning `Value::null()`
        // here would be incorrect: the JIT'd caller has no way to tell
        // a sentinel "miss" from a real null return, so it'd treat the
        // sentinel as the call's result.
        if ic.kind == 1 {
            let mut call_ptr = jit_ptr;
            let mut closure_for_fallback: *mut ObjClosure = std::ptr::null_mut();
            if let Some(vm) = unsafe { vm_ref() } {
                let live = vm
                    .engine
                    .jit_code
                    .get(ic.func_id as usize)
                    .copied()
                    .unwrap_or(std::ptr::null());
                if live.is_null() {
                    // Function genuinely lost its JIT code (eviction or
                    // interpreter-mode run). The kind=1 IC still has the
                    // closure pointer, so dispatch through
                    // `call_closure_jit_or_sync` — that's the canonical
                    // path that picks the right backend (re-JIT if
                    // re-tiered, bytecode interpret otherwise). Clear
                    // the kind=1 entry first so subsequent visits go
                    // through the slow path on the JIT side too. We
                    // reset the closure_for_fallback pointer outside the
                    // borrow so we don't double-borrow `vm` on the call.
                    closure_for_fallback = ic.closure as *mut ObjClosure;
                    unsafe { (*ic_ptr).clear() };
                } else if live != jit_ptr {
                    let mut fresh = ic;
                    fresh.jit_ptr = live;
                    unsafe { (*ic_ptr).store(fresh) };
                    call_ptr = live;
                }
            }
            if !closure_for_fallback.is_null() {
                // Outside the `if let Some(vm)` borrow above so we can
                // call `call_closure_jit_or_sync`, which itself takes
                // `&mut vm`. Re-fetch the VM here.
                if let Some(vm) = unsafe { vm_ref() } {
                    return call_closure_jit_or_sync(
                        vm,
                        closure_for_fallback,
                        &collect_args(),
                        None,
                    );
                }
                return Value::null().to_bits();
            }
            return unsafe { call_jit_with_shadow_raw(call_ptr, &collect_args()) };
        }
        // Same revalidation as the kind=1 fast path — a tier-up after
        // IC install could relocate the code blob and the cached
        // pointer would PAC-fault on dereference (or point at
        // unrelated bytes and produce silent miscompiles).
        //
        // On `live == null` (function lost its JIT code entirely)
        // dispatch through `call_closure_jit_or_sync` with the
        // cached closure. On `live != jit_ptr` refresh the IC and
        // call the new address.
        let mut call_ptr = jit_ptr;
        let mut closure_for_fallback: *mut ObjClosure = std::ptr::null_mut();
        if let Some(vm) = unsafe { vm_ref() } {
            let live = vm
                .engine
                .jit_code
                .get(ic.func_id as usize)
                .copied()
                .unwrap_or(std::ptr::null());
            if live.is_null() {
                closure_for_fallback = ic.closure as *mut ObjClosure;
                unsafe { (*ic_ptr).clear() };
            } else if live != jit_ptr {
                let mut fresh = ic;
                fresh.jit_ptr = live;
                unsafe { (*ic_ptr).store(fresh) };
                call_ptr = live;
            }
        }
        if !closure_for_fallback.is_null() {
            if let Some(vm) = unsafe { vm_ref() } {
                return call_closure_jit_or_sync(vm, closure_for_fallback, &collect_args(), None);
            }
            return Value::null().to_bits();
        }
        // Save and set context for the callee.
        let saved_ctx = read_jit_ctx();
        mutate_jit_ctx(|ctx| {
            ctx.current_func_id = ic.func_id;
            ctx.closure = ic.closure as *mut u8;
        });
        let depth = jit_depth();
        set_jit_depth(depth + 1);
        let result = unsafe { call_jit_with_shadow_raw(call_ptr, &collect_args()) };
        set_jit_depth(depth);
        set_jit_context(saved_ctx);
        result
    })();
    jit_roots_restore_len(root_base);
    result
}

/// Raw call into JIT code without VM reference (for IC dispatch).
#[inline(always)]
unsafe fn call_jit_with_shadow_raw(fn_ptr: *const u8, args: &[Value]) -> u64 {
    let j = jit_state();
    let mut link = EntryLink::new();
    unsafe { enter_link(j, &mut link) };
    let result = unsafe { call_jit_with_shadow_raw_inner(fn_ptr, args) };
    unsafe { leave_link(j, &link) };
    result
}

#[inline(always)]
unsafe fn call_jit_with_shadow_raw_inner(fn_ptr: *const u8, args: &[Value]) -> u64 {
    unsafe {
        match args.len() {
            0 => {
                let f: extern "C" fn() -> u64 = std::mem::transmute(fn_ptr);
                f()
            }
            1 => {
                let f: extern "C" fn(u64) -> u64 = std::mem::transmute(fn_ptr);
                f(args[0].to_bits())
            }
            2 => {
                let f: extern "C" fn(u64, u64) -> u64 = std::mem::transmute(fn_ptr);
                f(args[0].to_bits(), args[1].to_bits())
            }
            3 => {
                let f: extern "C" fn(u64, u64, u64) -> u64 = std::mem::transmute(fn_ptr);
                f(args[0].to_bits(), args[1].to_bits(), args[2].to_bits())
            }
            4 => {
                let f: extern "C" fn(u64, u64, u64, u64) -> u64 = std::mem::transmute(fn_ptr);
                f(
                    args[0].to_bits(),
                    args[1].to_bits(),
                    args[2].to_bits(),
                    args[3].to_bits(),
                )
            }
            5 => {
                let f: extern "C" fn(u64, u64, u64, u64, u64) -> u64 = std::mem::transmute(fn_ptr);
                f(
                    args[0].to_bits(),
                    args[1].to_bits(),
                    args[2].to_bits(),
                    args[3].to_bits(),
                    args[4].to_bits(),
                )
            }
            6 => {
                let f: extern "C" fn(u64, u64, u64, u64, u64, u64) -> u64 =
                    std::mem::transmute(fn_ptr);
                f(
                    args[0].to_bits(),
                    args[1].to_bits(),
                    args[2].to_bits(),
                    args[3].to_bits(),
                    args[4].to_bits(),
                    args[5].to_bits(),
                )
            }
            7 => {
                let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64) -> u64 =
                    std::mem::transmute(fn_ptr);
                f(
                    args[0].to_bits(),
                    args[1].to_bits(),
                    args[2].to_bits(),
                    args[3].to_bits(),
                    args[4].to_bits(),
                    args[5].to_bits(),
                    args[6].to_bits(),
                )
            }
            _ => {
                let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64) -> u64 =
                    std::mem::transmute(fn_ptr);
                f(
                    args[0].to_bits(),
                    args[1].to_bits(),
                    args[2].to_bits(),
                    args[3].to_bits(),
                    args[4].to_bits(),
                    args[5].to_bits(),
                    args[6].to_bits(),
                    args[7].to_bits(),
                )
            }
        }
    }
}

fn call_static_self_inner(extra_args: &[u64]) -> u64 {
    let vm = unsafe { vm_ref() };
    let vm = match vm {
        Some(v) => v,
        None => return Value::null().to_bits(),
    };
    let ctx = read_jit_ctx();
    let closure_ptr = ctx.closure as *mut ObjClosure;
    let defining_class = ctx.defining_class as *mut crate::runtime::object::ObjClass;
    if closure_ptr.is_null() || defining_class.is_null() {
        return Value::null().to_bits();
    }

    let class_val = Value::object(defining_class as *mut u8);
    match extra_args.len() {
        0 => call_closure_jit_or_sync(vm, closure_ptr, &[class_val], Some(defining_class)),
        1 => call_closure_jit_or_sync(
            vm,
            closure_ptr,
            &[class_val, Value::from_bits(extra_args[0])],
            Some(defining_class),
        ),
        2 => call_closure_jit_or_sync(
            vm,
            closure_ptr,
            &[
                class_val,
                Value::from_bits(extra_args[0]),
                Value::from_bits(extra_args[1]),
            ],
            Some(defining_class),
        ),
        3 => call_closure_jit_or_sync(
            vm,
            closure_ptr,
            &[
                class_val,
                Value::from_bits(extra_args[0]),
                Value::from_bits(extra_args[1]),
                Value::from_bits(extra_args[2]),
            ],
            Some(defining_class),
        ),
        _ => call_closure_jit_or_sync(
            vm,
            closure_ptr,
            &[
                class_val,
                Value::from_bits(extra_args[0]),
                Value::from_bits(extra_args[1]),
                Value::from_bits(extra_args[2]),
                Value::from_bits(extra_args[3]),
            ],
            Some(defining_class),
        ),
    }
}

#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_call_static_self_0() -> u64 {
    call_static_self_inner(&[])
}

#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_call_static_self_1(a0: u64) -> u64 {
    call_static_self_inner(&[a0])
}

#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_call_static_self_2(a0: u64, a1: u64) -> u64 {
    call_static_self_inner(&[a0, a1])
}

#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_call_static_self_3(a0: u64, a1: u64, a2: u64) -> u64 {
    call_static_self_inner(&[a0, a1, a2])
}

#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_call_static_self_4(a0: u64, a1: u64, a2: u64, a3: u64) -> u64 {
    call_static_self_inner(&[a0, a1, a2, a3])
}

/// Internal: dispatch a super call. Walks to superclass and dispatches method.
fn dispatch_super_call(recv: Value, method_sym: crate::intern::SymbolId, args: &[Value]) -> u64 {
    with_rooted_args(args, |args| {
        let recv = args.first().copied().unwrap_or(recv);
        dispatch_super_call_rooted(recv, method_sym, args)
    })
}

fn dispatch_super_call_rooted(
    recv: Value,
    method_sym: crate::intern::SymbolId,
    args: &[Value],
) -> u64 {
    let vm = unsafe { vm_ref() };
    let vm = match vm {
        Some(v) => v,
        None => return Value::null().to_bits(),
    };

    // Use defining_class from JIT context to find the correct superclass.
    // class_of(recv) gives the runtime class (e.g. EqualityConstraint when
    // super() is called from BinaryConstraint.new), which is wrong — it would
    // look up EqualityConstraint.superclass = BinaryConstraint instead of
    // BinaryConstraint.superclass = Constraint.
    let ctx = read_jit_ctx();
    let defining_class = if !ctx.defining_class.is_null() {
        ctx.defining_class as *mut crate::runtime::object::ObjClass
    } else {
        // Fallback: try the fiber's current frame's defining_class
        let fiber = vm.fiber;
        if !fiber.is_null() {
            unsafe {
                (*fiber)
                    .mir_frames
                    .last()
                    .and_then(|f| f.defining_class)
                    .unwrap_or_else(|| vm.class_of(recv))
            }
        } else {
            vm.class_of(recv)
        }
    };

    let superclass = unsafe { (*defining_class).superclass };
    if superclass.is_null() {
        return Value::null().to_bits();
    }

    // Walk the superclass hierarchy to find the method and its defining class.
    // Constructors are registered under "static:new(_)" etc., but super calls
    // use the bare method symbol "new(_)".
    let lookup = unsafe {
        find_method_with_class(superclass, method_sym).or_else(|| {
            let method_name = vm.interner.resolve(method_sym);
            let static_name = format!("static:{}", method_name);
            let static_sym = vm.interner.intern(&static_name);
            find_method_with_class(superclass, static_sym)
        })
    };
    match lookup {
        Some((Method::Native(native_fn), _dc)) => native_fn(vm, args).to_bits(),
        Some((Method::Host(host_fn, context), _dc)) => host_fn(vm, context, args).to_bits(),
        Some((Method::ForeignC(foreign_fn), _dc)) => {
            crate::runtime::foreign::dispatch_foreign_c(vm, foreign_fn, args).to_bits()
        }
        Some((Method::ForeignCDynamic(idx), _dc)) => {
            crate::runtime::foreign::dispatch_dynamic(vm, idx, args).to_bits()
        }
        Some((Method::Closure(closure_ptr), dc)) => {
            call_closure_jit_or_sync(vm, closure_ptr, args, Some(dc))
        }
        Some((Method::Constructor(closure_ptr), dc)) => {
            // Super-constructor: args[0] is already 'this' (the
            // newly allocated instance), not the class. The
            // legacy JIT-IC stale-pointer hazard that motivated
            // forcing call_closure_sync here doesn't fire when
            // the callee's jit_code slot is set by AOT (no IC
            // patching, no concurrent tier-up). When it is set,
            // route through call_closure_jit_or_sync so the AOT
            // body actually runs — call_closure_sync would walk
            // the stub MIR `register_aot_function` left and
            // panic on `mir.blocks[0]`.
            if args.is_empty() {
                return Value::null().to_bits();
            }
            let func_id = unsafe { (*(*closure_ptr).function).fn_id } as usize;
            let has_aot_body = vm
                .engine
                .jit_code
                .get(func_id)
                .copied()
                .map(|p| !p.is_null())
                .unwrap_or(false);
            if has_aot_body {
                call_closure_jit_or_sync(vm, closure_ptr, args, Some(dc))
            } else {
                vm.call_closure_sync(closure_ptr, args, Some(dc))
                    .map(|v| v.to_bits())
                    .unwrap_or(args[0].to_bits())
            }
        }
        None => Value::null().to_bits(),
    }
}

/// `dispatch_super_call` from a body compiled for a method of
/// `class`: the class is the compile's, not the context's, since a
/// body entered by a direct call from other compiled code finds the
/// context still naming the caller's.
fn dispatch_super_call_from(
    class: u64,
    recv: Value,
    method_sym: crate::intern::SymbolId,
    args: &[Value],
) -> u64 {
    let ctx = read_jit_ctx();
    let saved = ctx.defining_class;
    unsafe { (*jit_state()).ctx.defining_class = class as *mut u8 };
    let r = dispatch_super_call(recv, method_sym, args);
    unsafe { (*jit_state()).ctx.defining_class = saved };
    r
}

/// Super call with 1 arg from a method of `class`: `[class, method_sym, this]`.
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_super_call_from_1(class: u64, method: u64, this: u64) -> u64 {
    let recv = Value::from_bits(this);
    let sym = crate::intern::SymbolId::from_raw(method as u32);
    dispatch_super_call_from(class, recv, sym, &[recv])
}
/// Super call with 2 args from a method of `class`.
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_super_call_from_2(class: u64, method: u64, this: u64, a0: u64) -> u64 {
    let recv = Value::from_bits(this);
    let sym = crate::intern::SymbolId::from_raw(method as u32);
    dispatch_super_call_from(class, recv, sym, &[recv, Value::from_bits(a0)])
}
/// Super call with 3 args from a method of `class`.
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_super_call_from_3(
    class: u64,
    method: u64,
    this: u64,
    a0: u64,
    a1: u64,
) -> u64 {
    let recv = Value::from_bits(this);
    let sym = crate::intern::SymbolId::from_raw(method as u32);
    dispatch_super_call_from(
        class,
        recv,
        sym,
        &[recv, Value::from_bits(a0), Value::from_bits(a1)],
    )
}
/// Super call with 4 args from a method of `class`.
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_super_call_from_4(
    class: u64,
    method: u64,
    this: u64,
    a0: u64,
    a1: u64,
    a2: u64,
) -> u64 {
    let recv = Value::from_bits(this);
    let sym = crate::intern::SymbolId::from_raw(method as u32);
    dispatch_super_call_from(
        class,
        recv,
        sym,
        &[
            recv,
            Value::from_bits(a0),
            Value::from_bits(a1),
            Value::from_bits(a2),
        ],
    )
}

/// Super call with 0 args. Codegen: `[method_sym]` (no receiver — shouldn't happen in practice)
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_super_call_0(method: u64) -> u64 {
    let _ = method;
    Value::null().to_bits()
}
/// Super call with 1 arg. Codegen: `[method_sym, this]`
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_super_call_1(method: u64, this: u64) -> u64 {
    let recv = Value::from_bits(this);
    let sym = crate::intern::SymbolId::from_raw(method as u32);
    dispatch_super_call(recv, sym, &[recv])
}
/// Super call with 2 args. Codegen: `[method_sym, this, a0]`
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_super_call_2(method: u64, this: u64, a0: u64) -> u64 {
    let recv = Value::from_bits(this);
    let sym = crate::intern::SymbolId::from_raw(method as u32);
    dispatch_super_call(recv, sym, &[recv, Value::from_bits(a0)])
}
/// Super call with 3 args. Codegen: `[method_sym, this, a0, a1]`
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_super_call_3(method: u64, this: u64, a0: u64, a1: u64) -> u64 {
    let recv = Value::from_bits(this);
    let sym = crate::intern::SymbolId::from_raw(method as u32);
    dispatch_super_call(
        recv,
        sym,
        &[recv, Value::from_bits(a0), Value::from_bits(a1)],
    )
}
/// Super call with 4 args. Codegen: `[method_sym, this, a0, a1, a2]`
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_super_call_4(method: u64, this: u64, a0: u64, a1: u64, a2: u64) -> u64 {
    let recv = Value::from_bits(this);
    let sym = crate::intern::SymbolId::from_raw(method as u32);
    dispatch_super_call(
        recv,
        sym,
        &[
            recv,
            Value::from_bits(a0),
            Value::from_bits(a1),
            Value::from_bits(a2),
        ],
    )
}

/// Allocate a new empty list.
/// Create a list and populate it with the given elements.
fn make_list_impl(elements: &[u64]) -> u64 {
    let vm = unsafe { vm_ref() };
    let vm = match vm {
        Some(v) => v,
        None => return Value::null().to_bits(),
    };

    let root_len_before = jit_roots_snapshot_len();
    for &elem in elements {
        push_jit_root(Value::from_bits(elem));
    }

    let list_ptr = vm.gc.alloc_list_sized(elements.len());
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

/// Add a single element to an existing list.
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_list_add(list_val: u64, elem: u64) {
    let list = Value::from_bits(list_val);
    let elem_v = Value::from_bits(elem);
    if let Some(ptr) = list.as_object() {
        let list_ptr = ptr as *mut crate::runtime::object::ObjList;
        unsafe {
            (*list_ptr).add(elem_v);
        }
    }
}

#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_make_list() -> u64 {
    make_list_impl(&[])
}

#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_make_list_1(a0: u64) -> u64 {
    make_list_impl(&[a0])
}

#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_make_list_2(a0: u64, a1: u64) -> u64 {
    make_list_impl(&[a0, a1])
}

#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_make_list_3(a0: u64, a1: u64, a2: u64) -> u64 {
    make_list_impl(&[a0, a1, a2])
}

#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_make_list_4(a0: u64, a1: u64, a2: u64, a3: u64) -> u64 {
    make_list_impl(&[a0, a1, a2, a3])
}

/// Set a key-value pair on a map object.
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_map_set(map_val: u64, key: u64, value: u64) {
    let map = Value::from_bits(map_val);
    let key_v = Value::from_bits(key);
    let value_v = Value::from_bits(value);
    if let Some(ptr) = map.as_object() {
        let map_ptr = ptr as *mut ObjMap;
        unsafe {
            (*map_ptr).set(key_v, value_v);
        }
    }
}

#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_make_map() -> u64 {
    let vm = unsafe { vm_ref() };
    let vm = match vm {
        Some(v) => v,
        None => return Value::null().to_bits(),
    };

    let map_ptr = vm.gc.alloc_map();
    unsafe {
        (*map_ptr).header.class = vm.map_class;
    }
    let val = Value::object(map_ptr as *mut u8);
    unsafe { finish_alloc(vm, val) }
}

/// Allocate a new range.
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_make_range(from: u64, to: u64, inclusive: u64) -> u64 {
    let vm = unsafe { vm_ref() };
    let vm = match vm {
        Some(v) => v,
        None => return Value::null().to_bits(),
    };

    let from_val = f64::from_bits(from);
    let to_val = f64::from_bits(to);
    let is_inclusive = inclusive != 0;

    let range_ptr = vm.gc.alloc_range(from_val, to_val, is_inclusive);
    unsafe {
        (*range_ptr).header.class = vm.range_class;
    }
    let val = Value::object(range_ptr as *mut u8);
    unsafe { finish_alloc(vm, val) }
}

/// Helper: allocate closure and populate upvalues from a slice of NaN-boxed values.
fn make_closure_inner(fn_id: u64, upvalue_vals: &[u64]) -> u64 {
    let vm = unsafe { vm_ref() };
    let vm = match vm {
        Some(v) => v,
        None => return Value::null().to_bits(),
    };

    let fn_ptr = vm.closure_fn(fn_id as u32, upvalue_vals.len() as u16);
    let closure_ptr = vm.gc.alloc_closure(fn_ptr);
    let defining_class = read_jit_ctx().defining_class as *mut ObjClass;
    unsafe {
        (*closure_ptr).header.class = vm.fn_class;
        // The body reaches the static fields of the class whose
        // method is making it.
        (*closure_ptr).defining_class = defining_class;
        if !defining_class.is_null() {}
    }
    // Root the closure before upvalue allocations.
    push_jit_root(Value::object(closure_ptr as *mut u8));

    // Populate upvalues with captured values (pre-closed); the
    // closure is read back through its root each iteration.
    let closure_root_idx = jit_roots_len() - 1;
    for (i, &uv_bits) in upvalue_vals.iter().enumerate() {
        let captured_val = Value::from_bits(uv_bits);
        let uv_obj = vm.gc.alloc_upvalue(std::ptr::null_mut());
        unsafe {
            (*uv_obj).closed = captured_val;
            (*uv_obj).location = &mut (*uv_obj).closed as *mut Value;
            let live_closure_val = jit_root_at(closure_root_idx);
            let live_closure = live_closure_val.as_object().expect("closure root vanished")
                as *mut crate::runtime::object::ObjClosure;
            if i < (*live_closure).upvalues.len() {
                (&mut (*live_closure).upvalues)[i] = uv_obj;
            }
        }
    }

    // The closure (and its function object) were pushed as
    // intermediate roots during the upvalue allocation loop; read
    // the rooted value back before popping.
    let closure_val = pop_jit_root().expect("closure root vanished");
    unsafe { finish_alloc(vm, closure_val) }
}

/// Allocate a closure with 0 upvalues.
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_make_closure_0(fn_id: u64) -> u64 {
    make_closure_inner(fn_id, &[])
}
/// Allocate a closure with 1 upvalue.
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_make_closure_1(fn_id: u64, uv0: u64) -> u64 {
    make_closure_inner(fn_id, &[uv0])
}
/// Allocate a closure with 2 upvalues.
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_make_closure_2(fn_id: u64, uv0: u64, uv1: u64) -> u64 {
    make_closure_inner(fn_id, &[uv0, uv1])
}
/// Allocate a closure with 3 upvalues.
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_make_closure_3(fn_id: u64, uv0: u64, uv1: u64, uv2: u64) -> u64 {
    make_closure_inner(fn_id, &[uv0, uv1, uv2])
}
/// Allocate a closure with 4 upvalues.
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_make_closure_4(fn_id: u64, uv0: u64, uv1: u64, uv2: u64, uv3: u64) -> u64 {
    make_closure_inner(fn_id, &[uv0, uv1, uv2, uv3])
}
/// Allocate a closure with 5 upvalues.
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_make_closure_5(
    fn_id: u64,
    uv0: u64,
    uv1: u64,
    uv2: u64,
    uv3: u64,
    uv4: u64,
) -> u64 {
    make_closure_inner(fn_id, &[uv0, uv1, uv2, uv3, uv4])
}
/// Allocate a closure with 6 upvalues.
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_make_closure_6(
    fn_id: u64,
    uv0: u64,
    uv1: u64,
    uv2: u64,
    uv3: u64,
    uv4: u64,
    uv5: u64,
) -> u64 {
    make_closure_inner(fn_id, &[uv0, uv1, uv2, uv3, uv4, uv5])
}
/// Allocate a closure with 7 upvalues.
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_make_closure_7(
    fn_id: u64,
    uv0: u64,
    uv1: u64,
    uv2: u64,
    uv3: u64,
    uv4: u64,
    uv5: u64,
    uv6: u64,
) -> u64 {
    make_closure_inner(fn_id, &[uv0, uv1, uv2, uv3, uv4, uv5, uv6])
}
/// Allocate a closure with 8 upvalues. AOT bodies that capture
/// more than 8 upvalues fall through to the generic `wren_make_
/// closure_n` slow path so the lowering doesn't silently truncate.
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_make_closure_8(
    fn_id: u64,
    uv0: u64,
    uv1: u64,
    uv2: u64,
    uv3: u64,
    uv4: u64,
    uv5: u64,
    uv6: u64,
    uv7: u64,
) -> u64 {
    make_closure_inner(fn_id, &[uv0, uv1, uv2, uv3, uv4, uv5, uv6, uv7])
}
/// Allocate a closure with `n` upvalues read from a contiguous
/// `[u64; n]` buffer the lowering builds on the JIT stack. Used as
/// the > 8-upvalue fallback. Without it, lowering had to truncate
/// the upvalue list past arity 4, so a 5+-upvalue closure read
/// past the end of its `Vec<*mut ObjUpvalue>` and crashed at the
/// first access of the dropped index — exactly what happened to
/// `Session.cookie`'s 7-upvalue middleware in the web spec.
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub unsafe extern "C" fn wren_make_closure_n(fn_id: u64, count: u64, upvalues: *const u64) -> u64 {
    if upvalues.is_null() {
        return make_closure_inner(fn_id, &[]);
    }
    let slice = unsafe { std::slice::from_raw_parts(upvalues, count as usize) };
    make_closure_inner(fn_id, slice)
}

/// Concatenate two strings.
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_string_concat(a: u64, b: u64) -> u64 {
    let va = Value::from_bits(a);
    let vb = Value::from_bits(b);

    let vm = unsafe { vm_ref() };
    let vm = match vm {
        Some(v) => v,
        None => return Value::null().to_bits(),
    };

    let sa = crate::runtime::vm_interp::value_to_string(vm, va);
    let sb = crate::runtime::vm_interp::value_to_string(vm, vb);
    let concatenated = format!("{}{}", sa, sb);
    let val = vm.new_string(concatenated);
    unsafe { finish_alloc(vm, val) }
}

/// Convert a value to its string representation.
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_to_string(val: u64) -> u64 {
    let v = Value::from_bits(val);

    let vm = unsafe { vm_ref() };
    let vm = match vm {
        Some(v) => v,
        None => return Value::null().to_bits(),
    };

    let s = crate::runtime::vm_interp::value_to_string(vm, v);
    let val = vm.new_string(s);
    unsafe { finish_alloc(vm, val) }
}

/// Materialize a string literal from its interned symbol id.
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_const_string(sym_idx: u64) -> u64 {
    let vm = unsafe { vm_ref() };
    let vm = match vm {
        Some(v) => v,
        None => return Value::null().to_bits(),
    };

    let sym = crate::intern::SymbolId::from_raw(sym_idx as u32);
    let val = vm.new_string(vm.interner.resolve(sym).to_string());
    unsafe { finish_alloc(vm, val) }
}

/// Type check: is value an instance of class?
/// class_sym is a SymbolId identifying the class name.
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_is_type(val: u64, class_sym: u64) -> u64 {
    let v = Value::from_bits(val);
    let target_sym = crate::intern::SymbolId::from_raw(class_sym as u32);

    let vm = unsafe { vm_ref() };
    let vm = match vm {
        Some(vm) => vm,
        None => return Value::bool(false).to_bits(),
    };

    // Walk the class hierarchy
    let mut class = vm.class_of(v);
    loop {
        if class.is_null() {
            break;
        }
        let class_name = unsafe { (*class).name };
        if class_name == target_sym {
            return Value::bool(true).to_bits();
        }
        class = unsafe { (*class).superclass };
    }
    Value::bool(false).to_bits()
}

/// Subscript get (`list[idx]` or `map[key]`).
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_subscript_get(receiver: u64, index: u64) -> u64 {
    let recv = Value::from_bits(receiver);
    let idx = Value::from_bits(index);

    if recv.is_object() {
        let ptr = recv.as_object().unwrap();
        let header = ptr as *const ObjHeader;
        let obj_type = unsafe { (*header).obj_type };

        match obj_type {
            ObjType::List => {
                let list = ptr as *const ObjList;
                if let Some(i) = fast_list_index(idx, unsafe { (*list).count as usize })
                    && let Some(val) = unsafe { (*list).get(i) }
                {
                    return val.to_bits();
                }
            }
            ObjType::Map => {
                let map = ptr as *const ObjMap;
                let map_key = MapKey::new(idx);
                if let Some(val) = unsafe { (*map).entries.get(&map_key) } {
                    return val.to_bits();
                }
            }
            ObjType::String => {
                let string = ptr as *const ObjString;
                let s = unsafe { (*string).as_str() };
                // `str[i]` — single char at a char index.
                if let Some(n) = idx.as_num() {
                    let mut i = n as i64;
                    // Use cached char count; template engines walk
                    // long strings via repeated `str[i]` and a fresh
                    // `chars().count()` each time turns the loop into
                    // O(N²).
                    let char_count = unsafe { (*string).char_count_cached() } as i64;
                    if i < 0 {
                        i += char_count;
                    }
                    if i >= 0
                        && i < char_count
                        && let Some(ch) = s.chars().nth(i as usize)
                    {
                        let vm = unsafe { vm_ref() };
                        if let Some(vm) = vm {
                            return vm.new_string(ch.to_string()).to_bits();
                        }
                    }
                }
                // `str[range]` — substring slice. Without this the JIT
                // lowering of `Instruction::SubscriptGet` silently returns
                // null for every range-based slice on a String, which
                // breaks any parser / string processor that tier-ups.
                // The interpreter handles this via the native `[_]` method
                // in `runtime/core/string.rs::subscript`; this path mirrors
                // the Range branch of that function.
                if idx.is_object() {
                    let idx_ptr = idx.as_object().unwrap();
                    let idx_header = idx_ptr as *const ObjHeader;
                    if unsafe { (*idx_header).obj_type } == ObjType::Range {
                        use crate::runtime::object::ObjRange;
                        let range = unsafe { &*(idx_ptr as *const ObjRange) };
                        let len = unsafe { (*string).char_count_cached() } as i64;
                        let normalize = |mut v: i64| -> i64 {
                            if v < 0 {
                                v += len
                            }
                            v
                        };
                        let from = normalize(range.from as i64);
                        let to_raw = normalize(range.to as i64);
                        let end = if range.is_inclusive {
                            to_raw + 1
                        } else {
                            to_raw
                        };
                        if from >= 0 && end >= from && end <= len {
                            let slice: String = s
                                .chars()
                                .skip(from as usize)
                                .take((end - from) as usize)
                                .collect();
                            let vm = unsafe { vm_ref() };
                            if let Some(vm) = vm {
                                return vm.new_string(slice).to_bits();
                            }
                        }
                    }
                }
            }
            ObjType::TypedArray => {
                use crate::runtime::object::{ObjTypedArray, TypedArrayKind};
                let arr = ptr as *const ObjTypedArray;
                if let Some(n) = idx.as_num() {
                    let count = unsafe { (*arr).count as i64 };
                    // Wren negative-index convention.
                    let raw = n as i64;
                    let i = if raw < 0 { raw + count } else { raw };
                    if i < 0 || i >= count {
                        return Value::null().to_bits();
                    }
                    let i = i as usize;
                    let kind = unsafe { (*arr).kind_tag() };
                    let v = match kind {
                        TypedArrayKind::U8 => unsafe { (*arr).get_u8(i).unwrap_or(0) as f64 },
                        TypedArrayKind::I32 => unsafe { (*arr).get_i32(i).unwrap_or(0) as f64 },
                        TypedArrayKind::F32 => unsafe { (*arr).get_f32(i).unwrap_or(0.0) as f64 },
                        TypedArrayKind::F64 => unsafe { (*arr).get_f64(i).unwrap_or(0.0) },
                    };
                    return Value::num(v).to_bits();
                }
            }
            ObjType::Simd => {
                let simd = ptr as *const ObjSimd;
                if let Some(n) = idx.as_num() {
                    let raw = n as i64;
                    let i = if raw < 0 { raw + 4 } else { raw };
                    if !(0..4).contains(&i) {
                        return Value::null().to_bits();
                    }
                    let i = i as usize;
                    let v = match unsafe { (*simd).kind_tag() } {
                        crate::runtime::object::SimdKind::F32x4 => unsafe {
                            (*simd).get_f32(i).unwrap_or(0.0) as f64
                        },
                        crate::runtime::object::SimdKind::I32x4 => unsafe {
                            (*simd).get_i32(i).unwrap_or(0) as f64
                        },
                    };
                    return Value::num(v).to_bits();
                }
            }
            _ => {}
        }
    }
    // Fallback: the receiver has a `[_]` method defined on its class
    // (user class with `subscript` overload, StringByteSequence,
    // StringCodePointSequence, etc.). Dispatch through the method
    // table the same way the interpreter does — missing this path
    // silently returned null for every non-builtin `foo[bar]` in
    // JIT'd code, which broke `c.bytes[0]`-style patterns and any
    // user class overloading `[_]`.
    let vm = unsafe { vm_ref() };
    if let Some(vm) = vm
        && let Some(v) = vm.call_method_on(recv, "[_]", &[idx])
    {
        return v.to_bits();
    }
    Value::null().to_bits()
}

/// Subscript set (`list[idx] = val` or `map[key] = val`).
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_subscript_set(receiver: u64, index: u64, value: u64) -> u64 {
    let recv = Value::from_bits(receiver);
    let idx = Value::from_bits(index);
    let value = Value::from_bits(value);

    if recv.is_object() {
        let ptr = recv.as_object().unwrap();
        let header = ptr as *const ObjHeader;
        let obj_type = unsafe { (*header).obj_type };

        match obj_type {
            ObjType::List => {
                let list = ptr as *mut ObjList;
                if let Some(i) = fast_list_index(idx, unsafe { (*list).count as usize }) {
                    unsafe {
                        (*list).set(i, value);
                    }
                    return value.to_bits();
                }
            }
            ObjType::Map => {
                let map = ptr as *mut ObjMap;
                let map_key = MapKey::new(idx);
                unsafe {
                    (*map).entries.insert(map_key, value);
                }
                return value.to_bits();
            }
            ObjType::TypedArray => {
                use crate::runtime::object::{ObjTypedArray, TypedArrayKind};
                let arr = ptr as *mut ObjTypedArray;
                if let Some(n) = idx.as_num() {
                    let count = unsafe { (*arr).count as i64 };
                    let raw = n as i64;
                    let i = if raw < 0 { raw + count } else { raw };
                    if i < 0 || i >= count {
                        return value.to_bits();
                    }
                    let i = i as usize;
                    let kind = unsafe { (*arr).kind_tag() };
                    if let Some(v) = value.as_num() {
                        match kind {
                            TypedArrayKind::U8 => {
                                if (0.0..=255.0).contains(&v) && v.fract() == 0.0 {
                                    unsafe { (*arr).set_u8(i, v as u8) };
                                }
                            }
                            TypedArrayKind::I32 => {
                                if v.fract() == 0.0 && v >= i32::MIN as f64 && v <= i32::MAX as f64
                                {
                                    unsafe { (*arr).set_i32(i, v as i32) };
                                }
                            }
                            TypedArrayKind::F32 => unsafe { (*arr).set_f32(i, v as f32) },
                            TypedArrayKind::F64 => unsafe { (*arr).set_f64(i, v) },
                        }
                    }
                    return value.to_bits();
                }
            }
            _ => {}
        }
    }
    // Fallback: dispatch to the `[_]=(_)` method for user classes or
    // builtins with custom subscript-set overloads. Same rationale as
    // the matching fallback in wren_subscript_get — without this the
    // JIT silently drops `foo[bar] = baz` for every non-builtin
    // receiver and returns null-but-"arity 1" runtime errors at the
    // next use site.
    let vm = unsafe { vm_ref() };
    if let Some(vm) = vm
        && let Some(v) = vm.call_method_on(recv, "[_]=(_)", &[idx, value])
    {
        return v.to_bits();
    }
    value.to_bits()
}

/// Get an upvalue by index from the current closure in JitContext.
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_get_upvalue(index: u64) -> u64 {
    let ctx = read_jit_ctx();
    if ctx.closure.is_null() {
        return Value::null().to_bits();
    }
    let closure = ctx.closure as *const ObjClosure;
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

/// Set an upvalue by index on the current closure in JitContext.
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_set_upvalue(index: u64, value: u64) -> u64 {
    let ctx = read_jit_ctx();
    if ctx.closure.is_null() {
        return value;
    }
    let closure = ctx.closure as *const ObjClosure;
    let idx = index as usize;
    let value = Value::from_bits(value);
    unsafe {
        let upvalues = &(*closure).upvalues;
        if idx < upvalues.len() {
            let uv = upvalues[idx];
            (*uv).set(value);
        }
    }
    value.to_bits()
}

/// Get a static field from the defining class.
/// field_sym is the raw SymbolId index.
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_get_static_field(field_sym: u64) -> u64 {
    let ctx = read_jit_ctx();
    if ctx.defining_class.is_null() {
        return Value::null().to_bits();
    }
    let class = ctx.defining_class as *const ObjClass;
    let sym = crate::intern::SymbolId::from_raw(field_sym as u32);
    unsafe {
        (*class)
            .static_fields
            .get(&sym)
            .copied()
            .unwrap_or(Value::null())
            .to_bits()
    }
}

/// Set a static field on the defining class.
/// field_sym is the raw SymbolId index, value is the NaN-boxed value.
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_set_static_field(field_sym: u64, value: u64) -> u64 {
    let ctx = read_jit_ctx();
    if ctx.defining_class.is_null() {
        return value;
    }
    let class = ctx.defining_class as *mut ObjClass;
    let sym = crate::intern::SymbolId::from_raw(field_sym as u32);
    let value = Value::from_bits(value);
    unsafe {
        (*class).static_fields.insert(sym, value);
    }
    value.to_bits()
}

/// Guard: check that value is an instance of the expected class.
/// Returns the value if check passes, traps otherwise.
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_guard_class(value: u64, class: u64) -> u64 {
    // For now, always pass the guard. A proper implementation would
    // check the class hierarchy and deoptimize on mismatch.
    let _ = class;
    value
}

/// Guard: check that value's class implements the expected protocol.
/// Returns the value if check passes (always passes for now).
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_guard_protocol(value: u64, protocol_id: u64) -> u64 {
    let _ = protocol_id;
    value
}

// ---------------------------------------------------------------------------
// Guard deoptimization — invalidate JIT + re-execute via interpreter
// ---------------------------------------------------------------------------

/// A speculative guard failed at the entry of `func_id`'s compiled
/// code, before any side effect: the function goes back to the
/// interpreter until an unspeculated compile lands, and this call is
/// re-run there with its original arguments.
#[cfg(feature = "host")]
fn deopt_impl(func_id: u32, args: &[u64]) -> u64 {
    let vm = unsafe { vm_ref() };
    let vm = match vm {
        Some(v) => v,
        None => return Value::null().to_bits(),
    };
    let id = crate::runtime::engine::FuncId(func_id);
    if env_flag(&TIER_TRACE, "WLIFT_TIER_TRACE") {
        eprintln!(
            "tier-trace: [{:.2}ms] deopt FuncId({}) argc={}",
            crate::runtime::engine::trace_clock_ms(),
            func_id,
            args.len()
        );
    }
    // The bailout reloads the bead to interpreted first, so the
    // recompile can go through the broker.
    let _decision = vm.engine.tier.record_bailout(id, 0, 0);
    vm.note_speculation_failed(id);
    run_interpreted(vm, func_id, args)
}

/// Run `func_id` on `args` in the interpreter and return its result.
#[cfg(any(feature = "host", feature = "aot_runtime"))]
fn run_interpreted(vm: &mut crate::runtime::vm::VM, func_id: u32, args: &[u64]) -> u64 {
    let id = crate::runtime::engine::FuncId(func_id);
    let values: Vec<Value> = args.iter().map(|&a| Value::from_bits(a)).collect();

    // The running closure when the dispatcher recorded one for this
    // function; a method reached through a direct call has only the
    // caller's, so it is dispatched again by name instead.
    let ctx = read_jit_ctx();
    let closure_ptr = ctx.closure as *mut ObjClosure;
    let closure_matches =
        !closure_ptr.is_null() && unsafe { (*(*closure_ptr).function).fn_id } == func_id;
    if closure_matches {
        let defining_class = if ctx.defining_class.is_null() {
            None
        } else {
            Some(ctx.defining_class as *mut crate::runtime::object::ObjClass)
        };
        return vm
            .call_closure_sync(closure_ptr, &values, defining_class)
            .map(|v| v.to_bits())
            .unwrap_or(Value::null().to_bits());
    }
    let Some(name) = vm.engine.get_mir(id).map(|m| m.name) else {
        return Value::null().to_bits();
    };
    let Some(&recv) = values.first() else {
        return Value::null().to_bits();
    };
    dispatch_call(recv, name.index() as u64, &values)
}

/// The deopt buffer is a word stream: a tag word, then the words of
/// the register's source. Bit 32 of the tag marks a range rebuilt from
/// its two bound words, bit 33 its inclusiveness; the low 32 bits are
/// the register.
const DEOPT_RANGE: u64 = 1 << 32;
const DEOPT_INCLUSIVE: u64 = 1 << 33;
/// Bit 34 marks an instance rebuilt from its field words, which follow
/// its class word and identity word; bits 40-55 hold the field count.
const DEOPT_OBJECT: u64 = 1 << 34;
const DEOPT_FIELDS_SHIFT: u32 = 40;

/// The tag word of `r` in a deopt buffer.
pub fn deopt_tag(r: &crate::mir::DeoptReg) -> u64 {
    match &r.source {
        crate::mir::DeoptSource::Value(_) => r.reg as u64,
        crate::mir::DeoptSource::Range { inclusive, .. } => {
            r.reg as u64 | DEOPT_RANGE | if *inclusive { DEOPT_INCLUSIVE } else { 0 }
        }
        crate::mir::DeoptSource::Object { fields, .. } => {
            r.reg as u64 | DEOPT_OBJECT | ((fields.len() as u64) << DEOPT_FIELDS_SHIFT)
        }
    }
}

/// The constant words that follow the tag of `r`, before its operands.
pub fn deopt_consts(r: &crate::mir::DeoptReg) -> Vec<u64> {
    match &r.source {
        crate::mir::DeoptSource::Object { class, id, .. } => vec![*class as u64, *id as u64],
        _ => Vec::new(),
    }
}

/// Words `r` takes in a deopt buffer, tag included.
pub fn deopt_words(r: &crate::mir::DeoptReg) -> usize {
    1 + deopt_consts(r).len() + r.source.operands().len()
}

/// The registers `words` describe, in the layout `deopt_tag` and
/// `deopt_words` lay down: a range or an instance the compiled body
/// kept as scalars is allocated again and rooted for the caller.
fn decode_deopt_words(vm: &mut crate::runtime::vm::VM, words: &[u64]) -> Vec<(u32, Value)> {
    let mut regs: Vec<(u32, Value)> = Vec::new();
    let mut objects: Vec<(u32, Value)> = Vec::new();
    let mut i = 0;
    while i < words.len() {
        let tag = words[i];
        let reg = tag as u32;
        if tag & DEOPT_RANGE != 0 {
            let from = f64::from_bits(words[i + 1]);
            let to = f64::from_bits(words[i + 2]);
            let range_ptr = vm.gc.alloc_range(from, to, tag & DEOPT_INCLUSIVE != 0);
            unsafe {
                (*range_ptr).header.class = vm.range_class;
            }
            let val = Value::object(range_ptr as *mut u8);
            push_jit_root(val);
            regs.push((reg, val));
            i += 3;
        } else if tag & DEOPT_OBJECT != 0 {
            let class = words[i + 1] as *mut crate::runtime::object::ObjClass;
            let id = words[i + 2] as u32;
            let nfields = (tag >> DEOPT_FIELDS_SHIFT) as usize & 0xffff;
            let val = match objects.iter().find(|(k, _)| *k == id) {
                Some((_, v)) => *v,
                None => {
                    let inst = vm.gc.alloc_instance(class);
                    unsafe {
                        (*inst).header.class = class;
                        for f in 0..nfields.min((*inst).num_fields as usize) {
                            (*inst).set_field_unchecked(f, Value::from_bits(words[i + 3 + f]));
                        }
                    }
                    let val = Value::object(inst as *mut u8);
                    push_jit_root(val);
                    objects.push((id, val));
                    val
                }
            };
            regs.push((reg, val));
            i += 3 + nfields;
        } else {
            regs.push((reg, Value::from_bits(words[i + 1])));
            i += 2;
        }
    }
    regs
}

/// A mid-body speculation in `func_id` failed: resume the interpreter
/// at bytecode offset `pc` with the registers described by the `n`
/// words in `buf` and return the function's result.
///
/// # Safety
/// `buf` must point at `n` readable u64s laid out as `deopt_tag` and
/// `deopt_words` describe; compiled code passes its own stack buffer.
#[cfg(feature = "host")]
#[cfg_attr(not(target_arch = "wasm32"), unsafe(no_mangle))]
pub unsafe extern "C" fn wren_deopt_at(func_id: u64, pc: u64, n: u64, buf: *const u64) -> u64 {
    let vm = unsafe { vm_ref() };
    let vm = match vm {
        Some(v) => v,
        None => return Value::null().to_bits(),
    };
    // The frame goes on in the interpreter; a trace leaves it out.
    shadow_current();
    let func_id = func_id as u32 as u64;
    let words: Vec<u64> = (0..n as usize).map(|i| unsafe { *buf.add(i) }).collect();
    let root_len_before = jit_roots_snapshot_len();
    let regs = decode_deopt_words(vm, &words);
    let id = crate::runtime::engine::FuncId(func_id as u32);
    if env_flag(&TIER_TRACE, "WLIFT_TIER_TRACE") {
        eprintln!(
            "tier-trace: [{:.2}ms] deopt FuncId({}) at pc={} live={}",
            crate::runtime::engine::trace_clock_ms(),
            func_id,
            pc,
            n
        );
    }
    let _decision = vm.engine.tier.record_bailout(id, 0, 0);
    vm.note_speculation_failed(id);
    vm.engine.deopt_exits += 1;
    // The running closure when the dispatcher recorded one for this
    // function, else the one the method was bound with.
    let ctx = read_jit_ctx();
    let ctx_closure = ctx.closure as *mut ObjClosure;
    let (closure, defining_class) = if !ctx_closure.is_null()
        && unsafe { (*(*ctx_closure).function).fn_id } == func_id as u32
    {
        (
            ctx_closure,
            ctx.defining_class as *mut crate::runtime::object::ObjClass,
        )
    } else {
        match vm.engine.method_binding.get(func_id as usize) {
            Some(&(c, d)) if !c.is_null() => (c, d),
            _ => {
                vm.has_error = true;
                note_error_pending();
                vm.note_raise();
                vm.last_error = Some(format!(
                    "deoptimisation of FuncId({}) found no closure to resume",
                    func_id
                ));
                return Value::null().to_bits();
            }
        }
    };
    let result = vm
        .resume_method_sync(closure, defining_class, id, pc as u32, &regs)
        .map(|v| v.to_bits())
        .unwrap_or(Value::null().to_bits());
    jit_roots_restore_len(root_len_before);
    result
}

/// Deopt `func_id` with its `n` entry arguments in `buf`.
///
/// # Safety
/// `buf` must point at `n` readable u64s; compiled code passes its own
/// stack buffer.
#[cfg(feature = "host")]
#[cfg_attr(not(target_arch = "wasm32"), unsafe(no_mangle))]
pub unsafe extern "C" fn wren_deopt_n(func_id: u64, n: u64, buf: *const u64) -> u64 {
    let args: Vec<u64> = (0..n as usize).map(|i| unsafe { *buf.add(i) }).collect();
    shadow_current();
    deopt_impl(func_id as u32, &args)
}

// ---------------------------------------------------------------------------
// Boxed NaN-boxed arithmetic runtime functions
// ---------------------------------------------------------------------------

/// Helper: unbox a NaN-boxed u64 as f64.
#[inline(always)]
fn unbox_num(bits: u64) -> f64 {
    f64::from_bits(bits)
}

/// Helper: box an f64 as NaN-boxed u64.
#[inline(always)]
fn box_num(n: f64) -> u64 {
    Value::num(n).to_bits()
}

#[cfg(any(feature = "host", feature = "aot_runtime"))]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_num_add(a: u64, b: u64) -> u64 {
    wren_arith_dispatch(a, b, "+(_)", "+", |x, y| x + y)
}

#[cfg(any(feature = "host", feature = "aot_runtime"))]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_num_sub(a: u64, b: u64) -> u64 {
    wren_arith_dispatch(a, b, "-(_)", "-", |x, y| x - y)
}

#[cfg(any(feature = "host", feature = "aot_runtime"))]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_num_mul(a: u64, b: u64) -> u64 {
    wren_arith_dispatch(a, b, "*(_)", "*", |x, y| x * y)
}

#[cfg(any(feature = "host", feature = "aot_runtime"))]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_num_div(a: u64, b: u64) -> u64 {
    wren_arith_dispatch(a, b, "/(_)", "/", |x, y| x / y)
}

#[cfg(any(feature = "host", feature = "aot_runtime"))]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_num_mod(a: u64, b: u64) -> u64 {
    wren_arith_dispatch(a, b, "%(_)", "%", |x, y| x % y)
}

// ---------------------------------------------------------------------------
// Bitwise — declared by the Cranelift lowering for `Instruction::BitAnd /
// BitOr / BitXor / BitNot / Shl / Shr`. Wren truncates Num operands to u32
// before the op (per the bytecode interpreter's `Op::BitAnd` / etc.). Hatch
// packages use these for hash mixing; without the symbols the staticlib
// failed to link any AOT object that touched those ops.
// ---------------------------------------------------------------------------

#[cfg(any(feature = "host", feature = "aot_runtime"))]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_bit_and(a: u64, b: u64) -> u64 {
    wren_bit_binop(a, b, "&(_)", "&", |x, y| x & y)
}

#[cfg(any(feature = "host", feature = "aot_runtime"))]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_bit_or(a: u64, b: u64) -> u64 {
    wren_bit_binop(a, b, "|(_)", "|", |x, y| x | y)
}

#[cfg(any(feature = "host", feature = "aot_runtime"))]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_bit_xor(a: u64, b: u64) -> u64 {
    wren_bit_binop(a, b, "^(_)", "^", |x, y| x ^ y)
}

#[cfg(any(feature = "host", feature = "aot_runtime"))]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_bit_shl(a: u64, b: u64) -> u64 {
    wren_bit_binop(a, b, "<<(_)", "<<", |x, y| x.wrapping_shl(y & 31))
}

#[cfg(any(feature = "host", feature = "aot_runtime"))]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_bit_shr(a: u64, b: u64) -> u64 {
    wren_bit_binop(a, b, ">>(_)", ">>", |x, y| x.wrapping_shr(y & 31))
}

#[cfg(any(feature = "host", feature = "aot_runtime"))]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_bit_not(a: u64) -> u64 {
    let va = Value::from_bits(a);
    if va.is_num() {
        let n = Value::num_to_u32_wrapping(unbox_num(a));
        return box_num((!n) as f64);
    }
    match unsafe { vm_ref() } {
        Some(vm) if vm.has_error => Value::null().to_bits(),
        Some(vm) => {
            let sym = vm.interner.lookup("~").or_else(|| vm.interner.lookup("!"));
            if let Some(sym) = sym {
                let class = vm.class_of(va);
                if let Some((method, _dc)) = unsafe { find_method_with_class(class, sym) } {
                    return dispatch_method(vm, method, &[va], None);
                }
            }
            raise_method_not_found(vm, va, "~")
        }
        None => Value::null().to_bits(),
    }
}

#[inline]
fn wren_bit_binop(
    a: u64,
    b: u64,
    method_with_paren: &str,
    method_bare: &str,
    fast: impl FnOnce(u32, u32) -> u32,
) -> u64 {
    let va = Value::from_bits(a);
    if va.is_num() {
        let x = Value::num_to_u32_wrapping(unbox_num(a));
        let y = Value::num_to_u32_wrapping(unbox_num(b));
        return box_num(fast(x, y) as f64);
    }
    match unsafe { vm_ref() } {
        Some(vm) if vm.has_error => Value::null().to_bits(),
        Some(vm) => {
            let sym = vm
                .interner
                .lookup(method_with_paren)
                .or_else(|| vm.interner.lookup(method_bare));
            if let Some(sym) = sym {
                let class = vm.class_of(va);
                if let Some((method, _dc)) = unsafe { find_method_with_class(class, sym) } {
                    return dispatch_method(vm, method, &[va, Value::from_bits(b)], None);
                }
            }
            raise_method_not_found(vm, va, method_with_paren)
        }
        None => Value::null().to_bits(),
    }
}

/// Raise the interpreter's method-not-found error for `recv` and return
/// the null the caller hands back; the interpreter picks the error up
/// at the next boundary.
fn raise_method_not_found(vm: &mut crate::runtime::vm::VM, recv: Value, method: &str) -> u64 {
    // The first error stands; compiled code keeps running on nulls
    // until the interpreter unwinds it.
    if !vm.has_error {
        let class_name = vm.class_name_of(recv);
        vm.runtime_error(format!("{} does not implement '{}'", class_name, method));
    }
    Value::null().to_bits()
}

/// Common path for arithmetic-operator slow paths: if the receiver
/// is a Num, run the f64 op; otherwise dispatch the user-defined
/// operator method (e.g. `Mat4 * Mat4` → `Mat4.* (o)`). Mirrors
/// what `Op::Mul` / `Op::Sub` / etc. do in the bytecode interpreter
/// via `bc_boxed_binop!` + `try_operator_dispatch`.
#[inline]
fn wren_arith_dispatch(
    a: u64,
    b: u64,
    method_with_paren: &str,
    method_bare: &str,
    fast: impl FnOnce(f64, f64) -> f64,
) -> u64 {
    let va = Value::from_bits(a);
    if va.is_num() {
        return box_num(fast(unbox_num(a), unbox_num(b)));
    }
    match unsafe { vm_ref() } {
        Some(vm) if vm.has_error => Value::null().to_bits(),
        Some(vm) => {
            let sym = vm
                .interner
                .lookup(method_with_paren)
                .or_else(|| vm.interner.lookup(method_bare));
            if let Some(sym) = sym {
                let class = vm.class_of(va);
                if let Some((method, _dc)) = unsafe { find_method_with_class(class, sym) } {
                    return dispatch_method(vm, method, &[va, Value::from_bits(b)], None);
                }
            }
            raise_method_not_found(vm, va, method_with_paren)
        }
        None => Value::null().to_bits(),
    }
}

#[cfg(any(feature = "host", feature = "aot_runtime"))]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_num_neg(a: u64) -> u64 {
    let va = Value::from_bits(a);
    if va.is_num() {
        return box_num(-unbox_num(a));
    }
    // Non-numeric: dispatch as prefix - method
    match unsafe { vm_ref() } {
        Some(vm) if vm.has_error => Value::null().to_bits(),
        Some(vm) => {
            let sym = vm
                .interner
                .lookup("-()")
                .or_else(|| vm.interner.lookup("-"));
            if let Some(sym) = sym {
                let class = vm.class_of(va);
                if let Some((method, _dc)) = unsafe { find_method_with_class(class, sym) } {
                    return dispatch_method(vm, method, &[va], None);
                }
            }
            raise_method_not_found(vm, va, "-()")
        }
        None => Value::null().to_bits(),
    }
}

// ---------------------------------------------------------------------------
// Boxed comparison runtime functions
// ---------------------------------------------------------------------------

#[cfg(any(feature = "host", feature = "aot_runtime"))]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_cmp_lt(a: u64, b: u64) -> u64 {
    wren_cmp_dispatch(a, b, "<(_)", "<", |x, y| x < y)
}

#[cfg(any(feature = "host", feature = "aot_runtime"))]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_cmp_gt(a: u64, b: u64) -> u64 {
    wren_cmp_dispatch(a, b, ">(_)", ">", |x, y| x > y)
}

#[cfg(any(feature = "host", feature = "aot_runtime"))]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_cmp_le(a: u64, b: u64) -> u64 {
    wren_cmp_dispatch(a, b, "<=(_)", "<=", |x, y| x <= y)
}

#[cfg(any(feature = "host", feature = "aot_runtime"))]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_cmp_ge(a: u64, b: u64) -> u64 {
    wren_cmp_dispatch(a, b, ">=(_)", ">=", |x, y| x >= y)
}

/// Same dispatch pattern as `wren_arith_dispatch`, but for ordering
/// comparisons. User-defined `<(_)`, `>(_)`, `<=(_)`, `>=(_)` need
/// to be honoured for non-Num receivers; the bare-fast-path version
/// silently returned `Bool(NaN < NaN) == false` for any object pair,
/// which silently broke `Comparable.compareTo`-style protocols.
#[inline]
fn wren_cmp_dispatch(
    a: u64,
    b: u64,
    method_with_paren: &str,
    method_bare: &str,
    fast: impl FnOnce(f64, f64) -> bool,
) -> u64 {
    let va = Value::from_bits(a);
    if va.is_num() {
        return Value::bool(fast(unbox_num(a), unbox_num(b))).to_bits();
    }
    match unsafe { vm_ref() } {
        Some(vm) if vm.has_error => Value::bool(false).to_bits(),
        Some(vm) => {
            let sym = vm
                .interner
                .lookup(method_with_paren)
                .or_else(|| vm.interner.lookup(method_bare));
            if let Some(sym) = sym {
                let class = vm.class_of(va);
                if let Some((method, _dc)) = unsafe { find_method_with_class(class, sym) } {
                    return dispatch_method(vm, method, &[va, Value::from_bits(b)], None);
                }
            }
            raise_method_not_found(vm, va, method_with_paren);
            Value::bool(false).to_bits()
        }
        None => Value::bool(false).to_bits(),
    }
}

#[cfg(any(feature = "host", feature = "aot_runtime"))]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_cmp_eq(a: u64, b: u64) -> u64 {
    // Wren's `==` is overloadable per class. Mirror the
    // interpreter's `Op::CmpEq` handler: when the LHS is a
    // non-primitive object (i.e. user class instance — not a
    // String, Num, Bool, or Null), dispatch through the user's
    // `==(_)` method via the standard call helper. Otherwise
    // fall back to `Value::equals`, which short-circuits on equal
    // bits + handles NaN + does string content compare.
    let lhs = Value::from_bits(a);
    let rhs = Value::from_bits(b);
    if lhs.is_object()
        && !lhs.is_string_object()
        && !rhs.is_null()
        && !rhs.is_bool()
        && !rhs.is_num()
        && let Some(vm) = unsafe { vm_ref() }
    {
        let sym = vm.interner.intern("==(_)");
        return dispatch_call(lhs, sym.index() as u64, &[lhs, rhs]);
    }
    Value::bool(lhs.equals(rhs)).to_bits()
}

#[cfg(any(feature = "host", feature = "aot_runtime"))]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_cmp_ne(a: u64, b: u64) -> u64 {
    let lhs = Value::from_bits(a);
    let rhs = Value::from_bits(b);
    if lhs.is_object()
        && !lhs.is_string_object()
        && !rhs.is_null()
        && !rhs.is_bool()
        && !rhs.is_num()
        && let Some(vm) = unsafe { vm_ref() }
    {
        let sym = vm.interner.intern("!=(_)");
        return dispatch_call(lhs, sym.index() as u64, &[lhs, rhs]);
    }
    Value::bool(!lhs.equals(rhs)).to_bits()
}

// ---------------------------------------------------------------------------
// Boxed logical
// ---------------------------------------------------------------------------

#[cfg(any(feature = "host", feature = "aot_runtime"))]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_not(a: u64) -> u64 {
    Value::bool(Value::from_bits(a).is_falsy()).to_bits()
}

#[cfg(any(feature = "host", feature = "aot_runtime"))]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_is_truthy(value: u64) -> u64 {
    let v = Value::from_bits(value);
    // Return raw 0/1 (not NaN-boxed) so JmpZero can branch correctly.
    if v.is_falsy() { 0u64 } else { 1u64 }
}

// ---------------------------------------------------------------------------
// FP transcendental wrappers (raw f64 bits in/out, for JIT CallRuntime)
// ---------------------------------------------------------------------------

#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_fp_sin(bits: u64) -> u64 {
    f64::from_bits(bits).sin().to_bits()
}
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_fp_cos(bits: u64) -> u64 {
    f64::from_bits(bits).cos().to_bits()
}
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_fp_tan(bits: u64) -> u64 {
    f64::from_bits(bits).tan().to_bits()
}
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_fp_asin(bits: u64) -> u64 {
    f64::from_bits(bits).asin().to_bits()
}
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_fp_acos(bits: u64) -> u64 {
    f64::from_bits(bits).acos().to_bits()
}
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_fp_atan(bits: u64) -> u64 {
    f64::from_bits(bits).atan().to_bits()
}
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_fp_log(bits: u64) -> u64 {
    f64::from_bits(bits).ln().to_bits()
}
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_fp_log2(bits: u64) -> u64 {
    f64::from_bits(bits).log2().to_bits()
}
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_fp_exp(bits: u64) -> u64 {
    f64::from_bits(bits).exp().to_bits()
}
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_fp_cbrt(bits: u64) -> u64 {
    f64::from_bits(bits).cbrt().to_bits()
}
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_fp_atan2(a: u64, b: u64) -> u64 {
    f64::from_bits(a).atan2(f64::from_bits(b)).to_bits()
}
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_fp_pow(a: u64, b: u64) -> u64 {
    f64::from_bits(a).powf(f64::from_bits(b)).to_bits()
}
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_fp_min(a: u64, b: u64) -> u64 {
    f64::from_bits(a).min(f64::from_bits(b)).to_bits()
}
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_fp_max(a: u64, b: u64) -> u64 {
    f64::from_bits(a).max(f64::from_bits(b)).to_bits()
}

// ---------------------------------------------------------------------------
// ABI physical register mappings
// ---------------------------------------------------------------------------

/// ABI argument registers for each target (physical register indices).
#[allow(dead_code)]
fn abi_regs(target: super::Target) -> (&'static [u32], u32, u32, u32) {
    // Returns (arg_regs, ret_reg, call_scratch, copy_scratch)
    match target {
        super::Target::Aarch64 => (
            &[0, 1, 2, 3, 4, 5], // X0-X5
            0,                   // return in X0
            16,                  // X16 (IP0) for function pointer
            17,                  // X17 (IP1) for cycle breaking
        ),
        super::Target::X86_64 => (
            &[7, 6, 2, 1, 8, 9], // RDI, RSI, RDX, RCX, R8, R9
            0,                   // return in RAX
            11,                  // R11 for function pointer
            10,                  // R10 for cycle breaking / second spill scratch
        ),
        super::Target::Wasm => (&[], 0, 0, 0),
    }
}

// ---------------------------------------------------------------------------
// Address resolution
// ---------------------------------------------------------------------------

/// Names only JIT code uses: the tiering callbacks and the native
/// backends' shadow-root slot. A program's runtime exports none of them.
#[cfg(feature = "host")]
pub const JIT_ONLY_FN_NAMES: &[&str] = &[
    "wren_current_shadow_roots_ptr",
    "wren_cold_loop_hot",
    "wren_tier_tick",
    "wren_retier",
    "wren_deopt_n",
    "wren_deopt_at",
];

/// Every other name [`resolve`] answers: what a JIT module registers up
/// front, so a body links without a symbol lookup in the process, and
/// what an AOT program's runtime object must export.
#[cfg(any(feature = "host", feature = "aot_runtime"))]
pub const RUNTIME_FN_NAMES: &[&str] = &[
    "wren_get_module_var",
    "wren_set_module_var",
    "wren_call_0",
    "wren_call_1",
    "wren_call_2",
    "wren_call_3",
    "wren_call_4",
    "wren_call_5",
    "wren_call_6",
    "wren_call_7",
    "wren_call_8",
    "wren_call_dynamic",
    "wren_call_static_self_0",
    "wren_call_static_self_1",
    "wren_call_static_self_2",
    "wren_call_static_self_3",
    "wren_call_static_self_4",
    "wren_super_call_0",
    "wren_super_call_1",
    "wren_super_call_2",
    "wren_super_call_3",
    "wren_super_call_4",
    "wren_super_call_from_1",
    "wren_super_call_from_2",
    "wren_super_call_from_3",
    "wren_super_call_from_4",
    "wren_load_jit_ptr",
    "wren_load_jit_closure",
    "wren_aot_check_error",
    "wren_jit_roots_snapshot",
    "wren_jit_roots_restore",
    "wren_known_call_0",
    "wren_known_call_1",
    "wren_known_call_2",
    "wren_known_call_3",
    "wren_known_call_0_nocheck",
    "wren_known_call_1_nocheck",
    "wren_known_call_2_nocheck",
    "wren_known_call_3_nocheck",
    "wren_construct_0",
    "wren_construct_1",
    "wren_construct_2",
    "wren_construct_3",
    "wren_ic_call_0",
    "wren_ic_call_1",
    "wren_ic_call_2",
    "wren_ic_call_3",
    "wren_make_list",
    "wren_make_list_1",
    "wren_make_list_2",
    "wren_make_list_3",
    "wren_make_list_4",
    "wren_list_add",
    "wren_make_map",
    "wren_map_set",
    "wren_make_range",
    "wren_make_closure_0",
    "wren_make_closure_1",
    "wren_make_closure_2",
    "wren_make_closure_3",
    "wren_make_closure_4",
    "wren_make_closure_5",
    "wren_make_closure_6",
    "wren_make_closure_7",
    "wren_make_closure_8",
    "wren_make_closure_n",
    "wren_string_concat",
    "wren_osr_post",
    "wren_osr_take",
    "wren_to_string",
    "wren_const_string",
    "wren_is_type",
    "wren_guard_class",
    "wren_guard_protocol",
    "wren_subscript_get",
    "wren_subscript_set",
    "wren_num_add",
    "wren_num_sub",
    "wren_num_mul",
    "wren_num_div",
    "wren_num_mod",
    "wren_num_neg",
    "wren_bit_and",
    "wren_bit_or",
    "wren_bit_xor",
    "wren_bit_not",
    "wren_bit_shl",
    "wren_bit_shr",
    "wren_alloc_simd4f",
    "wren_alloc_simd4i",
    "wren_cmp_lt",
    "wren_cmp_gt",
    "wren_cmp_le",
    "wren_cmp_ge",
    "wren_cmp_eq",
    "wren_cmp_ne",
    "wren_not",
    "wren_is_truthy",
    "wren_get_upvalue",
    "wren_set_upvalue",
    "wren_shadow_store",
    "wren_shadow_load",
    "wren_enter_shadow_frame",
    "wren_exit_shadow_frame",
    "wren_get_static_field",
    "wren_set_static_field",
    "wren_fp_sin",
    "wren_fp_cos",
    "wren_fp_tan",
    "wren_fp_asin",
    "wren_fp_acos",
    "wren_fp_atan",
    "wren_fp_log",
    "wren_fp_log2",
    "wren_fp_exp",
    "wren_fp_cbrt",
    "wren_fp_atan2",
    "wren_fp_pow",
    "wren_fp_min",
    "wren_fp_max",
    "wren_jit_frame_push",
    "wren_jit_frame_pop",
    "wren_ic_enter",
    "wren_ic_leave",
    "wren_alloc_instance",
    "wren_ic_ctor_0",
    "wren_ic_ctor_1",
    "wren_ic_ctor_2",
    "wren_ic_ctor_3",
    "wren_ic_native_0",
    "wren_ic_native_1",
    "wren_ic_native_2",
    "wren_ic_native_3",
    "wren_ic_host_0",
    "wren_ic_host_1",
    "wren_ic_host_2",
    "wren_ic_host_3",
];

/// Resolve a runtime function name to its address.
/// Returns `None` if the name is unknown.
#[cfg(any(feature = "host", feature = "aot_runtime"))]
pub fn resolve(name: &str) -> Option<usize> {
    match name {
        "wren_get_module_var" => Some(wren_get_module_var as *const () as usize),
        "wren_set_module_var" => Some(wren_set_module_var as *const () as usize),
        // Arity-specific call dispatch
        "wren_call_0" => Some(wren_call_0 as *const () as usize),
        "wren_call_1" => Some(wren_call_1 as *const () as usize),
        "wren_call_2" => Some(wren_call_2 as *const () as usize),
        "wren_call_3" => Some(wren_call_3 as *const () as usize),
        "wren_call_4" => Some(wren_call_4 as *const () as usize),
        "wren_call_5" => Some(wren_call_5 as *const () as usize),
        "wren_call_6" => Some(wren_call_6 as *const () as usize),
        "wren_call_7" => Some(wren_call_7 as *const () as usize),
        "wren_call_8" => Some(wren_call_8 as *const () as usize),
        "wren_call_dynamic" => Some(wren_call_dynamic as *const () as usize),
        "wren_call_static_self_0" => Some(wren_call_static_self_0 as *const () as usize),
        "wren_call_static_self_1" => Some(wren_call_static_self_1 as *const () as usize),
        "wren_call_static_self_2" => Some(wren_call_static_self_2 as *const () as usize),
        "wren_call_static_self_3" => Some(wren_call_static_self_3 as *const () as usize),
        "wren_call_static_self_4" => Some(wren_call_static_self_4 as *const () as usize),
        "wren_super_call_0" => Some(wren_super_call_0 as *const () as usize),
        "wren_super_call_1" => Some(wren_super_call_1 as *const () as usize),
        "wren_super_call_2" => Some(wren_super_call_2 as *const () as usize),
        "wren_super_call_3" => Some(wren_super_call_3 as *const () as usize),
        "wren_super_call_4" => Some(wren_super_call_4 as *const () as usize),
        "wren_super_call_from_1" => Some(wren_super_call_from_1 as *const () as usize),
        "wren_super_call_from_2" => Some(wren_super_call_from_2 as *const () as usize),
        "wren_super_call_from_3" => Some(wren_super_call_from_3 as *const () as usize),
        "wren_super_call_from_4" => Some(wren_super_call_from_4 as *const () as usize),
        // Collections
        // Known-function dispatch (devirtualized)
        "wren_load_jit_ptr" => Some(wren_load_jit_ptr as *const () as usize),
        "wren_load_jit_closure" => Some(wren_load_jit_closure as *const () as usize),
        "wren_aot_check_error" => Some(wren_aot_check_error as *const () as usize),
        "wren_jit_roots_snapshot" => Some(wren_jit_roots_snapshot as *const () as usize),
        "wren_jit_roots_restore" => Some(wren_jit_roots_restore as *const () as usize),
        "wren_known_call_0" => Some(wren_known_call_0 as *const () as usize),
        "wren_known_call_1" => Some(wren_known_call_1 as *const () as usize),
        "wren_known_call_2" => Some(wren_known_call_2 as *const () as usize),
        "wren_known_call_3" => Some(wren_known_call_3 as *const () as usize),
        "wren_known_call_0_nocheck" => Some(wren_known_call_0_nocheck as *const () as usize),
        "wren_known_call_1_nocheck" => Some(wren_known_call_1_nocheck as *const () as usize),
        "wren_known_call_2_nocheck" => Some(wren_known_call_2_nocheck as *const () as usize),
        "wren_known_call_3_nocheck" => Some(wren_known_call_3_nocheck as *const () as usize),
        "wren_construct_0" => Some(wren_construct_0 as *const () as usize),
        "wren_construct_1" => Some(wren_construct_1 as *const () as usize),
        "wren_construct_2" => Some(wren_construct_2 as *const () as usize),
        "wren_construct_3" => Some(wren_construct_3 as *const () as usize),
        // Indirect IC dispatch (lightweight JIT-to-JIT calls)
        "wren_ic_call_0" => Some(wren_ic_call_0 as *const () as usize),
        "wren_ic_call_1" => Some(wren_ic_call_1 as *const () as usize),
        "wren_ic_call_2" => Some(wren_ic_call_2 as *const () as usize),
        "wren_ic_call_3" => Some(wren_ic_call_3 as *const () as usize),
        // Collections
        "wren_make_list" => Some(wren_make_list as *const () as usize),
        "wren_make_list_1" => Some(wren_make_list_1 as *const () as usize),
        "wren_make_list_2" => Some(wren_make_list_2 as *const () as usize),
        "wren_make_list_3" => Some(wren_make_list_3 as *const () as usize),
        "wren_make_list_4" => Some(wren_make_list_4 as *const () as usize),
        "wren_list_add" => Some(wren_list_add as *const () as usize),
        "wren_make_map" => Some(wren_make_map as *const () as usize),
        "wren_map_set" => Some(wren_map_set as *const () as usize),
        "wren_make_range" => Some(wren_make_range as *const () as usize),
        // Arity-specific closure creation
        "wren_make_closure_0" => Some(wren_make_closure_0 as *const () as usize),
        "wren_make_closure_1" => Some(wren_make_closure_1 as *const () as usize),
        "wren_make_closure_2" => Some(wren_make_closure_2 as *const () as usize),
        "wren_make_closure_3" => Some(wren_make_closure_3 as *const () as usize),
        "wren_make_closure_4" => Some(wren_make_closure_4 as *const () as usize),
        "wren_make_closure_5" => Some(wren_make_closure_5 as *const () as usize),
        "wren_make_closure_6" => Some(wren_make_closure_6 as *const () as usize),
        "wren_make_closure_7" => Some(wren_make_closure_7 as *const () as usize),
        "wren_make_closure_8" => Some(wren_make_closure_8 as *const () as usize),
        "wren_make_closure_n" => Some(wren_make_closure_n as *const () as usize),
        // Strings
        "wren_string_concat" => Some(wren_string_concat as *const () as usize),
        "wren_osr_post" => Some(wren_osr_post as *const () as usize),
        #[cfg(feature = "host")]
        "wren_cold_loop_hot" => Some(wren_cold_loop_hot as *const () as usize),
        "wren_osr_take" => Some(wren_osr_take as *const () as usize),
        #[cfg(feature = "host")]
        "wren_tier_tick" => Some(wren_tier_tick as *const () as usize),
        #[cfg(feature = "host")]
        "wren_retier" => Some(wren_retier as *const () as usize),
        "wren_to_string" => Some(wren_to_string as *const () as usize),
        "wren_const_string" => Some(wren_const_string as *const () as usize),
        // Type checks & guards
        "wren_is_type" => Some(wren_is_type as *const () as usize),
        "wren_guard_class" => Some(wren_guard_class as *const () as usize),
        "wren_guard_protocol" => Some(wren_guard_protocol as *const () as usize),
        // Guard deoptimization (arity-specific)
        #[cfg(feature = "host")]
        "wren_deopt_n" => Some(wren_deopt_n as *const () as usize),
        #[cfg(feature = "host")]
        "wren_deopt_at" => Some(wren_deopt_at as *const () as usize),
        // Subscript
        "wren_subscript_get" => Some(wren_subscript_get as *const () as usize),
        "wren_subscript_set" => Some(wren_subscript_set as *const () as usize),
        // Boxed arithmetic
        "wren_num_add" => Some(wren_num_add as *const () as usize),
        "wren_num_sub" => Some(wren_num_sub as *const () as usize),
        "wren_num_mul" => Some(wren_num_mul as *const () as usize),
        "wren_num_div" => Some(wren_num_div as *const () as usize),
        "wren_num_mod" => Some(wren_num_mod as *const () as usize),
        "wren_num_neg" => Some(wren_num_neg as *const () as usize),
        // Boxed bitwise
        "wren_bit_and" => Some(wren_bit_and as *const () as usize),
        "wren_bit_or" => Some(wren_bit_or as *const () as usize),
        "wren_bit_xor" => Some(wren_bit_xor as *const () as usize),
        "wren_bit_not" => Some(wren_bit_not as *const () as usize),
        "wren_bit_shl" => Some(wren_bit_shl as *const () as usize),
        "wren_bit_shr" => Some(wren_bit_shr as *const () as usize),
        "wren_alloc_simd4f" => Some(wren_alloc_simd4f as *const () as usize),
        "wren_alloc_simd4i" => Some(wren_alloc_simd4i as *const () as usize),
        // Boxed comparisons
        "wren_cmp_lt" => Some(wren_cmp_lt as *const () as usize),
        "wren_cmp_gt" => Some(wren_cmp_gt as *const () as usize),
        "wren_cmp_le" => Some(wren_cmp_le as *const () as usize),
        "wren_cmp_ge" => Some(wren_cmp_ge as *const () as usize),
        "wren_cmp_eq" => Some(wren_cmp_eq as *const () as usize),
        "wren_cmp_ne" => Some(wren_cmp_ne as *const () as usize),
        // Logical
        "wren_not" => Some(wren_not as *const () as usize),
        "wren_is_truthy" => Some(wren_is_truthy as *const () as usize),
        // Upvalues
        "wren_get_upvalue" => Some(wren_get_upvalue as *const () as usize),
        "wren_set_upvalue" => Some(wren_set_upvalue as *const () as usize),
        "wren_shadow_store" => Some(wren_shadow_store as *const () as usize),
        "wren_shadow_load" => Some(wren_shadow_load as *const () as usize),
        "wren_current_shadow_roots_ptr" => {
            Some(std::ptr::addr_of_mut!(CURRENT_NATIVE_SHADOW_ROOTS) as usize)
        }
        "wren_enter_shadow_frame" => Some(wren_enter_shadow_frame as *const () as usize),
        "wren_exit_shadow_frame" => Some(wren_exit_shadow_frame as *const () as usize),
        // Static fields
        "wren_get_static_field" => Some(wren_get_static_field as *const () as usize),
        "wren_set_static_field" => Some(wren_set_static_field as *const () as usize),
        // FP transcendentals (raw f64 bits in/out)
        "wren_fp_sin" => Some(wren_fp_sin as *const () as usize),
        "wren_fp_cos" => Some(wren_fp_cos as *const () as usize),
        "wren_fp_tan" => Some(wren_fp_tan as *const () as usize),
        "wren_fp_asin" => Some(wren_fp_asin as *const () as usize),
        "wren_fp_acos" => Some(wren_fp_acos as *const () as usize),
        "wren_fp_atan" => Some(wren_fp_atan as *const () as usize),
        "wren_fp_log" => Some(wren_fp_log as *const () as usize),
        "wren_fp_log2" => Some(wren_fp_log2 as *const () as usize),
        "wren_fp_exp" => Some(wren_fp_exp as *const () as usize),
        "wren_fp_cbrt" => Some(wren_fp_cbrt as *const () as usize),
        "wren_fp_atan2" => Some(wren_fp_atan2 as *const () as usize),
        "wren_fp_pow" => Some(wren_fp_pow as *const () as usize),
        "wren_fp_min" => Some(wren_fp_min as *const () as usize),
        "wren_fp_max" => Some(wren_fp_max as *const () as usize),
        // JIT frame registration for GC stack walking
        "wren_jit_frame_push" => Some(wren_jit_frame_push as *const () as usize),
        "wren_jit_frame_pop" => Some(wren_jit_frame_pop as *const () as usize),
        // Inline IC context swap for kind=6 (non-leaf JIT dispatch)
        "wren_ic_enter" => Some(wren_ic_enter as *const () as usize),
        "wren_ic_leave" => Some(wren_ic_leave as *const () as usize),
        // Inline constructor dispatch for kind=3
        "wren_alloc_instance" => Some(wren_alloc_instance as *const () as usize),
        "wren_ic_ctor_0" => Some(wren_ic_ctor_0 as *const () as usize),
        "wren_ic_ctor_1" => Some(wren_ic_ctor_1 as *const () as usize),
        "wren_ic_ctor_2" => Some(wren_ic_ctor_2 as *const () as usize),
        "wren_ic_ctor_3" => Some(wren_ic_ctor_3 as *const () as usize),
        // Inline IC native dispatch for kind=4
        "wren_ic_native_0" => Some(wren_ic_native_0 as *const () as usize),
        "wren_ic_native_1" => Some(wren_ic_native_1 as *const () as usize),
        "wren_ic_native_2" => Some(wren_ic_native_2 as *const () as usize),
        "wren_ic_native_3" => Some(wren_ic_native_3 as *const () as usize),
        // Inline IC host dispatch for kind=8
        "wren_ic_host_0" => Some(wren_ic_host_0 as *const () as usize),
        "wren_ic_host_1" => Some(wren_ic_host_1 as *const () as usize),
        "wren_ic_host_2" => Some(wren_ic_host_2 as *const () as usize),
        "wren_ic_host_3" => Some(wren_ic_host_3 as *const () as usize),
        _ => None,
    }
}

/// Register a JIT function's frame pointer for GC stack walking.
/// Called from JIT prologue with FP, func_id, and return address.
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_jit_frame_push(fp: u64, func_id: u64) {
    // Return address is at [fp + 8] (saved LR in the JIT frame).
    let ret_addr = if fp != 0 {
        unsafe { *((fp as usize + 8) as *const usize) }
    } else {
        0
    };
    push_jit_frame(fp as usize, func_id as u32, ret_addr);
}

/// Unregister a JIT function's frame pointer.
/// Called from JIT epilogue before return.
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_jit_frame_pop() {
    pop_jit_frame();
}

// ---------------------------------------------------------------------------
// Inline IC native dispatch (kind=4)
// ---------------------------------------------------------------------------

/// Thin wrapper for native method dispatch from inline IC.
/// Takes (native_fn_ptr, receiver, args..., jit_fp) — the jit_fp is captured
/// by #[naked] wrappers to register the JIT frame for GC.
// IC native trampolines transmute a `u64` carried in the IC entry
// to a `NativeFn` pointer — only sound on host targets where fn
// pointers are 64-bit. wasm32 has 32-bit fn pointers, and there's
// no JIT to populate the IC slot anyway, so the whole helper is
// host-only.
#[cfg(any(feature = "host", feature = "aot_runtime"))]
macro_rules! ic_native_inner {
    ($name:ident, $($arg:ident),*) => {
        extern "C" fn $name(native_fn: u64, recv: u64, $($arg: u64,)* jit_fp: u64) -> u64 {
            // Read return address from the JIT frame for precise safepoint scanning.
            let ret_addr = if jit_fp != 0 {
                unsafe { *((jit_fp as usize + 8) as *const usize) }
            } else { 0 };
            push_jit_frame(jit_fp as usize, read_jit_ctx().current_func_id as u32, ret_addr);
            let result = match unsafe { vm_ref() } {
                Some(vm) => {
                    let f: crate::runtime::object::NativeFn = unsafe { std::mem::transmute(native_fn as usize) };
                    f(vm, &[Value::from_bits(recv), $(Value::from_bits($arg)),*]).to_bits()
                }
                None => Value::null().to_bits(),
            };
            pop_jit_frame();
            result
        }
    };
}

#[cfg(any(feature = "host", feature = "aot_runtime"))]
ic_native_inner!(wren_ic_native_0_inner,);
#[cfg(any(feature = "host", feature = "aot_runtime"))]
ic_native_inner!(wren_ic_native_1_inner, a0);
#[cfg(any(feature = "host", feature = "aot_runtime"))]
ic_native_inner!(wren_ic_native_2_inner, a0, a1);
#[cfg(any(feature = "host", feature = "aot_runtime"))]
ic_native_inner!(wren_ic_native_3_inner, a0, a1, a2);

/// Host method dispatch from an inline IC hit (kind=8): the fn and its
/// context word come from the entry, the receiver and arguments from
/// the call. The thread's JIT state is read once.
#[cfg(any(feature = "host", feature = "aot_runtime"))]
macro_rules! ic_host_inner {
    ($name:ident, $($arg:ident),*) => {
        extern "C" fn $name(host_fn: u64, context: u64, recv: u64, $($arg: u64,)* jit_fp: u64) -> u64 {
            let ret_addr = if jit_fp != 0 {
                unsafe { *((jit_fp as usize + 8) as *const usize) }
            } else { 0 };
            let j = jit_state();
            let func_id = unsafe { (*j).ctx.current_func_id } as u32;
            push_frame_on(j, jit_fp as usize, func_id, ret_addr);
            let result = match unsafe { vm_at(j) } {
                Some(vm) => {
                    let f: crate::runtime::object::HostFn = unsafe { std::mem::transmute(host_fn as usize) };
                    let result = f(vm, context as usize, &[Value::from_bits(recv), $(Value::from_bits($arg)),*]).to_bits();
                    match vm.pending_fiber_action.take() {
                        Some(action) => handle_jit_fiber_action(vm, action),
                        None => result,
                    }
                }
                None => Value::null().to_bits(),
            };
            unsafe { (*j).frames.pop() };
            result
        }
    };
}

#[cfg(any(feature = "host", feature = "aot_runtime"))]
ic_host_inner!(wren_ic_host_0_inner,);
#[cfg(any(feature = "host", feature = "aot_runtime"))]
ic_host_inner!(wren_ic_host_1_inner, a0);
#[cfg(any(feature = "host", feature = "aot_runtime"))]
ic_host_inner!(wren_ic_host_2_inner, a0, a1);
#[cfg(any(feature = "host", feature = "aot_runtime"))]
ic_host_inner!(wren_ic_host_3_inner, a0, a1, a2);

/// # Safety
/// Called only from JIT-compiled code via inline IC dispatch (kind=8).
#[cfg(all(target_arch = "aarch64", feature = "host"))]
#[unsafe(naked)]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub unsafe extern "C" fn wren_ic_host_0(_hfn: u64, _ctx: u64, _recv: u64) -> u64 {
    core::arch::naked_asm!("mov x3, x29", "b {inner}", inner = sym wren_ic_host_0_inner);
}
#[cfg(all(
    not(target_arch = "aarch64"),
    any(feature = "host", feature = "aot_runtime")
))]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_ic_host_0(hfn: u64, ctx: u64, recv: u64) -> u64 {
    wren_ic_host_0_inner(hfn, ctx, recv, 0)
}

/// # Safety
/// Called only from JIT-compiled code via inline IC dispatch (kind=8).
#[cfg(all(target_arch = "aarch64", feature = "host"))]
#[unsafe(naked)]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub unsafe extern "C" fn wren_ic_host_1(_hfn: u64, _ctx: u64, _recv: u64, _a0: u64) -> u64 {
    core::arch::naked_asm!("mov x4, x29", "b {inner}", inner = sym wren_ic_host_1_inner);
}
#[cfg(all(
    not(target_arch = "aarch64"),
    any(feature = "host", feature = "aot_runtime")
))]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_ic_host_1(hfn: u64, ctx: u64, recv: u64, a0: u64) -> u64 {
    wren_ic_host_1_inner(hfn, ctx, recv, a0, 0)
}

/// # Safety
/// Called only from JIT-compiled code via inline IC dispatch (kind=8).
#[cfg(all(target_arch = "aarch64", feature = "host"))]
#[unsafe(naked)]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub unsafe extern "C" fn wren_ic_host_2(
    _hfn: u64,
    _ctx: u64,
    _recv: u64,
    _a0: u64,
    _a1: u64,
) -> u64 {
    core::arch::naked_asm!("mov x5, x29", "b {inner}", inner = sym wren_ic_host_2_inner);
}
#[cfg(all(
    not(target_arch = "aarch64"),
    any(feature = "host", feature = "aot_runtime")
))]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_ic_host_2(hfn: u64, ctx: u64, recv: u64, a0: u64, a1: u64) -> u64 {
    wren_ic_host_2_inner(hfn, ctx, recv, a0, a1, 0)
}

/// # Safety
/// Called only from JIT-compiled code via inline IC dispatch (kind=8).
#[cfg(all(target_arch = "aarch64", feature = "host"))]
#[unsafe(naked)]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub unsafe extern "C" fn wren_ic_host_3(
    _hfn: u64,
    _ctx: u64,
    _recv: u64,
    _a0: u64,
    _a1: u64,
    _a2: u64,
) -> u64 {
    core::arch::naked_asm!("mov x6, x29", "b {inner}", inner = sym wren_ic_host_3_inner);
}
#[cfg(all(
    not(target_arch = "aarch64"),
    any(feature = "host", feature = "aot_runtime")
))]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_ic_host_3(hfn: u64, ctx: u64, recv: u64, a0: u64, a1: u64, a2: u64) -> u64 {
    wren_ic_host_3_inner(hfn, ctx, recv, a0, a1, a2, 0)
}

// #[naked] wrappers capture x29 (JIT FP) as the last argument.
/// # Safety
/// Called only from JIT-compiled code via inline IC dispatch (kind=4).
#[cfg(all(target_arch = "aarch64", feature = "host"))]
#[unsafe(naked)]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub unsafe extern "C" fn wren_ic_native_0(_nfn: u64, _recv: u64) -> u64 {
    core::arch::naked_asm!("mov x2, x29", "b {inner}", inner = sym wren_ic_native_0_inner);
}
#[cfg(all(
    not(target_arch = "aarch64"),
    any(feature = "host", feature = "aot_runtime")
))]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_ic_native_0(nfn: u64, recv: u64) -> u64 {
    wren_ic_native_0_inner(nfn, recv, 0)
}

/// # Safety
/// Called only from JIT-compiled code via inline IC dispatch (kind=4).
#[cfg(all(target_arch = "aarch64", feature = "host"))]
#[unsafe(naked)]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub unsafe extern "C" fn wren_ic_native_1(_nfn: u64, _recv: u64, _a0: u64) -> u64 {
    core::arch::naked_asm!("mov x3, x29", "b {inner}", inner = sym wren_ic_native_1_inner);
}
#[cfg(all(
    not(target_arch = "aarch64"),
    any(feature = "host", feature = "aot_runtime")
))]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_ic_native_1(nfn: u64, recv: u64, a0: u64) -> u64 {
    wren_ic_native_1_inner(nfn, recv, a0, 0)
}

/// # Safety
/// Called only from JIT-compiled code via inline IC dispatch (kind=4).
#[cfg(all(target_arch = "aarch64", feature = "host"))]
#[unsafe(naked)]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub unsafe extern "C" fn wren_ic_native_2(_nfn: u64, _recv: u64, _a0: u64, _a1: u64) -> u64 {
    core::arch::naked_asm!("mov x4, x29", "b {inner}", inner = sym wren_ic_native_2_inner);
}
#[cfg(all(
    not(target_arch = "aarch64"),
    any(feature = "host", feature = "aot_runtime")
))]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_ic_native_2(nfn: u64, recv: u64, a0: u64, a1: u64) -> u64 {
    wren_ic_native_2_inner(nfn, recv, a0, a1, 0)
}

/// # Safety
/// Called only from JIT-compiled code via inline IC dispatch (kind=4).
#[cfg(all(target_arch = "aarch64", feature = "host"))]
#[unsafe(naked)]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub unsafe extern "C" fn wren_ic_native_3(
    _nfn: u64,
    _recv: u64,
    _a0: u64,
    _a1: u64,
    _a2: u64,
) -> u64 {
    core::arch::naked_asm!("mov x5, x29", "b {inner}", inner = sym wren_ic_native_3_inner);
}
#[cfg(all(
    not(target_arch = "aarch64"),
    any(feature = "host", feature = "aot_runtime")
))]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_ic_native_3(nfn: u64, recv: u64, a0: u64, a1: u64, a2: u64) -> u64 {
    wren_ic_native_3_inner(nfn, recv, a0, a1, a2, 0)
}

/// Get the stable address of the thread-local JitContext.
/// Used to set x19 at interpreter→JIT entry points.
#[inline(always)]
pub fn jit_ctx_ptr() -> *mut JitContext {
    unsafe { &mut (*jit_state()).ctx as *mut JitContext }
}

// ---------------------------------------------------------------------------
// Inline IC constructor dispatch (kind=3)
// ---------------------------------------------------------------------------

/// Single-call constructor dispatch: alloc instance + call constructor body.
/// Replaces the 50-function dispatch chain with one Rust function call.
macro_rules! ic_ctor_inner {
    ($name:ident, $($arg:ident),*) => {
        extern "C" fn $name(class_val: u64, closure: u64, $($arg: u64,)* jit_fp: u64) -> u64 {
            push_jit_frame(jit_fp as usize, read_jit_ctx().current_func_id as u32,
                if jit_fp != 0 { unsafe { *((jit_fp as usize + 8) as *const usize) } } else { 0 });
            let result = match unsafe { vm_ref() } {
                Some(vm) => {
                    let class_ptr = Value::from_bits(class_val)
                        .as_object()
                        .unwrap_or(std::ptr::null_mut()) as *mut ObjClass;
                    if class_ptr.is_null() {
                        Value::null().to_bits()
                    } else {
                        let closure_ptr = closure as *mut ObjClosure;
                        let ctor_args = &[$(Value::from_bits($arg)),*];
                        vm.call_constructor_sync(class_ptr, closure_ptr, ctor_args).to_bits()
                    }
                }
                None => Value::null().to_bits(),
            };
            pop_jit_frame();
            result
        }
    };
}

ic_ctor_inner!(wren_ic_ctor_0_inner,);
ic_ctor_inner!(wren_ic_ctor_1_inner, a0);
ic_ctor_inner!(wren_ic_ctor_2_inner, a0, a1);
ic_ctor_inner!(wren_ic_ctor_3_inner, a0, a1, a2);

/// # Safety
/// Called only from JIT-compiled code via inline IC constructor dispatch (kind=3).
#[cfg(all(target_arch = "aarch64", feature = "host"))]
#[unsafe(naked)]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub unsafe extern "C" fn wren_ic_ctor_0(_cls: u64, _closure: u64) -> u64 {
    core::arch::naked_asm!("mov x2, x29", "b {inner}", inner = sym wren_ic_ctor_0_inner);
}
#[cfg(all(
    not(target_arch = "aarch64"),
    any(feature = "host", feature = "aot_runtime")
))]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_ic_ctor_0(cls: u64, closure: u64) -> u64 {
    wren_ic_ctor_0_inner(cls, closure, 0)
}

/// # Safety
/// Called only from JIT-compiled code via inline IC constructor dispatch (kind=3).
#[cfg(all(target_arch = "aarch64", feature = "host"))]
#[unsafe(naked)]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub unsafe extern "C" fn wren_ic_ctor_1(_cls: u64, _closure: u64, _a0: u64) -> u64 {
    core::arch::naked_asm!("mov x3, x29", "b {inner}", inner = sym wren_ic_ctor_1_inner);
}
#[cfg(all(
    not(target_arch = "aarch64"),
    any(feature = "host", feature = "aot_runtime")
))]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_ic_ctor_1(cls: u64, closure: u64, a0: u64) -> u64 {
    wren_ic_ctor_1_inner(cls, closure, a0, 0)
}

/// # Safety
/// Called only from JIT-compiled code via inline IC constructor dispatch (kind=3).
#[cfg(all(target_arch = "aarch64", feature = "host"))]
#[unsafe(naked)]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub unsafe extern "C" fn wren_ic_ctor_2(_cls: u64, _closure: u64, _a0: u64, _a1: u64) -> u64 {
    core::arch::naked_asm!("mov x4, x29", "b {inner}", inner = sym wren_ic_ctor_2_inner);
}
#[cfg(all(
    not(target_arch = "aarch64"),
    any(feature = "host", feature = "aot_runtime")
))]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_ic_ctor_2(cls: u64, closure: u64, a0: u64, a1: u64) -> u64 {
    wren_ic_ctor_2_inner(cls, closure, a0, a1, 0)
}

/// # Safety
/// Called only from JIT-compiled code via inline IC constructor dispatch (kind=3).
#[cfg(all(target_arch = "aarch64", feature = "host"))]
#[unsafe(naked)]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub unsafe extern "C" fn wren_ic_ctor_3(
    _cls: u64,
    _closure: u64,
    _a0: u64,
    _a1: u64,
    _a2: u64,
) -> u64 {
    core::arch::naked_asm!("mov x5, x29", "b {inner}", inner = sym wren_ic_ctor_3_inner);
}
#[cfg(all(
    not(target_arch = "aarch64"),
    any(feature = "host", feature = "aot_runtime")
))]
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_ic_ctor_3(cls: u64, closure: u64, a0: u64, a1: u64, a2: u64) -> u64 {
    wren_ic_ctor_3_inner(cls, closure, a0, a1, a2, 0)
}

/// A fresh instance of the class object in `class_val`, ready for its
/// initialiser, which compiled code then calls directly.
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_alloc_instance(class_val: u64) -> u64 {
    let class_ptr = Value::from_bits(class_val)
        .as_object()
        .unwrap_or(std::ptr::null_mut()) as *mut ObjClass;
    if class_ptr.is_null() {
        return Value::null().to_bits();
    }
    match unsafe { vm_ref() } {
        Some(vm) => {
            let instance = vm.gc.alloc_instance(class_ptr);
            unsafe { finish_alloc(vm, Value::object(instance as *mut u8)) }
        }
        None => Value::null().to_bits(),
    }
}

/// Allocate a `Simd4f` from raw lane bits.
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_alloc_simd4f(l0: u64, l1: u64, l2: u64, l3: u64) -> u64 {
    match unsafe { vm_ref() } {
        Some(vm) => vm
            .new_simd(
                SimdKind::F32x4,
                [l0 as u32, l1 as u32, l2 as u32, l3 as u32],
            )
            .to_bits(),
        None => Value::null().to_bits(),
    }
}

/// Allocate a `Simd4i` from raw lane bits.
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_alloc_simd4i(l0: u64, l1: u64, l2: u64, l3: u64) -> u64 {
    match unsafe { vm_ref() } {
        Some(vm) => vm
            .new_simd(
                SimdKind::I32x4,
                [l0 as u32, l1 as u32, l2 as u32, l3 as u32],
            )
            .to_bits(),
        None => Value::null().to_bits(),
    }
}

/// Check if a function is a trivial getter (get_field + return).
/// Used by speculative inlining to inline getter bodies at call sites.
pub fn trivial_getter_check(func_id: crate::runtime::engine::FuncId) -> Option<u16> {
    use crate::mir::{Instruction, Terminator};
    // Access the MIR through thread-local JIT context
    let ctx = read_jit_ctx();
    let vm = unsafe {
        if ctx.vm.is_null() {
            return None;
        }
        &*(ctx.vm as *const crate::runtime::vm::VM)
    };
    let mir = vm.engine.get_mir(func_id)?;
    if mir.blocks.len() != 1 {
        return None;
    }
    let block = &mir.blocks[0];
    let mut self_param = None;
    let mut getter = None;
    for (_vid, inst) in &block.instructions {
        match inst {
            Instruction::BlockParam(0) if self_param.is_none() => self_param = Some(*_vid),
            Instruction::GetField(recv, idx) if getter.is_none() && Some(*recv) == self_param => {
                getter = Some((*_vid, *idx));
            }
            _ => return None,
        }
    }
    match (getter, &block.terminator) {
        (Some((ret_vid, field_idx)), Terminator::Return(v)) if *v == ret_vid => Some(field_idx),
        _ => None,
    }
}

/// Inline IC enter: set JitContext for a non-leaf JIT call (kind=6).
/// Saves current_func_id, sets new func_id + closure. Returns saved_func_id.
/// Skips depth tracking — native stack overflow is the backstop for infinite recursion.
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_ic_enter(func_id: u64, closure: u64) -> u64 {
    unsafe {
        let ctx = &mut (*jit_state()).ctx;
        let saved = ctx.current_func_id;
        ctx.current_func_id = func_id;
        ctx.closure = closure as *mut u8;
        saved
    }
}

/// Inline IC leave: restore current_func_id after a non-leaf JIT call.
#[cfg_attr(
    any(not(target_arch = "wasm32"), feature = "aot_runtime"),
    unsafe(no_mangle)
)]
pub extern "C" fn wren_ic_leave(saved_func_id: u64) {
    unsafe {
        (*jit_state()).ctx.current_func_id = saved_func_id;
    }
}

// ---------------------------------------------------------------------------
// Liveness analysis for post-regalloc instruction stream
// ---------------------------------------------------------------------------

/// Compute live-out GP registers for every instruction via iterative dataflow.
///
/// Returns a Vec where `result[i]` is the set of GP register indices that are
/// live immediately after instruction `i` executes.
#[allow(dead_code)]
fn compute_live_out(insts: &[MachInst]) -> Vec<std::collections::HashSet<u32>> {
    use std::collections::{HashMap, HashSet};

    let n = insts.len();
    if n == 0 {
        return vec![];
    }

    // Map label id → instruction index
    let mut label_to_idx: HashMap<u32, usize> = HashMap::new();
    for (i, inst) in insts.iter().enumerate() {
        if let MachInst::DefLabel(lbl) = inst {
            label_to_idx.insert(lbl.0, i);
        }
    }

    // Helper: get jump targets for an instruction
    fn jump_targets(inst: &MachInst) -> Vec<u32> {
        match inst {
            MachInst::Jmp { target } => vec![target.0],
            MachInst::JmpIf { target, .. } => vec![target.0],
            MachInst::JmpZero { target, .. } => vec![target.0],
            MachInst::JmpNonZero { target, .. } => vec![target.0],
            MachInst::TestBitJmpZero { target, .. } => vec![target.0],
            MachInst::TestBitJmpNonZero { target, .. } => vec![target.0],
            _ => vec![],
        }
    }

    // Determine if instruction falls through to the next
    fn falls_through(inst: &MachInst) -> bool {
        !matches!(inst, MachInst::Jmp { .. } | MachInst::Ret | MachInst::Trap)
    }

    // Precompute uses/defs per instruction (GP only)
    let mut inst_uses: Vec<HashSet<u32>> = Vec::with_capacity(n);
    let mut inst_defs: Vec<HashSet<u32>> = Vec::with_capacity(n);
    for inst in insts {
        let mut uses = HashSet::new();
        for u in inst.uses() {
            if u.class == super::RegClass::Gp {
                uses.insert(u.index);
            }
        }
        inst_uses.push(uses);

        let mut defs = HashSet::new();
        if let Some(d) = inst.def()
            && d.class == super::RegClass::Gp
        {
            defs.insert(d.index);
        }
        inst_defs.push(defs);
    }

    // Build successor map
    let mut successors: Vec<Vec<usize>> = Vec::with_capacity(n);
    for (i, inst) in insts.iter().enumerate() {
        let mut succs = Vec::new();
        for t in jump_targets(inst) {
            if let Some(&idx) = label_to_idx.get(&t) {
                succs.push(idx);
            }
        }
        if falls_through(inst) && i + 1 < n {
            succs.push(i + 1);
        }
        successors.push(succs);
    }

    // Iterative dataflow: live_in[i] = use[i] ∪ (live_out[i] - def[i])
    //                     live_out[i] = ∪ live_in[s] for s in succ(i)
    let mut live_in: Vec<HashSet<u32>> = vec![HashSet::new(); n];
    let mut live_out: Vec<HashSet<u32>> = vec![HashSet::new(); n];

    loop {
        let mut changed = false;
        // Process in reverse for faster convergence
        for i in (0..n).rev() {
            // live_out[i] = union of live_in[succ]
            let mut new_out = HashSet::new();
            for &s in &successors[i] {
                new_out.extend(&live_in[s]);
            }

            // live_in[i] = use[i] ∪ (live_out[i] - def[i])
            let mut new_in = new_out.clone();
            for d in &inst_defs[i] {
                new_in.remove(d);
            }
            new_in.extend(&inst_uses[i]);

            if new_in != live_in[i] || new_out != live_out[i] {
                live_in[i] = new_in;
                live_out[i] = new_out;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    live_out
}

// ---------------------------------------------------------------------------
// CallRuntime → ABI setup + CallInd lowering pass
// ---------------------------------------------------------------------------

use crate::codegen::{Label, MachFunc, MachInst, Mem, VReg};

/// Lower all `CallRuntime` instructions after register allocation, using the
/// final register/spill assignments to build correct ABI setup sequences.
///
/// Register-sourced arguments are scheduled with a physical-register parallel
/// copy resolver, while spilled arguments are loaded directly from their stack
/// slots into ABI argument registers after any clobber-sensitive register moves.
#[cfg(any(feature = "host", feature = "aot_runtime"))]
pub fn link_runtime_calls(
    mf: &mut MachFunc,
    target: super::Target,
    assignments: &std::collections::HashMap<VReg, crate::codegen::regalloc::Location>,
) {
    let (abi_args, abi_ret, call_scratch, copy_scratch) = abi_regs(target);
    if abi_args.is_empty() {
        return;
    }
    let frame_ptr = match target {
        super::Target::X86_64 => VReg::gp(5),
        super::Target::Aarch64 => VReg::gp(29),
        super::Target::Wasm => return,
    };

    enum PendingCall {
        Runtime(usize, &'static str, Vec<VReg>, Option<VReg>),
        Local(usize, Label, Vec<VReg>, Option<VReg>),
        Indirect(usize, VReg, Vec<VReg>, Option<VReg>),
    }

    let mut call_sites: Vec<PendingCall> = Vec::new();
    for (orig_i, inst) in mf.insts.iter().enumerate() {
        match inst {
            MachInst::CallRuntime { name, args, ret } => {
                call_sites.push(PendingCall::Runtime(orig_i, name, args.clone(), *ret));
            }
            MachInst::CallLocal { target, args, ret } => {
                call_sites.push(PendingCall::Local(orig_i, *target, args.clone(), *ret));
            }
            MachInst::CallIndirectAbi { target, args, ret } => {
                call_sites.push(PendingCall::Indirect(orig_i, *target, args.clone(), *ret));
            }
            _ => {}
        }
    }

    for call_site in call_sites.into_iter().rev() {
        let (orig_i, new_insts) = match call_site {
            PendingCall::Runtime(orig_i, name, args, ret) => {
                let Some(addr) = resolve(name) else {
                    continue;
                };
                let new_insts = lower_linked_call(
                    &args,
                    ret,
                    abi_args,
                    abi_ret,
                    call_scratch,
                    copy_scratch,
                    frame_ptr,
                    assignments,
                    LinkedCallTarget::Runtime(addr as u64),
                );
                (orig_i, new_insts)
            }
            PendingCall::Local(orig_i, label, args, ret) => {
                let new_insts = lower_linked_call(
                    &args,
                    ret,
                    abi_args,
                    abi_ret,
                    call_scratch,
                    copy_scratch,
                    frame_ptr,
                    assignments,
                    LinkedCallTarget::Local(label),
                );
                (orig_i, new_insts)
            }
            PendingCall::Indirect(orig_i, target_vreg, args, ret) => {
                let new_insts = lower_linked_call(
                    &args,
                    ret,
                    abi_args,
                    abi_ret,
                    call_scratch,
                    copy_scratch,
                    frame_ptr,
                    assignments,
                    LinkedCallTarget::Indirect(target_vreg),
                );
                (orig_i, new_insts)
            }
        };

        mf.insts.splice(orig_i..=orig_i, new_insts);
    }
}

enum LinkedCallTarget {
    Runtime(u64),
    Local(Label),
    Indirect(VReg),
}

#[allow(clippy::too_many_arguments)]
#[cfg(any(feature = "host", feature = "aot_runtime"))]
fn lower_linked_call(
    args: &[VReg],
    ret: Option<VReg>,
    abi_args: &[u32],
    abi_ret: u32,
    call_scratch: u32,
    copy_scratch: u32,
    frame_ptr: VReg,
    assignments: &std::collections::HashMap<VReg, crate::codegen::regalloc::Location>,
    target: LinkedCallTarget,
) -> Vec<MachInst> {
    let mut reg_moves: Vec<(u32, u32)> = Vec::new();
    let mut spill_loads: Vec<(i32, u32)> = Vec::new();

    for (idx, vreg) in args.iter().enumerate().take(abi_args.len()) {
        let dst = abi_args[idx];
        match assignments.get(vreg) {
            Some(crate::codegen::regalloc::Location::Reg(phys)) => {
                reg_moves.push((phys.hw_enc as u32, dst));
            }
            Some(crate::codegen::regalloc::Location::Spill(offset)) => {
                spill_loads.push((*offset, dst));
            }
            None => reg_moves.push((vreg.index, dst)),
        }
    }

    let resolved = resolve_parallel_copy(&reg_moves, copy_scratch);
    let mut new_insts: Vec<MachInst> = Vec::new();
    for (src, dst) in resolved {
        new_insts.push(MachInst::Mov {
            dst: VReg::gp(dst),
            src: VReg::gp(src),
        });
    }

    for (offset, dst) in spill_loads {
        new_insts.push(MachInst::Ldr {
            dst: VReg::gp(dst),
            mem: Mem::new(frame_ptr, offset),
        });
    }

    match target {
        LinkedCallTarget::Runtime(addr) => {
            new_insts.push(MachInst::LoadImm {
                dst: VReg::gp(call_scratch),
                bits: addr,
            });
            new_insts.push(MachInst::CallInd {
                target: VReg::gp(call_scratch),
            });
        }
        LinkedCallTarget::Local(label) => {
            new_insts.push(MachInst::CallLabel { target: label });
        }
        LinkedCallTarget::Indirect(target_vreg) => {
            match assignments.get(&target_vreg) {
                Some(crate::codegen::regalloc::Location::Reg(phys))
                    if phys.hw_enc as u32 != call_scratch =>
                {
                    new_insts.insert(
                        0,
                        MachInst::Mov {
                            dst: VReg::gp(call_scratch),
                            src: VReg::gp(phys.hw_enc as u32),
                        },
                    );
                }
                Some(crate::codegen::regalloc::Location::Spill(offset)) => {
                    new_insts.insert(
                        0,
                        MachInst::Ldr {
                            dst: VReg::gp(call_scratch),
                            mem: Mem::new(frame_ptr, *offset),
                        },
                    );
                }
                _ => {}
            }
            new_insts.push(MachInst::CallInd {
                target: VReg::gp(call_scratch),
            });
        }
    }

    if let Some(ret_vreg) = ret {
        match assignments.get(&ret_vreg) {
            Some(crate::codegen::regalloc::Location::Reg(phys))
                if phys.hw_enc as u32 != abi_ret =>
            {
                new_insts.push(MachInst::Mov {
                    dst: VReg::gp(phys.hw_enc as u32),
                    src: VReg::gp(abi_ret),
                });
            }
            Some(crate::codegen::regalloc::Location::Spill(offset)) => {
                new_insts.push(MachInst::Str {
                    src: VReg::gp(abi_ret),
                    mem: Mem::new(frame_ptr, *offset),
                });
            }
            _ => {}
        }
    }

    // Shadow reloads removed — GC writes forwarded pointers directly to
    // spill slots via stack map write-back. JIT frame is registered by
    // #[naked] wren_call_N wrappers.

    new_insts
}

#[allow(dead_code)]
#[cfg(any(feature = "host", feature = "aot_runtime"))]
fn target_data_addr(name: &'static str) -> u64 {
    resolve(name).unwrap_or(0) as u64
}

/// Resolve a set of parallel register moves so no source is clobbered before
/// it is read. Uses a scratch register to break cycles.
///
/// Input: slice of (src_phys, dst_phys) pairs.
/// Output: ordered sequence of (src, dst) moves that is safe to execute sequentially.
fn resolve_parallel_copy(moves: &[(u32, u32)], scratch: u32) -> Vec<(u32, u32)> {
    // Filter identity moves (src == dst).
    let mut remaining: Vec<(u32, u32)> = moves.iter().filter(|(s, d)| s != d).cloned().collect();

    if remaining.is_empty() {
        return vec![];
    }

    let mut result = Vec::new();

    // Emit moves whose destination is NOT a source of any other move.
    // These are safe because writing the destination won't clobber anything.
    let mut changed = true;
    while changed {
        changed = false;
        let mut i = 0;
        while i < remaining.len() {
            let (_, dst) = remaining[i];
            let dst_is_source = remaining
                .iter()
                .enumerate()
                .any(|(j, (s, _))| j != i && *s == dst);
            if !dst_is_source {
                result.push(remaining.remove(i));
                changed = true;
            } else {
                i += 1;
            }
        }
    }

    // Only cycles remain. Break each cycle using the scratch register.
    while !remaining.is_empty() {
        let (first_src, first_dst) = remaining.remove(0);
        // Save first source to scratch so its register can be overwritten.
        result.push((first_src, scratch));

        // Follow the cycle chain, collecting moves.
        let mut chain = Vec::new();
        let mut current = first_dst;
        while let Some(idx) = remaining.iter().position(|(s, _)| *s == current) {
            let (_, next_dst) = remaining.remove(idx);
            chain.push((current, next_dst));
            current = next_dst;
        }

        // Emit chain moves in REVERSE order so each destination is written
        // before its register is read as a source by a subsequent move.
        // Example cycle r2→r0→r1→r2: after saving r2 to scratch,
        // chain = [(r0,r1), (r1,r2)]. Reversed: (r1,r2) then (r0,r1).
        // This writes r2=r1 first (r2 is free), then r1=r0 (r1 is free).
        for &(src, dst) in chain.iter().rev() {
            result.push((src, dst));
        }

        // Close the cycle: move from scratch to the first destination.
        result.push((scratch, first_dst));
    }

    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::regalloc::Location;
    use crate::codegen::{PhysReg, Target};

    #[test]
    fn resolve_known_functions() {
        assert!(resolve("wren_get_module_var").is_some());
        assert!(resolve("wren_call_0").is_some());
        assert!(resolve("wren_make_list").is_some());
        assert!(resolve("wren_subscript_get").is_some());
    }

    #[test]
    fn every_listed_name_resolves() {
        for name in RUNTIME_FN_NAMES.iter().chain(JIT_ONLY_FN_NAMES) {
            assert!(resolve(name).is_some(), "{name}");
        }
    }

    #[test]
    fn resolve_unknown_returns_none() {
        assert!(resolve("nonexistent_fn").is_none());
    }

    #[test]
    fn function_pointers_are_nonzero() {
        assert_ne!(resolve("wren_get_module_var").unwrap(), 0);
        assert_ne!(resolve("wren_call_0").unwrap(), 0);
    }

    #[test]
    fn test_jit_context_thread_local() {
        // Default context has null VM
        assert!(with_context(|_| ()).is_none());

        // Set a context
        let mut dummy_vm = [0u8; 1024];
        set_jit_context(JitContext {
            module_vars: std::ptr::null_mut(),
            module_var_count: 0,
            vm: dummy_vm.as_mut_ptr(),
            module_name: std::ptr::null(),
            module_name_len: 0,
            current_func_id: u32::MAX as u64,
            closure: std::ptr::null_mut(),
            defining_class: std::ptr::null_mut(),
            jit_code_base: std::ptr::null(),
            jit_code_len: 0,
        });
        assert!(with_context(|_| ()).is_some());

        // Clean up
        set_jit_context(JitContext::default());
    }

    #[test]
    fn test_get_module_var_with_context() {
        let mut vars: Vec<u64> = vec![
            Value::num(42.0).to_bits(),
            Value::bool(true).to_bits(),
            Value::null().to_bits(),
        ];
        let mut dummy_vm = [0u8; 4096];

        set_jit_context(JitContext {
            module_vars: vars.as_mut_ptr(),
            module_var_count: 3,
            vm: dummy_vm.as_mut_ptr(),
            module_name: std::ptr::null(),
            module_name_len: 0,
            current_func_id: u32::MAX as u64,
            closure: std::ptr::null_mut(),
            defining_class: std::ptr::null_mut(),
            jit_code_base: std::ptr::null(),
            jit_code_len: 0,
        });

        // Read first module var (num 42)
        let result = wren_get_module_var(0);
        assert_eq!(result, Value::num(42.0).to_bits());

        // Read second (bool true)
        let result = wren_get_module_var(1);
        assert_eq!(result, Value::bool(true).to_bits());

        // Out of bounds
        let result = wren_get_module_var(99);
        assert_eq!(result, Value::null().to_bits());

        // Clean up
        set_jit_context(JitContext::default());
    }

    #[test]
    fn test_set_module_var_with_context() {
        let mut vars: Vec<u64> = vec![Value::null().to_bits(); 3];
        let mut dummy_vm = [0u8; 4096];

        set_jit_context(JitContext {
            module_vars: vars.as_mut_ptr(),
            module_var_count: 3,
            vm: dummy_vm.as_mut_ptr(),
            module_name: std::ptr::null(),
            module_name_len: 0,
            current_func_id: u32::MAX as u64,
            closure: std::ptr::null_mut(),
            defining_class: std::ptr::null_mut(),
            jit_code_base: std::ptr::null(),
            jit_code_len: 0,
        });

        let new_val = Value::num(99.0).to_bits();
        wren_set_module_var(1, new_val);

        // Verify the module var was updated
        assert_eq!(vars[1], new_val);

        // Clean up
        set_jit_context(JitContext::default());
    }

    #[test]
    fn test_saved_jit_context_roots_round_trip() {
        clear_jit_roots();

        let mut dummy_vm = [0u8; 8];
        let mut old_closure = [0u8; 8];
        let mut old_class = [0u8; 8];
        let saved_ctx = JitContext {
            module_vars: std::ptr::null_mut(),
            module_var_count: 0,
            vm: dummy_vm.as_mut_ptr(),
            module_name: std::ptr::null(),
            module_name_len: 0,
            current_func_id: 7,
            closure: old_closure.as_mut_ptr(),
            defining_class: old_class.as_mut_ptr(),
            jit_code_base: std::ptr::null(),
            jit_code_len: 0,
        };

        let root_len_before = root_saved_jit_context(saved_ctx);
        let mut forwarded_closure = [0u8; 8];
        let mut forwarded_class = [0u8; 8];
        let mut roots = take_jit_roots();
        roots[root_len_before] = Value::object(forwarded_closure.as_mut_ptr());
        roots[root_len_before + 1] = Value::object(forwarded_class.as_mut_ptr());
        set_jit_roots(roots);

        restore_rooted_jit_context(saved_ctx, root_len_before);

        let restored = read_jit_ctx();
        assert_eq!(restored.closure, forwarded_closure.as_mut_ptr());
        assert_eq!(restored.defining_class, forwarded_class.as_mut_ptr());
        assert_eq!(jit_roots_snapshot_len(), 0);

        clear_jit_roots();
        set_jit_context(JitContext::default());
    }

    #[test]
    fn test_link_inserts_abi_moves_aarch64() {
        // Simulate post-regalloc state: args already in physical registers.
        let mut mf = MachFunc::new("test".into());
        let arg0 = VReg::gp(10);
        let arg1 = VReg::gp(11);
        let ret = VReg::gp(12);

        mf.emit(MachInst::LoadImm {
            dst: arg0,
            bits: 10,
        });
        mf.emit(MachInst::LoadImm {
            dst: arg1,
            bits: 20,
        });
        mf.emit(MachInst::CallRuntime {
            name: "wren_num_add",
            args: vec![arg0, arg1],
            ret: Some(ret),
        });
        mf.emit(MachInst::Ret);

        let before_len = mf.insts.len();
        let assignments = std::collections::HashMap::from([
            (arg0, Location::Reg(PhysReg::gp(3))),
            (arg1, Location::Reg(PhysReg::gp(5))),
            (ret, Location::Reg(PhysReg::gp(3))),
        ]);
        link_runtime_calls(&mut mf, Target::Aarch64, &assignments);

        // CallRuntime replaced: 2 arg movs + LoadImm + CallInd + ret mov = 5 new
        assert_eq!(mf.insts.len(), before_len + 4);

        let has_x0_move = mf.insts.iter().any(
            |inst| matches!(inst, MachInst::Mov { dst, src } if dst.index == 0 && src.index == 3),
        );
        assert!(has_x0_move, "should move arg0 into X0");

        let has_x1_move = mf.insts.iter().any(
            |inst| matches!(inst, MachInst::Mov { dst, src } if dst.index == 1 && src.index == 5),
        );
        assert!(has_x1_move, "should move arg1 into X1");
    }

    #[test]
    fn test_link_uses_call_scratch_sentinel() {
        let mut mf = MachFunc::new("test".into());
        let arg0 = VReg::gp(10);
        let arg1 = VReg::gp(11);

        mf.emit(MachInst::CallRuntime {
            name: "wren_num_add",
            args: vec![arg0, arg1],
            ret: None,
        });

        let assignments = std::collections::HashMap::from([
            (arg0, Location::Reg(PhysReg::gp(3))),
            (arg1, Location::Reg(PhysReg::gp(4))),
        ]);
        link_runtime_calls(&mut mf, Target::Aarch64, &assignments);

        let has_call_scratch = mf
            .insts
            .iter()
            .any(|inst| matches!(inst, MachInst::LoadImm { dst, .. } if dst.index == 16));
        let has_call_ind = mf
            .insts
            .iter()
            .any(|inst| matches!(inst, MachInst::CallInd { target } if target.index == 16));

        assert!(
            has_call_scratch,
            "call lowering should use the call scratch sentinel"
        );
        assert!(
            has_call_ind,
            "call lowering should indirect through the call scratch sentinel"
        );
    }

    #[test]
    fn test_runtime_call_parallel_copy_resolves_with_abi_overlap() {
        let mut mf = MachFunc::new("test".into());
        let recv = VReg::gp(10);
        let method = VReg::gp(11);
        let arg = VReg::gp(12);

        mf.emit(MachInst::CallRuntime {
            name: "wren_call_1",
            args: vec![recv, method, arg],
            ret: None,
        });

        let assignments = std::collections::HashMap::from([
            (recv, Location::Reg(PhysReg::gp(2))),
            (method, Location::Reg(PhysReg::gp(0))),
            (arg, Location::Reg(PhysReg::gp(1))),
        ]);
        link_runtime_calls(&mut mf, Target::Aarch64, &assignments);

        let mut regs = std::collections::HashMap::<u32, &'static str>::from([
            (0, "method"),
            (1, "arg"),
            (2, "recv"),
            (17, "scratch"),
        ]);

        for inst in &mf.insts {
            match inst {
                MachInst::Mov { dst, src } => {
                    let val = regs.get(&src.index).copied().unwrap_or("unknown");
                    regs.insert(dst.index, val);
                }
                MachInst::LoadImm { dst, .. } if dst.index == 16 => break,
                _ => {}
            }
        }

        assert_eq!(regs.get(&0), Some(&"recv"));
        assert_eq!(regs.get(&1), Some(&"method"));
        assert_eq!(regs.get(&2), Some(&"arg"));
    }

    #[test]
    fn test_parallel_copy_no_cycle() {
        // (3→0, 5→1) — no conflicts
        let result = resolve_parallel_copy(&[(3, 0), (5, 1)], 17);
        assert_eq!(result.len(), 2);
        // Both moves should appear, order doesn't matter for correctness
        assert!(result.contains(&(3, 0)));
        assert!(result.contains(&(5, 1)));
    }

    #[test]
    fn test_parallel_copy_swap_cycle() {
        // (0→1, 1→0) — classic swap cycle
        let result = resolve_parallel_copy(&[(0, 1), (1, 0)], 17);
        // Should use scratch: 3 moves total
        assert_eq!(result.len(), 3, "swap should produce 3 moves: {:?}", result);
        assert!(
            result.iter().any(|(_, d)| *d == 17),
            "should use scratch as dst"
        );
        assert!(
            result.iter().any(|(s, _)| *s == 17),
            "should use scratch as src"
        );
    }

    #[test]
    fn test_parallel_copy_identity_filtered() {
        // (0→0, 1→1) — all identity, nothing to do
        let result = resolve_parallel_copy(&[(0, 0), (1, 1)], 17);
        assert!(result.is_empty());
    }

    #[test]
    fn test_boxed_arithmetic_still_works() {
        let a = Value::num(10.0).to_bits();
        let b = Value::num(3.0).to_bits();
        assert_eq!(wren_num_add(a, b), Value::num(13.0).to_bits());
        assert_eq!(wren_num_sub(a, b), Value::num(7.0).to_bits());
        assert_eq!(wren_num_mul(a, b), Value::num(30.0).to_bits());
        assert_eq!(wren_cmp_lt(a, b), Value::bool(false).to_bits());
        assert_eq!(wren_cmp_gt(a, b), Value::bool(true).to_bits());
        assert_eq!(wren_cmp_eq(a, a), Value::bool(true).to_bits());
    }
}
