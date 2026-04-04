use crate::runtime::object::{
    MapKey, Method, ObjClass, ObjClosure, ObjHeader, ObjInstance, ObjList, ObjMap, ObjString,
    ObjType,
};
/// Runtime functions callable from JIT-compiled code.
///
/// These are `extern "C"` functions that the JIT emits `CallRuntime` for.
/// The `resolve` function maps a runtime function name to its address so
/// the codegen can patch `CallRuntime` → `CallInd` with the actual pointer.
///
/// Convention:
/// - All values are passed/returned as NaN-boxed `u64`
/// - VM context is accessed via thread-local `JIT_CONTEXT`
/// - x86_64: System V ABI (RDI, RSI, RDX, RCX, R8, R9 → RAX)
/// - aarch64: AAPCS64 (X0-X7 → X0)
use crate::runtime::value::Value;
use std::sync::OnceLock;

/// Flat shadow root storage. All shadow frames share a single contiguous
/// Vec<Value>. Push/pop is offset arithmetic — zero heap allocation after
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

/// Stack of active JIT frames for GC stack walking.
/// Each entry: (frame_pointer, func_id, return_address).
/// The return_address identifies the active safepoint for precise root scanning.
static mut JIT_FRAME_STACK: [(usize, u32, usize); 64] = [(0, 0, 0); 64];
static mut JIT_FRAME_COUNT: u32 = 0;

/// Push a JIT frame for GC visibility.
#[inline(always)]
pub fn push_jit_frame(fp: usize, func_id: u32, ret_addr: usize) {
    unsafe {
        let idx = JIT_FRAME_COUNT as usize;
        if idx < 64 {
            std::ptr::addr_of_mut!(JIT_FRAME_STACK)
                .cast::<(usize, u32, usize)>()
                .add(idx)
                .write((fp, func_id, ret_addr));
            JIT_FRAME_COUNT += 1;
        }
    }
}

/// Pop a JIT frame.
#[inline(always)]
pub fn pop_jit_frame() {
    unsafe {
        JIT_FRAME_COUNT = JIT_FRAME_COUNT.saturating_sub(1);
    }
}

/// Get all active JIT frames (fp, func_id, ret_addr) for GC stack walking.
pub fn jit_frame_entries() -> &'static [(usize, u32, usize)] {
    unsafe { &JIT_FRAME_STACK[..JIT_FRAME_COUNT as usize] }
}

#[inline(always)]
fn trace_jit_ic(msg: impl FnOnce() -> String) {
    if std::env::var_os("WLIFT_TRACE_JIT_IC").is_some() {
        eprintln!("{}", msg());
    }
}

#[inline(always)]
fn decode_method_and_ic(packed: u64) -> (crate::intern::SymbolId, Option<usize>) {
    let method = crate::intern::SymbolId::from_raw(packed as u32);
    let ic_tag = (packed >> 32) as u32;
    let ic_idx = if ic_tag == 0 {
        None
    } else {
        Some((ic_tag - 1) as usize)
    };
    (method, ic_idx)
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
fn current_jit_callsite_ic(
    vm: &mut crate::runtime::vm::VM,
    ic_idx: usize,
) -> Option<*mut crate::mir::bytecode::CallSiteIC> {
    let ctx = read_jit_ctx();
    let func_id = if ctx.current_func_id != u32::MAX as u64 {
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
    _vm: &crate::runtime::vm::VM,
    fn_ptr: *const u8,
    _func_id: crate::runtime::engine::FuncId,
    args: &[Value],
) -> u64 {
    call_jit_cached(fn_ptr, args)
}

#[inline(always)]
unsafe fn call_jit_cached(fn_ptr: *const u8, args: &[Value]) -> u64 {
    // Ensure x19 holds JitContext pointer for the JIT code.
    #[cfg(target_arch = "aarch64")]
    {
        let ctx_ptr = jit_ctx_ptr() as u64;
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
        _ => {
            let f: extern "C" fn(u64, u64, u64, u64) -> u64 = std::mem::transmute(fn_ptr);
            f(
                args[0].to_bits(),
                args[1].to_bits(),
                args[2].to_bits(),
                args[3].to_bits(),
            )
        }
    }
}

#[inline(always)]
fn trivial_getter_field_index(
    vm: &crate::runtime::vm::VM,
    closure_ptr: *mut ObjClosure,
) -> Option<u16> {
    // Performance note: this function is called on every IC population.
    // It checks the callee's MIR to see if it's a trivial getter.
    use crate::mir::{Instruction, Terminator};

    let func_id = crate::runtime::engine::FuncId(unsafe { (*(*closure_ptr).function).fn_id });
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
                && vm.engine.jit_leaf.get(fn_idx).copied().unwrap_or(false)
            {
                crate::mir::bytecode::CallSiteIC {
                    class: cache_key_class as usize,
                    jit_ptr,
                    closure: closure_ptr as *const u8,
                    func_id: fn_idx as u64,
                    kind: 1,
                }
            } else if !jit_ptr.is_null()
                && !jit_disabled()
                && allow_nonleaf_native(vm, crate::runtime::engine::FuncId(fn_idx as u32))
            {
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
    };

    unsafe {
        *ic_ptr = entry;
    }
    trace_jit_ic(|| {
        format!(
            "jit-ic: populate kind={} ic_ptr=0x{:x}",
            unsafe { (*ic_ptr).kind },
            ic_ptr as usize
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
            vm.engine.request_tier_up(func_id, &vm.interner);
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

    if jit_ptr.is_null() || !vm.engine.jit_leaf.get(fn_idx).copied().unwrap_or(false) {
        return false;
    }

    unsafe {
        *ic_ptr = crate::mir::bytecode::CallSiteIC {
            class: cache_key_class as usize,
            jit_ptr,
            closure: closure_ptr as *const u8,
            func_id: func_id.0 as u64,
            kind: 1,
        };
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
        vm.engine.request_tier_up(func_id, &vm.interner);
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
    let ic = unsafe { &mut *ic_ptr };
    if ic.class != cache_key_class as usize || ic.class == 0 {
        if ic.func_id != 0 {
            vm.engine
                .note_ic_miss(crate::runtime::engine::FuncId(ic.func_id as u32));
        }
        return None;
    }

    match ic.kind {
        1 => {
            if args.len() > 4 {
                *ic = crate::mir::bytecode::CallSiteIC::default();
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
                *ic = crate::mir::bytecode::CallSiteIC::default();
                return None;
            }
            vm.engine.note_ic_hit(func_id);
            trace_jit_ic(|| format!("jit-ic: hit kind=1 func={}", ic.func_id));
            Some(unsafe { call_jit_with_shadow(vm, live_ptr, func_id, args) })
        }
        2 => {
            let closure_ptr = ic.closure as *mut ObjClosure;
            if closure_ptr.is_null() {
                *ic = crate::mir::bytecode::CallSiteIC::default();
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
                let live_ptr = unsafe { (*ic_ptr).jit_ptr };
                vm.engine
                    .note_ic_hit(crate::runtime::engine::FuncId(unsafe {
                        (*ic_ptr).func_id as u32
                    }));
                trace_jit_ic(|| {
                    format!("jit-ic: hit upgraded kind=1 func={}", unsafe {
                        (*ic_ptr).func_id
                    })
                });
                let fid = crate::runtime::engine::FuncId(unsafe { (*ic_ptr).func_id as u32 });
                return Some(unsafe { call_jit_with_shadow(vm, live_ptr, fid, args) });
            }
            vm.engine.note_ic_hit(func_id);
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
                *ic = crate::mir::bytecode::CallSiteIC::default();
                return Some(Value::null().to_bits());
            }
            let func_id = crate::runtime::engine::FuncId(ic.func_id as u32);
            maybe_request_next_tier(vm, func_id);
            if vm.engine.has_pending_compilations() {
                vm.engine.poll_compilations();
            }
            vm.engine.note_ic_hit(func_id);
            trace_jit_ic(|| format!("jit-ic: hit kind=3 func={}", ic.func_id));
            Some(
                vm.call_constructor_sync(class_ptr, closure_ptr, &args[1..])
                    .to_bits(),
            )
        }
        4 => {
            trace_jit_ic(|| "jit-ic: hit kind=4".to_string());
            let native_fn: crate::runtime::object::NativeFn =
                unsafe { std::mem::transmute(ic.closure) };
            Some(native_fn(vm, args).to_bits())
        }
        5 => {
            let instance = recv
                .as_object()
                .map(|p| p as *mut ObjInstance)
                .unwrap_or(std::ptr::null_mut());
            if instance.is_null() {
                *ic = crate::mir::bytecode::CallSiteIC::default();
                return None;
            }
            if !ic.closure.is_null() {
                let func_id = crate::runtime::engine::FuncId(unsafe {
                    (*(*(ic.closure as *mut ObjClosure)).function).fn_id
                });
                vm.engine.note_ic_hit(func_id);
            }
            trace_jit_ic(|| format!("jit-ic: hit kind=5 field={}", ic.func_id));
            let fields_ptr = unsafe { (*instance).fields };
            if fields_ptr.is_null() {
                *ic = crate::mir::bytecode::CallSiteIC::default();
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
                *ic = crate::mir::bytecode::CallSiteIC::default();
                return None;
            }
            vm.engine.note_ic_hit(func_id);
            // Direct field access via UnsafeCell — no 48-byte copy.
            let saved_func_id = JIT_CTX.with(|c| unsafe {
                let ctx = &mut *c.get();
                let old = ctx.current_func_id;
                ctx.current_func_id = fn_idx as u64;
                ctx.closure = ic.closure as *mut u8;
                old
            });
            JIT_DEPTH.with(|d| d.set(depth + 1));
            let result = unsafe { call_jit_cached(live_ptr, args) };
            JIT_DEPTH.with(|d| d.set(depth));
            JIT_CTX.with(|c| unsafe {
                (*c.get()).current_func_id = saved_func_id;
            });
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
    /// Pointer to module variable storage (Vec<Value> data pointer).
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

thread_local! {
    /// JIT context — UnsafeCell for zero-copy direct field access.
    /// Cell<JitContext> copies 48 bytes on every get/set. UnsafeCell gives
    /// direct pointer access with no copying — critical for the hot dispatch path.
    static JIT_CTX: std::cell::UnsafeCell<JitContext> = const { std::cell::UnsafeCell::new(JitContext {
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
    }) };

    /// JIT root set — UnsafeCell for zero-overhead Vec mutation.
    /// SAFETY: single-threaded access within each thread (no concurrent borrows).
    static JIT_ROOTS_STORE: std::cell::UnsafeCell<Vec<Value>> = const { std::cell::UnsafeCell::new(Vec::new()) };

    /// Flat shadow root stack — zero-alloc push/pop after warmup.
    static FLAT_SHADOW: std::cell::UnsafeCell<FlatShadowStack> =
        const { std::cell::UnsafeCell::new(FlatShadowStack {
            roots: Vec::new(),
            boundaries: Vec::new(),
        }) };


    /// JIT native recursion depth — guards against native stack overflow.
    /// When depth exceeds MAX_JIT_DEPTH, calls fall back to interpreter dispatch.
    static JIT_DEPTH: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };

    /// When true, all JIT dispatch is disabled (fall back to interpreter).
    /// Used by shadow check to run interpreter-only path for comparison.
    static JIT_DISABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

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

/// Read the current JIT native recursion depth.
#[inline(always)]
pub fn jit_depth() -> u32 {
    JIT_DEPTH.with(|d| d.get())
}

/// Set the JIT native recursion depth.
#[inline(always)]
pub fn set_jit_depth(depth: u32) {
    JIT_DEPTH.with(|d| d.set(depth));
}

/// Check if JIT dispatch is disabled (for shadow check mode).
#[inline(always)]
pub fn jit_disabled() -> bool {
    JIT_DISABLED.with(|d| d.get())
}

#[inline(always)]
pub fn shadow_nonleaf_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("WLIFT_SHADOW_NONLEAF").is_some())
}

/// Set up the JIT context before calling compiled code.
#[inline(always)]
pub fn set_jit_context(ctx: JitContext) {
    JIT_CTX.with(|c| unsafe { *c.get() = ctx });
}

/// Read the current JIT context (zero-copy reference via UnsafeCell).
#[inline(always)]
pub fn read_jit_ctx() -> JitContext {
    JIT_CTX.with(|c| unsafe { *c.get() })
}

/// Mutate the JIT context in place (no copy — direct field access).
#[inline(always)]
pub fn mutate_jit_ctx(f: impl FnOnce(&mut JitContext)) {
    JIT_CTX.with(|c| unsafe { f(&mut *c.get()) });
}

// ---------------------------------------------------------------------------
// JIT root set — GC-visible heap pointers held by native frames
// ---------------------------------------------------------------------------

/// Push a value into the JIT root set so GC can see it.
#[inline(always)]
pub fn push_jit_root(v: Value) {
    JIT_ROOTS_STORE.with(|r| unsafe { (*r.get()).push(v) });
}

/// Take all JIT roots for GC scanning. Returns the values (caller must write back
/// after collection via `set_jit_roots` for nursery forwarding).
pub fn take_jit_roots() -> Vec<Value> {
    JIT_ROOTS_STORE.with(|r| unsafe { std::mem::take(&mut *r.get()) })
}

/// Write back JIT roots after GC (with nursery-forwarded pointers).
pub fn set_jit_roots(roots: Vec<Value>) {
    JIT_ROOTS_STORE.with(|r| unsafe { *r.get() = roots });
}

/// Pop and return the last JIT root, if any.
#[inline(always)]
pub fn pop_jit_root() -> Option<Value> {
    JIT_ROOTS_STORE.with(|r| unsafe { (*r.get()).pop() })
}

/// Clear JIT roots (called after native code returns to interpreter).
#[inline(always)]
pub fn clear_jit_roots() {
    JIT_ROOTS_STORE.with(|r| unsafe {
        let v = &mut *r.get();
        v.clear();
        // Prevent capacity from growing unboundedly after root spikes.
        if v.capacity() > 256 {
            v.shrink_to(256);
        }
    });
}

/// Get current JIT roots count (for save/restore around re-entrant calls).
#[inline(always)]
#[allow(dead_code)]
fn jit_roots_len() -> usize {
    JIT_ROOTS_STORE.with(|r| unsafe { (*r.get()).len() })
}

/// Truncate JIT roots back to a saved length (pop roots added by this frame).
#[inline(always)]
#[allow(dead_code)]
fn jit_roots_truncate(len: usize) {
    JIT_ROOTS_STORE.with(|r| unsafe { (*r.get()).truncate(len) });
}

/// Get the current JIT roots length for save/restore by external callers.
#[inline(always)]
pub fn jit_roots_snapshot_len() -> usize {
    JIT_ROOTS_STORE.with(|r| unsafe { (*r.get()).len() })
}

/// Read the JIT root at the given index (for reading GC-forwarded pointers).
#[inline(always)]
pub fn jit_root_at(idx: usize) -> crate::runtime::value::Value {
    JIT_ROOTS_STORE.with(|r| unsafe { (&(*r.get()))[idx] })
}

/// Truncate JIT roots to the given length (public version for external callers).
#[inline(always)]
pub fn jit_roots_restore_len(len: usize) {
    JIT_ROOTS_STORE.with(|r| unsafe { (*r.get()).truncate(len) });
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

/// Write back JitContext's GC-managed pointers after collection (nursery forwarding).
pub fn update_jit_context_roots(closure: Value, defining_class: Value) {
    mutate_jit_ctx(|ctx| {
        ctx.closure = closure.as_object().unwrap_or(std::ptr::null_mut());
        ctx.defining_class = defining_class.as_object().unwrap_or(std::ptr::null_mut());
    });
}

#[inline(always)]
fn root_saved_jit_context(ctx: JitContext) -> usize {
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

#[inline(always)]
fn restore_rooted_jit_context(mut saved_ctx: JitContext, root_len_before: usize) {
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

pub fn take_native_shadow_roots() -> (Vec<usize>, Vec<Value>) {
    FLAT_SHADOW.with(|s| unsafe {
        let stack = &mut *s.get();
        set_current_native_shadow_roots_ptr(std::ptr::null_mut());
        if stack.boundaries.is_empty() {
            return (Vec::new(), Vec::new());
        }
        // Build per-frame lengths from boundary offsets.
        let mut lengths = Vec::with_capacity(stack.boundaries.len());
        for i in 0..stack.boundaries.len() {
            let start = stack.boundaries[i] as usize;
            let end = if i + 1 < stack.boundaries.len() {
                stack.boundaries[i + 1] as usize
            } else {
                stack.roots.len()
            };
            lengths.push(end - start);
        }
        let roots = std::mem::take(&mut stack.roots);
        stack.boundaries.clear();
        (lengths, roots)
    })
}

pub fn set_native_shadow_roots(lengths: Vec<usize>, roots: Vec<Value>) {
    FLAT_SHADOW.with(|s| unsafe {
        let stack = &mut *s.get();
        stack.roots = roots;
        stack.boundaries.clear();
        let mut offset = 0u16;
        for len in &lengths {
            stack.boundaries.push(offset);
            offset += *len as u16;
        }
        sync_flat_shadow_ptr(stack);
    });
}

#[allow(dead_code)]
#[inline(always)]
fn current_shadow_slot_count(
    vm: &crate::runtime::vm::VM,
    func_id: crate::runtime::engine::FuncId,
) -> usize {
    vm.engine
        .jit_metadata
        .get(func_id.0 as usize)
        .and_then(|meta| meta.as_ref())
        // Only return nonzero when the function actually has shadow stores
        // (safepoints). Functions with boxed values but no safepoints (e.g.,
        // pure arithmetic) don't emit shadow stores and don't need a frame.
        .filter(|meta| !meta.safepoints.is_empty())
        .map(|meta| meta.boxed_values.len())
        .unwrap_or(0)
}

#[inline(always)]
fn nonleaf_shadow_safe(
    vm: &crate::runtime::vm::VM,
    func_id: crate::runtime::engine::FuncId,
) -> bool {
    vm.engine
        .jit_metadata
        .get(func_id.0 as usize)
        .and_then(|meta| meta.as_ref())
        .map(|meta| meta.spill_safe_nonleaf)
        .unwrap_or(false)
}

#[inline(always)]
fn nonleaf_moving_gc_safe(
    _vm: &crate::runtime::vm::VM,
    _func_id: crate::runtime::engine::FuncId,
) -> bool {
    // All non-leaf JIT functions are safe for nested execution:
    // - JIT frame registration (#[naked] wren_call_N) makes frames visible to GC
    // - force_boxed_gp_spills ensures all boxed values are in known spill slots
    // - GC stack walker (scan_native_stack_roots) reads/writes spill slots directly
    // - x20 holds JitContext pointer, eliminating TLS for context access
    true
    /* Old conservative check (no longer needed with stack map infrastructure):
    vm.engine
        .jit_metadata
        .get(func_id.0 as usize)
        .and_then(|meta| meta.as_ref())
        .map(|meta| {
            meta.spill_safe_nonleaf
                && meta
                    .safepoints
                    .iter()
                    .map(|sp| sp.live_roots.len())
                    .max()
                    .unwrap_or(0)
                    <= 5
                && meta.safepoints.iter().all(|sp| {
                    sp.live_roots.iter().all(|root| {
                        matches!(
                            root.location,
                            crate::codegen::native_meta::RootLocation::Spill(_)
                        )
                    })
                })
        })
        .unwrap_or(false)
    */
}

fn trace_nonleaf_gate(
    vm: &crate::runtime::vm::VM,
    func_id: crate::runtime::engine::FuncId,
    allow_nonleaf_native: bool,
) {
    if std::env::var_os("WLIFT_TRACE_NONLEAF").is_none() {
        return;
    }

    let (boxed_values, safepoints, max_live_roots, spill_safe_nonleaf) = vm
        .engine
        .jit_metadata
        .get(func_id.0 as usize)
        .and_then(|meta| meta.as_ref())
        .map(|meta| {
            (
                meta.boxed_values.len(),
                meta.safepoints.len(),
                meta.safepoints
                    .iter()
                    .map(|sp| sp.live_roots.len())
                    .max()
                    .unwrap_or(0),
                meta.spill_safe_nonleaf,
            )
        })
        .unwrap_or((0, 0, 0, false));
    let name = vm
        .engine
        .get_mir(func_id)
        .map(|mir| vm.interner.resolve(mir.name).to_string())
        .unwrap_or_else(|| "<unknown>".to_string());
    eprintln!(
        "nonleaf gate: FuncId({}) {} allow={} spill_safe={} boxed={} safepoints={} max_live_roots={}",
        func_id.0,
        name,
        allow_nonleaf_native,
        spill_safe_nonleaf,
        boxed_values,
        safepoints,
        max_live_roots
    );
}

pub fn allow_nonleaf_native(
    vm: &crate::runtime::vm::VM,
    func_id: crate::runtime::engine::FuncId,
) -> bool {
    // With Cranelift, all non-leaf JIT functions are safe to call —
    // Cranelift handles register allocation and stack frames correctly.
    // The old backend needed shadow-stack/spill-slot safety checks.
    #[cfg(feature = "cranelift")]
    {
        let _ = (vm, func_id);
        return true;
    }
    #[cfg(not(feature = "cranelift"))]
    {
        let allow_nonleaf_native = match vm.config.gc_strategy {
            crate::runtime::gc_trait::GcStrategy::Generational => {
                nonleaf_moving_gc_safe(vm, func_id)
            }
            _ => nonleaf_shadow_safe(vm, func_id),
        };
        trace_nonleaf_gate(vm, func_id, allow_nonleaf_native);
        allow_nonleaf_native
    }
}

pub fn allow_root_nonleaf_native(
    vm: &crate::runtime::vm::VM,
    func_id: crate::runtime::engine::FuncId,
) -> bool {
    let allow_root_nonleaf_native = match vm.config.gc_strategy {
        crate::runtime::gc_trait::GcStrategy::Generational => nonleaf_moving_gc_safe(vm, func_id),
        _ => nonleaf_shadow_safe(vm, func_id),
    };
    trace_nonleaf_gate(vm, func_id, allow_root_nonleaf_native);
    allow_root_nonleaf_native
}

#[inline(always)]
fn trace_native_entry(
    vm: &crate::runtime::vm::VM,
    func_id: crate::runtime::engine::FuncId,
    kind: &str,
) {
    if std::env::var_os("WLIFT_TRACE_NATIVE_ENTRY").is_none() {
        return;
    }
    let name = vm
        .engine
        .get_mir(func_id)
        .map(|mir| vm.interner.resolve(mir.name).to_string())
        .unwrap_or_else(|| "<unknown>".to_string());
    eprintln!("native-entry: {kind} FuncId({}) {}", func_id.0, name);
}

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
pub extern "C" fn wren_enter_shadow_frame(slot_count: u64) {
    let count = slot_count as usize;
    if count > 0 {
        push_native_shadow_frame(count);
    }
}

/// Callee-managed shadow frame: pop in JIT epilogue.
pub extern "C" fn wren_exit_shadow_frame() {
    pop_native_shadow_frame();
}

/// Get the VM pointer from the JIT context.
/// # Safety
/// Caller must ensure the VM pointer is valid.
#[inline(always)]
unsafe fn vm_ref() -> Option<&'static mut crate::runtime::vm::VM> {
    let ctx = read_jit_ctx();
    if ctx.vm.is_null() {
        None
    } else {
        Some(&mut *(ctx.vm as *mut crate::runtime::vm::VM))
    }
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

/// Load the JIT code pointer for a given function ID.
/// Returns the function pointer as u64 (0 if not compiled).
/// Used by CallKnownFunc to do direct JIT-to-JIT calls.
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

/// Record an old->young edge for a just-performed object write.
pub extern "C" fn wren_write_barrier(source: u64, value: u64) -> u64 {
    let source = Value::from_bits(source);
    let value = Value::from_bits(value);
    if let Some(ptr) = source.as_object() {
        unsafe {
            if let Some(vm) = vm_ref() {
                vm.gc.write_barrier(ptr as *mut ObjHeader, value);
            }
        }
    }
    value.to_bits()
}

/// Try to call a closure via native code if it's compiled; fall back to interpreter.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn call_closure_jit_or_sync(
    vm: &mut crate::runtime::vm::VM,
    closure_ptr: *mut ObjClosure,
    args: &[Value],
    defining_class: Option<*mut crate::runtime::object::ObjClass>,
) -> u64 {
    // Save the caller's JIT context so we can restore it after the nested call.
    // Without this, interpreter fallback paths (call_closure_sync → run_fiber)
    // can overwrite the context via dispatch_closure_bc, corrupting the caller's
    // module_vars / closure / defining_class when control returns.
    let saved_ctx = read_jit_ctx();
    let saved_ctx_root_len = root_saved_jit_context(saved_ctx);

    // Update context for the callee: set closure, defining_class, and
    // ensure vm is set (may be null if we're called from a fresh thread
    // or after a context restore).
    mutate_jit_ctx(|ctx| {
        if ctx.vm.is_null() {
            ctx.vm = vm as *mut _ as *mut u8;
        }
        ctx.current_func_id = unsafe { (*(*closure_ptr).function).fn_id } as u64;
        ctx.closure = closure_ptr as *mut u8;
        ctx.defining_class = defining_class
            .map(|p| p as *mut u8)
            .unwrap_or(std::ptr::null_mut());
    });

    // Install completed compilations so we can dispatch natively.
    if vm.engine.has_pending_compilations() {
        vm.engine.poll_compilations();
    }
    let mut had_native_candidate = false;
    if args.len() <= 4 {
        let func_id = crate::runtime::engine::FuncId(unsafe { (*(*closure_ptr).function).fn_id });

        let native_fn_ptr: Option<*const u8> = if JIT_DISABLED.with(|d| d.get()) {
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
            // Guard: check native recursion depth to prevent stack overflow.
            // Each JIT call level uses ~1-2KB native stack; limit to 256 levels.
            let depth = JIT_DEPTH.with(|d| d.get());
            let is_leaf = vm
                .engine
                .jit_leaf
                .get(func_id.0 as usize)
                .copied()
                .unwrap_or(false);
            let allow_nonleaf_native = allow_nonleaf_native(vm, func_id);
            // Nested non-leaf native calls are not safe yet: if the callee
            // allocates or re-enters the interpreter, the caller's native frame
            // still has live boxed values with no stack map / root metadata.
            // That is fatal for moving GC and can still lead to collection of
            // live objects under non-moving GC. Keep top-level non-leaf native
            // execution available, but route nested non-leaf calls through the
            // interpreter until native frame rooting exists.
            if !is_leaf && !allow_nonleaf_native {
                vm.engine.note_fallback_to_interpreter(func_id);
                let result = vm
                    .call_closure_sync(closure_ptr, args, defining_class)
                    .map(|v| v.to_bits())
                    .unwrap_or(Value::null().to_bits());
                restore_rooted_jit_context(saved_ctx, saved_ctx_root_len);
                return result;
            }

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
                JIT_DEPTH.with(|d| d.set(depth + 1));
                let root_len_before = jit_roots_snapshot_len();
                // Shadow frame push/pop handled by call_jit_with_shadow.
                let result = unsafe { call_jit_with_shadow(vm, fn_ptr, func_id, args) };
                JIT_DEPTH.with(|d| d.set(depth));

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
    let idx = method.index() as usize;
    let mut c = cls;
    while !c.is_null() {
        let cls_ref = &*c;
        if idx < cls_ref.methods.len() {
            if let Some(m) = &cls_ref.methods[idx] {
                return Some((*m, c));
            }
        }
        c = (*c).superclass;
    }
    None
}

/// Internal: dispatch a method call with a pre-built args slice.
fn dispatch_call(recv: Value, method_packed: u64, args: &[Value]) -> u64 {
    dispatch_call_rooted(recv, method_packed, args)
}

fn dispatch_call_rooted(recv: Value, method_packed: u64, args: &[Value]) -> u64 {
    let vm = unsafe { vm_ref() };
    let vm = match vm {
        Some(v) => v,
        None => return Value::null().to_bits(),
    };
    let (method_sym, ic_idx) = decode_method_and_ic(method_packed);

    // Match the interpreter's closure call fast path. Fn.call(...) is a stub
    // on the Fn class; actual closure invocation must pass only the user args.
    if vm.is_call_sym(method_sym) && recv.is_object() {
        if let Some(ptr) = recv.as_object() {
            let header = ptr as *const ObjHeader;
            if unsafe { (*header).obj_type } == ObjType::Closure {
                let closure_ptr = ptr as *mut ObjClosure;
                return call_closure_jit_or_sync(vm, closure_ptr, &args[1..], None);
            }
        }
    }

    let class = vm.class_of(recv);
    let cache_key_class = cache_key_class(vm, recv, class);
    let ic_ptr = ic_idx.and_then(|idx| current_jit_callsite_ic(vm, idx));

    if let Some(ic_ptr) = ic_ptr {
        if let Some(result) = try_dispatch_callsite_ic(vm, ic_ptr, recv, args, cache_key_class) {
            return result;
        }
    }

    // Check method cache first (avoids find_method + static symbol resolution).
    if let Some((m, dc)) = vm.method_cache.lookup(cache_key_class, method_sym) {
        if let Some(ic_ptr) = ic_ptr {
            populate_callsite_ic(vm, ic_ptr, cache_key_class, m, dc);
        }
        return dispatch_method(vm, m, args, Some(dc));
    }

    // Cache miss: full hierarchy lookup to get the correct defining class.
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
                    Value::null().to_bits()
                }
            } else {
                Value::null().to_bits()
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

    match action {
        FiberAction::Call { target, value } | FiberAction::Transfer { target, value } => {
            let caller = vm.fiber;
            unsafe {
                if !caller.is_null() {
                    (*caller).state = FiberState::Suspended;
                }
                if is_call {
                    (*target).caller = caller;
                }
            }
            let target_state = unsafe { (*target).state };
            if target_state == FiberState::Suspended {
                // Resuming a suspended fiber: deliver the value.
                unsafe {
                    if let Some(dst) = (*target).resume_value_dst.take() {
                        if let Some(frame) = (*target).mir_frames.last_mut() {
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
            vm.fiber = target;
            let result = crate::runtime::vm_interp::run_fiber(vm);
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
                if let Some(v) = yield_val {
                    return v.to_bits();
                }
            }
            match result {
                Ok(v) => v.to_bits(),
                Err(e) => {
                    // Propagate the runtime error from the child fiber.
                    vm.has_error = true;
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

/// Dispatch a resolved method entry.
#[inline(always)]
fn dispatch_method(
    vm: &mut crate::runtime::vm::VM,
    method: Method,
    args: &[Value],
    defining_class: Option<*mut crate::runtime::object::ObjClass>,
) -> u64 {
    match method {
        Method::Native(native_fn) => {
            let result = native_fn(vm, args).to_bits();
            // Check if the native method requested a fiber action (e.g. fiber.call()).
            // If so, handle it synchronously here — the JIT caller expects a return
            // value and has no mechanism to perform a fiber switch itself.
            if let Some(action) = vm.pending_fiber_action.take() {
                return handle_jit_fiber_action(vm, action);
            }
            result
        }
        Method::Closure(cp) => {
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
                        vm.engine.request_tier_up(func_id, &vm.interner);
                    }
                }
            }

            if args.len() <= 4 && !jit_disabled() {
                let fn_ptr = vm
                    .engine
                    .jit_code
                    .get(fn_idx)
                    .copied()
                    .unwrap_or(std::ptr::null());
                let is_leaf = vm.engine.jit_leaf.get(fn_idx).copied().unwrap_or(false);
                // With Cranelift, allow non-leaf JIT dispatch — Cranelift
                // handles register allocation and call conventions correctly.
                // The old backend required is_leaf to avoid spill-slot bugs.
                #[cfg(feature = "cranelift")]
                let allow_jit = !fn_ptr.is_null();
                #[cfg(not(feature = "cranelift"))]
                let allow_jit = !fn_ptr.is_null() && is_leaf;
                if allow_jit {
                    let saved_ctx = read_jit_ctx();
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

/// Call a method with 0 extra args. Codegen: [receiver, method_sym]
/// On aarch64, wren_call_N uses #[naked] wrappers to capture the caller's
/// frame pointer (x29) at zero cost to JIT code. The FP is passed as the
/// LAST argument to the inner function, which pushes it to JIT_FRAME_STACK
/// for GC stack walking.
///
/// # Safety
/// Called only from JIT-compiled code via `CallRuntime`. The receiver and
/// method arguments are NaN-boxed values produced by the JIT.
#[cfg(target_arch = "aarch64")]
#[unsafe(naked)]
pub unsafe extern "C" fn wren_call_0(_receiver: u64, _method: u64) -> u64 {
    core::arch::naked_asm!(
        "mov x2, x29",       // pass JIT FP as 3rd arg
        "mov x3, x30",       // pass return address as 4th arg
        "b {inner}",
        inner = sym wren_call_0_inner,
    );
}
#[cfg(target_arch = "x86_64")]
#[unsafe(naked)]
pub unsafe extern "C" fn wren_call_0(_receiver: u64, _method: u64) -> u64 {
    // SysV: rdi=receiver, rsi=method → inner gets rdx=jit_fp, rcx=ret_addr
    core::arch::naked_asm!(
        "mov rdx, rbp",       // pass JIT FP (caller's RBP) as 3rd arg
        "mov rcx, [rsp]",     // pass return address as 4th arg
        "jmp {inner}",
        inner = sym wren_call_0_inner,
    );
}
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
pub extern "C" fn wren_call_0(receiver: u64, method: u64) -> u64 {
    wren_call_0_inner(receiver, method, 0, 0)
}
extern "C" fn wren_call_0_inner(receiver: u64, method: u64, jit_fp: u64, ret_addr: u64) -> u64 {
    push_jit_frame(
        jit_fp as usize,
        read_jit_ctx().current_func_id as u32,
        ret_addr as usize,
    );
    let recv = Value::from_bits(receiver);
    let result = dispatch_call(recv, method, &[recv]);
    pop_jit_frame();
    result
}

/// # Safety
/// Called only from JIT-compiled code via `CallRuntime`.
#[cfg(target_arch = "aarch64")]
#[unsafe(naked)]
pub unsafe extern "C" fn wren_call_1(_receiver: u64, _method: u64, _a0: u64) -> u64 {
    core::arch::naked_asm!(
        "mov x3, x29",
        "mov x4, x30",
        "b {inner}",
        inner = sym wren_call_1_inner,
    );
}
#[cfg(target_arch = "x86_64")]
#[unsafe(naked)]
pub unsafe extern "C" fn wren_call_1(_receiver: u64, _method: u64, _a0: u64) -> u64 {
    // SysV: rdi=receiver, rsi=method, rdx=a0 → inner gets rcx=jit_fp, r8=ret_addr
    core::arch::naked_asm!(
        "mov rcx, rbp",
        "mov r8, [rsp]",
        "jmp {inner}",
        inner = sym wren_call_1_inner,
    );
}
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
pub extern "C" fn wren_call_1(receiver: u64, method: u64, a0: u64) -> u64 {
    wren_call_1_inner(receiver, method, a0, 0, 0)
}
extern "C" fn wren_call_1_inner(
    receiver: u64,
    method: u64,
    a0: u64,
    jit_fp: u64,
    ret_addr: u64,
) -> u64 {
    push_jit_frame(
        jit_fp as usize,
        read_jit_ctx().current_func_id as u32,
        ret_addr as usize,
    );
    let recv = Value::from_bits(receiver);
    let result = dispatch_call(recv, method, &[recv, Value::from_bits(a0)]);
    pop_jit_frame();
    result
}

/// # Safety
/// Called only from JIT-compiled code via `CallRuntime`.
#[cfg(target_arch = "aarch64")]
#[unsafe(naked)]
pub unsafe extern "C" fn wren_call_2(_receiver: u64, _method: u64, _a0: u64, _a1: u64) -> u64 {
    core::arch::naked_asm!(
        "mov x4, x29",
        "mov x5, x30",
        "b {inner}",
        inner = sym wren_call_2_inner,
    );
}
#[cfg(target_arch = "x86_64")]
#[unsafe(naked)]
pub unsafe extern "C" fn wren_call_2(_receiver: u64, _method: u64, _a0: u64, _a1: u64) -> u64 {
    // SysV: rdi=receiver, rsi=method, rdx=a0, rcx=a1 → inner gets r8=jit_fp, r9=ret_addr
    core::arch::naked_asm!(
        "mov r8, rbp",
        "mov r9, [rsp]",
        "jmp {inner}",
        inner = sym wren_call_2_inner,
    );
}
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
pub extern "C" fn wren_call_2(receiver: u64, method: u64, a0: u64, a1: u64) -> u64 {
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
    push_jit_frame(
        jit_fp as usize,
        read_jit_ctx().current_func_id as u32,
        ret_addr as usize,
    );
    let recv = Value::from_bits(receiver);
    let result = dispatch_call(
        recv,
        method,
        &[recv, Value::from_bits(a0), Value::from_bits(a1)],
    );
    pop_jit_frame();
    result
}

/// # Safety
/// Called only from JIT-compiled code via `CallRuntime`.
#[cfg(target_arch = "aarch64")]
#[unsafe(naked)]
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
#[cfg(target_arch = "x86_64")]
pub extern "C" fn wren_call_3(receiver: u64, method: u64, a0: u64, a1: u64, a2: u64) -> u64 {
    // Can't use #[naked] for 7 args (exceeds 6 SysV register args).
    // Read the caller's frame pointer from the stack frame chain instead.
    // After this function's prologue: [rbp] = saved caller RBP, [rbp+8] = ret addr.
    // The JIT function's FP is the saved caller RBP: [rbp] points to the JIT frame.
    let jit_fp: u64;
    let ret_addr: u64;
    unsafe {
        core::arch::asm!(
            "mov {fp}, [rbp]",      // caller's RBP = JIT frame pointer
            "mov {ra}, [rbp+8]",    // this function's return address
            fp = out(reg) jit_fp,
            ra = out(reg) ret_addr,
        );
    }
    wren_call_3_inner(receiver, method, a0, a1, a2, jit_fp, ret_addr)
}
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
pub extern "C" fn wren_call_3(receiver: u64, method: u64, a0: u64, a1: u64, a2: u64) -> u64 {
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
    push_jit_frame(
        jit_fp as usize,
        read_jit_ctx().current_func_id as u32,
        ret_addr as usize,
    );
    let recv = Value::from_bits(receiver);
    let result = dispatch_call(
        recv,
        method,
        &[
            recv,
            Value::from_bits(a0),
            Value::from_bits(a1),
            Value::from_bits(a2),
        ],
    );
    pop_jit_frame();
    result
}

/// # Safety
/// Called only from JIT-compiled code via `CallRuntime`.
#[cfg(target_arch = "aarch64")]
#[unsafe(naked)]
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
#[cfg(not(target_arch = "aarch64"))]
pub extern "C" fn wren_call_4(
    receiver: u64,
    method: u64,
    a0: u64,
    a1: u64,
    a2: u64,
    a3: u64,
) -> u64 {
    wren_call_4_inner(receiver, method, a0, a1, a2, a3, 0, 0)
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
    push_jit_frame(
        jit_fp as usize,
        read_jit_ctx().current_func_id as u32,
        ret_addr as usize,
    );
    let recv = Value::from_bits(receiver);
    let result = dispatch_call(
        recv,
        method,
        &[
            recv,
            Value::from_bits(a0),
            Value::from_bits(a1),
            Value::from_bits(a2),
            Value::from_bits(a3),
        ],
    );
    pop_jit_frame();
    result
}

// ---------------------------------------------------------------------------
// Indirect IC dispatch: lightweight JIT-to-JIT calls via IC entry.
// These are ~10x faster than wren_call_N because they skip method lookup.
// Called from Cranelift JIT code when the indirect IC class check passes.
// The IC entry provides the jit_ptr; these functions set up minimal context
// (func_id + closure) and do a direct call.
// ---------------------------------------------------------------------------

/// IC call with 0 extra args. Signature: (ic_ptr, recv) -> result
pub extern "C" fn wren_ic_call_0(ic_ptr: u64, recv: u64) -> u64 {
    wren_ic_call_inner(ic_ptr, &[recv])
}

/// IC call with 1 extra arg. Signature: (ic_ptr, recv, a0) -> result
pub extern "C" fn wren_ic_call_1(ic_ptr: u64, recv: u64, a0: u64) -> u64 {
    wren_ic_call_inner(ic_ptr, &[recv, a0])
}

/// IC call with 2 extra args. Signature: (ic_ptr, recv, a0, a1) -> result
pub extern "C" fn wren_ic_call_2(ic_ptr: u64, recv: u64, a0: u64, a1: u64) -> u64 {
    wren_ic_call_inner(ic_ptr, &[recv, a0, a1])
}

/// IC call with 3 extra args.
pub extern "C" fn wren_ic_call_3(ic_ptr: u64, recv: u64, a0: u64, a1: u64, a2: u64) -> u64 {
    wren_ic_call_inner(ic_ptr, &[recv, a0, a1, a2])
}

fn wren_ic_call_inner(ic_ptr_raw: u64, args: &[u64]) -> u64 {
    let ic = unsafe { &*(ic_ptr_raw as *const crate::mir::bytecode::CallSiteIC) };
    let jit_ptr = ic.jit_ptr;
    if jit_ptr.is_null() {
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
    let args_val: smallvec::SmallVec<[Value; 5]> =
        args.iter().map(|&a| Value::from_bits(a)).collect();
    let result = unsafe { call_jit_with_shadow_raw(jit_ptr, &args_val) };
    set_jit_depth(depth);
    set_jit_context(saved_ctx);
    result
}

/// Raw call into JIT code without VM reference (for IC dispatch).
#[inline(always)]
unsafe fn call_jit_with_shadow_raw(fn_ptr: *const u8, args: &[Value]) -> u64 {
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
        _ => {
            let f: extern "C" fn(u64, u64, u64, u64) -> u64 = std::mem::transmute(fn_ptr);
            f(
                args[0].to_bits(),
                args[1].to_bits(),
                args[2].to_bits(),
                args[3].to_bits(),
            )
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

pub extern "C" fn wren_call_static_self_0() -> u64 {
    call_static_self_inner(&[])
}

pub extern "C" fn wren_call_static_self_1(a0: u64) -> u64 {
    call_static_self_inner(&[a0])
}

pub extern "C" fn wren_call_static_self_2(a0: u64, a1: u64) -> u64 {
    call_static_self_inner(&[a0, a1])
}

pub extern "C" fn wren_call_static_self_3(a0: u64, a1: u64, a2: u64) -> u64 {
    call_static_self_inner(&[a0, a1, a2])
}

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
        Some((Method::Closure(closure_ptr), dc)) => {
            call_closure_jit_or_sync(vm, closure_ptr, args, Some(dc))
        }
        Some((Method::Constructor(closure_ptr), dc)) => {
            // Super constructor calls are correct: args[0] is already 'this' (the
            // newly allocated instance), not the class. Use call_closure_sync to
            // avoid GC-unsafe JIT dispatch with potentially stale 'this' pointer.
            // args[0] = this (instance), args[1..] = user constructor args.
            if args.is_empty() {
                return Value::null().to_bits();
            }
            vm.call_closure_sync(closure_ptr, args, Some(dc))
                .map(|v| v.to_bits())
                .unwrap_or(args[0].to_bits())
        }
        None => Value::null().to_bits(),
    }
}

/// Super call with 0 args. Codegen: [method_sym] (no receiver — shouldn't happen in practice)
pub extern "C" fn wren_super_call_0(method: u64) -> u64 {
    let _ = method;
    Value::null().to_bits()
}
/// Super call with 1 arg. Codegen: [method_sym, this]
pub extern "C" fn wren_super_call_1(method: u64, this: u64) -> u64 {
    let recv = Value::from_bits(this);
    let sym = crate::intern::SymbolId::from_raw(method as u32);
    dispatch_super_call(recv, sym, &[recv])
}
/// Super call with 2 args. Codegen: [method_sym, this, a0]
pub extern "C" fn wren_super_call_2(method: u64, this: u64, a0: u64) -> u64 {
    let recv = Value::from_bits(this);
    let sym = crate::intern::SymbolId::from_raw(method as u32);
    dispatch_super_call(recv, sym, &[recv, Value::from_bits(a0)])
}
/// Super call with 3 args. Codegen: [method_sym, this, a0, a1]
pub extern "C" fn wren_super_call_3(method: u64, this: u64, a0: u64, a1: u64) -> u64 {
    let recv = Value::from_bits(this);
    let sym = crate::intern::SymbolId::from_raw(method as u32);
    dispatch_super_call(
        recv,
        sym,
        &[recv, Value::from_bits(a0), Value::from_bits(a1)],
    )
}
/// Super call with 4 args. Codegen: [method_sym, this, a0, a1, a2]
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
    push_jit_root(val);
    val.to_bits()
}

/// Add a single element to an existing list.
pub extern "C" fn wren_list_add(list_val: u64, elem: u64) {
    let list = Value::from_bits(list_val);
    if let Some(ptr) = list.as_object() {
        let list_ptr = ptr as *mut crate::runtime::object::ObjList;
        unsafe {
            (*list_ptr).add(Value::from_bits(elem));
        }
    }
}

pub extern "C" fn wren_make_list() -> u64 {
    make_list_impl(&[])
}

pub extern "C" fn wren_make_list_1(a0: u64) -> u64 {
    make_list_impl(&[a0])
}

pub extern "C" fn wren_make_list_2(a0: u64, a1: u64) -> u64 {
    make_list_impl(&[a0, a1])
}

pub extern "C" fn wren_make_list_3(a0: u64, a1: u64, a2: u64) -> u64 {
    make_list_impl(&[a0, a1, a2])
}

pub extern "C" fn wren_make_list_4(a0: u64, a1: u64, a2: u64, a3: u64) -> u64 {
    make_list_impl(&[a0, a1, a2, a3])
}

/// Allocate a new empty map.
/// Set a key-value pair on a map object.
pub extern "C" fn wren_map_set(map_val: u64, key: u64, value: u64) {
    let map = Value::from_bits(map_val);
    if let Some(ptr) = map.as_object() {
        let map_ptr = ptr as *mut ObjMap;
        unsafe {
            (*map_ptr).set(Value::from_bits(key), Value::from_bits(value));
        }
    }
}

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
    push_jit_root(val);
    val.to_bits()
}

/// Allocate a new range.
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
    push_jit_root(val);
    val.to_bits()
}

/// Helper: allocate closure and populate upvalues from a slice of NaN-boxed values.
fn make_closure_inner(fn_id: u64, upvalue_vals: &[u64]) -> u64 {
    let vm = unsafe { vm_ref() };
    let vm = match vm {
        Some(v) => v,
        None => return Value::null().to_bits(),
    };

    let func_id = fn_id as u32;
    let name_sym = vm.interner.intern("<closure>");
    let uv_count = upvalue_vals.len() as u16;
    let arity = vm
        .engine
        .get_mir(crate::runtime::engine::FuncId(func_id))
        .map(|mir| mir.arity)
        .unwrap_or(0);
    let fn_ptr = vm.gc.alloc_fn(name_sym, arity, uv_count, func_id);
    unsafe {
        (*fn_ptr).header.class = vm.fn_class;
    }
    // Root the fn object before further allocations.
    push_jit_root(Value::object(fn_ptr as *mut u8));

    let closure_ptr = vm.gc.alloc_closure(fn_ptr);
    unsafe {
        (*closure_ptr).header.class = vm.fn_class;
    }
    // Root the closure before upvalue allocations.
    push_jit_root(Value::object(closure_ptr as *mut u8));

    // Populate upvalues with captured values (pre-closed)
    for (i, &uv_bits) in upvalue_vals.iter().enumerate() {
        let captured_val = Value::from_bits(uv_bits);
        let uv_obj = vm.gc.alloc_upvalue(std::ptr::null_mut());
        unsafe {
            (*uv_obj).closed = captured_val;
            (*uv_obj).location = &mut (*uv_obj).closed as *mut Value;
            if i < (*closure_ptr).upvalues.len() {
                (&mut (*closure_ptr).upvalues)[i] = uv_obj;
            }
        }
    }

    Value::object(closure_ptr as *mut u8).to_bits()
}

/// Allocate a closure with 0 upvalues.
pub extern "C" fn wren_make_closure_0(fn_id: u64) -> u64 {
    make_closure_inner(fn_id, &[])
}
/// Allocate a closure with 1 upvalue.
pub extern "C" fn wren_make_closure_1(fn_id: u64, uv0: u64) -> u64 {
    make_closure_inner(fn_id, &[uv0])
}
/// Allocate a closure with 2 upvalues.
pub extern "C" fn wren_make_closure_2(fn_id: u64, uv0: u64, uv1: u64) -> u64 {
    make_closure_inner(fn_id, &[uv0, uv1])
}
/// Allocate a closure with 3 upvalues.
pub extern "C" fn wren_make_closure_3(fn_id: u64, uv0: u64, uv1: u64, uv2: u64) -> u64 {
    make_closure_inner(fn_id, &[uv0, uv1, uv2])
}
/// Allocate a closure with 4 upvalues.
pub extern "C" fn wren_make_closure_4(fn_id: u64, uv0: u64, uv1: u64, uv2: u64, uv3: u64) -> u64 {
    make_closure_inner(fn_id, &[uv0, uv1, uv2, uv3])
}

/// Concatenate two strings.
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
    push_jit_root(val);
    val.to_bits()
}

/// Convert a value to its string representation.
pub extern "C" fn wren_to_string(val: u64) -> u64 {
    let v = Value::from_bits(val);

    let vm = unsafe { vm_ref() };
    let vm = match vm {
        Some(v) => v,
        None => return Value::null().to_bits(),
    };

    let s = crate::runtime::vm_interp::value_to_string(vm, v);
    let val = vm.new_string(s);
    push_jit_root(val);
    val.to_bits()
}

/// Materialize a string literal from its interned symbol id.
pub extern "C" fn wren_const_string(sym_idx: u64) -> u64 {
    let vm = unsafe { vm_ref() };
    let vm = match vm {
        Some(v) => v,
        None => return Value::null().to_bits(),
    };

    let sym = crate::intern::SymbolId::from_raw(sym_idx as u32);
    let val = vm.new_string(vm.interner.resolve(sym).to_string());
    push_jit_root(val);
    val.to_bits()
}

/// Type check: is value an instance of class?
/// class_sym is a SymbolId identifying the class name.
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

/// Subscript get (list[idx] or map[key]).
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
                if let Some(n) = idx.as_num() {
                    let i = n as usize;
                    let count = unsafe { (*list).count as usize };
                    if i < count {
                        if let Some(val) = unsafe { (*list).get(i) } {
                            return val.to_bits();
                        }
                    }
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
                if let Some(n) = idx.as_num() {
                    let i = n as usize;
                    let s = unsafe { (*string).as_str() };
                    if let Some(ch) = s.chars().nth(i) {
                        let vm = unsafe { vm_ref() };
                        if let Some(vm) = vm {
                            return vm.new_string(ch.to_string()).to_bits();
                        }
                    }
                }
            }
            _ => {}
        }
    }
    Value::null().to_bits()
}

/// Subscript set (list[idx] = val or map[key] = val).
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
                if let Some(n) = idx.as_num() {
                    let i = n as usize;
                    let count = unsafe { (*list).count as usize };
                    if i < count {
                        unsafe {
                            (*list).set(i, value);
                            if let Some(vm) = vm_ref() {
                                vm.gc.write_barrier(ptr as *mut ObjHeader, value);
                            }
                        }
                        return value.to_bits();
                    }
                }
            }
            ObjType::Map => {
                let map = ptr as *mut ObjMap;
                let map_key = MapKey::new(idx);
                unsafe {
                    (*map).entries.insert(map_key, value);
                    if let Some(vm) = vm_ref() {
                        vm.gc.write_barrier(ptr as *mut ObjHeader, idx);
                        vm.gc.write_barrier(ptr as *mut ObjHeader, value);
                    }
                }
                return value.to_bits();
            }
            _ => {}
        }
    }
    value.to_bits()
}

/// Get an upvalue by index from the current closure in JitContext.
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
            if let Some(vm) = vm_ref() {
                vm.gc.write_barrier(uv as *mut ObjHeader, value);
            }
        }
    }
    value.to_bits()
}

/// Get a static field from the defining class.
/// field_sym is the raw SymbolId index.
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
        if let Some(vm) = vm_ref() {
            vm.gc.write_barrier(class as *mut ObjHeader, value);
        }
    }
    value.to_bits()
}

/// Guard: check that value is an instance of the expected class.
/// Returns the value if check passes, traps otherwise.
pub extern "C" fn wren_guard_class(value: u64, class: u64) -> u64 {
    // For now, always pass the guard. A proper implementation would
    // check the class hierarchy and deoptimize on mismatch.
    let _ = class;
    value
}

/// Guard: check that value's class implements the expected protocol.
/// Returns the value if check passes (always passes for now).
pub extern "C" fn wren_guard_protocol(value: u64, protocol_id: u64) -> u64 {
    let _ = protocol_id;
    value
}

// ---------------------------------------------------------------------------
// Guard deoptimization — invalidate JIT + re-execute via interpreter
// ---------------------------------------------------------------------------

/// Common deopt implementation: invalidate JIT code for the current function
/// and re-execute it via the interpreter with the given args.
fn deopt_impl(args: &[u64]) -> u64 {
    let vm = unsafe { vm_ref() };
    let vm = match vm {
        Some(v) => v,
        None => return Value::null().to_bits(),
    };
    let ctx = read_jit_ctx();
    let closure_ptr = ctx.closure as *mut ObjClosure;
    if closure_ptr.is_null() {
        return Value::null().to_bits();
    }
    let func_id = unsafe { (*(*closure_ptr).function).fn_id } as usize;

    vm.engine
        .note_deopt_to_baseline(crate::runtime::engine::FuncId(func_id as u32));

    // Re-execute via interpreter with the original args.
    let defining_class = if ctx.defining_class.is_null() {
        None
    } else {
        Some(ctx.defining_class as *mut crate::runtime::object::ObjClass)
    };
    let values: Vec<Value> = args.iter().map(|&a| Value::from_bits(a)).collect();
    vm.call_closure_sync(closure_ptr, &values, defining_class)
        .map(|v| v.to_bits())
        .unwrap_or(Value::null().to_bits())
}

/// Deopt with 0 args (standalone function, no receiver).
pub extern "C" fn wren_deopt_0() -> u64 {
    deopt_impl(&[])
}

/// Deopt with 1 arg (e.g. standalone function with 1 param, or method with `this` only).
pub extern "C" fn wren_deopt_1(a0: u64) -> u64 {
    deopt_impl(&[a0])
}

/// Deopt with 2 args (e.g. method with `this` + 1 param).
pub extern "C" fn wren_deopt_2(a0: u64, a1: u64) -> u64 {
    deopt_impl(&[a0, a1])
}

/// Deopt with 3 args.
pub extern "C" fn wren_deopt_3(a0: u64, a1: u64, a2: u64) -> u64 {
    deopt_impl(&[a0, a1, a2])
}

/// Deopt with 4 args.
pub extern "C" fn wren_deopt_4(a0: u64, a1: u64, a2: u64, a3: u64) -> u64 {
    deopt_impl(&[a0, a1, a2, a3])
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

pub extern "C" fn wren_num_add(a: u64, b: u64) -> u64 {
    let va = Value::from_bits(a);
    if va.is_num() {
        return box_num(unbox_num(a) + unbox_num(b));
    }
    // Non-numeric receiver: dispatch as method call (string concat, operator overload)
    match unsafe { vm_ref() } {
        Some(vm) => {
            let sym = vm
                .interner
                .lookup("+(_)")
                .or_else(|| vm.interner.lookup("+"));
            if let Some(sym) = sym {
                let class = vm.class_of(va);
                if let Some((method, _dc)) = unsafe { find_method_with_class(class, sym) } {
                    return dispatch_method(vm, method, &[va, Value::from_bits(b)], None);
                }
            }
            Value::null().to_bits()
        }
        None => Value::null().to_bits(),
    }
}

pub extern "C" fn wren_num_sub(a: u64, b: u64) -> u64 {
    box_num(unbox_num(a) - unbox_num(b))
}

pub extern "C" fn wren_num_mul(a: u64, b: u64) -> u64 {
    box_num(unbox_num(a) * unbox_num(b))
}

pub extern "C" fn wren_num_div(a: u64, b: u64) -> u64 {
    box_num(unbox_num(a) / unbox_num(b))
}

pub extern "C" fn wren_num_mod(a: u64, b: u64) -> u64 {
    box_num(unbox_num(a) % unbox_num(b))
}

pub extern "C" fn wren_num_neg(a: u64) -> u64 {
    let va = Value::from_bits(a);
    if va.is_num() {
        return box_num(-unbox_num(a));
    }
    // Non-numeric: dispatch as prefix - method
    match unsafe { vm_ref() } {
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
            Value::null().to_bits()
        }
        None => Value::null().to_bits(),
    }
}

// ---------------------------------------------------------------------------
// Boxed comparison runtime functions
// ---------------------------------------------------------------------------

pub extern "C" fn wren_cmp_lt(a: u64, b: u64) -> u64 {
    Value::bool(unbox_num(a) < unbox_num(b)).to_bits()
}

pub extern "C" fn wren_cmp_gt(a: u64, b: u64) -> u64 {
    Value::bool(unbox_num(a) > unbox_num(b)).to_bits()
}

pub extern "C" fn wren_cmp_le(a: u64, b: u64) -> u64 {
    Value::bool(unbox_num(a) <= unbox_num(b)).to_bits()
}

pub extern "C" fn wren_cmp_ge(a: u64, b: u64) -> u64 {
    Value::bool(unbox_num(a) >= unbox_num(b)).to_bits()
}

pub extern "C" fn wren_cmp_eq(a: u64, b: u64) -> u64 {
    Value::bool(a == b).to_bits()
}

pub extern "C" fn wren_cmp_ne(a: u64, b: u64) -> u64 {
    Value::bool(a != b).to_bits()
}

// ---------------------------------------------------------------------------
// Boxed logical
// ---------------------------------------------------------------------------

pub extern "C" fn wren_not(a: u64) -> u64 {
    Value::bool(Value::from_bits(a).is_falsy()).to_bits()
}

pub extern "C" fn wren_is_truthy(value: u64) -> u64 {
    let v = Value::from_bits(value);
    // Return raw 0/1 (not NaN-boxed) so JmpZero can branch correctly.
    if v.is_falsy() {
        0u64
    } else {
        1u64
    }
}

// ---------------------------------------------------------------------------
// FP transcendental wrappers (raw f64 bits in/out, for JIT CallRuntime)
// ---------------------------------------------------------------------------

pub extern "C" fn wren_fp_sin(bits: u64) -> u64 {
    f64::from_bits(bits).sin().to_bits()
}
pub extern "C" fn wren_fp_cos(bits: u64) -> u64 {
    f64::from_bits(bits).cos().to_bits()
}
pub extern "C" fn wren_fp_tan(bits: u64) -> u64 {
    f64::from_bits(bits).tan().to_bits()
}
pub extern "C" fn wren_fp_asin(bits: u64) -> u64 {
    f64::from_bits(bits).asin().to_bits()
}
pub extern "C" fn wren_fp_acos(bits: u64) -> u64 {
    f64::from_bits(bits).acos().to_bits()
}
pub extern "C" fn wren_fp_atan(bits: u64) -> u64 {
    f64::from_bits(bits).atan().to_bits()
}
pub extern "C" fn wren_fp_log(bits: u64) -> u64 {
    f64::from_bits(bits).ln().to_bits()
}
pub extern "C" fn wren_fp_log2(bits: u64) -> u64 {
    f64::from_bits(bits).log2().to_bits()
}
pub extern "C" fn wren_fp_exp(bits: u64) -> u64 {
    f64::from_bits(bits).exp().to_bits()
}
pub extern "C" fn wren_fp_cbrt(bits: u64) -> u64 {
    f64::from_bits(bits).cbrt().to_bits()
}
pub extern "C" fn wren_fp_atan2(a: u64, b: u64) -> u64 {
    f64::from_bits(a).atan2(f64::from_bits(b)).to_bits()
}
pub extern "C" fn wren_fp_pow(a: u64, b: u64) -> u64 {
    f64::from_bits(a).powf(f64::from_bits(b)).to_bits()
}
pub extern "C" fn wren_fp_min(a: u64, b: u64) -> u64 {
    f64::from_bits(a).min(f64::from_bits(b)).to_bits()
}
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

/// Resolve a runtime function name to its address.
/// Returns `None` if the name is unknown.
pub fn resolve(name: &str) -> Option<usize> {
    match name {
        "wren_get_module_var" => Some(wren_get_module_var as *const () as usize),
        "wren_set_module_var" => Some(wren_set_module_var as *const () as usize),
        "wren_write_barrier" => Some(wren_write_barrier as *const () as usize),
        // Arity-specific call dispatch
        "wren_call_0" => Some(wren_call_0 as *const () as usize),
        "wren_call_1" => Some(wren_call_1 as *const () as usize),
        "wren_call_2" => Some(wren_call_2 as *const () as usize),
        "wren_call_3" => Some(wren_call_3 as *const () as usize),
        "wren_call_4" => Some(wren_call_4 as *const () as usize),
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
        // Collections
        // JIT pointer loading for CallKnownFunc
        "wren_load_jit_ptr" => Some(wren_load_jit_ptr as *const () as usize),
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
        // Strings
        "wren_string_concat" => Some(wren_string_concat as *const () as usize),
        "wren_to_string" => Some(wren_to_string as *const () as usize),
        "wren_const_string" => Some(wren_const_string as *const () as usize),
        // Type checks & guards
        "wren_is_type" => Some(wren_is_type as *const () as usize),
        "wren_guard_class" => Some(wren_guard_class as *const () as usize),
        "wren_guard_protocol" => Some(wren_guard_protocol as *const () as usize),
        // Guard deoptimization (arity-specific)
        "wren_deopt_0" => Some(wren_deopt_0 as *const () as usize),
        "wren_deopt_1" => Some(wren_deopt_1 as *const () as usize),
        "wren_deopt_2" => Some(wren_deopt_2 as *const () as usize),
        "wren_deopt_3" => Some(wren_deopt_3 as *const () as usize),
        "wren_deopt_4" => Some(wren_deopt_4 as *const () as usize),
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
        _ => None,
    }
}

/// Register a JIT function's frame pointer for GC stack walking.
/// Called from JIT prologue with FP, func_id, and return address.
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
pub extern "C" fn wren_jit_frame_pop() {
    pop_jit_frame();
}

// ---------------------------------------------------------------------------
// Inline IC native dispatch (kind=4)
// ---------------------------------------------------------------------------

/// Thin wrapper for native method dispatch from inline IC.
/// Takes (native_fn_ptr, receiver, args..., jit_fp) — the jit_fp is captured
/// by #[naked] wrappers to register the JIT frame for GC.
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
                    let f: crate::runtime::object::NativeFn = unsafe { std::mem::transmute(native_fn) };
                    f(vm, &[Value::from_bits(recv), $(Value::from_bits($arg)),*]).to_bits()
                }
                None => Value::null().to_bits(),
            };
            pop_jit_frame();
            result
        }
    };
}

ic_native_inner!(wren_ic_native_0_inner,);
ic_native_inner!(wren_ic_native_1_inner, a0);
ic_native_inner!(wren_ic_native_2_inner, a0, a1);
ic_native_inner!(wren_ic_native_3_inner, a0, a1, a2);

// #[naked] wrappers capture x29 (JIT FP) as the last argument.
/// # Safety
/// Called only from JIT-compiled code via inline IC dispatch (kind=4).
#[cfg(target_arch = "aarch64")]
#[unsafe(naked)]
pub unsafe extern "C" fn wren_ic_native_0(_nfn: u64, _recv: u64) -> u64 {
    core::arch::naked_asm!("mov x2, x29", "b {inner}", inner = sym wren_ic_native_0_inner);
}
#[cfg(not(target_arch = "aarch64"))]
pub extern "C" fn wren_ic_native_0(nfn: u64, recv: u64) -> u64 {
    wren_ic_native_0_inner(nfn, recv, 0)
}

/// # Safety
/// Called only from JIT-compiled code via inline IC dispatch (kind=4).
#[cfg(target_arch = "aarch64")]
#[unsafe(naked)]
pub unsafe extern "C" fn wren_ic_native_1(_nfn: u64, _recv: u64, _a0: u64) -> u64 {
    core::arch::naked_asm!("mov x3, x29", "b {inner}", inner = sym wren_ic_native_1_inner);
}
#[cfg(not(target_arch = "aarch64"))]
pub extern "C" fn wren_ic_native_1(nfn: u64, recv: u64, a0: u64) -> u64 {
    wren_ic_native_1_inner(nfn, recv, a0, 0)
}

/// # Safety
/// Called only from JIT-compiled code via inline IC dispatch (kind=4).
#[cfg(target_arch = "aarch64")]
#[unsafe(naked)]
pub unsafe extern "C" fn wren_ic_native_2(_nfn: u64, _recv: u64, _a0: u64, _a1: u64) -> u64 {
    core::arch::naked_asm!("mov x4, x29", "b {inner}", inner = sym wren_ic_native_2_inner);
}
#[cfg(not(target_arch = "aarch64"))]
pub extern "C" fn wren_ic_native_2(nfn: u64, recv: u64, a0: u64, a1: u64) -> u64 {
    wren_ic_native_2_inner(nfn, recv, a0, a1, 0)
}

/// # Safety
/// Called only from JIT-compiled code via inline IC dispatch (kind=4).
#[cfg(target_arch = "aarch64")]
#[unsafe(naked)]
pub unsafe extern "C" fn wren_ic_native_3(
    _nfn: u64,
    _recv: u64,
    _a0: u64,
    _a1: u64,
    _a2: u64,
) -> u64 {
    core::arch::naked_asm!("mov x5, x29", "b {inner}", inner = sym wren_ic_native_3_inner);
}
#[cfg(not(target_arch = "aarch64"))]
pub extern "C" fn wren_ic_native_3(nfn: u64, recv: u64, a0: u64, a1: u64, a2: u64) -> u64 {
    wren_ic_native_3_inner(nfn, recv, a0, a1, a2, 0)
}

/// Get the stable address of the thread-local JitContext.
/// Used to set x19 at interpreter→JIT entry points.
#[inline(always)]
pub fn jit_ctx_ptr() -> *mut JitContext {
    JIT_CTX.with(|c| c.get())
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
#[cfg(target_arch = "aarch64")]
#[unsafe(naked)]
pub unsafe extern "C" fn wren_ic_ctor_0(_cls: u64, _closure: u64) -> u64 {
    core::arch::naked_asm!("mov x2, x29", "b {inner}", inner = sym wren_ic_ctor_0_inner);
}
#[cfg(not(target_arch = "aarch64"))]
pub extern "C" fn wren_ic_ctor_0(cls: u64, closure: u64) -> u64 {
    wren_ic_ctor_0_inner(cls, closure, 0)
}

/// # Safety
/// Called only from JIT-compiled code via inline IC constructor dispatch (kind=3).
#[cfg(target_arch = "aarch64")]
#[unsafe(naked)]
pub unsafe extern "C" fn wren_ic_ctor_1(_cls: u64, _closure: u64, _a0: u64) -> u64 {
    core::arch::naked_asm!("mov x3, x29", "b {inner}", inner = sym wren_ic_ctor_1_inner);
}
#[cfg(not(target_arch = "aarch64"))]
pub extern "C" fn wren_ic_ctor_1(cls: u64, closure: u64, a0: u64) -> u64 {
    wren_ic_ctor_1_inner(cls, closure, a0, 0)
}

/// # Safety
/// Called only from JIT-compiled code via inline IC constructor dispatch (kind=3).
#[cfg(target_arch = "aarch64")]
#[unsafe(naked)]
pub unsafe extern "C" fn wren_ic_ctor_2(_cls: u64, _closure: u64, _a0: u64, _a1: u64) -> u64 {
    core::arch::naked_asm!("mov x4, x29", "b {inner}", inner = sym wren_ic_ctor_2_inner);
}
#[cfg(not(target_arch = "aarch64"))]
pub extern "C" fn wren_ic_ctor_2(cls: u64, closure: u64, a0: u64, a1: u64) -> u64 {
    wren_ic_ctor_2_inner(cls, closure, a0, a1, 0)
}

/// # Safety
/// Called only from JIT-compiled code via inline IC constructor dispatch (kind=3).
#[cfg(target_arch = "aarch64")]
#[unsafe(naked)]
pub unsafe extern "C" fn wren_ic_ctor_3(
    _cls: u64,
    _closure: u64,
    _a0: u64,
    _a1: u64,
    _a2: u64,
) -> u64 {
    core::arch::naked_asm!("mov x5, x29", "b {inner}", inner = sym wren_ic_ctor_3_inner);
}
#[cfg(not(target_arch = "aarch64"))]
pub extern "C" fn wren_ic_ctor_3(cls: u64, closure: u64, a0: u64, a1: u64, a2: u64) -> u64 {
    wren_ic_ctor_3_inner(cls, closure, a0, a1, a2, 0)
}

/// Allocate an ObjInstance for a class. Used by inline constructor IC (kind=3).
/// Takes the class as a NaN-boxed Value (receiver of the constructor call).
/// Returns the new instance as a NaN-boxed Value.
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
            Value::object(instance as *mut u8).to_bits()
        }
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
pub extern "C" fn wren_ic_enter(func_id: u64, closure: u64) -> u64 {
    JIT_CTX.with(|c| unsafe {
        let ctx = &mut *c.get();
        let saved = ctx.current_func_id;
        ctx.current_func_id = func_id;
        ctx.closure = closure as *mut u8;
        saved
    })
}

/// Inline IC leave: restore current_func_id after a non-leaf JIT call.
pub extern "C" fn wren_ic_leave(saved_func_id: u64) {
    JIT_CTX.with(|c| unsafe {
        (*c.get()).current_func_id = saved_func_id;
    });
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
        if let Some(d) = inst.def() {
            if d.class == super::RegClass::Gp {
                defs.insert(d.index);
            }
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
pub fn link_runtime_calls(
    mf: &mut MachFunc,
    target: super::Target,
    assignments: &std::collections::HashMap<VReg, crate::codegen::regalloc::Location>,
    native_meta: Option<&crate::codegen::native_meta::NativeFrameMetadata>,
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
        Runtime(
            usize,
            &'static str,
            Vec<VReg>,
            Option<VReg>,
            Option<Vec<crate::codegen::native_meta::LiveRootMetadata>>,
        ),
        Local(
            usize,
            Label,
            Vec<VReg>,
            Option<VReg>,
            Option<Vec<crate::codegen::native_meta::LiveRootMetadata>>,
        ),
        Indirect(usize, VReg, Vec<VReg>, Option<VReg>),
    }

    let mut call_sites: Vec<PendingCall> = Vec::new();
    let mut safepoint_iter = native_meta
        .map(|meta| meta.safepoints.iter())
        .into_iter()
        .flatten();
    for (orig_i, inst) in mf.insts.iter().enumerate() {
        match inst {
            MachInst::CallRuntime { name, args, ret } => {
                let live_roots = if *name != "wren_shadow_store"
                    && *name != "wren_shadow_load"
                    && *name != "wren_enter_shadow_frame"
                    && *name != "wren_exit_shadow_frame"
                {
                    safepoint_iter.next().map(|sp| sp.live_roots.clone())
                } else {
                    None
                };
                call_sites.push(PendingCall::Runtime(
                    orig_i,
                    name,
                    args.clone(),
                    *ret,
                    live_roots,
                ));
            }
            MachInst::CallLocal { target, args, ret } => {
                let live_roots = safepoint_iter.next().map(|sp| sp.live_roots.clone());
                call_sites.push(PendingCall::Local(
                    orig_i,
                    *target,
                    args.clone(),
                    *ret,
                    live_roots,
                ));
            }
            MachInst::CallIndirectAbi { target, args, ret } => {
                call_sites.push(PendingCall::Indirect(orig_i, *target, args.clone(), *ret));
            }
            _ => {}
        }
    }

    for call_site in call_sites.into_iter().rev() {
        let (orig_i, new_insts) = match call_site {
            PendingCall::Runtime(orig_i, name, args, ret, live_roots) => {
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
                    live_roots.as_deref(),
                );
                (orig_i, new_insts)
            }
            PendingCall::Local(orig_i, label, args, ret, live_roots) => {
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
                    live_roots.as_deref(),
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
                    None,
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
    live_roots: Option<&[crate::codegen::native_meta::LiveRootMetadata]>,
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
    let _ = live_roots;

    new_insts
}

#[allow(dead_code)]
fn target_data_addr(name: &'static str) -> u64 {
    resolve(name).unwrap_or(0) as u64
}

#[allow(dead_code)]
fn append_shadow_reloads(
    new_insts: &mut Vec<MachInst>,
    shadow_roots_ptr_addr: u64,
    call_scratch: u32,
    copy_scratch: u32,
    frame_ptr: VReg,
    live_roots: &[crate::codegen::native_meta::LiveRootMetadata],
) {
    if shadow_roots_ptr_addr == 0 {
        return;
    }

    for root in live_roots {
        let crate::codegen::native_meta::RootLocation::Spill(offset) = root.location else {
            continue;
        };
        new_insts.push(MachInst::LoadImm {
            dst: VReg::gp(call_scratch),
            bits: shadow_roots_ptr_addr,
        });
        new_insts.push(MachInst::Ldr {
            dst: VReg::gp(copy_scratch),
            mem: Mem::new(VReg::gp(call_scratch), 0),
        });
        new_insts.push(MachInst::Ldr {
            dst: VReg::gp(call_scratch),
            mem: Mem::new(VReg::gp(copy_scratch), (root.slot as i32) * 8),
        });
        new_insts.push(MachInst::Str {
            src: VReg::gp(call_scratch),
            mem: Mem::new(frame_ptr, offset),
        });
    }
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

    // Phase 1: Emit moves whose destination is NOT a source of any other move.
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

    // Phase 2: Only cycles remain. Break each cycle using the scratch register.
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
    fn test_native_shadow_roots_round_trip() {
        push_native_shadow_frame(2);
        assert_eq!(
            wren_shadow_store(0, Value::num(7.0).to_bits()),
            Value::num(7.0).to_bits()
        );
        assert_eq!(
            wren_shadow_store(1, Value::bool(true).to_bits()),
            Value::bool(true).to_bits()
        );

        let (lengths, roots) = take_native_shadow_roots();
        assert_eq!(lengths, vec![2]);
        assert_eq!(roots, vec![Value::num(7.0), Value::bool(true)]);

        set_native_shadow_roots(lengths, vec![Value::null(), Value::num(11.0)]);
        let (lengths, roots) = take_native_shadow_roots();
        assert_eq!(lengths, vec![2]);
        assert_eq!(roots, vec![Value::null(), Value::num(11.0)]);

        set_native_shadow_roots(Vec::new(), Vec::new());
        pop_native_shadow_frame();
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
        link_runtime_calls(&mut mf, Target::Aarch64, &assignments, None);

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
        link_runtime_calls(&mut mf, Target::Aarch64, &assignments, None);

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
    fn test_shadow_reloads_do_not_use_abi_return_as_scratch() {
        let ret = VReg::gp(12);
        let mut mf = MachFunc::new("test".into());
        mf.emit(MachInst::CallRuntime {
            name: "wren_not",
            args: vec![VReg::gp(10)],
            ret: Some(ret),
        });
        mf.emit(MachInst::Ret);

        let assignments = std::collections::HashMap::from([
            (VReg::gp(10), Location::Reg(PhysReg::gp(3))),
            (ret, Location::Reg(PhysReg::gp(0))),
            (VReg::gp(20), Location::Spill(-8)),
        ]);
        let native_meta = crate::codegen::native_meta::NativeFrameMetadata {
            boxed_values: vec![VReg::gp(20)],
            safepoints: vec![crate::codegen::native_meta::SafepointMetadata {
                ordinal: 0,
                inst_index: 0,
                code_offset: 0,
                kind: crate::codegen::native_meta::SafepointKind::CallRuntime,
                live_roots: vec![crate::codegen::native_meta::LiveRootMetadata {
                    slot: 0,
                    location: crate::codegen::native_meta::RootLocation::Spill(-8),
                }],
            }],
            spill_safe_nonleaf: true,
        };

        link_runtime_calls(&mut mf, Target::Aarch64, &assignments, Some(&native_meta));

        let call_idx = mf
            .insts
            .iter()
            .position(|inst| matches!(inst, MachInst::CallInd { .. }))
            .expect("linked call should contain CallInd");
        let clobbers_ret = mf.insts[call_idx + 1..].iter().any(|inst| {
            matches!(
                inst,
                MachInst::Ldr {
                    dst,
                    mem: Mem { base, offset: 0 }
                } if *dst == VReg::gp(0) && *base == VReg::gp(17)
            )
        });
        assert!(
            !clobbers_ret,
            "shadow reloads must not reuse the ABI return register as a scratch"
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
        link_runtime_calls(&mut mf, Target::Aarch64, &assignments, None);

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
