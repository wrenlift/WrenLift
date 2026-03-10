use crate::runtime::object::{
    MapKey, Method, ObjClass, ObjClosure, ObjHeader, ObjList, ObjMap, ObjString, ObjType,
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
    /// Current closure pointer (for upvalue access from JIT-compiled closure bodies).
    pub closure: *mut u8,
    /// The class that defines the current method (for static field access).
    pub defining_class: *mut u8,
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
            closure: std::ptr::null_mut(),
            defining_class: std::ptr::null_mut(),
        }
    }
}

// ---------------------------------------------------------------------------
// JIT context and root set — using Cell/UnsafeCell for zero-overhead access.
// thread_local! ensures test parallelism safety; Cell/UnsafeCell eliminates
// RefCell borrow-checking overhead on the hot path.
// ---------------------------------------------------------------------------

thread_local! {
    /// JIT context — Cell<JitContext> for zero-overhead get/set (JitContext is Copy).
    static JIT_CTX: std::cell::Cell<JitContext> = const { std::cell::Cell::new(JitContext {
        module_vars: std::ptr::null_mut(),
        module_var_count: 0,
        vm: std::ptr::null_mut(),
        module_name: std::ptr::null(),
        module_name_len: 0,
        closure: std::ptr::null_mut(),
        defining_class: std::ptr::null_mut(),
    }) };

    /// JIT root set — UnsafeCell for zero-overhead Vec mutation.
    /// SAFETY: single-threaded access within each thread (no concurrent borrows).
    static JIT_ROOTS_STORE: std::cell::UnsafeCell<Vec<Value>> = const { std::cell::UnsafeCell::new(Vec::new()) };

    /// JIT native recursion depth — guards against native stack overflow.
    /// When depth exceeds MAX_JIT_DEPTH, calls fall back to interpreter dispatch.
    static JIT_DEPTH: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
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

/// Set up the JIT context before calling compiled code.
#[inline(always)]
pub fn set_jit_context(ctx: JitContext) {
    JIT_CTX.with(|c| c.set(ctx));
}

/// Read the current JIT context.
#[inline(always)]
fn read_jit_ctx() -> JitContext {
    JIT_CTX.with(|c| c.get())
}

/// Mutate the JIT context in place.
#[inline(always)]
fn mutate_jit_ctx(f: impl FnOnce(&mut JitContext)) {
    JIT_CTX.with(|c| {
        let mut ctx = c.get();
        f(&mut ctx);
        c.set(ctx);
    });
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

/// Clear JIT roots (called after native code returns to interpreter).
#[inline(always)]
pub fn clear_jit_roots() {
    JIT_ROOTS_STORE.with(|r| unsafe { (*r.get()).clear() });
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
    let vm = unsafe { vm_ref() };
    if let Some(vm) = vm {
        let ctx = read_jit_ctx();
        if !ctx.module_name.is_null() {
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

/// Try to call a closure via native code if it's compiled; fall back to interpreter.
fn call_closure_jit_or_sync(
    vm: &mut crate::runtime::vm::VM,
    closure_ptr: *mut ObjClosure,
    args: &[Value],
) -> u64 {
    // Install completed compilations so we can dispatch natively.
    vm.engine.poll_compilations();
    if args.len() <= 4 {
        let func_id = crate::runtime::engine::FuncId(unsafe { (*(*closure_ptr).function).fn_id });
        // Check if this function has been compiled to native code.
        let native_fn_ptr: Option<*const u8> =
            if let Some(crate::runtime::engine::FuncBody::Compiled { executable, .. }) =
                vm.engine.get_function(func_id)
            {
                if executable.is_native() {
                    Some(executable.native_ptr())
                } else {
                    None
                }
            } else {
                None
            };

        if let Some(fn_ptr) = native_fn_ptr {
            // Guard: check native recursion depth to prevent stack overflow.
            // Each JIT call level uses ~1-2KB native stack; limit to 256 levels.
            let depth = JIT_DEPTH.with(|d| d.get());
            if depth < MAX_JIT_DEPTH {
                JIT_DEPTH.with(|d| d.set(depth + 1));
                // Native-to-native dispatch: call compiled code directly.
                let result = unsafe {
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
                            let f: extern "C" fn(u64, u64, u64) -> u64 =
                                std::mem::transmute(fn_ptr);
                            f(args[0].to_bits(), args[1].to_bits(), args[2].to_bits())
                        }
                        _ => {
                            let f: extern "C" fn(u64, u64, u64, u64) -> u64 =
                                std::mem::transmute(fn_ptr);
                            f(
                                args[0].to_bits(),
                                args[1].to_bits(),
                                args[2].to_bits(),
                                args[3].to_bits(),
                            )
                        }
                    }
                };
                JIT_DEPTH.with(|d| d.set(depth));
                return result;
            }
            // depth >= MAX_JIT_DEPTH: fall through to interpreter path.
        }
    }

    // Fall back to interpreter.
    vm.call_closure_sync(closure_ptr, args)
        .map(|v| v.to_bits())
        .unwrap_or(Value::null().to_bits())
}

/// Internal: dispatch a method call with a pre-built args slice.
fn dispatch_call(recv: Value, method_sym: crate::intern::SymbolId, args: &[Value]) -> u64 {
    let vm = unsafe { vm_ref() };
    let vm = match vm {
        Some(v) => v,
        None => return Value::null().to_bits(),
    };

    let class = vm.class_of(recv);

    // For class receivers, use the actual class object as cache key
    // (not class_class, since all classes share that metaclass).
    let cache_key_class = if class == vm.class_class {
        recv.as_object().unwrap_or(std::ptr::null_mut()) as *mut ObjClass
    } else {
        class
    };

    // Check method cache first (avoids find_method + static symbol resolution).
    if let Some((m, _dc)) = vm.method_cache.lookup(cache_key_class, method_sym) {
        return dispatch_method(vm, m, args);
    }

    // Cache miss: full lookup.
    let method_entry = unsafe { (*class).find_method(method_sym).cloned() };

    match method_entry {
        Some(m) => {
            vm.method_cache
                .insert(cache_key_class, method_sym, m.clone(), class);
            dispatch_method(vm, m, args)
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
                        .and_then(|sym| unsafe { (*cache_key_class).find_method(sym).cloned() })
                } else {
                    None
                };
                if let Some(m) = found {
                    vm.method_cache
                        .insert(cache_key_class, method_sym, m.clone(), cache_key_class);
                    dispatch_method(vm, m, args)
                } else {
                    Value::null().to_bits()
                }
            } else {
                Value::null().to_bits()
            }
        }
    }
}

/// Dispatch a resolved method entry.
#[inline(always)]
fn dispatch_method(vm: &mut crate::runtime::vm::VM, method: Method, args: &[Value]) -> u64 {
    match method {
        Method::Native(native_fn) => native_fn(vm, args).to_bits(),
        Method::Closure(cp) => call_closure_jit_or_sync(vm, cp, args),
        Method::Constructor(cp) => call_closure_jit_or_sync(vm, cp, args),
    }
}

/// Call a method with 0 extra args. Codegen: [receiver, method_sym]
pub extern "C" fn wren_call_0(receiver: u64, method: u64) -> u64 {
    let recv = Value::from_bits(receiver);
    let sym = crate::intern::SymbolId::from_raw(method as u32);
    dispatch_call(recv, sym, &[recv])
}

/// Call with 1 extra arg. Codegen: [receiver, method_sym, arg0]
pub extern "C" fn wren_call_1(receiver: u64, method: u64, a0: u64) -> u64 {
    let recv = Value::from_bits(receiver);
    let sym = crate::intern::SymbolId::from_raw(method as u32);
    dispatch_call(recv, sym, &[recv, Value::from_bits(a0)])
}

/// Call with 2 extra args.
pub extern "C" fn wren_call_2(receiver: u64, method: u64, a0: u64, a1: u64) -> u64 {
    let recv = Value::from_bits(receiver);
    let sym = crate::intern::SymbolId::from_raw(method as u32);
    dispatch_call(
        recv,
        sym,
        &[recv, Value::from_bits(a0), Value::from_bits(a1)],
    )
}

/// Call with 3 extra args.
pub extern "C" fn wren_call_3(receiver: u64, method: u64, a0: u64, a1: u64, a2: u64) -> u64 {
    let recv = Value::from_bits(receiver);
    let sym = crate::intern::SymbolId::from_raw(method as u32);
    dispatch_call(
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

/// Call with 4 extra args (max via registers).
pub extern "C" fn wren_call_4(
    receiver: u64,
    method: u64,
    a0: u64,
    a1: u64,
    a2: u64,
    a3: u64,
) -> u64 {
    let recv = Value::from_bits(receiver);
    let sym = crate::intern::SymbolId::from_raw(method as u32);
    dispatch_call(
        recv,
        sym,
        &[
            recv,
            Value::from_bits(a0),
            Value::from_bits(a1),
            Value::from_bits(a2),
            Value::from_bits(a3),
        ],
    )
}

/// Internal: dispatch a super call. Walks to superclass and dispatches method.
fn dispatch_super_call(recv: Value, method_sym: crate::intern::SymbolId, args: &[Value]) -> u64 {
    let vm = unsafe { vm_ref() };
    let vm = match vm {
        Some(v) => v,
        None => return Value::null().to_bits(),
    };

    let class = vm.class_of(recv);
    let superclass = unsafe { (*class).superclass };
    if superclass.is_null() {
        return Value::null().to_bits();
    }

    let method_entry = unsafe { (*superclass).find_method(method_sym).cloned() };
    match method_entry {
        Some(Method::Native(native_fn)) => native_fn(vm, args).to_bits(),
        Some(Method::Closure(closure_ptr)) => call_closure_jit_or_sync(vm, closure_ptr, args),
        Some(Method::Constructor(closure_ptr)) => call_closure_jit_or_sync(vm, closure_ptr, args),
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
pub extern "C" fn wren_make_list() -> u64 {
    let vm = unsafe { vm_ref() };
    let vm = match vm {
        Some(v) => v,
        None => return Value::null().to_bits(),
    };

    let list_ptr = vm.gc.alloc_list();
    unsafe {
        (*list_ptr).header.class = vm.list_class;
    }
    let val = Value::object(list_ptr as *mut u8);
    push_jit_root(val);
    val.to_bits()
}

/// Allocate a new empty map.
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
                            (*list).set(i, Value::from_bits(value));
                        }
                        return value;
                    }
                }
            }
            ObjType::Map => {
                let map = ptr as *mut ObjMap;
                let map_key = MapKey::new(idx);
                unsafe {
                    (*map).entries.insert(map_key, Value::from_bits(value));
                }
                return value;
            }
            _ => {}
        }
    }
    value
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
    unsafe {
        let upvalues = &(*closure).upvalues;
        if idx < upvalues.len() {
            let uv = upvalues[idx];
            (*uv).set(Value::from_bits(value));
        }
    }
    value
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
    unsafe {
        (*class).static_fields.insert(sym, Value::from_bits(value));
    }
    value
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
    box_num(unbox_num(a) + unbox_num(b))
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
    box_num(-unbox_num(a))
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
            11,                  // R11 for cycle breaking (safe: copy resolves before ptr load)
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
        // Arity-specific call dispatch
        "wren_call_0" => Some(wren_call_0 as *const () as usize),
        "wren_call_1" => Some(wren_call_1 as *const () as usize),
        "wren_call_2" => Some(wren_call_2 as *const () as usize),
        "wren_call_3" => Some(wren_call_3 as *const () as usize),
        "wren_call_4" => Some(wren_call_4 as *const () as usize),
        "wren_super_call_0" => Some(wren_super_call_0 as *const () as usize),
        "wren_super_call_1" => Some(wren_super_call_1 as *const () as usize),
        "wren_super_call_2" => Some(wren_super_call_2 as *const () as usize),
        "wren_super_call_3" => Some(wren_super_call_3 as *const () as usize),
        "wren_super_call_4" => Some(wren_super_call_4 as *const () as usize),
        // Collections
        "wren_make_list" => Some(wren_make_list as *const () as usize),
        "wren_make_map" => Some(wren_make_map as *const () as usize),
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
        // Type checks & guards
        "wren_is_type" => Some(wren_is_type as *const () as usize),
        "wren_guard_class" => Some(wren_guard_class as *const () as usize),
        "wren_guard_protocol" => Some(wren_guard_protocol as *const () as usize),
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
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// CallRuntime → ABI setup + CallInd lowering pass
// ---------------------------------------------------------------------------

use crate::codegen::{MachFunc, MachInst, VReg};

/// Lower all `CallRuntime` instructions to proper ABI calling sequences.
///
/// This pass MUST run AFTER register allocation and sentinel fixup, when all
/// vregs are physical register indices. It uses parallel copy resolution to
/// avoid the classic register-clobbering problem with sequential moves.
///
/// Sequence per call:
/// 1. Parallel copy: move arguments from their physical registers to ABI arg registers
/// 2. Load function address into scratch register
/// 3. Indirect call via scratch
/// 4. Move return value from ABI return register to destination
pub fn link_runtime_calls(mf: &mut MachFunc, target: super::Target) {
    let (abi_args, abi_ret, call_scratch, copy_scratch) = abi_regs(target);
    if abi_args.is_empty() {
        return;
    } // Wasm: no runtime calls

    // Build caller-saved register set for this target.
    let caller_saved: &[super::PhysReg] = match target {
        super::Target::Aarch64 => super::phys_aarch64::ABI.gp_caller_saved,
        super::Target::X86_64 => super::phys_x86_64::ABI.gp_caller_saved,
        super::Target::Wasm => return,
    };
    let caller_saved_set: std::collections::HashSet<u32> =
        caller_saved.iter().map(|r| r.hw_enc as u32).collect();

    let mut i = 0;
    while i < mf.insts.len() {
        if let MachInst::CallRuntime { name, args, ret } = &mf.insts[i] {
            let name = *name;
            let args = args.clone();
            let ret = *ret;

            if let Some(addr) = resolve(name) {
                // Compute which caller-saved GP registers are live across this call:
                // any register referenced (used or defined) by instructions after this call.
                let mut live_across: std::collections::HashSet<u32> =
                    std::collections::HashSet::new();
                for j in (i + 1)..mf.insts.len() {
                    for u in mf.insts[j].uses() {
                        if u.class == super::RegClass::Gp {
                            live_across.insert(u.index);
                        }
                    }
                    if let Some(d) = mf.insts[j].def() {
                        if d.class == super::RegClass::Gp {
                            live_across.insert(d.index);
                        }
                    }
                }

                // Only save caller-saved regs that are live, excluding scratch and
                // result registers. ABI arg registers are NOT excluded because
                // they may hold live values that the parallel copy will overwrite
                // (e.g., n sits in x1 and x1 is also the second ABI arg slot).
                //
                // abi_ret (x0) is only excluded when ret_vreg == abi_ret, because
                // in that case the pop would overwrite the call result. When
                // ret_vreg != abi_ret, the result Mov happens before pops, so
                // popping abi_ret safely restores the old live value.
                let mut excluded: std::collections::HashSet<u32> = std::collections::HashSet::new();
                if ret.is_none_or(|rv| rv.index == abi_ret) {
                    excluded.insert(abi_ret);
                }
                excluded.insert(call_scratch);
                excluded.insert(copy_scratch);
                // Also exclude the destination register — pop must not overwrite
                // the call result that was moved there.
                if let Some(rv) = ret {
                    excluded.insert(rv.index);
                }

                let mut save_regs: Vec<u32> = caller_saved_set
                    .intersection(&live_across)
                    .copied()
                    .filter(|r| !excluded.contains(r))
                    .collect();
                save_regs.sort(); // deterministic order

                let mut new_insts: Vec<MachInst> = Vec::new();

                // 0. Push caller-saved live registers
                for &reg in &save_regs {
                    new_insts.push(MachInst::Push { src: VReg::gp(reg) });
                }

                // 1. Build (src_phys, dst_abi) move pairs and resolve in parallel
                let moves: Vec<(u32, u32)> = args
                    .iter()
                    .enumerate()
                    .filter(|(idx, _)| *idx < abi_args.len())
                    .map(|(idx, vreg)| (vreg.index, abi_args[idx]))
                    .collect();

                for (src, dst) in resolve_parallel_copy(&moves, copy_scratch) {
                    new_insts.push(MachInst::Mov {
                        dst: VReg::gp(dst),
                        src: VReg::gp(src),
                    });
                }

                // 2. Load function address into scratch register
                new_insts.push(MachInst::LoadImm {
                    dst: VReg::gp(call_scratch),
                    bits: addr as u64,
                });

                // 3. Indirect call
                new_insts.push(MachInst::CallInd {
                    target: VReg::gp(call_scratch),
                });

                // 4. Move return value from ABI return register
                if let Some(ret_vreg) = ret {
                    if ret_vreg.index != abi_ret {
                        new_insts.push(MachInst::Mov {
                            dst: ret_vreg,
                            src: VReg::gp(abi_ret),
                        });
                    }
                }

                // 5. Pop caller-saved live registers (reverse order)
                for &reg in save_regs.iter().rev() {
                    new_insts.push(MachInst::Pop { dst: VReg::gp(reg) });
                }

                let seq_len = new_insts.len();
                mf.insts.splice(i..=i, new_insts);
                i += seq_len;
            } else {
                i += 1;
            }
        } else {
            i += 1;
        }
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

        // Follow the cycle chain.
        let mut current = first_dst;
        while let Some(idx) = remaining.iter().position(|(s, _)| *s == current) {
            let (_, next_dst) = remaining.remove(idx);
            result.push((current, next_dst));
            current = next_dst;
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
            closure: std::ptr::null_mut(),
            defining_class: std::ptr::null_mut(),
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
            closure: std::ptr::null_mut(),
            defining_class: std::ptr::null_mut(),
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
            closure: std::ptr::null_mut(),
            defining_class: std::ptr::null_mut(),
        });

        let new_val = Value::num(99.0).to_bits();
        wren_set_module_var(1, new_val);

        // Verify the module var was updated
        assert_eq!(vars[1], new_val);

        // Clean up
        set_jit_context(JitContext::default());
    }

    #[test]
    fn test_link_inserts_abi_moves_aarch64() {
        // Simulate post-regalloc state: args already in physical registers.
        let mut mf = MachFunc::new("test".into());
        // Pretend regalloc assigned arg0 to X3, arg1 to X5, ret to X3.
        let arg0 = VReg::gp(3);
        let arg1 = VReg::gp(5);
        let ret = VReg::gp(3);

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
        link_runtime_calls(&mut mf, crate::codegen::Target::Aarch64);

        // CallRuntime replaced: 2 arg movs + LoadImm + CallInd + ret mov = 5 new
        assert_eq!(mf.insts.len(), before_len + 4);

        // Verify moves target ABI physical registers (X0=0, X1=1)
        let has_x0_move = mf
            .insts
            .iter()
            .any(|inst| matches!(inst, MachInst::Mov { dst, .. } if dst.index == 0));
        assert!(has_x0_move, "should have move to X0");

        let has_x1_move = mf
            .insts
            .iter()
            .any(|inst| matches!(inst, MachInst::Mov { dst, .. } if dst.index == 1));
        assert!(has_x1_move, "should have move to X1");
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
