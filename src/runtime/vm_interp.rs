use smallvec::SmallVec;

/// Fiber-based MIR interpreter with bytecode dispatch.
///
/// All execution happens within a Fiber context. The interpreter loop
/// runs on the active fiber's `mir_frames` stack:
/// - Call instructions push new frames onto the fiber
/// - Return pops a frame and passes the result to the caller
/// - When the last frame is popped, the fiber is Done
///
/// Instructions are decoded from a flat bytecode stream (Op enum),
/// with operands read inline. Pure instruction evaluation (arithmetic,
/// comparisons, guards, box/unbox, bitwise, constants, moves) is inlined.
/// VM-specific instructions (calls, collections, module vars, string
/// allocation, fields, closures) have full VM access.
use std::rc::Rc;

use crate::intern::SymbolId;
use crate::mir::bytecode::{read_u16, read_u32, read_u8, BcConst, BytecodeFunction, Op};
use crate::mir::interp::InterpError;
use crate::mir::{BlockId, ValueId};
use crate::runtime::engine::ExecutionMode;
use crate::runtime::engine::FuncId;
use crate::runtime::object::*;
use crate::runtime::value::Value;
use crate::runtime::vm::{FiberAction, VM};

// ---------------------------------------------------------------------------
// Global inline method cache
// ---------------------------------------------------------------------------

/// Direct-mapped method cache: (class_ptr, method_sym) → (Method, defining_class).
/// Eliminates HashMap lookups for monomorphic call sites.
const METHOD_CACHE_SIZE: usize = 1024; // power of 2

struct MethodCacheEntry {
    class: *mut ObjClass,
    method_raw: u32,
    result: Method,
    defining_class: *mut ObjClass,
}

pub struct MethodCache {
    entries: Vec<Option<MethodCacheEntry>>,
}

impl Default for MethodCache {
    fn default() -> Self {
        Self::new()
    }
}

impl MethodCache {
    pub fn new() -> Self {
        let mut entries = Vec::with_capacity(METHOD_CACHE_SIZE);
        entries.resize_with(METHOD_CACHE_SIZE, || None);
        Self { entries }
    }

    #[inline(always)]
    fn hash(class: *mut ObjClass, method: SymbolId) -> usize {
        ((class as usize).wrapping_mul(2654435761) ^ (method.index() as usize))
            & (METHOD_CACHE_SIZE - 1)
    }

    #[inline]
    pub fn lookup(
        &self,
        class: *mut ObjClass,
        method: SymbolId,
    ) -> Option<(Method, *mut ObjClass)> {
        let idx = Self::hash(class, method);
        if let Some(entry) = unsafe { self.entries.get_unchecked(idx) } {
            if entry.class == class && entry.method_raw == method.index() {
                return Some((entry.result, entry.defining_class));
            }
        }
        None
    }

    #[inline]
    pub fn insert(
        &mut self,
        class: *mut ObjClass,
        method: SymbolId,
        result: Method,
        defining_class: *mut ObjClass,
    ) {
        let idx = Self::hash(class, method);
        self.entries[idx] = Some(MethodCacheEntry {
            class,
            method_raw: method.index(),
            result,
            defining_class,
        });
    }

    /// Invalidate all cache entries. Must be called after GC since
    /// nursery promotion can relocate ObjClass/ObjClosure pointers.
    pub fn invalidate(&mut self) {
        for entry in &mut self.entries {
            *entry = None;
        }
    }
}

/// Sentinel value for uninitialized register slots.
const UNDEF: Value = Value::UNDEFINED;

#[inline]
fn call_native_with_frame_sync(
    vm: &mut VM,
    fiber: *mut ObjFiber,
    pc: u32,
    values: &mut Vec<Value>,
    func: NativeFn,
    args: &[Value],
) -> Value {
    unsafe {
        if let Some(frame) = (*fiber).mir_frames.last_mut() {
            frame.values = std::mem::take(values);
            frame.pc = pc;
        }
    }

    // Catch panics so a native-code crash surfaces as a
    // fiber-catchable runtime error instead of aborting the
    // whole process. Wren is a library runtime — a bug in a
    // third-party crate called from a Wren script shouldn't
    // take down the host.
    let result = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| func(vm, args))) {
        Ok(v) => v,
        Err(panic) => {
            let msg = panic_message(&panic);
            vm.has_error = true;
            vm.last_error = Some(format!("native panic: {}", msg));
            Value::null()
        }
    };

    unsafe {
        if let Some(frame) = (*fiber).mir_frames.last_mut() {
            *values = std::mem::take(&mut frame.values);
        }
    }

    result
}

/// Recover the payload string of a caught panic. Panics can
/// carry any type via `panic!(some_value)`, but in practice the
/// vast majority are either `&'static str` (literal panic
/// messages) or `String` (formatted `panic!("{}", ...)`). Fall
/// back to a placeholder for anything else.
fn panic_message(panic: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = panic.downcast_ref::<&'static str>() {
        return s.to_string();
    }
    if let Some(s) = panic.downcast_ref::<String>() {
        return s.clone();
    }
    "(no message)".to_string()
}

/// Mirror of `call_native_with_frame_sync` for C-ABI foreign methods
/// bound from a dynamic library. Saves the current interpreter frame
/// state around the call so the foreign function can re-enter the VM
/// without clobbering live register files.
#[inline]
fn call_foreign_dynamic_with_frame_sync(
    vm: &mut VM,
    fiber: *mut ObjFiber,
    pc: u32,
    values: &mut Vec<Value>,
    idx: u32,
    args: &[Value],
) -> Value {
    // Mirrors `call_foreign_c_with_frame_sync` for the dynamic
    // plugin path. Same frame-sync + panic-guard discipline; the
    // only difference is the dispatcher we call into resolves the
    // (lib, sym) from the index and forwards across the wasm
    // module boundary via the JS-side bridge.
    unsafe {
        if let Some(frame) = (*fiber).mir_frames.last_mut() {
            frame.values = std::mem::take(values);
            frame.pc = pc;
        }
    }
    let result = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        crate::runtime::foreign::dispatch_dynamic(vm, idx, args)
    })) {
        Ok(v) => v,
        Err(panic) => {
            let msg = panic_message(&panic);
            vm.has_error = true;
            vm.last_error = Some(format!("foreign panic: {}", msg));
            Value::null()
        }
    };
    unsafe {
        if let Some(frame) = (*fiber).mir_frames.last_mut() {
            *values = std::mem::take(&mut frame.values);
        }
    }
    result
}

fn call_foreign_c_with_frame_sync(
    vm: &mut VM,
    fiber: *mut ObjFiber,
    pc: u32,
    values: &mut Vec<Value>,
    func: crate::runtime::object::ForeignCFn,
    args: &[Value],
) -> Value {
    unsafe {
        if let Some(frame) = (*fiber).mir_frames.last_mut() {
            frame.values = std::mem::take(values);
            frame.pc = pc;
        }
    }

    // Foreign code is even less trustworthy than first-party
    // natives — the same panic protection applies.
    let result = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        crate::runtime::foreign::dispatch_foreign_c(vm, func, args)
    })) {
        Ok(v) => v,
        Err(panic) => {
            let msg = panic_message(&panic);
            vm.has_error = true;
            vm.last_error = Some(format!("foreign panic: {}", msg));
            Value::null()
        }
    };

    unsafe {
        if let Some(frame) = (*fiber).mir_frames.last_mut() {
            *values = std::mem::take(&mut frame.values);
        }
    }

    result
}

fn try_run_root_frame_native(
    vm: &mut VM,
    fiber: *mut ObjFiber,
    func_id: FuncId,
    module_name: &Rc<String>,
) -> Result<Option<Value>, RuntimeError> {
    #[inline(always)]
    fn trace_root_native(msg: impl FnOnce() -> String) {
        if std::env::var_os("WLIFT_TRACE_ROOT_NATIVE").is_some() {
            eprintln!("{}", msg());
        }
    }

    if vm.engine.mode == ExecutionMode::Interpreter || crate::codegen::runtime_fns::jit_disabled() {
        return Ok(None);
    }

    let fn_idx = func_id.0 as usize;
    let native_fn_ptr = vm
        .engine
        .jit_code
        .get(fn_idx)
        .copied()
        .unwrap_or(std::ptr::null());
    if native_fn_ptr.is_null() {
        return Ok(None);
    }

    let is_leaf = vm.engine.jit_leaf.get(fn_idx).copied().unwrap_or(false);
    let allow_nonleaf_native = crate::codegen::runtime_fns::allow_root_nonleaf_native(vm, func_id);
    if !is_leaf && !allow_nonleaf_native {
        return Ok(None);
    }

    let saved_ctx = crate::codegen::runtime_fns::read_jit_ctx();
    if !saved_ctx.vm.is_null() {
        return Ok(None);
    }

    let jit_depth = crate::codegen::runtime_fns::jit_depth();
    if jit_depth >= crate::codegen::runtime_fns::MAX_JIT_DEPTH {
        return Ok(None);
    }

    let vm_ptr = vm as *mut VM as *mut u8;
    let mod_name_bytes = module_name.as_bytes();
    let (mv_ptr, mv_count) = vm
        .engine
        .modules
        .get(module_name.as_str())
        .map(|m| (m.vars.as_ptr() as *mut u64, m.vars.len() as u32))
        .unwrap_or((std::ptr::null_mut(), 0));
    let fiber_root_idx = crate::codegen::runtime_fns::jit_roots_snapshot_len();
    crate::codegen::runtime_fns::push_jit_root(Value::object(fiber as *mut u8));
    crate::codegen::runtime_fns::set_jit_context(crate::codegen::runtime_fns::JitContext {
        module_vars: mv_ptr,
        module_var_count: mv_count,
        vm: vm_ptr,
        module_name: mod_name_bytes.as_ptr(),
        module_name_len: mod_name_bytes.len() as u32,
        current_func_id: func_id.0 as u64,
        closure: std::ptr::null_mut(),
        defining_class: std::ptr::null_mut(),
        jit_code_base: vm.engine.jit_code.as_ptr(),
        jit_code_len: vm.engine.jit_code.len() as u32,
    });

    vm.engine.note_native_entry(func_id);
    if std::env::var_os("WLIFT_TRACE_NATIVE_ENTRY").is_some() {
        let name = vm
            .engine
            .get_mir(func_id)
            .map(|mir| vm.interner.resolve(mir.name).to_string())
            .unwrap_or_else(|| "<unknown>".to_string());
        eprintln!("native-entry: root FuncId({}) {}", func_id.0, name);
    }
    crate::codegen::runtime_fns::set_jit_depth(jit_depth + 1);
    let result_bits = unsafe { call_jit_fn(native_fn_ptr, &[]) };
    crate::codegen::runtime_fns::set_jit_depth(jit_depth);

    let live_fiber = if crate::codegen::runtime_fns::jit_roots_snapshot_len() > fiber_root_idx {
        crate::codegen::runtime_fns::jit_root_at(fiber_root_idx)
            .as_object()
            .map(|p| p as *mut ObjFiber)
            .unwrap_or(fiber)
    } else {
        fiber
    };
    crate::codegen::runtime_fns::jit_roots_restore_len(fiber_root_idx);
    trace_root_native(|| {
        format!(
            "root-native: return func={} fiber={:p} live_fiber={:p} vm_fiber_before={:p}",
            func_id.0, fiber, live_fiber, vm.fiber
        )
    });
    vm.fiber = live_fiber;
    crate::codegen::runtime_fns::set_jit_context(saved_ctx);

    if vm.has_error {
        vm.has_error = false;
        let err = vm
            .last_error
            .take()
            .unwrap_or_else(|| "runtime error in native entry".to_string());
        return Err(RuntimeError::Error(err));
    }

    Ok(Some(Value::from_bits(result_bits)))
}

enum OsrTransfer {
    NotEntered,
    ContinueFiberLoop,
    Return(Value),
}

#[allow(clippy::too_many_arguments)] // OSR transfer needs every piece of the live frame context.
fn try_enter_loop_osr(
    vm: &mut VM,
    fiber: *mut ObjFiber,
    func_id: FuncId,
    module_name: &Rc<String>,
    closure: Option<*mut ObjClosure>,
    defining_class: Option<*mut ObjClass>,
    return_dst: Option<ValueId>,
    bc: &BytecodeFunction,
    branch_offset: u32,
    target_offset: u32,
    values: &mut Vec<Value>,
    stop_depth: Option<usize>,
) -> Result<OsrTransfer, RuntimeError> {
    if vm.engine.mode != ExecutionMode::Tiered || crate::codegen::runtime_fns::jit_disabled() {
        return Ok(OsrTransfer::NotEntered);
    }
    // Phase 4: allow method/closure OSR when an env-gated kill switch is not
    // set. Keep a fast opt-out so production deployments can disable this path
    // if a regression surfaces before we build GC/deopt coverage.
    if (closure.is_some() || defining_class.is_some())
        && std::env::var_os("WLIFT_DISABLE_METHOD_OSR").is_some()
    {
        return Ok(OsrTransfer::NotEntered);
    }

    let Some(point) = bc
        .osr_points
        .iter()
        .find(|point| point.branch_offset == branch_offset && point.target_offset == target_offset)
    else {
        return Ok(OsrTransfer::NotEntered);
    };
    if point.param_regs.len() > 4 {
        return Ok(OsrTransfer::NotEntered);
    }

    let mut osr_args = SmallVec::<[Value; 4]>::new();
    for &reg in &point.param_regs {
        let Some(value) = values.get(reg as usize).copied() else {
            return Ok(OsrTransfer::NotEntered);
        };
        if value.is_undefined() {
            return Ok(OsrTransfer::NotEntered);
        }
        osr_args.push(value);
    }

    let Some(entry) = vm
        .engine
        .active_osr_entry(func_id, point.target_block, osr_args.len())
    else {
        return Ok(OsrTransfer::NotEntered);
    };

    let jit_depth = crate::codegen::runtime_fns::jit_depth();
    if jit_depth >= crate::codegen::runtime_fns::MAX_JIT_DEPTH {
        return Ok(OsrTransfer::NotEntered);
    }
    // Continuation-safe OSR: nested transfers are allowed (jit_depth > 0)
    // because the save/restore discipline below mirrors regular native-from-
    // native dispatch — JIT context, roots, and depth are all snapshotted
    // before the call and restored after. A kill switch keeps a quick opt-
    // out while we finish deopt/fiber-action coverage.
    if jit_depth > 0 && std::env::var_os("WLIFT_DISABLE_NESTED_OSR").is_some() {
        return Ok(OsrTransfer::NotEntered);
    }

    unsafe {
        if let Some(frame) = (*fiber).mir_frames.last_mut() {
            frame.pc = target_offset;
            frame.values = std::mem::take(values);
            frame.bc_ptr = bc as *const BytecodeFunction;
        }
    }

    let saved_ctx = crate::codegen::runtime_fns::read_jit_ctx();
    let saved_ctx_root_idx = crate::codegen::runtime_fns::jit_roots_snapshot_len();
    crate::codegen::runtime_fns::push_jit_root(if saved_ctx.closure.is_null() {
        Value::null()
    } else {
        Value::object(saved_ctx.closure)
    });
    crate::codegen::runtime_fns::push_jit_root(if saved_ctx.defining_class.is_null() {
        Value::null()
    } else {
        Value::object(saved_ctx.defining_class)
    });
    let current_ctx_root_idx = crate::codegen::runtime_fns::jit_roots_snapshot_len();
    crate::codegen::runtime_fns::push_jit_root(
        closure
            .map(|ptr| Value::object(ptr as *mut u8))
            .unwrap_or_else(Value::null),
    );
    crate::codegen::runtime_fns::push_jit_root(
        defining_class
            .map(|ptr| Value::object(ptr as *mut u8))
            .unwrap_or_else(Value::null),
    );
    let fiber_root_idx = crate::codegen::runtime_fns::jit_roots_snapshot_len();
    crate::codegen::runtime_fns::push_jit_root(Value::object(fiber as *mut u8));

    let vm_ptr = vm as *mut VM as *mut u8;
    let mod_name_bytes = module_name.as_bytes();
    let (mv_ptr, mv_count) = vm
        .engine
        .modules
        .get(module_name.as_str())
        .map(|m| (m.vars.as_ptr() as *mut u64, m.vars.len() as u32))
        .unwrap_or((std::ptr::null_mut(), 0));
    crate::codegen::runtime_fns::set_jit_context(crate::codegen::runtime_fns::JitContext {
        module_vars: mv_ptr,
        module_var_count: mv_count,
        vm: vm_ptr,
        module_name: mod_name_bytes.as_ptr(),
        module_name_len: mod_name_bytes.len() as u32,
        current_func_id: func_id.0 as u64,
        closure: crate::codegen::runtime_fns::jit_root_at(current_ctx_root_idx)
            .as_object()
            .unwrap_or(std::ptr::null_mut()),
        defining_class: crate::codegen::runtime_fns::jit_root_at(current_ctx_root_idx + 1)
            .as_object()
            .unwrap_or(std::ptr::null_mut()),
        jit_code_base: vm.engine.jit_code.as_ptr(),
        jit_code_len: vm.engine.jit_code.len() as u32,
    });

    if std::env::var_os("WLIFT_OSR_TRACE").is_some() {
        let name = vm
            .engine
            .get_mir(func_id)
            .map(|mir| vm.interner.resolve(mir.name).to_string())
            .unwrap_or_else(|| "<unknown>".to_string());
        eprintln!(
            "osr-trace: enter FuncId({}) {} bb{} argc={}",
            func_id.0,
            name,
            point.target_block.0,
            osr_args.len()
        );
    }

    vm.engine.note_native_entry(func_id);
    vm.engine.note_osr_entry(func_id);
    crate::codegen::runtime_fns::set_jit_depth(jit_depth + 1);
    let result_bits = unsafe { call_jit_fn(entry.ptr, &osr_args) };
    crate::codegen::runtime_fns::set_jit_depth(jit_depth);

    let live_fiber = if crate::codegen::runtime_fns::jit_roots_snapshot_len() > fiber_root_idx {
        crate::codegen::runtime_fns::jit_root_at(fiber_root_idx)
            .as_object()
            .map(|p| p as *mut ObjFiber)
            .unwrap_or(fiber)
    } else {
        fiber
    };
    vm.fiber = live_fiber;

    let mut restored_ctx = saved_ctx;
    restored_ctx.closure = crate::codegen::runtime_fns::jit_root_at(saved_ctx_root_idx)
        .as_object()
        .unwrap_or(std::ptr::null_mut());
    restored_ctx.defining_class = crate::codegen::runtime_fns::jit_root_at(saved_ctx_root_idx + 1)
        .as_object()
        .unwrap_or(std::ptr::null_mut());
    crate::codegen::runtime_fns::jit_roots_restore_len(saved_ctx_root_idx);
    crate::codegen::runtime_fns::set_jit_context(restored_ctx);

    if vm.has_error {
        vm.has_error = false;
        let err = vm
            .last_error
            .take()
            .unwrap_or_else(|| "runtime error in OSR entry".to_string());
        return Err(RuntimeError::Error(err));
    }

    // Deopt: fiber actions (yield / transfer / suspend) triggered inside the
    // OSR'd native loop can't resume at an arbitrary iteration once the
    // frame has been popped. Until we have a proper interpreter-resume
    // continuation for these cases, surface a clear runtime error instead
    // of silently continuing past the action.
    if vm.pending_fiber_action.is_some() {
        vm.pending_fiber_action = None;
        return Err(RuntimeError::Unsupported(
            "fiber yield/transfer from inside a JIT-OSR loop is not yet supported".into(),
        ));
    }

    let result = Value::from_bits(result_bits);
    let frame_values = unsafe {
        (*live_fiber)
            .mir_frames
            .pop()
            .map(|frame| frame.values)
            .unwrap_or_default()
    };
    if vm.register_pool.len() < 128 {
        vm.register_pool.push(frame_values);
    }

    let remaining_depth = unsafe { (*live_fiber).mir_frames.len() };
    if stop_depth == Some(remaining_depth) {
        return Ok(OsrTransfer::Return(result));
    }
    if remaining_depth == 0 {
        unsafe {
            (*live_fiber).state = FiberState::Done;
        }
        let caller = unsafe { (*live_fiber).caller };
        if !caller.is_null() {
            unsafe {
                (*live_fiber).caller = std::ptr::null_mut();
            }
            resume_caller(vm, caller, result);
            return Ok(OsrTransfer::ContinueFiberLoop);
        }
        return Ok(OsrTransfer::Return(result));
    }

    if let Some(dst) = return_dst {
        unsafe {
            if let Some(caller_frame) = (*live_fiber).mir_frames.last_mut() {
                set_reg(&mut caller_frame.values, dst.0 as u16, result);
            }
        }
    }
    Ok(OsrTransfer::ContinueFiberLoop)
}

// ---------------------------------------------------------------------------
// Runtime errors
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum RuntimeError {
    MethodNotFound { class_name: String, method: String },
    GuardFailed(&'static str),
    Unsupported(String),
    StepLimitExceeded,
    StackOverflow,
    Error(String),
    Unreachable,
}

/// Source location context attached when reporting runtime errors.
#[derive(Debug, Clone)]
pub struct SourceLoc {
    pub span: crate::ast::Span,
    pub module: Rc<String>,
}

impl std::fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RuntimeError::MethodNotFound { class_name, method } => {
                write!(f, "{} does not implement '{}'", class_name, method)
            }
            RuntimeError::GuardFailed(ty) => write!(f, "type guard failed: expected {}", ty),
            RuntimeError::Unsupported(msg) => write!(f, "unsupported: {}", msg),
            RuntimeError::StepLimitExceeded => write!(f, "step limit exceeded"),
            RuntimeError::StackOverflow => write!(f, "stack overflow"),
            RuntimeError::Error(msg) => write!(f, "{}", msg),
            RuntimeError::Unreachable => write!(f, "reached unreachable code"),
        }
    }
}

impl From<InterpError> for RuntimeError {
    fn from(e: InterpError) -> Self {
        match e {
            InterpError::Unsupported(msg) => RuntimeError::Unsupported(msg),
            InterpError::StepLimitExceeded => RuntimeError::StepLimitExceeded,
            InterpError::Unreachable => RuntimeError::Unreachable,
            other => RuntimeError::Error(other.to_string()),
        }
    }
}

// ---------------------------------------------------------------------------
// Inline register helpers
// ---------------------------------------------------------------------------

#[inline(always)]
fn get_reg(values: &[Value], idx: u16) -> Value {
    unsafe { *values.get_unchecked(idx as usize) }
}

#[inline(always)]
fn set_reg(values: &mut Vec<Value>, idx: u16, val: Value) {
    let i = idx as usize;
    if i < values.len() {
        unsafe {
            *values.get_unchecked_mut(i) = val;
        }
    } else {
        values.resize(i + 1, UNDEF);
        values[i] = val;
    }
}

/// Read a u16 at a fixed bytecode offset without advancing pc.
#[inline(always)]
fn read_u16_at(code: &[u8], offset: u32) -> u16 {
    let i = offset as usize;
    unsafe {
        let lo = *code.get_unchecked(i);
        let hi = *code.get_unchecked(i + 1);
        u16::from_le_bytes([lo, hi])
    }
}

/// Build a SmallVec of arg values from bytecode arg register indices.
/// Used by the slow path after the fast path declines.
#[inline]
fn build_arg_vals(
    recv: Value,
    code: &[u8],
    arg_regs_pc: u32,
    argc: usize,
    values: &[Value],
) -> SmallVec<[Value; 8]> {
    let mut arg_vals: SmallVec<[Value; 8]> = SmallVec::with_capacity(1 + argc);
    arg_vals.push(recv);
    let mut apc = arg_regs_pc;
    for _ in 0..argc {
        let arg_reg = read_u16(code, &mut apc);
        arg_vals.push(get_reg(values, arg_reg));
    }
    arg_vals
}

// ---------------------------------------------------------------------------
// Macros for repetitive opcode patterns (must precede run_fiber)
// ---------------------------------------------------------------------------

/// Boxed binary arithmetic: try numeric, fall back to operator dispatch.
/// Returns true if a closure frame was pushed (caller should continue 'fiber_loop).
macro_rules! bc_boxed_binop {
    ($vm:expr, $fiber:expr, $pc:expr, $values:expr, $module_name:expr, $bc_ptr:expr, $op:expr, $method:expr) => {{
        let code = &unsafe { &*$bc_ptr }.code;
        let dst = read_u16(code, $pc);
        let lhs_reg = read_u16(code, $pc);
        let rhs_reg = read_u16(code, $pc);
        let a = get_reg($values, lhs_reg);
        let b = get_reg($values, rhs_reg);
        match (a.as_num(), b.as_num()) {
            (Some(x), Some(y)) => {
                let f: fn(f64, f64) -> f64 = $op;
                set_reg($values, dst, Value::num(f(x, y)));
                false
            }
            _ => {
                let recv = a;
                let arg = b;
                try_operator_dispatch(
                    $vm,
                    $fiber,
                    recv,
                    $method,
                    &[arg],
                    $pc,
                    $values,
                    dst,
                    $module_name,
                    $bc_ptr,
                )?
            }
        }
    }};
}

/// Unboxed f64 binary op.
macro_rules! bc_f64_binop {
    ($pc:expr, $values:expr, $op:expr, $bc:expr) => {{
        let code = &$bc.code;
        let dst = read_u16(code, $pc);
        let lhs_reg = read_u16(code, $pc);
        let rhs_reg = read_u16(code, $pc);
        let a = get_reg($values, lhs_reg)
            .as_num()
            .ok_or(RuntimeError::GuardFailed("num"))?;
        let b = get_reg($values, rhs_reg)
            .as_num()
            .ok_or(RuntimeError::GuardFailed("num"))?;
        let f: fn(f64, f64) -> f64 = $op;
        set_reg($values, dst, Value::num(f(a, b)));
    }};
}

/// Boxed comparison: try numeric, fall back to operator dispatch.
/// Returns true if a closure frame was pushed.
macro_rules! bc_boxed_cmp {
    ($vm:expr, $fiber:expr, $pc:expr, $values:expr, $module_name:expr, $bc_ptr:expr, $op:expr, $method:expr) => {{
        let code = &unsafe { &*$bc_ptr }.code;
        let dst = read_u16(code, $pc);
        let lhs_reg = read_u16(code, $pc);
        let rhs_reg = read_u16(code, $pc);
        let a = get_reg($values, lhs_reg);
        let b = get_reg($values, rhs_reg);
        match (a.as_num(), b.as_num()) {
            (Some(x), Some(y)) => {
                let f: fn(f64, f64) -> bool = $op;
                set_reg($values, dst, Value::bool(f(x, y)));
                false
            }
            _ => {
                let recv = a;
                let arg = b;
                try_operator_dispatch(
                    $vm,
                    $fiber,
                    recv,
                    $method,
                    &[arg],
                    $pc,
                    $values,
                    dst,
                    $module_name,
                    $bc_ptr,
                )?
            }
        }
    }};
}

/// Unboxed f64 comparison.
macro_rules! bc_f64_cmp {
    ($pc:expr, $values:expr, $op:expr, $bc:expr) => {{
        let code = &$bc.code;
        let dst = read_u16(code, $pc);
        let lhs_reg = read_u16(code, $pc);
        let rhs_reg = read_u16(code, $pc);
        let a = get_reg($values, lhs_reg)
            .as_num()
            .ok_or(RuntimeError::GuardFailed("num"))?;
        let b = get_reg($values, rhs_reg)
            .as_num()
            .ok_or(RuntimeError::GuardFailed("num"))?;
        let f: fn(f64, f64) -> bool = $op;
        set_reg($values, dst, Value::bool(f(a, b)));
    }};
}

/// Bitwise binary op: try numeric, fall back to operator dispatch.
/// Returns true if a closure frame was pushed.
macro_rules! bc_bitwise_binop {
    ($vm:expr, $fiber:expr, $pc:expr, $values:expr, $module_name:expr, $bc_ptr:expr, $op:expr, $method:expr) => {{
        let code = &unsafe { &*$bc_ptr }.code;
        let dst = read_u16(code, $pc);
        let lhs_reg = read_u16(code, $pc);
        let rhs_reg = read_u16(code, $pc);
        let a = get_reg($values, lhs_reg);
        let b = get_reg($values, rhs_reg);
        match (a.as_num(), b.as_num()) {
            (Some(x), Some(y)) => {
                let f: fn(i32, i32) -> i32 = $op;
                set_reg($values, dst, Value::num(f(x as i32, y as i32) as f64));
                false
            }
            _ => {
                let recv = a;
                let arg = b;
                try_operator_dispatch(
                    $vm,
                    $fiber,
                    recv,
                    $method,
                    &[arg],
                    $pc,
                    $values,
                    dst,
                    $module_name,
                    $bc_ptr,
                )?
            }
        }
    }};
}

// ---------------------------------------------------------------------------
// Fiber-based bytecode interpreter
// ---------------------------------------------------------------------------

fn run_fiber_with_stop_depth(
    vm: &mut VM,
    stop_depth: Option<usize>,
) -> Result<Value, RuntimeError> {
    if vm.fiber.is_null() {
        return Err(RuntimeError::Error("no active fiber".into()));
    }

    // `stop_depth` is a frame count on the fiber that was active when
    // this run loop was entered — typically a native-to-Wren bridge
    // (`vm.call_fn_native`, constructor sync path) that pushes a frame
    // and waits for it to unwind. If a `Fiber.try` / `Fiber.abort`
    // switches fibers mid-run, frame counts on the new fiber are
    // unrelated, and comparing `stop_depth` against them would
    // prematurely bail out — returning the wrong value to the bridge.
    // Remember the starting fiber and only honour `stop_depth` on it.
    let stop_fiber = vm.fiber;

    let mut steps: usize = 0;
    // Back-edge counter: sampled for tier-up polling on hot loops.
    let mut backedge_counter: u32 = 0;

    // Outer loop: re-entered when we push/pop a call frame or switch fibers
    'fiber_loop: loop {
        if vm.fiber.is_null() {
            // A no-caller yield cleared `vm.fiber` to hand
            // control back to the host scheduler. The fiber
            // itself is still alive in the host's parked list
            // and will be resumed via a future fresh
            // `run_fiber` call once its awaited future settles.
            return Ok(Value::null());
        }
        let mut fiber = vm.fiber;
        unsafe {
            (*fiber).state = FiberState::Running;
        }

        let frame_count = unsafe { (*fiber).mir_frames.len() };
        if frame_count == 0 {
            // Check if this fiber was resumed with a value from a child fiber
            // (JIT barrier path: frames were temporarily removed).
            let resume_val = unsafe { (*fiber).jit_resume_value.take() };
            unsafe {
                (*fiber).state = FiberState::Done;
            }
            return Ok(resume_val.unwrap_or(Value::null()));
        }

        // Load execution state from the top frame into locals. Inline
        // call/return sites re-enter `'fiber_loop` rather than mutating
        // these in place so `module_name` (and the `module_vars_ptr`
        // cache derived from it) always match the active frame —
        // crucial when the callee lives in a different module than the
        // caller.
        let (func_id, mut pc, mut values, module_name, closure, _defining_class, return_dst) = unsafe {
            let frame = (*fiber).mir_frames.last_mut().unwrap();
            (
                frame.func_id,
                frame.pc,
                std::mem::take(&mut frame.values),
                frame.module_name.clone(),
                frame.closure,
                frame.defining_class,
                frame.return_dst,
            )
        };

        if pc == 0 && closure.is_none() && return_dst.is_none() {
            if let Some(return_val) = try_run_root_frame_native(vm, fiber, func_id, &module_name)? {
                fiber = vm.fiber;
                if std::env::var_os("WLIFT_TRACE_ROOT_NATIVE").is_some() {
                    let frame_count = if fiber.is_null() {
                        0
                    } else {
                        unsafe { (*fiber).mir_frames.len() }
                    };
                    eprintln!(
                        "root-native: handoff func={} fiber={:p} vm_fiber={:p} frames={} caller={:p}",
                        func_id.0,
                        fiber,
                        vm.fiber,
                        frame_count,
                        if fiber.is_null() {
                            std::ptr::null_mut()
                        } else {
                            unsafe { (*fiber).caller }
                        }
                    );
                }
                unsafe {
                    (*fiber).mir_frames.pop();
                    (*fiber).state = FiberState::Done;
                }
                values.clear();
                if vm.register_pool.len() < 128 {
                    vm.register_pool.push(values);
                }
                let caller = unsafe { (*fiber).caller };
                if !caller.is_null() {
                    unsafe {
                        (*fiber).caller = std::ptr::null_mut();
                    }
                    resume_caller(vm, caller, return_val);
                    continue 'fiber_loop;
                }
                return Ok(return_val);
            }
        }

        // Get bytecode (lazily compiled on first use).
        // Use raw pointer to avoid Arc clone overhead in the hot loop.
        let mut bc_ptr = vm
            .engine
            .ensure_bytecode(func_id)
            .ok_or_else(|| RuntimeError::Error("invalid func_id".into()))?;
        // SAFETY: bc_ptr is stable — bytecode is never freed once compiled.
        let mut bc = unsafe { &*bc_ptr };
        let mut code: &[u8] = &bc.code;

        // Cache raw pointer to module vars — eliminates HashMap lookup per
        // GetModuleVar/SetModuleVar (was 42% of runtime in method_call).
        // Recomputed on each `'fiber_loop` iteration so inline
        // call/return into a different module picks up the right
        // slot table.
        let module_vars_ptr: *mut Vec<Value> = vm
            .engine
            .modules
            .get_mut(module_name.as_str())
            .map(|m| &mut m.vars as *mut Vec<Value>)
            .unwrap_or(std::ptr::null_mut());

        // Set JIT context for the current frame so JIT-compiled
        // functions dispatched from the IC fast path see the active
        // frame's module vars, func id, etc. Refreshed on every
        // `'fiber_loop` iteration rather than once per fiber —
        // a fiber switch (from `fib.try()`, `Fiber.abort`, etc.)
        // re-enters the loop with the same non-null `ctx.vm`, but
        // `module_vars` / `current_func_id` now belong to the
        // previous fiber. Skipping the refresh made the first call
        // after a cross-fiber abort read stale pointers and return
        // UNDEFINED.
        {
            let (mv_ptr, mv_count) = if !module_vars_ptr.is_null() {
                let v = unsafe { &*module_vars_ptr };
                (v.as_ptr() as *mut u64, v.len() as u32)
            } else {
                (std::ptr::null_mut(), 0)
            };
            let mod_name_bytes = module_name.as_bytes();
            crate::codegen::runtime_fns::mutate_jit_ctx(|ctx| {
                ctx.vm = vm as *mut _ as *mut u8;
                ctx.module_vars = mv_ptr;
                ctx.module_var_count = mv_count;
                ctx.module_name = mod_name_bytes.as_ptr();
                ctx.module_name_len = mod_name_bytes.len() as u32;
                ctx.current_func_id = func_id.0 as u64;
                ctx.jit_code_base = vm.engine.jit_code.as_ptr();
                ctx.jit_code_len = vm.engine.jit_code.len() as u32;
            });
        }

        // Root-frame threaded dispatch — disabled pending proper
        // frame lifecycle handling (popping root frame + fiber_loop
        // re-entry causes infinite null output). Wasm builds gate
        // out the threaded module entirely (it relies on the JIT
        // runtime helpers); the `cfg` keeps the disabled-by-default
        // block from breaking the wasm type check.
        #[cfg(feature = "host")]
        if false {
            let fn_idx_tc = func_id.0 as usize;
            let has_jit = !vm
                .engine
                .jit_code
                .get(fn_idx_tc)
                .copied()
                .unwrap_or(std::ptr::null())
                .is_null();
            if pc == 0
                && !has_jit
                && vm.engine.mode != ExecutionMode::Interpreter
                && crate::mir::threaded::threaded_depth_ok()
                && crate::mir::threaded::threaded_enabled()
            {
                let _ = vm.engine.ensure_threaded_code(func_id, &vm.interner);
                let has_tc = vm
                    .engine
                    .threaded_code
                    .get(fn_idx_tc)
                    .and_then(|t| t.as_ref())
                    .and_then(|t| t.as_ref())
                    .is_some();
                if has_tc {
                    let (mv_ptr, mv_count) = if !module_vars_ptr.is_null() {
                        let v = unsafe { &*module_vars_ptr };
                        (v.as_ptr() as *mut u64, v.len() as u32)
                    } else {
                        (std::ptr::null_mut(), 0)
                    };
                    let vm_ptr = vm as *mut VM as *mut u8;
                    let mod_name_bytes = module_name.as_bytes();
                    crate::codegen::runtime_fns::mutate_jit_ctx(|ctx| {
                        ctx.vm = vm_ptr;
                        ctx.module_vars = mv_ptr;
                        ctx.module_var_count = mv_count;
                        ctx.module_name = mod_name_bytes.as_ptr();
                        ctx.module_name_len = mod_name_bytes.len() as u32;
                        ctx.current_func_id = func_id.0 as u64;
                        ctx.jit_code_base = vm.engine.jit_code.as_ptr();
                        ctx.jit_code_len = vm.engine.jit_code.len() as u32;
                    });
                    let tc = vm.engine.threaded_code[fn_idx_tc]
                        .as_ref()
                        .unwrap()
                        .as_ref()
                        .unwrap();
                    let recycled = vm.register_pool.pop();
                    let (result, regs_back) = crate::mir::threaded::execute_threaded(
                        tc, &values, mv_ptr, mv_count, vm_ptr, None, recycled,
                    );
                    if vm.register_pool.len() < 128 {
                        vm.register_pool.push(regs_back);
                    }
                    unsafe {
                        (*fiber).mir_frames.pop();
                    }
                    if let Some(rdst) = return_dst {
                        unsafe {
                            if let Some(caller_frame) = (*fiber).mir_frames.last_mut() {
                                caller_frame.values[rdst.0 as usize] = result;
                            }
                        }
                    }
                    values.clear();
                    if vm.register_pool.len() < 128 {
                        vm.register_pool.push(values);
                    }
                    continue 'fiber_loop;
                }
            }
        }

        // Pre-compute safepoint interval (config doesn't change mid-loop).
        // step_limit == 0 means "unlimited" — common for long-running
        // servers and event loops. Use a large interval so the per-tick
        // safepoint cost stays negligible while still hitting GC + bg
        // compile polls reasonably often.
        let unlimited_steps = vm.config.step_limit == 0;
        let check_interval: usize = if unlimited_steps {
            0xFFF
        } else if vm.config.step_limit <= 0xFF {
            0xF
        } else if vm.config.step_limit <= 0xFFF {
            0xFF
        } else {
            0xFFF
        };

        // Inner dispatch loop: decodes opcodes from the flat bytecode stream.
        loop {
            if steps & check_interval == 0 {
                if !unlimited_steps && steps > vm.config.step_limit {
                    return Err(RuntimeError::StepLimitExceeded);
                }
                if vm.gc.should_collect() || vm.gc_requested {
                    vm.gc_requested = false;
                    unsafe {
                        if let Some(frame) = (*fiber).mir_frames.last_mut() {
                            frame.values = std::mem::take(&mut values);
                            frame.pc = pc;
                        }
                    }
                    vm.collect_garbage();
                    vm.method_cache.invalidate();
                    vm.engine.invalidate_inline_caches();
                    vm.engine.poll_compilations();
                    vm.engine.drain_compile_queue(&vm.interner);
                    fiber = vm.fiber;
                    unsafe {
                        if let Some(frame) = (*fiber).mir_frames.last_mut() {
                            values = std::mem::take(&mut frame.values);
                        }
                    }
                }
                // SIGUSR1-driven hot reload — cheap atomic load when
                // no signal is pending. Snapshots the current frame
                // around the reload because reload_module re-enters
                // the interpreter to run the new module's top-level.
                if super::vm::reload_pending() {
                    unsafe {
                        if let Some(frame) = (*fiber).mir_frames.last_mut() {
                            frame.values = std::mem::take(&mut values);
                            frame.pc = pc;
                        }
                    }
                    vm.check_pending_reload();
                    fiber = vm.fiber;
                    unsafe {
                        if let Some(frame) = (*fiber).mir_frames.last_mut() {
                            values = std::mem::take(&mut frame.values);
                        }
                    }
                }
            }
            steps += 1;

            let func_name = vm
                .engine
                .get_mir(func_id)
                .map(|m| vm.interner.resolve(m.name).to_string())
                .unwrap_or_else(|| "<unknown>".to_string());
            debug_assert!(
                (pc as usize) < code.len(),
                "pc {} out of bounds for code len {} in func {:?} ({}) module {}",
                pc,
                code.len(),
                func_id,
                func_name,
                module_name
            );
            let op = unsafe { *code.get_unchecked(pc as usize) };
            pc += 1;

            // SAFETY: op values come from our encoder, all valid Op discriminants
            match unsafe { Op::from_u8_unchecked(op) } {
                // =============================================================
                // Constants (3B-5B)
                // =============================================================
                Op::ConstNull => {
                    let dst = read_u16(code, &mut pc);
                    set_reg(&mut values, dst, Value::null());
                }
                Op::ConstTrue => {
                    let dst = read_u16(code, &mut pc);
                    set_reg(&mut values, dst, Value::bool(true));
                }
                Op::ConstFalse => {
                    let dst = read_u16(code, &mut pc);
                    set_reg(&mut values, dst, Value::bool(false));
                }
                Op::ConstNum => {
                    let dst = read_u16(code, &mut pc);
                    let idx = read_u16(code, &mut pc);
                    let val = match bc.constants[idx as usize] {
                        BcConst::F64(n) => Value::num(n),
                        BcConst::I64(n) => Value::num(n as f64),
                    };
                    set_reg(&mut values, dst, val);
                }
                Op::ConstF64 => {
                    let dst = read_u16(code, &mut pc);
                    let idx = read_u16(code, &mut pc);
                    let val = match bc.constants[idx as usize] {
                        BcConst::F64(n) => Value::num(n),
                        BcConst::I64(n) => Value::num(n as f64),
                    };
                    set_reg(&mut values, dst, val);
                }
                Op::ConstI64 => {
                    let dst = read_u16(code, &mut pc);
                    let idx = read_u16(code, &mut pc);
                    let val = match bc.constants[idx as usize] {
                        BcConst::I64(n) => Value::num(n as f64),
                        BcConst::F64(n) => Value::num(n),
                    };
                    set_reg(&mut values, dst, val);
                }
                Op::ConstString => {
                    let dst = read_u16(code, &mut pc);
                    let sym_idx = read_u16(code, &mut pc);
                    let sym = SymbolId::from_raw(sym_idx as u32);
                    let s = vm.interner.resolve(sym).to_string();
                    set_reg(&mut values, dst, vm.new_string(s));
                }

                // =============================================================
                // Module variables
                // =============================================================
                Op::GetModuleVar => {
                    let dst = read_u16(code, &mut pc);
                    let slot = read_u16(code, &mut pc) as usize;
                    let val = if !module_vars_ptr.is_null() {
                        let vars = unsafe { &*module_vars_ptr };
                        vars.get(slot).copied().unwrap_or(Value::null())
                    } else {
                        Value::null()
                    };
                    set_reg(&mut values, dst, val);
                }
                Op::SetModuleVar => {
                    let dst = read_u16(code, &mut pc);
                    let val_reg = read_u16(code, &mut pc);
                    let slot = read_u16(code, &mut pc) as usize;
                    let v = get_reg(&values, val_reg);
                    if !module_vars_ptr.is_null() {
                        let vars = unsafe { &mut *module_vars_ptr };
                        while vars.len() <= slot {
                            vars.push(Value::null());
                        }
                        vars[slot] = v;
                    }
                    set_reg(&mut values, dst, v);
                }

                // =============================================================
                // Upvalues
                // =============================================================
                Op::GetUpvalue => {
                    let dst = read_u16(code, &mut pc);
                    let idx = read_u16(code, &mut pc) as usize;
                    let frame_closure = unsafe { (*fiber).mir_frames.last().unwrap().closure };
                    let val = if let Some(closure_ptr) = frame_closure {
                        let upvalues = unsafe { &(*closure_ptr).upvalues };
                        if idx < upvalues.len() {
                            unsafe { (*upvalues[idx]).get() }
                        } else {
                            Value::null()
                        }
                    } else {
                        Value::null()
                    };
                    set_reg(&mut values, dst, val);
                }
                Op::SetUpvalue => {
                    let dst = read_u16(code, &mut pc);
                    let val_reg = read_u16(code, &mut pc);
                    let idx = read_u16(code, &mut pc) as usize;
                    let v = get_reg(&values, val_reg);
                    let frame_closure = unsafe { (*fiber).mir_frames.last().unwrap().closure };
                    if let Some(closure_ptr) = frame_closure {
                        let upvalues = unsafe { &mut (*closure_ptr).upvalues };
                        if idx < upvalues.len() {
                            let uv_ptr = upvalues[idx];
                            unsafe { (*uv_ptr).set(v) };
                            vm.gc.write_barrier(uv_ptr as *mut ObjHeader, v);
                        }
                    }
                    set_reg(&mut values, dst, v);
                }

                // =============================================================
                // Block parameters (pre-bound by branch; no-op if already set)
                // =============================================================
                Op::BlockParam => {
                    let dst = read_u16(code, &mut pc);
                    let _param_idx = read_u16(code, &mut pc);
                    // Already bound by branch instruction, skip if set
                    let idx = dst as usize;
                    if idx < values.len() && !matches!(values[idx], v if v.is_undefined()) {
                        continue;
                    }
                    set_reg(&mut values, dst, Value::null());
                }

                // =============================================================
                // Static fields
                // =============================================================
                Op::GetStaticField => {
                    let dst = read_u16(code, &mut pc);
                    let sym_idx = read_u16(code, &mut pc);
                    let sym = SymbolId::from_raw(sym_idx as u32);
                    let frame = unsafe { (*fiber).mir_frames.last().unwrap() };
                    let val = if let Some(class_ptr) = frame.defining_class {
                        unsafe {
                            (*class_ptr)
                                .static_fields
                                .get(&sym)
                                .copied()
                                .unwrap_or(Value::null())
                        }
                    } else {
                        Value::null()
                    };
                    set_reg(&mut values, dst, val);
                }
                Op::SetStaticField => {
                    let dst = read_u16(code, &mut pc);
                    let val_reg = read_u16(code, &mut pc);
                    let sym_idx = read_u16(code, &mut pc);
                    let sym = SymbolId::from_raw(sym_idx as u32);
                    let v = get_reg(&values, val_reg);
                    let frame = unsafe { (*fiber).mir_frames.last().unwrap() };
                    if let Some(class_ptr) = frame.defining_class {
                        unsafe {
                            (*class_ptr).static_fields.insert(sym, v);
                        }
                        vm.gc.write_barrier(class_ptr as *mut ObjHeader, v);
                    }
                    set_reg(&mut values, dst, v);
                }

                // =============================================================
                // Unary ops (5B: op + dst + src)
                // =============================================================
                Op::Neg => {
                    let dst = read_u16(code, &mut pc);
                    let src = read_u16(code, &mut pc);
                    let a = get_reg(&values, src);
                    match a.as_num() {
                        Some(n) => set_reg(&mut values, dst, Value::num(-n)),
                        None => {
                            let recv = a;
                            let result = try_operator_dispatch(
                                vm,
                                fiber,
                                recv,
                                "-",
                                &[],
                                &mut pc,
                                &mut values,
                                dst,
                                &module_name,
                                bc_ptr,
                            )?;
                            if result {
                                continue 'fiber_loop;
                            }
                        }
                    }
                }
                Op::NegF64 => {
                    let dst = read_u16(code, &mut pc);
                    let src = read_u16(code, &mut pc);
                    let n = get_reg(&values, src)
                        .as_num()
                        .ok_or(RuntimeError::GuardFailed("num"))?;
                    set_reg(&mut values, dst, Value::num(-n));
                }
                Op::Not => {
                    let dst = read_u16(code, &mut pc);
                    let src = read_u16(code, &mut pc);
                    let a = get_reg(&values, src);
                    set_reg(&mut values, dst, Value::bool(!a.is_truthy_wren()));
                }
                Op::BitNot => {
                    let dst = read_u16(code, &mut pc);
                    let src = read_u16(code, &mut pc);
                    let a = get_reg(&values, src);
                    match a.as_num() {
                        Some(n) => {
                            let i = n as i32;
                            set_reg(&mut values, dst, Value::num((!i) as f64));
                        }
                        None => {
                            let recv = a;
                            let result = try_operator_dispatch(
                                vm,
                                fiber,
                                recv,
                                "~",
                                &[],
                                &mut pc,
                                &mut values,
                                dst,
                                &module_name,
                                bc_ptr,
                            )?;
                            if result {
                                continue 'fiber_loop;
                            }
                        }
                    }
                }
                Op::Unbox => {
                    // With NaN-boxed Values, unbox is a no-op (value is already boxed).
                    // Just verify it's a number and copy.
                    let dst = read_u16(code, &mut pc);
                    let src = read_u16(code, &mut pc);
                    let a = get_reg(&values, src);
                    if a.as_num().is_none() {
                        return Err(RuntimeError::Error("unbox: not a number".into()));
                    }
                    set_reg(&mut values, dst, a);
                }
                Op::BoxOp => {
                    let dst = read_u16(code, &mut pc);
                    let src = read_u16(code, &mut pc);
                    let val = get_reg(&values, src);
                    set_reg(&mut values, dst, val);
                }
                Op::Move => {
                    let dst = read_u16(code, &mut pc);
                    let src = read_u16(code, &mut pc);
                    let val = get_reg(&values, src);
                    set_reg(&mut values, dst, val);
                }
                Op::ToStringOp => {
                    let dst = read_u16(code, &mut pc);
                    let src = read_u16(code, &mut pc);
                    let s = value_to_string(vm, get_reg(&values, src));
                    set_reg(&mut values, dst, vm.new_string(s));
                }
                Op::GuardNum => {
                    let dst = read_u16(code, &mut pc);
                    let src = read_u16(code, &mut pc);
                    let v = get_reg(&values, src);
                    if !v.is_num() {
                        return Err(RuntimeError::GuardFailed("Num"));
                    }
                    set_reg(&mut values, dst, v);
                }
                Op::GuardBool => {
                    let dst = read_u16(code, &mut pc);
                    let src = read_u16(code, &mut pc);
                    let v = get_reg(&values, src);
                    if !v.is_bool() {
                        return Err(RuntimeError::GuardFailed("Bool"));
                    }
                    set_reg(&mut values, dst, v);
                }

                // =============================================================
                // Guard/type ops (7B: op + dst + src + imm16)
                // =============================================================
                Op::GuardClass => {
                    let dst = read_u16(code, &mut pc);
                    let src = read_u16(code, &mut pc);
                    let _class_sym = read_u16(code, &mut pc);
                    let val = get_reg(&values, src);
                    set_reg(&mut values, dst, val);
                }
                Op::GuardProtocol => {
                    let dst = read_u16(code, &mut pc);
                    let src = read_u16(code, &mut pc);
                    let _proto_sym = read_u16(code, &mut pc);
                    let val = get_reg(&values, src);
                    set_reg(&mut values, dst, val);
                }
                Op::IsType => {
                    let dst = read_u16(code, &mut pc);
                    let val_reg = read_u16(code, &mut pc);
                    let class_sym_idx = read_u16(code, &mut pc);
                    let class_sym = SymbolId::from_raw(class_sym_idx as u32);
                    let value = get_reg(&values, val_reg);
                    let mut cls = vm.class_of(value);
                    let mut matched = false;
                    while !cls.is_null() {
                        if unsafe { (*cls).name } == class_sym {
                            matched = true;
                            break;
                        }
                        cls = unsafe { (*cls).superclass };
                    }
                    set_reg(&mut values, dst, Value::bool(matched));
                }

                // =============================================================
                // Field access
                // =============================================================
                Op::GetField => {
                    let dst = read_u16(code, &mut pc);
                    let recv_reg = read_u16(code, &mut pc);
                    let field_idx = read_u16(code, &mut pc) as usize;
                    let recv = get_reg(&values, recv_reg);
                    let val = if let Some(ptr) = recv.as_object() {
                        let inst = ptr as *const ObjInstance;
                        unsafe { (*inst).get_field_unchecked(field_idx) }
                    } else {
                        Value::null()
                    };
                    set_reg(&mut values, dst, val);
                }
                Op::SetField => {
                    let dst = read_u16(code, &mut pc);
                    let recv_reg = read_u16(code, &mut pc);
                    let field_idx = read_u16(code, &mut pc) as usize;
                    let val_reg = read_u16(code, &mut pc);
                    let recv = get_reg(&values, recv_reg);
                    let val = get_reg(&values, val_reg);
                    if let Some(ptr) = recv.as_object() {
                        let inst = ptr as *mut ObjInstance;
                        unsafe { (*inst).set_field_unchecked(field_idx, val) };
                        vm.gc.write_barrier(ptr as *mut ObjHeader, val);
                    }
                    set_reg(&mut values, dst, val);
                }

                // =============================================================
                // Math intrinsics
                // =============================================================
                Op::MathUnaryF64 => {
                    let dst = read_u16(code, &mut pc);
                    let src = read_u16(code, &mut pc);
                    let math_op_byte = read_u8(code, &mut pc);
                    let _pad = read_u8(code, &mut pc); // alignment pad
                    let math_op = crate::mir::bytecode::u8_to_math_unary(math_op_byte);
                    let n = get_reg(&values, src)
                        .as_num()
                        .ok_or(RuntimeError::GuardFailed("num"))?;
                    set_reg(&mut values, dst, Value::num(math_op.apply(n)));
                }
                Op::MathBinaryF64 => {
                    let dst = read_u16(code, &mut pc);
                    let lhs = read_u16(code, &mut pc);
                    let rhs = read_u16(code, &mut pc);
                    let math_op_byte = read_u8(code, &mut pc);
                    let math_op = crate::mir::bytecode::u8_to_math_binary(math_op_byte);
                    let a = get_reg(&values, lhs)
                        .as_num()
                        .ok_or(RuntimeError::GuardFailed("num"))?;
                    let b = get_reg(&values, rhs)
                        .as_num()
                        .ok_or(RuntimeError::GuardFailed("num"))?;
                    set_reg(&mut values, dst, Value::num(math_op.apply(a, b)));
                }

                // =============================================================
                // Binary arithmetic (7B: op + dst + lhs + rhs)
                // =============================================================
                Op::Add => {
                    if bc_boxed_binop!(
                        vm,
                        fiber,
                        &mut pc,
                        &mut values,
                        &module_name,
                        bc_ptr,
                        |x: f64, y: f64| x + y,
                        "+(_)"
                    ) {
                        continue 'fiber_loop;
                    }
                }
                Op::Sub => {
                    if bc_boxed_binop!(
                        vm,
                        fiber,
                        &mut pc,
                        &mut values,
                        &module_name,
                        bc_ptr,
                        |x: f64, y: f64| x - y,
                        "-(_)"
                    ) {
                        continue 'fiber_loop;
                    }
                }
                Op::Mul => {
                    if bc_boxed_binop!(
                        vm,
                        fiber,
                        &mut pc,
                        &mut values,
                        &module_name,
                        bc_ptr,
                        |x: f64, y: f64| x * y,
                        "*(_)"
                    ) {
                        continue 'fiber_loop;
                    }
                }
                Op::Div => {
                    if bc_boxed_binop!(
                        vm,
                        fiber,
                        &mut pc,
                        &mut values,
                        &module_name,
                        bc_ptr,
                        |x: f64, y: f64| x / y,
                        "/(_)"
                    ) {
                        continue 'fiber_loop;
                    }
                }
                Op::Mod => {
                    if bc_boxed_binop!(
                        vm,
                        fiber,
                        &mut pc,
                        &mut values,
                        &module_name,
                        bc_ptr,
                        |x: f64, y: f64| x % y,
                        "%(_)"
                    ) {
                        continue 'fiber_loop;
                    }
                }

                Op::AddF64 => {
                    bc_f64_binop!(&mut pc, &mut values, |x, y| x + y, &bc);
                }
                Op::SubF64 => {
                    bc_f64_binop!(&mut pc, &mut values, |x, y| x - y, &bc);
                }
                Op::MulF64 => {
                    bc_f64_binop!(&mut pc, &mut values, |x, y| x * y, &bc);
                }
                Op::DivF64 => {
                    bc_f64_binop!(&mut pc, &mut values, |x, y| x / y, &bc);
                }
                Op::ModF64 => {
                    bc_f64_binop!(&mut pc, &mut values, |x, y| x % y, &bc);
                }

                // =============================================================
                // Comparisons
                // =============================================================
                Op::CmpLt => {
                    if bc_boxed_cmp!(
                        vm,
                        fiber,
                        &mut pc,
                        &mut values,
                        &module_name,
                        bc_ptr,
                        |x: f64, y: f64| x < y,
                        "<(_)"
                    ) {
                        continue 'fiber_loop;
                    }
                }
                Op::CmpGt => {
                    if bc_boxed_cmp!(
                        vm,
                        fiber,
                        &mut pc,
                        &mut values,
                        &module_name,
                        bc_ptr,
                        |x: f64, y: f64| x > y,
                        ">(_)"
                    ) {
                        continue 'fiber_loop;
                    }
                }
                Op::CmpLe => {
                    if bc_boxed_cmp!(
                        vm,
                        fiber,
                        &mut pc,
                        &mut values,
                        &module_name,
                        bc_ptr,
                        |x: f64, y: f64| x <= y,
                        "<=(_)"
                    ) {
                        continue 'fiber_loop;
                    }
                }
                Op::CmpGe => {
                    if bc_boxed_cmp!(
                        vm,
                        fiber,
                        &mut pc,
                        &mut values,
                        &module_name,
                        bc_ptr,
                        |x: f64, y: f64| x >= y,
                        ">=(_)"
                    ) {
                        continue 'fiber_loop;
                    }
                }
                Op::CmpEq => {
                    let dst = read_u16(code, &mut pc);
                    let lhs_reg = read_u16(code, &mut pc);
                    let rhs_reg = read_u16(code, &mut pc);
                    let lhs = get_reg(&values, lhs_reg);
                    let rhs = get_reg(&values, rhs_reg);
                    // For non-instance objects, try operator dispatch
                    if lhs.is_object()
                        && !lhs.is_string_object()
                        && !rhs.is_null()
                        && !rhs.is_bool()
                        && !rhs.is_num()
                    {
                        let result = try_operator_dispatch(
                            vm,
                            fiber,
                            lhs,
                            "==(_)",
                            &[rhs],
                            &mut pc,
                            &mut values,
                            dst,
                            &module_name,
                            bc_ptr,
                        )?;
                        if result {
                            continue 'fiber_loop;
                        }
                    } else {
                        set_reg(&mut values, dst, Value::bool(lhs == rhs));
                    }
                }
                Op::CmpNe => {
                    let dst = read_u16(code, &mut pc);
                    let lhs_reg = read_u16(code, &mut pc);
                    let rhs_reg = read_u16(code, &mut pc);
                    let lhs = get_reg(&values, lhs_reg);
                    let rhs = get_reg(&values, rhs_reg);
                    if lhs.is_object()
                        && !lhs.is_string_object()
                        && !rhs.is_null()
                        && !rhs.is_bool()
                        && !rhs.is_num()
                    {
                        let result = try_operator_dispatch(
                            vm,
                            fiber,
                            lhs,
                            "!=(_)",
                            &[rhs],
                            &mut pc,
                            &mut values,
                            dst,
                            &module_name,
                            bc_ptr,
                        )?;
                        if result {
                            continue 'fiber_loop;
                        }
                    } else {
                        set_reg(&mut values, dst, Value::bool(lhs != rhs));
                    }
                }

                Op::CmpLtF64 => {
                    bc_f64_cmp!(&mut pc, &mut values, |x, y| x < y, &bc);
                }
                Op::CmpGtF64 => {
                    bc_f64_cmp!(&mut pc, &mut values, |x, y| x > y, &bc);
                }
                Op::CmpLeF64 => {
                    bc_f64_cmp!(&mut pc, &mut values, |x, y| x <= y, &bc);
                }
                Op::CmpGeF64 => {
                    bc_f64_cmp!(&mut pc, &mut values, |x, y| x >= y, &bc);
                }

                // =============================================================
                // Bitwise ops
                // =============================================================
                Op::BitAnd => {
                    if bc_bitwise_binop!(
                        vm,
                        fiber,
                        &mut pc,
                        &mut values,
                        &module_name,
                        bc_ptr,
                        |x: i32, y: i32| x & y,
                        "&(_)"
                    ) {
                        continue 'fiber_loop;
                    }
                }
                Op::BitOr => {
                    if bc_bitwise_binop!(
                        vm,
                        fiber,
                        &mut pc,
                        &mut values,
                        &module_name,
                        bc_ptr,
                        |x: i32, y: i32| x | y,
                        "|(_)"
                    ) {
                        continue 'fiber_loop;
                    }
                }
                Op::BitXor => {
                    let dst = read_u16(code, &mut pc);
                    let lhs_reg = read_u16(code, &mut pc);
                    let rhs_reg = read_u16(code, &mut pc);
                    let a = get_reg(&values, lhs_reg);
                    let b = get_reg(&values, rhs_reg);
                    match (a.as_num(), b.as_num()) {
                        (Some(x), Some(y)) => {
                            set_reg(
                                &mut values,
                                dst,
                                Value::num(((x as i32) ^ (y as i32)) as f64),
                            );
                        }
                        _ => {
                            return Err(RuntimeError::Error("bitwise xor: not numbers".into()));
                        }
                    }
                }
                Op::Shl => {
                    if bc_bitwise_binop!(
                        vm,
                        fiber,
                        &mut pc,
                        &mut values,
                        &module_name,
                        bc_ptr,
                        |x: i32, y: i32| x << (y & 31),
                        "<<(_)"
                    ) {
                        continue 'fiber_loop;
                    }
                }
                Op::Shr => {
                    if bc_bitwise_binop!(
                        vm,
                        fiber,
                        &mut pc,
                        &mut values,
                        &module_name,
                        bc_ptr,
                        |x: i32, y: i32| x >> (y & 31),
                        ">>(_)"
                    ) {
                        continue 'fiber_loop;
                    }
                }

                // =============================================================
                // Range
                // =============================================================
                Op::MakeRange => {
                    let dst = read_u16(code, &mut pc);
                    let from_reg = read_u16(code, &mut pc);
                    let to_reg = read_u16(code, &mut pc);
                    let inclusive = read_u8(code, &mut pc) != 0;
                    let f = get_reg(&values, from_reg)
                        .as_num()
                        .ok_or(RuntimeError::GuardFailed("num"))?;
                    let t = get_reg(&values, to_reg)
                        .as_num()
                        .ok_or(RuntimeError::GuardFailed("num"))?;
                    set_reg(&mut values, dst, vm.new_range(f, t, inclusive));
                }

                // =============================================================
                // Method calls
                // =============================================================
                Op::Call => {
                    let dst = read_u16(code, &mut pc);
                    let recv_reg = read_u16(code, &mut pc);
                    let method_idx = read_u16(code, &mut pc);
                    let ic_idx = read_u16(code, &mut pc) as usize;
                    let argc = read_u8(code, &mut pc) as usize;
                    let method = SymbolId::from_raw(method_idx as u32);

                    let recv_val = get_reg(&values, recv_reg);

                    // Record where arg register indices start in bytecode,
                    // then advance pc past them. Args are read lazily.
                    let arg_regs_pc = pc;
                    pc += (argc as u32) * 2;

                    // ── IC fast path: monomorphic inline cache ──────────────
                    // Stripped to bare minimum: class check → call → store.
                    // No tier-up profiling, no IC re-validation, no stats.
                    // Tier-up happens via slow path + back-edge counting.
                    let ic_table = unsafe { &*bc.ic_table.get() };
                    if ic_idx < ic_table.len() {
                        let ic = &ic_table[ic_idx];
                        if ic.kind == 1 && recv_val.is_object() {
                            let obj_ptr = unsafe { recv_val.as_object().unwrap_unchecked() };
                            let recv_class =
                                unsafe { (*(obj_ptr as *const ObjHeader)).class as usize };
                            let fn_idx_ic = ic.func_id as usize;
                            let is_leaf =
                                vm.engine.jit_leaf.get(fn_idx_ic).copied().unwrap_or(false);
                            // The IC kind=1 inline JIT-leaf dispatch
                            // passes recv + args in registers, which the
                            // GC root scanner can't see (no JIT-frame
                            // stack maps). To stay sound we restrict the
                            // fast path to callees that transitively
                            // can't fire a GC: no allocations in the
                            // body, and no calls that allocate either.
                            // `func_is_alloc_free` is computed by the
                            // module-level purity / alloc-free pass and
                            // lives as a Vec<bool> indexed by FuncId for
                            // O(1) lookup on this hot path.
                            //
                            // Setting `WLIFT_ENABLE_IC_JIT=1` overrides
                            // the alloc-free gate for benchmarks /
                            // diagnosis. `WLIFT_DISABLE_IC_JIT=1` turns
                            // the fast path off entirely.
                            // BISECT: temporarily revert the alloc-free
                            // gate to the previous behavior (env-var
                            // opt-in only). The Linux x86_64 CI bench
                            // is dumping core on delta_blue and the
                            // alloc-free auto-enable is the most
                            // likely culprit — the analysis runs over
                            // engine MIR concurrently with JIT
                            // submissions, and a stale read could
                            // route a not-actually-alloc-free callee
                            // into the IC fast path.
                            let force_on = std::env::var_os("WLIFT_ENABLE_IC_JIT").is_some();
                            if recv_class == ic.class && is_leaf && force_on {
                                if std::env::var_os("WLIFT_TRACE_IC_JIT").is_some() {
                                    let fn_idx_ic = ic.func_id as usize;
                                    let name = vm
                                        .engine
                                        .get_mir(FuncId(fn_idx_ic as u32))
                                        .map(|m| vm.interner.resolve(m.name).to_string())
                                        .unwrap_or_else(|| "<unknown>".into());
                                    eprintln!(
                                        "ic-jit: dispatch fn_idx={} name='{}' argc={}",
                                        fn_idx_ic, name, argc
                                    );
                                }
                                let jit_ptr = ic.jit_ptr;
                                ensure_ctx_reg();
                                // Read recv + arg values, then push them
                                // as JIT roots so a GC inside the callee
                                // updates these pointers before the JIT'd
                                // body dereferences them. Without this,
                                // a generational promote during the
                                // callee leaves the u64 args pointing at
                                // freed memory; the JIT callee reads
                                // garbage and the caller's `values`
                                // entries (now updated via frame.values
                                // tracing) disagree with the args the
                                // callee was invoked with.
                                let arg_count = argc;
                                let mut arg_regs: SmallVec<[u16; 4]> = SmallVec::new();
                                for i in 0..arg_count {
                                    arg_regs.push(read_u16_at(code, arg_regs_pc + (i as u32) * 2));
                                }
                                let root_base =
                                    crate::codegen::runtime_fns::jit_roots_snapshot_len();
                                crate::codegen::runtime_fns::push_jit_root(recv_val);
                                for &reg in &arg_regs {
                                    crate::codegen::runtime_fns::push_jit_root(get_reg(
                                        &values, reg,
                                    ));
                                }
                                // Publish caller's register file before
                                // the JIT call so its remaining live
                                // pointers also get traced through
                                // mir_frames.
                                unsafe {
                                    let frame = (*fiber).mir_frames.last_mut().unwrap();
                                    frame.pc = pc;
                                    frame.values = std::mem::take(&mut values);
                                }
                                let result_bits = unsafe {
                                    let recv_bits =
                                        crate::codegen::runtime_fns::jit_root_at(root_base)
                                            .to_bits();
                                    match argc {
                                        0 => {
                                            let f: extern "C" fn(u64) -> u64 =
                                                std::mem::transmute(jit_ptr);
                                            f(recv_bits)
                                        }
                                        1 => {
                                            let a1 = crate::codegen::runtime_fns::jit_root_at(
                                                root_base + 1,
                                            )
                                            .to_bits();
                                            let f: extern "C" fn(u64, u64) -> u64 =
                                                std::mem::transmute(jit_ptr);
                                            f(recv_bits, a1)
                                        }
                                        2 => {
                                            let a1 = crate::codegen::runtime_fns::jit_root_at(
                                                root_base + 1,
                                            )
                                            .to_bits();
                                            let a2 = crate::codegen::runtime_fns::jit_root_at(
                                                root_base + 2,
                                            )
                                            .to_bits();
                                            let f: extern "C" fn(u64, u64, u64) -> u64 =
                                                std::mem::transmute(jit_ptr);
                                            f(recv_bits, a1, a2)
                                        }
                                        _ => {
                                            let a1 = crate::codegen::runtime_fns::jit_root_at(
                                                root_base + 1,
                                            )
                                            .to_bits();
                                            let a2 = crate::codegen::runtime_fns::jit_root_at(
                                                root_base + 2,
                                            )
                                            .to_bits();
                                            let a3 = crate::codegen::runtime_fns::jit_root_at(
                                                root_base + 3,
                                            )
                                            .to_bits();
                                            let f: extern "C" fn(u64, u64, u64, u64) -> u64 =
                                                std::mem::transmute(jit_ptr);
                                            f(recv_bits, a1, a2, a3)
                                        }
                                    }
                                };
                                crate::codegen::runtime_fns::jit_roots_restore_len(root_base);
                                // Reload the (possibly GC-updated)
                                // register file from the frame.
                                unsafe {
                                    values = std::mem::take(
                                        &mut (*fiber).mir_frames.last_mut().unwrap().values,
                                    );
                                }
                                set_reg(&mut values, dst, Value::from_bits(result_bits));
                                steps += 1;
                                continue;
                            }
                        }
                    }

                    // Save pc before slow-path dispatch (for error reporting / frame push)
                    unsafe {
                        if let Some(frame) = (*fiber).mir_frames.last_mut() {
                            frame.pc = pc;
                        }
                    }

                    // ── Full dispatch (populates IC on miss) ─────────────────
                    // Closure call() fast path
                    if vm.is_call_sym(method) && recv_val.is_object() {
                        let ptr = recv_val.as_object().unwrap();
                        let header = ptr as *const ObjHeader;
                        if unsafe { (*header).obj_type } == ObjType::Closure {
                            let closure_ptr = ptr as *mut ObjClosure;
                            let arg_vals =
                                build_arg_vals(recv_val, code, arg_regs_pc, argc, &values);
                            dispatch_closure_bc(
                                vm,
                                fiber,
                                closure_ptr,
                                &arg_vals[1..],
                                pc,
                                values,
                                &module_name,
                                ValueId(dst as u32),
                                None,
                                bc_ptr,
                            )?;
                            continue 'fiber_loop;
                        }
                    }

                    // Build arg values for the slow path
                    let arg_vals = build_arg_vals(recv_val, code, arg_regs_pc, argc, &values);

                    let class = vm.class_of(recv_val);

                    let cache_key_class = if class == vm.class_class {
                        recv_val.as_object().unwrap() as *mut ObjClass
                    } else {
                        class
                    };

                    let (method_entry, defining_class) =
                        if let Some((m, dc)) = vm.method_cache.lookup(cache_key_class, method) {
                            (Some(m), Some(dc))
                        } else {
                            let result = if class == vm.class_class {
                                let recv_class_ptr = cache_key_class;
                                let method_str = vm.interner.resolve(method);
                                let prefix = b"static:";
                                let total = prefix.len() + method_str.len();
                                let mut buf = [0u8; 128];
                                let found = if total <= buf.len() {
                                    buf[..prefix.len()].copy_from_slice(prefix);
                                    buf[prefix.len()..total].copy_from_slice(method_str.as_bytes());
                                    let static_str =
                                        unsafe { std::str::from_utf8_unchecked(&buf[..total]) };
                                    vm.interner.lookup(static_str).and_then(|sym| {
                                        unsafe { (*recv_class_ptr).find_method(sym).cloned() }
                                            .map(|m| (m, recv_class_ptr))
                                    })
                                } else {
                                    None
                                };
                                found.or_else(|| unsafe { find_method_with_class(class, method) })
                            } else {
                                unsafe { find_method_with_class(class, method) }
                            };
                            if let Some((ref m, dc)) = result {
                                vm.method_cache.insert(cache_key_class, method, *m, dc);
                            }
                            match result {
                                Some((m, c)) => (Some(m), Some(c)),
                                None => (None, None),
                            }
                        };

                    match method_entry {
                        Some(m @ (Method::Native(_) | Method::ForeignC(_) | Method::ForeignCDynamic(_))) => {
                            let result = match m {
                                Method::Native(func) => call_native_with_frame_sync(
                                    vm,
                                    fiber,
                                    pc,
                                    &mut values,
                                    func,
                                    &arg_vals,
                                ),
                                Method::ForeignC(func) => call_foreign_c_with_frame_sync(
                                    vm,
                                    fiber,
                                    pc,
                                    &mut values,
                                    func,
                                    &arg_vals,
                                ),
                                Method::ForeignCDynamic(idx) => call_foreign_dynamic_with_frame_sync(
                                    vm,
                                    fiber,
                                    pc,
                                    &mut values,
                                    idx,
                                    &arg_vals,
                                ),
                                _ => unreachable!(),
                            };
                            if vm.has_error {
                                vm.has_error = false;
                                let fiber_is_try = unsafe { (*fiber).is_try };
                                if fiber_is_try {
                                    let err_msg = vm.last_error.take().unwrap_or_default();
                                    let err_val = vm.new_string(err_msg);
                                    unsafe {
                                        (*fiber).error = err_val;
                                        (*fiber).is_try = false;
                                        (*fiber).state = FiberState::Done;
                                    }
                                    let caller = unsafe { (*fiber).caller };
                                    if !caller.is_null() {
                                        unsafe {
                                            (*fiber).caller = std::ptr::null_mut();
                                        }
                                        resume_caller(vm, caller, err_val);
                                        continue 'fiber_loop;
                                    }
                                    return Ok(err_val);
                                }
                                return Err(RuntimeError::Error(
                                    "runtime error in native method".into(),
                                ));
                            }
                            if let Some(action) = vm.pending_fiber_action.take() {
                                handle_fiber_action_bc(
                                    vm,
                                    fiber,
                                    action,
                                    pc,
                                    values,
                                    ValueId(dst as u32),
                                )?;
                                continue 'fiber_loop;
                            }
                            set_reg(&mut values, dst, result);
                        }
                        Some(Method::Closure(closure_ptr)) => {
                            // Try to populate IC for JIT leaf methods.
                            if vm.engine.mode != ExecutionMode::Interpreter && argc <= 3 {
                                let fn_ptr = unsafe { (*closure_ptr).function };
                                let fn_idx = unsafe { (*fn_ptr).fn_id } as usize;
                                let jit_ptr = vm
                                    .engine
                                    .jit_code
                                    .get(fn_idx)
                                    .copied()
                                    .unwrap_or(std::ptr::null());
                                // For Cranelift: only inline-dispatch leaf functions.
                                // Non-leaf functions use dispatch_closure_bc which
                                // properly saves/restores JIT context for re-entrant calls.
                                #[cfg(feature = "cranelift")]
                                let jit_dispatch_ok = !jit_ptr.is_null()
                                    && vm.engine.jit_leaf.get(fn_idx).copied().unwrap_or(false);
                                #[cfg(not(feature = "cranelift"))]
                                let jit_dispatch_ok = !jit_ptr.is_null()
                                    && vm.engine.jit_leaf.get(fn_idx).copied().unwrap_or(false);
                                if std::env::var_os("WLIFT_TRACE_JIT_CALL").is_some() {
                                    eprintln!(
                                        "JIT-CHECK: fn_idx={} argc={} jit_null={} ok={}",
                                        fn_idx,
                                        argc,
                                        jit_ptr.is_null(),
                                        jit_dispatch_ok
                                    );
                                }
                                if jit_dispatch_ok {
                                    // Populate IC for this call site
                                    let ic_table = unsafe { &mut *bc.ic_table.get() };
                                    if ic_idx < ic_table.len() {
                                        ic_table[ic_idx] = crate::mir::bytecode::CallSiteIC {
                                            class: cache_key_class as usize,
                                            jit_ptr,
                                            closure: closure_ptr as *const u8,
                                            func_id: fn_idx as u64,
                                            kind: 1, // JIT leaf
                                        };
                                    }
                                    // Swap in the callee's module context.
                                    // Cross-module calls (e.g. a hot-loop in
                                    // crypto.spec calling `Expect.that` defined
                                    // in @hatch:assert) need the callee's own
                                    // `module_vars` pointer for `GetModuleVar @N`
                                    // to resolve to the right class — otherwise
                                    // the JIT reads the CALLER's slot N, and
                                    // downstream method lookup fails silently
                                    // with Null (classic tiered-mode "Null.toBe"
                                    // flake).
                                    //
                                    // The saved_ctx is restored immediately after
                                    // the JIT call returns so the interpreter
                                    // continues with its own module context.
                                    // Fast path for intra-module calls:
                                    // compare the callee's module name to
                                    // the current context's. If equal, the
                                    // module_vars pointer is already right
                                    // and we skip the lookup + re-write.
                                    // Keeps recursive method dispatch (fib,
                                    // method_call, binary_trees) at zero
                                    // additional overhead after the fix.
                                    let saved_ctx = crate::codegen::runtime_fns::read_jit_ctx();
                                    let callee_module =
                                        vm.engine.func_module(FuncId(fn_idx as u32));
                                    let same_module = match callee_module {
                                        Some(mn) => {
                                            let bytes = mn.as_bytes();
                                            bytes.as_ptr() == saved_ctx.module_name
                                                && bytes.len() as u32 == saved_ctx.module_name_len
                                        }
                                        None => true,
                                    };
                                    crate::codegen::runtime_fns::mutate_jit_ctx(|ctx| {
                                        ctx.current_func_id = fn_idx as u64;
                                        ctx.closure = closure_ptr as *mut u8;
                                        ctx.defining_class = defining_class
                                            .map(|p| p as *mut u8)
                                            .unwrap_or(std::ptr::null_mut());
                                        if !same_module {
                                            if let Some(mod_name) = callee_module {
                                                if let Some(m) =
                                                    vm.engine.modules.get(mod_name.as_str())
                                                {
                                                    ctx.module_vars = m.vars.as_ptr() as *mut u64;
                                                    ctx.module_var_count = m.vars.len() as u32;
                                                    let bytes = mod_name.as_bytes();
                                                    ctx.module_name = bytes.as_ptr();
                                                    ctx.module_name_len = bytes.len() as u32;
                                                }
                                            }
                                        }
                                    });
                                    ensure_ctx_reg();
                                    // Push args as JIT roots BEFORE the
                                    // call so a GC fired by the callee
                                    // updates these pointers before the
                                    // callee dereferences them. The
                                    // u64-bit-cast snapshots in the
                                    // function-call register slots are
                                    // invisible to the GC otherwise —
                                    // generational promote then leaves
                                    // them pointing at freed memory and
                                    // the callee reads garbage.
                                    let root_base =
                                        crate::codegen::runtime_fns::jit_roots_snapshot_len();
                                    for v in arg_vals.iter() {
                                        crate::codegen::runtime_fns::push_jit_root(*v);
                                    }
                                    // Publish the caller's register file
                                    // to its frame BEFORE the JIT call
                                    // so GC running mid-callee can see
                                    // every Value still live in the
                                    // caller and update its pointers if
                                    // objects move.
                                    unsafe {
                                        let frame = (*fiber).mir_frames.last_mut().unwrap();
                                        frame.pc = pc;
                                        frame.values = std::mem::take(&mut values);
                                    }
                                    let result_bits = unsafe {
                                        let r0 =
                                            crate::codegen::runtime_fns::jit_root_at(root_base)
                                                .to_bits();
                                        match argc {
                                            0 => {
                                                let f: extern "C" fn(u64) -> u64 =
                                                    std::mem::transmute(jit_ptr);
                                                f(r0)
                                            }
                                            1 => {
                                                let r1 = crate::codegen::runtime_fns::jit_root_at(
                                                    root_base + 1,
                                                )
                                                .to_bits();
                                                let f: extern "C" fn(u64, u64) -> u64 =
                                                    std::mem::transmute(jit_ptr);
                                                f(r0, r1)
                                            }
                                            2 => {
                                                let r1 = crate::codegen::runtime_fns::jit_root_at(
                                                    root_base + 1,
                                                )
                                                .to_bits();
                                                let r2 = crate::codegen::runtime_fns::jit_root_at(
                                                    root_base + 2,
                                                )
                                                .to_bits();
                                                let f: extern "C" fn(u64, u64, u64) -> u64 =
                                                    std::mem::transmute(jit_ptr);
                                                f(r0, r1, r2)
                                            }
                                            _ => {
                                                let r1 = crate::codegen::runtime_fns::jit_root_at(
                                                    root_base + 1,
                                                )
                                                .to_bits();
                                                let r2 = crate::codegen::runtime_fns::jit_root_at(
                                                    root_base + 2,
                                                )
                                                .to_bits();
                                                let r3 = crate::codegen::runtime_fns::jit_root_at(
                                                    root_base + 3,
                                                )
                                                .to_bits();
                                                let f: extern "C" fn(u64, u64, u64, u64) -> u64 =
                                                    std::mem::transmute(jit_ptr);
                                                f(r0, r1, r2, r3)
                                            }
                                        }
                                    };
                                    crate::codegen::runtime_fns::jit_roots_restore_len(root_base);
                                    // Reload the caller's register file —
                                    // GC during the call may have updated
                                    // entries through the frame.values
                                    // root we published above.
                                    unsafe {
                                        values = std::mem::take(
                                            &mut (*fiber).mir_frames.last_mut().unwrap().values,
                                        );
                                    }
                                    // Restore the caller's module context so
                                    // the next interpreter ops read from the
                                    // right module_vars.
                                    crate::codegen::runtime_fns::set_jit_context(saved_ctx);
                                    vm.engine.note_native_entry(FuncId(fn_idx as u32));
                                    set_reg(&mut values, dst, Value::from_bits(result_bits));
                                    steps += 1;
                                    continue;
                                }
                            }
                            // ── Inline closure dispatch ──
                            let target_fn_ptr = unsafe { (*closure_ptr).function };
                            let target_func_id = FuncId(unsafe { (*target_fn_ptr).fn_id });
                            let fn_idx = target_func_id.0 as usize;

                            // Check if already JIT-compiled → use dispatch_closure_bc for JIT path
                            if vm.engine.mode != ExecutionMode::Interpreter {
                                let jit_ptr = vm
                                    .engine
                                    .jit_code
                                    .get(fn_idx)
                                    .copied()
                                    .unwrap_or(std::ptr::null());
                                if !jit_ptr.is_null() || {
                                    let should_tier_up = vm.engine.record_call(target_func_id);
                                    if should_tier_up {
                                        vm.engine.request_tier_up(target_func_id, &vm.interner);
                                    }
                                    vm.engine.poll_compilations();
                                    // Re-check after poll
                                    !vm.engine
                                        .jit_code
                                        .get(fn_idx)
                                        .copied()
                                        .unwrap_or(std::ptr::null())
                                        .is_null()
                                } {
                                    dispatch_closure_bc(
                                        vm,
                                        fiber,
                                        closure_ptr,
                                        &arg_vals,
                                        pc,
                                        values,
                                        &module_name,
                                        ValueId(dst as u32),
                                        defining_class,
                                        bc_ptr,
                                    )?;
                                    continue 'fiber_loop;
                                }
                            }

                            // ── Inline frame push: avoid fiber_loop restart ──
                            let target_bc_ptr =
                                vm.engine.ensure_bytecode(target_func_id).ok_or_else(|| {
                                    RuntimeError::Error("invalid closure func_id".into())
                                })?;
                            let target_bc = unsafe { &*target_bc_ptr };

                            // Allocate register file
                            let reg_count = target_bc.register_count as usize;
                            let mut new_values = if let Some(mut recycled) = vm.register_pool.pop()
                            {
                                recycled.resize(reg_count, UNDEF);
                                new_values_fill_undef(&mut recycled, reg_count);
                                recycled
                            } else {
                                vec![UNDEF; reg_count]
                            };
                            // Bind params
                            for &(dst_reg, param_idx) in target_bc.param_offsets.iter() {
                                if (param_idx as usize) < arg_vals.len()
                                    && (dst_reg as usize) < reg_count
                                {
                                    new_values[dst_reg as usize] = arg_vals[param_idx as usize];
                                }
                            }

                            // Check call depth
                            let depth = unsafe { (*fiber).mir_frames.len() };
                            if depth >= vm.config.max_call_depth {
                                return Err(RuntimeError::StackOverflow);
                            }

                            // Save caller frame
                            unsafe {
                                let frame = (*fiber).mir_frames.last_mut().unwrap();
                                frame.pc = pc;
                                frame.values = values;
                            }

                            // Save caller's bc_ptr before pushing new frame
                            unsafe {
                                (*fiber).mir_frames.last_mut().unwrap().bc_ptr = bc_ptr;
                            }

                            // Push new frame. `module_name` must be the callee's
                            // defining module, not the caller's — otherwise
                            // `GetModuleVar @idx` inside the callee reads the
                            // caller's slot, which is a different variable
                            // (see `Engine::func_module`). Fall back to the
                            // caller's module only when the callee has no
                            // recorded module (isolated test paths).
                            vm.engine.note_interpreted_entry(target_func_id);
                            let callee_module = vm
                                .engine
                                .func_module(target_func_id)
                                .cloned()
                                .unwrap_or_else(|| Rc::clone(&module_name));
                            unsafe {
                                (*fiber).mir_frames.push(MirCallFrame {
                                    func_id: target_func_id,
                                    current_block: BlockId(0),
                                    ip: 0,
                                    pc: 0,
                                    values: new_values,
                                    module_name: callee_module,
                                    return_dst: Some(ValueId(dst as u32)),
                                    closure: Some(closure_ptr),
                                    defining_class,
                                    bc_ptr: target_bc_ptr,
                                });
                            }

                            // Re-enter the fiber loop so locals refresh from
                            // the newly pushed callee frame — including
                            // `module_name` and the derived `module_vars_ptr`,
                            // which determine how `GetModuleVar` resolves
                            // inside the callee.
                            continue 'fiber_loop;
                        }
                        Some(Method::Constructor(closure_ptr)) => {
                            let recv_class = recv_val.as_object().unwrap() as *mut ObjClass;
                            let target_fn_ptr = unsafe { (*closure_ptr).function };
                            let target_func_id = FuncId(unsafe { (*target_fn_ptr).fn_id });

                            // Tier-up profiling + poll BEFORE allocation.
                            // poll_compilations() may install JIT code, changing
                            // whether we take the JIT or interpreter path.
                            if vm.engine.mode != ExecutionMode::Interpreter {
                                let should_tier_up = vm.engine.record_call(target_func_id);
                                if should_tier_up {
                                    vm.engine.request_tier_up(target_func_id, &vm.interner);
                                }
                                vm.engine.poll_compilations();
                            }

                            // JIT dispatch: call_constructor_sync handles its own
                            // instance allocation, avoiding double-allocation.
                            // call_constructor_sync_impl checks has_active_interp_frames
                            // to avoid mid-recursion tier-up bugs.
                            let fn_idx = target_func_id.0 as usize;
                            let ctor_jit = vm
                                .engine
                                .jit_code
                                .get(fn_idx)
                                .copied()
                                .unwrap_or(std::ptr::null());
                            if !ctor_jit.is_null() && argc <= 3 {
                                // Save interpreter state to fiber frame before
                                // calling into JIT (may trigger nested interpreter
                                // calls or GC that modifies fiber state).
                                unsafe {
                                    let frame = (*fiber).mir_frames.last_mut().unwrap();
                                    frame.pc = pc;
                                    frame.values = values;
                                    frame.bc_ptr = bc_ptr;
                                }
                                let result = vm.call_constructor_sync(
                                    recv_class,
                                    closure_ptr,
                                    &arg_vals[1..],
                                );
                                // Re-read interpreter state (may have been modified
                                // by nested calls or GC during constructor execution).
                                values = unsafe {
                                    std::mem::take(
                                        &mut (*fiber).mir_frames.last_mut().unwrap().values,
                                    )
                                };
                                bc_ptr = unsafe { (*fiber).mir_frames.last().unwrap().bc_ptr };
                                bc = unsafe { &*bc_ptr };
                                code = &bc.code;
                                set_reg(&mut values, dst, result);
                                steps += 1;
                                continue;
                            }

                            // Interpreter path: allocate instance here (no JIT code).
                            let instance = vm.gc.alloc_instance(recv_class);
                            let instance_val = Value::object(instance as *mut u8);

                            let target_bc_ptr =
                                vm.engine.ensure_bytecode(target_func_id).ok_or_else(|| {
                                    RuntimeError::Error("invalid ctor func_id".into())
                                })?;
                            let target_bc = unsafe { &*target_bc_ptr };

                            let reg_count = target_bc.register_count as usize;
                            let mut new_values = if let Some(mut recycled) = vm.register_pool.pop()
                            {
                                recycled.resize(reg_count, UNDEF);
                                new_values_fill_undef(&mut recycled, reg_count);
                                recycled
                            } else {
                                vec![UNDEF; reg_count]
                            };
                            // Bind params directly: arg[0]=instance, rest from arg_vals
                            for &(dst_reg, param_idx) in target_bc.param_offsets.iter() {
                                let val = if param_idx == 0 {
                                    instance_val
                                } else if (param_idx as usize) < arg_vals.len() {
                                    arg_vals[param_idx as usize]
                                } else {
                                    UNDEF
                                };
                                if (dst_reg as usize) < reg_count {
                                    new_values[dst_reg as usize] = val;
                                }
                            }

                            let depth = unsafe { (*fiber).mir_frames.len() };
                            if depth >= vm.config.max_call_depth {
                                return Err(RuntimeError::StackOverflow);
                            }

                            unsafe {
                                let frame = (*fiber).mir_frames.last_mut().unwrap();
                                frame.pc = pc;
                                frame.values = values;
                                frame.bc_ptr = bc_ptr;
                            }

                            unsafe {
                                vm.engine.note_interpreted_entry(target_func_id);
                                let callee_module = vm
                                    .engine
                                    .func_module(target_func_id)
                                    .cloned()
                                    .unwrap_or_else(|| Rc::clone(&module_name));
                                (*fiber).mir_frames.push(MirCallFrame {
                                    func_id: target_func_id,
                                    current_block: BlockId(0),
                                    ip: 0,
                                    pc: 0,
                                    values: new_values,
                                    module_name: callee_module,
                                    return_dst: Some(ValueId(dst as u32)),
                                    closure: Some(closure_ptr),
                                    defining_class,
                                    bc_ptr: target_bc_ptr,
                                });
                            }

                            continue 'fiber_loop;
                        }
                        None => {
                            let method_name = vm.interner.resolve(method).to_string();
                            let class_name = vm.class_name_of(recv_val);
                            // Save `pc` into the frame before bailing so
                            // `extract_error_location` reads the actual
                            // failing-call PC. Without this the frame
                            // keeps whatever pc was last persisted by
                            // a successful dispatch — which can be far
                            // past the failing call (often in a later
                            // statement), and the diagnostic span lands
                            // on a comment several lines down.
                            unsafe {
                                let frame = (*fiber).mir_frames.last_mut().unwrap();
                                frame.pc = pc;
                            }
                            // Honour `Fiber.try` for method-not-found
                            // errors the same way the native-error path
                            // already does.
                            let msg =
                                format!("{} does not implement '{}'", class_name, method_name);
                            if let Some(err_val) =
                                unsafe { route_method_error_through_fiber_try(vm, fiber, msg) }
                            {
                                let caller_resumed = vm.fiber != fiber;
                                if caller_resumed {
                                    continue 'fiber_loop;
                                }
                                return Ok(err_val);
                            }
                            return Err(RuntimeError::MethodNotFound {
                                class_name,
                                method: method_name,
                            });
                        }
                    }
                }

                Op::CallStaticSelf => {
                    let dst = read_u16(code, &mut pc);
                    let argc = read_u8(code, &mut pc) as usize;

                    let (closure_ptr, defining_class) = unsafe {
                        let frame = (*fiber).mir_frames.last().unwrap();
                        (frame.closure, frame.defining_class)
                    };
                    let closure_ptr = closure_ptr.ok_or_else(|| {
                        RuntimeError::Error("static self call outside closure frame".into())
                    })?;
                    let defining_class = defining_class.ok_or_else(|| {
                        RuntimeError::Error("static self call without defining class".into())
                    })?;

                    let mut arg_vals: SmallVec<[Value; 8]> = SmallVec::with_capacity(argc + 1);
                    arg_vals.push(Value::object(defining_class as *mut u8));
                    for _ in 0..argc {
                        let arg_reg = read_u16(code, &mut pc);
                        arg_vals.push(get_reg(&values, arg_reg));
                    }

                    dispatch_closure_bc(
                        vm,
                        fiber,
                        closure_ptr,
                        &arg_vals,
                        pc,
                        values,
                        &module_name,
                        ValueId(dst as u32),
                        Some(defining_class),
                        bc_ptr,
                    )?;
                    continue 'fiber_loop;
                }

                Op::SuperCall => {
                    let dst = read_u16(code, &mut pc);
                    let method_idx = read_u16(code, &mut pc);
                    let argc = read_u8(code, &mut pc) as usize;
                    let method = SymbolId::from_raw(method_idx as u32);

                    let mut arg_vals: SmallVec<[Value; 8]> = SmallVec::with_capacity(argc);
                    for _ in 0..argc {
                        let arg_reg = read_u16(code, &mut pc);
                        arg_vals.push(get_reg(&values, arg_reg));
                    }

                    let defining = unsafe { (*fiber).mir_frames.last().unwrap().defining_class };
                    let superclass = defining.and_then(|cls| {
                        let sup = unsafe { (*cls).superclass };
                        if sup.is_null() {
                            None
                        } else {
                            Some(sup)
                        }
                    });

                    if let Some(super_cls) = superclass {
                        let found = unsafe { find_method_with_class(super_cls, method) };
                        let (method_entry, found_class) = match found {
                            Some((m, c)) => (Some(m), Some(c)),
                            None => {
                                let method_name = vm.interner.resolve(method).to_string();
                                let static_name = format!("static:{}", method_name);
                                let static_sym = vm.interner.intern(&static_name);
                                match unsafe { find_method_with_class(super_cls, static_sym) } {
                                    Some((m, c)) => (Some(m), Some(c)),
                                    None => (None, None),
                                }
                            }
                        };
                        match method_entry {
                            Some(m @ (Method::Native(_) | Method::ForeignC(_) | Method::ForeignCDynamic(_))) => {
                                let result = match m {
                                    Method::Native(func) => call_native_with_frame_sync(
                                        vm,
                                        fiber,
                                        pc,
                                        &mut values,
                                        func,
                                        &arg_vals,
                                    ),
                                    Method::ForeignC(func) => call_foreign_c_with_frame_sync(
                                        vm,
                                        fiber,
                                        pc,
                                        &mut values,
                                        func,
                                        &arg_vals,
                                    ),
                                    Method::ForeignCDynamic(idx) => call_foreign_dynamic_with_frame_sync(
                                        vm,
                                        fiber,
                                        pc,
                                        &mut values,
                                        idx,
                                        &arg_vals,
                                    ),
                                    _ => unreachable!(),
                                };
                                set_reg(&mut values, dst, result);
                            }
                            Some(Method::Closure(closure_ptr)) => {
                                dispatch_closure_bc(
                                    vm,
                                    fiber,
                                    closure_ptr,
                                    &arg_vals,
                                    pc,
                                    values,
                                    &module_name,
                                    ValueId(dst as u32),
                                    found_class,
                                    bc_ptr,
                                )?;
                                continue 'fiber_loop;
                            }
                            Some(Method::Constructor(closure_ptr)) => {
                                dispatch_closure_bc(
                                    vm,
                                    fiber,
                                    closure_ptr,
                                    &arg_vals,
                                    pc,
                                    values,
                                    &module_name,
                                    ValueId(dst as u32),
                                    found_class,
                                    bc_ptr,
                                )?;
                                continue 'fiber_loop;
                            }
                            None => {
                                let method_name = vm.interner.resolve(method).to_string();
                                let class_name =
                                    unsafe { vm.interner.resolve((*super_cls).name).to_string() };
                                // Same `frame.pc = pc` save as the
                                // regular Op::Call site above — without
                                // it the diagnostic span lands on
                                // whatever statement was last
                                // persisted, not the failing super call.
                                unsafe {
                                    let frame = (*fiber).mir_frames.last_mut().unwrap();
                                    frame.pc = pc;
                                }
                                let msg =
                                    format!("{} does not implement '{}'", class_name, method_name);
                                if let Some(err_val) =
                                    unsafe { route_method_error_through_fiber_try(vm, fiber, msg) }
                                {
                                    let caller_resumed = vm.fiber != fiber;
                                    if caller_resumed {
                                        continue 'fiber_loop;
                                    }
                                    return Ok(err_val);
                                }
                                return Err(RuntimeError::MethodNotFound {
                                    class_name,
                                    method: method_name,
                                });
                            }
                        }
                    } else {
                        return Err(RuntimeError::Error(
                            "super called with no superclass".into(),
                        ));
                    }
                }

                // =============================================================
                // Subscripts
                // =============================================================
                Op::SubscriptGet => {
                    let dst = read_u16(code, &mut pc);
                    let recv_reg = read_u16(code, &mut pc);
                    let argc = read_u8(code, &mut pc) as usize;
                    let recv = get_reg(&values, recv_reg);
                    let mut all_args: SmallVec<[Value; 4]> = SmallVec::with_capacity(1 + argc);
                    all_args.push(recv);
                    for _ in 0..argc {
                        let arg_reg = read_u16(code, &mut pc);
                        all_args.push(get_reg(&values, arg_reg));
                    }
                    let class = vm.class_of(recv);
                    let sym = subscript_get_sym(vm, argc);
                    match unsafe { (*class).find_method(sym).cloned() } {
                        Some(m @ (Method::Native(_) | Method::ForeignC(_) | Method::ForeignCDynamic(_))) => {
                            let result = match m {
                                Method::Native(func) => call_native_with_frame_sync(
                                    vm,
                                    fiber,
                                    pc,
                                    &mut values,
                                    func,
                                    &all_args,
                                ),
                                Method::ForeignC(func) => call_foreign_c_with_frame_sync(
                                    vm,
                                    fiber,
                                    pc,
                                    &mut values,
                                    func,
                                    &all_args,
                                ),
                                Method::ForeignCDynamic(idx) => call_foreign_dynamic_with_frame_sync(
                                    vm,
                                    fiber,
                                    pc,
                                    &mut values,
                                    idx,
                                    &all_args,
                                ),
                                _ => unreachable!(),
                            };
                            set_reg(&mut values, dst, result);
                        }
                        Some(Method::Closure(closure_ptr)) => {
                            dispatch_closure_bc(
                                vm,
                                fiber,
                                closure_ptr,
                                &all_args,
                                pc,
                                values,
                                &module_name,
                                ValueId(dst as u32),
                                None,
                                bc_ptr,
                            )?;
                            continue 'fiber_loop;
                        }
                        _ => {
                            return Err(RuntimeError::Unsupported(format!(
                                "subscript get with arity {}",
                                argc
                            )));
                        }
                    }
                }
                Op::SubscriptSet => {
                    let dst = read_u16(code, &mut pc);
                    let recv_reg = read_u16(code, &mut pc);
                    let argc = read_u8(code, &mut pc) as usize;
                    let recv = get_reg(&values, recv_reg);
                    let mut all_args: SmallVec<[Value; 4]> = SmallVec::with_capacity(2 + argc);
                    all_args.push(recv);
                    for _ in 0..argc {
                        let arg_reg = read_u16(code, &mut pc);
                        all_args.push(get_reg(&values, arg_reg));
                    }
                    let val_reg = read_u16(code, &mut pc);
                    all_args.push(get_reg(&values, val_reg));
                    let class = vm.class_of(recv);
                    let sym = subscript_set_sym(vm, argc);
                    match unsafe { (*class).find_method(sym).cloned() } {
                        Some(m @ (Method::Native(_) | Method::ForeignC(_) | Method::ForeignCDynamic(_))) => {
                            let result = match m {
                                Method::Native(func) => call_native_with_frame_sync(
                                    vm,
                                    fiber,
                                    pc,
                                    &mut values,
                                    func,
                                    &all_args,
                                ),
                                Method::ForeignC(func) => call_foreign_c_with_frame_sync(
                                    vm,
                                    fiber,
                                    pc,
                                    &mut values,
                                    func,
                                    &all_args,
                                ),
                                Method::ForeignCDynamic(idx) => call_foreign_dynamic_with_frame_sync(
                                    vm,
                                    fiber,
                                    pc,
                                    &mut values,
                                    idx,
                                    &all_args,
                                ),
                                _ => unreachable!(),
                            };
                            set_reg(&mut values, dst, result);
                        }
                        Some(Method::Closure(closure_ptr)) => {
                            dispatch_closure_bc(
                                vm,
                                fiber,
                                closure_ptr,
                                &all_args,
                                pc,
                                values,
                                &module_name,
                                ValueId(dst as u32),
                                None,
                                bc_ptr,
                            )?;
                            continue 'fiber_loop;
                        }
                        _ => {
                            return Err(RuntimeError::Unsupported(format!(
                                "subscript set with arity {}",
                                argc
                            )));
                        }
                    }
                }

                // =============================================================
                // Collections
                // =============================================================
                Op::MakeList => {
                    let dst = read_u16(code, &mut pc);
                    let count = read_u8(code, &mut pc) as usize;
                    let mut elements = Vec::with_capacity(count);
                    for _ in 0..count {
                        let reg = read_u16(code, &mut pc);
                        elements.push(get_reg(&values, reg));
                    }
                    // `vm.new_list` allocates and may trigger GC.
                    // Publish the caller's register file to its frame
                    // so every still-live pointer in `values` gets
                    // traced through `mir_frames`. Without this,
                    // generational promote moves objects behind the
                    // local Vec — the next op then dereferences a
                    // stale pointer and surfaces as "Null does not
                    // implement X" on a wrapper that's actually
                    // still alive at its post-promote address.
                    unsafe {
                        let frame = (*fiber).mir_frames.last_mut().unwrap();
                        frame.pc = pc;
                        frame.values = std::mem::take(&mut values);
                    }
                    let result = vm.new_list(elements);
                    unsafe {
                        values =
                            std::mem::take(&mut (*fiber).mir_frames.last_mut().unwrap().values);
                    }
                    set_reg(&mut values, dst, result);
                }
                Op::MakeMap => {
                    let dst = read_u16(code, &mut pc);
                    let count = read_u8(code, &mut pc) as usize;
                    let mut entries = Vec::with_capacity(count);
                    let root_len_before = crate::codegen::runtime_fns::jit_roots_snapshot_len();
                    for _ in 0..count {
                        let k_reg = read_u16(code, &mut pc);
                        let v_reg = read_u16(code, &mut pc);
                        let key = get_reg(&values, k_reg);
                        let val = get_reg(&values, v_reg);
                        crate::codegen::runtime_fns::push_jit_root(key);
                        crate::codegen::runtime_fns::push_jit_root(val);
                        entries.push(());
                    }
                    // Publish the caller's register file before any
                    // allocator path runs — see Op::MakeList. The
                    // key/value JIT-roots above protect the new
                    // map's own contents; only the surrounding
                    // local Vec needs the frame.values dance.
                    unsafe {
                        let frame = (*fiber).mir_frames.last_mut().unwrap();
                        frame.pc = pc;
                        frame.values = std::mem::take(&mut values);
                    }
                    let map_val = vm.new_map();
                    crate::codegen::runtime_fns::push_jit_root(map_val);
                    if count > 0 {
                        let map_ptr =
                            map_val.as_object().unwrap() as *mut crate::runtime::object::ObjMap;
                        for i in 0..entries.len() {
                            let key =
                                crate::codegen::runtime_fns::jit_root_at(root_len_before + i * 2);
                            let val = crate::codegen::runtime_fns::jit_root_at(
                                root_len_before + i * 2 + 1,
                            );
                            unsafe { (*map_ptr).set(key, val) };
                        }
                    }
                    let map_val = crate::codegen::runtime_fns::jit_root_at(
                        root_len_before + entries.len() * 2,
                    );
                    crate::codegen::runtime_fns::jit_roots_restore_len(root_len_before);
                    unsafe {
                        values =
                            std::mem::take(&mut (*fiber).mir_frames.last_mut().unwrap().values);
                    }
                    set_reg(&mut values, dst, map_val);
                }
                Op::StringConcat => {
                    let dst = read_u16(code, &mut pc);
                    let count = read_u8(code, &mut pc) as usize;
                    let mut result = String::new();
                    for _ in 0..count {
                        let reg = read_u16(code, &mut pc);
                        result.push_str(&value_to_string(vm, get_reg(&values, reg)));
                    }
                    // Same rationale as MakeList — `vm.new_string` allocates.
                    unsafe {
                        let frame = (*fiber).mir_frames.last_mut().unwrap();
                        frame.pc = pc;
                        frame.values = std::mem::take(&mut values);
                    }
                    let result_val = vm.new_string(result);
                    unsafe {
                        values =
                            std::mem::take(&mut (*fiber).mir_frames.last_mut().unwrap().values);
                    }
                    set_reg(&mut values, dst, result_val);
                }

                // =============================================================
                // Closures
                // =============================================================
                Op::MakeClosure => {
                    let dst = read_u16(code, &mut pc);
                    let fn_id = read_u32(code, &mut pc);
                    let uv_count = read_u8(code, &mut pc) as usize;

                    let closure_name = vm.interner.intern("<closure>");
                    let arity = vm
                        .engine
                        .get_mir(FuncId(fn_id))
                        .map(|mir| mir.arity)
                        .unwrap_or(0);
                    let fn_ptr = vm.gc.alloc_fn(closure_name, arity, uv_count as u16, fn_id);
                    unsafe {
                        (*fn_ptr).header.class = vm.fn_class;
                        (*fn_ptr).trivial_getter_field = vm
                            .engine
                            .trivial_getter_fields
                            .get(fn_id as usize)
                            .copied()
                            .flatten()
                            .unwrap_or(u16::MAX);
                        (*fn_ptr).trivial_setter_field = vm
                            .engine
                            .trivial_setter_fields
                            .get(fn_id as usize)
                            .copied()
                            .flatten()
                            .unwrap_or(u16::MAX);
                    }
                    let closure_ptr = vm.gc.alloc_closure(fn_ptr);
                    unsafe {
                        (*closure_ptr).header.class = vm.fn_class;
                    }

                    for i in 0..uv_count {
                        let uv_reg = read_u16(code, &mut pc);
                        let captured_val = get_reg(&values, uv_reg);
                        let uv_obj = vm.gc.alloc_upvalue(std::ptr::null_mut());
                        unsafe {
                            (*uv_obj).closed = captured_val;
                            (*uv_obj).location = &mut (*uv_obj).closed as *mut Value;
                            if i < (*closure_ptr).upvalues.len() {
                                (&mut (*closure_ptr).upvalues)[i] = uv_obj;
                            }
                        }
                    }

                    set_reg(&mut values, dst, Value::object(closure_ptr as *mut u8));
                }

                // =============================================================
                // Terminators
                // =============================================================
                Op::Return => {
                    let src = read_u16(code, &mut pc);
                    let return_val = get_reg(&values, src);
                    let return_dst = unsafe { (*fiber).mir_frames.last().unwrap().return_dst };
                    unsafe { (*fiber).mir_frames.pop() };
                    // Recycle the callee's register file
                    values.clear();
                    if vm.register_pool.len() < 128 {
                        vm.register_pool.push(values);
                    }
                    let remaining_depth = unsafe { (*fiber).mir_frames.len() };
                    if fiber == stop_fiber && stop_depth == Some(remaining_depth) {
                        return Ok(return_val);
                    }
                    if remaining_depth == 0 {
                        unsafe { (*fiber).state = FiberState::Done };
                        let caller = unsafe { (*fiber).caller };
                        if !caller.is_null() {
                            unsafe { (*fiber).caller = std::ptr::null_mut() };
                            resume_caller(vm, caller, return_val);
                            continue 'fiber_loop;
                        }
                        return Ok(return_val);
                    }
                    // Write return value into the caller's slot, then
                    // re-enter the fiber loop so all locals refresh
                    // from the (now-top) caller frame — including
                    // `module_name` / `module_vars_ptr`, which may
                    // differ when the callee lived in another module.
                    unsafe {
                        let frame = (*fiber).mir_frames.last_mut().unwrap();
                        if let Some(dst) = return_dst {
                            set_reg(&mut frame.values, dst.0 as u16, return_val);
                        }
                        if frame.bc_ptr.is_null() {
                            frame.bc_ptr = vm
                                .engine
                                .ensure_bytecode(frame.func_id)
                                .ok_or_else(|| RuntimeError::Error("invalid func_id".into()))?;
                        }
                    }
                    continue 'fiber_loop;
                }
                Op::ReturnNull => {
                    let return_dst = unsafe { (*fiber).mir_frames.last().unwrap().return_dst };
                    unsafe { (*fiber).mir_frames.pop() };
                    // Recycle the callee's register file
                    values.clear();
                    if vm.register_pool.len() < 128 {
                        vm.register_pool.push(values);
                    }
                    let remaining_depth = unsafe { (*fiber).mir_frames.len() };
                    if fiber == stop_fiber && stop_depth == Some(remaining_depth) {
                        return Ok(Value::null());
                    }
                    if remaining_depth == 0 {
                        unsafe { (*fiber).state = FiberState::Done };
                        let caller = unsafe { (*fiber).caller };
                        if !caller.is_null() {
                            unsafe { (*fiber).caller = std::ptr::null_mut() };
                            resume_caller(vm, caller, Value::null());
                            continue 'fiber_loop;
                        }
                        return Ok(Value::null());
                    }
                    // Write null into the caller's slot, then
                    // re-enter the fiber loop (see Op::Return for the
                    // module-refresh rationale).
                    unsafe {
                        let frame = (*fiber).mir_frames.last_mut().unwrap();
                        if let Some(dst) = return_dst {
                            set_reg(&mut frame.values, dst.0 as u16, Value::null());
                        }
                        if frame.bc_ptr.is_null() {
                            frame.bc_ptr = vm
                                .engine
                                .ensure_bytecode(frame.func_id)
                                .ok_or_else(|| RuntimeError::Error("invalid func_id".into()))?;
                        }
                    }
                    continue 'fiber_loop;
                }
                Op::Unreachable => {
                    return Err(RuntimeError::Unreachable);
                }
                Op::Branch => {
                    let branch_offset = pc - 1;
                    let target = read_u32(code, &mut pc);
                    let argc = read_u8(code, &mut pc) as usize;
                    // Block-param bindings are a PARALLEL assignment:
                    // every `dst = src` logically happens "at once."
                    // Read all sources before writing any destinations
                    // so cyclic patterns (a = b, b = a) work —
                    // especially on loop back-edges where argv-style
                    // phis feed their own slot through mutated
                    // neighbours.
                    let mut bindings: [(u16, Value); 16] = [(0, Value::null()); 16];
                    let mut heap_bindings: Vec<(u16, Value)> = Vec::new();
                    let use_heap = argc > bindings.len();
                    for item in bindings.iter_mut().take(argc) {
                        let dst_reg = read_u16(code, &mut pc);
                        let src_reg = read_u16(code, &mut pc);
                        let val = get_reg(&values, src_reg);
                        if use_heap {
                            heap_bindings.push((dst_reg, val));
                        } else {
                            *item = (dst_reg, val);
                        }
                    }
                    if use_heap {
                        for &(dst_reg, val) in &heap_bindings {
                            set_reg(&mut values, dst_reg, val);
                        }
                    } else {
                        for &(dst_reg, val) in &bindings[..argc] {
                            set_reg(&mut values, dst_reg, val);
                        }
                    }
                    // Back-edge detection: if jumping backward, this is a
                    // loop iteration. Trigger JIT compilation after enough
                    // iterations so hot loops in infrequently-called functions
                    // (like the script's main function) still get compiled.
                    // Also drain the compile queue here — this is a safe point
                    // between iterations (not mid-call-dispatch).
                    if target < branch_offset && vm.engine.mode == ExecutionMode::Tiered {
                        // Sample tier-up polling every 64 back-edges to keep
                        // the hot loop fast. We still bump call_count on every
                        // back-edge (cheap inline increment), but the expensive
                        // pending-compile polling only runs every 64 iterations.
                        backedge_counter = backedge_counter.wrapping_add(1);
                        let should_tier_up = vm.engine.record_call(func_id);
                        if std::env::var_os("WLIFT_OSR_TRACE").is_some()
                            && (backedge_counter == 1 || should_tier_up)
                        {
                            let name = vm
                                .engine
                                .get_mir(func_id)
                                .map(|mir| vm.interner.resolve(mir.name).to_string())
                                .unwrap_or_else(|| "<unknown>".to_string());
                            eprintln!(
                                "osr-trace: backedge FuncId({}) {} count={} tier_up={}",
                                func_id.0, name, backedge_counter, should_tier_up
                            );
                        }
                        if should_tier_up {
                            vm.engine.request_tier_up(func_id, &vm.interner);
                        }
                        if should_tier_up
                            || vm.engine.has_pending_compilations()
                            || (backedge_counter & 63) == 0
                        {
                            if vm.engine.has_pending_compilations() {
                                // Save frame state before draining.
                                unsafe {
                                    if let Some(frame) = (*fiber).mir_frames.last_mut() {
                                        frame.values = std::mem::take(&mut values);
                                        frame.pc = pc;
                                    }
                                }
                                vm.engine.poll_compilations();
                                vm.engine.drain_compile_queue(&vm.interner);
                                unsafe {
                                    if let Some(frame) = (*fiber).mir_frames.last_mut() {
                                        values = std::mem::take(&mut frame.values);
                                    }
                                }
                            }
                            // Read current-frame context so inline-push call
                            // sites don't leave stale closure/class/module for
                            // the OSR transfer.
                            let (cur_module_name, cur_closure, cur_defining_class, cur_return_dst) = unsafe {
                                let frame = (*fiber).mir_frames.last().unwrap();
                                (
                                    Rc::clone(&frame.module_name),
                                    frame.closure,
                                    frame.defining_class,
                                    frame.return_dst,
                                )
                            };
                            match try_enter_loop_osr(
                                vm,
                                fiber,
                                func_id,
                                &cur_module_name,
                                cur_closure,
                                cur_defining_class,
                                cur_return_dst,
                                bc,
                                branch_offset,
                                target,
                                &mut values,
                                stop_depth,
                            )? {
                                OsrTransfer::NotEntered => {}
                                OsrTransfer::ContinueFiberLoop => continue 'fiber_loop,
                                OsrTransfer::Return(value) => return Ok(value),
                            }
                        }
                    }
                    pc = target;
                }
                Op::CondBranch => {
                    let branch_offset = pc - 1;
                    let cond_reg = read_u16(code, &mut pc);
                    let cond = get_reg(&values, cond_reg);

                    // Read true branch
                    let true_off = read_u32(code, &mut pc);
                    let t_argc = read_u8(code, &mut pc) as usize;
                    // Save position and skip true params
                    let true_params_start = pc;
                    pc += (t_argc * 4) as u32; // each param pair is 4 bytes (2+2)

                    // Read false branch
                    let false_off = read_u32(code, &mut pc);
                    let f_argc = read_u8(code, &mut pc) as usize;
                    let false_params_start = pc;
                    let branch_end = false_params_start + (f_argc * 4) as u32;
                    // Don't need to advance pc — both branches jump to target offset

                    let target = if cond.is_truthy_wren() {
                        // Parallel-assign true-branch bindings (read all
                        // sources before any write — see Op::Branch).
                        let mut p = true_params_start;
                        let mut bindings: [(u16, Value); 16] = [(0, Value::null()); 16];
                        let mut heap_bindings: Vec<(u16, Value)> = Vec::new();
                        let use_heap = t_argc > bindings.len();
                        for item in bindings.iter_mut().take(t_argc) {
                            let dst_reg = read_u16(code, &mut p);
                            let src_reg = read_u16(code, &mut p);
                            let val = get_reg(&values, src_reg);
                            if use_heap {
                                heap_bindings.push((dst_reg, val));
                            } else {
                                *item = (dst_reg, val);
                            }
                        }
                        if use_heap {
                            for &(dst_reg, val) in &heap_bindings {
                                set_reg(&mut values, dst_reg, val);
                            }
                        } else {
                            for &(dst_reg, val) in &bindings[..t_argc] {
                                set_reg(&mut values, dst_reg, val);
                            }
                        }
                        true_off
                    } else {
                        // Same parallel-assign for the false path.
                        let mut p = false_params_start;
                        let mut bindings: [(u16, Value); 16] = [(0, Value::null()); 16];
                        let mut heap_bindings: Vec<(u16, Value)> = Vec::new();
                        let use_heap = f_argc > bindings.len();
                        for item in bindings.iter_mut().take(f_argc) {
                            let dst_reg = read_u16(code, &mut p);
                            let src_reg = read_u16(code, &mut p);
                            let val = get_reg(&values, src_reg);
                            if use_heap {
                                heap_bindings.push((dst_reg, val));
                            } else {
                                *item = (dst_reg, val);
                            }
                        }
                        if use_heap {
                            for &(dst_reg, val) in &heap_bindings {
                                set_reg(&mut values, dst_reg, val);
                            }
                        } else {
                            for &(dst_reg, val) in &bindings[..f_argc] {
                                set_reg(&mut values, dst_reg, val);
                            }
                        }
                        false_off
                    };

                    if target < branch_offset && vm.engine.mode == ExecutionMode::Tiered {
                        backedge_counter = backedge_counter.wrapping_add(1);
                        let should_tier_up = vm.engine.record_call(func_id);
                        if std::env::var_os("WLIFT_OSR_TRACE").is_some()
                            && (backedge_counter == 1 || should_tier_up)
                        {
                            let name = vm
                                .engine
                                .get_mir(func_id)
                                .map(|mir| vm.interner.resolve(mir.name).to_string())
                                .unwrap_or_else(|| "<unknown>".to_string());
                            eprintln!(
                                "osr-trace: cond-backedge FuncId({}) {} count={} tier_up={}",
                                func_id.0, name, backedge_counter, should_tier_up
                            );
                        }
                        if should_tier_up {
                            vm.engine.request_tier_up(func_id, &vm.interner);
                        }
                        if should_tier_up
                            || vm.engine.has_pending_compilations()
                            || (backedge_counter & 63) == 0
                        {
                            if vm.engine.has_pending_compilations() {
                                unsafe {
                                    if let Some(frame) = (*fiber).mir_frames.last_mut() {
                                        frame.values = std::mem::take(&mut values);
                                        frame.pc = branch_end;
                                    }
                                }
                                vm.engine.poll_compilations();
                                vm.engine.drain_compile_queue(&vm.interner);
                                unsafe {
                                    if let Some(frame) = (*fiber).mir_frames.last_mut() {
                                        values = std::mem::take(&mut frame.values);
                                    }
                                }
                            }
                            let (cur_module_name, cur_closure, cur_defining_class, cur_return_dst) = unsafe {
                                let frame = (*fiber).mir_frames.last().unwrap();
                                (
                                    Rc::clone(&frame.module_name),
                                    frame.closure,
                                    frame.defining_class,
                                    frame.return_dst,
                                )
                            };
                            match try_enter_loop_osr(
                                vm,
                                fiber,
                                func_id,
                                &cur_module_name,
                                cur_closure,
                                cur_defining_class,
                                cur_return_dst,
                                bc,
                                branch_offset,
                                target,
                                &mut values,
                                stop_depth,
                            )? {
                                OsrTransfer::NotEntered => {}
                                OsrTransfer::ContinueFiberLoop => continue 'fiber_loop,
                                OsrTransfer::Return(value) => return Ok(value),
                            }
                        }
                    }
                    pc = target;
                }
            } // match op
        } // inner dispatch loop
    } // fiber_loop
}

// ---------------------------------------------------------------------------
// Backward-compatible wrapper
// ---------------------------------------------------------------------------

/// Execute a MIR function using a temporary fiber.
pub fn eval_in_vm(
    vm: &mut VM,
    func: &crate::mir::MirFunction,
    module_vars: &mut Vec<Value>,
) -> Result<Value, RuntimeError> {
    if func.blocks.is_empty() {
        return Ok(Value::null());
    }

    let func_id = vm.engine.register_function(func.clone());

    let module_name = "__eval__".to_string();
    vm.engine.modules.insert(
        module_name.clone(),
        crate::runtime::engine::ModuleEntry {
            top_level: func_id,
            vars: std::mem::take(module_vars),
            var_names: Vec::new(),
        },
    );

    // Get bytecode to determine register count
    let reg_count = vm
        .engine
        .get_bytecode(func_id)
        .map(|bc| bc.register_count as usize)
        .unwrap_or(func.next_value as usize);

    let module_name_rc = Rc::new(module_name.clone());
    let fiber = vm.gc.alloc_fiber();
    unsafe {
        (*fiber).header.class = vm.fiber_class;
        // Pre-bind entry block params by scanning bytecode
        let reg_file = vec![UNDEF; reg_count];
        (*fiber).mir_frames.push(MirCallFrame {
            func_id,
            current_block: BlockId(0),
            ip: 0,
            pc: 0,
            values: reg_file,
            module_name: module_name_rc,
            return_dst: None,
            closure: None,
            defining_class: None,
            bc_ptr: std::ptr::null(),
        });
    }

    let prev_fiber = vm.fiber;
    vm.fiber = fiber;

    let result = run_fiber(vm);

    vm.fiber = prev_fiber;

    if let Some(entry) = vm.engine.modules.get(&module_name) {
        *module_vars = entry.vars.clone();
    }

    result
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Create a register file pre-sized for a MIR function.
#[inline]
pub fn new_register_file(mir: &crate::mir::MirFunction) -> Vec<Value> {
    vec![UNDEF; mir.next_value as usize]
}

/// Fill a recycled register file with UNDEF up to `count` slots.
#[inline(always)]
fn new_values_fill_undef(values: &mut [Value], count: usize) {
    for v in values[..count].iter_mut() {
        *v = UNDEF;
    }
}

/// Run the active fiber (vm.fiber) until it finishes, yields, or errors.
///
/// The fiber must have at least one MirCallFrame pushed before calling this.
/// Uses flat bytecode dispatch for efficient interpretation.
pub fn run_fiber(vm: &mut VM) -> Result<Value, RuntimeError> {
    // On wasm32, install the VM as the current thread-local so
    // JIT'd helper calls (`wren_call`, `wren_make_*`, etc.) can
    // find it without a VM ref through the wasm ABI. Save/
    // restore handles re-entrance — the scheduler can resume a
    // parked fiber via `run_fiber` while another `run_fiber` is
    // already on the call stack, and each level needs to see
    // the right VM (always the same one in single-threaded
    // wasm, but the discipline keeps the contract clear).
    //
    // Host has Cranelift-driven calling convention that threads
    // VM through stack args; no thread-local needed.
    #[cfg(target_arch = "wasm32")]
    let prev_vm = crate::runtime::tier::enter_vm(vm as *mut VM);
    let result = run_fiber_with_stop_depth(vm, None);
    #[cfg(target_arch = "wasm32")]
    crate::runtime::tier::exit_vm(prev_vm);
    result
}

/// Run the active fiber until its frame stack shrinks back to `stop_depth`.
///
/// Used for synchronous native->interpreter calls that push one callee frame
/// onto the current fiber and must stop before resuming the native caller.
pub fn run_fiber_until_depth(vm: &mut VM, stop_depth: usize) -> Result<Value, RuntimeError> {
    run_fiber_with_stop_depth(vm, Some(stop_depth))
}

/// Dispatch a closure call in bytecode mode: save current frame, push new frame.
#[allow(clippy::too_many_arguments)]
fn dispatch_closure_bc(
    vm: &mut VM,
    fiber: *mut ObjFiber,
    closure_ptr: *mut ObjClosure,
    arg_vals: &[Value],
    pc: u32,
    values: Vec<Value>,
    module_name: &Rc<String>,
    return_dst: ValueId,
    defining_class: Option<*mut ObjClass>,
    caller_bc_ptr: *const BytecodeFunction,
) -> Result<(), RuntimeError> {
    dispatch_closure_bc_inner(
        vm,
        fiber,
        closure_ptr,
        arg_vals,
        pc,
        values,
        module_name,
        return_dst,
        defining_class,
        caller_bc_ptr,
        true,
    )
}

/// Inner dispatch with `allow_threaded` flag. When false (called from
/// try_operator_dispatch), skip the threaded path because the bytecode
/// loop's `continue 'fiber_loop` expects a frame push/pop cycle.
#[allow(clippy::too_many_arguments)] // Closure dispatch fans out caller state; wrapping in a struct would churn every callsite.
fn dispatch_closure_bc_inner(
    vm: &mut VM,
    fiber: *mut ObjFiber,
    closure_ptr: *mut ObjClosure,
    arg_vals: &[Value],
    pc: u32,
    mut values: Vec<Value>,
    module_name: &Rc<String>,
    return_dst: ValueId,
    defining_class: Option<*mut ObjClass>,
    caller_bc_ptr: *const BytecodeFunction,
    allow_threaded: bool,
) -> Result<(), RuntimeError> {
    let fn_ptr = unsafe { (*closure_ptr).function };
    let target_func_id = FuncId(unsafe { (*fn_ptr).fn_id });
    let fn_idx = target_func_id.0 as usize;

    // Wasm tier-up dispatch + trigger. On wasm builds, hot Wren
    // methods get compiled to a per-function wasm module
    // (see `wlift_wasm::tier_up`); the resulting `WebAssembly`
    // function reference lives in the `wasm_jit_slots` table.
    //
    //   1. If a slot's already set, dispatch via the host's JS
    //      shim instead of running bytecode.
    //   2. Otherwise, bump the call counter; on threshold, ask
    //      the runtime's tier broker (`runtime::tier::try_compile`)
    //      to compile + install. The very next call hits path 1.
    //
    // Capped at args.len() <= 3 (receiver + 2 positional) today
    // because the JS call shims only support up to 2 args.
    // Phase 4 swaps `tier::dispatch` for a shared funcref table
    // + `call_indirect`, eliminating the per-arity cap and the
    // JS-roundtrip cost on inter-fn calls.
    #[cfg(target_arch = "wasm32")]
    if arg_vals.len() <= 2 {
        crate::runtime::tier::bump_dispatch_hook_hits();
        if let Some(slot) = vm.engine.wasm_jit_slot(target_func_id) {
            // `arg_vals` in this function is the caller's
            // `&caller_arg_vals[1..]` — receiver already
            // stripped at the call site (line ~2021 in
            // `Op::Call`). So `arg_vals` is exactly the
            // positional args; pass through as-is.
            let args_bits: Vec<u64> = arg_vals.iter().map(|v| v.to_bits()).collect();
            // Export name only matters for Phase 4's table
            // multiplexing; per-instance dispatch ignores it.
            let result_bits = crate::runtime::tier::dispatch(slot, "", &args_bits);
            let result_val = Value::from_bits(result_bits);
            set_reg(&mut values, return_dst.0 as u16, result_val);
            unsafe {
                if let Some(frame) = (*fiber).mir_frames.last_mut() {
                    frame.pc = pc;
                    frame.values = values;
                }
            }
            return Ok(());
        }
        // Not yet JIT'd — bump counter and try compile on
        // threshold crossing. We only do this on the *crossing*
        // (count == THRESHOLD) so a function that fails to
        // compile doesn't retry-spam every subsequent call.
        let count = vm.engine.bump_wasm_call_count(target_func_id);
        if count == crate::runtime::engine::WASM_JIT_THRESHOLD {
            if let Some(mir_arc) = vm.engine.get_mir(target_func_id) {
                if let Some(slot) = crate::runtime::tier::try_compile(&mir_arc) {
                    vm.engine.install_wasm_jit_slot(target_func_id, slot);
                    // Don't dispatch this call through the JIT —
                    // the BC interpreter has already set up
                    // partial state. Fall through to the BC path
                    // and let the *next* invocation of this
                    // function take the JIT path.
                }
            }
        }
    }

    // Fast JIT dispatch: check the Vec-indexed jit_code array (O(1) lookup).
    // Skip tier-up profiling entirely for already-compiled functions.
    if vm.engine.mode != ExecutionMode::Interpreter
        && arg_vals.len() <= 4
        && !crate::codegen::runtime_fns::jit_disabled()
    {
        let native_fn_ptr = vm
            .engine
            .jit_code
            .get(fn_idx)
            .copied()
            .unwrap_or(std::ptr::null());

        // Always sample argument types for profile-guided specialization,
        // even after baseline compilation. The optimized tier needs this
        // profile data to insert GuardNum guards for TypeSpecialize.
        vm.engine.sample_arg_types(target_func_id, arg_vals);
        if native_fn_ptr.is_null() {
            // Not yet compiled: do tier-up profiling.
            let should_tier_up = vm.engine.record_call(target_func_id);
            if should_tier_up {
                vm.engine.request_tier_up(target_func_id, &vm.interner);
            }
        }
        // Always poll — and always re-read jit_code AFTER poll to avoid
        // use-after-free when poll_compilations replaces a previously-compiled
        // function with a recompiled version (dropping the old ExecutableBuffer).
        vm.engine.poll_compilations();
        // Drain compile queue: dispatch_closure_bc is a safe point because
        // caller state has been saved and we're about to push a new frame.
        // Note: drain_compile_queue is NOT safe to call here — it modifies
        // engine state (IC invalidation, jit_code changes) that can corrupt
        // the interpreter's assumptions during execution. Queue processing
        // is handled by request_tier_up when new functions hit the threshold.
        let fn_ptr_raw = vm
            .engine
            .jit_code
            .get(fn_idx)
            .copied()
            .unwrap_or(std::ptr::null());

        if !fn_ptr_raw.is_null() {
            let allow_shadow_nonleaf =
                crate::codegen::runtime_fns::allow_nonleaf_native(vm, target_func_id);
            // Only dispatch leaf functions via JIT. Non-leaf JIT dispatch is
            // correct but too slow (each method call pays ~100ns Rust FFI
            // overhead vs ~5ns interpreter frame push). Needs inline caching
            // + direct JIT-to-JIT calls to be viable.
            let jit_depth = crate::codegen::runtime_fns::jit_depth();
            if jit_depth < crate::codegen::runtime_fns::MAX_JIT_DEPTH {
                // Save caller's frame for GC tracing.
                unsafe {
                    let frame = (*fiber).mir_frames.last_mut().unwrap();
                    frame.pc = pc;
                    frame.values = values;
                }

                // Set up JitContext with the CALLEE's module context.
                // The `module_name` parameter is the caller's module; the
                // callee may live in a different module (e.g. a hot loop in
                // crypto.spec calling `Expect.that` defined in
                // @hatch:assert). Use `func_module` to recover the callee's
                // defining module so `GetModuleVar @N` resolves correctly
                // inside the JIT-compiled body — otherwise it reads the
                // caller's slot N and silently returns Null, which later
                // surfaces as "Null does not implement 'foo'".
                let vm_ptr = vm as *mut VM as *mut u8;
                let callee_module_name = vm
                    .engine
                    .func_module(target_func_id)
                    .cloned()
                    .unwrap_or_else(|| Rc::clone(module_name));
                let mod_name_bytes = callee_module_name.as_bytes();
                let (mv_ptr, mv_count) = vm
                    .engine
                    .modules
                    .get(callee_module_name.as_str())
                    .map(|m| (m.vars.as_ptr() as *mut u64, m.vars.len() as u32))
                    .unwrap_or((std::ptr::null_mut(), 0));
                crate::codegen::runtime_fns::set_jit_context(
                    crate::codegen::runtime_fns::JitContext {
                        module_vars: mv_ptr,
                        module_var_count: mv_count,
                        vm: vm_ptr,
                        module_name: mod_name_bytes.as_ptr(),
                        module_name_len: mod_name_bytes.len() as u32,
                        current_func_id: target_func_id.0 as u64,
                        closure: closure_ptr as *mut u8,
                        defining_class: defining_class
                            .map(|p| p as *mut u8)
                            .unwrap_or(std::ptr::null_mut()),
                        jit_code_base: vm.engine.jit_code.as_ptr(),
                        jit_code_len: vm.engine.jit_code.len() as u32,
                    },
                );

                // With Cranelift, dispatch all JIT functions (leaf + non-leaf).
                // The old backend only dispatched leaf functions to avoid
                // spill-slot bugs in non-leaf code.
                #[cfg(feature = "cranelift")]
                let dispatch_jit = true;
                #[cfg(not(feature = "cranelift"))]
                let dispatch_jit = vm.engine.jit_leaf.get(fn_idx).copied().unwrap_or(false);

                if dispatch_jit {
                    vm.engine.note_native_entry(target_func_id);
                    crate::codegen::runtime_fns::set_jit_depth(jit_depth + 1);
                    let root_len_before = crate::codegen::runtime_fns::jit_roots_snapshot_len();
                    let result_bits = unsafe { call_jit_fn(fn_ptr_raw, arg_vals) };
                    crate::codegen::runtime_fns::jit_roots_restore_len(root_len_before);
                    crate::codegen::runtime_fns::set_jit_depth(jit_depth);

                    let mut values = unsafe {
                        (*fiber)
                            .mir_frames
                            .last_mut()
                            .map(|f| std::mem::take(&mut f.values))
                            .unwrap_or_default()
                    };
                    let result_val = Value::from_bits(result_bits);

                    set_reg(&mut values, return_dst.0 as u16, result_val);
                    unsafe {
                        if let Some(frame) = (*fiber).mir_frames.last_mut() {
                            frame.pc = pc;
                            frame.values = values;
                        }
                    }
                    return Ok(());
                }
                if allow_shadow_nonleaf {
                    let result_bits = crate::codegen::runtime_fns::call_closure_jit_or_sync(
                        vm,
                        closure_ptr,
                        arg_vals,
                        defining_class,
                    );

                    let mut values = unsafe {
                        (*fiber)
                            .mir_frames
                            .last_mut()
                            .map(|f| std::mem::take(&mut f.values))
                            .unwrap_or_default()
                    };
                    let result_val = Value::from_bits(result_bits);

                    set_reg(&mut values, return_dst.0 as u16, result_val);
                    unsafe {
                        if let Some(frame) = (*fiber).mir_frames.last_mut() {
                            frame.pc = pc;
                            frame.values = values;
                        }
                    }
                    return Ok(());
                }
                vm.engine.note_fallback_to_interpreter(target_func_id);
                // Non-leaf: fall through to interpreter path.
                // Restore values from saved frame before continuing.
                values = unsafe {
                    (*fiber)
                        .mir_frames
                        .last_mut()
                        .map(|f| std::mem::take(&mut f.values))
                        .unwrap_or_default()
                };
            }
        }
    }

    // Tier-up profiling for functions that didn't take the JIT fast path above
    // (either >4 args, or interpreter mode).
    if vm.engine.mode != ExecutionMode::Interpreter {
        let should_tier_up = vm.engine.record_call(target_func_id);
        if should_tier_up {
            vm.engine.request_tier_up(target_func_id, &vm.interner);
        }
        vm.engine.poll_compilations();
    }

    // Threaded dispatch: use pre-decoded threaded code instead of
    // bytecode interpretation. Only for functions that DON'T have JIT
    // code (those already use the faster JIT path above). Threaded is
    // a middle tier between bytecode and JIT: faster decode than bytecode,
    // but slower than native JIT. Skipped when called from
    // try_operator_dispatch (allow_threaded=false).
    // Skip threaded for constructors — they need special instance
    // allocation that the threaded path doesn't handle. Also check if
    // the function name starts with "new(" as a heuristic.
    let is_constructor = vm
        .engine
        .get_mir(target_func_id)
        .map(|m| {
            let name = vm.interner.resolve(m.name);
            name.starts_with("new(") || name.starts_with("new ")
        })
        .unwrap_or(false);
    let fn_has_jit = !vm
        .engine
        .jit_code
        .get(fn_idx)
        .copied()
        .unwrap_or(std::ptr::null())
        .is_null();
    let threaded_bypasses_osr = if vm.engine.mode == ExecutionMode::Tiered
        && std::env::var_os("WLIFT_DISABLE_METHOD_OSR").is_none()
    {
        vm.engine
            .ensure_bytecode(target_func_id)
            .map(|bc_ptr| unsafe {
                let bc = &*bc_ptr;
                !bc.osr_points.is_empty()
            })
            .unwrap_or(false)
    } else {
        false
    };
    #[cfg(feature = "host")]
    if allow_threaded
        && !fn_has_jit
        && !is_constructor
        && !threaded_bypasses_osr
        && vm.engine.mode != ExecutionMode::Interpreter
        && crate::mir::threaded::threaded_depth_ok()
        && crate::mir::threaded::threaded_enabled()
    {
        // Ensure threaded code exists (lazy init).
        let _ = vm.engine.ensure_threaded_code(target_func_id, &vm.interner);
        // Now borrow the threaded code immutably + modules separately.
        let has_tc = vm
            .engine
            .threaded_code
            .get(fn_idx)
            .and_then(|t| t.as_ref())
            .and_then(|t| t.as_ref())
            .is_some();
        if has_tc {
            // Resolve module_vars against the *callee's* defining
            // module, not the caller's. Without this, a route handler
            // closure registered in `main.wren` and dispatched via the
            // threaded path from `@hatch:web`'s listen loop would see
            // `@hatch:web`'s module-var slots — every `GetModuleVar @N`
            // inside the closure would resolve to whatever class
            // happens to live at slot N in @hatch:web (e.g. `page`
            // would silently rewrite to the `HxResponse` class), and
            // anything that did `arg is String` would die with
            // "Right operand must be a class" because String is at
            // a different slot in main vs @hatch:web.
            let callee_module_name = vm
                .engine
                .func_module(target_func_id)
                .cloned()
                .unwrap_or_else(|| Rc::clone(module_name));
            let (mv_ptr, mv_count) = vm
                .engine
                .modules
                .get(callee_module_name.as_str())
                .map(|m| (m.vars.as_ptr() as *mut u64, m.vars.len() as u32))
                .unwrap_or((std::ptr::null_mut(), 0));
            let vm_ptr = vm as *mut VM as *mut u8;
            // Share the bytecode function's IC table with the threaded
            // interpreter so IC entries populated by the bytecode slow
            // path are immediately available to the threaded fast path.
            let bc_ic_table = vm
                .engine
                .ensure_bytecode(target_func_id)
                .map(|bc_ptr| unsafe {
                    let bc = &*bc_ptr;
                    bc.ic_table.get()
                });
            // Set JIT context so wren_call_N slow path works.
            let mod_name_bytes = callee_module_name.as_bytes();
            crate::codegen::runtime_fns::mutate_jit_ctx(|ctx| {
                ctx.vm = vm_ptr;
                ctx.module_vars = mv_ptr;
                ctx.module_var_count = mv_count;
                ctx.module_name = mod_name_bytes.as_ptr();
                ctx.module_name_len = mod_name_bytes.len() as u32;
                ctx.current_func_id = target_func_id.0 as u64;
                ctx.closure = closure_ptr as *mut u8;
                ctx.defining_class = defining_class
                    .map(|p| p as *mut u8)
                    .unwrap_or(std::ptr::null_mut());
                ctx.jit_code_base = vm.engine.jit_code.as_ptr();
                ctx.jit_code_len = vm.engine.jit_code.len() as u32;
            });
            let tc = vm.engine.threaded_code[fn_idx]
                .as_ref()
                .unwrap()
                .as_ref()
                .unwrap();
            let recycled = vm.register_pool.pop();
            let (result, regs_back) = crate::mir::threaded::execute_threaded(
                tc,
                arg_vals,
                mv_ptr,
                mv_count,
                vm_ptr,
                bc_ic_table,
                recycled,
            );
            if vm.register_pool.len() < 128 {
                vm.register_pool.push(regs_back);
            }
            // Store result in caller's frame and return.
            set_reg(&mut values, return_dst.0 as u16, result);
            unsafe {
                if let Some(frame) = (*fiber).mir_frames.last_mut() {
                    frame.pc = pc;
                    frame.values = values;
                }
            }
            return Ok(());
        }
    }

    vm.engine.note_interpreted_entry(target_func_id);

    // Get bytecode for target function (lazily compiled, survives tier-up).
    // Use raw pointer to avoid Arc clone overhead.
    let target_bc_ptr = vm
        .engine
        .ensure_bytecode(target_func_id)
        .ok_or_else(|| RuntimeError::Error("invalid closure func_id".into()))?;
    let target_bc = unsafe { &*target_bc_ptr };

    // Pre-bind entry block params using cached param_offsets.
    // Reuse a register file from the pool when possible to avoid heap alloc.
    let reg_count = target_bc.register_count as usize;
    let mut new_values = if let Some(mut recycled) = vm.register_pool.pop() {
        recycled.resize(reg_count, UNDEF);
        // Fill all slots to UNDEF (resize only fills *new* slots)
        new_values_fill_undef(&mut recycled, reg_count);
        recycled
    } else {
        vec![UNDEF; reg_count]
    };
    for &(dst_reg, param_idx) in target_bc.param_offsets.iter() {
        if (param_idx as usize) < arg_vals.len() && (dst_reg as usize) < reg_count {
            new_values[dst_reg as usize] = arg_vals[param_idx as usize];
        }
    }

    // Check call depth
    let depth = unsafe { (*fiber).mir_frames.len() };
    if depth >= vm.config.max_call_depth {
        return Err(RuntimeError::StackOverflow);
    }

    // Save caller frame state
    unsafe {
        let frame = (*fiber).mir_frames.last_mut().unwrap();
        frame.pc = pc;
        frame.values = values;
        frame.bc_ptr = caller_bc_ptr;
    }

    // Push new frame. Use the callee's defining module if recorded so
    // `GetModuleVar` inside reads against its own module's slots, not
    // the caller's (Rc::clone is cheap — just a refcount bump).
    let frame_module = vm
        .engine
        .func_module(target_func_id)
        .cloned()
        .unwrap_or_else(|| Rc::clone(module_name));
    unsafe {
        (*fiber).mir_frames.push(MirCallFrame {
            func_id: target_func_id,
            current_block: BlockId(0),
            ip: 0,
            pc: 0,
            values: new_values,
            module_name: frame_module,
            return_dst: Some(return_dst),
            closure: Some(closure_ptr),
            defining_class,
            bc_ptr: target_bc_ptr,
        });
    }

    Ok(())
}

/// Set x19 = JitContext pointer for JIT code (aarch64 only).
/// Must be called before any JIT function entry.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn ensure_ctx_reg() {
    let ctx_ptr = crate::codegen::runtime_fns::jit_ctx_ptr() as u64;
    unsafe {
        core::arch::asm!(
            "mov x20, {ctx}",
            ctx = in(reg) ctx_ptr,
            lateout("x20") _,
            options(nostack, nomem),
        );
    }
}
#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn ensure_ctx_reg() {}

/// Transmute and call a JIT function pointer with the given args (max 4).
/// Public wrapper for call_jit_fn (used by call_constructor_sync in vm.rs).
///
/// # Safety
/// `fn_ptr` must point to a valid JIT-compiled function whose ABI matches
/// the number of arguments provided in `args`. The JIT code must be safe to
/// execute from the current thread context.
pub unsafe fn call_jit_fn_pub(fn_ptr: *const u8, args: &[Value]) -> u64 {
    call_jit_fn(fn_ptr, args)
}

/// Sets x20 = JitContext pointer before the call (preserved by callee-saved ABI).
#[inline(always)]
unsafe fn call_jit_fn(fn_ptr: *const u8, args: &[Value]) -> u64 {
    ensure_ctx_reg();
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

/// Resume a caller fiber with a value.
/// If `fiber` was launched via `.try()` / `.try(arg)`, route a
/// raised runtime error through its `error` slot, mark it done,
/// and resume the caller (if any). Returns `Some(value)` if the
/// error was caught — outer caller should `continue 'fiber_loop`
/// (caller resumed) or `return Ok(value)` (no caller, fiber run
/// terminates here). Returns `None` if the fiber wasn't in try
/// mode — the error should propagate to `Err(...)` like before.
///
/// Mirrors the native-side error catch in the bytecode `Op::Call`
/// dispatcher so non-native error sites (method-not-found, super
/// chain miss) get the same `Fiber.try` semantics. Without this,
/// `Fiber.try { B.new().missing() }` aborts the process even
/// though `Fiber.try { Fiber.abort("nope") }` is caught cleanly.
unsafe fn route_method_error_through_fiber_try(
    vm: &mut VM,
    fiber: *mut ObjFiber,
    err_msg: String,
) -> Option<Value> {
    unsafe {
        if !(*fiber).is_try {
            return None;
        }
        let err_val = vm.new_string(err_msg);
        (*fiber).error = err_val;
        (*fiber).is_try = false;
        (*fiber).state = FiberState::Done;
        let caller = (*fiber).caller;
        if !caller.is_null() {
            (*fiber).caller = std::ptr::null_mut();
            resume_caller(vm, caller, err_val);
        }
        Some(err_val)
    }
}

fn resume_caller(vm: &mut VM, caller: *mut ObjFiber, value: Value) {
    unsafe {
        if let Some(dst) = (*caller).resume_value_dst.take() {
            if let Some(frame) = (*caller).mir_frames.last_mut() {
                set_reg(&mut frame.values, dst.0 as u16, value);
            } else {
                // JIT barrier path: caller's frames were temporarily removed.
                // Store the resume value so handle_jit_fiber_action can read it.
                (*caller).jit_resume_value = Some(value);
            }
        } else if (*caller).mir_frames.is_empty() {
            // No resume_value_dst and no frames: JIT barrier path.
            (*caller).jit_resume_value = Some(value);
        }
    }
    vm.fiber = caller;
}

/// Handle a fiber action (call/yield/transfer) triggered by a native method.
fn handle_fiber_action_bc(
    vm: &mut VM,
    fiber: *mut ObjFiber,
    action: FiberAction,
    pc: u32,
    values: Vec<Value>,
    call_dst: ValueId,
) -> Result<(), RuntimeError> {
    // Save current fiber's execution state
    unsafe {
        if let Some(frame) = (*fiber).mir_frames.last_mut() {
            frame.pc = pc;
            frame.values = values;
        }
    }

    match action {
        FiberAction::Call { target, value } => {
            unsafe {
                (*fiber).state = FiberState::Suspended;
                (*fiber).resume_value_dst = Some(call_dst);
                (*target).caller = fiber;
            }
            let target_state = unsafe { (*target).state };
            if target_state == FiberState::Suspended {
                unsafe {
                    if let Some(dst) = (*target).resume_value_dst.take() {
                        if let Some(frame) = (*target).mir_frames.last_mut() {
                            set_reg(&mut frame.values, dst.0 as u16, value);
                        }
                    }
                }
            }
            vm.fiber = target;
            Ok(())
        }
        FiberAction::Yield { value } => {
            unsafe {
                (*fiber).state = FiberState::Suspended;
                (*fiber).resume_value_dst = Some(call_dst);
            }
            let caller = unsafe { (*fiber).caller };
            if caller.is_null() {
                // No caller to resume — typical when the host
                // scheduler resumed a parked fiber and that fiber
                // awaits a *second* future. The first yield
                // returned to its caller and cleared the slot;
                // subsequent yields have nowhere to return.
                // Clear `vm.fiber` so `run_fiber` exits cleanly
                // on its next loop iteration; the host scheduler
                // resumes the parked fiber via `Browser.parkSelf`
                // + future settlement (the fiber has to be
                // registered in the host's parked list before
                // this yield — an unparked yield with no caller
                // is still a stuck suspension, but that's the
                // user's misuse to detect, not a runtime error).
                let _ = value;
                vm.fiber = std::ptr::null_mut();
                return Ok(());
            }
            unsafe {
                (*fiber).caller = std::ptr::null_mut();
            }
            resume_caller(vm, caller, value);
            Ok(())
        }
        FiberAction::Transfer { target, value } => {
            unsafe {
                (*fiber).state = FiberState::Suspended;
                (*fiber).resume_value_dst = Some(call_dst);
            }
            let target_state = unsafe { (*target).state };
            if target_state == FiberState::Suspended {
                unsafe {
                    if let Some(dst) = (*target).resume_value_dst.take() {
                        if let Some(frame) = (*target).mir_frames.last_mut() {
                            set_reg(&mut frame.values, dst.0 as u16, value);
                        }
                    }
                }
            }
            vm.fiber = target;
            Ok(())
        }
        FiberAction::Suspend => {
            unsafe {
                (*fiber).state = FiberState::Suspended;
                (*fiber).resume_value_dst = Some(call_dst);
            }
            let caller = unsafe { (*fiber).caller };
            if !caller.is_null() {
                unsafe {
                    (*fiber).caller = std::ptr::null_mut();
                }
                resume_caller(vm, caller, Value::null());
            }
            Ok(())
        }
    }
}

/// Try to dispatch an operator as a method call on the receiver's class.
/// Returns true if a closure frame was pushed (caller should continue 'fiber_loop).
/// Returns false if native method was called and result stored in values[dst].
#[allow(clippy::too_many_arguments)]
fn try_operator_dispatch(
    vm: &mut VM,
    fiber: *mut ObjFiber,
    recv: Value,
    method_str: &str,
    args: &[Value],
    pc: &mut u32,
    values: &mut Vec<Value>,
    dst: u16,
    module_name: &Rc<String>,
    caller_bc_ptr: *const BytecodeFunction,
) -> Result<bool, RuntimeError> {
    let method_sym = vm.interner.intern(method_str);
    let class = vm.class_of(recv);
    // Check method cache first
    let method_entry = if let Some((m, _dc)) = vm.method_cache.lookup(class, method_sym) {
        Some(m)
    } else {
        let found = unsafe { (*class).find_method(method_sym).cloned() };
        if let Some(ref m) = found {
            vm.method_cache.insert(class, method_sym, *m, class);
        }
        found
    };
    match method_entry {
        Some(m @ (Method::Native(_) | Method::ForeignC(_) | Method::ForeignCDynamic(_))) => {
            let mut arg_vals: SmallVec<[Value; 4]> = SmallVec::with_capacity(1 + args.len());
            arg_vals.push(recv);
            arg_vals.extend_from_slice(args);
            let result = match m {
                Method::Native(func) => {
                    call_native_with_frame_sync(vm, fiber, *pc, values, func, &arg_vals)
                }
                Method::ForeignC(func) => {
                    call_foreign_c_with_frame_sync(vm, fiber, *pc, values, func, &arg_vals)
                }
                Method::ForeignCDynamic(idx) => {
                    call_foreign_dynamic_with_frame_sync(vm, fiber, *pc, values, idx, &arg_vals)
                }
                _ => unreachable!(),
            };
            set_reg(values, dst, result);
            Ok(false)
        }
        Some(Method::Closure(closure_ptr)) => {
            let mut arg_vals: SmallVec<[Value; 4]> = SmallVec::with_capacity(1 + args.len());
            arg_vals.push(recv);
            arg_vals.extend_from_slice(args);
            // Use allow_threaded=false: the bytecode loop expects a
            // frame push/pop cycle for operator dispatch.
            dispatch_closure_bc_inner(
                vm,
                fiber,
                closure_ptr,
                &arg_vals,
                *pc,
                std::mem::take(values),
                module_name,
                ValueId(dst as u32),
                None,
                caller_bc_ptr,
                false, // no threaded — bytecode loop handles return
            )?;
            Ok(true)
        }
        _ => Err(RuntimeError::Error(format!(
            "{} does not implement '{}'",
            vm.class_name_of(recv),
            method_str
        ))),
    }
}

/// Walk the class hierarchy to find the method and its defining class in a single pass.
unsafe fn find_method_with_class(
    cls: *mut ObjClass,
    method: SymbolId,
) -> Option<(Method, *mut ObjClass)> {
    let idx = method.index() as usize;
    let mut c = cls;
    while !c.is_null() {
        let cls = &*c;
        if idx < cls.methods.len() {
            if let Some(m) = &cls.methods[idx] {
                return Some((*m, c));
            }
        }
        c = (*c).superclass;
    }
    None
}

/// Return the interned SymbolId for a subscript-get signature.
#[inline]
fn subscript_get_sym(vm: &mut VM, arity: usize) -> SymbolId {
    let sig: &str = match arity {
        1 => "[_]",
        2 => "[_,_]",
        3 => "[_,_,_]",
        4 => "[_,_,_,_]",
        _ => {
            let s = format!("[{}]", vec!["_"; arity].join(","));
            return vm.interner.intern(&s);
        }
    };
    vm.interner.intern(sig)
}

/// Return the interned SymbolId for a subscript-set signature.
#[inline]
fn subscript_set_sym(vm: &mut VM, arity: usize) -> SymbolId {
    let sig: &str = match arity {
        1 => "[_]=(_)",
        2 => "[_,_]=(_)",
        3 => "[_,_,_]=(_)",
        4 => "[_,_,_,_]=(_)",
        _ => {
            let s = format!("[{}]=(_)", vec!["_"; arity].join(","));
            return vm.interner.intern(&s);
        }
    };
    vm.interner.intern(sig)
}

pub fn value_to_string(vm: &VM, value: Value) -> String {
    if value.is_null() {
        "null".to_string()
    } else if let Some(b) = value.as_bool() {
        if b {
            "true".to_string()
        } else {
            "false".to_string()
        }
    } else if let Some(n) = value.as_num() {
        if n == n.floor() && n.abs() < 1e15 && !n.is_infinite() {
            format!("{}", n as i64)
        } else {
            format!("{}", n)
        }
    } else if value.is_object() {
        let ptr = value.as_object().unwrap();
        let header = unsafe { &*(ptr as *const ObjHeader) };
        match header.obj_type {
            ObjType::String => {
                let obj = unsafe { &*(ptr as *const ObjString) };
                obj.value.clone()
            }
            // Core object types format with their shape, matching
            // what `System.print` emits. Prior to this fix, string
            // interpolation (`"%(x)"`) hit the "instance of <class>"
            // fallback for everything non-String, so `"%([1,2])"`
            // read as "instance of List" instead of "[1, 2]".
            ObjType::List => {
                let list = unsafe { &*(ptr as *const crate::runtime::object::ObjList) };
                let parts: Vec<String> = (0..list.len())
                    .map(|i| value_to_string(vm, list.get(i).unwrap_or(Value::null())))
                    .collect();
                format!("[{}]", parts.join(", "))
            }
            ObjType::Map => {
                let map = unsafe { &*(ptr as *const crate::runtime::object::ObjMap) };
                let parts: Vec<String> = map
                    .entries
                    .iter()
                    .map(|(k, v)| {
                        format!("{}: {}", value_to_string(vm, k.0), value_to_string(vm, *v))
                    })
                    .collect();
                format!("{{{}}}", parts.join(", "))
            }
            ObjType::Range => {
                let range = unsafe { &*(ptr as *const crate::runtime::object::ObjRange) };
                let op = if range.is_inclusive { ".." } else { "..." };
                format!(
                    "{}{}{}",
                    value_to_string(vm, Value::num(range.from)),
                    op,
                    value_to_string(vm, Value::num(range.to))
                )
            }
            ObjType::Fn | ObjType::Closure => "Fn".to_string(),
            ObjType::Fiber => "Fiber".to_string(),
            // Classes stringify to their own name (`List`, `String`,
            // a user class's declared name, etc.), matching what
            // `Class.toString` returns when dispatched explicitly.
            ObjType::Class => {
                let class = unsafe { &*(ptr as *const crate::runtime::object::ObjClass) };
                vm.interner.resolve(class.name).to_string()
            }
            _ => {
                let class_name = vm.class_name_of(value);
                format!("instance of {}", class_name)
            }
        }
    } else {
        "?".to_string()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "cranelift")]
    use crate::mir::MirType;
    use crate::mir::{Instruction, MirFunction, Terminator};
    use crate::runtime::vm::{VMConfig, VM};

    fn make_vm() -> VM {
        VM::new(VMConfig::default())
    }

    #[test]
    fn test_return_constant() {
        let mut vm = make_vm();
        let name = vm.interner.intern("test");
        let mut f = MirFunction::new(name, 0);
        let bb = f.new_block();
        let v0 = f.new_value();
        f.block_mut(bb)
            .instructions
            .push((v0, Instruction::ConstNum(42.0)));
        f.block_mut(bb).terminator = Terminator::Return(v0);

        let mut vars = vec![];
        let result = eval_in_vm(&mut vm, &f, &mut vars).unwrap();
        assert_eq!(result.as_num().unwrap(), 42.0);
    }

    #[test]
    fn test_return_null() {
        let mut vm = make_vm();
        let name = vm.interner.intern("test");
        let mut f = MirFunction::new(name, 0);
        let bb = f.new_block();
        f.block_mut(bb).terminator = Terminator::ReturnNull;

        let mut vars = vec![];
        let result = eval_in_vm(&mut vm, &f, &mut vars).unwrap();
        assert!(result.is_null());
    }

    #[test]
    fn test_arithmetic() {
        let mut vm = make_vm();
        let name = vm.interner.intern("test");
        let mut f = MirFunction::new(name, 0);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        {
            let b = f.block_mut(bb);
            b.instructions.push((v0, Instruction::ConstNum(10.0)));
            b.instructions.push((v1, Instruction::ConstNum(3.0)));
            b.instructions.push((v2, Instruction::Add(v0, v1)));
            b.terminator = Terminator::Return(v2);
        }

        let mut vars = vec![];
        let result = eval_in_vm(&mut vm, &f, &mut vars).unwrap();
        assert_eq!(result.as_num().unwrap(), 13.0);
    }

    #[test]
    fn test_string_creation() {
        let mut vm = make_vm();
        let hello_sym = vm.interner.intern("hello");
        let name = vm.interner.intern("test");
        let mut f = MirFunction::new(name, 0);
        let bb = f.new_block();
        let v0 = f.new_value();
        f.block_mut(bb)
            .instructions
            .push((v0, Instruction::ConstString(hello_sym.index())));
        f.block_mut(bb).terminator = Terminator::Return(v0);

        let mut vars = vec![];
        let result = eval_in_vm(&mut vm, &f, &mut vars).unwrap();
        assert!(result.is_object());
    }

    #[test]
    fn test_call_native_method() {
        let mut vm = make_vm();
        let abs_sym = vm.interner.intern("abs");
        let name = vm.interner.intern("test");
        let mut f = MirFunction::new(name, 0);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        {
            let b = f.block_mut(bb);
            b.instructions.push((v0, Instruction::ConstNum(-5.0)));
            b.instructions.push((
                v1,
                Instruction::Call {
                    receiver: v0,
                    method: abs_sym,
                    args: vec![],
                    pure_call: false,
                },
            ));
            b.terminator = Terminator::Return(v1);
        }

        let mut vars = vec![];
        let result = eval_in_vm(&mut vm, &f, &mut vars).unwrap();
        assert_eq!(result.as_num().unwrap(), 5.0);
    }

    #[test]
    fn test_call_native_binary() {
        let mut vm = make_vm();
        let add_sym = vm.interner.intern("+(_)");
        let name = vm.interner.intern("test");
        let mut f = MirFunction::new(name, 0);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        {
            let b = f.block_mut(bb);
            b.instructions.push((v0, Instruction::ConstNum(10.0)));
            b.instructions.push((v1, Instruction::ConstNum(20.0)));
            b.instructions.push((
                v2,
                Instruction::Call {
                    receiver: v0,
                    method: add_sym,
                    args: vec![v1],
                    pure_call: false,
                },
            ));
            b.terminator = Terminator::Return(v2);
        }

        let mut vars = vec![];
        let result = eval_in_vm(&mut vm, &f, &mut vars).unwrap();
        assert_eq!(result.as_num().unwrap(), 30.0);
    }

    #[test]
    fn test_make_list() {
        let mut vm = make_vm();
        let name = vm.interner.intern("test");
        let mut f = MirFunction::new(name, 0);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        {
            let b = f.block_mut(bb);
            b.instructions.push((v0, Instruction::ConstNum(1.0)));
            b.instructions.push((v1, Instruction::ConstNum(2.0)));
            b.instructions
                .push((v2, Instruction::MakeList(vec![v0, v1])));
            b.terminator = Terminator::Return(v2);
        }

        let mut vars = vec![];
        let result = eval_in_vm(&mut vm, &f, &mut vars).unwrap();
        assert!(result.is_object());
    }

    #[test]
    fn test_module_vars() {
        let mut vm = make_vm();
        let name = vm.interner.intern("test");
        let mut f = MirFunction::new(name, 0);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        {
            let b = f.block_mut(bb);
            b.instructions.push((v0, Instruction::ConstNum(99.0)));
            b.instructions.push((v1, Instruction::SetModuleVar(0, v0)));
            b.instructions.push((v2, Instruction::GetModuleVar(0)));
            b.terminator = Terminator::Return(v2);
        }

        let mut vars = vec![];
        let result = eval_in_vm(&mut vm, &f, &mut vars).unwrap();
        assert_eq!(result.as_num().unwrap(), 99.0);
    }

    #[test]
    fn test_loop_with_branch() {
        let mut vm = make_vm();
        let name = vm.interner.intern("test");
        let mut f = MirFunction::new(name, 0);

        let bb0 = f.new_block();
        let bb1 = f.new_block();
        let bb2 = f.new_block();
        let bb3 = f.new_block();

        let v_zero = f.new_value();
        let v_one = f.new_value();
        {
            let b = f.block_mut(bb0);
            b.instructions.push((v_zero, Instruction::ConstNum(0.0)));
            b.instructions.push((v_one, Instruction::ConstNum(1.0)));
            b.terminator = Terminator::Branch {
                target: bb1,
                args: vec![v_zero, v_one],
            };
        }

        let v_sum_p = f.new_value();
        let v_i_p = f.new_value();
        let v_five = f.new_value();
        let v_cond = f.new_value();
        {
            let b = f.block_mut(bb1);
            b.instructions.push((v_sum_p, Instruction::BlockParam(0)));
            b.instructions.push((v_i_p, Instruction::BlockParam(1)));
            b.instructions.push((v_five, Instruction::ConstNum(5.0)));
            b.instructions
                .push((v_cond, Instruction::CmpLe(v_i_p, v_five)));
            b.terminator = Terminator::CondBranch {
                condition: v_cond,
                true_target: bb2,
                true_args: vec![v_sum_p, v_i_p],
                false_target: bb3,
                false_args: vec![v_sum_p],
            };
        }

        let v_sum_b = f.new_value();
        let v_i_b = f.new_value();
        let v_sum2 = f.new_value();
        let v_one2 = f.new_value();
        let v_i2 = f.new_value();
        {
            let b = f.block_mut(bb2);
            b.instructions.push((v_sum_b, Instruction::BlockParam(0)));
            b.instructions.push((v_i_b, Instruction::BlockParam(1)));
            b.instructions
                .push((v_sum2, Instruction::Add(v_sum_b, v_i_b)));
            b.instructions.push((v_one2, Instruction::ConstNum(1.0)));
            b.instructions.push((v_i2, Instruction::Add(v_i_b, v_one2)));
            b.terminator = Terminator::Branch {
                target: bb1,
                args: vec![v_sum2, v_i2],
            };
        }

        let v_result = f.new_value();
        {
            let b = f.block_mut(bb3);
            b.instructions.push((v_result, Instruction::BlockParam(0)));
            b.terminator = Terminator::Return(v_result);
        }

        let mut vars = vec![];
        let result = eval_in_vm(&mut vm, &f, &mut vars).unwrap();
        assert_eq!(result.as_num().unwrap(), 15.0);
    }

    #[cfg(feature = "cranelift")]
    #[test]
    fn test_tiered_cond_branch_backedge_enters_osr() {
        let config = VMConfig {
            execution_mode: ExecutionMode::Tiered,
            jit_threshold: 1,
            opt_threshold: u32::MAX,
            ..VMConfig::default()
        };
        let mut vm = VM::new(config);
        vm.engine.collect_tier_stats = true;

        let name = vm.interner.intern("cond_backedge_osr_test");
        let mut f = MirFunction::new(name, 0);

        let entry = f.new_block();
        let loop_bb = f.new_block();
        let exit = f.new_block();

        let v_zero = f.new_value();
        {
            let b = f.block_mut(entry);
            b.instructions.push((v_zero, Instruction::ConstNum(0.0)));
            b.terminator = Terminator::Branch {
                target: loop_bb,
                args: vec![v_zero],
            };
        }

        let v_i = f.new_value();
        let v_limit = f.new_value();
        let v_cond = f.new_value();
        let v_one = f.new_value();
        let v_next = f.new_value();
        {
            let b = f.block_mut(loop_bb);
            b.params.push((v_i, MirType::Value));
            b.instructions.push((v_i, Instruction::BlockParam(0)));
            b.instructions
                .push((v_limit, Instruction::ConstNum(100_000.0)));
            b.instructions
                .push((v_cond, Instruction::CmpLt(v_i, v_limit)));
            b.instructions.push((v_one, Instruction::ConstNum(1.0)));
            b.instructions.push((v_next, Instruction::Add(v_i, v_one)));
            b.terminator = Terminator::CondBranch {
                condition: v_cond,
                true_target: loop_bb,
                true_args: vec![v_next],
                false_target: exit,
                false_args: vec![v_i],
            };
        }

        let v_result = f.new_value();
        {
            let b = f.block_mut(exit);
            b.params.push((v_result, MirType::Value));
            b.instructions.push((v_result, Instruction::BlockParam(0)));
            b.terminator = Terminator::Return(v_result);
        }

        let mut vars = vec![];
        let result = eval_in_vm(&mut vm, &f, &mut vars).unwrap();
        assert_eq!(result.as_num().unwrap(), 100_000.0);

        let osr_entries: u64 = vm
            .engine
            .tier_stats
            .iter()
            .map(|stats| stats.osr_entries)
            .sum();
        assert!(
            osr_entries > 0,
            "expected the conditional back-edge loop to enter OSR"
        );
    }

    #[test]
    fn test_method_not_found() {
        let mut vm = make_vm();
        let bogus = vm.interner.intern("bogus");
        let name = vm.interner.intern("test");
        let mut f = MirFunction::new(name, 0);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        {
            let b = f.block_mut(bb);
            b.instructions.push((v0, Instruction::ConstNum(1.0)));
            b.instructions.push((
                v1,
                Instruction::Call {
                    receiver: v0,
                    method: bogus,
                    args: vec![],
                    pure_call: false,
                },
            ));
            b.terminator = Terminator::Return(v1);
        }

        let mut vars = vec![];
        let result = eval_in_vm(&mut vm, &f, &mut vars);
        assert!(matches!(result, Err(RuntimeError::MethodNotFound { .. })));
    }

    #[test]
    fn test_run_fiber_directly() {
        let mut vm = make_vm();
        let name = vm.interner.intern("test");
        let mut f = MirFunction::new(name, 0);
        let bb = f.new_block();
        let v0 = f.new_value();
        f.block_mut(bb)
            .instructions
            .push((v0, Instruction::ConstNum(77.0)));
        f.block_mut(bb).terminator = Terminator::Return(v0);

        let reg_size = f.next_value as usize;
        let func_id = vm.engine.register_function(f);
        let fiber = vm.gc.alloc_fiber();
        unsafe {
            (*fiber).header.class = vm.fiber_class;
            (*fiber).mir_frames.push(MirCallFrame {
                func_id,
                current_block: BlockId(0),
                ip: 0,
                pc: 0,
                values: vec![UNDEF; reg_size],
                module_name: Rc::new("__test__".to_string()),
                return_dst: None,
                closure: None,
                defining_class: None,
                bc_ptr: std::ptr::null(),
            });
        }
        vm.fiber = fiber;

        let result = run_fiber(&mut vm).unwrap();
        assert_eq!(result.as_num().unwrap(), 77.0);
        assert_eq!(unsafe { (*fiber).state }, FiberState::Done);
    }

    #[test]
    fn test_fiber_state_transitions() {
        let mut vm = make_vm();
        let name = vm.interner.intern("test");
        let mut f = MirFunction::new(name, 0);
        let bb = f.new_block();
        f.block_mut(bb).terminator = Terminator::ReturnNull;

        let reg_size = f.next_value as usize;
        let func_id = vm.engine.register_function(f);
        let fiber = vm.gc.alloc_fiber();
        unsafe {
            (*fiber).header.class = vm.fiber_class;
            assert_eq!((*fiber).state, FiberState::New);
            (*fiber).mir_frames.push(MirCallFrame {
                func_id,
                current_block: BlockId(0),
                ip: 0,
                pc: 0,
                values: vec![UNDEF; reg_size],
                module_name: Rc::new("__test__".to_string()),
                return_dst: None,
                closure: None,
                defining_class: None,
                bc_ptr: std::ptr::null(),
            });
        }
        vm.fiber = fiber;

        let _ = run_fiber(&mut vm).unwrap();
        assert_eq!(unsafe { (*fiber).state }, FiberState::Done);
        assert_eq!(unsafe { (*fiber).mir_frames.len() }, 0);
    }
}
