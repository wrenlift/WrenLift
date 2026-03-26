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

    let result = func(vm, args);

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

    let mut steps: usize = 0;

    // Outer loop: re-entered when we push/pop a call frame or switch fibers
    'fiber_loop: loop {
        let mut fiber = vm.fiber;
        unsafe {
            (*fiber).state = FiberState::Running;
        }

        let frame_count = unsafe { (*fiber).mir_frames.len() };
        if frame_count == 0 {
            unsafe {
                (*fiber).state = FiberState::Done;
            }
            return Ok(Value::null());
        }

        // Load execution state from the top frame into locals.
        // These are mutable so that inline call/return can update them
        // without restarting the fiber loop.
        let (mut func_id, mut pc, mut values, module_name, closure, return_dst) = unsafe {
            let frame = (*fiber).mir_frames.last_mut().unwrap();
            (
                frame.func_id,
                frame.pc,
                std::mem::take(&mut frame.values),
                frame.module_name.clone(),
                frame.closure,
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
        let module_vars_ptr: *mut Vec<Value> = vm
            .engine
            .modules
            .get_mut(module_name.as_str())
            .map(|m| &mut m.vars as *mut Vec<Value>)
            .unwrap_or(std::ptr::null_mut());

        // Inner dispatch loop: decodes opcodes from the flat bytecode stream.
        loop {
            // GC safepoint: check every 4096 instructions
            if steps & 0xFFF == 0 {
                if steps > vm.config.step_limit {
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
                    // Only kind 1 (JIT leaf) is inlined here. Adding more IC
                    // kinds inline causes LLVM to bloat the dispatch loop and
                    // regress performance on tight method-call benchmarks.
                    let ic_table = unsafe { &mut *bc.ic_table.get() };
                    if ic_idx < ic_table.len() {
                        let ic = &mut ic_table[ic_idx];
                        if ic.kind == 1 {
                            let func_id = FuncId(ic.func_id as u32);
                            if vm.engine.mode == ExecutionMode::Tiered {
                                let should_tier_up = vm.engine.record_call(func_id);
                                if should_tier_up {
                                    vm.engine.request_tier_up(func_id, &vm.interner);
                                }
                                if vm.engine.has_pending_compilations() {
                                    vm.engine.poll_compilations();
                                }
                            }
                            let recv_class = if recv_val.is_object() {
                                let obj_ptr = unsafe { recv_val.as_object().unwrap_unchecked() };
                                unsafe { (*(obj_ptr as *const ObjHeader)).class as usize }
                            } else {
                                0
                            };
                            if recv_class == ic.class && recv_class != 0 {
                                let fn_idx = ic.func_id as usize;
                                let live_ptr = vm
                                    .engine
                                    .jit_code
                                    .get(fn_idx)
                                    .copied()
                                    .unwrap_or(std::ptr::null());
                                if live_ptr.is_null()
                                    || live_ptr != ic.jit_ptr
                                    || !vm.engine.jit_leaf.get(fn_idx).copied().unwrap_or(false)
                                {
                                    *ic = crate::mir::bytecode::CallSiteIC::default();
                                } else {
                                    let recv_bits = recv_val.to_bits();
                                    vm.engine.note_ic_hit(func_id);
                                    ensure_ctx_reg();
                                    let result_bits = unsafe {
                                        match argc {
                                            0 => {
                                                let f: extern "C" fn(u64) -> u64 =
                                                    std::mem::transmute(live_ptr);
                                                f(recv_bits)
                                            }
                                            1 => {
                                                let a1 = read_u16_at(code, arg_regs_pc);
                                                let f: extern "C" fn(u64, u64) -> u64 =
                                                    std::mem::transmute(live_ptr);
                                                f(recv_bits, get_reg(&values, a1).to_bits())
                                            }
                                            2 => {
                                                let a1 = read_u16_at(code, arg_regs_pc);
                                                let a2 = read_u16_at(code, arg_regs_pc + 2);
                                                let f: extern "C" fn(u64, u64, u64) -> u64 =
                                                    std::mem::transmute(live_ptr);
                                                f(
                                                    recv_bits,
                                                    get_reg(&values, a1).to_bits(),
                                                    get_reg(&values, a2).to_bits(),
                                                )
                                            }
                                            _ => {
                                                let a1 = read_u16_at(code, arg_regs_pc);
                                                let a2 = read_u16_at(code, arg_regs_pc + 2);
                                                let a3 = read_u16_at(code, arg_regs_pc + 4);
                                                let f: extern "C" fn(u64, u64, u64, u64) -> u64 =
                                                    std::mem::transmute(live_ptr);
                                                f(
                                                    recv_bits,
                                                    get_reg(&values, a1).to_bits(),
                                                    get_reg(&values, a2).to_bits(),
                                                    get_reg(&values, a3).to_bits(),
                                                )
                                            }
                                        }
                                    };
                                    vm.engine.note_native_entry(func_id);
                                    set_reg(&mut values, dst, Value::from_bits(result_bits));
                                    steps += 1;
                                    continue;
                                }
                            } else {
                                vm.engine.note_ic_miss(func_id);
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
                        Some(Method::Native(func)) => {
                            let result = call_native_with_frame_sync(
                                vm,
                                fiber,
                                pc,
                                &mut values,
                                func,
                                &arg_vals,
                            );
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
                                if !jit_ptr.is_null()
                                    && vm.engine.jit_leaf.get(fn_idx).copied().unwrap_or(false)
                                {
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
                                    ensure_ctx_reg();
                                    let recv_bits = recv_val.to_bits();
                                    let result_bits = unsafe {
                                        match argc {
                                            0 => {
                                                let f: extern "C" fn(u64) -> u64 =
                                                    std::mem::transmute(jit_ptr);
                                                f(recv_bits)
                                            }
                                            1 => {
                                                let f: extern "C" fn(u64, u64) -> u64 =
                                                    std::mem::transmute(jit_ptr);
                                                f(arg_vals[0].to_bits(), arg_vals[1].to_bits())
                                            }
                                            2 => {
                                                let f: extern "C" fn(u64, u64, u64) -> u64 =
                                                    std::mem::transmute(jit_ptr);
                                                f(
                                                    arg_vals[0].to_bits(),
                                                    arg_vals[1].to_bits(),
                                                    arg_vals[2].to_bits(),
                                                )
                                            }
                                            _ => {
                                                let f: extern "C" fn(u64, u64, u64, u64) -> u64 =
                                                    std::mem::transmute(jit_ptr);
                                                f(
                                                    arg_vals[0].to_bits(),
                                                    arg_vals[1].to_bits(),
                                                    arg_vals[2].to_bits(),
                                                    arg_vals[3].to_bits(),
                                                )
                                            }
                                        }
                                    };
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

                            // Push new frame
                            vm.engine.note_interpreted_entry(target_func_id);
                            unsafe {
                                (*fiber).mir_frames.push(MirCallFrame {
                                    func_id: target_func_id,
                                    current_block: BlockId(0),
                                    ip: 0,
                                    pc: 0,
                                    values: new_values,
                                    module_name: Rc::clone(&module_name),
                                    return_dst: Some(ValueId(dst as u32)),
                                    closure: Some(closure_ptr),
                                    defining_class,
                                    bc_ptr: target_bc_ptr,
                                });
                            }

                            // Update locals to execute callee inline (skip fiber_loop restart)
                            pc = 0;
                            values = unsafe {
                                std::mem::take(&mut (*fiber).mir_frames.last_mut().unwrap().values)
                            };
                            bc_ptr = target_bc_ptr;
                            bc = unsafe { &*target_bc_ptr };
                            code = &bc.code;
                            continue;
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
                                (*fiber).mir_frames.push(MirCallFrame {
                                    func_id: target_func_id,
                                    current_block: BlockId(0),
                                    ip: 0,
                                    pc: 0,
                                    values: new_values,
                                    module_name: Rc::clone(&module_name),
                                    return_dst: Some(ValueId(dst as u32)),
                                    closure: Some(closure_ptr),
                                    defining_class,
                                    bc_ptr: target_bc_ptr,
                                });
                            }

                            pc = 0;
                            values = unsafe {
                                std::mem::take(&mut (*fiber).mir_frames.last_mut().unwrap().values)
                            };
                            bc_ptr = target_bc_ptr;
                            bc = unsafe { &*target_bc_ptr };
                            code = &bc.code;
                            continue;
                        }
                        None => {
                            let method_name = vm.interner.resolve(method).to_string();
                            let class_name = vm.class_name_of(recv_val);
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
                            Some(Method::Native(func)) => {
                                let result = call_native_with_frame_sync(
                                    vm,
                                    fiber,
                                    pc,
                                    &mut values,
                                    func,
                                    &arg_vals,
                                );
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
                        Some(Method::Native(func)) => {
                            let result = call_native_with_frame_sync(
                                vm,
                                fiber,
                                pc,
                                &mut values,
                                func,
                                &all_args,
                            );
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
                        Some(Method::Native(func)) => {
                            let result = call_native_with_frame_sync(
                                vm,
                                fiber,
                                pc,
                                &mut values,
                                func,
                                &all_args,
                            );
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
                    set_reg(&mut values, dst, vm.new_list(elements));
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
                    set_reg(&mut values, dst, vm.new_string(result));
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
                    if stop_depth == Some(remaining_depth) {
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
                    // ── Inline return: restore caller state without fiber_loop restart ──
                    unsafe {
                        let frame = (*fiber).mir_frames.last_mut().unwrap();
                        if let Some(dst) = return_dst {
                            set_reg(&mut frame.values, dst.0 as u16, return_val);
                        }
                        func_id = frame.func_id;
                        pc = frame.pc;
                        values = std::mem::take(&mut frame.values);
                        bc_ptr = if !frame.bc_ptr.is_null() {
                            frame.bc_ptr
                        } else {
                            vm.engine
                                .ensure_bytecode(func_id)
                                .ok_or_else(|| RuntimeError::Error("invalid func_id".into()))?
                        };
                    }
                    bc = unsafe { &*bc_ptr };
                    code = &bc.code;
                    continue;
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
                    if stop_depth == Some(remaining_depth) {
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
                    // ── Inline return: restore caller state without fiber_loop restart ──
                    unsafe {
                        let frame = (*fiber).mir_frames.last_mut().unwrap();
                        if let Some(dst) = return_dst {
                            set_reg(&mut frame.values, dst.0 as u16, Value::null());
                        }
                        func_id = frame.func_id;
                        pc = frame.pc;
                        values = std::mem::take(&mut frame.values);
                        bc_ptr = if !frame.bc_ptr.is_null() {
                            frame.bc_ptr
                        } else {
                            vm.engine
                                .ensure_bytecode(func_id)
                                .ok_or_else(|| RuntimeError::Error("invalid func_id".into()))?
                        };
                    }
                    bc = unsafe { &*bc_ptr };
                    code = &bc.code;
                    continue;
                }
                Op::Unreachable => {
                    return Err(RuntimeError::Unreachable);
                }
                Op::Branch => {
                    let target = read_u32(code, &mut pc);
                    let argc = read_u8(code, &mut pc) as usize;
                    // Bind block params: read [dst, src] pairs
                    for _ in 0..argc {
                        let dst_reg = read_u16(code, &mut pc);
                        let src_reg = read_u16(code, &mut pc);
                        let val = get_reg(&values, src_reg);
                        set_reg(&mut values, dst_reg, val);
                    }
                    pc = target;
                }
                Op::CondBranch => {
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
                    // Don't need to advance pc — both branches jump to target offset

                    if cond.is_truthy_wren() {
                        // Bind true params
                        let mut p = true_params_start;
                        for _ in 0..t_argc {
                            let dst_reg = read_u16(code, &mut p);
                            let src_reg = read_u16(code, &mut p);
                            let val = get_reg(&values, src_reg);
                            set_reg(&mut values, dst_reg, val);
                        }
                        pc = true_off;
                    } else {
                        // Bind false params
                        let mut p = false_params_start;
                        for _ in 0..f_argc {
                            let dst_reg = read_u16(code, &mut p);
                            let src_reg = read_u16(code, &mut p);
                            let val = get_reg(&values, src_reg);
                            set_reg(&mut values, dst_reg, val);
                        }
                        pc = false_off;
                    }
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
    run_fiber_with_stop_depth(vm, None)
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
    mut values: Vec<Value>,
    module_name: &Rc<String>,
    return_dst: ValueId,
    defining_class: Option<*mut ObjClass>,
    caller_bc_ptr: *const BytecodeFunction,
) -> Result<(), RuntimeError> {
    let fn_ptr = unsafe { (*closure_ptr).function };
    let target_func_id = FuncId(unsafe { (*fn_ptr).fn_id });
    let fn_idx = target_func_id.0 as usize;

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

        if native_fn_ptr.is_null() {
            // Not yet compiled: do tier-up profiling + type sampling.
            let should_tier_up = vm.engine.record_call(target_func_id);
            // Sample argument types for profile-guided specialization.
            vm.engine.sample_arg_types(target_func_id, arg_vals);
            if should_tier_up {
                vm.engine.request_tier_up(target_func_id, &vm.interner);
            }
        }
        // Always poll — and always re-read jit_code AFTER poll to avoid
        // use-after-free when poll_compilations replaces a previously-compiled
        // function with a recompiled version (dropping the old ExecutableBuffer).
        vm.engine.poll_compilations();
        let fn_ptr_raw = vm
            .engine
            .jit_code
            .get(fn_idx)
            .copied()
            .unwrap_or(std::ptr::null());

        if !fn_ptr_raw.is_null() {
            let is_leaf = vm.engine.jit_leaf.get(fn_idx).copied().unwrap_or(false);
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

                // Set up base JitContext (module_vars, vm, module_name).
                let vm_ptr = vm as *mut VM as *mut u8;
                let mod_name_bytes = module_name.as_bytes();
                let (mv_ptr, mv_count) = vm
                    .engine
                    .modules
                    .get(module_name.as_str())
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
                    },
                );

                if is_leaf {
                    vm.engine.note_native_entry(target_func_id);
                    crate::codegen::runtime_fns::set_jit_depth(jit_depth + 1);
                    let root_len_before = crate::codegen::runtime_fns::jit_roots_snapshot_len();
                    let result_bits = unsafe { call_jit_fn(fn_ptr_raw, arg_vals) };
                    crate::codegen::runtime_fns::jit_roots_restore_len(root_len_before);
                    crate::codegen::runtime_fns::set_jit_depth(jit_depth);

                    // Take values back from frame (GC may have moved objects).
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

    // Push new frame (Rc::clone is cheap — just refcount bump)
    unsafe {
        (*fiber).mir_frames.push(MirCallFrame {
            func_id: target_func_id,
            current_block: BlockId(0),
            ip: 0,
            pc: 0,
            values: new_values,
            module_name: Rc::clone(module_name),
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
fn resume_caller(vm: &mut VM, caller: *mut ObjFiber, value: Value) {
    unsafe {
        if let Some(dst) = (*caller).resume_value_dst.take() {
            if let Some(frame) = (*caller).mir_frames.last_mut() {
                set_reg(&mut frame.values, dst.0 as u16, value);
            }
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
                return Err(RuntimeError::Error(
                    "Fiber.yield called with no caller".into(),
                ));
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
        Some(Method::Native(func)) => {
            let mut arg_vals: SmallVec<[Value; 4]> = SmallVec::with_capacity(1 + args.len());
            arg_vals.push(recv);
            arg_vals.extend_from_slice(args);
            let result = call_native_with_frame_sync(vm, fiber, *pc, values, func, &arg_vals);
            set_reg(values, dst, result);
            Ok(false)
        }
        Some(Method::Closure(closure_ptr)) => {
            let mut arg_vals: SmallVec<[Value; 4]> = SmallVec::with_capacity(1 + args.len());
            arg_vals.push(recv);
            arg_vals.extend_from_slice(args);
            dispatch_closure_bc(
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
