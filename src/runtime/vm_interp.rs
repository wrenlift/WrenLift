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
use smallvec::SmallVec;

use crate::intern::SymbolId;
use crate::mir::bytecode::{read_u16, read_u32, read_u8, BcConst, BytecodeFunction, Op};
use crate::mir::interp::{InterpError, InterpValue};
use crate::mir::{BlockId, ValueId};
use crate::runtime::engine::FuncId;
use crate::runtime::object::*;
use crate::runtime::value::Value;
use crate::runtime::vm::{FiberAction, VM};

/// Sentinel value for uninitialized register slots.
const UNDEF: InterpValue = InterpValue::Boxed(Value::UNDEFINED);

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
fn get_reg(values: &[InterpValue], idx: u16) -> InterpValue {
    unsafe { *values.get_unchecked(idx as usize) }
}

#[inline(always)]
fn get_reg_val(values: &[InterpValue], idx: u16) -> Value {
    unsafe { values.get_unchecked(idx as usize).to_value() }
}

#[inline(always)]
fn set_reg(values: &mut Vec<InterpValue>, idx: u16, val: InterpValue) {
    let i = idx as usize;
    if i >= values.len() {
        values.resize(i + 1, UNDEF);
    }
    unsafe { *values.get_unchecked_mut(i) = val; }
}

// ---------------------------------------------------------------------------
// Macros for repetitive opcode patterns (must precede run_fiber)
// ---------------------------------------------------------------------------

/// Boxed binary arithmetic: try numeric, fall back to operator dispatch.
/// Returns true if a closure frame was pushed (caller should continue 'fiber_loop).
macro_rules! bc_boxed_binop {
    ($vm:expr, $fiber:expr, $pc:expr, $values:expr, $module_name:expr, $bc:expr, $op:expr, $method:expr) => {{
        let code = &$bc.code;
        let dst = read_u16(code, $pc);
        let lhs_reg = read_u16(code, $pc);
        let rhs_reg = read_u16(code, $pc);
        let a = get_reg($values, lhs_reg);
        let b = get_reg($values, rhs_reg);
        match (a.as_f64(), b.as_f64()) {
            (Ok(x), Ok(y)) => {
                let f: fn(f64, f64) -> f64 = $op;
                set_reg($values, dst, InterpValue::Boxed(Value::num(f(x, y))));
                false
            }
            _ => {
                let recv = a.to_value();
                let arg = b.to_value();
                try_operator_dispatch(
                    $vm, $fiber, recv, $method, &[arg], $pc, $values, dst,
                    $module_name, $bc,
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
        let a = get_reg($values, lhs_reg).as_f64().map_err(RuntimeError::from)?;
        let b = get_reg($values, rhs_reg).as_f64().map_err(RuntimeError::from)?;
        let f: fn(f64, f64) -> f64 = $op;
        set_reg($values, dst, InterpValue::F64(f(a, b)));
    }};
}

/// Boxed comparison: try numeric, fall back to operator dispatch.
/// Returns true if a closure frame was pushed.
macro_rules! bc_boxed_cmp {
    ($vm:expr, $fiber:expr, $pc:expr, $values:expr, $module_name:expr, $bc:expr, $op:expr, $method:expr) => {{
        let code = &$bc.code;
        let dst = read_u16(code, $pc);
        let lhs_reg = read_u16(code, $pc);
        let rhs_reg = read_u16(code, $pc);
        let a = get_reg($values, lhs_reg);
        let b = get_reg($values, rhs_reg);
        match (a.as_f64(), b.as_f64()) {
            (Ok(x), Ok(y)) => {
                let f: fn(f64, f64) -> bool = $op;
                set_reg($values, dst, InterpValue::Boxed(Value::bool(f(x, y))));
                false
            }
            _ => {
                let recv = a.to_value();
                let arg = b.to_value();
                try_operator_dispatch(
                    $vm, $fiber, recv, $method, &[arg], $pc, $values, dst,
                    $module_name, $bc,
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
        let a = get_reg($values, lhs_reg).as_f64().map_err(RuntimeError::from)?;
        let b = get_reg($values, rhs_reg).as_f64().map_err(RuntimeError::from)?;
        let f: fn(f64, f64) -> bool = $op;
        set_reg($values, dst, InterpValue::Bool(f(a, b)));
    }};
}

/// Bitwise binary op: try numeric, fall back to operator dispatch.
/// Returns true if a closure frame was pushed.
macro_rules! bc_bitwise_binop {
    ($vm:expr, $fiber:expr, $pc:expr, $values:expr, $module_name:expr, $bc:expr, $op:expr, $method:expr) => {{
        let code = &$bc.code;
        let dst = read_u16(code, $pc);
        let lhs_reg = read_u16(code, $pc);
        let rhs_reg = read_u16(code, $pc);
        let a = get_reg($values, lhs_reg);
        let b = get_reg($values, rhs_reg);
        match (a.as_f64(), b.as_f64()) {
            (Ok(x), Ok(y)) => {
                let f: fn(i32, i32) -> i32 = $op;
                set_reg($values, dst, InterpValue::Boxed(Value::num(f(x as i32, y as i32) as f64)));
                false
            }
            _ => {
                let recv = a.to_value();
                let arg = b.to_value();
                try_operator_dispatch(
                    $vm, $fiber, recv, $method, &[arg], $pc, $values, dst,
                    $module_name, $bc,
                )?
            }
        }
    }};
}

// ---------------------------------------------------------------------------
// Fiber-based bytecode interpreter
// ---------------------------------------------------------------------------

/// Run the active fiber (vm.fiber) until it finishes, yields, or errors.
///
/// The fiber must have at least one MirCallFrame pushed before calling this.
/// Uses flat bytecode dispatch for efficient interpretation.
pub fn run_fiber(vm: &mut VM) -> Result<Value, RuntimeError> {
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
        let (func_id, mut pc, mut values, module_name) = unsafe {
            let frame = (*fiber).mir_frames.last_mut().unwrap();
            (
                frame.func_id,
                frame.pc,
                std::mem::take(&mut frame.values),
                frame.module_name.clone(),
            )
        };

        // Get bytecode (lazily compiled on first use).
        // Use raw pointer to avoid Arc clone overhead in the hot loop.
        let bc_ptr = vm
            .engine
            .ensure_bytecode(func_id)
            .ok_or_else(|| RuntimeError::Error("invalid func_id".into()))?;
        // SAFETY: bc_ptr is stable — bytecode is never freed once compiled.
        let bc = unsafe { &*bc_ptr };
        let code = &bc.code;

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

            debug_assert!((pc as usize) < code.len());
            let op = unsafe { *code.get_unchecked(pc as usize) };
            pc += 1;

            // SAFETY: op values come from our encoder, all valid Op discriminants
            match unsafe { Op::from_u8_unchecked(op) } {
                // =============================================================
                // Constants (3B-5B)
                // =============================================================
                Op::ConstNull => {
                    let dst = read_u16(code, &mut pc);
                    set_reg(&mut values, dst, InterpValue::Boxed(Value::null()));
                }
                Op::ConstTrue => {
                    let dst = read_u16(code, &mut pc);
                    set_reg(&mut values, dst, InterpValue::Boxed(Value::bool(true)));
                }
                Op::ConstFalse => {
                    let dst = read_u16(code, &mut pc);
                    set_reg(&mut values, dst, InterpValue::Boxed(Value::bool(false)));
                }
                Op::ConstNum => {
                    let dst = read_u16(code, &mut pc);
                    let idx = read_u16(code, &mut pc);
                    let val = match bc.constants[idx as usize] {
                        BcConst::F64(n) => InterpValue::Boxed(Value::num(n)),
                        BcConst::I64(n) => InterpValue::Boxed(Value::num(n as f64)),
                    };
                    set_reg(&mut values, dst, val);
                }
                Op::ConstF64 => {
                    let dst = read_u16(code, &mut pc);
                    let idx = read_u16(code, &mut pc);
                    let val = match bc.constants[idx as usize] {
                        BcConst::F64(n) => InterpValue::F64(n),
                        BcConst::I64(n) => InterpValue::F64(n as f64),
                    };
                    set_reg(&mut values, dst, val);
                }
                Op::ConstI64 => {
                    let dst = read_u16(code, &mut pc);
                    let idx = read_u16(code, &mut pc);
                    let val = match bc.constants[idx as usize] {
                        BcConst::I64(n) => InterpValue::I64(n),
                        BcConst::F64(n) => InterpValue::I64(n as i64),
                    };
                    set_reg(&mut values, dst, val);
                }
                Op::ConstString => {
                    let dst = read_u16(code, &mut pc);
                    let sym_idx = read_u16(code, &mut pc);
                    let sym = SymbolId::from_raw(sym_idx as u32);
                    let s = vm.interner.resolve(sym).to_string();
                    set_reg(&mut values, dst, InterpValue::Boxed(vm.new_string(s)));
                }

                // =============================================================
                // Module variables
                // =============================================================
                Op::GetModuleVar => {
                    let dst = read_u16(code, &mut pc);
                    let slot = read_u16(code, &mut pc) as usize;
                    let val = vm
                        .engine
                        .modules
                        .get(module_name.as_str())
                        .and_then(|m| m.vars.get(slot).copied())
                        .unwrap_or(Value::null());
                    set_reg(&mut values, dst, InterpValue::Boxed(val));
                }
                Op::SetModuleVar => {
                    let dst = read_u16(code, &mut pc);
                    let val_reg = read_u16(code, &mut pc);
                    let slot = read_u16(code, &mut pc) as usize;
                    let v = get_reg_val(&values, val_reg);
                    if let Some(entry) = vm.engine.modules.get_mut(module_name.as_str()) {
                        while entry.vars.len() <= slot {
                            entry.vars.push(Value::null());
                        }
                        entry.vars[slot] = v;
                    }
                    set_reg(&mut values, dst, InterpValue::Boxed(v));
                }

                // =============================================================
                // Upvalues
                // =============================================================
                Op::GetUpvalue => {
                    let dst = read_u16(code, &mut pc);
                    let idx = read_u16(code, &mut pc) as usize;
                    let frame_closure =
                        unsafe { (*fiber).mir_frames.last().unwrap().closure };
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
                    set_reg(&mut values, dst, InterpValue::Boxed(val));
                }
                Op::SetUpvalue => {
                    let dst = read_u16(code, &mut pc);
                    let val_reg = read_u16(code, &mut pc);
                    let idx = read_u16(code, &mut pc) as usize;
                    let v = get_reg_val(&values, val_reg);
                    let frame_closure =
                        unsafe { (*fiber).mir_frames.last().unwrap().closure };
                    if let Some(closure_ptr) = frame_closure {
                        let upvalues = unsafe { &mut (*closure_ptr).upvalues };
                        if idx < upvalues.len() {
                            unsafe { (*upvalues[idx]).set(v) };
                        }
                    }
                    set_reg(&mut values, dst, InterpValue::Boxed(v));
                }

                // =============================================================
                // Block parameters (pre-bound by branch; no-op if already set)
                // =============================================================
                Op::BlockParam => {
                    let dst = read_u16(code, &mut pc);
                    let _param_idx = read_u16(code, &mut pc);
                    // Already bound by branch instruction, skip if set
                    let idx = dst as usize;
                    if idx < values.len()
                        && !matches!(values[idx], InterpValue::Boxed(v) if v.is_undefined())
                    {
                        continue;
                    }
                    set_reg(&mut values, dst, InterpValue::Boxed(Value::null()));
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
                    set_reg(&mut values, dst, InterpValue::Boxed(val));
                }
                Op::SetStaticField => {
                    let dst = read_u16(code, &mut pc);
                    let val_reg = read_u16(code, &mut pc);
                    let sym_idx = read_u16(code, &mut pc);
                    let sym = SymbolId::from_raw(sym_idx as u32);
                    let v = get_reg_val(&values, val_reg);
                    let frame = unsafe { (*fiber).mir_frames.last().unwrap() };
                    if let Some(class_ptr) = frame.defining_class {
                        unsafe {
                            (*class_ptr).static_fields.insert(sym, v);
                        }
                    }
                    set_reg(&mut values, dst, InterpValue::Boxed(v));
                }

                // =============================================================
                // Unary ops (5B: op + dst + src)
                // =============================================================
                Op::Neg => {
                    let dst = read_u16(code, &mut pc);
                    let src = read_u16(code, &mut pc);
                    let a = get_reg(&values, src);
                    match a.as_f64() {
                        Ok(n) => set_reg(&mut values, dst, InterpValue::Boxed(Value::num(-n))),
                        Err(_) => {
                            let recv = a.to_value();
                            let result = try_operator_dispatch(
                                vm, fiber, recv, "-", &[], &mut pc, &mut values, dst,
                                &module_name, &bc,
                            )?;
                            if result { continue 'fiber_loop; }
                        }
                    }
                }
                Op::NegF64 => {
                    let dst = read_u16(code, &mut pc);
                    let src = read_u16(code, &mut pc);
                    let n = get_reg(&values, src).as_f64().map_err(RuntimeError::from)?;
                    set_reg(&mut values, dst, InterpValue::F64(-n));
                }
                Op::Not => {
                    let dst = read_u16(code, &mut pc);
                    let src = read_u16(code, &mut pc);
                    let a = get_reg(&values, src);
                    set_reg(
                        &mut values,
                        dst,
                        InterpValue::Boxed(Value::bool(!a.is_truthy())),
                    );
                }
                Op::BitNot => {
                    let dst = read_u16(code, &mut pc);
                    let src = read_u16(code, &mut pc);
                    let a = get_reg(&values, src);
                    match a.as_f64() {
                        Ok(n) => {
                            let i = n as i32;
                            set_reg(
                                &mut values,
                                dst,
                                InterpValue::Boxed(Value::num((!i) as f64)),
                            );
                        }
                        Err(_) => {
                            let recv = a.to_value();
                            let result = try_operator_dispatch(
                                vm, fiber, recv, "~", &[], &mut pc, &mut values, dst,
                                &module_name, &bc,
                            )?;
                            if result { continue 'fiber_loop; }
                        }
                    }
                }
                Op::Unbox => {
                    let dst = read_u16(code, &mut pc);
                    let src = read_u16(code, &mut pc);
                    let a = get_reg(&values, src);
                    let result = match a {
                        InterpValue::Boxed(val) => match val.as_num() {
                            Some(n) => InterpValue::F64(n),
                            None => {
                                return Err(RuntimeError::Error(
                                    "unbox: not a number".into(),
                                ))
                            }
                        },
                        InterpValue::F64(n) => InterpValue::F64(n),
                        InterpValue::I64(n) => InterpValue::F64(n as f64),
                        InterpValue::Bool(_) => {
                            return Err(RuntimeError::Error("unbox: bool".into()))
                        }
                    };
                    set_reg(&mut values, dst, result);
                }
                Op::BoxOp => {
                    let dst = read_u16(code, &mut pc);
                    let src = read_u16(code, &mut pc);
                    let val = get_reg(&values, src).to_value();
                    set_reg(&mut values, dst, InterpValue::Boxed(val));
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
                    let s = value_to_string(vm, get_reg_val(&values, src));
                    set_reg(&mut values, dst, InterpValue::Boxed(vm.new_string(s)));
                }
                Op::GuardNum => {
                    let dst = read_u16(code, &mut pc);
                    let src = read_u16(code, &mut pc);
                    let v = get_reg(&values, src);
                    match v {
                        InterpValue::Boxed(val) if val.is_num() => {
                            set_reg(&mut values, dst, v);
                        }
                        InterpValue::F64(_) | InterpValue::I64(_) => {
                            set_reg(&mut values, dst, v);
                        }
                        _ => return Err(RuntimeError::GuardFailed("Num")),
                    }
                }
                Op::GuardBool => {
                    let dst = read_u16(code, &mut pc);
                    let src = read_u16(code, &mut pc);
                    let v = get_reg(&values, src);
                    match v {
                        InterpValue::Boxed(val) if val.is_bool() => {
                            set_reg(&mut values, dst, v);
                        }
                        InterpValue::Bool(_) => {
                            set_reg(&mut values, dst, v);
                        }
                        _ => return Err(RuntimeError::GuardFailed("Bool")),
                    }
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
                    let value = get_reg_val(&values, val_reg);
                    let mut cls = vm.class_of(value);
                    let mut matched = false;
                    while !cls.is_null() {
                        if unsafe { (*cls).name } == class_sym {
                            matched = true;
                            break;
                        }
                        cls = unsafe { (*cls).superclass };
                    }
                    set_reg(&mut values, dst, InterpValue::Boxed(Value::bool(matched)));
                }

                // =============================================================
                // Field access
                // =============================================================
                Op::GetField => {
                    let dst = read_u16(code, &mut pc);
                    let recv_reg = read_u16(code, &mut pc);
                    let field_idx = read_u16(code, &mut pc) as usize;
                    let recv = get_reg_val(&values, recv_reg);
                    let val = if let Some(ptr) = recv.as_object() {
                        let inst = ptr as *const ObjInstance;
                        unsafe { (*inst).get_field(field_idx) }.unwrap_or(Value::null())
                    } else {
                        Value::null()
                    };
                    set_reg(&mut values, dst, InterpValue::Boxed(val));
                }
                Op::SetField => {
                    let dst = read_u16(code, &mut pc);
                    let recv_reg = read_u16(code, &mut pc);
                    let field_idx = read_u16(code, &mut pc) as usize;
                    let val_reg = read_u16(code, &mut pc);
                    let recv = get_reg_val(&values, recv_reg);
                    let val = get_reg_val(&values, val_reg);
                    if let Some(ptr) = recv.as_object() {
                        let inst = ptr as *mut ObjInstance;
                        unsafe { (*inst).set_field(field_idx, val) };
                    }
                    set_reg(&mut values, dst, InterpValue::Boxed(val));
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
                    let n = get_reg(&values, src).as_f64().map_err(RuntimeError::from)?;
                    set_reg(&mut values, dst, InterpValue::F64(math_op.apply(n)));
                }
                Op::MathBinaryF64 => {
                    let dst = read_u16(code, &mut pc);
                    let lhs = read_u16(code, &mut pc);
                    let rhs = read_u16(code, &mut pc);
                    let math_op_byte = read_u8(code, &mut pc);
                    let math_op = crate::mir::bytecode::u8_to_math_binary(math_op_byte);
                    let a = get_reg(&values, lhs).as_f64().map_err(RuntimeError::from)?;
                    let b = get_reg(&values, rhs).as_f64().map_err(RuntimeError::from)?;
                    set_reg(&mut values, dst, InterpValue::F64(math_op.apply(a, b)));
                }

                // =============================================================
                // Binary arithmetic (7B: op + dst + lhs + rhs)
                // =============================================================
                Op::Add => { if bc_boxed_binop!(vm, fiber, &mut pc, &mut values, &module_name, &bc, |x: f64, y: f64| x + y, "+(_)") { continue 'fiber_loop; } }
                Op::Sub => { if bc_boxed_binop!(vm, fiber, &mut pc, &mut values, &module_name, &bc, |x: f64, y: f64| x - y, "-(_)") { continue 'fiber_loop; } }
                Op::Mul => { if bc_boxed_binop!(vm, fiber, &mut pc, &mut values, &module_name, &bc, |x: f64, y: f64| x * y, "*(_)") { continue 'fiber_loop; } }
                Op::Div => { if bc_boxed_binop!(vm, fiber, &mut pc, &mut values, &module_name, &bc, |x: f64, y: f64| x / y, "/(_)") { continue 'fiber_loop; } }
                Op::Mod => { if bc_boxed_binop!(vm, fiber, &mut pc, &mut values, &module_name, &bc, |x: f64, y: f64| x % y, "%(_)") { continue 'fiber_loop; } }

                Op::AddF64 => { bc_f64_binop!(&mut pc, &mut values, |x, y| x + y, &bc); }
                Op::SubF64 => { bc_f64_binop!(&mut pc, &mut values, |x, y| x - y, &bc); }
                Op::MulF64 => { bc_f64_binop!(&mut pc, &mut values, |x, y| x * y, &bc); }
                Op::DivF64 => { bc_f64_binop!(&mut pc, &mut values, |x, y| x / y, &bc); }
                Op::ModF64 => { bc_f64_binop!(&mut pc, &mut values, |x, y| x % y, &bc); }

                // =============================================================
                // Comparisons
                // =============================================================
                Op::CmpLt => { if bc_boxed_cmp!(vm, fiber, &mut pc, &mut values, &module_name, &bc, |x: f64, y: f64| x < y, "<(_)") { continue 'fiber_loop; } }
                Op::CmpGt => { if bc_boxed_cmp!(vm, fiber, &mut pc, &mut values, &module_name, &bc, |x: f64, y: f64| x > y, ">(_)") { continue 'fiber_loop; } }
                Op::CmpLe => { if bc_boxed_cmp!(vm, fiber, &mut pc, &mut values, &module_name, &bc, |x: f64, y: f64| x <= y, "<=(_)") { continue 'fiber_loop; } }
                Op::CmpGe => { if bc_boxed_cmp!(vm, fiber, &mut pc, &mut values, &module_name, &bc, |x: f64, y: f64| x >= y, ">=(_)") { continue 'fiber_loop; } }
                Op::CmpEq => {
                    let dst = read_u16(code, &mut pc);
                    let lhs_reg = read_u16(code, &mut pc);
                    let rhs_reg = read_u16(code, &mut pc);
                    let lhs = get_reg_val(&values, lhs_reg);
                    let rhs = get_reg_val(&values, rhs_reg);
                    // For non-instance objects, try operator dispatch
                    if lhs.is_object()
                        && !lhs.is_string_object()
                        && !rhs.is_null()
                        && !rhs.is_bool()
                        && !rhs.is_num()
                    {
                        let result = try_operator_dispatch(
                            vm, fiber, lhs, "==(_)", &[rhs], &mut pc, &mut values, dst,
                            &module_name, &bc,
                        )?;
                        if result { continue 'fiber_loop; }
                    } else {
                        set_reg(&mut values, dst, InterpValue::Boxed(Value::bool(lhs == rhs)));
                    }
                }
                Op::CmpNe => {
                    let dst = read_u16(code, &mut pc);
                    let lhs_reg = read_u16(code, &mut pc);
                    let rhs_reg = read_u16(code, &mut pc);
                    let lhs = get_reg_val(&values, lhs_reg);
                    let rhs = get_reg_val(&values, rhs_reg);
                    if lhs.is_object()
                        && !lhs.is_string_object()
                        && !rhs.is_null()
                        && !rhs.is_bool()
                        && !rhs.is_num()
                    {
                        let result = try_operator_dispatch(
                            vm, fiber, lhs, "!=(_)", &[rhs], &mut pc, &mut values, dst,
                            &module_name, &bc,
                        )?;
                        if result { continue 'fiber_loop; }
                    } else {
                        set_reg(&mut values, dst, InterpValue::Boxed(Value::bool(lhs != rhs)));
                    }
                }

                Op::CmpLtF64 => { bc_f64_cmp!(&mut pc, &mut values, |x, y| x < y, &bc); }
                Op::CmpGtF64 => { bc_f64_cmp!(&mut pc, &mut values, |x, y| x > y, &bc); }
                Op::CmpLeF64 => { bc_f64_cmp!(&mut pc, &mut values, |x, y| x <= y, &bc); }
                Op::CmpGeF64 => { bc_f64_cmp!(&mut pc, &mut values, |x, y| x >= y, &bc); }

                // =============================================================
                // Bitwise ops
                // =============================================================
                Op::BitAnd => { if bc_bitwise_binop!(vm, fiber, &mut pc, &mut values, &module_name, &bc, |x: i32, y: i32| x & y, "&(_)") { continue 'fiber_loop; } }
                Op::BitOr => { if bc_bitwise_binop!(vm, fiber, &mut pc, &mut values, &module_name, &bc, |x: i32, y: i32| x | y, "|(_)") { continue 'fiber_loop; } }
                Op::BitXor => {
                    let dst = read_u16(code, &mut pc);
                    let lhs_reg = read_u16(code, &mut pc);
                    let rhs_reg = read_u16(code, &mut pc);
                    let a = get_reg(&values, lhs_reg);
                    let b = get_reg(&values, rhs_reg);
                    match (a.as_f64(), b.as_f64()) {
                        (Ok(x), Ok(y)) => {
                            set_reg(&mut values, dst, InterpValue::Boxed(Value::num(((x as i32) ^ (y as i32)) as f64)));
                        }
                        _ => {
                            return Err(RuntimeError::Error("bitwise xor: not numbers".into()));
                        }
                    }
                }
                Op::Shl => { if bc_bitwise_binop!(vm, fiber, &mut pc, &mut values, &module_name, &bc, |x: i32, y: i32| x << (y & 31), "<<(_)") { continue 'fiber_loop; } }
                Op::Shr => { if bc_bitwise_binop!(vm, fiber, &mut pc, &mut values, &module_name, &bc, |x: i32, y: i32| x >> (y & 31), ">>(_)") { continue 'fiber_loop; } }

                // =============================================================
                // Range
                // =============================================================
                Op::MakeRange => {
                    let dst = read_u16(code, &mut pc);
                    let from_reg = read_u16(code, &mut pc);
                    let to_reg = read_u16(code, &mut pc);
                    let inclusive = read_u8(code, &mut pc) != 0;
                    let f = get_reg(&values, from_reg)
                        .as_f64()
                        .map_err(RuntimeError::from)?;
                    let t = get_reg(&values, to_reg)
                        .as_f64()
                        .map_err(RuntimeError::from)?;
                    set_reg(
                        &mut values,
                        dst,
                        InterpValue::Boxed(vm.new_range(f, t, inclusive)),
                    );
                }

                // =============================================================
                // Method calls
                // =============================================================
                Op::Call => {
                    let dst = read_u16(code, &mut pc);
                    let recv_reg = read_u16(code, &mut pc);
                    let method_idx = read_u16(code, &mut pc);
                    let argc = read_u8(code, &mut pc) as usize;
                    let method = SymbolId::from_raw(method_idx as u32);

                    let recv_val = get_reg_val(&values, recv_reg);
                    let mut arg_vals: SmallVec<[Value; 8]> =
                        SmallVec::with_capacity(1 + argc);
                    arg_vals.push(recv_val);
                    for _ in 0..argc {
                        let arg_reg = read_u16(code, &mut pc);
                        arg_vals.push(get_reg_val(&values, arg_reg));
                    }

                    // Save pc before any call dispatch (for error reporting)
                    unsafe {
                        if let Some(frame) = (*fiber).mir_frames.last_mut() {
                            frame.pc = pc;
                        }
                    }

                    // Fast path: if receiver is a closure and method is call(...),
                    // dispatch directly. Check first char 'c' to avoid interning.
                    if recv_val.is_object() {
                        let method_str = vm.interner.resolve(method);
                        if method_str.as_bytes().first() == Some(&b'c')
                            && method_str.starts_with("call")
                        {
                            let ptr = recv_val.as_object().unwrap();
                            let header = ptr as *const ObjHeader;
                            if unsafe { (*header).obj_type } == ObjType::Closure {
                                let closure_ptr = ptr as *mut ObjClosure;
                                dispatch_closure_bc(
                                    vm, fiber, closure_ptr, &arg_vals[1..],
                                    pc, values, &module_name,
                                    ValueId(dst as u32), None,
                                )?;
                                continue 'fiber_loop;
                            }
                        }
                    }

                    let class = vm.class_of(recv_val);

                    // For class receivers, try static method first (common case
                    // for Foo.bar() calls). Uses stack buffer + lookup to avoid heap alloc.
                    let (method_entry, defining_class) =
                        if class == vm.class_class {
                            let recv_class_ptr =
                                recv_val.as_object().unwrap() as *mut ObjClass;
                            let method_str = vm.interner.resolve(method);
                            let prefix = b"static:";
                            let total = prefix.len() + method_str.len();
                            let mut buf = [0u8; 128];
                            let found = if total <= buf.len() {
                                buf[..prefix.len()].copy_from_slice(prefix);
                                buf[prefix.len()..total]
                                    .copy_from_slice(method_str.as_bytes());
                                let static_str = unsafe {
                                    std::str::from_utf8_unchecked(&buf[..total])
                                };
                                vm.interner.lookup(static_str).and_then(|sym| {
                                    unsafe { (*recv_class_ptr).find_method(sym).cloned() }
                                })
                            } else {
                                None
                            };
                            match found {
                                Some(m) => (Some(m), Some(recv_class_ptr)),
                                None => {
                                    match unsafe { find_method_with_class(class, method) } {
                                        Some((m, c)) => (Some(m), Some(c)),
                                        None => (None, None),
                                    }
                                }
                            }
                        } else {
                            match unsafe { find_method_with_class(class, method) } {
                                Some((m, c)) => (Some(m), Some(c)),
                                None => (None, None),
                            }
                        };

                    match method_entry {
                        Some(Method::Native(func)) => {
                            let result = func(vm, &arg_vals);
                            if vm.has_error {
                                vm.has_error = false;
                                let fiber_is_try = unsafe { (*fiber).is_try };
                                if fiber_is_try {
                                    let err_msg =
                                        vm.last_error.take().unwrap_or_default();
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
                                    vm, fiber, action, pc, values,
                                    ValueId(dst as u32),
                                )?;
                                continue 'fiber_loop;
                            }
                            set_reg(&mut values, dst, InterpValue::Boxed(result));
                        }
                        Some(Method::Closure(closure_ptr)) => {
                            dispatch_closure_bc(
                                vm, fiber, closure_ptr, &arg_vals,
                                pc, values, &module_name,
                                ValueId(dst as u32), defining_class,
                            )?;
                            continue 'fiber_loop;
                        }
                        Some(Method::Constructor(closure_ptr)) => {
                            let recv_class =
                                recv_val.as_object().unwrap() as *mut ObjClass;
                            let instance = vm.gc.alloc_instance(recv_class);
                            let instance_val = Value::object(instance as *mut u8);
                            let mut ctor_args = arg_vals.clone();
                            ctor_args[0] = instance_val;
                            dispatch_closure_bc(
                                vm, fiber, closure_ptr, &ctor_args,
                                pc, values, &module_name,
                                ValueId(dst as u32), defining_class,
                            )?;
                            continue 'fiber_loop;
                        }
                        None => {
                            let method_name =
                                vm.interner.resolve(method).to_string();
                            let class_name = vm.class_name_of(recv_val);
                            return Err(RuntimeError::MethodNotFound {
                                class_name,
                                method: method_name,
                            });
                        }
                    }
                }

                Op::SuperCall => {
                    let dst = read_u16(code, &mut pc);
                    let method_idx = read_u16(code, &mut pc);
                    let argc = read_u8(code, &mut pc) as usize;
                    let method = SymbolId::from_raw(method_idx as u32);

                    let mut arg_vals: SmallVec<[Value; 8]> =
                        SmallVec::with_capacity(argc);
                    for _ in 0..argc {
                        let arg_reg = read_u16(code, &mut pc);
                        arg_vals.push(get_reg_val(&values, arg_reg));
                    }

                    let defining =
                        unsafe { (*fiber).mir_frames.last().unwrap().defining_class };
                    let superclass = defining.and_then(|cls| {
                        let sup = unsafe { (*cls).superclass };
                        if sup.is_null() { None } else { Some(sup) }
                    });

                    if let Some(super_cls) = superclass {
                        let found =
                            unsafe { find_method_with_class(super_cls, method) };
                        let (method_entry, found_class) = match found {
                            Some((m, c)) => (Some(m), Some(c)),
                            None => {
                                let method_name =
                                    vm.interner.resolve(method).to_string();
                                let static_name =
                                    format!("static:{}", method_name);
                                let static_sym = vm.interner.intern(&static_name);
                                match unsafe {
                                    find_method_with_class(super_cls, static_sym)
                                } {
                                    Some((m, c)) => (Some(m), Some(c)),
                                    None => (None, None),
                                }
                            }
                        };
                        match method_entry {
                            Some(Method::Native(func)) => {
                                set_reg(
                                    &mut values,
                                    dst,
                                    InterpValue::Boxed(func(vm, &arg_vals)),
                                );
                            }
                            Some(Method::Closure(closure_ptr)) => {
                                dispatch_closure_bc(
                                    vm, fiber, closure_ptr, &arg_vals,
                                    pc, values, &module_name,
                                    ValueId(dst as u32), found_class,
                                )?;
                                continue 'fiber_loop;
                            }
                            Some(Method::Constructor(closure_ptr)) => {
                                dispatch_closure_bc(
                                    vm, fiber, closure_ptr, &arg_vals,
                                    pc, values, &module_name,
                                    ValueId(dst as u32), found_class,
                                )?;
                                continue 'fiber_loop;
                            }
                            None => {
                                let method_name =
                                    vm.interner.resolve(method).to_string();
                                let class_name = unsafe {
                                    vm.interner.resolve((*super_cls).name).to_string()
                                };
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
                    let recv = get_reg_val(&values, recv_reg);
                    let mut all_args: SmallVec<[Value; 4]> =
                        SmallVec::with_capacity(1 + argc);
                    all_args.push(recv);
                    for _ in 0..argc {
                        let arg_reg = read_u16(code, &mut pc);
                        all_args.push(get_reg_val(&values, arg_reg));
                    }
                    let class = vm.class_of(recv);
                    let sym = subscript_get_sym(vm, argc);
                    match unsafe { (*class).find_method(sym).cloned() } {
                        Some(Method::Native(func)) => {
                            set_reg(
                                &mut values,
                                dst,
                                InterpValue::Boxed(func(vm, &all_args)),
                            );
                        }
                        Some(Method::Closure(closure_ptr)) => {
                            dispatch_closure_bc(
                                vm, fiber, closure_ptr, &all_args,
                                pc, values, &module_name,
                                ValueId(dst as u32), None,
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
                    let recv = get_reg_val(&values, recv_reg);
                    let mut all_args: SmallVec<[Value; 4]> =
                        SmallVec::with_capacity(2 + argc);
                    all_args.push(recv);
                    for _ in 0..argc {
                        let arg_reg = read_u16(code, &mut pc);
                        all_args.push(get_reg_val(&values, arg_reg));
                    }
                    let val_reg = read_u16(code, &mut pc);
                    all_args.push(get_reg_val(&values, val_reg));
                    let class = vm.class_of(recv);
                    let sym = subscript_set_sym(vm, argc);
                    match unsafe { (*class).find_method(sym).cloned() } {
                        Some(Method::Native(func)) => {
                            set_reg(
                                &mut values,
                                dst,
                                InterpValue::Boxed(func(vm, &all_args)),
                            );
                        }
                        Some(Method::Closure(closure_ptr)) => {
                            dispatch_closure_bc(
                                vm, fiber, closure_ptr, &all_args,
                                pc, values, &module_name,
                                ValueId(dst as u32), None,
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
                        elements.push(get_reg_val(&values, reg));
                    }
                    set_reg(
                        &mut values,
                        dst,
                        InterpValue::Boxed(vm.new_list(elements)),
                    );
                }
                Op::MakeMap => {
                    let dst = read_u16(code, &mut pc);
                    let count = read_u8(code, &mut pc) as usize;
                    let map_val = vm.new_map();
                    if count > 0 {
                        let map_ptr = map_val.as_object().unwrap()
                            as *mut crate::runtime::object::ObjMap;
                        for _ in 0..count {
                            let k_reg = read_u16(code, &mut pc);
                            let v_reg = read_u16(code, &mut pc);
                            let key = get_reg_val(&values, k_reg);
                            let val = get_reg_val(&values, v_reg);
                            unsafe { (*map_ptr).set(key, val) };
                        }
                    }
                    set_reg(&mut values, dst, InterpValue::Boxed(map_val));
                }
                Op::StringConcat => {
                    let dst = read_u16(code, &mut pc);
                    let count = read_u8(code, &mut pc) as usize;
                    let mut result = String::new();
                    for _ in 0..count {
                        let reg = read_u16(code, &mut pc);
                        result.push_str(&value_to_string(
                            vm,
                            get_reg_val(&values, reg),
                        ));
                    }
                    set_reg(
                        &mut values,
                        dst,
                        InterpValue::Boxed(vm.new_string(result)),
                    );
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
                    let fn_ptr =
                        vm.gc
                            .alloc_fn(closure_name, arity, uv_count as u16, fn_id);
                    unsafe {
                        (*fn_ptr).header.class = vm.fn_class;
                    }
                    let closure_ptr = vm.gc.alloc_closure(fn_ptr);
                    unsafe {
                        (*closure_ptr).header.class = vm.fn_class;
                    }

                    for i in 0..uv_count {
                        let uv_reg = read_u16(code, &mut pc);
                        let captured_val = get_reg_val(&values, uv_reg);
                        let uv_obj = vm.gc.alloc_upvalue(std::ptr::null_mut());
                        unsafe {
                            (*uv_obj).closed = captured_val;
                            (*uv_obj).location = &mut (*uv_obj).closed as *mut Value;
                            if i < (*closure_ptr).upvalues.len() {
                                (&mut (*closure_ptr).upvalues)[i] = uv_obj;
                            }
                        }
                    }

                    set_reg(
                        &mut values,
                        dst,
                        InterpValue::Boxed(Value::object(closure_ptr as *mut u8)),
                    );
                }

                // =============================================================
                // Terminators
                // =============================================================
                Op::Return => {
                    let src = read_u16(code, &mut pc);
                    let return_val = get_reg_val(&values, src);
                    let return_dst =
                        unsafe { (*fiber).mir_frames.last().unwrap().return_dst };
                    unsafe { (*fiber).mir_frames.pop() };
                    // Recycle the callee's register file
                    values.clear();
                    vm.register_pool.push(values);
                    if unsafe { (*fiber).mir_frames.is_empty() } {
                        unsafe { (*fiber).state = FiberState::Done };
                        let caller = unsafe { (*fiber).caller };
                        if !caller.is_null() {
                            unsafe { (*fiber).caller = std::ptr::null_mut() };
                            resume_caller(vm, caller, return_val);
                            continue 'fiber_loop;
                        }
                        return Ok(return_val);
                    }
                    if let Some(dst) = return_dst {
                        unsafe {
                            let frame = (*fiber).mir_frames.last_mut().unwrap();
                            set_reg(
                                &mut frame.values,
                                dst.0 as u16,
                                InterpValue::Boxed(return_val),
                            );
                        }
                    }
                    continue 'fiber_loop;
                }
                Op::ReturnNull => {
                    let return_dst =
                        unsafe { (*fiber).mir_frames.last().unwrap().return_dst };
                    unsafe { (*fiber).mir_frames.pop() };
                    // Recycle the callee's register file
                    values.clear();
                    vm.register_pool.push(values);
                    if unsafe { (*fiber).mir_frames.is_empty() } {
                        unsafe { (*fiber).state = FiberState::Done };
                        let caller = unsafe { (*fiber).caller };
                        if !caller.is_null() {
                            unsafe { (*fiber).caller = std::ptr::null_mut() };
                            resume_caller(vm, caller, Value::null());
                            continue 'fiber_loop;
                        }
                        return Ok(Value::null());
                    }
                    if let Some(dst) = return_dst {
                        unsafe {
                            let frame = (*fiber).mir_frames.last_mut().unwrap();
                            set_reg(
                                &mut frame.values,
                                dst.0 as u16,
                                InterpValue::Boxed(Value::null()),
                            );
                        }
                    }
                    continue 'fiber_loop;
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

                    if cond.is_truthy() {
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
pub fn new_register_file(mir: &crate::mir::MirFunction) -> Vec<InterpValue> {
    vec![UNDEF; mir.next_value as usize]
}

/// Fill a recycled register file with UNDEF up to `count` slots.
#[inline(always)]
fn new_values_fill_undef(values: &mut [InterpValue], count: usize) {
    for v in values[..count].iter_mut() {
        *v = UNDEF;
    }
}

/// Dispatch a closure call in bytecode mode: save current frame, push new frame.
#[allow(clippy::too_many_arguments)]
fn dispatch_closure_bc(
    vm: &mut VM,
    fiber: *mut ObjFiber,
    closure_ptr: *mut ObjClosure,
    arg_vals: &[Value],
    pc: u32,
    values: Vec<InterpValue>,
    module_name: &Rc<String>,
    return_dst: ValueId,
    defining_class: Option<*mut ObjClass>,
) -> Result<(), RuntimeError> {
    let fn_ptr = unsafe { (*closure_ptr).function };
    let target_func_id = FuncId(unsafe { (*fn_ptr).fn_id });

    // Tier-up profiling: record call and queue background compilation if hot
    let should_tier_up = vm.engine.record_call(target_func_id);
    if should_tier_up {
        vm.engine.request_tier_up(target_func_id, &vm.interner);
    }

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
            new_values[dst_reg as usize] = InterpValue::Boxed(arg_vals[param_idx as usize]);
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
        });
    }

    Ok(())
}

/// Resume a caller fiber with a value.
fn resume_caller(vm: &mut VM, caller: *mut ObjFiber, value: Value) {
    unsafe {
        if let Some(dst) = (*caller).resume_value_dst.take() {
            if let Some(frame) = (*caller).mir_frames.last_mut() {
                set_reg(&mut frame.values, dst.0 as u16, InterpValue::Boxed(value));
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
    values: Vec<InterpValue>,
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
                            set_reg(
                                &mut frame.values,
                                dst.0 as u16,
                                InterpValue::Boxed(value),
                            );
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
                            set_reg(
                                &mut frame.values,
                                dst.0 as u16,
                                InterpValue::Boxed(value),
                            );
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
    values: &mut Vec<InterpValue>,
    dst: u16,
    module_name: &Rc<String>,
    _bc: &BytecodeFunction,
) -> Result<bool, RuntimeError> {
    let method_sym = vm.interner.intern(method_str);
    let class = vm.class_of(recv);
    let method_entry = unsafe { (*class).find_method(method_sym).cloned() };
    match method_entry {
        Some(Method::Native(func)) => {
            let mut arg_vals: SmallVec<[Value; 4]> = SmallVec::with_capacity(1 + args.len());
            arg_vals.push(recv);
            arg_vals.extend_from_slice(args);
            let result = func(vm, &arg_vals);
            set_reg(values, dst, InterpValue::Boxed(result));
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
    let mut c = cls;
    while !c.is_null() {
        if let Some(m) = (*c).methods.get(&method) {
            return Some((m.clone(), c));
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
            });
        }
        vm.fiber = fiber;

        let _ = run_fiber(&mut vm).unwrap();
        assert_eq!(unsafe { (*fiber).state }, FiberState::Done);
        assert_eq!(unsafe { (*fiber).mir_frames.len() }, 0);
    }
}
