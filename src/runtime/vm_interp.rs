/// Fiber-based MIR interpreter with VM access.
///
/// All execution happens within a Fiber context. The interpreter loop
/// runs on the active fiber's `mir_frames` stack:
/// - Call instructions push new frames onto the fiber
/// - Return pops a frame and passes the result to the caller
/// - When the last frame is popped, the fiber is Done
///
/// Pure instruction evaluation (arithmetic, comparisons, guards, box/unbox,
/// bitwise, constants, moves) is delegated to `mir::interp::eval_pure_instruction`.
/// This module only handles VM-specific instructions: calls, collections,
/// module vars, string allocation, fields, closures.

use std::collections::HashMap;

use crate::mir::interp::{eval_pure_instruction, InterpError, InterpValue};
use crate::mir::{BlockId, Instruction, MirFunction, Terminator, ValueId};
use crate::runtime::object::*;
use crate::runtime::value::Value;
use crate::runtime::vm::VM;

// ---------------------------------------------------------------------------
// Runtime errors
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum RuntimeError {
    MethodNotFound { class_name: String, method: String },
    GuardFailed(&'static str),
    Unsupported(String),
    StepLimitExceeded,
    Error(String),
    Unreachable,
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

const MAX_STEPS: usize = 1_000_000;

// ---------------------------------------------------------------------------
// Fiber-based MIR interpreter
// ---------------------------------------------------------------------------

/// Run the active fiber (vm.fiber) until it finishes, yields, or errors.
///
/// The fiber must have at least one MirCallFrame pushed before calling this.
pub fn run_fiber(vm: &mut VM) -> Result<Value, RuntimeError> {
    let fiber = vm.fiber;
    if fiber.is_null() {
        return Err(RuntimeError::Error("no active fiber".into()));
    }

    unsafe { (*fiber).state = FiberState::Running; }

    let mut steps: usize = 0;

    // Outer loop: re-entered when we push/pop a call frame
    'fiber_loop: loop {
        let frame_count = unsafe { (*fiber).mir_frames.len() };
        if frame_count == 0 {
            unsafe { (*fiber).state = FiberState::Done; }
            return Ok(Value::null());
        }

        // Load execution state from the top frame into locals.
        let (func_id, mut current_block, mut values, module_name, start_ip) = unsafe {
            let frame = (*fiber).mir_frames.last_mut().unwrap();
            (
                frame.func_id,
                frame.current_block,
                std::mem::take(&mut frame.values),
                frame.module_name.clone(),
                std::mem::replace(&mut frame.ip, 0),
            )
        };

        // Grab an Arc to the MIR — doesn't borrow the engine.
        let mir = vm.engine.get_mir(func_id)
            .ok_or_else(|| RuntimeError::Error("invalid func_id".into()))?;

        // Inner loop: walks basic blocks within the current function
        let mut first_block = true;
        loop {
            let block = &mir.blocks[current_block.0 as usize];
            let instructions = &block.instructions;
            let terminator = &block.terminator;

            let ip_start = if first_block { start_ip } else { 0 };
            first_block = false;

            let mut ip = ip_start;
            while ip < instructions.len() {
                let (dst, inst) = &instructions[ip];
                ip += 1;

                steps += 1;
                if steps > MAX_STEPS {
                    return Err(RuntimeError::StepLimitExceeded);
                }

                use Instruction::*;
                let result: InterpValue = match inst {
                    // -- BlockParam (pre-bound by bind_block_params) --
                    BlockParam(_) => {
                        if values.contains_key(dst) {
                            continue;
                        }
                        InterpValue::Boxed(Value::null())
                    }

                    // -- VM-specific: string allocation --
                    ConstString(idx) => {
                        let sym = crate::intern::SymbolId::from_raw(*idx);
                        let s = vm.interner.resolve(sym).to_string();
                        InterpValue::Boxed(vm.new_string(s))
                    }

                    // -- Module variables (through VM engine) --
                    GetModuleVar(slot) => {
                        let val = vm.engine.modules
                            .get(&module_name)
                            .and_then(|m| m.vars.get(*slot as usize).copied())
                            .unwrap_or(Value::null());
                        InterpValue::Boxed(val)
                    }
                    SetModuleVar(slot, val_id) => {
                        let v = get_val(&values, *val_id)?.to_value();
                        let idx = *slot as usize;
                        if let Some(entry) = vm.engine.modules.get_mut(&module_name) {
                            while entry.vars.len() <= idx {
                                entry.vars.push(Value::null());
                            }
                            entry.vars[idx] = v;
                        }
                        InterpValue::Boxed(v)
                    }

                    // -- Method calls --
                    Call { receiver, method, args } => {
                        let recv_val = get_val(&values, *receiver)?.to_value();
                        let mut arg_vals = vec![recv_val];
                        for arg in args {
                            arg_vals.push(get_val(&values, *arg)?.to_value());
                        }

                        let class = vm.class_of(recv_val);
                        let method_entry = unsafe { (*class).find_method(*method).cloned() };

                        match method_entry {
                            Some(Method::Native(func)) => {
                                let result = func(vm, &arg_vals);
                                if vm.has_error {
                                    vm.has_error = false;
                                    return Err(RuntimeError::Error(
                                        "runtime error in native method".into(),
                                    ));
                                }
                                InterpValue::Boxed(result)
                            }
                            Some(Method::Closure(_)) => {
                                return Err(RuntimeError::Unsupported("closure method dispatch".into()));
                            }
                            None => {
                                let method_name = vm.interner.resolve(*method).to_string();
                                let class_name = vm.class_name_of(recv_val);
                                return Err(RuntimeError::MethodNotFound { class_name, method: method_name });
                            }
                        }
                    }
                    SuperCall { .. } => return Err(RuntimeError::Unsupported("super calls".into())),

                    // -- Subscript --
                    SubscriptGet { receiver, args } => {
                        let recv = get_val(&values, *receiver)?.to_value();
                        let mut all_args = vec![recv];
                        for a in args {
                            all_args.push(get_val(&values, *a)?.to_value());
                        }
                        let class = vm.class_of(recv);
                        let sym = vm.interner.intern("[_]");
                        match unsafe { (*class).find_method(sym).cloned() } {
                            Some(Method::Native(func)) => InterpValue::Boxed(func(vm, &all_args)),
                            _ => return Err(RuntimeError::Unsupported("subscript get".into())),
                        }
                    }
                    SubscriptSet { receiver, args, value } => {
                        let recv = get_val(&values, *receiver)?.to_value();
                        let mut all_args = vec![recv];
                        for a in args {
                            all_args.push(get_val(&values, *a)?.to_value());
                        }
                        all_args.push(get_val(&values, *value)?.to_value());
                        let class = vm.class_of(recv);
                        let sym = vm.interner.intern("[_]=(_)");
                        match unsafe { (*class).find_method(sym).cloned() } {
                            Some(Method::Native(func)) => InterpValue::Boxed(func(vm, &all_args)),
                            _ => return Err(RuntimeError::Unsupported("subscript set".into())),
                        }
                    }

                    // -- Collections --
                    MakeList(elems) => {
                        let elements: Vec<Value> = elems.iter()
                            .map(|e| get_val(&values, *e).map(|v| v.to_value()))
                            .collect::<Result<_, _>>()?;
                        InterpValue::Boxed(vm.new_list(elements))
                    }
                    MakeMap(_) => InterpValue::Boxed(vm.new_map()),
                    MakeRange(from, to, inclusive) => {
                        let f = get_val(&values, *from)?.as_f64()
                            .map_err(RuntimeError::from)?;
                        let t = get_val(&values, *to)?.as_f64()
                            .map_err(RuntimeError::from)?;
                        InterpValue::Boxed(vm.new_range(f, t, *inclusive))
                    }

                    // -- String operations --
                    ToString(a) => {
                        let s = value_to_string(vm, get_val(&values, *a)?.to_value());
                        InterpValue::Boxed(vm.new_string(s))
                    }
                    StringConcat(parts) => {
                        let mut result = String::new();
                        for p in parts {
                            result.push_str(&value_to_string(vm, get_val(&values, *p)?.to_value()));
                        }
                        InterpValue::Boxed(vm.new_string(result))
                    }

                    // -- Type checking --
                    IsType(_, _) => InterpValue::Boxed(Value::bool(false)),

                    // -- Guard that needs VM class system --
                    GuardClass(a, _cls) => get_val(&values, *a)?,

                    // -- Not yet implemented --
                    GetField(..) => return Err(RuntimeError::Unsupported("field access".into())),
                    SetField(..) => return Err(RuntimeError::Unsupported("field set".into())),
                    GetUpvalue(_) => return Err(RuntimeError::Unsupported("upvalue get".into())),
                    SetUpvalue(_, _) => return Err(RuntimeError::Unsupported("upvalue set".into())),
                    MakeClosure { .. } => return Err(RuntimeError::Unsupported("closure creation".into())),

                    // -- Everything else: delegate to shared MIR evaluator --
                    _ => eval_pure_instruction(inst, &values)?,
                };

                values.insert(*dst, result);
            }

            // -- Terminator --
            steps += 1;
            if steps > MAX_STEPS {
                return Err(RuntimeError::StepLimitExceeded);
            }

            match &terminator {
                Terminator::Return(v) => {
                    let return_val = get_val(&values, *v)?.to_value();
                    let return_dst = unsafe { (*fiber).mir_frames.last().unwrap().return_dst };
                    unsafe { (*fiber).mir_frames.pop(); }
                    if unsafe { (*fiber).mir_frames.is_empty() } {
                        unsafe { (*fiber).state = FiberState::Done; }
                        return Ok(return_val);
                    }
                    if let Some(dst) = return_dst {
                        unsafe {
                            (*fiber).mir_frames.last_mut().unwrap()
                                .values.insert(dst, InterpValue::Boxed(return_val));
                        }
                    }
                    continue 'fiber_loop;
                }
                Terminator::ReturnNull => {
                    let return_dst = unsafe { (*fiber).mir_frames.last().unwrap().return_dst };
                    unsafe { (*fiber).mir_frames.pop(); }
                    if unsafe { (*fiber).mir_frames.is_empty() } {
                        unsafe { (*fiber).state = FiberState::Done; }
                        return Ok(Value::null());
                    }
                    if let Some(dst) = return_dst {
                        unsafe {
                            (*fiber).mir_frames.last_mut().unwrap()
                                .values.insert(dst, InterpValue::Boxed(Value::null()));
                        }
                    }
                    continue 'fiber_loop;
                }
                Terminator::Unreachable => {
                    return Err(RuntimeError::Unreachable);
                }
                Terminator::Branch { target, args } => {
                    bind_block_params(&mut values, &mir.blocks[target.0 as usize], args)?;
                    current_block = *target;
                }
                Terminator::CondBranch { condition, true_target, true_args, false_target, false_args } => {
                    let cond = get_val(&values, *condition)?;
                    if cond.is_truthy() {
                        bind_block_params(&mut values, &mir.blocks[true_target.0 as usize], true_args)?;
                        current_block = *true_target;
                    } else {
                        bind_block_params(&mut values, &mir.blocks[false_target.0 as usize], false_args)?;
                        current_block = *false_target;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Backward-compatible wrapper
// ---------------------------------------------------------------------------

/// Execute a MIR function using a temporary fiber.
pub fn eval_in_vm(
    vm: &mut VM,
    func: &MirFunction,
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
        },
    );

    let fiber = vm.gc.alloc_fiber();
    unsafe {
        (*fiber).header.class = vm.fiber_class;
        (*fiber).mir_frames.push(MirCallFrame {
            func_id,
            current_block: BlockId(0),
            ip: 0,
            values: HashMap::new(),
            module_name: module_name.clone(),
            return_dst: None,
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

fn get_val(values: &HashMap<ValueId, InterpValue>, id: ValueId) -> Result<InterpValue, RuntimeError> {
    values.get(&id).copied()
        .ok_or_else(|| RuntimeError::Error(format!("undefined value {:?}", id)))
}

fn bind_block_params(
    values: &mut HashMap<ValueId, InterpValue>,
    target_block: &crate::mir::BasicBlock,
    args: &[ValueId],
) -> Result<(), RuntimeError> {
    let arg_vals: Vec<InterpValue> = args.iter()
        .map(|a| get_val(values, *a))
        .collect::<Result<Vec<_>, _>>()?;

    let mut param_idx = 0;
    for (dst, inst) in &target_block.instructions {
        if matches!(inst, Instruction::BlockParam(_)) {
            if param_idx < arg_vals.len() {
                values.insert(*dst, arg_vals[param_idx]);
                param_idx += 1;
            }
        } else {
            break;
        }
    }
    Ok(())
}

pub fn value_to_string(vm: &VM, value: Value) -> String {
    if value.is_null() {
        "null".to_string()
    } else if let Some(b) = value.as_bool() {
        if b { "true".to_string() } else { "false".to_string() }
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
    use crate::runtime::vm::{VM, VMConfig};

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
        f.block_mut(bb).instructions.push((v0, Instruction::ConstNum(42.0)));
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
        f.block_mut(bb).instructions.push((v0, Instruction::ConstString(hello_sym.index())));
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
            b.instructions.push((v1, Instruction::Call {
                receiver: v0, method: abs_sym, args: vec![],
            }));
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
            b.instructions.push((v2, Instruction::Call {
                receiver: v0, method: add_sym, args: vec![v1],
            }));
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
            b.instructions.push((v2, Instruction::MakeList(vec![v0, v1])));
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
            b.terminator = Terminator::Branch { target: bb1, args: vec![v_zero, v_one] };
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
            b.instructions.push((v_cond, Instruction::CmpLe(v_i_p, v_five)));
            b.terminator = Terminator::CondBranch {
                condition: v_cond,
                true_target: bb2, true_args: vec![v_sum_p, v_i_p],
                false_target: bb3, false_args: vec![v_sum_p],
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
            b.instructions.push((v_sum2, Instruction::Add(v_sum_b, v_i_b)));
            b.instructions.push((v_one2, Instruction::ConstNum(1.0)));
            b.instructions.push((v_i2, Instruction::Add(v_i_b, v_one2)));
            b.terminator = Terminator::Branch { target: bb1, args: vec![v_sum2, v_i2] };
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
            b.instructions.push((v1, Instruction::Call {
                receiver: v0, method: bogus, args: vec![],
            }));
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
        f.block_mut(bb).instructions.push((v0, Instruction::ConstNum(77.0)));
        f.block_mut(bb).terminator = Terminator::Return(v0);

        let func_id = vm.engine.register_function(f);
        let fiber = vm.gc.alloc_fiber();
        unsafe {
            (*fiber).header.class = vm.fiber_class;
            (*fiber).mir_frames.push(MirCallFrame {
                func_id,
                current_block: BlockId(0),
                ip: 0,
                values: HashMap::new(),
                module_name: "__test__".to_string(),
                return_dst: None,
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

        let func_id = vm.engine.register_function(f);
        let fiber = vm.gc.alloc_fiber();
        unsafe {
            (*fiber).header.class = vm.fiber_class;
            assert_eq!((*fiber).state, FiberState::New);
            (*fiber).mir_frames.push(MirCallFrame {
                func_id,
                current_block: BlockId(0),
                ip: 0,
                values: HashMap::new(),
                module_name: "__test__".to_string(),
                return_dst: None,
            });
        }
        vm.fiber = fiber;

        let _ = run_fiber(&mut vm).unwrap();
        assert_eq!(unsafe { (*fiber).state }, FiberState::Done);
        assert_eq!(unsafe { (*fiber).mir_frames.len() }, 0);
    }
}
