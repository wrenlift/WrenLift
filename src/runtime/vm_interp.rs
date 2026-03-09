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
use crate::runtime::vm::{FiberAction, VM};

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

/// Source location context attached when reporting runtime errors.
#[derive(Debug, Clone)]
pub struct SourceLoc {
    pub span: crate::ast::Span,
    pub module: String,
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

const MAX_STEPS: usize = 100_000_000;

// ---------------------------------------------------------------------------
// Fiber-based MIR interpreter
// ---------------------------------------------------------------------------

/// Run the active fiber (vm.fiber) until it finishes, yields, or errors.
///
/// The fiber must have at least one MirCallFrame pushed before calling this.
/// On error, saves the current ip/block back to the fiber frame so
/// `extract_error_location` can look up the source span.
pub fn run_fiber(vm: &mut VM) -> Result<Value, RuntimeError> {
    if vm.fiber.is_null() {
        return Err(RuntimeError::Error("no active fiber".into()));
    }

    let mut steps: usize = 0;

    // Outer loop: re-entered when we push/pop a call frame or switch fibers
    'fiber_loop: loop {
        let fiber = vm.fiber;
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
        let mir = vm
            .engine
            .get_mir(func_id)
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

                // Save execution position to the frame so extract_error_location
                // can find the source span if a `?` propagates an error.
                unsafe {
                    if let Some(frame) = (*fiber).mir_frames.last_mut() {
                        frame.ip = ip;
                        frame.current_block = current_block;
                    }
                }

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
                        let val = vm
                            .engine
                            .modules
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
                    Call {
                        receiver,
                        method,
                        args,
                    } => {
                        let recv_val = get_val(&values, *receiver)?.to_value();
                        let mut arg_vals = vec![recv_val];
                        for arg in args {
                            arg_vals.push(get_val(&values, *arg)?.to_value());
                        }

                        // If receiver is a closure and method is call(...), dispatch directly
                        let method_name_str = vm.interner.resolve(*method).to_string();
                        if method_name_str.starts_with("call") && recv_val.is_object() {
                            let ptr = recv_val.as_object().unwrap();
                            let header = ptr as *const ObjHeader;
                            if unsafe { (*header).obj_type } == ObjType::Closure {
                                let closure_ptr = ptr as *mut ObjClosure;
                                // For Fn.call(), pass only the call args (skip receiver)
                                dispatch_closure(
                                    vm,
                                    fiber,
                                    closure_ptr,
                                    &arg_vals[1..],
                                    &mut current_block,
                                    ip,
                                    values,
                                    &module_name,
                                    *dst,
                                )?;
                                continue 'fiber_loop;
                            }
                        }

                        let class = vm.class_of(recv_val);
                        let mut method_entry = unsafe { (*class).find_method(*method).cloned() };
                        // Track which class the method was found on (for super dispatch)
                        let mut defining_class: Option<*mut ObjClass> =
                            unsafe { find_method_class(class, *method) };

                        // If not found and receiver IS a class, try static method
                        // on the class itself. Core library stores statics with
                        // "static:" prefix (e.g. "static:print(_)").
                        if method_entry.is_none() && class == vm.class_class {
                            let static_name = format!("static:{}", method_name_str);
                            let static_sym = vm.interner.intern(&static_name);
                            let recv_class_ptr = recv_val.as_object().unwrap() as *mut ObjClass;
                            method_entry =
                                unsafe { (*recv_class_ptr).find_method(static_sym).cloned() };
                            // For static/constructor methods, the defining class is the receiver class
                            defining_class = Some(recv_class_ptr);
                        }

                        match method_entry {
                            Some(Method::Native(func)) => {
                                let result = func(vm, &arg_vals);
                                if vm.has_error {
                                    vm.has_error = false;
                                    // If the fiber was invoked via try(), catch the error
                                    // and return it to the caller instead of propagating.
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
                                            unsafe { (*fiber).caller = std::ptr::null_mut() };
                                            resume_caller(vm, caller, err_val);
                                            continue 'fiber_loop;
                                        }
                                        return Ok(err_val);
                                    }
                                    return Err(RuntimeError::Error(
                                        "runtime error in native method".into(),
                                    ));
                                }
                                // Check for pending fiber action (yield/call/transfer)
                                if let Some(action) = vm.pending_fiber_action.take() {
                                    handle_fiber_action(
                                        vm,
                                        fiber,
                                        action,
                                        current_block,
                                        ip,
                                        values,
                                        *dst,
                                    )?;
                                    continue 'fiber_loop;
                                }
                                InterpValue::Boxed(result)
                            }
                            Some(Method::Closure(closure_ptr)) => {
                                dispatch_closure_with_class(
                                    vm,
                                    fiber,
                                    closure_ptr,
                                    &arg_vals,
                                    &mut current_block,
                                    ip,
                                    values,
                                    &module_name,
                                    *dst,
                                    defining_class,
                                )?;
                                continue 'fiber_loop;
                            }
                            Some(Method::Constructor(closure_ptr)) => {
                                // Allocate a new instance of the receiver class
                                let recv_class = recv_val.as_object().unwrap() as *mut ObjClass;
                                let instance = vm.gc.alloc_instance(recv_class);
                                let instance_val = Value::object(instance as *mut u8);

                                // Replace receiver (class) with the new instance
                                let mut ctor_args = arg_vals.clone();
                                ctor_args[0] = instance_val;

                                dispatch_closure_with_class(
                                    vm,
                                    fiber,
                                    closure_ptr,
                                    &ctor_args,
                                    &mut current_block,
                                    ip,
                                    values,
                                    &module_name,
                                    *dst,
                                    defining_class,
                                )?;
                                continue 'fiber_loop;
                            }
                            None => {
                                let method_name = vm.interner.resolve(*method).to_string();
                                let class_name = vm.class_name_of(recv_val);
                                return Err(RuntimeError::MethodNotFound {
                                    class_name,
                                    method: method_name,
                                });
                            }
                        }
                    }
                    SuperCall { method, args } => {
                        // args[0] is `this`, rest are actual arguments
                        let mut arg_vals: Vec<Value> = Vec::with_capacity(args.len());
                        for arg in args {
                            arg_vals.push(get_val(&values, *arg)?.to_value());
                        }

                        // Get the defining class from the current frame, then look at its superclass
                        let defining =
                            unsafe { (*fiber).mir_frames.last().unwrap().defining_class };
                        let superclass = defining.and_then(|cls| {
                            let sup = unsafe { (*cls).superclass };
                            if sup.is_null() {
                                None
                            } else {
                                Some(sup)
                            }
                        });

                        if let Some(super_cls) = superclass {
                            let mut method_entry =
                                unsafe { (*super_cls).find_method(*method).cloned() };
                            // If not found, try with "static:" prefix (for super constructor calls)
                            if method_entry.is_none() {
                                let method_name = vm.interner.resolve(*method).to_string();
                                let static_name = format!("static:{}", method_name);
                                let static_sym = vm.interner.intern(&static_name);
                                method_entry =
                                    unsafe { (*super_cls).find_method(static_sym).cloned() };
                            }
                            match method_entry {
                                Some(Method::Native(func)) => {
                                    InterpValue::Boxed(func(vm, &arg_vals))
                                }
                                Some(Method::Closure(closure_ptr)) => {
                                    let found_class =
                                        unsafe { find_method_class(super_cls, *method) };
                                    dispatch_closure_with_class(
                                        vm,
                                        fiber,
                                        closure_ptr,
                                        &arg_vals,
                                        &mut current_block,
                                        ip,
                                        values,
                                        &module_name,
                                        *dst,
                                        found_class,
                                    )?;
                                    continue 'fiber_loop;
                                }
                                Some(Method::Constructor(closure_ptr)) => {
                                    let found_class =
                                        unsafe { find_method_class(super_cls, *method) };
                                    dispatch_closure_with_class(
                                        vm,
                                        fiber,
                                        closure_ptr,
                                        &arg_vals,
                                        &mut current_block,
                                        ip,
                                        values,
                                        &module_name,
                                        *dst,
                                        found_class,
                                    )?;
                                    continue 'fiber_loop;
                                }
                                None => {
                                    let method_name = vm.interner.resolve(*method).to_string();
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

                    // -- Subscript --
                    SubscriptGet { receiver, args } => {
                        let recv = get_val(&values, *receiver)?.to_value();
                        let mut all_args = vec![recv];
                        for a in args {
                            all_args.push(get_val(&values, *a)?.to_value());
                        }
                        let class = vm.class_of(recv);
                        // Build signature: [_] or [_,_] etc based on arg count
                        let sig = format!("[{}]", vec!["_"; args.len()].join(","));
                        let sym = vm.interner.intern(&sig);
                        match unsafe { (*class).find_method(sym).cloned() } {
                            Some(Method::Native(func)) => InterpValue::Boxed(func(vm, &all_args)),
                            Some(Method::Closure(closure_ptr)) => {
                                dispatch_closure(
                                    vm,
                                    fiber,
                                    closure_ptr,
                                    &all_args,
                                    &mut current_block,
                                    ip,
                                    values,
                                    &module_name,
                                    *dst,
                                )?;
                                continue 'fiber_loop;
                            }
                            _ => {
                                return Err(RuntimeError::Unsupported(format!(
                                    "subscript get with sig '{}'",
                                    sig
                                )))
                            }
                        }
                    }
                    SubscriptSet {
                        receiver,
                        args,
                        value,
                    } => {
                        let recv = get_val(&values, *receiver)?.to_value();
                        let mut all_args = vec![recv];
                        for a in args {
                            all_args.push(get_val(&values, *a)?.to_value());
                        }
                        all_args.push(get_val(&values, *value)?.to_value());
                        let class = vm.class_of(recv);
                        // Build signature: [_]=(_) or [_,_]=(_) etc
                        let sig = format!("[{}]=(_)", vec!["_"; args.len()].join(","));
                        let sym = vm.interner.intern(&sig);
                        match unsafe { (*class).find_method(sym).cloned() } {
                            Some(Method::Native(func)) => InterpValue::Boxed(func(vm, &all_args)),
                            Some(Method::Closure(closure_ptr)) => {
                                dispatch_closure(
                                    vm,
                                    fiber,
                                    closure_ptr,
                                    &all_args,
                                    &mut current_block,
                                    ip,
                                    values,
                                    &module_name,
                                    *dst,
                                )?;
                                continue 'fiber_loop;
                            }
                            _ => {
                                return Err(RuntimeError::Unsupported(format!(
                                    "subscript set with sig '{}'",
                                    sig
                                )))
                            }
                        }
                    }

                    // -- Collections --
                    MakeList(elems) => {
                        let elements: Vec<Value> = elems
                            .iter()
                            .map(|e| get_val(&values, *e).map(|v| v.to_value()))
                            .collect::<Result<_, _>>()?;
                        InterpValue::Boxed(vm.new_list(elements))
                    }
                    MakeMap(pairs) => {
                        let map_val = vm.new_map();
                        if !pairs.is_empty() {
                            let map_ptr =
                                map_val.as_object().unwrap() as *mut crate::runtime::object::ObjMap;
                            for (k, v) in pairs {
                                let key = get_val(&values, *k)?.to_value();
                                let val = get_val(&values, *v)?.to_value();
                                unsafe {
                                    (*map_ptr).set(key, val);
                                }
                            }
                        }
                        InterpValue::Boxed(map_val)
                    }
                    MakeRange(from, to, inclusive) => {
                        let f = get_val(&values, *from)?
                            .as_f64()
                            .map_err(RuntimeError::from)?;
                        let t = get_val(&values, *to)?
                            .as_f64()
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
                    IsType(val_id, class_sym) => {
                        let value = get_val(&values, *val_id)?.to_value();
                        let target_name = vm.interner.resolve(*class_sym).to_string();
                        let mut cls = vm.class_of(value);
                        let mut matched = false;
                        while !cls.is_null() {
                            let name = unsafe { vm.interner.resolve((*cls).name).to_string() };
                            if name == target_name {
                                matched = true;
                                break;
                            }
                            cls = unsafe { (*cls).superclass };
                        }
                        InterpValue::Boxed(Value::bool(matched))
                    }

                    // -- Guards that need VM class system --
                    GuardClass(a, _cls) => get_val(&values, *a)?,
                    GuardProtocol(a, _proto) => get_val(&values, *a)?,

                    // -- Field access --
                    GetField(recv_id, field_idx) => {
                        let recv = get_val(&values, *recv_id)?.to_value();
                        if let Some(ptr) = recv.as_object() {
                            let inst = ptr as *const ObjInstance;
                            let val = unsafe { (*inst).get_field(*field_idx as usize) }
                                .unwrap_or(Value::null());
                            InterpValue::Boxed(val)
                        } else {
                            InterpValue::Boxed(Value::null())
                        }
                    }
                    SetField(recv_id, field_idx, val_id) => {
                        let recv = get_val(&values, *recv_id)?.to_value();
                        let val = get_val(&values, *val_id)?.to_value();
                        if let Some(ptr) = recv.as_object() {
                            let inst = ptr as *mut ObjInstance;
                            unsafe {
                                (*inst).set_field(*field_idx as usize, val);
                            }
                        }
                        InterpValue::Boxed(val)
                    }
                    GetStaticField(sym) => {
                        // Read from the defining class's static_fields map
                        let frame = unsafe { (*fiber).mir_frames.last().unwrap() };
                        if let Some(class_ptr) = frame.defining_class {
                            let val = unsafe {
                                (*class_ptr)
                                    .static_fields
                                    .get(sym)
                                    .copied()
                                    .unwrap_or(Value::null())
                            };
                            InterpValue::Boxed(val)
                        } else {
                            InterpValue::Boxed(Value::null())
                        }
                    }
                    SetStaticField(sym, val_id) => {
                        let val = get_val(&values, *val_id)?.to_value();
                        let frame = unsafe { (*fiber).mir_frames.last().unwrap() };
                        if let Some(class_ptr) = frame.defining_class {
                            unsafe {
                                (*class_ptr).static_fields.insert(*sym, val);
                            }
                        }
                        InterpValue::Boxed(val)
                    }
                    GetUpvalue(idx) => {
                        // Read from the current frame's closure upvalue array
                        let frame_closure = unsafe { (*fiber).mir_frames.last().unwrap().closure };
                        if let Some(closure_ptr) = frame_closure {
                            let upvalues = unsafe { &(*closure_ptr).upvalues };
                            if (*idx as usize) < upvalues.len() {
                                let uv = upvalues[*idx as usize];
                                InterpValue::Boxed(unsafe { (*uv).get() })
                            } else {
                                InterpValue::Boxed(Value::null())
                            }
                        } else {
                            InterpValue::Boxed(Value::null())
                        }
                    }
                    SetUpvalue(idx, val_id) => {
                        let v = get_val(&values, *val_id)?.to_value();
                        let frame_closure = unsafe { (*fiber).mir_frames.last().unwrap().closure };
                        if let Some(closure_ptr) = frame_closure {
                            let upvalues = unsafe { &(*closure_ptr).upvalues };
                            if (*idx as usize) < upvalues.len() {
                                let uv = upvalues[*idx as usize];
                                unsafe {
                                    (*uv).set(v);
                                }
                            }
                        }
                        InterpValue::Boxed(v)
                    }
                    MakeClosure {
                        fn_id,
                        upvalues: uv_ids,
                    } => {
                        let closure_name = vm.interner.intern("<closure>");
                        let uv_count = uv_ids.len() as u16;
                        let arity = vm
                            .engine
                            .get_mir(crate::runtime::engine::FuncId(*fn_id))
                            .map(|mir| mir.arity)
                            .unwrap_or(0);
                        let fn_ptr = vm.gc.alloc_fn(closure_name, arity, uv_count, *fn_id);
                        unsafe {
                            (*fn_ptr).header.class = vm.fn_class;
                        }
                        let closure_ptr = vm.gc.alloc_closure(fn_ptr);
                        unsafe {
                            (*closure_ptr).header.class = vm.fn_class;
                        }

                        // Populate upvalues with captured values (pre-closed)
                        for (i, uv_id) in uv_ids.iter().enumerate() {
                            let captured_val = get_val(&values, *uv_id)?.to_value();
                            // Allocate upvalue with null location, then store value in closed field
                            let uv_obj = vm.gc.alloc_upvalue(std::ptr::null_mut());
                            unsafe {
                                (*uv_obj).closed = captured_val;
                                (*uv_obj).location = &mut (*uv_obj).closed as *mut Value;
                                if i < (*closure_ptr).upvalues.len() {
                                    (&mut (*closure_ptr).upvalues)[i] = uv_obj;
                                }
                            }
                        }

                        InterpValue::Boxed(Value::object(closure_ptr as *mut u8))
                    }

                    // -- Everything else: delegate to shared MIR evaluator --
                    // If pure eval fails with a type mismatch, try operator dispatch.
                    _ => match eval_pure_instruction(inst, &values) {
                        Ok(v) => v,
                        Err(_) => {
                            if let Some((recv_id, method_str, arg_ids)) =
                                operator_dispatch_info(inst)
                            {
                                let recv_val = get_val(&values, recv_id)?.to_value();
                                let method_sym = vm.interner.intern(method_str);
                                let class = vm.class_of(recv_val);
                                let method_entry =
                                    unsafe { (*class).find_method(method_sym).cloned() };
                                match method_entry {
                                    Some(Method::Native(func)) => {
                                        let mut arg_vals = vec![recv_val];
                                        for &a in &arg_ids {
                                            arg_vals.push(get_val(&values, a)?.to_value());
                                        }
                                        InterpValue::Boxed(func(vm, &arg_vals))
                                    }
                                    Some(Method::Closure(closure_ptr)) => {
                                        let mut arg_vals = vec![recv_val];
                                        for &a in &arg_ids {
                                            arg_vals.push(get_val(&values, a)?.to_value());
                                        }
                                        dispatch_closure(
                                            vm,
                                            fiber,
                                            closure_ptr,
                                            &arg_vals,
                                            &mut current_block,
                                            ip,
                                            values,
                                            &module_name,
                                            *dst,
                                        )?;
                                        continue 'fiber_loop;
                                    }
                                    _ => {
                                        return Err(RuntimeError::Error(format!(
                                            "{} does not implement '{}'",
                                            vm.class_name_of(recv_val),
                                            method_str
                                        )))
                                    }
                                }
                            } else {
                                eval_pure_instruction(inst, &values)?
                            }
                        }
                    },
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
                    unsafe {
                        (*fiber).mir_frames.pop();
                    }
                    if unsafe { (*fiber).mir_frames.is_empty() } {
                        unsafe {
                            (*fiber).state = FiberState::Done;
                        }
                        // If this fiber has a caller, resume it with the return value
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
                    if let Some(dst) = return_dst {
                        unsafe {
                            (*fiber)
                                .mir_frames
                                .last_mut()
                                .unwrap()
                                .values
                                .insert(dst, InterpValue::Boxed(return_val));
                        }
                    }
                    continue 'fiber_loop;
                }
                Terminator::ReturnNull => {
                    let return_dst = unsafe { (*fiber).mir_frames.last().unwrap().return_dst };
                    unsafe {
                        (*fiber).mir_frames.pop();
                    }
                    if unsafe { (*fiber).mir_frames.is_empty() } {
                        unsafe {
                            (*fiber).state = FiberState::Done;
                        }
                        let caller = unsafe { (*fiber).caller };
                        if !caller.is_null() {
                            unsafe {
                                (*fiber).caller = std::ptr::null_mut();
                            }
                            resume_caller(vm, caller, Value::null());
                            continue 'fiber_loop;
                        }
                        return Ok(Value::null());
                    }
                    if let Some(dst) = return_dst {
                        unsafe {
                            (*fiber)
                                .mir_frames
                                .last_mut()
                                .unwrap()
                                .values
                                .insert(dst, InterpValue::Boxed(Value::null()));
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
                Terminator::CondBranch {
                    condition,
                    true_target,
                    true_args,
                    false_target,
                    false_args,
                } => {
                    let cond = get_val(&values, *condition)?;
                    if cond.is_truthy() {
                        bind_block_params(
                            &mut values,
                            &mir.blocks[true_target.0 as usize],
                            true_args,
                        )?;
                        current_block = *true_target;
                    } else {
                        bind_block_params(
                            &mut values,
                            &mir.blocks[false_target.0 as usize],
                            false_args,
                        )?;
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

fn get_val(
    values: &HashMap<ValueId, InterpValue>,
    id: ValueId,
) -> Result<InterpValue, RuntimeError> {
    values
        .get(&id)
        .copied()
        .ok_or_else(|| RuntimeError::Error(format!("undefined value {:?}", id)))
}

fn bind_block_params(
    values: &mut HashMap<ValueId, InterpValue>,
    target_block: &crate::mir::BasicBlock,
    args: &[ValueId],
) -> Result<(), RuntimeError> {
    let arg_vals: Vec<InterpValue> = args
        .iter()
        .map(|a| get_val(values, *a))
        .collect::<Result<Vec<_>, _>>()?;

    // First try .params (used by for-loops, ternary merges, etc.)
    if !target_block.params.is_empty() {
        for (i, (dst, _ty)) in target_block.params.iter().enumerate() {
            if i < arg_vals.len() {
                values.insert(*dst, arg_vals[i]);
            }
        }
        return Ok(());
    }

    // Fall back to BlockParam instructions (used by entry blocks)
    for (dst, inst) in &target_block.instructions {
        if let Instruction::BlockParam(param_idx) = inst {
            let idx = *param_idx as usize;
            if idx < arg_vals.len() {
                values.insert(*dst, arg_vals[idx]);
            }
        }
    }
    Ok(())
}

/// Dispatch a closure call: save current frame, push new frame with args bound to block params.
#[allow(clippy::too_many_arguments)]
fn dispatch_closure(
    vm: &mut VM,
    fiber: *mut ObjFiber,
    closure_ptr: *mut ObjClosure,
    arg_vals: &[Value],
    current_block: &mut BlockId,
    ip: usize,
    values: HashMap<ValueId, InterpValue>,
    module_name: &str,
    return_dst: ValueId,
) -> Result<(), RuntimeError> {
    dispatch_closure_with_class(
        vm,
        fiber,
        closure_ptr,
        arg_vals,
        current_block,
        ip,
        values,
        module_name,
        return_dst,
        None,
    )
}

/// Dispatch a closure call with an optional defining class (for super dispatch).
#[allow(clippy::too_many_arguments)]
fn dispatch_closure_with_class(
    vm: &mut VM,
    fiber: *mut ObjFiber,
    closure_ptr: *mut ObjClosure,
    arg_vals: &[Value],
    current_block: &mut BlockId,
    ip: usize,
    values: HashMap<ValueId, InterpValue>,
    module_name: &str,
    return_dst: ValueId,
    defining_class: Option<*mut ObjClass>,
) -> Result<(), RuntimeError> {
    let fn_ptr = unsafe { (*closure_ptr).function };
    let target_func_id = crate::runtime::engine::FuncId(unsafe { (*fn_ptr).fn_id });

    // Tier-up: record call and compile if threshold reached
    let should_tier_up = vm.engine.record_call(target_func_id);
    if should_tier_up {
        let interner = &vm.interner as *const crate::intern::Interner;
        // Safety: interner is not modified during tier_up; separate field from engine
        vm.engine.tier_up(target_func_id, unsafe { &*interner });
    }

    let target_mir = vm
        .engine
        .get_mir(target_func_id)
        .ok_or_else(|| RuntimeError::Error("invalid closure func_id".into()))?;

    // Bind args to entry block params using BlockParam index
    let entry_block = &target_mir.blocks[0];
    let mut new_values: HashMap<ValueId, InterpValue> = HashMap::new();
    for (dst, inst) in &entry_block.instructions {
        if let Instruction::BlockParam(param_idx) = inst {
            let idx = *param_idx as usize;
            if idx < arg_vals.len() {
                new_values.insert(*dst, InterpValue::Boxed(arg_vals[idx]));
            }
        }
    }

    // Save current frame state
    unsafe {
        let frame = (*fiber).mir_frames.last_mut().unwrap();
        frame.current_block = *current_block;
        frame.ip = ip;
        frame.values = values;
    }

    // Push new frame
    unsafe {
        (*fiber).mir_frames.push(MirCallFrame {
            func_id: target_func_id,
            current_block: BlockId(0),
            ip: 0,
            values: new_values,
            module_name: module_name.to_string(),
            return_dst: Some(return_dst),
            closure: Some(closure_ptr),
            defining_class,
        });
    }

    Ok(())
}

/// Resume a caller fiber with a value (used when a called fiber finishes or yields).
fn resume_caller(vm: &mut VM, caller: *mut ObjFiber, value: Value) {
    // Store the value in the caller's resume slot
    unsafe {
        if let Some(dst) = (*caller).resume_value_dst.take() {
            if let Some(frame) = (*caller).mir_frames.last_mut() {
                frame.values.insert(dst, InterpValue::Boxed(value));
            }
        }
    }
    vm.fiber = caller;
}

/// Handle a fiber action (call/yield/transfer) triggered by a native method.
/// Saves the current fiber's state and switches to the target fiber.
fn handle_fiber_action(
    vm: &mut VM,
    fiber: *mut ObjFiber,
    action: FiberAction,
    current_block: BlockId,
    ip: usize,
    values: HashMap<ValueId, InterpValue>,
    call_dst: ValueId,
) -> Result<(), RuntimeError> {
    // Save current fiber's execution state
    unsafe {
        if let Some(frame) = (*fiber).mir_frames.last_mut() {
            frame.current_block = current_block;
            frame.ip = ip;
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

            // If target is suspended, pass the value into its resume slot
            let target_state = unsafe { (*target).state };
            if target_state == FiberState::Suspended {
                unsafe {
                    if let Some(dst) = (*target).resume_value_dst.take() {
                        if let Some(frame) = (*target).mir_frames.last_mut() {
                            frame.values.insert(dst, InterpValue::Boxed(value));
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
                // Transfer doesn't set caller chain
            }

            let target_state = unsafe { (*target).state };
            if target_state == FiberState::Suspended {
                unsafe {
                    if let Some(dst) = (*target).resume_value_dst.take() {
                        if let Some(frame) = (*target).mir_frames.last_mut() {
                            frame.values.insert(dst, InterpValue::Boxed(value));
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
            // If there's a caller, return to it (like yield with null).
            // If no caller, execution stops (run_fiber will see no active fiber).
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

/// Map a MIR arithmetic/comparison instruction to Wren operator method info.
/// Returns (receiver, method_signature, arg_value_ids) if applicable.
fn operator_dispatch_info(inst: &Instruction) -> Option<(ValueId, &'static str, Vec<ValueId>)> {
    match inst {
        Instruction::Add(a, b) => Some((*a, "+(_)", vec![*b])),
        Instruction::Sub(a, b) => Some((*a, "-(_)", vec![*b])),
        Instruction::Mul(a, b) => Some((*a, "*(_)", vec![*b])),
        Instruction::Div(a, b) => Some((*a, "/(_)", vec![*b])),
        Instruction::Mod(a, b) => Some((*a, "%(_)", vec![*b])),
        Instruction::CmpLt(a, b) => Some((*a, "<(_)", vec![*b])),
        Instruction::CmpGt(a, b) => Some((*a, ">(_)", vec![*b])),
        Instruction::CmpLe(a, b) => Some((*a, "<=(_)", vec![*b])),
        Instruction::CmpGe(a, b) => Some((*a, ">=(_)", vec![*b])),
        Instruction::CmpEq(a, b) => Some((*a, "==(_)", vec![*b])),
        Instruction::CmpNe(a, b) => Some((*a, "!=(_)", vec![*b])),
        Instruction::Neg(a) => Some((*a, "-", vec![])),
        Instruction::Not(a) => Some((*a, "!", vec![])),
        Instruction::BitNot(a) => Some((*a, "~", vec![])),
        Instruction::BitAnd(a, b) => Some((*a, "&(_)", vec![*b])),
        Instruction::BitOr(a, b) => Some((*a, "|(_)", vec![*b])),
        Instruction::Shl(a, b) => Some((*a, "<<(_)", vec![*b])),
        Instruction::Shr(a, b) => Some((*a, ">>(_)", vec![*b])),
        _ => None,
    }
}

/// Walk the class hierarchy starting from `cls` to find which class owns `method`.
/// Returns `Some(cls)` for the class that has the method in its own table.
unsafe fn find_method_class(
    cls: *mut ObjClass,
    method: crate::intern::SymbolId,
) -> Option<*mut ObjClass> {
    let mut c = cls;
    while !c.is_null() {
        if (*c).methods.contains_key(&method) {
            return Some(c);
        }
        c = (*c).superclass;
    }
    None
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
