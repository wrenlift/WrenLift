use crate::runtime::engine::FuncId;
use crate::runtime::object::{
    FiberState, MirCallFrame, NativeContext, ObjClosure, ObjFiber, ObjHeader, ObjType,
};
use crate::runtime::value::Value;
use crate::runtime::vm::VM;
use std::collections::HashMap;

// --- Helpers ---

/// Extract an ObjFiber pointer from a Value (args[0] for instance methods).
unsafe fn as_fiber(val: Value) -> Option<*mut ObjFiber> {
    let ptr = val.as_object()?;
    let header = ptr as *const ObjHeader;
    if (*header).obj_type == ObjType::Fiber {
        Some(ptr as *mut ObjFiber)
    } else {
        None
    }
}

/// Extract an ObjClosure pointer from a Value.
unsafe fn as_closure(val: Value) -> Option<*mut ObjClosure> {
    let ptr = val.as_object()?;
    let header = ptr as *const ObjHeader;
    if (*header).obj_type == ObjType::Closure {
        Some(ptr as *mut ObjClosure)
    } else {
        None
    }
}

/// Set up a fiber's initial MIR frame from a closure.
unsafe fn setup_fiber_from_closure(fiber: *mut ObjFiber, closure: *mut ObjClosure) {
    let fn_ptr = (*closure).function;
    let func_id = FuncId((*fn_ptr).fn_id);

    (*fiber).mir_frames.push(MirCallFrame {
        func_id,
        current_block: crate::mir::BlockId(0),
        ip: 0,
        values: HashMap::new(),
        module_name: String::from("main"),
        return_dst: None,
        closure: Some(closure),
        defining_class: None,
    });
    (*fiber).state = FiberState::New;
}

// --- Static methods ---

fn fiber_new(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    // args[0] = Fiber class (receiver), args[1] = closure
    let closure_ptr = unsafe { as_closure(args[1]) };
    match closure_ptr {
        Some(closure) => {
            // Capture spawn-site stack trace if opt-in is enabled
            let spawn_trace = ctx.capture_spawn_trace();
            let fiber = ctx.alloc_fiber();
            unsafe {
                (*fiber).header.class = ctx.get_fiber_class();
                setup_fiber_from_closure(fiber, closure);
                (*fiber).spawn_trace = spawn_trace;
            }
            Value::object(fiber as *mut u8)
        }
        None => {
            ctx.runtime_error("Fiber.new expects a function.".to_string());
            Value::null()
        }
    }
}

fn fiber_abort(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    if args[1].is_object() {
        let class_name = ctx.get_class_name_of(args[1]);
        if class_name == "String" {
            let msg = super::as_string(args[1]);
            ctx.runtime_error(msg.to_string());
        } else {
            ctx.runtime_error("Runtime error.".to_string());
        }
    } else {
        ctx.runtime_error("Runtime error.".to_string());
    }
    Value::null()
}

fn fiber_current(ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    let fiber = ctx.get_current_fiber();
    if fiber.is_null() {
        Value::null()
    } else {
        Value::object(fiber as *mut u8)
    }
}

fn fiber_suspend(ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    ctx.set_fiber_action_suspend();
    Value::null()
}

fn fiber_yield_0(ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    ctx.set_fiber_action_yield(Value::null());
    Value::null()
}

fn fiber_yield_1(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    ctx.set_fiber_action_yield(args[1]);
    Value::null()
}

// --- Instance methods ---

fn fiber_call_0(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let fiber = unsafe { as_fiber(args[0]) };
    match fiber {
        Some(f) => {
            let state = unsafe { (*f).state };
            match state {
                FiberState::New | FiberState::Suspended => {
                    ctx.set_fiber_action_call(f, Value::null());
                }
                FiberState::Done => {
                    ctx.runtime_error("Cannot call a finished fiber.".to_string());
                }
                _ => {
                    ctx.runtime_error("Fiber has already been called.".to_string());
                }
            }
        }
        None => {
            ctx.runtime_error("Expected a fiber.".to_string());
        }
    }
    Value::null()
}

fn fiber_call_1(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let fiber = unsafe { as_fiber(args[0]) };
    match fiber {
        Some(f) => {
            let state = unsafe { (*f).state };
            match state {
                FiberState::New | FiberState::Suspended => {
                    ctx.set_fiber_action_call(f, args[1]);
                }
                FiberState::Done => {
                    ctx.runtime_error("Cannot call a finished fiber.".to_string());
                }
                _ => {
                    ctx.runtime_error("Fiber has already been called.".to_string());
                }
            }
        }
        None => {
            ctx.runtime_error("Expected a fiber.".to_string());
        }
    }
    Value::null()
}

fn fiber_error(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let fiber = unsafe { as_fiber(args[0]) };
    match fiber {
        Some(f) => unsafe { (*f).error },
        None => {
            ctx.runtime_error("Expected a fiber.".to_string());
            Value::null()
        }
    }
}

fn fiber_is_done(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let fiber = unsafe { as_fiber(args[0]) };
    match fiber {
        Some(f) => {
            let done = unsafe { (*f).state == FiberState::Done };
            Value::bool(done)
        }
        None => Value::bool(true),
    }
}

fn fiber_transfer_0(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let fiber = unsafe { as_fiber(args[0]) };
    match fiber {
        Some(f) => {
            let state = unsafe { (*f).state };
            match state {
                FiberState::New | FiberState::Suspended => {
                    ctx.set_fiber_action_transfer(f, Value::null());
                }
                FiberState::Done => {
                    ctx.runtime_error("Cannot transfer to a finished fiber.".to_string());
                }
                _ => {
                    ctx.runtime_error("Fiber is already running.".to_string());
                }
            }
        }
        None => {
            ctx.runtime_error("Expected a fiber.".to_string());
        }
    }
    Value::null()
}

fn fiber_transfer_1(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let fiber = unsafe { as_fiber(args[0]) };
    match fiber {
        Some(f) => {
            let state = unsafe { (*f).state };
            match state {
                FiberState::New | FiberState::Suspended => {
                    ctx.set_fiber_action_transfer(f, args[1]);
                }
                FiberState::Done => {
                    ctx.runtime_error("Cannot transfer to a finished fiber.".to_string());
                }
                _ => {
                    ctx.runtime_error("Fiber is already running.".to_string());
                }
            }
        }
        None => {
            ctx.runtime_error("Expected a fiber.".to_string());
        }
    }
    Value::null()
}

fn fiber_transfer_error(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let fiber = unsafe { as_fiber(args[0]) };
    match fiber {
        Some(f) => {
            let state = unsafe { (*f).state };
            match state {
                FiberState::New | FiberState::Suspended => {
                    unsafe {
                        (*f).error = args[1];
                    }
                    ctx.set_fiber_action_transfer(f, args[1]);
                }
                _ => {
                    ctx.runtime_error("Cannot transfer to fiber.".to_string());
                }
            }
        }
        None => {
            ctx.runtime_error("Expected a fiber.".to_string());
        }
    }
    Value::null()
}

fn fiber_stack_trace(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    if !ctx.fiber_stack_traces_enabled() {
        return ctx.alloc_string(
            "<stack traces not enabled — set VMConfig.fiber_stack_traces = true>".to_string(),
        );
    }
    let fiber = unsafe { as_fiber(args[0]) };
    match fiber {
        Some(f) => {
            let trace = ctx.get_stack_trace_string(f);
            ctx.alloc_string(trace)
        }
        None => {
            ctx.runtime_error("Expected a fiber.".to_string());
            Value::null()
        }
    }
}

fn fiber_try_0(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    // try() is like call() but catches runtime errors
    fiber_call_0(ctx, args)
}

fn fiber_try_1(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    fiber_call_1(ctx, args)
}

pub fn bind(vm: &mut VM) {
    let class = vm.fiber_class;

    // Static methods
    vm.primitive_static(class, "new(_)", fiber_new);
    vm.primitive_static(class, "abort(_)", fiber_abort);
    vm.primitive_static(class, "current", fiber_current);
    vm.primitive_static(class, "suspend()", fiber_suspend);
    vm.primitive_static(class, "yield()", fiber_yield_0);
    vm.primitive_static(class, "yield(_)", fiber_yield_1);

    // Instance methods
    vm.primitive(class, "call()", fiber_call_0);
    vm.primitive(class, "call(_)", fiber_call_1);
    vm.primitive(class, "error", fiber_error);
    vm.primitive(class, "isDone", fiber_is_done);
    vm.primitive(class, "transfer()", fiber_transfer_0);
    vm.primitive(class, "transfer(_)", fiber_transfer_1);
    vm.primitive(class, "transferError(_)", fiber_transfer_error);
    vm.primitive(class, "try()", fiber_try_0);
    vm.primitive(class, "try(_)", fiber_try_1);
    vm.primitive(class, "stackTrace", fiber_stack_trace);
}
