use crate::runtime::engine::FuncId;
use crate::runtime::object::{
    FiberState, MirCallFrame, NativeContext, ObjClosure, ObjFiber, ObjHeader, ObjMap, ObjType,
};
use crate::runtime::value::Value;
use crate::runtime::vm::VM;

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

/// Set up a fiber's initial MIR frame from a closure. `module_name`
/// inherits from the caller so module-scope identifiers (classes,
/// top-level vars) remain reachable inside the fiber's closure — a
/// hardcoded "main" used to silently return null for references
/// defined in any other module.
unsafe fn setup_fiber_from_closure(
    fiber: *mut ObjFiber,
    closure: *mut ObjClosure,
    module_name: std::rc::Rc<String>,
) {
    let fn_ptr = (*closure).function;
    let func_id = FuncId((*fn_ptr).fn_id);

    // The register file starts empty; it will be sized correctly when the
    // fiber's first frame is loaded in run_fiber.
    (*fiber).mir_frames.push(MirCallFrame {
        func_id,
        current_block: crate::mir::BlockId(0),
        ip: 0,
        pc: 0,
        values: Vec::new(),
        module_name,
        return_dst: None,
        closure: Some(closure),
        defining_class: None,
        bc_ptr: std::ptr::null(),
    });
    (*fiber).state = FiberState::New;
}

/// Pull the module name from the caller's topmost frame so a
/// newly-spawned fiber resolves `Greet` (etc.) against the module
/// where it was defined, not a hardcoded "main".
unsafe fn current_module_name(ctx: &mut dyn NativeContext) -> std::rc::Rc<String> {
    let caller = ctx.get_current_fiber();
    if caller.is_null() {
        return std::rc::Rc::new(String::from("main"));
    }
    (*caller)
        .mir_frames
        .last()
        .map(|f| f.module_name.clone())
        .unwrap_or_else(|| std::rc::Rc::new(String::from("main")))
}

// --- Static methods ---

fn fiber_new(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    // args[0] = Fiber class (receiver), args[1] = closure
    let closure_ptr = unsafe { as_closure(args[1]) };
    match closure_ptr {
        Some(closure) => {
            // Capture spawn-site stack trace if opt-in is enabled
            let spawn_trace = ctx.capture_spawn_trace();
            // Prefer the closure's defining module: a closure passed
            // across an import boundary still has to resolve
            // `GetModuleVar` against the slots it was compiled for.
            // Fall back to the caller's module for test closures
            // that aren't registered with a module.
            let fn_ptr = unsafe { (*closure).function };
            let func_id = unsafe { (*fn_ptr).fn_id };
            let module_name = ctx
                .func_module(func_id)
                .unwrap_or_else(|| unsafe { current_module_name(ctx) });

            // Snapshot parent's context bag + deadline before we alloc
            // (alloc can move objects on nursery GC, invalidating caller).
            // Cancellation does NOT inherit — each fiber decides its own
            // timeline, and a cancelled parent cascades only if the user
            // wires it up explicitly.
            let parent = ctx.get_current_fiber();
            let (parent_ctx_entries, parent_deadline) = unsafe {
                if parent.is_null() {
                    (Vec::new(), None)
                } else {
                    let entries = snapshot_context_entries(parent);
                    ((*parent).deadline_ms).map_or((entries.clone(), None), |d| (entries, Some(d)))
                }
            };

            let fiber = ctx.alloc_fiber();
            unsafe {
                (*fiber).header.class = ctx.get_fiber_class();
                setup_fiber_from_closure(fiber, closure, module_name);
                (*fiber).spawn_trace = spawn_trace;
                (*fiber).deadline_ms = parent_deadline;
            }

            // Populate the child's context map (allocates) only if parent
            // had entries. Keeps the zero-context case allocation-free.
            if !parent_ctx_entries.is_empty() {
                let child_map = ctx.alloc_map();
                unsafe {
                    if let Some(map_ptr) = child_map.as_object() {
                        let map = map_ptr as *mut ObjMap;
                        for (k, v) in parent_ctx_entries {
                            (*map).set(k, v);
                        }
                    }
                    (*fiber).context_map = child_map;
                }
            }

            Value::object(fiber as *mut u8)
        }
        None => {
            ctx.runtime_error("Fiber.new expects a function.".to_string());
            Value::null()
        }
    }
}

/// Snapshot a fiber's context entries into an owned Vec so the caller
/// can pass them to a later `alloc_map` without worrying about GC
/// moving the source map mid-copy.
unsafe fn snapshot_context_entries(fiber: *mut ObjFiber) -> Vec<(Value, Value)> {
    if fiber.is_null() || (*fiber).context_map.is_null() {
        return Vec::new();
    }
    let ptr = match (*fiber).context_map.as_object() {
        Some(p) => p,
        None => return Vec::new(),
    };
    let header = ptr as *const ObjHeader;
    if (*header).obj_type != ObjType::Map {
        return Vec::new();
    }
    let map = ptr as *const ObjMap;
    (*map)
        .entries
        .iter()
        .map(|(k, &v)| (k.value(), v))
        .collect()
}

/// Return (or lazily create + cache) a fiber's context map.
unsafe fn get_or_init_context_map(ctx: &mut dyn NativeContext, fiber: *mut ObjFiber) -> Value {
    if !(*fiber).context_map.is_null() {
        return (*fiber).context_map;
    }
    let map = ctx.alloc_map();
    (*fiber).context_map = map;
    map
}

// --- Context / cancellation / deadline ---

fn fiber_context_get(ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    let cur = ctx.get_current_fiber();
    if cur.is_null() {
        ctx.runtime_error("Fiber.context called outside a fiber.".to_string());
        return Value::null();
    }
    unsafe { get_or_init_context_map(ctx, cur) }
}

fn fiber_is_cancelled_static(ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    let cur = ctx.get_current_fiber();
    if cur.is_null() {
        return Value::bool(false);
    }
    unsafe { Value::bool((*cur).cancelled) }
}

fn fiber_cancel_static(ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    let cur = ctx.get_current_fiber();
    if !cur.is_null() {
        unsafe {
            (*cur).cancelled = true;
        }
    }
    Value::null()
}

fn fiber_deadline_ms_static(ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    let cur = ctx.get_current_fiber();
    if cur.is_null() {
        return Value::null();
    }
    unsafe {
        match (*cur).deadline_ms {
            Some(ms) => Value::num(ms),
            None => Value::null(),
        }
    }
}

fn fiber_set_deadline_ms_static(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let cur = ctx.get_current_fiber();
    if cur.is_null() {
        ctx.runtime_error("Fiber.setDeadlineMs called outside a fiber.".to_string());
        return Value::null();
    }
    unsafe {
        (*cur).deadline_ms = if args[1].is_null() {
            None
        } else {
            args[1].as_num()
        };
    }
    Value::null()
}

fn fiber_context_instance(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let fiber = unsafe { as_fiber(args[0]) };
    match fiber {
        Some(f) => unsafe { get_or_init_context_map(ctx, f) },
        None => {
            ctx.runtime_error("Expected a fiber.".to_string());
            Value::null()
        }
    }
}

fn fiber_cancel_instance(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let fiber = unsafe { as_fiber(args[0]) };
    match fiber {
        Some(f) => unsafe {
            (*f).cancelled = true;
        },
        None => {
            ctx.runtime_error("Expected a fiber.".to_string());
        }
    }
    Value::null()
}

fn fiber_is_cancelled_instance(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let fiber = unsafe { as_fiber(args[0]) };
    match fiber {
        Some(f) => unsafe { Value::bool((*f).cancelled) },
        None => {
            ctx.runtime_error("Expected a fiber.".to_string());
            Value::null()
        }
    }
}

fn fiber_deadline_ms_instance(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let fiber = unsafe { as_fiber(args[0]) };
    match fiber {
        Some(f) => unsafe {
            match (*f).deadline_ms {
                Some(ms) => Value::num(ms),
                None => Value::null(),
            }
        },
        None => {
            ctx.runtime_error("Expected a fiber.".to_string());
            Value::null()
        }
    }
}

fn fiber_set_deadline_ms_instance(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let fiber = unsafe { as_fiber(args[0]) };
    match fiber {
        Some(f) => unsafe {
            (*f).deadline_ms = if args[1].is_null() {
                None
            } else {
                args[1].as_num()
            };
        },
        None => {
            ctx.runtime_error("Expected a fiber.".to_string());
        }
    }
    Value::null()
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
    let fiber = unsafe { as_fiber(args[0]) };
    match fiber {
        Some(f) => {
            let state = unsafe { (*f).state };
            match state {
                FiberState::New | FiberState::Suspended => {
                    unsafe { (*f).is_try = true };
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

fn fiber_try_1(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let fiber = unsafe { as_fiber(args[0]) };
    match fiber {
        Some(f) => {
            let state = unsafe { (*f).state };
            match state {
                FiberState::New | FiberState::Suspended => {
                    unsafe { (*f).is_try = true };
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

pub fn bind(vm: &mut VM) {
    let class = vm.fiber_class;

    // Static methods
    vm.primitive_static(class, "new(_)", fiber_new);
    vm.primitive_static(class, "abort(_)", fiber_abort);
    vm.primitive_static(class, "current", fiber_current);
    vm.primitive_static(class, "suspend()", fiber_suspend);
    vm.primitive_static(class, "yield()", fiber_yield_0);
    vm.primitive_static(class, "yield(_)", fiber_yield_1);

    // Context / cancellation / deadline — current-fiber shortcuts.
    vm.primitive_static(class, "context", fiber_context_get);
    vm.primitive_static(class, "isCancelled", fiber_is_cancelled_static);
    vm.primitive_static(class, "cancel()", fiber_cancel_static);
    vm.primitive_static(class, "deadlineMs", fiber_deadline_ms_static);
    vm.primitive_static(class, "setDeadlineMs(_)", fiber_set_deadline_ms_static);

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

    // Per-instance context / cancellation / deadline (inspect / control
    // a fiber from outside it).
    vm.primitive(class, "context", fiber_context_instance);
    vm.primitive(class, "cancel()", fiber_cancel_instance);
    vm.primitive(class, "isCancelled", fiber_is_cancelled_instance);
    vm.primitive(class, "deadlineMs", fiber_deadline_ms_instance);
    vm.primitive(class, "setDeadlineMs(_)", fiber_set_deadline_ms_instance);
}
