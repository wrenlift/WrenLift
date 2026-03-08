use crate::runtime::object::NativeContext;
use crate::runtime::value::Value;
use crate::runtime::vm::VM;

// --- Static methods ---

fn fiber_new(ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    ctx.runtime_error("Fiber.new not yet implemented.".to_string());
    Value::null()
}

fn fiber_abort(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    if args[1].is_object() {
        // Attempt to treat it as a string error message.
        let class_name = ctx.get_class_name_of(args[1]);
        if class_name == "String" {
            let msg = super::as_string(args[1]);
            ctx.runtime_error(msg.to_string());
        } else {
            ctx.runtime_error("Runtime error.".to_string());
        }
    } else if args[1].is_null() {
        // abort(null) clears the error -- but for now just set a generic one.
        ctx.runtime_error("Runtime error.".to_string());
    } else {
        ctx.runtime_error("Runtime error.".to_string());
    }
    Value::null()
}

fn fiber_current(_ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    // Placeholder: return null.
    Value::null()
}

fn fiber_suspend(_ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    // Placeholder: return null.
    Value::null()
}

fn fiber_yield_0(ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    ctx.runtime_error("Fiber.yield not yet implemented.".to_string());
    Value::null()
}

fn fiber_yield_1(ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    ctx.runtime_error("Fiber.yield not yet implemented.".to_string());
    Value::null()
}

// --- Instance methods ---

fn fiber_call_0(ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    ctx.runtime_error("Fiber.call not yet implemented.".to_string());
    Value::null()
}

fn fiber_call_1(ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    ctx.runtime_error("Fiber.call not yet implemented.".to_string());
    Value::null()
}

fn fiber_error(_ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    // Placeholder: return null.
    Value::null()
}

fn fiber_is_done(_ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    // Placeholder: return true.
    Value::bool(true)
}

fn fiber_transfer_0(ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    ctx.runtime_error("Fiber.transfer not yet implemented.".to_string());
    Value::null()
}

fn fiber_transfer_1(ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    ctx.runtime_error("Fiber.transfer not yet implemented.".to_string());
    Value::null()
}

fn fiber_transfer_error(ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    ctx.runtime_error("Fiber.transferError not yet implemented.".to_string());
    Value::null()
}

fn fiber_try_0(ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    ctx.runtime_error("Fiber.try not yet implemented.".to_string());
    Value::null()
}

fn fiber_try_1(ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    ctx.runtime_error("Fiber.try not yet implemented.".to_string());
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
}
