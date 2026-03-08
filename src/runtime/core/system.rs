use crate::runtime::object::NativeContext;
use crate::runtime::value::Value;
use crate::runtime::vm::VM;
use std::time::{SystemTime, UNIX_EPOCH};

fn system_clock(_ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    Value::num(duration.as_secs_f64())
}

fn system_gc(_ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    // Trigger garbage collection. For now, no-op.
    Value::null()
}

fn system_write_string(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let s = super::as_string(args[1]);
    print!("{}", s);
    args[1]
}

pub fn bind(vm: &mut VM) {
    let class = vm.system_class;

    vm.primitive_static(class, "clock", system_clock);
    vm.primitive_static(class, "gc()", system_gc);
    vm.primitive_static(class, "writeString_(_)", system_write_string);
}
