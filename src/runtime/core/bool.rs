use crate::runtime::object::NativeContext;
use crate::runtime::value::Value;
use crate::runtime::vm::VM;

fn not(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    if args[0].as_bool().unwrap() {
        Value::bool(false)
    } else {
        Value::bool(true)
    }
}

fn to_string(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    if args[0].as_bool().unwrap() {
        ctx.alloc_string("true".to_string())
    } else {
        ctx.alloc_string("false".to_string())
    }
}

pub fn bind(vm: &mut VM) {
    vm.primitive(vm.bool_class, "!", not);
    vm.primitive(vm.bool_class, "toString", to_string);
}
