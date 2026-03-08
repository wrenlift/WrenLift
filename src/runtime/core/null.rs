use crate::runtime::object::NativeContext;
use crate::runtime::value::Value;
use crate::runtime::vm::VM;

fn not(_ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    Value::bool(true)
}

fn to_string(ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    ctx.alloc_string("null".to_string())
}

pub fn bind(vm: &mut VM) {
    vm.primitive(vm.null_class, "!", not);
    vm.primitive(vm.null_class, "toString", to_string);
}
