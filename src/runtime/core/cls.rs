use crate::runtime::object::{NativeContext, NativeFn, ObjClass};
use crate::runtime::value::Value;
use crate::runtime::vm::VM;

fn name(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let class_ptr = args[0].as_object().unwrap() as *const ObjClass;
    let symbol_id = unsafe { (*class_ptr).name };
    let name_str = ctx.resolve_symbol(symbol_id);
    ctx.alloc_string(name_str.to_string())
}

fn supertype(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let class_ptr = args[0].as_object().unwrap() as *const ObjClass;
    let superclass = unsafe { (*class_ptr).superclass };

    if superclass.is_null() {
        Value::null()
    } else {
        Value::object(superclass as *mut u8)
    }
}

fn to_string(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    name(ctx, args)
}

pub fn bind(vm: &mut VM) {
    let class = vm.class_class;

    vm.primitive(class, "name", name as NativeFn);
    vm.primitive(class, "supertype", supertype as NativeFn);
    vm.primitive(class, "toString", to_string as NativeFn);
}
