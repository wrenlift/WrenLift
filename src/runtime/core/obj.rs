use crate::runtime::object::{NativeContext, NativeFn, ObjClass};
use crate::runtime::value::Value;
use crate::runtime::vm::VM;

fn not(_ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    Value::bool(false)
}

fn eq(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    Value::bool(args[0].equals(args[1]))
}

fn neq(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    Value::bool(!args[0].equals(args[1]))
}

fn is(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    if !args[1].is_object() {
        ctx.runtime_error("Right operand must be a class.".to_string());
        return Value::null();
    }

    let target_class = args[1].as_object().unwrap() as *const ObjClass;
    let mut current = ctx.get_class_of(args[0]);

    while !current.is_null() {
        if std::ptr::eq(current, target_class) {
            return Value::bool(true);
        }
        current = unsafe { (*current).superclass };
    }

    Value::bool(false)
}

fn to_string(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let class_name = ctx.get_class_name_of(args[0]);
    let result = format!("instance of {}", class_name);
    ctx.alloc_string(result)
}

fn type_(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let class_ptr = ctx.get_class_of(args[0]);
    Value::object(class_ptr as *mut u8)
}

fn hash_val(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    args[0].to_bits().hash(&mut hasher);
    Value::num((hasher.finish() & 0x001F_FFFF_FFFF_FFFF) as f64)
}

fn same(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    Value::bool(args[1].to_bits() == args[2].to_bits())
}

pub fn bind(vm: &mut VM) {
    let class = vm.object_class;

    vm.primitive(class, "!", not as NativeFn);
    vm.primitive(class, "==(_)", eq as NativeFn);
    vm.primitive(class, "!=(_)", neq as NativeFn);
    vm.primitive(class, "is(_)", is as NativeFn);
    vm.primitive(class, "toString", to_string as NativeFn);
    vm.primitive(class, "type", type_ as NativeFn);
    vm.primitive(class, "hashCode", hash_val as NativeFn);

    vm.primitive_static(class, "same(_,_)", same as NativeFn);
}
