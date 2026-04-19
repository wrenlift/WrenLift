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

fn system_gc(ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    ctx.trigger_gc();
    Value::null()
}

fn system_write_string(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let s = super::as_string(args[1]);
    ctx.write_output(s);
    args[1]
}

fn format_object(ctx: &dyn NativeContext, value: Value) -> String {
    if value.is_null() {
        "null".to_string()
    } else if value.is_num() {
        let n = value.as_num().unwrap();
        if n == n.trunc() && n.is_finite() && n.abs() < 1e15 {
            format!("{}", n as i64)
        } else {
            format!("{}", n)
        }
    } else if value.is_bool() {
        (if value.as_bool().unwrap() {
            "true"
        } else {
            "false"
        })
        .to_string()
    } else if value.is_object() {
        if super::is_string(value) {
            super::as_string(value).to_string()
        } else {
            // For non-string objects, format by type
            let ptr = value.as_object().unwrap();
            let header = unsafe { &*(ptr as *const super::super::object::ObjHeader) };
            match header.obj_type {
                super::super::object::ObjType::List => {
                    let list = unsafe { &*(ptr as *const super::super::object::ObjList) };
                    let parts: Vec<String> = (0..list.len())
                        .map(|i| format_object(ctx, list.get(i).unwrap_or(Value::null())))
                        .collect();
                    format!("[{}]", parts.join(", "))
                }
                super::super::object::ObjType::Range => {
                    let range = unsafe { &*(ptr as *const super::super::object::ObjRange) };
                    let op = if range.is_inclusive { ".." } else { "..." };
                    format!(
                        "{}{}{}",
                        format_object(ctx, Value::num(range.from)),
                        op,
                        format_object(ctx, Value::num(range.to))
                    )
                }
                super::super::object::ObjType::Map => {
                    let map = unsafe { &*(ptr as *const super::super::object::ObjMap) };
                    let parts: Vec<String> = map
                        .entries
                        .iter()
                        .map(|(k, v)| format!("{}: {}", format_object(ctx, k.0), format_object(ctx, *v)))
                        .collect();
                    format!("{{{}}}", parts.join(", "))
                }
                super::super::object::ObjType::Fn => "Fn".to_string(),
                super::super::object::ObjType::Closure => "Fn".to_string(),
                super::super::object::ObjType::Fiber => "Fiber".to_string(),
                super::super::object::ObjType::Class => {
                    // `System.print(List)` should render the class's
                    // name, matching what `Class.toString` returns
                    // when dispatched explicitly. Falling through
                    // to "instance" would hide the real value and
                    // make dispatch-path bugs harder to spot.
                    let class =
                        unsafe { &*(ptr as *const super::super::object::ObjClass) };
                    ctx.resolve_symbol(class.name).to_string()
                }
                _ => "instance".to_string(),
            }
        }
    } else {
        String::new()
    }
}

fn system_print_no_arg(ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    ctx.write_output("\n");
    Value::null()
}

fn system_print(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let s = format_object(ctx, args[1]);
    ctx.write_output(&s);
    ctx.write_output("\n");
    args[1]
}

fn system_write(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let s = format_object(ctx, args[1]);
    ctx.write_output(&s);
    args[1]
}

fn system_print_all(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    if args[1].is_object() {
        let ptr = args[1].as_object().unwrap();
        let header = ptr as *const super::super::object::ObjHeader;
        if unsafe { (*header).obj_type } == super::super::object::ObjType::List {
            let list = unsafe { &*(ptr as *const super::super::object::ObjList) };
            for elem in list.as_slice() {
                ctx.write_output(&format_object(ctx, *elem));
            }
        } else {
            ctx.write_output(&format_object(ctx, args[1]));
        }
    } else {
        ctx.write_output(&format_object(ctx, args[1]));
    }
    ctx.write_output("\n");
    args[1]
}

fn system_write_all(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    if args[1].is_object() {
        let ptr = args[1].as_object().unwrap();
        let header = ptr as *const super::super::object::ObjHeader;
        if unsafe { (*header).obj_type } == super::super::object::ObjType::List {
            let list = unsafe { &*(ptr as *const super::super::object::ObjList) };
            for elem in list.as_slice() {
                ctx.write_output(&format_object(ctx, *elem));
            }
        } else {
            ctx.write_output(&format_object(ctx, args[1]));
        }
    } else {
        ctx.write_output(&format_object(ctx, args[1]));
    }
    args[1]
}

fn system_write_object(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let obj = args[1];
    // Try calling toString on the object
    if let Some(s) = ctx.call_method_on(obj, "toString", &[]) {
        if super::is_string(s) {
            let text = super::as_string(s);
            ctx.write_output(text);
            return Value::null();
        }
    }
    ctx.write_output("[invalid toString]");
    Value::null()
}

pub fn bind(vm: &mut VM) {
    let class = vm.system_class;

    vm.primitive_static(class, "clock", system_clock);
    vm.primitive_static(class, "gc()", system_gc);
    vm.primitive_static(class, "writeString_(_)", system_write_string);
    vm.primitive_static(class, "writeObject_(_)", system_write_object);
    vm.primitive_static(class, "print", system_print_no_arg);
    vm.primitive_static(class, "print(_)", system_print);
    vm.primitive_static(class, "write(_)", system_write);
    vm.primitive_static(class, "printAll(_)", system_print_all);
    vm.primitive_static(class, "writeAll(_)", system_write_all);
}
