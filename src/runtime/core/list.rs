use crate::runtime::object::{NativeContext, ObjList};
use crate::runtime::value::Value;
use crate::runtime::vm::VM;

fn receiver_list(args: &[Value]) -> &ObjList {
    unsafe {
        let ptr = args[0].as_object().unwrap();
        &*(ptr as *const ObjList)
    }
}

fn receiver_list_mut(args: &[Value]) -> &mut ObjList {
    unsafe {
        let ptr = args[0].as_object().unwrap();
        &mut *(ptr as *mut ObjList)
    }
}

// Static methods

fn list_new(ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    ctx.alloc_list(vec![])
}

fn list_filled(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let count = match super::validate_int(ctx, args[1], "Count") {
        Some(v) => v,
        None => return Value::null(),
    };
    if count < 0 {
        ctx.runtime_error("Count must be non-negative.".to_string());
        return Value::null();
    }
    let fill = args[2];
    let elements = vec![fill; count as usize];
    ctx.alloc_list(elements)
}

// Instance methods

fn list_subscript(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let list = receiver_list(args);
    let count = list.len();
    let index = match super::validate_index(ctx, args[1], count, "Subscript") {
        Some(v) => v,
        None => return Value::null(),
    };
    list.get(index).unwrap_or(Value::null())
}

fn list_subscript_set(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let count = receiver_list(args).len();
    let index = match super::validate_index(ctx, args[1], count, "Subscript") {
        Some(v) => v,
        None => return Value::null(),
    };
    let list = receiver_list_mut(args);
    list.set(index, args[2]);
    args[2]
}

fn list_add(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let list = receiver_list_mut(args);
    list.add(args[1]);
    args[0]
}

fn list_add_core(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let list = receiver_list_mut(args);
    list.add(args[1]);
    args[0]
}

fn list_clear(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let list = receiver_list_mut(args);
    list.clear();
    Value::null()
}

fn list_count(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let list = receiver_list(args);
    Value::num(list.len() as f64)
}

fn list_insert(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let count = receiver_list(args).len();
    let index = match super::validate_index(ctx, args[1], count + 1, "Index") {
        Some(v) => v,
        None => return Value::null(),
    };
    let list = receiver_list_mut(args);
    list.insert(index, args[2]);
    args[2]
}

fn list_iterate(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let list = receiver_list(args);
    if args[1].is_null() {
        if list.is_empty() {
            return Value::bool(false);
        }
        return Value::num(0.0);
    }

    if !args[1].is_num() {
        return Value::bool(false);
    }

    let index = args[1].as_num().unwrap() + 1.0;
    if index >= list.len() as f64 {
        return Value::bool(false);
    }
    Value::num(index)
}

fn list_iterator_value(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let list = receiver_list(args);
    let index = args[1].as_num().unwrap() as usize;
    list.get(index).unwrap_or(Value::null())
}

fn list_remove_at(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let count = receiver_list(args).len();
    let index = match super::validate_index(ctx, args[1], count, "Index") {
        Some(v) => v,
        None => return Value::null(),
    };
    let list = receiver_list_mut(args);
    list.remove(index).unwrap_or(Value::null())
}

fn list_remove(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let list = receiver_list_mut(args);
    if let Some(pos) = list.as_slice().iter().position(|v| *v == args[1]) {
        list.remove(pos);
        args[1]
    } else {
        Value::null()
    }
}

fn list_index_of(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let list = receiver_list(args);
    match list.as_slice().iter().position(|v| *v == args[1]) {
        Some(i) => Value::num(i as f64),
        None => Value::num(-1.0),
    }
}

fn list_swap(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let count = receiver_list(args).len();
    let index_a = match super::validate_index(ctx, args[1], count, "Index 0") {
        Some(v) => v,
        None => return Value::null(),
    };
    let index_b = match super::validate_index(ctx, args[2], count, "Index 1") {
        Some(v) => v,
        None => return Value::null(),
    };
    let list = receiver_list_mut(args);
    list.swap(index_a, index_b);
    Value::null()
}

pub fn bind(vm: &mut VM) {
    let cls = vm.list_class;

    vm.primitive_static(cls, "new()", list_new);
    vm.primitive_static(cls, "filled(_,_)", list_filled);

    vm.primitive(cls, "[_]", list_subscript);
    vm.primitive(cls, "[_]=(_)", list_subscript_set);
    vm.primitive(cls, "add(_)", list_add);
    vm.primitive(cls, "addCore_(_)", list_add_core);
    vm.primitive(cls, "clear()", list_clear);
    vm.primitive(cls, "count", list_count);
    vm.primitive(cls, "insert(_,_)", list_insert);
    vm.primitive(cls, "iterate(_)", list_iterate);
    vm.primitive(cls, "iteratorValue(_)", list_iterator_value);
    vm.primitive(cls, "removeAt(_)", list_remove_at);
    vm.primitive(cls, "remove(_)", list_remove);
    vm.primitive(cls, "indexOf(_)", list_index_of);
    vm.primitive(cls, "swap(_,_)", list_swap);
}
