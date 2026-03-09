use crate::runtime::object::{MapKey, NativeContext, ObjMap};
use crate::runtime::value::Value;
use crate::runtime::vm::VM;

fn receiver_map(args: &[Value]) -> &ObjMap {
    unsafe { &*(args[0].as_object().unwrap() as *const ObjMap) }
}

fn receiver_map_mut(args: &[Value]) -> &mut ObjMap {
    unsafe { &mut *(args[0].as_object().unwrap() as *mut ObjMap) }
}

// Static methods

fn map_new(ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    ctx.alloc_map()
}

// Instance methods

fn map_subscript(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let map = receiver_map(args);
    let key = MapKey::new(args[1]);
    match map.entries.get(&key) {
        Some(val) => *val,
        None => Value::null(),
    }
}

fn map_subscript_set(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let map = receiver_map_mut(args);
    let key = MapKey::new(args[1]);
    map.entries.insert(key, args[2]);
    args[2]
}

fn map_add_core(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let map = receiver_map_mut(args);
    let key = MapKey::new(args[1]);
    map.entries.insert(key, args[2]);
    args[0]
}

fn map_clear(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let map = receiver_map_mut(args);
    map.entries.clear();
    Value::null()
}

fn map_contains_key(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let map = receiver_map(args);
    let key = MapKey::new(args[1]);
    Value::bool(map.entries.contains_key(&key))
}

fn map_count(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let map = receiver_map(args);
    Value::num(map.entries.len() as f64)
}

fn map_remove(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let map = receiver_map_mut(args);
    let key = MapKey::new(args[1]);
    match map.entries.remove(&key) {
        Some(val) => val,
        None => Value::null(),
    }
}

fn map_iterate(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let map = receiver_map(args);

    if map.entries.is_empty() {
        return Value::bool(false);
    }

    let keys: Vec<&MapKey> = map.entries.keys().collect();

    if args[1].is_null() {
        return keys[0].value();
    }

    let current = MapKey::new(args[1]);
    for (i, key) in keys.iter().enumerate() {
        if **key == current {
            if i + 1 < keys.len() {
                return keys[i + 1].value();
            } else {
                return Value::bool(false);
            }
        }
    }

    Value::bool(false)
}

fn map_key_iterator_value(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    args[1]
}

fn map_value_iterator_value(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let map = receiver_map(args);
    let key = MapKey::new(args[1]);
    match map.entries.get(&key) {
        Some(val) => *val,
        None => Value::null(),
    }
}

fn map_keys(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let map = receiver_map(args);
    let keys: Vec<Value> = map.entries.keys().map(|k| k.value()).collect();
    ctx.alloc_list(keys)
}

fn map_values(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let map = receiver_map(args);
    let vals: Vec<Value> = map.entries.values().copied().collect();
    ctx.alloc_list(vals)
}

fn map_to_string(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let map = receiver_map(args);
    let mut result = String::from("{");
    let mut first = true;
    for (key, val) in &map.entries {
        if !first { result.push_str(", "); }
        first = false;
        let k = super::sequence::value_to_string(ctx, key.value());
        let v = super::sequence::value_to_string(ctx, *val);
        result.push_str(&format!("{}: {}", k, v));
    }
    result.push('}');
    ctx.alloc_string(result)
}

pub fn bind(vm: &mut VM) {
    let cls = vm.map_class;

    vm.primitive_static(cls, "new()", map_new);

    vm.primitive(cls, "[_]", map_subscript);
    vm.primitive(cls, "[_]=(_)", map_subscript_set);
    vm.primitive(cls, "addCore_(_,_)", map_add_core);
    vm.primitive(cls, "clear()", map_clear);
    vm.primitive(cls, "containsKey(_)", map_contains_key);
    vm.primitive(cls, "count", map_count);
    vm.primitive(cls, "remove(_)", map_remove);
    vm.primitive(cls, "iterate(_)", map_iterate);
    vm.primitive(cls, "keyIteratorValue_(_)", map_key_iterator_value);
    vm.primitive(cls, "valueIteratorValue_(_)", map_value_iterator_value);
    vm.primitive(cls, "keys", map_keys);
    vm.primitive(cls, "values", map_values);
    vm.primitive(cls, "toString", map_to_string);
}
