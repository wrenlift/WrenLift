use crate::mir::{AttrEntry, AttrValue};
use crate::runtime::object::{NativeContext, NativeFn, ObjClass, ObjList, ObjMap};
use crate::runtime::value::Value;
use crate::runtime::vm::VM;

/// One group bucket in the in-flight attribute map: its name (None for
/// ungrouped), the underlying `ObjMap`, and the running list of keys in
/// insertion order (each paired with the `ObjList` that accumulates that
/// key's values).
struct GroupBucket {
    name: Option<String>,
    map: *mut ObjMap,
    keys: Vec<(String, *mut ObjList)>,
}

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

/// Class.attributes — returns the class-level attribute table as a nested
/// Map `{ group?: { key: [values] } }`, or `null` if the class has no
/// runtime attributes.
fn attributes(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let class_ptr = args[0].as_object().unwrap() as *const ObjClass;
    let entries: &[AttrEntry] = unsafe { &(*class_ptr).attributes };
    if entries.is_empty() {
        return Value::null();
    }
    build_attr_map(ctx, entries)
}

/// Class.methodAttributes — returns `{ signature: { group?: { key: [values] } } }`
/// for every method that carries at least one runtime attribute, or `null`
/// if the class has none.
fn method_attributes(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let class_ptr = args[0].as_object().unwrap() as *const ObjClass;
    let method_attrs = unsafe { &(*class_ptr).method_attributes };
    if method_attrs.is_empty() {
        return Value::null();
    }
    let outer = ctx.alloc_map();
    let outer_ptr = outer.as_object().unwrap() as *mut ObjMap;
    // Collect to a stable order so the output is deterministic run-to-run.
    let mut keys: Vec<_> = method_attrs.keys().copied().collect();
    keys.sort_by_key(|k| k.index());
    for sig_sym in keys {
        let entries = method_attrs.get(&sig_sym).unwrap();
        let sig_name = ctx.resolve_symbol(sig_sym).to_string();
        let sig_key = ctx.alloc_string(sig_name);
        let inner = build_attr_map(ctx, entries);
        unsafe { (*outer_ptr).set(sig_key, inner) };
    }
    outer
}

/// Build one `{ group?: { key: [values] } }` map from a flat attribute list.
fn build_attr_map(ctx: &mut dyn NativeContext, entries: &[AttrEntry]) -> Value {
    let outer = ctx.alloc_map();
    let outer_ptr = outer.as_object().unwrap() as *mut ObjMap;

    let mut buckets: Vec<GroupBucket> = Vec::new();

    for entry in entries {
        let bucket_idx = match buckets.iter().position(|b| b.name == entry.group) {
            Some(i) => i,
            None => {
                let group_val = ctx.alloc_map();
                let group_ptr = group_val.as_object().unwrap() as *mut ObjMap;
                let key_val = match &entry.group {
                    Some(name) => ctx.alloc_string(name.clone()),
                    None => Value::null(),
                };
                unsafe { (*outer_ptr).set(key_val, group_val) };
                buckets.push(GroupBucket {
                    name: entry.group.clone(),
                    map: group_ptr,
                    keys: Vec::new(),
                });
                buckets.len() - 1
            }
        };

        let bucket = &mut buckets[bucket_idx];
        let list_ptr = match bucket.keys.iter().find(|(k, _)| k == &entry.key) {
            Some((_, p)) => *p,
            None => {
                let list_val = ctx.alloc_list(Vec::new());
                let list_ptr = list_val.as_object().unwrap() as *mut ObjList;
                let key_val = ctx.alloc_string(entry.key.clone());
                unsafe { (*bucket.map).set(key_val, list_val) };
                bucket.keys.push((entry.key.clone(), list_ptr));
                list_ptr
            }
        };

        let value = match &entry.value {
            None => Value::null(),
            Some(AttrValue::Num(n)) => Value::num(*n),
            Some(AttrValue::Bool(b)) => Value::bool(*b),
            Some(AttrValue::Null) => Value::null(),
            Some(AttrValue::Str(s)) => ctx.alloc_string(s.clone()),
            Some(AttrValue::Ident(s)) => ctx.alloc_string(s.clone()),
        };
        unsafe { (*list_ptr).add(value) };
    }

    outer
}

pub fn bind(vm: &mut VM) {
    let class = vm.class_class;

    vm.primitive(class, "name", name as NativeFn);
    vm.primitive(class, "supertype", supertype as NativeFn);
    vm.primitive(class, "toString", to_string as NativeFn);
    vm.primitive(class, "attributes", attributes as NativeFn);
    vm.primitive(class, "methodAttributes", method_attributes as NativeFn);
}
