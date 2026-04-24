//! Optional `uuid` module — UUID generation and parsing.
//!
//! Exposes v4 (random), v5 (namespace + name, SHA-1 based), and v7
//! (time-ordered random) generators. Parsing normalises to canonical
//! hyphenated lowercase form, or returns null on malformed input.
//!
//! v7 is the recommended choice for database keys — monotonic-ish
//! ordering lets B-tree indices stay tight, and the random tail
//! still gives 74 bits of entropy per UUID.

use uuid::Uuid;

use crate::runtime::object::{NativeContext, ObjHeader, ObjList, ObjString, ObjType};
use crate::runtime::value::Value;
use crate::runtime::vm::VM;

// --- Helpers ---------------------------------------------------

unsafe fn string_of_value(v: Value) -> Option<String> {
    if !v.is_object() {
        return None;
    }
    let ptr = v.as_object()?;
    let header = ptr as *const ObjHeader;
    if unsafe { (*header).obj_type } != ObjType::String {
        return None;
    }
    let s = ptr as *const ObjString;
    Some(unsafe { (*s).as_str().to_string() })
}

fn string_arg(ctx: &mut dyn NativeContext, v: Value, label: &str, field: &str) -> Option<String> {
    match unsafe { string_of_value(v) } {
        Some(s) => Some(s),
        None => {
            ctx.runtime_error(format!("{}: {} must be a string.", label, field));
            None
        }
    }
}

/// Namespace keywords — same names as RFC 4122 Appendix C constants.
fn namespace_from_name(name: &str) -> Option<Uuid> {
    match name {
        "dns" | "DNS" => Some(Uuid::NAMESPACE_DNS),
        "url" | "URL" => Some(Uuid::NAMESPACE_URL),
        "oid" | "OID" => Some(Uuid::NAMESPACE_OID),
        "x500" | "X500" => Some(Uuid::NAMESPACE_X500),
        _ => None,
    }
}

fn parse_namespace(ctx: &mut dyn NativeContext, v: Value, label: &str) -> Option<Uuid> {
    let s = string_arg(ctx, v, label, "namespace")?;
    if let Some(u) = namespace_from_name(&s) {
        return Some(u);
    }
    match Uuid::parse_str(&s) {
        Ok(u) => Some(u),
        Err(_) => {
            ctx.runtime_error(format!(
                "{}: namespace must be a UUID string or one of 'dns', 'url', 'oid', 'x500'.",
                label
            ));
            None
        }
    }
}

// --- Generators ------------------------------------------------

fn uuid_v4(ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    ctx.alloc_string(Uuid::new_v4().hyphenated().to_string())
}

fn uuid_v7(ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    ctx.alloc_string(Uuid::now_v7().hyphenated().to_string())
}

fn uuid_v5(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(ns) = parse_namespace(ctx, args[1], "Uuid.v5") else {
        return Value::null();
    };
    let Some(name) = string_arg(ctx, args[2], "Uuid.v5", "name") else {
        return Value::null();
    };
    let u = Uuid::new_v5(&ns, name.as_bytes());
    ctx.alloc_string(u.hyphenated().to_string())
}

fn uuid_nil(ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    ctx.alloc_string(Uuid::nil().hyphenated().to_string())
}

// --- Parsing / validation --------------------------------------

fn uuid_parse(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(s) = string_arg(ctx, args[1], "Uuid.parse", "text") else {
        return Value::null();
    };
    match Uuid::parse_str(&s) {
        // Return the canonicalised hyphenated lowercase form —
        // lets callers use string equality on output regardless
        // of input casing or braces.
        Ok(u) => ctx.alloc_string(u.hyphenated().to_string()),
        Err(_) => Value::null(),
    }
}

fn uuid_is_valid(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(s) = string_arg(ctx, args[1], "Uuid.isValid", "text") else {
        return Value::null();
    };
    Value::bool(Uuid::parse_str(&s).is_ok())
}

fn uuid_version(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(s) = string_arg(ctx, args[1], "Uuid.version", "text") else {
        return Value::null();
    };
    match Uuid::parse_str(&s) {
        Ok(u) => {
            let v = u.get_version_num();
            Value::num(v as f64)
        }
        Err(_) => Value::null(),
    }
}

// --- Byte conversion -------------------------------------------

fn uuid_to_bytes(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(s) = string_arg(ctx, args[1], "Uuid.toBytes", "text") else {
        return Value::null();
    };
    let u = match Uuid::parse_str(&s) {
        Ok(u) => u,
        Err(_) => {
            ctx.runtime_error(format!("Uuid.toBytes: invalid UUID string '{}'.", s));
            return Value::null();
        }
    };
    let bytes = u.as_bytes();
    let elements: Vec<Value> = bytes.iter().map(|&b| Value::num(b as f64)).collect();
    ctx.alloc_list(elements)
}

fn uuid_from_bytes(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let ptr = match args[1].as_object() {
        Some(p) => p as *const ObjList,
        None => {
            ctx.runtime_error("Uuid.fromBytes: bytes must be a list of 16 numbers.".to_string());
            return Value::null();
        }
    };
    let (count, data) = unsafe { ((*ptr).count as usize, (*ptr).elements) };
    if count != 16 {
        ctx.runtime_error(format!(
            "Uuid.fromBytes: expected 16 bytes, got {}.",
            count
        ));
        return Value::null();
    }
    let mut buf = [0u8; 16];
    for (i, byte) in buf.iter_mut().enumerate() {
        let v = unsafe { *data.add(i) };
        let n = match v.as_num() {
            Some(n) => n,
            None => {
                ctx.runtime_error("Uuid.fromBytes: bytes must be numbers.".to_string());
                return Value::null();
            }
        };
        if !(0.0..=255.0).contains(&n) || n.fract() != 0.0 {
            ctx.runtime_error(
                "Uuid.fromBytes: bytes must be integers in 0..=255.".to_string(),
            );
            return Value::null();
        }
        *byte = n as u8;
    }
    let u = Uuid::from_bytes(buf);
    ctx.alloc_string(u.hyphenated().to_string())
}

// --- Registration ----------------------------------------------

pub fn register(vm: &mut VM) -> *mut crate::runtime::object::ObjClass {
    let class = vm.make_class("UuidCore", vm.object_class);

    vm.primitive_static(class, "v4()", uuid_v4);
    vm.primitive_static(class, "v7()", uuid_v7);
    vm.primitive_static(class, "v5(_,_)", uuid_v5);
    vm.primitive_static(class, "nil()", uuid_nil);

    vm.primitive_static(class, "parse(_)", uuid_parse);
    vm.primitive_static(class, "isValid(_)", uuid_is_valid);
    vm.primitive_static(class, "version(_)", uuid_version);

    vm.primitive_static(class, "toBytes(_)", uuid_to_bytes);
    vm.primitive_static(class, "fromBytes(_)", uuid_from_bytes);

    class
}
