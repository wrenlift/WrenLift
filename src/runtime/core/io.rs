//! Optional `io` module — low-level byte / text conversions that
//! pure Wren can't do on its own. Exists so @hatch:io's `Buffer`
//! can round-trip between `List<Num>` byte form and Wren strings.
//!
//! Deliberately tiny: the stream interfaces live in @hatch:io as
//! pure Wren, and each concrete source (in-memory, process, file)
//! gets its own native primitives in the relevant module (proc,
//! fs). `io` just owns the codec so callers don't have to reach
//! into hash / os for it.

use crate::runtime::object::{NativeContext, ObjHeader, ObjString, ObjType};
use crate::runtime::value::Value;
use crate::runtime::vm::VM;

use super::bytes_from_byte_list as bytes_from_list;

/// `IoCore.bytesToString(list)` — decode UTF-8 bytes into a Wren
/// String. Invalid sequences become U+FFFD. We prefer lossy over
/// erroring because text pulled off a subprocess's stdout
/// commonly has tails of multi-byte characters split across
/// read boundaries; callers can validate separately if they care.
fn io_bytes_to_string(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(bytes) = bytes_from_list(ctx, args[1], "Io.bytesToString") else {
        return Value::null();
    };
    let s = String::from_utf8_lossy(&bytes).into_owned();
    ctx.alloc_string(s)
}

/// `IoCore.stringToBytes(s)` — return the UTF-8 byte form of a
/// Wren string as a List<Num>. Wren strings are already UTF-8
/// internally, so this is a pure copy.
fn io_string_to_bytes(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    if !args[1].is_object() {
        ctx.runtime_error("Io.stringToBytes: expected a string.".to_string());
        return Value::null();
    }
    let ptr = args[1].as_object().unwrap();
    let header = ptr as *const ObjHeader;
    unsafe {
        if (*header).obj_type != ObjType::String {
            ctx.runtime_error("Io.stringToBytes: expected a string.".to_string());
            return Value::null();
        }
    }
    let s = ptr as *const ObjString;
    let bytes: Vec<Value> = unsafe { (*s).as_str() }
        .as_bytes()
        .iter()
        .map(|&b| Value::num(b as f64))
        .collect();
    ctx.alloc_list(bytes)
}

pub fn register(vm: &mut VM) -> *mut crate::runtime::object::ObjClass {
    let class = vm.make_class("IoCore", vm.object_class);
    vm.primitive_static(class, "bytesToString(_)", io_bytes_to_string);
    vm.primitive_static(class, "stringToBytes(_)", io_string_to_bytes);
    class
}
