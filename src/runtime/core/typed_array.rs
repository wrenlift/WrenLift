//! Built-in ByteArray / Float32Array / Float64Array classes —
//! fixed-size contiguous numeric buffers backed by a raw
//! `Vec<u8>` on the GC heap.
//!
//! One storage type (`ObjTypedArray`) with a `kind` byte carries
//! all three; three separate Wren-facing classes provide
//! constructors and `is`-checks while sharing a single set of
//! accessor method implementations (kind-dispatched internally).
//!
//! Surface:
//!
//!   ByteArray.new(count)       — zero-init n-byte buffer
//!   Float32Array.new(count)    — zero-init n-element f32 buffer
//!   Float64Array.new(count)    — zero-init n-element f64 buffer
//!   Class.fromList(list)       — copy from a List<Num>
//!   Class.fromString(s)        — (ByteArray only) UTF-8 bytes
//!
//!   instance.count             — element count
//!   instance.byteLength        — count * elementSize
//!   instance[i]                — load
//!   instance[i] = v            — store
//!   instance.iterate(iter)     — Sequence iteration
//!   instance.iteratorValue(i)  — Sequence iteration
//!   instance.toList            — convert to List<Num>
//!   instance.toString          — debug repr

use crate::runtime::object::{
    NativeContext, ObjHeader, ObjList, ObjTypedArray, ObjType, TypedArrayKind,
};
use crate::runtime::value::Value;
use crate::runtime::vm::VM;

// --- Helpers ----------------------------------------------------

fn receiver_ta(args: &[Value]) -> &ObjTypedArray {
    unsafe {
        let ptr = args[0].as_object().unwrap();
        &*(ptr as *const ObjTypedArray)
    }
}

/// Pull a non-negative integer count out of a Value, or signal
/// an error. Used by every `new(_)` constructor.
fn count_arg(ctx: &mut dyn NativeContext, v: Value, label: &str) -> Option<u32> {
    match v.as_num() {
        Some(n) if n.is_finite() && n >= 0.0 && n.fract() == 0.0 && n <= u32::MAX as f64 => {
            Some(n as u32)
        }
        _ => {
            ctx.runtime_error(format!(
                "{}: count must be a non-negative integer.",
                label
            ));
            None
        }
    }
}

/// Validate + normalize an index against the element count, with
/// Wren-style negative index support.
fn index_arg(ctx: &mut dyn NativeContext, v: Value, count: u32, label: &str) -> Option<u32> {
    let raw = match v.as_num() {
        Some(n) if n.is_finite() && n.fract() == 0.0 => n as i64,
        _ => {
            ctx.runtime_error(format!("{}: index must be an integer.", label));
            return None;
        }
    };
    let c = count as i64;
    let i = if raw < 0 { raw + c } else { raw };
    if i < 0 || i >= c {
        ctx.runtime_error(format!("{}: index {} out of bounds (count {}).", label, raw, count));
        return None;
    }
    Some(i as u32)
}

/// Coerce a Value into a byte (0..=255 integer). Used by
/// ByteArray subscript-set.
fn byte_from_value(ctx: &mut dyn NativeContext, v: Value, label: &str) -> Option<u8> {
    match v.as_num() {
        Some(n) if n.is_finite() && (0.0..=255.0).contains(&n) && n.fract() == 0.0 => {
            Some(n as u8)
        }
        _ => {
            ctx.runtime_error(format!(
                "{}: value must be an integer in 0..=255.",
                label
            ));
            None
        }
    }
}

// --- Construction ----------------------------------------------

fn byte_array_new(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(n) = count_arg(ctx, args[1], "ByteArray.new") else {
        return Value::null();
    };
    ctx.alloc_typed_array(n, TypedArrayKind::U8)
}

fn float32_array_new(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(n) = count_arg(ctx, args[1], "Float32Array.new") else {
        return Value::null();
    };
    ctx.alloc_typed_array(n, TypedArrayKind::F32)
}

fn float64_array_new(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(n) = count_arg(ctx, args[1], "Float64Array.new") else {
        return Value::null();
    };
    ctx.alloc_typed_array(n, TypedArrayKind::F64)
}

/// `Class.fromList(list)` — allocate + copy every element from a
/// Wren List. Elements must be Num; ByteArray additionally
/// requires integer bytes.
fn from_list_generic(
    ctx: &mut dyn NativeContext,
    args: &[Value],
    kind: TypedArrayKind,
    label: &str,
) -> Value {
    if !args[1].is_object() {
        ctx.runtime_error(format!("{}: expected a List.", label));
        return Value::null();
    }
    let ptr = args[1].as_object().unwrap();
    let header = ptr as *const ObjHeader;
    if unsafe { (*header).obj_type } != ObjType::List {
        ctx.runtime_error(format!("{}: expected a List.", label));
        return Value::null();
    }
    let list = unsafe { &*(ptr as *const ObjList) };
    let count = list.count;
    let result = ctx.alloc_typed_array(count, kind);
    let arr_ptr = result.as_object().unwrap() as *mut ObjTypedArray;

    for i in 0..count as usize {
        let v = list.get(i).unwrap_or(Value::null());
        let n = match v.as_num() {
            Some(n) => n,
            None => {
                ctx.runtime_error(format!(
                    "{}: list[{}] must be a number.",
                    label, i
                ));
                return Value::null();
            }
        };
        match kind {
            TypedArrayKind::U8 => {
                if !(0.0..=255.0).contains(&n) || n.fract() != 0.0 {
                    ctx.runtime_error(format!(
                        "{}: list[{}] must be an integer in 0..=255.",
                        label, i
                    ));
                    return Value::null();
                }
                unsafe { (*arr_ptr).set_u8(i, n as u8) };
            }
            TypedArrayKind::F32 => unsafe { (*arr_ptr).set_f32(i, n as f32) },
            TypedArrayKind::F64 => unsafe { (*arr_ptr).set_f64(i, n) },
        }
    }
    result
}

fn byte_array_from_list(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    from_list_generic(ctx, args, TypedArrayKind::U8, "ByteArray.fromList")
}

fn float32_array_from_list(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    from_list_generic(ctx, args, TypedArrayKind::F32, "Float32Array.fromList")
}

fn float64_array_from_list(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    from_list_generic(ctx, args, TypedArrayKind::F64, "Float64Array.fromList")
}

/// ByteArray-only: allocate a buffer holding the UTF-8 bytes of a
/// String. Round-trips back to a String through `@hatch:io`.
fn byte_array_from_string(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(s) = super::validate_string(ctx, args[1], "ByteArray.fromString") else {
        return Value::null();
    };
    let bytes = s.as_bytes();
    let result = ctx.alloc_typed_array(bytes.len() as u32, TypedArrayKind::U8);
    let arr_ptr = result.as_object().unwrap() as *mut ObjTypedArray;
    unsafe {
        let dst = (*arr_ptr).as_bytes_mut();
        dst.copy_from_slice(bytes);
    }
    result
}

// --- Accessors -------------------------------------------------

fn ta_count(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    Value::num(receiver_ta(args).count as f64)
}

fn ta_byte_length(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    Value::num(receiver_ta(args).byte_len() as f64)
}

fn ta_subscript(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let arr = receiver_ta(args);
    let Some(i) = index_arg(ctx, args[1], arr.count, "TypedArray subscript") else {
        return Value::null();
    };
    let i = i as usize;
    match arr.kind_tag() {
        TypedArrayKind::U8 => Value::num(arr.get_u8(i).unwrap_or(0) as f64),
        TypedArrayKind::F32 => Value::num(arr.get_f32(i).unwrap_or(0.0) as f64),
        TypedArrayKind::F64 => Value::num(arr.get_f64(i).unwrap_or(0.0)),
    }
}

fn ta_subscript_set(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let arr_ptr = args[0].as_object().unwrap() as *mut ObjTypedArray;
    let (count, kind) = unsafe { ((*arr_ptr).count, (*arr_ptr).kind_tag()) };
    let Some(i) = index_arg(ctx, args[1], count, "TypedArray subscript=") else {
        return Value::null();
    };
    let i = i as usize;
    match kind {
        TypedArrayKind::U8 => {
            let Some(b) = byte_from_value(ctx, args[2], "ByteArray[_]=") else {
                return Value::null();
            };
            unsafe { (*arr_ptr).set_u8(i, b) };
        }
        TypedArrayKind::F32 => {
            let Some(n) = args[2].as_num() else {
                ctx.runtime_error("Float32Array[_]=: value must be a number.".to_string());
                return Value::null();
            };
            unsafe { (*arr_ptr).set_f32(i, n as f32) };
        }
        TypedArrayKind::F64 => {
            let Some(n) = args[2].as_num() else {
                ctx.runtime_error("Float64Array[_]=: value must be a number.".to_string());
                return Value::null();
            };
            unsafe { (*arr_ptr).set_f64(i, n) };
        }
    }
    args[2]
}

// Sequence iteration: iterate(null) → 0; iterate(i) → i+1 until out of range.
fn ta_iterate(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let arr = receiver_ta(args);
    if args[1].is_null() {
        return if arr.count == 0 {
            Value::bool(false)
        } else {
            Value::num(0.0)
        };
    }
    let Some(n) = args[1].as_num() else {
        return Value::bool(false);
    };
    let next = (n as i64) + 1;
    if next < 0 || next as u32 >= arr.count {
        Value::bool(false)
    } else {
        Value::num(next as f64)
    }
}

fn ta_iterator_value(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    // Reuse subscript's element-type dispatch.
    ta_subscript(ctx, args)
}

fn ta_to_list(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let arr = receiver_ta(args);
    let n = arr.count as usize;
    let mut out: Vec<Value> = Vec::with_capacity(n);
    match arr.kind_tag() {
        TypedArrayKind::U8 => {
            for i in 0..n {
                out.push(Value::num(arr.get_u8(i).unwrap_or(0) as f64));
            }
        }
        TypedArrayKind::F32 => {
            for i in 0..n {
                out.push(Value::num(arr.get_f32(i).unwrap_or(0.0) as f64));
            }
        }
        TypedArrayKind::F64 => {
            for i in 0..n {
                out.push(Value::num(arr.get_f64(i).unwrap_or(0.0)));
            }
        }
    }
    ctx.alloc_list(out)
}

fn ta_to_string(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let arr = receiver_ta(args);
    ctx.alloc_string(format!(
        "{}({})",
        arr.kind_tag().class_name(),
        arr.count
    ))
}

// --- Binding ----------------------------------------------------

pub fn bind(vm: &mut VM) {
    // ByteArray
    {
        let cls = vm.byte_array_class;
        vm.primitive_static(cls, "new(_)", byte_array_new);
        vm.primitive_static(cls, "fromList(_)", byte_array_from_list);
        vm.primitive_static(cls, "fromString(_)", byte_array_from_string);
        bind_shared_instance(vm, cls);
    }

    // Float32Array
    {
        let cls = vm.float32_array_class;
        vm.primitive_static(cls, "new(_)", float32_array_new);
        vm.primitive_static(cls, "fromList(_)", float32_array_from_list);
        bind_shared_instance(vm, cls);
    }

    // Float64Array
    {
        let cls = vm.float64_array_class;
        vm.primitive_static(cls, "new(_)", float64_array_new);
        vm.primitive_static(cls, "fromList(_)", float64_array_from_list);
        bind_shared_instance(vm, cls);
    }
}

fn bind_shared_instance(vm: &mut VM, cls: *mut crate::runtime::object::ObjClass) {
    vm.primitive(cls, "count", ta_count);
    vm.primitive(cls, "byteLength", ta_byte_length);
    vm.primitive(cls, "[_]", ta_subscript);
    vm.primitive(cls, "[_]=(_)", ta_subscript_set);
    vm.primitive(cls, "iterate(_)", ta_iterate);
    vm.primitive(cls, "iteratorValue(_)", ta_iterator_value);
    vm.primitive(cls, "toList", ta_to_list);
    vm.primitive(cls, "toString", ta_to_string);
}
