//! Built-in ByteArray / Int32Array / Float32Array / Float64Array classes —
//! fixed-size contiguous numeric buffers backed by a raw
//! `Vec<u8>` on the GC heap.
//!
//! One storage type (`ObjTypedArray`) with a `kind` byte carries
//! all four; separate Wren-facing classes provide
//! constructors and `is`-checks while sharing a single set of
//! accessor method implementations (kind-dispatched internally).
//!
//! Surface:
//!
//!   ByteArray.new(count)       — zero-init n-byte buffer
//!   Int32Array.new(count)      — zero-init n-element i32 buffer
//!   Float32Array.new(count)    — zero-init n-element f32 buffer
//!   Float64Array.new(count)    — zero-init n-element f64 buffer
//!   Class.fromList(list)       — copy from a `List<Num>`
//!   Class.fromString(s)        — (ByteArray only) UTF-8 bytes
//!
//!   instance.count             — element count
//!   instance.byteLength        — count * elementSize
//!   instance[i]                — load
//!   instance[i] = v            — store
//!   instance.iterate(iter)     — Sequence iteration
//!   instance.iteratorValue(i)  — Sequence iteration
//!   instance.toList            — convert to `List<Num>`
//!   instance.toString          — debug repr

use crate::runtime::object::{
    NativeContext, ObjHeader, ObjList, ObjType, ObjTypedArray, TypedArrayKind,
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
            ctx.runtime_error(format!("{}: count must be a non-negative integer.", label));
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
        ctx.runtime_error(format!(
            "{}: index {} out of bounds (count {}).",
            label, raw, count
        ));
        return None;
    }
    Some(i as u32)
}

/// Coerce a Value into a byte (0..=255 integer). Used by
/// ByteArray subscript-set.
fn byte_from_value(ctx: &mut dyn NativeContext, v: Value, label: &str) -> Option<u8> {
    match v.as_num() {
        Some(n) if n.is_finite() && (0.0..=255.0).contains(&n) && n.fract() == 0.0 => Some(n as u8),
        _ => {
            ctx.runtime_error(format!("{}: value must be an integer in 0..=255.", label));
            None
        }
    }
}

fn i32_from_value(ctx: &mut dyn NativeContext, v: Value, label: &str) -> Option<i32> {
    match v.as_num() {
        Some(n)
            if n.is_finite()
                && n.fract() == 0.0
                && n >= i32::MIN as f64
                && n <= i32::MAX as f64 =>
        {
            Some(n as i32)
        }
        _ => {
            ctx.runtime_error(format!(
                "{}: value must be an integer in {}..={}.",
                label,
                i32::MIN,
                i32::MAX
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

fn int32_array_new(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(n) = count_arg(ctx, args[1], "Int32Array.new") else {
        return Value::null();
    };
    ctx.alloc_typed_array(n, TypedArrayKind::I32)
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
                ctx.runtime_error(format!("{}: list[{}] must be a number.", label, i));
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
            TypedArrayKind::I32 => {
                if n.fract() != 0.0 || n < i32::MIN as f64 || n > i32::MAX as f64 {
                    ctx.runtime_error(format!(
                        "{}: list[{}] must be an integer in {}..={}.",
                        label,
                        i,
                        i32::MIN,
                        i32::MAX
                    ));
                    return Value::null();
                }
                unsafe { (*arr_ptr).set_i32(i, n as i32) };
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

fn int32_array_from_list(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    from_list_generic(ctx, args, TypedArrayKind::I32, "Int32Array.fromList")
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

/// ByteArray.toUtf8String — interpret the whole buffer as UTF-8 and
/// return a fresh String. One FFI hop, one validation pass, one
/// allocation; replaces the pure-Wren `chars.add(String.fromCodePoint(b))
/// + chars.join("")` pattern that was O(n) iterations + an O(n) join
/// over thousands of one-char Strings.
fn byte_array_to_utf8_string(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let bytes = receiver_ta_bytes(args);
    match std::str::from_utf8(bytes) {
        Ok(s) => ctx.alloc_string(s.to_string()),
        Err(_) => {
            ctx.runtime_error("ByteArray.toUtf8String: bytes are not valid UTF-8.".to_string());
            Value::null()
        }
    }
}

/// ByteArray.utf8Slice(off, len) — interpret bytes [off, off+len) as
/// UTF-8 and return a fresh String. The bounded slice variant that
/// per-token parsers (JSON / TOML / CSV) want — no temp ByteArray
/// allocation, one FFI hop per emitted token.
fn byte_array_utf8_slice(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let bytes = receiver_ta_bytes(args);
    let off = match args[1].as_num() {
        Some(n) if n.is_finite() && n >= 0.0 && n.fract() == 0.0 => n as usize,
        _ => {
            ctx.runtime_error("ByteArray.utf8Slice: off must be a non-negative integer.".into());
            return Value::null();
        }
    };
    let len = match args[2].as_num() {
        Some(n) if n.is_finite() && n >= 0.0 && n.fract() == 0.0 => n as usize,
        _ => {
            ctx.runtime_error("ByteArray.utf8Slice: len must be a non-negative integer.".into());
            return Value::null();
        }
    };
    let end = off.saturating_add(len);
    if end > bytes.len() {
        ctx.runtime_error(format!(
            "ByteArray.utf8Slice: off={} + len={} > buffer length {}.",
            off, len, bytes.len()));
        return Value::null();
    }
    match std::str::from_utf8(&bytes[off..end]) {
        Ok(s) => ctx.alloc_string(s.to_string()),
        Err(_) => {
            ctx.runtime_error("ByteArray.utf8Slice: slice is not valid UTF-8.".to_string());
            Value::null()
        }
    }
}

// Borrow the bytes of the receiver ByteArray. Helper for the
// to_utf8_string / utf8_slice primitives. The receiver lives for
// the duration of the foreign call; the cast extends its lifetime
// to the borrow, which is sound because the GC doesn't relocate
// while a primitive is on the stack.
fn receiver_ta_bytes<'a>(args: &'a [Value]) -> &'a [u8] {
    let arr = receiver_ta(args);
    let b = arr.as_bytes();
    unsafe { std::slice::from_raw_parts(b.as_ptr(), b.len()) }
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
        TypedArrayKind::I32 => Value::num(arr.get_i32(i).unwrap_or(0) as f64),
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
        TypedArrayKind::I32 => {
            let Some(n) = i32_from_value(ctx, args[2], "Int32Array[_]=") else {
                return Value::null();
            };
            unsafe { (*arr_ptr).set_i32(i, n) };
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
        TypedArrayKind::I32 => {
            for i in 0..n {
                out.push(Value::num(arr.get_i32(i).unwrap_or(0) as f64));
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
    ctx.alloc_string(format!("{}({})", arr.kind_tag().class_name(), arr.count))
}

// --- Native bulk decoders (ByteArray-only) ----------------------
//
// These exist so the glTF loader can decode the multi-MB .bin
// chunks of large assets without paying per-byte FFI for every
// f32/u16/u32 element. See memory/project_gltf_parser_perf for
// background. All offsets are in BYTES; bounds are checked once,
// then the inner copy is a tight Rust loop.

fn byte_array_read_f32_le(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let arr = receiver_ta(args);
    let bytes = arr.as_bytes();
    let off = match args[1].as_num() {
        Some(n) if n.is_finite() && n.fract() == 0.0 && n >= 0.0 => n as usize,
        _ => {
            ctx.runtime_error("ByteArray.readF32LE: byteOffset must be a non-negative integer".into());
            return Value::null();
        }
    };
    if off + 4 > bytes.len() {
        ctx.runtime_error(format!("ByteArray.readF32LE: byteOffset {} + 4 out of range (len {})", off, bytes.len()));
        return Value::null();
    }
    let v = f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
    Value::num(v as f64)
}

fn byte_array_read_u16_le(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let arr = receiver_ta(args);
    let bytes = arr.as_bytes();
    let off = match args[1].as_num() {
        Some(n) if n.is_finite() && n.fract() == 0.0 && n >= 0.0 => n as usize,
        _ => {
            ctx.runtime_error("ByteArray.readU16LE: byteOffset must be a non-negative integer".into());
            return Value::null();
        }
    };
    if off + 2 > bytes.len() {
        ctx.runtime_error(format!("ByteArray.readU16LE: byteOffset {} + 2 out of range (len {})", off, bytes.len()));
        return Value::null();
    }
    let v = u16::from_le_bytes([bytes[off], bytes[off + 1]]);
    Value::num(v as f64)
}

fn byte_array_read_u32_le(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let arr = receiver_ta(args);
    let bytes = arr.as_bytes();
    let off = match args[1].as_num() {
        Some(n) if n.is_finite() && n.fract() == 0.0 && n >= 0.0 => n as usize,
        _ => {
            ctx.runtime_error("ByteArray.readU32LE: byteOffset must be a non-negative integer".into());
            return Value::null();
        }
    };
    if off + 4 > bytes.len() {
        ctx.runtime_error(format!("ByteArray.readU32LE: byteOffset {} + 4 out of range (len {})", off, bytes.len()));
        return Value::null();
    }
    let v = u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
    Value::num(v as f64)
}

// Bulk copy: decode `count` little-endian f32 values starting at
// byte offset `src_byte_off` in the receiver ByteArray, write them
// into `dst` (a Float32Array) starting at element offset `dst_off`.
// `stride` is the byte stride between consecutive source f32s
// (≥ 4) — lets the loader honour glTF's strided bufferViews
// without a per-element FFI hop. Stride 0 means tight 4-byte packing.
fn byte_array_copy_to_float32_array(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let arr = receiver_ta(args);
    let src_bytes = arr.as_bytes();
    let src_off = match args[1].as_num() {
        Some(n) if n.is_finite() && n.fract() == 0.0 && n >= 0.0 => n as usize,
        _ => {
            ctx.runtime_error("ByteArray.copyToFloat32Array: srcByteOffset must be a non-negative integer".into());
            return Value::null();
        }
    };
    let count = match args[2].as_num() {
        Some(n) if n.is_finite() && n.fract() == 0.0 && n >= 0.0 => n as usize,
        _ => {
            ctx.runtime_error("ByteArray.copyToFloat32Array: count must be a non-negative integer".into());
            return Value::null();
        }
    };
    let stride = match args[5].as_num() {
        Some(n) if n.is_finite() && n.fract() == 0.0 && n >= 0.0 => {
            let s = n as usize;
            if s == 0 { 4 } else { s }
        }
        _ => 4,
    };
    if stride < 4 {
        ctx.runtime_error(format!("ByteArray.copyToFloat32Array: stride {} < 4", stride));
        return Value::null();
    }
    let dst_obj = match args[3].as_object() {
        Some(p) => p,
        None => {
            ctx.runtime_error("ByteArray.copyToFloat32Array: dst must be a Float32Array".into());
            return Value::null();
        }
    };
    let dst_off = match args[4].as_num() {
        Some(n) if n.is_finite() && n.fract() == 0.0 && n >= 0.0 => n as usize,
        _ => {
            ctx.runtime_error("ByteArray.copyToFloat32Array: dstOffset must be a non-negative integer".into());
            return Value::null();
        }
    };
    unsafe {
        let dst_arr = &mut *(dst_obj as *mut ObjTypedArray);
        if dst_arr.kind_tag() != TypedArrayKind::F32 {
            ctx.runtime_error("ByteArray.copyToFloat32Array: dst must be a Float32Array".into());
            return Value::null();
        }
        if count > 0 {
            let last_src = src_off + (count - 1) * stride + 4;
            if last_src > src_bytes.len() {
                ctx.runtime_error(format!("ByteArray.copyToFloat32Array: source overrun (need {}, have {})", last_src, src_bytes.len()));
                return Value::null();
            }
            if dst_off + count > dst_arr.count as usize {
                ctx.runtime_error(format!("ByteArray.copyToFloat32Array: dst overrun (need {}, have {})", dst_off + count, dst_arr.count));
                return Value::null();
            }
            for i in 0..count {
                let b = src_off + i * stride;
                let v = f32::from_le_bytes([src_bytes[b], src_bytes[b + 1], src_bytes[b + 2], src_bytes[b + 3]]);
                dst_arr.set_f32(dst_off + i, v);
            }
        }
    }
    Value::null()
}

// Bulk copy: decode `count` LE u16 values, widen to i32, write
// into an Int32Array. Same stride semantics; default stride 2.
fn byte_array_copy_to_int32_array_u16(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let arr = receiver_ta(args);
    let src_bytes = arr.as_bytes();
    let src_off = match args[1].as_num() {
        Some(n) if n.is_finite() && n.fract() == 0.0 && n >= 0.0 => n as usize,
        _ => {
            ctx.runtime_error("ByteArray.copyU16LEToInt32Array: srcByteOffset must be a non-negative integer".into());
            return Value::null();
        }
    };
    let count = match args[2].as_num() {
        Some(n) if n.is_finite() && n.fract() == 0.0 && n >= 0.0 => n as usize,
        _ => {
            ctx.runtime_error("ByteArray.copyU16LEToInt32Array: count must be a non-negative integer".into());
            return Value::null();
        }
    };
    let stride = match args[5].as_num() {
        Some(n) if n.is_finite() && n.fract() == 0.0 && n >= 0.0 => {
            let s = n as usize;
            if s == 0 { 2 } else { s }
        }
        _ => 2,
    };
    if stride < 2 {
        ctx.runtime_error(format!("ByteArray.copyU16LEToInt32Array: stride {} < 2", stride));
        return Value::null();
    }
    let dst_obj = match args[3].as_object() {
        Some(p) => p,
        None => {
            ctx.runtime_error("ByteArray.copyU16LEToInt32Array: dst must be an Int32Array".into());
            return Value::null();
        }
    };
    let dst_off = match args[4].as_num() {
        Some(n) if n.is_finite() && n.fract() == 0.0 && n >= 0.0 => n as usize,
        _ => {
            ctx.runtime_error("ByteArray.copyU16LEToInt32Array: dstOffset must be a non-negative integer".into());
            return Value::null();
        }
    };
    unsafe {
        let dst_arr = &mut *(dst_obj as *mut ObjTypedArray);
        if dst_arr.kind_tag() != TypedArrayKind::I32 {
            ctx.runtime_error("ByteArray.copyU16LEToInt32Array: dst must be an Int32Array".into());
            return Value::null();
        }
        if count > 0 {
            let last_src = src_off + (count - 1) * stride + 2;
            if last_src > src_bytes.len() {
                ctx.runtime_error(format!("ByteArray.copyU16LEToInt32Array: source overrun (need {}, have {})", last_src, src_bytes.len()));
                return Value::null();
            }
            if dst_off + count > dst_arr.count as usize {
                ctx.runtime_error(format!("ByteArray.copyU16LEToInt32Array: dst overrun (need {}, have {})", dst_off + count, dst_arr.count));
                return Value::null();
            }
            for i in 0..count {
                let b = src_off + i * stride;
                let v = u16::from_le_bytes([src_bytes[b], src_bytes[b + 1]]) as i32;
                dst_arr.set_i32(dst_off + i, v);
            }
        }
    }
    Value::null()
}

// Bulk copy: u8 → Int32Array. Cheapest variant — straight cast.
fn byte_array_copy_to_int32_array_u8(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let arr = receiver_ta(args);
    let src_bytes = arr.as_bytes();
    let src_off = match args[1].as_num() {
        Some(n) if n.is_finite() && n.fract() == 0.0 && n >= 0.0 => n as usize,
        _ => {
            ctx.runtime_error("ByteArray.copyU8ToInt32Array: srcByteOffset must be a non-negative integer".into());
            return Value::null();
        }
    };
    let count = match args[2].as_num() {
        Some(n) if n.is_finite() && n.fract() == 0.0 && n >= 0.0 => n as usize,
        _ => {
            ctx.runtime_error("ByteArray.copyU8ToInt32Array: count must be a non-negative integer".into());
            return Value::null();
        }
    };
    let stride = match args[5].as_num() {
        Some(n) if n.is_finite() && n.fract() == 0.0 && n >= 0.0 => {
            let s = n as usize;
            if s == 0 { 1 } else { s }
        }
        _ => 1,
    };
    if stride < 1 {
        ctx.runtime_error(format!("ByteArray.copyU8ToInt32Array: stride {} < 1", stride));
        return Value::null();
    }
    let dst_obj = match args[3].as_object() {
        Some(p) => p,
        None => {
            ctx.runtime_error("ByteArray.copyU8ToInt32Array: dst must be an Int32Array".into());
            return Value::null();
        }
    };
    let dst_off = match args[4].as_num() {
        Some(n) if n.is_finite() && n.fract() == 0.0 && n >= 0.0 => n as usize,
        _ => {
            ctx.runtime_error("ByteArray.copyU8ToInt32Array: dstOffset must be a non-negative integer".into());
            return Value::null();
        }
    };
    unsafe {
        let dst_arr = &mut *(dst_obj as *mut ObjTypedArray);
        if dst_arr.kind_tag() != TypedArrayKind::I32 {
            ctx.runtime_error("ByteArray.copyU8ToInt32Array: dst must be an Int32Array".into());
            return Value::null();
        }
        if count > 0 {
            let last_src = src_off + (count - 1) * stride + 1;
            if last_src > src_bytes.len() {
                ctx.runtime_error(format!("ByteArray.copyU8ToInt32Array: source overrun (need {}, have {})", last_src, src_bytes.len()));
                return Value::null();
            }
            if dst_off + count > dst_arr.count as usize {
                ctx.runtime_error(format!("ByteArray.copyU8ToInt32Array: dst overrun (need {}, have {})", dst_off + count, dst_arr.count));
                return Value::null();
            }
            for i in 0..count {
                dst_arr.set_i32(dst_off + i, src_bytes[src_off + i * stride] as i32);
            }
        }
    }
    Value::null()
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
        // Native bulk decoders for binary asset parsers (glTF etc).
        vm.primitive(cls, "readF32LE(_)", byte_array_read_f32_le);
        vm.primitive(cls, "readU16LE(_)", byte_array_read_u16_le);
        vm.primitive(cls, "readU32LE(_)", byte_array_read_u32_le);
        vm.primitive(cls, "copyToFloat32Array(_,_,_,_,_)",
            byte_array_copy_to_float32_array);
        vm.primitive(cls, "copyU16LEToInt32Array(_,_,_,_,_)",
            byte_array_copy_to_int32_array_u16);
        vm.primitive(cls, "copyU8ToInt32Array(_,_,_,_,_)",
            byte_array_copy_to_int32_array_u8);
        // Bulk UTF-8 decoders — replace the pure-Wren
        // `chars.add(String.fromCodePoint(b)); chars.join("")` pattern
        // that O(n²)'s on multi-MB buffers because the join allocates
        // a fresh String per accumulator slot.
        vm.primitive(cls, "toUtf8String", byte_array_to_utf8_string);
        vm.primitive(cls, "utf8Slice(_,_)", byte_array_utf8_slice);
    }

    // Int32Array
    {
        let cls = vm.int32_array_class;
        vm.primitive_static(cls, "new(_)", int32_array_new);
        vm.primitive_static(cls, "fromList(_)", int32_array_from_list);
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
