//! Built-in Simd / Simd4f / Simd4i classes.
//!
//! Storage lives in `ObjSimd`, a fixed-width inline 16-byte payload whose
//! interpretation is selected by `SimdKind`.

use crate::runtime::object::{
    NativeContext, ObjHeader, ObjSimd, ObjType, ObjTypedArray, SimdKind, TypedArrayKind,
};
use crate::runtime::value::Value;
use crate::runtime::vm::VM;

fn receiver_simd(args: &[Value]) -> &ObjSimd {
    unsafe {
        let ptr = args[0].as_object().unwrap();
        &*(ptr as *const ObjSimd)
    }
}

fn simd_arg(
    ctx: &mut dyn NativeContext,
    value: Value,
    expected: SimdKind,
    label: &str,
) -> Option<*mut ObjSimd> {
    let ptr = value.as_object().unwrap_or(std::ptr::null_mut());
    if ptr.is_null() {
        ctx.runtime_error(format!("{}: expected {}.", label, expected.class_name()));
        return None;
    }
    let header = ptr as *const ObjHeader;
    if unsafe { (*header).obj_type } != ObjType::Simd {
        ctx.runtime_error(format!("{}: expected {}.", label, expected.class_name()));
        return None;
    }
    let simd = ptr as *mut ObjSimd;
    if unsafe { (*simd).kind_tag() } != expected {
        ctx.runtime_error(format!("{}: expected {}.", label, expected.class_name()));
        return None;
    }
    Some(simd)
}

fn typed_array_arg(
    ctx: &mut dyn NativeContext,
    value: Value,
    expected: TypedArrayKind,
    label: &str,
) -> Option<*mut ObjTypedArray> {
    let ptr = value.as_object().unwrap_or(std::ptr::null_mut());
    if ptr.is_null() {
        ctx.runtime_error(format!("{}: expected {}.", label, expected.class_name()));
        return None;
    }
    let header = ptr as *const ObjHeader;
    if unsafe { (*header).obj_type } != ObjType::TypedArray {
        ctx.runtime_error(format!("{}: expected {}.", label, expected.class_name()));
        return None;
    }
    let arr = ptr as *mut ObjTypedArray;
    if unsafe { (*arr).kind_tag() } != expected {
        ctx.runtime_error(format!("{}: expected {}.", label, expected.class_name()));
        return None;
    }
    Some(arr)
}

fn finite_f32_arg(ctx: &mut dyn NativeContext, value: Value, label: &str) -> Option<f32> {
    match value.as_num() {
        Some(n) if n.is_finite() => Some(n as f32),
        _ => {
            ctx.runtime_error(format!("{}: expected a finite number.", label));
            None
        }
    }
}

fn i32_arg(ctx: &mut dyn NativeContext, value: Value, label: &str) -> Option<i32> {
    match value.as_num() {
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
                "{}: expected an integer in {}..={}.",
                label,
                i32::MIN,
                i32::MAX
            ));
            None
        }
    }
}

fn lane_index_arg(
    ctx: &mut dyn NativeContext,
    value: Value,
    label: &str,
    allow_negative: bool,
) -> Option<usize> {
    let raw = match value.as_num() {
        Some(n) if n.is_finite() && n.fract() == 0.0 => n as i64,
        _ => {
            ctx.runtime_error(format!("{}: index must be an integer.", label));
            return None;
        }
    };
    let lane = if allow_negative && raw < 0 { raw + 4 } else { raw };
    if !(0..4).contains(&lane) {
        ctx.runtime_error(format!("{}: index {} out of bounds.", label, raw));
        return None;
    }
    Some(lane as usize)
}

fn array_offset_arg(ctx: &mut dyn NativeContext, value: Value, label: &str) -> Option<usize> {
    match value.as_num() {
        Some(n) if n.is_finite() && n.fract() == 0.0 && n >= 0.0 => Some(n as usize),
        _ => {
            ctx.runtime_error(format!("{}: offset must be a non-negative integer.", label));
            None
        }
    }
}

fn make_simd(ctx: &mut dyn NativeContext, kind: SimdKind, lanes: [u32; 4]) -> Value {
    ctx.alloc_simd(kind, lanes)
}

fn simd_count(_ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    Value::num(4.0)
}

fn simd_byte_length(_ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    Value::num(16.0)
}

fn simd_subscript(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let simd = receiver_simd(args);
    let Some(i) = lane_index_arg(ctx, args[1], "Simd subscript", true) else {
        return Value::null();
    };
    match simd.kind_tag() {
        SimdKind::F32x4 => Value::num(simd.get_f32(i).unwrap_or(0.0) as f64),
        SimdKind::I32x4 => Value::num(simd.get_i32(i).unwrap_or(0) as f64),
    }
}

fn simd_iterate(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    if args[1].is_null() {
        return Value::num(0.0);
    }
    let Some(n) = args[1].as_num() else {
        return Value::bool(false);
    };
    let next = (n as i64) + 1;
    if !(0..4).contains(&next) {
        Value::bool(false)
    } else {
        Value::num(next as f64)
    }
}

fn simd_iterator_value(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    simd_subscript(ctx, args)
}

fn simd_replace_lane(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let simd = receiver_simd(args);
    let Some(i) = lane_index_arg(ctx, args[1], "Simd.replaceLane", true) else {
        return Value::null();
    };
    let mut lanes = simd.lanes;
    match simd.kind_tag() {
        SimdKind::F32x4 => {
            let Some(v) = finite_f32_arg(ctx, args[2], "Simd4f.replaceLane") else {
                return Value::null();
            };
            lanes[i] = v.to_bits();
        }
        SimdKind::I32x4 => {
            let Some(v) = i32_arg(ctx, args[2], "Simd4i.replaceLane") else {
                return Value::null();
            };
            lanes[i] = v as u32;
        }
    }
    make_simd(ctx, simd.kind_tag(), lanes)
}

fn simd_shuffle(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let simd = receiver_simd(args);
    let mut lanes = [0u32; 4];
    for dst in 0..4 {
        let Some(src) = lane_index_arg(ctx, args[1 + dst], "Simd.shuffle", false) else {
            return Value::null();
        };
        lanes[dst] = simd.lanes[src];
    }
    make_simd(ctx, simd.kind_tag(), lanes)
}

fn simd_to_string(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let simd = receiver_simd(args);
    let text = match simd.kind_tag() {
        SimdKind::F32x4 => format!(
            "Simd4f({}, {}, {}, {})",
            simd.get_f32(0).unwrap_or(0.0),
            simd.get_f32(1).unwrap_or(0.0),
            simd.get_f32(2).unwrap_or(0.0),
            simd.get_f32(3).unwrap_or(0.0)
        ),
        SimdKind::I32x4 => format!(
            "Simd4i({}, {}, {}, {})",
            simd.get_i32(0).unwrap_or(0),
            simd.get_i32(1).unwrap_or(0),
            simd.get_i32(2).unwrap_or(0),
            simd.get_i32(3).unwrap_or(0)
        ),
    };
    ctx.alloc_string(text)
}

fn simd4f_new(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let mut lanes = [0u32; 4];
    for i in 0..4 {
        let Some(v) = finite_f32_arg(ctx, args[1 + i], "Simd4f.new") else {
            return Value::null();
        };
        lanes[i] = v.to_bits();
    }
    make_simd(ctx, SimdKind::F32x4, lanes)
}

fn simd4i_new(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let mut lanes = [0u32; 4];
    for i in 0..4 {
        let Some(v) = i32_arg(ctx, args[1 + i], "Simd4i.new") else {
            return Value::null();
        };
        lanes[i] = v as u32;
    }
    make_simd(ctx, SimdKind::I32x4, lanes)
}

fn simd4f_splat(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(v) = finite_f32_arg(ctx, args[1], "Simd4f.splat") else {
        return Value::null();
    };
    make_simd(ctx, SimdKind::F32x4, [v.to_bits(); 4])
}

fn simd4i_splat(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(v) = i32_arg(ctx, args[1], "Simd4i.splat") else {
        return Value::null();
    };
    make_simd(ctx, SimdKind::I32x4, [v as u32; 4])
}

fn simd4f_load(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(arr) = typed_array_arg(ctx, args[1], TypedArrayKind::F32, "Simd4f.load") else {
        return Value::null();
    };
    let Some(offset) = array_offset_arg(ctx, args[2], "Simd4f.load") else {
        return Value::null();
    };
    let count = unsafe { (*arr).count as usize };
    if offset + 4 > count {
        ctx.runtime_error("Simd4f.load: array too short for 4 lanes.".to_string());
        return Value::null();
    }
    let mut lanes = [0u32; 4];
    for (lane, slot) in lanes.iter_mut().enumerate() {
        *slot = unsafe { (*arr).get_f32(offset + lane).unwrap_or(0.0) }.to_bits();
    }
    make_simd(ctx, SimdKind::F32x4, lanes)
}

fn simd4i_load(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(arr) = typed_array_arg(ctx, args[1], TypedArrayKind::I32, "Simd4i.load") else {
        return Value::null();
    };
    let Some(offset) = array_offset_arg(ctx, args[2], "Simd4i.load") else {
        return Value::null();
    };
    let count = unsafe { (*arr).count as usize };
    if offset + 4 > count {
        ctx.runtime_error("Simd4i.load: array too short for 4 lanes.".to_string());
        return Value::null();
    }
    let mut lanes = [0u32; 4];
    for (lane, slot) in lanes.iter_mut().enumerate() {
        *slot = unsafe { (*arr).get_i32(offset + lane).unwrap_or(0) as u32 };
    }
    make_simd(ctx, SimdKind::I32x4, lanes)
}

fn simd4f_store(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let simd = receiver_simd(args);
    let Some(arr) = typed_array_arg(ctx, args[1], TypedArrayKind::F32, "Simd4f.store") else {
        return Value::null();
    };
    let Some(offset) = array_offset_arg(ctx, args[2], "Simd4f.store") else {
        return Value::null();
    };
    let count = unsafe { (*arr).count as usize };
    if offset + 4 > count {
        ctx.runtime_error("Simd4f.store: array too short for 4 lanes.".to_string());
        return Value::null();
    }
    for lane in 0..4 {
        unsafe { (*arr).set_f32(offset + lane, simd.get_f32(lane).unwrap_or(0.0)) };
    }
    args[0]
}

fn simd4i_store(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let simd = receiver_simd(args);
    let Some(arr) = typed_array_arg(ctx, args[1], TypedArrayKind::I32, "Simd4i.store") else {
        return Value::null();
    };
    let Some(offset) = array_offset_arg(ctx, args[2], "Simd4i.store") else {
        return Value::null();
    };
    let count = unsafe { (*arr).count as usize };
    if offset + 4 > count {
        ctx.runtime_error("Simd4i.store: array too short for 4 lanes.".to_string());
        return Value::null();
    }
    for lane in 0..4 {
        unsafe { (*arr).set_i32(offset + lane, simd.get_i32(lane).unwrap_or(0)) };
    }
    args[0]
}

fn simd4f_to_float32_array(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let simd = receiver_simd(args);
    let result = ctx.alloc_typed_array(4, TypedArrayKind::F32);
    let arr = result.as_object().unwrap() as *mut ObjTypedArray;
    for lane in 0..4 {
        unsafe { (*arr).set_f32(lane, simd.get_f32(lane).unwrap_or(0.0)) };
    }
    result
}

fn simd4i_to_int32_array(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let simd = receiver_simd(args);
    let result = ctx.alloc_typed_array(4, TypedArrayKind::I32);
    let arr = result.as_object().unwrap() as *mut ObjTypedArray;
    for lane in 0..4 {
        unsafe { (*arr).set_i32(lane, simd.get_i32(lane).unwrap_or(0)) };
    }
    result
}

fn simd4f_reinterpret_as_int(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    make_simd(ctx, SimdKind::I32x4, receiver_simd(args).lanes)
}

fn simd4i_reinterpret_as_float(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    make_simd(ctx, SimdKind::F32x4, receiver_simd(args).lanes)
}

fn simd4f_select(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(mask) = simd_arg(ctx, args[1], SimdKind::I32x4, "Simd4f.select") else {
        return Value::null();
    };
    let Some(on_true) = simd_arg(ctx, args[2], SimdKind::F32x4, "Simd4f.select") else {
        return Value::null();
    };
    let Some(on_false) = simd_arg(ctx, args[3], SimdKind::F32x4, "Simd4f.select") else {
        return Value::null();
    };
    let mut lanes = [0u32; 4];
    for (i, slot) in lanes.iter_mut().enumerate() {
        let use_true = unsafe { (*mask).get_i32(i).unwrap_or(0) } != 0;
        *slot = if use_true {
            unsafe { (*on_true).lane_bits(i).unwrap_or(0) }
        } else {
            unsafe { (*on_false).lane_bits(i).unwrap_or(0) }
        };
    }
    make_simd(ctx, SimdKind::F32x4, lanes)
}

fn simd4i_select(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(mask) = simd_arg(ctx, args[1], SimdKind::I32x4, "Simd4i.select") else {
        return Value::null();
    };
    let Some(on_true) = simd_arg(ctx, args[2], SimdKind::I32x4, "Simd4i.select") else {
        return Value::null();
    };
    let Some(on_false) = simd_arg(ctx, args[3], SimdKind::I32x4, "Simd4i.select") else {
        return Value::null();
    };
    let mut lanes = [0u32; 4];
    for (i, slot) in lanes.iter_mut().enumerate() {
        let use_true = unsafe { (*mask).get_i32(i).unwrap_or(0) } != 0;
        *slot = if use_true {
            unsafe { (*on_true).lane_bits(i).unwrap_or(0) }
        } else {
            unsafe { (*on_false).lane_bits(i).unwrap_or(0) }
        };
    }
    make_simd(ctx, SimdKind::I32x4, lanes)
}

fn simd4f_binary(
    ctx: &mut dyn NativeContext,
    args: &[Value],
    label: &str,
    op: impl Fn(f32, f32) -> f32,
) -> Value {
    let left = receiver_simd(args);
    let Some(right) = simd_arg(ctx, args[1], SimdKind::F32x4, label) else {
        return Value::null();
    };
    let mut lanes = [0u32; 4];
    for (i, slot) in lanes.iter_mut().enumerate() {
        let a = left.get_f32(i).unwrap_or(0.0);
        let b = unsafe { (*right).get_f32(i).unwrap_or(0.0) };
        *slot = op(a, b).to_bits();
    }
    make_simd(ctx, SimdKind::F32x4, lanes)
}

fn simd4i_binary(
    ctx: &mut dyn NativeContext,
    args: &[Value],
    label: &str,
    op: impl Fn(i32, i32) -> i32,
) -> Value {
    let left = receiver_simd(args);
    let Some(right) = simd_arg(ctx, args[1], SimdKind::I32x4, label) else {
        return Value::null();
    };
    let mut lanes = [0u32; 4];
    for (i, slot) in lanes.iter_mut().enumerate() {
        let a = left.get_i32(i).unwrap_or(0);
        let b = unsafe { (*right).get_i32(i).unwrap_or(0) };
        *slot = op(a, b) as u32;
    }
    make_simd(ctx, SimdKind::I32x4, lanes)
}

fn simd4f_compare(
    ctx: &mut dyn NativeContext,
    args: &[Value],
    label: &str,
    op: impl Fn(f32, f32) -> bool,
) -> Value {
    let left = receiver_simd(args);
    let Some(right) = simd_arg(ctx, args[1], SimdKind::F32x4, label) else {
        return Value::null();
    };
    let mut lanes = [0u32; 4];
    for (i, slot) in lanes.iter_mut().enumerate() {
        let a = left.get_f32(i).unwrap_or(0.0);
        let b = unsafe { (*right).get_f32(i).unwrap_or(0.0) };
        *slot = if op(a, b) { u32::MAX } else { 0 };
    }
    make_simd(ctx, SimdKind::I32x4, lanes)
}

fn simd4i_compare(
    ctx: &mut dyn NativeContext,
    args: &[Value],
    label: &str,
    op: impl Fn(i32, i32) -> bool,
) -> Value {
    let left = receiver_simd(args);
    let Some(right) = simd_arg(ctx, args[1], SimdKind::I32x4, label) else {
        return Value::null();
    };
    let mut lanes = [0u32; 4];
    for (i, slot) in lanes.iter_mut().enumerate() {
        let a = left.get_i32(i).unwrap_or(0);
        let b = unsafe { (*right).get_i32(i).unwrap_or(0) };
        *slot = if op(a, b) { u32::MAX } else { 0 };
    }
    make_simd(ctx, SimdKind::I32x4, lanes)
}

fn simd4f_add(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    simd4f_binary(ctx, args, "Simd4f.+", |a, b| a + b)
}
fn simd4f_sub(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    simd4f_binary(ctx, args, "Simd4f.-", |a, b| a - b)
}
fn simd4f_mul(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    simd4f_binary(ctx, args, "Simd4f.*", |a, b| a * b)
}
fn simd4f_div(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    simd4f_binary(ctx, args, "Simd4f./", |a, b| a / b)
}
fn simd4f_min(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    simd4f_binary(ctx, args, "Simd4f.min", |a, b| a.min(b))
}
fn simd4f_max(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    simd4f_binary(ctx, args, "Simd4f.max", |a, b| a.max(b))
}
fn simd4f_neg(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let simd = receiver_simd(args);
    let mut lanes = [0u32; 4];
    for (i, slot) in lanes.iter_mut().enumerate() {
        *slot = (-simd.get_f32(i).unwrap_or(0.0)).to_bits();
    }
    make_simd(ctx, SimdKind::F32x4, lanes)
}
fn simd4f_abs(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let simd = receiver_simd(args);
    let mut lanes = [0u32; 4];
    for (i, slot) in lanes.iter_mut().enumerate() {
        *slot = simd.get_f32(i).unwrap_or(0.0).abs().to_bits();
    }
    make_simd(ctx, SimdKind::F32x4, lanes)
}
fn simd4f_sqrt(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let simd = receiver_simd(args);
    let mut lanes = [0u32; 4];
    for (i, slot) in lanes.iter_mut().enumerate() {
        *slot = simd.get_f32(i).unwrap_or(0.0).sqrt().to_bits();
    }
    make_simd(ctx, SimdKind::F32x4, lanes)
}
fn simd4f_eq(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    simd4f_compare(ctx, args, "Simd4f.==", |a, b| a == b)
}
fn simd4f_ne(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    simd4f_compare(ctx, args, "Simd4f.!=", |a, b| a != b)
}
fn simd4f_lt(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    simd4f_compare(ctx, args, "Simd4f.<", |a, b| a < b)
}
fn simd4f_le(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    simd4f_compare(ctx, args, "Simd4f.<=", |a, b| a <= b)
}
fn simd4f_gt(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    simd4f_compare(ctx, args, "Simd4f.>", |a, b| a > b)
}
fn simd4f_ge(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    simd4f_compare(ctx, args, "Simd4f.>=", |a, b| a >= b)
}

fn simd4i_add(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    simd4i_binary(ctx, args, "Simd4i.+", i32::wrapping_add)
}
fn simd4i_sub(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    simd4i_binary(ctx, args, "Simd4i.-", i32::wrapping_sub)
}
fn simd4i_mul(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    simd4i_binary(ctx, args, "Simd4i.*", i32::wrapping_mul)
}
fn simd4i_and(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    simd4i_binary(ctx, args, "Simd4i.&", |a, b| a & b)
}
fn simd4i_or(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    simd4i_binary(ctx, args, "Simd4i.|", |a, b| a | b)
}
fn simd4i_xor(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    simd4i_binary(ctx, args, "Simd4i.^", |a, b| a ^ b)
}
fn simd4i_min(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    simd4i_binary(ctx, args, "Simd4i.min", |a, b| a.min(b))
}
fn simd4i_max(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    simd4i_binary(ctx, args, "Simd4i.max", |a, b| a.max(b))
}
fn simd4i_neg(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let simd = receiver_simd(args);
    let mut lanes = [0u32; 4];
    for (i, slot) in lanes.iter_mut().enumerate() {
        *slot = simd.get_i32(i).unwrap_or(0).wrapping_neg() as u32;
    }
    make_simd(ctx, SimdKind::I32x4, lanes)
}
fn simd4i_not(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let simd = receiver_simd(args);
    let mut lanes = [0u32; 4];
    for (i, slot) in lanes.iter_mut().enumerate() {
        *slot = !simd.lane_bits(i).unwrap_or(0);
    }
    make_simd(ctx, SimdKind::I32x4, lanes)
}
fn simd4i_abs(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let simd = receiver_simd(args);
    let mut lanes = [0u32; 4];
    for (i, slot) in lanes.iter_mut().enumerate() {
        *slot = simd.get_i32(i).unwrap_or(0).wrapping_abs() as u32;
    }
    make_simd(ctx, SimdKind::I32x4, lanes)
}
fn simd4i_shl(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let simd = receiver_simd(args);
    let Some(shift) = i32_arg(ctx, args[1], "Simd4i.<<") else {
        return Value::null();
    };
    let shift = (shift as u32) & 31;
    let mut lanes = [0u32; 4];
    for (i, slot) in lanes.iter_mut().enumerate() {
        *slot = simd.get_i32(i).unwrap_or(0).wrapping_shl(shift) as u32;
    }
    make_simd(ctx, SimdKind::I32x4, lanes)
}
fn simd4i_shr(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let simd = receiver_simd(args);
    let Some(shift) = i32_arg(ctx, args[1], "Simd4i.>>") else {
        return Value::null();
    };
    let shift = (shift as u32) & 31;
    let mut lanes = [0u32; 4];
    for (i, slot) in lanes.iter_mut().enumerate() {
        *slot = (simd.get_i32(i).unwrap_or(0) >> shift) as u32;
    }
    make_simd(ctx, SimdKind::I32x4, lanes)
}
fn simd4i_eq(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    simd4i_compare(ctx, args, "Simd4i.==", |a, b| a == b)
}
fn simd4i_ne(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    simd4i_compare(ctx, args, "Simd4i.!=", |a, b| a != b)
}
fn simd4i_lt(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    simd4i_compare(ctx, args, "Simd4i.<", |a, b| a < b)
}
fn simd4i_le(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    simd4i_compare(ctx, args, "Simd4i.<=", |a, b| a <= b)
}
fn simd4i_gt(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    simd4i_compare(ctx, args, "Simd4i.>", |a, b| a > b)
}
fn simd4i_ge(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    simd4i_compare(ctx, args, "Simd4i.>=", |a, b| a >= b)
}

fn simd4i_all_true(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let simd = receiver_simd(args);
    Value::bool((0..4).all(|i| simd.get_i32(i).unwrap_or(0) != 0))
}

fn simd4i_any_true(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let simd = receiver_simd(args);
    Value::bool((0..4).any(|i| simd.get_i32(i).unwrap_or(0) != 0))
}

fn simd4i_bitmask(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let simd = receiver_simd(args);
    let mut mask = 0u32;
    for lane in 0..4 {
        let bits = simd.lane_bits(lane).unwrap_or(0);
        if (bits & 0x8000_0000) != 0 {
            mask |= 1 << lane;
        }
    }
    Value::num(mask as f64)
}

pub fn bind(vm: &mut VM) {
    let simd = vm.simd_class;
    vm.primitive(simd, "count", simd_count);
    vm.primitive(simd, "byteLength", simd_byte_length);
    vm.primitive(simd, "[_]", simd_subscript);
    vm.primitive(simd, "iterate(_)", simd_iterate);
    vm.primitive(simd, "iteratorValue(_)", simd_iterator_value);
    let replace_lane_sig = "replaceLane(_,_)";
    vm.primitive(simd, replace_lane_sig, simd_replace_lane);
    vm.primitive(simd, "shuffle(_,_,_,_)", simd_shuffle);
    vm.primitive(simd, "toString", simd_to_string);

    let f = vm.simd4f_class;
    vm.primitive_static(f, "new(_,_,_,_)", simd4f_new);
    vm.primitive_static(f, "splat(_)", simd4f_splat);
    vm.primitive_static(f, "load(_,_)", simd4f_load);
    vm.primitive_static(f, "select(_,_,_)", simd4f_select);
    vm.primitive(f, "store(_,_)", simd4f_store);
    vm.primitive(f, "toFloat32Array", simd4f_to_float32_array);
    vm.primitive(f, "reinterpretAsInt", simd4f_reinterpret_as_int);
    vm.primitive(f, "+(_)", simd4f_add);
    vm.primitive(f, "-(_)", simd4f_sub);
    vm.primitive(f, "*(_)", simd4f_mul);
    vm.primitive(f, "/(_)", simd4f_div);
    vm.primitive(f, "-", simd4f_neg);
    vm.primitive(f, "min(_)", simd4f_min);
    vm.primitive(f, "max(_)", simd4f_max);
    vm.primitive(f, "abs", simd4f_abs);
    vm.primitive(f, "sqrt", simd4f_sqrt);
    vm.primitive(f, "==(_)", simd4f_eq);
    vm.primitive(f, "!=(_)", simd4f_ne);
    vm.primitive(f, "<(_)", simd4f_lt);
    vm.primitive(f, "<=(_)", simd4f_le);
    vm.primitive(f, ">(_)", simd4f_gt);
    vm.primitive(f, ">=(_)", simd4f_ge);

    let i = vm.simd4i_class;
    vm.primitive_static(i, "new(_,_,_,_)", simd4i_new);
    vm.primitive_static(i, "splat(_)", simd4i_splat);
    vm.primitive_static(i, "load(_,_)", simd4i_load);
    vm.primitive_static(i, "select(_,_,_)", simd4i_select);
    vm.primitive(i, "store(_,_)", simd4i_store);
    vm.primitive(i, "toInt32Array", simd4i_to_int32_array);
    vm.primitive(i, "reinterpretAsFloat", simd4i_reinterpret_as_float);
    vm.primitive(i, "+(_)", simd4i_add);
    vm.primitive(i, "-(_)", simd4i_sub);
    vm.primitive(i, "*(_)", simd4i_mul);
    vm.primitive(i, "-", simd4i_neg);
    vm.primitive(i, "&(_)", simd4i_and);
    vm.primitive(i, "|(_)", simd4i_or);
    vm.primitive(i, "^(_)", simd4i_xor);
    vm.primitive(i, "~", simd4i_not);
    vm.primitive(i, "<<(_)", simd4i_shl);
    vm.primitive(i, ">>(_)", simd4i_shr);
    vm.primitive(i, "min(_)", simd4i_min);
    vm.primitive(i, "max(_)", simd4i_max);
    vm.primitive(i, "abs", simd4i_abs);
    vm.primitive(i, "allTrue", simd4i_all_true);
    vm.primitive(i, "anyTrue", simd4i_any_true);
    vm.primitive(i, "bitmask", simd4i_bitmask);
    vm.primitive(i, "==(_)", simd4i_eq);
    vm.primitive(i, "!=(_)", simd4i_ne);
    vm.primitive(i, "<(_)", simd4i_lt);
    vm.primitive(i, "<=(_)", simd4i_le);
    vm.primitive(i, ">(_)", simd4i_gt);
    vm.primitive(i, ">=(_)", simd4i_ge);
}
