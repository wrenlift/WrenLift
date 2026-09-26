//! Byte offsets of the object fields compiled code reads inline.
//!
//! One [`Layout`] per pointer width: the JIT reads the host's through
//! the constants below, AOT picks one from the target triple. Lowering,
//! constants and the bootstrap read a table, never `offset_of!`;
//! [`Layout::measured`] is what the tables are checked against, natively
//! by the tests here and for wasm32 under wasmtime.

use super::object::*;
use crate::capi::WliftAotMethodDesc;
use crate::codegen::runtime_fns::JitContext;

macro_rules! layout {
    ($($field:ident),* $(,)?) => {
        /// Field offsets and object sizes, in bytes, for one pointer width.
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub struct Layout {
            $(pub $field: i32,)*
        }

        impl Layout {
            /// Every entry by name, in declaration order.
            pub fn entries(&self) -> Vec<(&'static str, i32)> {
                vec![$((stringify!($field), self.$field)),*]
            }

            /// A layout from `entries` values in declaration order.
            pub fn from_values(values: &[i32]) -> Option<Layout> {
                let mut it = values.iter().copied();
                let layout = Layout { $($field: it.next()?,)* };
                it.next().is_none().then_some(layout)
            }
        }
    };
}

layout! {
    ptr_size,
    header_obj_type,
    header_gc_mark,
    header_flags,
    header_class,
    header_size,
    class_field_kinds,
    class_flags,
    class_num_fields,
    instance_num_fields,
    instance_fields,
    instance_size,
    list_count,
    list_capacity,
    list_elements,
    list_elem_class,
    list_size,
    typed_array_count,
    typed_array_kind,
    typed_array_data,
    typed_array_size,
    simd_kind,
    simd_lanes,
    simd_size,
    closure_function,
    closure_upvalues_data,
    upvalue_location,
    upvalue_closed,
    jit_ctx_module_vars,
    jit_ctx_vm,
    jit_ctx_current_func_id,
    jit_ctx_closure,
    jit_ctx_defining_class,
    jit_ctx_jit_code_base,
    method_desc_sig_len,
    method_desc_fn_ptr,
    method_desc_arity,
    method_desc_flags,
    method_desc_size,
}

impl Layout {
    /// 64-bit pointers: every native host.
    pub const LP64: Layout = Layout {
        ptr_size: 8,
        header_obj_type: 0,
        header_gc_mark: 1,
        header_flags: 2,
        header_class: 8,
        header_size: 16,
        class_field_kinds: 16,
        class_flags: 24,
        class_num_fields: 64,
        instance_num_fields: 16,
        instance_fields: 24,
        instance_size: 32,
        list_count: 16,
        list_capacity: 20,
        list_elements: 24,
        list_elem_class: 32,
        list_size: 40,
        typed_array_count: 16,
        typed_array_kind: 20,
        typed_array_data: 24,
        typed_array_size: 32,
        simd_kind: 16,
        simd_lanes: 24,
        simd_size: 40,
        closure_function: 16,
        closure_upvalues_data: 32,
        upvalue_location: 16,
        upvalue_closed: 24,
        jit_ctx_module_vars: 0,
        jit_ctx_vm: 16,
        jit_ctx_current_func_id: 40,
        jit_ctx_closure: 48,
        jit_ctx_defining_class: 56,
        jit_ctx_jit_code_base: 64,
        method_desc_sig_len: 8,
        method_desc_fn_ptr: 16,
        method_desc_arity: 24,
        method_desc_flags: 25,
        method_desc_size: 32,
    };

    /// 32-bit pointers: wasm32.
    pub const ILP32: Layout = Layout {
        ptr_size: 4,
        header_obj_type: 0,
        header_gc_mark: 1,
        header_flags: 2,
        header_class: 4,
        header_size: 8,
        class_field_kinds: 8,
        class_flags: 12,
        class_num_fields: 36,
        instance_num_fields: 8,
        instance_fields: 16,
        instance_size: 24,
        list_count: 8,
        list_capacity: 12,
        list_elements: 16,
        list_elem_class: 20,
        list_size: 24,
        typed_array_count: 8,
        typed_array_kind: 12,
        typed_array_data: 16,
        typed_array_size: 20,
        simd_kind: 8,
        simd_lanes: 16,
        simd_size: 32,
        closure_function: 8,
        closure_upvalues_data: 16,
        upvalue_location: 8,
        upvalue_closed: 16,
        jit_ctx_module_vars: 0,
        jit_ctx_vm: 8,
        jit_ctx_current_func_id: 24,
        jit_ctx_closure: 32,
        jit_ctx_defining_class: 36,
        jit_ctx_jit_code_base: 40,
        method_desc_sig_len: 4,
        method_desc_fn_ptr: 8,
        method_desc_arity: 12,
        method_desc_flags: 13,
        method_desc_size: 16,
    };

    /// The layout of this build.
    pub const HOST: Layout = if cfg!(target_pointer_width = "64") {
        Layout::LP64
    } else {
        Layout::ILP32
    };

    /// The layout for a target triple.
    pub fn for_triple(triple: &str) -> Layout {
        Layout::for_pointer_bytes(if triple.starts_with("wasm32") { 4 } else { 8 })
    }

    /// The layout for a pointer width in bytes.
    pub fn for_pointer_bytes(bytes: u32) -> Layout {
        if bytes == 4 {
            Layout::ILP32
        } else {
            Layout::LP64
        }
    }

    /// The layout this build actually has, for checking the tables.
    pub fn measured() -> Layout {
        use std::mem::{offset_of, size_of};
        let at = |n: usize| n as i32;
        // The Vec's data pointer, found by value: a one-element Vec's
        // capacity and length are 1, never its buffer's address.
        let closure = ObjClosure::new(std::ptr::null_mut(), 1);
        let vec_at = offset_of!(ObjClosure, upvalues);
        let words = size_of::<Vec<*mut ObjUpvalue>>() / size_of::<usize>();
        let data = (0..words)
            .map(|i| vec_at + i * size_of::<usize>())
            .find(|&off| {
                let word = unsafe {
                    ((&closure as *const ObjClosure as *const u8).add(off) as *const usize)
                        .read_unaligned()
                };
                word == closure.upvalues.as_ptr() as usize
            })
            .expect("a Vec holds its data pointer");
        Layout {
            ptr_size: at(size_of::<usize>()),
            header_obj_type: at(offset_of!(ObjHeader, obj_type)),
            header_gc_mark: at(offset_of!(ObjHeader, gc_mark)),
            header_flags: at(offset_of!(ObjHeader, flags)),
            header_class: at(offset_of!(ObjHeader, class)),
            header_size: at(size_of::<ObjHeader>()),
            class_field_kinds: at(offset_of!(ObjClass, field_kinds_ptr)),
            class_flags: at(offset_of!(ObjClass, flags)),
            class_num_fields: at(offset_of!(ObjClass, num_fields)),
            instance_num_fields: at(offset_of!(ObjInstance, num_fields)),
            instance_fields: at(offset_of!(ObjInstance, fields)),
            instance_size: at(size_of::<ObjInstance>()),
            list_count: at(offset_of!(ObjList, count)),
            list_capacity: at(offset_of!(ObjList, capacity)),
            list_elements: at(offset_of!(ObjList, elements)),
            list_elem_class: at(offset_of!(ObjList, elem_class)),
            list_size: at(size_of::<ObjList>()),
            typed_array_count: at(offset_of!(ObjTypedArray, count)),
            typed_array_kind: at(offset_of!(ObjTypedArray, kind)),
            typed_array_data: at(offset_of!(ObjTypedArray, data)),
            typed_array_size: at(size_of::<ObjTypedArray>()),
            simd_kind: at(offset_of!(ObjSimd, kind)),
            simd_lanes: at(offset_of!(ObjSimd, lanes)),
            simd_size: at(size_of::<ObjSimd>()),
            closure_function: at(offset_of!(ObjClosure, function)),
            closure_upvalues_data: at(data),
            upvalue_location: at(offset_of!(ObjUpvalue, location)),
            upvalue_closed: at(offset_of!(ObjUpvalue, closed)),
            jit_ctx_module_vars: at(offset_of!(JitContext, module_vars)),
            jit_ctx_vm: at(offset_of!(JitContext, vm)),
            jit_ctx_current_func_id: at(offset_of!(JitContext, current_func_id)),
            jit_ctx_closure: at(offset_of!(JitContext, closure)),
            jit_ctx_defining_class: at(offset_of!(JitContext, defining_class)),
            jit_ctx_jit_code_base: at(offset_of!(JitContext, jit_code_base)),
            method_desc_sig_len: at(offset_of!(WliftAotMethodDesc, sig_len)),
            method_desc_fn_ptr: at(offset_of!(WliftAotMethodDesc, fn_ptr)),
            method_desc_arity: at(offset_of!(WliftAotMethodDesc, arity)),
            method_desc_flags: at(offset_of!(WliftAotMethodDesc, flags)),
            method_desc_size: at(size_of::<WliftAotMethodDesc>()),
        }
    }
}

/// Write [`Layout::measured`] into `out` in declaration order and return
/// the entry count, so a test can read wasm32's layout from a module.
///
/// # Safety
/// `out` must be writable for `cap` i32s.
#[cfg(all(target_arch = "wasm32", feature = "aot_runtime"))]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn wlift_layout_probe(out: *mut i32, cap: usize) -> usize {
    let entries = Layout::measured().entries();
    for (i, (_, v)) in entries.iter().enumerate().take(cap) {
        unsafe { *out.add(i) = *v };
    }
    entries.len()
}

// -- ObjHeader -----------------------------------------------------------------

pub const HEADER_OBJ_TYPE: i32 = Layout::HOST.header_obj_type; // u8
pub const HEADER_GC_MARK: i32 = Layout::HOST.header_gc_mark; // u8
pub const HEADER_FLAGS: i32 = Layout::HOST.header_flags; // u8
pub const HEADER_CLASS: i32 = Layout::HOST.header_class; // *mut ObjClass
pub const HEADER_SIZE: i32 = Layout::HOST.header_size;

// -- ObjClass ------------------------------------------------------------------

/// `*mut u8`: one `FIELD_*` byte per instance field, or null.
pub const CLASS_FIELD_KINDS: i32 = Layout::HOST.class_field_kinds;
/// u8 of `CLASS_FLAG_*` bits.
pub const CLASS_FLAGS: i32 = Layout::HOST.class_flags;
/// u16: instance field count.
pub const CLASS_NUM_FIELDS: i32 = Layout::HOST.class_num_fields;
/// `ObjType::Instance` as the header's type byte.
pub const OBJ_TYPE_INSTANCE: u8 = 9;
/// The class or an ancestor other than Object defines `==` or `!=`, so
/// equality on its instances is not identity.
pub const CLASS_FLAG_EQ: u8 = 1;

// -- ObjInstance ---------------------------------------------------------------

pub const INSTANCE_NUM_FIELDS: i32 = Layout::HOST.instance_num_fields; // u32
pub const INSTANCE_FIELDS: i32 = Layout::HOST.instance_fields; // *mut Value
pub const INSTANCE_SIZE: i32 = Layout::HOST.instance_size;

// -- ObjList -------------------------------------------------------------------

pub const LIST_COUNT: i32 = Layout::HOST.list_count; // u32
pub const LIST_CAPACITY: i32 = Layout::HOST.list_capacity; // u32
pub const LIST_ELEMENTS: i32 = Layout::HOST.list_elements; // *mut Value
pub const LIST_ELEM_CLASS: i32 = Layout::HOST.list_elem_class; // usize
pub const LIST_SIZE: i32 = Layout::HOST.list_size;

// -- ObjTypedArray -------------------------------------------------------------
//
// Shared backing storage for ByteArray / Int32Array / Float32Array / Float64Array.
// The `kind` byte (0=U8, 1=F32, 2=F64, 3=I32) drives element size + load/store
// width.

pub const TYPED_ARRAY_COUNT: i32 = Layout::HOST.typed_array_count; // u32 — element count
pub const TYPED_ARRAY_KIND: i32 = Layout::HOST.typed_array_kind; // u8 — TypedArrayKind tag
pub const TYPED_ARRAY_DATA: i32 = Layout::HOST.typed_array_data; // *mut u8 — raw backing buffer
pub const TYPED_ARRAY_SIZE: i32 = Layout::HOST.typed_array_size;

// ObjType discriminant for TypedArray. Must match the
// `ObjType::TypedArray` variant position (13th, 0-indexed = 12).
pub const OBJ_TYPE_TYPED_ARRAY: u8 = 12;

// TypedArrayKind tag values. Must match the repr(u8) enum in
// `runtime::object::TypedArrayKind`.
pub const TA_KIND_U8: u8 = 0;
pub const TA_KIND_F32: u8 = 1;
pub const TA_KIND_F64: u8 = 2;
pub const TA_KIND_I32: u8 = 3;

// -- ObjSimd -------------------------------------------------------------------

pub const SIMD_KIND: i32 = Layout::HOST.simd_kind; // u8 — SimdKind tag
pub const SIMD_LANES: i32 = Layout::HOST.simd_lanes; // [u32; 4] raw lane payload
pub const SIMD_SIZE: i32 = Layout::HOST.simd_size;

// ObjType discriminant for ObjSimd.
pub const OBJ_TYPE_SIMD: u8 = 13;

// SimdKind tag values. Must match `runtime::object::SimdKind`.
pub const SIMD_KIND_F32X4: u8 = 0;
pub const SIMD_KIND_I32X4: u8 = 1;

// -- ObjClosure ----------------------------------------------------------------
//
// `upvalues` is a `Vec<*mut ObjUpvalue>`; CLOSURE_UPVALUES_DATA is where
// its data pointer lives, not where the Vec starts. Rust does not fix a
// Vec's field order, so `Layout::measured` finds the pointer by value.

pub const CLOSURE_FUNCTION: i32 = Layout::HOST.closure_function; // *mut ObjFn
pub const CLOSURE_UPVALUES_DATA: i32 = Layout::HOST.closure_upvalues_data; // *mut *mut ObjUpvalue

// -- ObjUpvalue ----------------------------------------------------------------

pub const UPVALUE_LOCATION: i32 = Layout::HOST.upvalue_location; // *mut Value (open) or &closed
pub const UPVALUE_CLOSED: i32 = Layout::HOST.upvalue_closed; // Value (closed-over storage)

// -- JitContext ----------------------------------------------------------------
//
// Layout in `runtime_fns::JitContext`. Used by AOT codegen to
// inline upvalue / static-field access without a helper hop —
// load through the TLS context pointer + the right offset.

pub const JIT_CTX_MODULE_VARS: i32 = Layout::HOST.jit_ctx_module_vars;
pub const JIT_CTX_VM: i32 = Layout::HOST.jit_ctx_vm;
pub const JIT_CTX_CURRENT_FUNC_ID: i32 = Layout::HOST.jit_ctx_current_func_id;
pub const JIT_CTX_CLOSURE: i32 = Layout::HOST.jit_ctx_closure;
pub const JIT_CTX_DEFINING_CLASS: i32 = Layout::HOST.jit_ctx_defining_class;
pub const JIT_CTX_JIT_CODE_BASE: i32 = Layout::HOST.jit_ctx_jit_code_base;

// -- Value size ----------------------------------------------------------------

pub const VALUE_SIZE: i32 = 8; // NaN-boxed u64

/// The entries where `table` differs from `actual`, one per line.
pub fn layout_mismatches(table: &Layout, actual: &Layout) -> String {
    table
        .entries()
        .into_iter()
        .zip(actual.entries())
        .filter(|((_, t), (_, a))| t != a)
        .map(|((name, t), (_, a))| format!("{name}: table {t}, actual {a}\n"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::value::Value;

    #[test]
    fn the_host_table_is_the_layout_this_build_has() {
        let actual = Layout::measured();
        assert!(
            Layout::HOST == actual,
            "object_layout's table is stale:\n{}",
            layout_mismatches(&Layout::HOST, &actual)
        );
    }

    #[test]
    fn a_triple_picks_its_pointer_width() {
        assert_eq!(Layout::for_triple("wasm32-wasip1"), Layout::ILP32);
        assert_eq!(Layout::for_triple("x86_64-unknown-linux-gnu"), Layout::LP64);
        assert_eq!(Layout::ILP32.ptr_size, 4);
        assert_eq!(Layout::LP64.ptr_size, 8);
    }

    #[test]
    fn values_round_trip_in_declaration_order() {
        let values: Vec<i32> = Layout::LP64.entries().iter().map(|e| e.1).collect();
        assert_eq!(Layout::from_values(&values), Some(Layout::LP64));
        assert_eq!(Layout::from_values(&values[1..]), None);
    }

    /// Compiled code encodes these as immediates.
    #[test]
    fn tags_match_their_enums() {
        assert_eq!(ObjType::Instance as u8, OBJ_TYPE_INSTANCE);
        assert_eq!(ObjType::TypedArray as u8, OBJ_TYPE_TYPED_ARRAY);
        assert_eq!(TypedArrayKind::U8 as u8, TA_KIND_U8);
        assert_eq!(TypedArrayKind::F32 as u8, TA_KIND_F32);
        assert_eq!(TypedArrayKind::F64 as u8, TA_KIND_F64);
        assert_eq!(TypedArrayKind::I32 as u8, TA_KIND_I32);
        assert_eq!(ObjType::Simd as u8, OBJ_TYPE_SIMD);
        assert_eq!(SimdKind::F32x4 as u8, SIMD_KIND_F32X4);
        assert_eq!(SimdKind::I32x4 as u8, SIMD_KIND_I32X4);
        assert_eq!(std::mem::size_of::<Value>(), VALUE_SIZE as usize);
    }
}
