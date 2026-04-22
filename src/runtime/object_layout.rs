/// Compile-time offset constants for `#[repr(C)]` object structs.
///
/// These constants are used by the JIT codegen for GEP-style inline memory
/// access. They MUST match the actual Rust struct layouts — the static
/// assertions at the bottom verify this.
// Object/Value types used in test assertions only.
#[cfg(test)]
use {super::object::*, super::value::Value};

// -- ObjHeader (24 bytes on 64-bit) -----------------------------------------

pub const HEADER_OBJ_TYPE: i32 = 0; // u8
pub const HEADER_GC_MARK: i32 = 1; // u8
pub const HEADER_GENERATION: i32 = 2; // u8
                                      // 5 bytes padding
pub const HEADER_NEXT: i32 = 8; // *mut ObjHeader
pub const HEADER_CLASS: i32 = 16; // *mut ObjClass
pub const HEADER_SIZE: i32 = 24;

// -- ObjInstance (40 bytes) --------------------------------------------------

pub const INSTANCE_NUM_FIELDS: i32 = 24; // u32
                                         // 4 bytes padding
pub const INSTANCE_FIELDS: i32 = 32; // *mut Value
pub const INSTANCE_SIZE: i32 = 40;

// -- ObjList (40 bytes) -----------------------------------------------------

pub const LIST_COUNT: i32 = 24; // u32
pub const LIST_CAPACITY: i32 = 28; // u32
pub const LIST_ELEMENTS: i32 = 32; // *mut Value
pub const LIST_SIZE: i32 = 40;

// -- ObjTypedArray (40 bytes) -----------------------------------------------
//
// Shared backing storage for ByteArray / Float32Array / Float64Array.
// The `kind` byte (0=U8, 1=F32, 2=F64) drives element size + load/store
// width.

pub const TYPED_ARRAY_COUNT: i32 = 24; // u32 — element count
pub const TYPED_ARRAY_KIND: i32 = 28; // u8 — TypedArrayKind tag
pub const TYPED_ARRAY_DATA: i32 = 32; // *mut u8 — raw backing buffer
pub const TYPED_ARRAY_SIZE: i32 = 40;

// ObjType discriminant for TypedArray. Must match the
// `ObjType::TypedArray` variant position (13th, 0-indexed = 12).
pub const OBJ_TYPE_TYPED_ARRAY: u8 = 12;

// TypedArrayKind tag values. Must match the repr(u8) enum in
// `runtime::object::TypedArrayKind`.
pub const TA_KIND_U8: u8 = 0;
pub const TA_KIND_F32: u8 = 1;
pub const TA_KIND_F64: u8 = 2;

// -- Value size --------------------------------------------------------------

pub const VALUE_SIZE: i32 = 8; // NaN-boxed u64

// -- Static assertions -------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_header_layout() {
        assert_eq!(std::mem::size_of::<ObjHeader>(), HEADER_SIZE as usize);
        assert_eq!(memoffset_of!(ObjHeader, obj_type), HEADER_OBJ_TYPE as usize);
        assert_eq!(memoffset_of!(ObjHeader, gc_mark), HEADER_GC_MARK as usize);
        assert_eq!(
            memoffset_of!(ObjHeader, generation),
            HEADER_GENERATION as usize
        );
        assert_eq!(memoffset_of!(ObjHeader, next), HEADER_NEXT as usize);
        assert_eq!(memoffset_of!(ObjHeader, class), HEADER_CLASS as usize);
    }

    #[test]
    fn verify_instance_layout() {
        assert_eq!(std::mem::size_of::<ObjInstance>(), INSTANCE_SIZE as usize);
        assert_eq!(
            memoffset_of!(ObjInstance, num_fields),
            INSTANCE_NUM_FIELDS as usize
        );
        assert_eq!(memoffset_of!(ObjInstance, fields), INSTANCE_FIELDS as usize);
    }

    #[test]
    fn verify_list_layout() {
        assert_eq!(std::mem::size_of::<ObjList>(), LIST_SIZE as usize);
        assert_eq!(memoffset_of!(ObjList, count), LIST_COUNT as usize);
        assert_eq!(memoffset_of!(ObjList, capacity), LIST_CAPACITY as usize);
        assert_eq!(memoffset_of!(ObjList, elements), LIST_ELEMENTS as usize);
    }

    #[test]
    fn verify_typed_array_layout() {
        assert_eq!(
            std::mem::size_of::<ObjTypedArray>(),
            TYPED_ARRAY_SIZE as usize
        );
        assert_eq!(
            memoffset_of!(ObjTypedArray, count),
            TYPED_ARRAY_COUNT as usize
        );
        assert_eq!(
            memoffset_of!(ObjTypedArray, kind),
            TYPED_ARRAY_KIND as usize
        );
        assert_eq!(
            memoffset_of!(ObjTypedArray, data),
            TYPED_ARRAY_DATA as usize
        );
        // Discriminant + kind tag values: the JIT codegen encodes
        // these as immediates, so a bump of the enum would break
        // the emitted machine code.
        assert_eq!(ObjType::TypedArray as u8, OBJ_TYPE_TYPED_ARRAY);
        assert_eq!(TypedArrayKind::U8 as u8, TA_KIND_U8);
        assert_eq!(TypedArrayKind::F32 as u8, TA_KIND_F32);
        assert_eq!(TypedArrayKind::F64 as u8, TA_KIND_F64);
    }

    #[test]
    fn verify_value_size() {
        assert_eq!(std::mem::size_of::<Value>(), VALUE_SIZE as usize);
    }

    /// Compute the offset of a field within a struct.
    macro_rules! memoffset_of {
        ($ty:ty, $field:ident) => {{
            let uninit = std::mem::MaybeUninit::<$ty>::uninit();
            let base = uninit.as_ptr();
            let field_ptr = unsafe { std::ptr::addr_of!((*base).$field) };
            (field_ptr as usize) - (base as usize)
        }};
    }
    use memoffset_of;
}
