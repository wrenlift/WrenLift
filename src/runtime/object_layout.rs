/// Compile-time offset constants for `#[repr(C)]` object structs.
///
/// These constants are used by the JIT codegen for GEP-style inline memory
/// access. They MUST match the actual Rust struct layouts — the static
/// assertions at the bottom verify this.

// Object/Value types used in test assertions only.
#[cfg(test)]
use {super::object::*, super::value::Value};

// -- ObjHeader (24 bytes on 64-bit) -----------------------------------------

pub const HEADER_OBJ_TYPE: i32 = 0;   // u8
pub const HEADER_GC_MARK: i32 = 1;    // u8
pub const HEADER_GENERATION: i32 = 2; // u8
// 5 bytes padding
pub const HEADER_NEXT: i32 = 8;       // *mut ObjHeader
pub const HEADER_CLASS: i32 = 16;     // *mut ObjClass
pub const HEADER_SIZE: i32 = 24;

// -- ObjInstance (40 bytes) --------------------------------------------------

pub const INSTANCE_NUM_FIELDS: i32 = 24; // u32
// 4 bytes padding
pub const INSTANCE_FIELDS: i32 = 32;    // *mut Value
pub const INSTANCE_SIZE: i32 = 40;

// -- ObjList (40 bytes) -----------------------------------------------------

pub const LIST_COUNT: i32 = 24;       // u32
pub const LIST_CAPACITY: i32 = 28;    // u32
pub const LIST_ELEMENTS: i32 = 32;    // *mut Value
pub const LIST_SIZE: i32 = 40;

// -- Value size --------------------------------------------------------------

pub const VALUE_SIZE: i32 = 8;        // NaN-boxed u64

// -- Static assertions -------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_header_layout() {
        assert_eq!(std::mem::size_of::<ObjHeader>(), HEADER_SIZE as usize);
        assert_eq!(
            memoffset_of!(ObjHeader, obj_type),
            HEADER_OBJ_TYPE as usize
        );
        assert_eq!(
            memoffset_of!(ObjHeader, gc_mark),
            HEADER_GC_MARK as usize
        );
        assert_eq!(
            memoffset_of!(ObjHeader, generation),
            HEADER_GENERATION as usize
        );
        assert_eq!(
            memoffset_of!(ObjHeader, next),
            HEADER_NEXT as usize
        );
        assert_eq!(
            memoffset_of!(ObjHeader, class),
            HEADER_CLASS as usize
        );
    }

    #[test]
    fn verify_instance_layout() {
        assert_eq!(std::mem::size_of::<ObjInstance>(), INSTANCE_SIZE as usize);
        assert_eq!(
            memoffset_of!(ObjInstance, num_fields),
            INSTANCE_NUM_FIELDS as usize
        );
        assert_eq!(
            memoffset_of!(ObjInstance, fields),
            INSTANCE_FIELDS as usize
        );
    }

    #[test]
    fn verify_list_layout() {
        assert_eq!(std::mem::size_of::<ObjList>(), LIST_SIZE as usize);
        assert_eq!(
            memoffset_of!(ObjList, count),
            LIST_COUNT as usize
        );
        assert_eq!(
            memoffset_of!(ObjList, capacity),
            LIST_CAPACITY as usize
        );
        assert_eq!(
            memoffset_of!(ObjList, elements),
            LIST_ELEMENTS as usize
        );
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
