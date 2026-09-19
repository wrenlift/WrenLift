//! What the collector shares with the rest of the runtime: the object
//! graph walk, the plausibility check on a pointer read from memory
//! the collector does not own, and the counters.
use super::object::*;
use super::value::Value;

#[derive(Clone, Copy, Debug, Default)]
pub struct GcStats {
    pub collections: u32,
    pub total_allocated: usize,
    pub total_freed: usize,
    pub objects_allocated: usize,
    pub objects_freed: usize,
    pub peak_objects: usize,
    /// Total time spent in GC (nanoseconds).
    pub gc_time_ns: u64,
}

// A header pointer read from a slot is followed only when it lies in
// the user-space range a heap object can occupy: not the first pages,
// which no allocation maps, and not the NaN-box tag space, which is
// where a Value mistaken for a pointer lands. A slot that fails the
// check is skipped rather than followed.
const MIN_VALID_HEAP_ADDR: usize = 0x10000;
// wasm32 addresses fill the type, so the cap is only for 64-bit hosts.
#[cfg(not(target_pointer_width = "32"))]
const MAX_VALID_HEAP_ADDR: usize = 1 << 52;
#[cfg(target_pointer_width = "32")]
const MAX_VALID_HEAP_ADDR: usize = usize::MAX;

pub(super) fn is_valid_obj_ptr(header: *mut ObjHeader) -> bool {
    let addr = header as usize;
    (MIN_VALID_HEAP_ADDR..MAX_VALID_HEAP_ADDR).contains(&addr)
}

/// The object `val` refers to, if it is one and its address is plausible.
#[inline(always)]
pub(super) fn object_of(val: Value) -> Option<*mut ObjHeader> {
    let header = val.as_object()? as *mut ObjHeader;
    is_valid_obj_ptr(header).then_some(header)
}

#[inline(always)]
fn child_ptr<F: FnMut(*mut ObjHeader)>(ptr: *mut ObjHeader, f: &mut F) {
    if !ptr.is_null() && is_valid_obj_ptr(ptr) {
        f(ptr);
    }
}

#[inline(always)]
fn child_value<F: FnMut(*mut ObjHeader)>(val: Value, f: &mut F) {
    if let Some(header) = object_of(val) {
        f(header);
    }
}

/// Call `f` with every object `header` refers to: its class, then its
/// fields' objects. Null and implausible pointers are skipped, so a
/// corrupt slot costs a leaked reference rather than a fault.
pub(super) unsafe fn for_each_child<F: FnMut(*mut ObjHeader)>(header: *mut ObjHeader, f: &mut F) {
    unsafe {
        if !is_valid_obj_ptr(header) {
            return;
        }
        child_ptr((*header).class as *mut ObjHeader, f);

        match (*header).obj_type {
            ObjType::String
            | ObjType::Fn
            | ObjType::Range
            | ObjType::Foreign
            | ObjType::TypedArray
            | ObjType::Simd
            | ObjType::Buffer => {}

            ObjType::List => {
                let list = &*(header as *mut ObjList);
                if let Some(buffer) = list.buffer_object() {
                    child_ptr(buffer, f);
                }
                for &val in list.as_slice() {
                    child_value(val, f);
                }
            }

            ObjType::Map => {
                let map = &*(header as *mut ObjMap);
                for (key, &val) in &map.entries {
                    child_value(key.value(), f);
                    child_value(val, f);
                }
            }

            ObjType::Closure => {
                let closure = &*(header as *mut ObjClosure);
                child_ptr(closure.function as *mut ObjHeader, f);
                for &uv in &closure.upvalues {
                    child_ptr(uv as *mut ObjHeader, f);
                }
                child_ptr(closure.defining_class as *mut ObjHeader, f);
            }

            ObjType::Upvalue => {
                let uv = &*(header as *mut ObjUpvalue);
                child_value(uv.closed, f);
            }

            ObjType::Fiber => {
                let fiber = &*(header as *mut ObjFiber);
                for &val in &fiber.stack {
                    child_value(val, f);
                }
                for frame in &fiber.frames {
                    child_ptr(frame.closure as *mut ObjHeader, f);
                }
                for frame in &fiber.mir_frames {
                    for &val in &frame.values {
                        child_value(val, f);
                    }
                    if let Some(closure) = frame.closure {
                        child_ptr(closure as *mut ObjHeader, f);
                    }
                    if let Some(class) = frame.defining_class {
                        child_ptr(class as *mut ObjHeader, f);
                    }
                }
                child_ptr(fiber.caller as *mut ObjHeader, f);
                child_value(fiber.error, f);
                child_value(fiber.context_map, f);
                if let Some(v) = fiber.jit_resume_value {
                    child_value(v, f);
                }
                #[cfg(feature = "host")]
                {
                    child_value(fiber.krio_return_value, f);
                    for &val in &fiber.krio_jit_roots {
                        child_value(val, f);
                    }
                }
            }

            ObjType::Class => {
                let class = &*(header as *mut ObjClass);
                child_ptr(class.superclass as *mut ObjHeader, f);
                for method in class.methods.iter().flatten() {
                    match method {
                        Method::Closure(ptr) | Method::Constructor(ptr) => {
                            child_ptr(*ptr as *mut ObjHeader, f);
                        }
                        Method::Native(_)
                        | Method::Host(..)
                        | Method::ForeignC(_)
                        | Method::ForeignCDynamic(_) => {}
                    }
                }
                for &val in class.static_fields.values() {
                    child_value(val, f);
                }
            }

            ObjType::Instance => {
                let inst = &*(header as *mut ObjInstance);
                if !inst.fields.is_null() {
                    for i in 0..inst.num_fields as usize {
                        child_value(*inst.fields.add(i), f);
                    }
                }
            }

            ObjType::Module => {
                let module = &*(header as *mut ObjModule);
                for &val in &module.variables {
                    child_value(val, f);
                }
            }
        }
    }
}

/// Drop what an object owns in place; the collector frees its cells.
pub(super) unsafe fn drop_in_place_by_type(header: *mut ObjHeader) {
    unsafe {
        match (*header).obj_type {
            ObjType::String => std::ptr::drop_in_place(header as *mut ObjString),
            ObjType::List => std::ptr::drop_in_place(header as *mut ObjList),
            ObjType::Map => std::ptr::drop_in_place(header as *mut ObjMap),
            ObjType::Range => std::ptr::drop_in_place(header as *mut ObjRange),
            ObjType::Fn => std::ptr::drop_in_place(header as *mut ObjFn),
            ObjType::Closure => std::ptr::drop_in_place(header as *mut ObjClosure),
            ObjType::Upvalue => std::ptr::drop_in_place(header as *mut ObjUpvalue),
            ObjType::Fiber => std::ptr::drop_in_place(header as *mut ObjFiber),
            ObjType::Class => std::ptr::drop_in_place(header as *mut ObjClass),
            ObjType::Instance => std::ptr::drop_in_place(header as *mut ObjInstance),
            ObjType::Foreign => std::ptr::drop_in_place(header as *mut ObjForeign),
            ObjType::Module => std::ptr::drop_in_place(header as *mut ObjModule),
            ObjType::TypedArray => std::ptr::drop_in_place(header as *mut ObjTypedArray),
            ObjType::Simd => std::ptr::drop_in_place(header as *mut ObjSimd),
            ObjType::Buffer => {}
        }
    }
}
