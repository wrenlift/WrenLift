//! Host-side exports for the `wlift_abi` plugin surface.
//!
//! Every function here is `#[no_mangle] pub unsafe extern "C"` — the
//! dynamic linker resolves a plugin cdylib's `extern "C"` references
//! against these symbols at dlopen time (with the host binary built
//! `-Wl,--export-dynamic` or platform equivalent). On wasm the same
//! `extern "C"` blocks resolve against these exports at static-link
//! time.
//!
//! The plugin side of the contract lives in `crates/wlift_abi`. The
//! two files together define every value that crosses the plugin
//! boundary; nothing in `runtime::vm`, `runtime::object`, or
//! `runtime::gc` reaches plugin code anymore.
//!
//! # ABI versioning
//!
//! [`PLUGIN_ABI_VERSION`] must be bumped on any wire-shape change
//! (Value bit layout, ObjType / TypedArrayKind discriminant
//! renumbering, function signature change, function removal). The
//! plugin loader (`runtime::foreign`) compares plugin and host
//! versions at dlopen and aborts cleanly on mismatch, replacing the
//! previous silent SIGSEGV when struct layouts drifted.

use crate::runtime::object::{NativeContext, ObjHeader, ObjList, ObjMap, ObjString, ObjType, ObjTypedArray, TypedArrayKind};
use crate::runtime::value::Value;
use crate::runtime::vm::VM;

/// Host-side ABI version. Mirror of `wlift_abi::ABI_VERSION`; bump
/// in lockstep when changing either side.
pub const PLUGIN_ABI_VERSION: u32 = 2;

// --- Helpers ---------------------------------------------------------------

#[inline(always)]
unsafe fn vm_ref<'a>(vm: *mut ()) -> &'a mut VM {
    unsafe { &mut *(vm as *mut VM) }
}

#[inline(always)]
fn obj_ptr(v: u64) -> Option<*mut u8> {
    Value::from_bits(v).as_object()
}

#[inline(always)]
fn typed_obj<T>(v: u64, expected: ObjType) -> Option<*mut T> {
    let p = obj_ptr(v)?;
    let hdr = p as *const ObjHeader;
    if unsafe { (*hdr).obj_type } == expected {
        Some(p as *mut T)
    } else {
        None
    }
}

// --- ABI handshake ---------------------------------------------------------

#[no_mangle]
pub extern "C" fn wlift_plugin_abi_version() -> u32 {
    PLUGIN_ABI_VERSION
}

// --- VM stack ops ----------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn wlift_plugin_slot(vm: *mut (), idx: u32) -> u64 {
    let vm = unsafe { vm_ref(vm) };
    vm.api_stack
        .get(idx as usize)
        .copied()
        .unwrap_or_else(Value::null)
        .to_bits()
}

#[no_mangle]
pub unsafe extern "C" fn wlift_plugin_set_return(vm: *mut (), value: u64) {
    let vm = unsafe { vm_ref(vm) };
    let v = Value::from_bits(value);
    if vm.api_stack.is_empty() {
        vm.api_stack.push(v);
    } else {
        vm.api_stack[0] = v;
    }
}

#[no_mangle]
pub unsafe extern "C" fn wlift_plugin_runtime_error(vm: *mut (), msg: *const u8, len: u32) {
    let vm = unsafe { vm_ref(vm) };
    let bytes = unsafe { std::slice::from_raw_parts(msg, len as usize) };
    let s = std::str::from_utf8(bytes).unwrap_or("<invalid utf-8>").to_string();
    vm.runtime_error(s);
}

// --- Allocators ------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn wlift_plugin_alloc_string(
    vm: *mut (),
    bytes: *const u8,
    len: u32,
) -> u64 {
    let vm = unsafe { vm_ref(vm) };
    let slice = unsafe { std::slice::from_raw_parts(bytes, len as usize) };
    // String content can be non-UTF-8 if the plugin is bridging raw
    // bytes (rare), but the runtime expects valid UTF-8. Reject by
    // raising a runtime error and returning null.
    let s = match std::str::from_utf8(slice) {
        Ok(s) => s.to_string(),
        Err(_) => {
            vm.runtime_error("wlift_plugin_alloc_string: not valid UTF-8".to_string());
            return Value::null().to_bits();
        }
    };
    vm.alloc_string(s).to_bits()
}

#[no_mangle]
pub unsafe extern "C" fn wlift_plugin_alloc_list(vm: *mut (), capacity: u32) -> u64 {
    let vm = unsafe { vm_ref(vm) };
    let mut storage = Vec::with_capacity(capacity as usize);
    // alloc_list expects an initial element vector; passing empty
    // gives us an empty list with at least `capacity` slots reserved
    // by the underlying allocator once the first add happens.
    let _ = capacity;
    storage.clear();
    vm.alloc_list(storage).to_bits()
}

#[no_mangle]
pub unsafe extern "C" fn wlift_plugin_alloc_map(vm: *mut ()) -> u64 {
    let vm = unsafe { vm_ref(vm) };
    vm.alloc_map().to_bits()
}

#[no_mangle]
pub unsafe extern "C" fn wlift_plugin_alloc_typed_array(
    vm: *mut (),
    count: u32,
    kind: u8,
) -> u64 {
    let vm = unsafe { vm_ref(vm) };
    let Some(kind) = TypedArrayKind::from_u8(kind) else {
        vm.runtime_error(format!(
            "wlift_plugin_alloc_typed_array: unknown kind tag {}",
            kind
        ));
        return Value::null().to_bits();
    };
    vm.alloc_typed_array(count, kind).to_bits()
}

// --- Object inspection -----------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn wlift_plugin_obj_type(value: u64) -> u8 {
    let Some(p) = obj_ptr(value) else {
        return 0xFF;
    };
    let hdr = p as *const ObjHeader;
    unsafe { (*hdr).obj_type as u8 }
}

#[no_mangle]
pub unsafe extern "C" fn wlift_plugin_string_bytes(
    value: u64,
    out_ptr: *mut *const u8,
    out_len: *mut u32,
) -> bool {
    let Some(s) = typed_obj::<ObjString>(value, ObjType::String) else {
        return false;
    };
    let bytes = unsafe { (*s).as_str().as_bytes() };
    unsafe {
        *out_ptr = bytes.as_ptr();
        *out_len = bytes.len() as u32;
    }
    true
}

#[no_mangle]
pub unsafe extern "C" fn wlift_plugin_list_count(value: u64) -> u32 {
    match typed_obj::<ObjList>(value, ObjType::List) {
        Some(l) => unsafe { (*l).count },
        None => 0,
    }
}

#[no_mangle]
pub unsafe extern "C" fn wlift_plugin_list_get(value: u64, idx: u32) -> u64 {
    let Some(l) = typed_obj::<ObjList>(value, ObjType::List) else {
        return Value::null().to_bits();
    };
    let count = unsafe { (*l).count };
    if idx >= count {
        return Value::null().to_bits();
    }
    let v = unsafe { *(*l).elements.add(idx as usize) };
    v.to_bits()
}

#[no_mangle]
pub unsafe extern "C" fn wlift_plugin_list_add(vm: *mut (), list: u64, value: u64) {
    let _ = vm; // GC rooting handled inside ObjList::add (ensure_capacity)
    let Some(l) = typed_obj::<ObjList>(list, ObjType::List) else {
        return;
    };
    unsafe { (*l).add(Value::from_bits(value)) }
}

#[no_mangle]
pub unsafe extern "C" fn wlift_plugin_map_count(value: u64) -> u32 {
    match typed_obj::<ObjMap>(value, ObjType::Map) {
        Some(m) => unsafe { (*m).entries.len() as u32 },
        None => 0,
    }
}

#[no_mangle]
pub unsafe extern "C" fn wlift_plugin_map_iter_next(
    value: u64,
    cursor: u32,
    out_key: *mut u64,
    out_value: *mut u64,
) -> u32 {
    let Some(m) = typed_obj::<ObjMap>(value, ObjType::Map) else {
        return 0;
    };
    let entries = unsafe { &(*m).entries };
    // Cursor is 1-based — 0 means "start of iteration"; the return
    // value is the cursor for the NEXT call (or 0 when done).
    let idx = cursor as usize;
    if idx >= entries.len() {
        return 0;
    }
    let (k, v) = entries.get_index(idx).unwrap();
    unsafe {
        *out_key = k.0.to_bits();
        *out_value = v.to_bits();
    }
    (cursor + 1)
}

#[no_mangle]
pub unsafe extern "C" fn wlift_plugin_map_set(vm: *mut (), map: u64, key: u64, value: u64) {
    let _ = vm; // ObjMap::set handles its own resize-grow; key + value bits already on the heap
    let Some(m) = typed_obj::<ObjMap>(map, ObjType::Map) else {
        return;
    };
    unsafe { (*m).set(Value::from_bits(key), Value::from_bits(value)) }
}

#[no_mangle]
pub unsafe extern "C" fn wlift_plugin_typed_array_kind(value: u64) -> u8 {
    let Some(a) = typed_obj::<ObjTypedArray>(value, ObjType::TypedArray) else {
        return 0xFF;
    };
    unsafe { (*a).kind_tag() as u8 }
}

#[no_mangle]
pub unsafe extern "C" fn wlift_plugin_typed_array_bytes(
    value: u64,
    out_ptr: *mut *mut u8,
    out_len: *mut u32,
) -> bool {
    let Some(a) = typed_obj::<ObjTypedArray>(value, ObjType::TypedArray) else {
        return false;
    };
    let bytes = unsafe { (*a).as_bytes_mut() };
    unsafe {
        *out_ptr = bytes.as_mut_ptr();
        *out_len = bytes.len() as u32;
    }
    true
}

// --- Static plugin registration (wasm path) --------------------------------

/// Register a `(library, symbol)` → foreign-function-pointer mapping.
/// Wasm static-link callers invoke this at module init so dispatch
/// can find their methods without a `dlopen`.
///
/// Native dlopen'd plugins don't need this — the host's
/// `runtime::foreign::resolve_symbol` looks symbols up by name via
/// `libloading::Symbol`. The wasm runtime can't dlopen, hence the
/// pre-registration path. This export is a no-op on native so the
/// plugin source stays target-agnostic.
#[cfg(target_arch = "wasm32")]
#[no_mangle]
pub unsafe extern "C" fn wlift_plugin_register_symbol(
    plugin: *const u8,
    plugin_len: u32,
    name: *const u8,
    name_len: u32,
    fn_ptr: *const (),
) {
    let plib = unsafe { std::slice::from_raw_parts(plugin, plugin_len as usize) };
    let pname = unsafe { std::slice::from_raw_parts(name, name_len as usize) };
    let library = std::str::from_utf8(plib).unwrap_or("");
    let symbol = std::str::from_utf8(pname).unwrap_or("");
    if library.is_empty() || symbol.is_empty() {
        return;
    }
    // SAFETY: caller asserts fn_ptr points at an `extern "C" fn(*mut VM)`.
    let func: unsafe extern "C" fn(*mut VM) = unsafe { std::mem::transmute(fn_ptr) };
    crate::runtime::foreign::register_plugin_symbol_unsafe(library, symbol, func);
}

#[cfg(not(target_arch = "wasm32"))]
#[no_mangle]
pub unsafe extern "C" fn wlift_plugin_register_symbol(
    _plugin: *const u8,
    _plugin_len: u32,
    _name: *const u8,
    _name_len: u32,
    _fn_ptr: *const (),
) {
    // No-op on native: foreign symbols are resolved by `dlsym` at
    // dispatch time, not pre-registered.
}
