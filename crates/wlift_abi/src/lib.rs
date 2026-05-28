//! Stable C-shaped ABI for WrenLift plugins.
//!
//! A plugin cdylib `use`s this crate (and nothing else from the
//! wren_lift workspace). It calls the host through `extern "C"`
//! functions whose symbols are exported with `#[no_mangle]` by
//! the host binary; the dynamic linker resolves the plugin's
//! external refs against the host's exports at `dlopen` time.
//!
//! Native binaries built from this workspace pass
//! `-Wl,--export-dynamic` (or platform equivalent) so the host's
//! `#[no_mangle]` symbols stay visible to dlopen'd plugins.
//!
//! On wasm, plugins are rlib-linked into `wlift_wasm`; the wasm
//! linker resolves the same `extern "C"` decls against the host
//! exports at static-link time. Identical surface, identical
//! plugin source, two link strategies.
//!
//! # Stability contract
//!
//! Every item in this crate is part of the plugin ABI. Adding a
//! new function is fine. Removing or re-shaping an existing one
//! requires bumping [`ABI_VERSION`] so the host can refuse to load
//! older plugin bundles with a clear error.

#![no_std]
#![allow(clippy::missing_safety_doc)]

// -- ABI version ------------------------------------------------------------

/// Compiled-in ABI version. Plugins call [`wlift_plugin_abi_version`]
/// at load time and compare against this; mismatched versions abort
/// with a "rebuild plugin against wren_lift ≥ X" message rather
/// than the SIGSEGV that struct-layout drift used to produce.
///
/// Bump on every wire-shape change:
/// - Value bit layout
/// - ObjType / TypedArrayKind discriminant renumbering
/// - Function signature change
/// - Function removal
///
/// Stays at 1 during the migration window — the new
/// `wlift_plugin_*` exports are additive, so plugins built against
/// `wlift_abi` are still load-compatible with the host's existing
/// v1 handshake. Bump to 2 in Phase E once the unmigrated
/// VM-direct surface is retired.
pub const ABI_VERSION: u32 = 1;

// -- Value ------------------------------------------------------------------

/// NaN-boxed 64-bit value. Matches `wren_lift::runtime::value::Value`
/// bit-for-bit; the host passes raw bits across the boundary and
/// this side just sees the encoded `u64`.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct Value(pub u64);

/// Base quiet-NaN payload. Every non-number value sets these bits.
pub const QNAN: u64 = 0x7FFC_0000_0000_0000;
pub const TAG_NULL: u64 = QNAN;
pub const TAG_FALSE: u64 = QNAN | 1;
pub const TAG_TRUE: u64 = QNAN | 2;
pub const TAG_UNDEFINED: u64 = QNAN | 3;
const SIGN_BIT: u64 = 1 << 63;
pub const TAG_OBJ: u64 = SIGN_BIT | QNAN;
pub const PTR_MASK: u64 = 0x0000_FFFF_FFFF_FFFF;

impl Value {
    pub const NULL: Value = Value(TAG_NULL);
    pub const FALSE: Value = Value(TAG_FALSE);
    pub const TRUE: Value = Value(TAG_TRUE);

    #[inline(always)]
    pub fn num(n: f64) -> Self {
        Value(n.to_bits())
    }

    #[inline(always)]
    pub fn bool(b: bool) -> Self {
        if b {
            Value::TRUE
        } else {
            Value::FALSE
        }
    }

    #[inline(always)]
    pub fn is_null(self) -> bool {
        self.0 == TAG_NULL
    }

    #[inline(always)]
    pub fn is_bool(self) -> bool {
        self.0 == TAG_TRUE || self.0 == TAG_FALSE
    }

    #[inline(always)]
    pub fn is_num(self) -> bool {
        (self.0 & QNAN) != QNAN
    }

    #[inline(always)]
    pub fn is_object(self) -> bool {
        (self.0 & TAG_OBJ) == TAG_OBJ
    }

    #[inline(always)]
    pub fn as_num(self) -> Option<f64> {
        if self.is_num() {
            Some(f64::from_bits(self.0))
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn as_bool(self) -> Option<bool> {
        match self.0 {
            TAG_TRUE => Some(true),
            TAG_FALSE => Some(false),
            _ => None,
        }
    }
}

// -- Object discriminants ---------------------------------------------------

/// Mirrors `wren_lift::runtime::object::ObjType` byte-for-byte.
/// `wlift_plugin_obj_type` returns this raw discriminant; the
/// plugin matches on it to decide which inspection helper to call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ObjType {
    String = 0,
    List = 1,
    Map = 2,
    Range = 3,
    Fn = 4,
    Closure = 5,
    Upvalue = 6,
    Fiber = 7,
    Class = 8,
    Instance = 9,
    Foreign = 10,
    Module = 11,
    TypedArray = 12,
    Simd = 13,
}

impl ObjType {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(ObjType::String),
            1 => Some(ObjType::List),
            2 => Some(ObjType::Map),
            3 => Some(ObjType::Range),
            4 => Some(ObjType::Fn),
            5 => Some(ObjType::Closure),
            6 => Some(ObjType::Upvalue),
            7 => Some(ObjType::Fiber),
            8 => Some(ObjType::Class),
            9 => Some(ObjType::Instance),
            10 => Some(ObjType::Foreign),
            11 => Some(ObjType::Module),
            12 => Some(ObjType::TypedArray),
            13 => Some(ObjType::Simd),
            _ => None,
        }
    }
}

/// Mirrors `wren_lift::runtime::object::TypedArrayKind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TypedArrayKind {
    U8 = 0,
    F32 = 1,
    F64 = 2,
    I32 = 3,
}

impl TypedArrayKind {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(TypedArrayKind::U8),
            1 => Some(TypedArrayKind::F32),
            2 => Some(TypedArrayKind::F64),
            3 => Some(TypedArrayKind::I32),
            _ => None,
        }
    }

    pub fn element_size(self) -> usize {
        match self {
            TypedArrayKind::U8 => 1,
            TypedArrayKind::F32 | TypedArrayKind::I32 => 4,
            TypedArrayKind::F64 => 8,
        }
    }
}

// -- Opaque VM pointer ------------------------------------------------------

/// Opaque handle the host hands to every plugin entry point.
/// Plugins must not dereference it; only pass it back to
/// `wlift_plugin_*` functions.
pub enum WrenVm {}

// -- Host exports (resolved by the dynamic / wasm linker) -------------------

extern "C" {
    /// Returns the host's [`ABI_VERSION`]. Plugins compare against
    /// their compiled-in constant at load time.
    pub fn wlift_plugin_abi_version() -> u32;

    // --- VM-stack ops ------------------------------------------------------

    /// Read the api_stack slot at `idx`. Slot 0 holds the receiver
    /// on entry / the return value on exit; slots 1..N hold args.
    /// Out-of-bounds returns [`TAG_NULL`].
    pub fn wlift_plugin_slot(vm: *mut WrenVm, idx: u32) -> u64;

    /// Write the return-value slot (api_stack[0]).
    pub fn wlift_plugin_set_return(vm: *mut WrenVm, value: u64);

    /// Raise a runtime error with the given UTF-8 message. The
    /// fiber will abort once the foreign call returns.
    pub fn wlift_plugin_runtime_error(vm: *mut WrenVm, msg: *const u8, len: u32);

    // --- Allocators (return Value bits directly) ---------------------------

    pub fn wlift_plugin_alloc_string(vm: *mut WrenVm, bytes: *const u8, len: u32) -> u64;
    pub fn wlift_plugin_alloc_list(vm: *mut WrenVm, capacity: u32) -> u64;
    pub fn wlift_plugin_alloc_map(vm: *mut WrenVm) -> u64;
    pub fn wlift_plugin_alloc_typed_array(vm: *mut WrenVm, count: u32, kind: u8) -> u64;

    // --- Object inspection (by Value bits) ---------------------------------

    /// Returns the [`ObjType`] discriminant for an object value, or
    /// `0xFF` if the value isn't an object.
    pub fn wlift_plugin_obj_type(value: u64) -> u8;

    /// Borrow the UTF-8 bytes of an ObjString. Returns `false` if
    /// `value` isn't a string. The slice stays valid until the next
    /// allocator call on the same VM.
    pub fn wlift_plugin_string_bytes(
        value: u64,
        out_ptr: *mut *const u8,
        out_len: *mut u32,
    ) -> bool;

    pub fn wlift_plugin_list_count(value: u64) -> u32;
    pub fn wlift_plugin_list_get(value: u64, idx: u32) -> u64;
    pub fn wlift_plugin_list_add(vm: *mut WrenVm, list: u64, value: u64);

    pub fn wlift_plugin_map_count(value: u64) -> u32;

    /// Iterate map entries. Pass `cursor = 0` to start; each call
    /// writes the entry's key + value through the out pointers and
    /// returns the next cursor (or `0` when iteration is complete).
    pub fn wlift_plugin_map_iter_next(
        value: u64,
        cursor: u32,
        out_key: *mut u64,
        out_value: *mut u64,
    ) -> u32;

    pub fn wlift_plugin_map_set(vm: *mut WrenVm, map: u64, key: u64, value: u64);

    pub fn wlift_plugin_typed_array_kind(value: u64) -> u8;

    /// Borrow the raw byte buffer of an ObjTypedArray. Out pointer
    /// is writable — callers may mutate in place. Returns `false`
    /// if `value` isn't a typed array.
    pub fn wlift_plugin_typed_array_bytes(
        value: u64,
        out_ptr: *mut *mut u8,
        out_len: *mut u32,
    ) -> bool;

    // --- Static plugin registration (wasm path) ----------------------------

    /// Register a `(plugin_name, fn_name) -> fn_ptr` mapping the
    /// host uses to dispatch foreign methods. Used by the wasm
    /// static-link path; native cdylibs are dlopen'd by name and
    /// don't need this.
    pub fn wlift_plugin_register_symbol(
        plugin: *const u8,
        plugin_len: u32,
        name: *const u8,
        name_len: u32,
        fn_ptr: *const (),
    );
}

// -- Ergonomic wrappers -----------------------------------------------------
//
// Plugins call these instead of writing `unsafe { ... }` around
// every extern call. Each wraps an extern with the minimal safety
// invariants documented inline.

/// Read an argument slot. Returns [`Value::NULL`] for out-of-bounds.
///
/// # Safety
/// `vm` must be the live `*mut WrenVm` the host handed to the
/// current foreign-method entry point. Same contract for every
/// wrapper below — the function's docs don't repeat it.
#[inline]
pub unsafe fn slot(vm: *mut WrenVm, idx: u32) -> Value {
    Value(unsafe { wlift_plugin_slot(vm, idx) })
}

/// Write the return-value slot (api_stack[0]).
#[inline]
pub unsafe fn set_return(vm: *mut WrenVm, value: Value) {
    unsafe { wlift_plugin_set_return(vm, value.0) }
}

/// Raise a runtime error. The fiber aborts when the foreign call
/// returns.
#[inline]
pub unsafe fn runtime_error(vm: *mut WrenVm, msg: &str) {
    unsafe {
        wlift_plugin_runtime_error(vm, msg.as_ptr(), msg.len() as u32);
    }
}

/// Allocate an [`ObjString`] from a Rust string slice.
#[inline]
pub unsafe fn alloc_string(vm: *mut WrenVm, s: &str) -> Value {
    Value(unsafe { wlift_plugin_alloc_string(vm, s.as_ptr(), s.len() as u32) })
}

/// Allocate an empty list with the given reserved capacity.
#[inline]
pub unsafe fn alloc_list(vm: *mut WrenVm, capacity: u32) -> Value {
    Value(unsafe { wlift_plugin_alloc_list(vm, capacity) })
}

/// Allocate an empty map.
#[inline]
pub unsafe fn alloc_map(vm: *mut WrenVm) -> Value {
    Value(unsafe { wlift_plugin_alloc_map(vm) })
}

/// Allocate a typed array with `count` elements of the given kind.
#[inline]
pub unsafe fn alloc_typed_array(vm: *mut WrenVm, count: u32, kind: TypedArrayKind) -> Value {
    Value(unsafe { wlift_plugin_alloc_typed_array(vm, count, kind as u8) })
}

/// Type-discriminate a Value without dereferencing the underlying
/// object pointer.
#[inline]
pub fn obj_type(value: Value) -> Option<ObjType> {
    if !value.is_object() {
        return None;
    }
    ObjType::from_u8(unsafe { wlift_plugin_obj_type(value.0) })
}

/// Get a string's bytes. The slice borrows from the GC heap and
/// is invalidated by the next allocator call.
#[inline]
pub fn string_bytes(value: Value) -> Option<&'static [u8]> {
    let mut ptr: *const u8 = core::ptr::null();
    let mut len: u32 = 0;
    let ok = unsafe { wlift_plugin_string_bytes(value.0, &mut ptr, &mut len) };
    if !ok || ptr.is_null() {
        return None;
    }
    Some(unsafe { core::slice::from_raw_parts(ptr, len as usize) })
}

/// Get a string's bytes as a `&str`. Returns `None` if not a string
/// or if the bytes aren't valid UTF-8 (should never happen — Wren
/// strings are UTF-8 by construction).
#[inline]
pub fn string_str(value: Value) -> Option<&'static str> {
    core::str::from_utf8(string_bytes(value)?).ok()
}

/// Number of elements in a list.
#[inline]
pub fn list_count(value: Value) -> u32 {
    unsafe { wlift_plugin_list_count(value.0) }
}

/// Read a list element by index. Out-of-bounds returns [`Value::NULL`].
#[inline]
pub fn list_get(value: Value, idx: u32) -> Value {
    Value(unsafe { wlift_plugin_list_get(value.0, idx) })
}

/// GC-rooted append. Required if the caller plans to allocate
/// further values after `list` — the plain field write would lose
/// the element if a minor GC fires between the write and the next
/// allocator call.
#[inline]
pub unsafe fn list_add(vm: *mut WrenVm, list: Value, value: Value) {
    unsafe { wlift_plugin_list_add(vm, list.0, value.0) }
}

/// Number of entries in a map.
#[inline]
pub fn map_count(value: Value) -> u32 {
    unsafe { wlift_plugin_map_count(value.0) }
}

/// Iterator over map entries, in insertion order.
pub struct MapIter {
    map: Value,
    cursor: u32,
}

impl Iterator for MapIter {
    type Item = (Value, Value);
    fn next(&mut self) -> Option<Self::Item> {
        let mut k: u64 = 0;
        let mut v: u64 = 0;
        let next = unsafe { wlift_plugin_map_iter_next(self.map.0, self.cursor, &mut k, &mut v) };
        if next == 0 {
            return None;
        }
        self.cursor = next;
        Some((Value(k), Value(v)))
    }
}

#[inline]
pub fn map_iter(map: Value) -> MapIter {
    MapIter { map, cursor: 0 }
}

#[inline]
pub unsafe fn map_set(vm: *mut WrenVm, map: Value, key: Value, value: Value) {
    unsafe { wlift_plugin_map_set(vm, map.0, key.0, value.0) }
}

/// Read the element-type tag of a typed array.
#[inline]
pub fn typed_array_kind(value: Value) -> Option<TypedArrayKind> {
    let tag = unsafe { wlift_plugin_typed_array_kind(value.0) };
    TypedArrayKind::from_u8(tag)
}

/// Borrow the raw byte buffer of a typed array. The slice borrows
/// from the GC heap; invalidated by the next allocator call.
#[inline]
pub fn typed_array_bytes_mut(value: Value) -> Option<&'static mut [u8]> {
    let mut ptr: *mut u8 = core::ptr::null_mut();
    let mut len: u32 = 0;
    let ok = unsafe { wlift_plugin_typed_array_bytes(value.0, &mut ptr, &mut len) };
    if !ok || ptr.is_null() {
        return None;
    }
    Some(unsafe { core::slice::from_raw_parts_mut(ptr, len as usize) })
}

#[inline]
pub fn typed_array_bytes(value: Value) -> Option<&'static [u8]> {
    typed_array_bytes_mut(value).map(|s| &*s)
}

/// Register a `(plugin, name) -> fn` mapping. Wasm static-link
/// callers invoke this at module init; native cdylibs are
/// dlopen'd by symbol and don't need it.
///
/// # Safety
/// `fn_ptr` must be a valid `extern "C" fn(*mut WrenVm)` matching
/// the host's foreign-method dispatch signature.
#[inline]
pub unsafe fn register_symbol(plugin: &str, name: &str, fn_ptr: *const ()) {
    unsafe {
        wlift_plugin_register_symbol(
            plugin.as_ptr(),
            plugin.len() as u32,
            name.as_ptr(),
            name.len() as u32,
            fn_ptr,
        )
    }
}
