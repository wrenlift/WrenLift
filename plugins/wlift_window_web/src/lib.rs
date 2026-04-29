//! `wlift_window_web` — browser-canvas window provider for the
//! WrenLift game framework. Pure C-ABI plugin: zero Rust deps,
//! every capability is a wasm import.
//!
//! Architecture
//! ------------
//!
//! Each foreign method is a thin trampoline that:
//!
//!   1. Reads its arguments off the VM slot stack via the host's
//!      Wren C API (`wrenGetSlot*` — pointer-free, called
//!      directly across the wasm-module boundary).
//!   2. Delegates the actual work to a JS-provided DOM bridge
//!      (`wlift_dom_*`) — canvas creation, event listener
//!      install, queue drain.
//!   3. Writes the result back via `wrenSetSlot*` or, for the
//!      drain path, lets the bridge build the Wren list directly
//!      against the live VM.
//!
//! The plugin owns *no* DOM state of its own; the JS loader's
//! per-canvas state lives in the host's address space (where it
//! can directly touch `wlift_wasm`'s exports). The plugin just
//! references canvases by integer handle.
//!
//! Build
//! -----
//!
//!   cargo build -p wlift_window_web --target wasm32-unknown-unknown --release

#![allow(clippy::missing_safety_doc)]

use core::ffi::c_void;

// ---------------------------------------------------------------
// Wren C API imports (pointer-free) — provided by `wlift_wasm`'s
// `#[no_mangle] pub extern "C" fn wren*` exports. The JS loader
// wires these directly from the host's exports object, so calls
// happen inside the host's wasm context with no JS hop.
#[link(wasm_import_module = "env")]
unsafe extern "C" {
    fn wrenEnsureSlots(vm: *mut c_void, num_slots: i32);
    fn wrenGetSlotDouble(vm: *mut c_void, slot: i32) -> f64;
    fn wrenSetSlotDouble(vm: *mut c_void, slot: i32, value: f64);
    fn wrenSetSlotBool(vm: *mut c_void, slot: i32, value: bool);
    fn wrenSetSlotNull(vm: *mut c_void, slot: i32);
    fn wrenSetSlotNewMap(vm: *mut c_void, slot: i32);
    fn wrenSetMapValue(vm: *mut c_void, map_slot: i32, key_slot: i32, value_slot: i32);
}

// String-shaped slot ops cross plugin↔host memory; the loader's
// wlift_get_slot_str / wlift_set_slot_str bridges handle the
// memcpy.
#[link(wasm_import_module = "env")]
unsafe extern "C" {
    fn wlift_get_slot_str(vm: *mut c_void, slot: i32, out: *mut u8, max: i32) -> i32;
    fn wlift_set_slot_str(vm: *mut c_void, slot: i32, ptr: *const u8, len: i32);
}

// ---------------------------------------------------------------
// DOM bridge — JS-provided. Plugin tells the bridge what to do;
// bridge owns all canvas / event-listener state.
//
// `parent_id` is a CSS-id-style selector ("stage" → `#stage`).
// Empty string means "attach to <body>".
//
// `wlift_dom_create_canvas` returns an opaque u32 handle (0 on
// failure). All other DOM fns take that handle.
#[link(wasm_import_module = "env")]
unsafe extern "C" {
    fn wlift_dom_create_canvas(
        parent_id_ptr: *const u8,
        parent_id_len: i32,
        width: u32,
        height: u32,
    ) -> u32;
    fn wlift_dom_destroy_canvas(handle: u32);
    fn wlift_dom_canvas_width(handle: u32) -> u32;
    fn wlift_dom_canvas_height(handle: u32) -> u32;
    fn wlift_dom_close_requested(handle: u32) -> i32;
    fn wlift_dom_set_canvas_attribute(
        handle: u32,
        name_ptr: *const u8,
        name_len: i32,
        value_ptr: *const u8,
        value_len: i32,
    );
    /// Drain the canvas's event queue and append each event as a
    /// Map onto the Wren list at `list_slot`. Bridge does all the
    /// slot building via `wrenSetSlotNewMap` / `wrenInsertInList`.
    fn wlift_dom_drain_events_into_list(vm: *mut c_void, list_slot: i32, handle: u32);
}

// ---------------------------------------------------------------
// Helpers for descriptor-Map decode
// ---------------------------------------------------------------

#[link(wasm_import_module = "env")]
unsafe extern "C" {
    fn wrenGetSlotType(vm: *mut c_void, slot: i32) -> i32;
    fn wrenGetSlotBool(vm: *mut c_void, slot: i32) -> bool;
    fn wrenGetMapContainsKey(vm: *mut c_void, map_slot: i32, key_slot: i32) -> bool;
    fn wrenGetMapValue(vm: *mut c_void, map_slot: i32, key_slot: i32, value_slot: i32);
}

const WREN_TYPE_BOOL: i32 = 0;
const WREN_TYPE_NUM: i32 = 1;
const WREN_TYPE_STRING: i32 = 6;

/// Look up `key` in the Map at `map_slot`. Stores the value at
/// `dest_slot` (or null if absent). Returns the WrenType code of
/// the value (or `5` (`WREN_TYPE_NULL`) for missing keys).
unsafe fn map_lookup(
    vm: *mut c_void,
    map_slot: i32,
    key: &str,
    key_slot: i32,
    dest_slot: i32,
) -> i32 {
    unsafe {
        wlift_set_slot_str(vm, key_slot, key.as_ptr(), key.len() as i32);
        if !wrenGetMapContainsKey(vm, map_slot, key_slot) {
            wrenSetSlotNull(vm, dest_slot);
            return 5;
        }
        wrenGetMapValue(vm, map_slot, key_slot, dest_slot);
        wrenGetSlotType(vm, dest_slot)
    }
}

unsafe fn map_num(vm: *mut c_void, map_slot: i32, key: &str, default: f64) -> f64 {
    unsafe {
        wrenEnsureSlots(vm, 4);
        let ty = map_lookup(vm, map_slot, key, 2, 3);
        if ty == WREN_TYPE_NUM {
            wrenGetSlotDouble(vm, 3)
        } else {
            default
        }
    }
}

unsafe fn map_bool(vm: *mut c_void, map_slot: i32, key: &str, default: bool) -> bool {
    unsafe {
        wrenEnsureSlots(vm, 4);
        let ty = map_lookup(vm, map_slot, key, 2, 3);
        if ty == WREN_TYPE_BOOL {
            wrenGetSlotBool(vm, 3)
        } else {
            default
        }
    }
}

/// Read a String at `slot` into a stack buffer (≤256 bytes). Used
/// for "title" and the "parent_id" descriptor field.
unsafe fn map_string_into<'a>(
    vm: *mut c_void,
    map_slot: i32,
    key: &str,
    buf: &'a mut [u8; 256],
) -> Option<&'a [u8]> {
    unsafe {
        wrenEnsureSlots(vm, 4);
        let ty = map_lookup(vm, map_slot, key, 2, 3);
        if ty != WREN_TYPE_STRING {
            return None;
        }
        let n = wlift_get_slot_str(vm, 3, buf.as_mut_ptr(), buf.len() as i32);
        if n <= 0 {
            return None;
        }
        Some(&buf[..n as usize])
    }
}

// ---------------------------------------------------------------
// Foreign methods — exported by name; the loader registers each
// against the foreign-method registry on plugin install.
// ---------------------------------------------------------------

/// `Window.create_(descriptor)` — descriptor keys:
///   "title":     String  (default "wlift")
///   "width":     Num     (default 800)
///   "height":    Num     (default 600)
///   "parent":    String  (default "stage")  — CSS id of the host
///                                             element to attach
///                                             under.
///   "resizable": Bool    (default true)     — currently unused;
///                                             page CSS owns
///                                             resize behaviour.
///
/// Returns the canvas handle (Num) at slot 0, or `-1` on failure.
#[no_mangle]
pub unsafe extern "C" fn wlift_window_create(vm: *mut c_void) {
    let mut title_buf = [0u8; 256];
    let mut parent_buf = [0u8; 256];

    let title = unsafe { map_string_into(vm, 1, "title", &mut title_buf) };
    let parent = unsafe { map_string_into(vm, 1, "parent", &mut parent_buf) }
        .unwrap_or(b"stage");
    let width = unsafe { map_num(vm, 1, "width", 800.0) } as u32;
    let height = unsafe { map_num(vm, 1, "height", 600.0) } as u32;
    let _resizable = unsafe { map_bool(vm, 1, "resizable", true) };

    let handle = unsafe {
        wlift_dom_create_canvas(parent.as_ptr(), parent.len() as i32, width, height)
    };

    if handle == 0 {
        unsafe { wrenSetSlotDouble(vm, 0, -1.0) };
        return;
    }

    if let Some(t) = title {
        unsafe {
            wlift_dom_set_canvas_attribute(
                handle,
                b"title".as_ptr(),
                5,
                t.as_ptr(),
                t.len() as i32,
            );
        }
    }

    unsafe { wrenSetSlotDouble(vm, 0, handle as f64) };
}

/// `Window.destroy_(id)` — removes the canvas from the DOM and
/// releases its event-listener closures.
#[no_mangle]
pub unsafe extern "C" fn wlift_window_destroy(vm: *mut c_void) {
    let handle = unsafe { wrenGetSlotDouble(vm, 1) } as u32;
    unsafe { wlift_dom_destroy_canvas(handle) };
    unsafe { wrenSetSlotNull(vm, 0) };
}

/// `Window.pump_(_)` — no-op on web. The browser already runs the
/// event loop; events arrive asynchronously into the queue. Kept
/// for API parity with the native `wlift_window`.
#[no_mangle]
pub unsafe extern "C" fn wlift_window_pump(vm: *mut c_void) {
    unsafe { wrenSetSlotNull(vm, 0) };
}

#[no_mangle]
pub unsafe extern "C" fn wlift_window_close_requested(vm: *mut c_void) {
    let handle = unsafe { wrenGetSlotDouble(vm, 1) } as u32;
    let flag = unsafe { wlift_dom_close_requested(handle) } != 0;
    unsafe { wrenSetSlotBool(vm, 0, flag) };
}

/// `Window.size_(_)` — returns `{ "width": Num, "height": Num }`.
#[no_mangle]
pub unsafe extern "C" fn wlift_window_size(vm: *mut c_void) {
    let handle = unsafe { wrenGetSlotDouble(vm, 1) } as u32;
    let w = unsafe { wlift_dom_canvas_width(handle) };
    let h = unsafe { wlift_dom_canvas_height(handle) };
    unsafe {
        wrenEnsureSlots(vm, 4);
        wrenSetSlotNewMap(vm, 0);
        wlift_set_slot_str(vm, 2, b"width".as_ptr(), 5);
        wrenSetSlotDouble(vm, 3, w as f64);
        wrenSetMapValue(vm, 0, 2, 3);
        wlift_set_slot_str(vm, 2, b"height".as_ptr(), 6);
        wrenSetSlotDouble(vm, 3, h as f64);
        wrenSetMapValue(vm, 0, 2, 3);
    }
}

/// `Window.drainEvents_(_)` — returns a List of event Maps. Each
/// map carries at minimum a `"type"` key; mouse events also
/// include `"x"`, `"y"`, `"button"`; keyboard events include
/// `"key"`. The bridge owns the per-canvas event queue and
/// builds the Wren list directly against the live VM.
#[no_mangle]
pub unsafe extern "C" fn wlift_window_drain_events(vm: *mut c_void) {
    let handle = unsafe { wrenGetSlotDouble(vm, 1) } as u32;
    unsafe {
        wrenEnsureSlots(vm, 6);
        wlift_dom_drain_events_into_list(vm, 0, handle);
    }
}
