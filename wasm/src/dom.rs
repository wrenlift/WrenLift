//! DOM bridge — first foreign module that needs the **main
//! thread's** globals (`document`), not just the worker scope's.
//!
//! Worker mode can't call `document.querySelector` directly: a
//! Web Worker has no DOM. The shape we use for every DOM-touching
//! foreign method:
//!
//!   1. `dom_*` foreign method mints a future handle on the wasm
//!      side, then calls a `_wlift_dom_*` JS shim (just like
//!      `setTimeout` / `fetch`).
//!
//!   2. In main-thread mode (`createWlift({ mode: "main" })`)
//!      the shim runs `document.*` directly and resolves the
//!      handle inline.
//!
//!   3. In worker mode (`createWlift()`, the default) the
//!      shim *cannot* see the DOM, so it `postMessage`s a
//!      `dom-op` request to the main thread; the page's
//!      `WorkerWlift` adapter dispatches it, runs the actual
//!      DOM call, and posts a `dom-result` back. The worker's
//!      message handler then settles the same handle from
//!      `wlift.js`.
//!
//!   4. Wasi (the smoke test) has no DOM either; we resolve
//!      synchronously with a deterministic `MOCK_DOM_*` payload
//!      so the bridge wiring is verifiable end-to-end.
//!
//! Surface today:
//!
//!   * `Dom.text(selector)` / `Dom.setText(selector, value)` —
//!     read/write `textContent`.
//!   * `Dom.getAttribute(selector, name)` /
//!     `Dom.setAttribute(selector, name, value)` — read/write
//!     attributes.
//!   * `Dom.addClass(selector, name)` /
//!     `Dom.removeClass(selector, name)` — toggle CSS classes.
//!   * `Dom.queryAll(selector)` — count of matching elements
//!     (string-encoded so the existing `Future<String>`
//!     transport can carry it without protocol churn).
//!
//! Adding more (`appendChild`, `removeChild`, custom-element
//! lifecycle, etc.) follows the same template — pull strings
//! out of the api_stack, mint a future, dispatch by cfg.

use wren_lift::runtime::object::{NativeContext, ObjString};
use wren_lift::runtime::value::Value;
use wren_lift::runtime::vm::VM;

use crate::create_pending_future;
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
use crate::resolve_future;

// ---------------------------------------------------------------------------
// Internal helpers — every foreign method below pulls 1–3 strings
// from the api_stack and writes a future-handle return slot. The
// helpers keep the bodies focused on the dispatch (browser vs
// wasi) rather than the marshalling boilerplate.
// ---------------------------------------------------------------------------

unsafe fn read_string_slot(vm: &mut VM, slot: usize, op: &str, arg_name: &str) -> Option<String> {
    let v = vm.api_stack.get(slot).copied().unwrap_or(Value::null());
    if v.is_string_object() {
        let ptr = v.as_object().expect("string object pointer");
        unsafe { Some((*(ptr as *const ObjString)).as_str().to_string()) }
    } else {
        vm.runtime_error(format!("{}: {} must be a String.", op, arg_name));
        None
    }
}

unsafe fn write_handle(vm: &mut VM, handle: u32) {
    if vm.api_stack.is_empty() {
        vm.api_stack.push(Value::num(handle as f64));
    } else {
        vm.api_stack[0] = Value::num(handle as f64);
    }
}

// ---------------------------------------------------------------------------
// Foreign-method shims
// ---------------------------------------------------------------------------

/// `Dom.text(selector)` — returns a future that resolves to the
/// `textContent` of the first matching element. Browser path
/// defers to a JS shim; wasi resolves with `MOCK_DOM_TEXT:…`.
///
/// # Safety
///
/// Reads `selector` (String) from slot 1, writes the future
/// handle (Num) to slot 0.
pub unsafe extern "C" fn dom_text(vm: *mut VM) {
    unsafe {
        let vm_ref = &mut *vm;
        let Some(selector) = read_string_slot(vm_ref, 1, "Dom.text", "selector") else {
            return;
        };
        let handle = create_pending_future();
        #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
        {
            crate::js::wlift_dom_text(handle, &selector);
        }
        #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
        {
            resolve_future(handle, format!("MOCK_DOM_TEXT:{}", selector));
        }
        write_handle(vm_ref, handle);
    }
}

/// `Dom.setText(selector, value)` — assigns `textContent` on the
/// first matching element.
///
/// # Safety
///
/// Reads `selector`, `value` (Strings); writes the future handle.
pub unsafe extern "C" fn dom_set_text(vm: *mut VM) {
    unsafe {
        let vm_ref = &mut *vm;
        let Some(selector) = read_string_slot(vm_ref, 1, "Dom.setText", "selector") else {
            return;
        };
        let Some(value) = read_string_slot(vm_ref, 2, "Dom.setText", "value") else {
            return;
        };
        let handle = create_pending_future();
        #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
        {
            crate::js::wlift_dom_set_text(handle, &selector, &value);
        }
        #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
        {
            let _ = value;
            resolve_future(handle, format!("MOCK_DOM_SET_TEXT:{}", selector));
        }
        write_handle(vm_ref, handle);
    }
}

/// `Dom.getAttribute(selector, name)` — reads an attribute on
/// the first matching element. Resolves to `""` for missing.
///
/// # Safety
///
/// Reads `selector`, `name` (Strings); writes the future handle.
pub unsafe extern "C" fn dom_get_attribute(vm: *mut VM) {
    unsafe {
        let vm_ref = &mut *vm;
        let Some(selector) = read_string_slot(vm_ref, 1, "Dom.getAttribute", "selector") else {
            return;
        };
        let Some(name) = read_string_slot(vm_ref, 2, "Dom.getAttribute", "name") else {
            return;
        };
        let handle = create_pending_future();
        #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
        {
            crate::js::wlift_dom_get_attribute(handle, &selector, &name);
        }
        #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
        {
            let _ = name;
            resolve_future(handle, format!("MOCK_DOM_GET_ATTR:{}", selector));
        }
        write_handle(vm_ref, handle);
    }
}

/// `Dom.setAttribute(selector, name, value)` — sets an attribute
/// on the first matching element.
///
/// # Safety
///
/// Reads `selector`, `name`, `value` (Strings); writes the
/// future handle.
pub unsafe extern "C" fn dom_set_attribute(vm: *mut VM) {
    unsafe {
        let vm_ref = &mut *vm;
        let Some(selector) = read_string_slot(vm_ref, 1, "Dom.setAttribute", "selector") else {
            return;
        };
        let Some(name) = read_string_slot(vm_ref, 2, "Dom.setAttribute", "name") else {
            return;
        };
        let Some(value) = read_string_slot(vm_ref, 3, "Dom.setAttribute", "value") else {
            return;
        };
        let handle = create_pending_future();
        #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
        {
            crate::js::wlift_dom_set_attribute(handle, &selector, &name, &value);
        }
        #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
        {
            let _ = (name, value);
            resolve_future(handle, format!("MOCK_DOM_SET_ATTR:{}", selector));
        }
        write_handle(vm_ref, handle);
    }
}

/// `Dom.addClass(selector, name)` — `classList.add(name)` on the
/// first matching element.
///
/// # Safety
///
/// Reads `selector`, `name` (Strings); writes the future handle.
pub unsafe extern "C" fn dom_add_class(vm: *mut VM) {
    unsafe {
        let vm_ref = &mut *vm;
        let Some(selector) = read_string_slot(vm_ref, 1, "Dom.addClass", "selector") else {
            return;
        };
        let Some(name) = read_string_slot(vm_ref, 2, "Dom.addClass", "name") else {
            return;
        };
        let handle = create_pending_future();
        #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
        {
            crate::js::wlift_dom_add_class(handle, &selector, &name);
        }
        #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
        {
            let _ = name;
            resolve_future(handle, format!("MOCK_DOM_ADD_CLASS:{}", selector));
        }
        write_handle(vm_ref, handle);
    }
}

/// `Dom.removeClass(selector, name)` — `classList.remove(name)`
/// on the first matching element.
///
/// # Safety
///
/// Reads `selector`, `name` (Strings); writes the future handle.
pub unsafe extern "C" fn dom_remove_class(vm: *mut VM) {
    unsafe {
        let vm_ref = &mut *vm;
        let Some(selector) = read_string_slot(vm_ref, 1, "Dom.removeClass", "selector") else {
            return;
        };
        let Some(name) = read_string_slot(vm_ref, 2, "Dom.removeClass", "name") else {
            return;
        };
        let handle = create_pending_future();
        #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
        {
            crate::js::wlift_dom_remove_class(handle, &selector, &name);
        }
        #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
        {
            let _ = name;
            resolve_future(handle, format!("MOCK_DOM_REMOVE_CLASS:{}", selector));
        }
        write_handle(vm_ref, handle);
    }
}

/// `Dom.queryAll(selector)` — resolves to the count of matching
/// elements as a decimal string (e.g. `"3"`). String-encoded so
/// it rides the existing `Future<String>` transport without
/// adding a numeric variant.
///
/// # Safety
///
/// Reads `selector` (String); writes the future handle.
pub unsafe extern "C" fn dom_query_all(vm: *mut VM) {
    unsafe {
        let vm_ref = &mut *vm;
        let Some(selector) = read_string_slot(vm_ref, 1, "Dom.queryAll", "selector") else {
            return;
        };
        let handle = create_pending_future();
        #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
        {
            crate::js::wlift_dom_query_all(handle, &selector);
        }
        #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
        {
            resolve_future(handle, format!("MOCK_DOM_QUERY_ALL:{}", selector));
        }
        write_handle(vm_ref, handle);
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Publish DOM foreign methods under library name `"dom"`.
pub fn register_static_symbols() {
    use wren_lift::runtime::foreign::register_plugin_symbol_unsafe;
    register_plugin_symbol_unsafe("dom", "dom_text", dom_text);
    register_plugin_symbol_unsafe("dom", "dom_set_text", dom_set_text);
    register_plugin_symbol_unsafe("dom", "dom_get_attribute", dom_get_attribute);
    register_plugin_symbol_unsafe("dom", "dom_set_attribute", dom_set_attribute);
    register_plugin_symbol_unsafe("dom", "dom_add_class", dom_add_class);
    register_plugin_symbol_unsafe("dom", "dom_remove_class", dom_remove_class);
    register_plugin_symbol_unsafe("dom", "dom_query_all", dom_query_all);
}

// ---------------------------------------------------------------------------
// Companion Wren source
// ---------------------------------------------------------------------------

/// Wren-side `Dom` shim. Embeds the same `Future` class as
/// `BROWSER_WREN`; the smoke binary inlines its own — both
/// versions stay byte-for-byte identical.
pub const DOM_WREN: &str = r#"
#!native = "dom"
foreign class DomCore {
    #!symbol = "dom_text"
    foreign static textHandle(selector)
    #!symbol = "dom_set_text"
    foreign static setTextHandle(selector, value)
    #!symbol = "dom_get_attribute"
    foreign static getAttributeHandle(selector, name)
    #!symbol = "dom_set_attribute"
    foreign static setAttributeHandle(selector, name, value)
    #!symbol = "dom_add_class"
    foreign static addClassHandle(selector, name)
    #!symbol = "dom_remove_class"
    foreign static removeClassHandle(selector, name)
    #!symbol = "dom_query_all"
    foreign static queryAllHandle(selector)
}

class Dom {
    static text(selector)                 { Future.new_(DomCore.textHandle(selector)) }
    static setText(selector, value)       { Future.new_(DomCore.setTextHandle(selector, value)) }
    static getAttribute(selector, name)   { Future.new_(DomCore.getAttributeHandle(selector, name)) }
    static setAttribute(selector, n, v)   { Future.new_(DomCore.setAttributeHandle(selector, n, v)) }
    static addClass(selector, name)       { Future.new_(DomCore.addClassHandle(selector, name)) }
    static removeClass(selector, name)    { Future.new_(DomCore.removeClassHandle(selector, name)) }
    static queryAll(selector)             { Future.new_(DomCore.queryAllHandle(selector)) }
}
"#;
