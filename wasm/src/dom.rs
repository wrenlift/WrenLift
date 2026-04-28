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
//! Today we ship two methods — `Dom.text(selector)` and
//! `Dom.setText(selector, value)`. They're the smallest pair that
//! exercises both directions of the bridge (read returns a value;
//! write returns `null`). Adding more (`appendChild`,
//! `setAttribute`, `addEventListener`) follows the same template.

use wren_lift::runtime::object::{NativeContext, ObjString};
use wren_lift::runtime::value::Value;
use wren_lift::runtime::vm::VM;

use crate::create_pending_future;
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
use crate::resolve_future;

// ---------------------------------------------------------------------------
// Foreign-method shims
// ---------------------------------------------------------------------------

/// `Dom.text(selector)` — returns a future that resolves to the
/// `textContent` of the first element matching `selector`, or
/// `""` when nothing matches. Browser path defers to a JS shim;
/// wasi resolves with `MOCK_DOM_TEXT:<selector>` so the smoke
/// test can verify the round-trip.
///
/// # Safety
///
/// Foreign-method ABI: reads `selector` (String) from slot 1,
/// writes the future handle (Num) to slot 0.
pub unsafe extern "C" fn dom_text(vm: *mut VM) {
    unsafe {
        let vm_ref = &mut *vm;
        let arg = vm_ref.api_stack.get(1).copied().unwrap_or(Value::null());
        let selector: String = if arg.is_string_object() {
            let ptr = arg.as_object().expect("string object pointer");
            (*(ptr as *const ObjString)).as_str().to_string()
        } else {
            vm_ref.runtime_error("Dom.text: selector must be a String.".to_string());
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

        if vm_ref.api_stack.is_empty() {
            vm_ref.api_stack.push(Value::num(handle as f64));
        } else {
            vm_ref.api_stack[0] = Value::num(handle as f64);
        }
    }
}

/// `Dom.setText(selector, value)` — assigns `textContent` on the
/// first matching element. Returns a future that resolves to the
/// empty string once the write lands (gives the awaiting fiber
/// something to synchronise on, in case the worker→main hop is
/// the slow leg). Wasi mocks with `MOCK_DOM_SET_TEXT:<selector>`.
///
/// # Safety
///
/// Reads `selector` (String) from slot 1, `value` (String) from
/// slot 2; writes the future handle (Num) to slot 0.
pub unsafe extern "C" fn dom_set_text(vm: *mut VM) {
    unsafe {
        let vm_ref = &mut *vm;
        let sel_arg = vm_ref.api_stack.get(1).copied().unwrap_or(Value::null());
        let val_arg = vm_ref.api_stack.get(2).copied().unwrap_or(Value::null());
        let selector: String = if sel_arg.is_string_object() {
            let ptr = sel_arg.as_object().expect("string object pointer");
            (*(ptr as *const ObjString)).as_str().to_string()
        } else {
            vm_ref.runtime_error("Dom.setText: selector must be a String.".to_string());
            return;
        };
        let value: String = if val_arg.is_string_object() {
            let ptr = val_arg.as_object().expect("string object pointer");
            (*(ptr as *const ObjString)).as_str().to_string()
        } else {
            vm_ref.runtime_error("Dom.setText: value must be a String.".to_string());
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

        if vm_ref.api_stack.is_empty() {
            vm_ref.api_stack.push(Value::num(handle as f64));
        } else {
            vm_ref.api_stack[0] = Value::num(handle as f64);
        }
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Publish DOM foreign methods under library name `"dom"`. Wren
/// `foreign class` declarations annotated with `#!native = "dom"`
/// resolve here.
pub fn register_static_symbols() {
    use wren_lift::runtime::foreign::register_plugin_symbol_unsafe;
    register_plugin_symbol_unsafe("dom", "dom_text", dom_text);
    register_plugin_symbol_unsafe("dom", "dom_set_text", dom_set_text);
}

// ---------------------------------------------------------------------------
// Companion Wren source
// ---------------------------------------------------------------------------

/// Wren-side `Dom` shim. Embeds the same `Future` class as
/// `BROWSER_WREN`; in a real package install only one copy ships
/// per VM, but the smoke binary inlines its own — both versions
/// stay byte-for-byte identical so they drop into either spot.
pub const DOM_WREN: &str = r#"
#!native = "dom"
foreign class DomCore {
    #!symbol = "dom_text"
    foreign static textHandle(selector)
    #!symbol = "dom_set_text"
    foreign static setTextHandle(selector, value)
}

class Dom {
    static text(selector)              { Future.new_(DomCore.textHandle(selector)) }
    static setText(selector, value)    { Future.new_(DomCore.setTextHandle(selector, value)) }
}
"#;
