//! Browser-bridge foreign methods.
//!
//! Lives in the `wlift_wasm` crate so it can call into the
//! `_wbindgen_*` exports the wasm-bindgen runtime ships. Each
//! foreign method follows the same pattern: mint a handle via
//! [`runtime::foreign::create_pending_future`-equivalent], kick
//! off the JS-side async work, and return the handle wrapped in
//! a Wren `Future`. JS calls back into the module via the
//! exports in `lib.rs` (`resolve_future` / `reject_future`) when
//! the underlying Promise settles; the awaiting fiber's
//! `Fiber.yield` loop notices on its next tick.
//!
//! Today this module hosts:
//!
//!   * `Browser.setTimeout(ms)` — resolves to `null` after `ms`
//!     milliseconds. Drives a JS `setTimeout` in browser hosts;
//!     under wasi (the smoke test) we resolve synchronously after
//!     a fixed number of yield-loop ticks since there's no event
//!     loop to wait on. That divergence is intentional: the goal
//!     is to verify the *bridge wiring* end-to-end, and the wasi
//!     path doesn't need to actually wait — only to demonstrate
//!     that an awaiting fiber wakes up once its handle flips.
//!
//!   * `Browser.peekState(handle)` / `Browser.takeValue(handle)`
//!     — the foreign methods Wren's `Future` class uses to poll
//!     the runtime-side handle store.
//!
//! All exports are registered into the wlift foreign-method
//! registry under the library name `"browser"` and bind to the
//! `Browser` Wren foreign class declared at the bottom of this
//! file's accompanying Wren source (see `BROWSER_WREN`).

use wren_lift::runtime::object::{NativeContext, ObjString};
use wren_lift::runtime::value::Value;
use wren_lift::runtime::vm::VM;

use crate::{create_pending_future, future_state, future_store_take_value, FutureState};
// `resolve_future` is only used on the wasi/native fallback path
// where there's no event loop to wait on. The browser path goes
// through the `_wlift_*` JS shims instead.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
use crate::resolve_future;

// ---------------------------------------------------------------------------
// Foreign-method shims
// ---------------------------------------------------------------------------

/// `Browser.setTimeout(ms)` — returns a future handle (Num) that
/// resolves to `null` after `ms` milliseconds. The browser path
/// hooks into JS `setTimeout`; the wasi path resolves
/// synchronously since there's no event loop to schedule against.
///
/// # Safety
///
/// Foreign-method ABI: reads `ms` from slot 1, writes the handle
/// (or null on bad args) to slot 0.
pub unsafe extern "C" fn browser_set_timeout(vm: *mut VM) {
    unsafe {
        let vm_ref = &mut *vm;
        let arg = vm_ref.api_stack.get(1).copied().unwrap_or(Value::null());
        let ms = match arg.as_num() {
            Some(n) if n.is_finite() && n >= 0.0 => n,
            _ => {
                vm_ref.runtime_error(
                    "Browser.setTimeout: ms must be a non-negative finite number.".to_string(),
                );
                return;
            }
        };
        let handle = create_pending_future();

        // Browser: dispatch to JS via the imported `_wlift_set_timeout`.
        // Wasi: resolve synchronously — the fiber that awaits this
        // handle still has to yield once before seeing the
        // resolution, which is enough to exercise the bridge end-
        // to-end without an event loop.
        #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
        {
            // Browser host. The JS shim:
            //
            //   export function _wlift_set_timeout(handle, ms) {
            //       setTimeout(() => wlift_resolve_future(handle, ""), ms);
            //   }
            //
            // bound via wasm-bindgen `extern "C"` block in lib.rs.
            crate::js::wlift_set_timeout(handle, ms);
        }
        #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
        {
            // Wasi / native. No event loop — resolve right away.
            // The await loop still yields once because the fiber
            // checks state before the next yield, but the value
            // is already in place.
            let _ = ms;
            crate::resolve_future(handle, String::new());
        }

        if vm_ref.api_stack.is_empty() {
            vm_ref.api_stack.push(Value::num(handle as f64));
        } else {
            vm_ref.api_stack[0] = Value::num(handle as f64);
        }
    }
}

/// `Browser.fetch(url)` — returns a future handle (Num) that
/// resolves to the response body (String). Browser path drives a
/// JS `fetch(url).then(r => r.text())`; wasi path resolves
/// synchronously with a mock payload so the bridge round-trip can
/// be exercised end-to-end without a network stack.
///
/// # Safety
///
/// Foreign-method ABI: reads `url` (String) from slot 1, writes
/// the handle (or null on bad args) to slot 0.
pub unsafe extern "C" fn browser_fetch(vm: *mut VM) {
    unsafe {
        let vm_ref = &mut *vm;
        let arg = vm_ref.api_stack.get(1).copied().unwrap_or(Value::null());
        let url: String = if arg.is_string_object() {
            let ptr = arg.as_object().expect("string object pointer");
            (*(ptr as *const ObjString)).as_str().to_string()
        } else {
            vm_ref.runtime_error("Browser.fetch: url must be a String.".to_string());
            return;
        };
        let handle = create_pending_future();

        #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
        {
            // Browser host. The JS shim:
            //
            //   export function _wlift_fetch(handle, url) {
            //       fetch(url)
            //         .then(r => r.text())
            //         .then(t => wlift_resolve_future(handle, t))
            //         .catch(e => wlift_reject_future(handle, String(e)));
            //   }
            //
            // bound via wasm-bindgen `extern "C"` block in lib.rs.
            crate::js::wlift_fetch(handle, &url);
        }
        #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
        {
            // Wasi / native. No fetch implementation — resolve with
            // a deterministic mock so tests can verify the bridge.
            // Real HTTP under wasi would need a wasi-http binding;
            // out of scope for the bridge smoke test.
            resolve_future(handle, format!("MOCK:{}", url));
        }

        if vm_ref.api_stack.is_empty() {
            vm_ref.api_stack.push(Value::num(handle as f64));
        } else {
            vm_ref.api_stack[0] = Value::num(handle as f64);
        }
    }
}

/// `Browser.peekState(handle)` — returns the future's current
/// state as a Num: 0 pending / 1 resolved / 2 rejected. Used by
/// `Future.await`'s `Fiber.yield` loop.
///
/// # Safety
///
/// Reads the handle from slot 1; writes the state to slot 0.
pub unsafe extern "C" fn browser_peek_state(vm: *mut VM) {
    unsafe {
        let vm_ref = &mut *vm;
        let arg = vm_ref.api_stack.get(1).copied().unwrap_or(Value::null());
        let handle = match arg.as_num() {
            Some(n) if n.is_finite() && n >= 0.0 => n as u32,
            _ => {
                vm_ref.runtime_error(
                    "Browser.peekState: handle must be a non-negative integer.".to_string(),
                );
                return;
            }
        };
        let state = match future_state(handle) {
            FutureState::Pending => 0.0,
            FutureState::Resolved => 1.0,
            FutureState::Rejected => 2.0,
        };
        if vm_ref.api_stack.is_empty() {
            vm_ref.api_stack.push(Value::num(state));
        } else {
            vm_ref.api_stack[0] = Value::num(state);
        }
    }
}

/// `Browser.takeValue(handle)` — returns the resolved value (a
/// String) or rejection error (also a String) and removes the
/// slot from the handle store. Wren's `Future.await` calls this
/// only after `peekState` returns non-zero.
///
/// # Safety
///
/// Reads the handle from slot 1; writes the result string (or
/// null) to slot 0.
pub unsafe extern "C" fn browser_take_value(vm: *mut VM) {
    unsafe {
        let vm_ref = &mut *vm;
        let arg = vm_ref.api_stack.get(1).copied().unwrap_or(Value::null());
        let handle = match arg.as_num() {
            Some(n) if n.is_finite() && n >= 0.0 => n as u32,
            _ => {
                vm_ref.runtime_error(
                    "Browser.takeValue: handle must be a non-negative integer.".to_string(),
                );
                return;
            }
        };
        let value = future_store_take_value(handle);
        let result = match value {
            Some(s) => vm_ref.new_string(s),
            None => Value::null(),
        };
        if vm_ref.api_stack.is_empty() {
            vm_ref.api_stack.push(result);
        } else {
            vm_ref.api_stack[0] = result;
        }
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Publish the `Browser.*` foreign methods to the runtime
/// registry. Called from `register_static_plugins` in `lib.rs`.
pub fn register_static_symbols() {
    use wren_lift::runtime::foreign::register_plugin_symbol_unsafe;
    register_plugin_symbol_unsafe("browser", "browser_set_timeout", browser_set_timeout);
    register_plugin_symbol_unsafe("browser", "browser_fetch", browser_fetch);
    register_plugin_symbol_unsafe("browser", "browser_peek_state", browser_peek_state);
    register_plugin_symbol_unsafe("browser", "browser_take_value", browser_take_value);
}

// ---------------------------------------------------------------------------
// Companion Wren source
// ---------------------------------------------------------------------------

/// Wren-side `Browser` + `Future` shim. The smoke binary embeds
/// this directly via `vm.interpret("browser", BROWSER_WREN)`.
/// Browser hosts can either embed it the same way or import it
/// from a real `@hatch:browser` package — same source either way.
pub const BROWSER_WREN: &str = r#"
// Foreign-method bridge to JS Promise-shaped APIs. Wren code
// awaits a Future; the JS side eventually flips the handle's
// state via `wlift_wasm.resolve_future(handle, value)`.
//
//   var data = Browser.fetch("...").await    // Phase 1.2+
//   Browser.setTimeout(50).await             // works today

#!native = "browser"
foreign class BrowserCore {
    #!symbol = "browser_set_timeout"
    foreign static setTimeoutHandle(ms)
    #!symbol = "browser_fetch"
    foreign static fetchHandle(url)
    #!symbol = "browser_peek_state"
    foreign static peekState(handle)
    #!symbol = "browser_take_value"
    foreign static takeValue(handle)
}

class Future {
    construct new_(handle) { _h = handle }
    handle { _h }
    state  { BrowserCore.peekState(_h) }
    await {
        // 0 = pending, 1 = resolved, 2 = rejected. Cache the
        // settled state BEFORE `takeValue` removes the slot —
        // a subsequent `peekState` on a missing handle reports
        // Rejected, which would mis-fire `Fiber.abort` on a
        // resolved Future.
        while (state == 0) Fiber.yield()
        var settled = state
        var v = BrowserCore.takeValue(_h)
        if (settled == 2) Fiber.abort(v == null ? "Future rejected" : v)
        return v
    }
}

class Browser {
    static setTimeout(ms) { Future.new_(BrowserCore.setTimeoutHandle(ms)) }
    static fetch(url)     { Future.new_(BrowserCore.fetchHandle(url)) }
}
"#;
