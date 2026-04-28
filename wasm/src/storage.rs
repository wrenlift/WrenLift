//! `localStorage` / `sessionStorage` bridge.
//!
//! Both Web Storage APIs are available in **Window** and
//! **WorkerGlobalScope** scopes — unlike `document`, the Storage
//! interface is exposed to dedicated workers in every shipping
//! browser. That means worker mode can call them directly
//! without the postMessage round-trip we use for DOM ops, and a
//! single inline-resolving shim covers both worker and main
//! mode. Operations are synchronous, so the future settles
//! before the foreign method returns to Wren — no scheduler
//! needed.
//!
//! Surface today (read/write/remove the obvious things; keys
//! follows the same template if a list-of-keys consumer arrives):
//!
//!   * `LocalStorage.get(key)`            → `String` | `""` if missing
//!   * `LocalStorage.set(key, value)`     → `""`
//!   * `LocalStorage.remove(key)`         → `""`
//!   * `LocalStorage.clear`               → `""`
//!   * `SessionStorage.*` — same shape, different backing store.
//!
//! The Rust foreign method takes `scope` (`"local"` | `"session"`)
//! as its first arg so we don't pay for two near-identical
//! method tables. Wren-side `LocalStorage` / `SessionStorage`
//! classes hard-code the scope so callers don't see it.

use wren_lift::runtime::object::{NativeContext, ObjString};
use wren_lift::runtime::value::Value;
use wren_lift::runtime::vm::VM;

use crate::create_pending_future;
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
use crate::resolve_future;

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

/// `Storage.get(scope, key)` — reads `scope`Storage[key]; returns
/// `""` when missing.
///
/// # Safety
///
/// Reads `scope`, `key` (Strings) from slots 1 and 2; writes the
/// future handle (Num) to slot 0.
pub unsafe extern "C" fn storage_get(vm: *mut VM) {
    unsafe {
        let vm_ref = &mut *vm;
        let Some(scope) = read_string_slot(vm_ref, 1, "Storage.get", "scope") else {
            return;
        };
        let Some(key) = read_string_slot(vm_ref, 2, "Storage.get", "key") else {
            return;
        };
        let handle = create_pending_future();
        #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
        {
            crate::js::wlift_storage_get(handle, &scope, &key);
        }
        #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
        {
            resolve_future(handle, format!("MOCK_STORAGE_GET:{}:{}", scope, key));
        }
        write_handle(vm_ref, handle);
    }
}

/// `Storage.set(scope, key, value)` — writes `scope`Storage[key]
/// = value.
///
/// # Safety
///
/// Reads `scope`, `key`, `value` (Strings); writes the future
/// handle.
pub unsafe extern "C" fn storage_set(vm: *mut VM) {
    unsafe {
        let vm_ref = &mut *vm;
        let Some(scope) = read_string_slot(vm_ref, 1, "Storage.set", "scope") else {
            return;
        };
        let Some(key) = read_string_slot(vm_ref, 2, "Storage.set", "key") else {
            return;
        };
        let Some(value) = read_string_slot(vm_ref, 3, "Storage.set", "value") else {
            return;
        };
        let handle = create_pending_future();
        #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
        {
            crate::js::wlift_storage_set(handle, &scope, &key, &value);
        }
        #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
        {
            let _ = value;
            resolve_future(handle, format!("MOCK_STORAGE_SET:{}:{}", scope, key));
        }
        write_handle(vm_ref, handle);
    }
}

/// `Storage.remove(scope, key)` — `scope`Storage.removeItem(key).
///
/// # Safety
///
/// Reads `scope`, `key` (Strings); writes the future handle.
pub unsafe extern "C" fn storage_remove(vm: *mut VM) {
    unsafe {
        let vm_ref = &mut *vm;
        let Some(scope) = read_string_slot(vm_ref, 1, "Storage.remove", "scope") else {
            return;
        };
        let Some(key) = read_string_slot(vm_ref, 2, "Storage.remove", "key") else {
            return;
        };
        let handle = create_pending_future();
        #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
        {
            crate::js::wlift_storage_remove(handle, &scope, &key);
        }
        #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
        {
            resolve_future(handle, format!("MOCK_STORAGE_REMOVE:{}:{}", scope, key));
        }
        write_handle(vm_ref, handle);
    }
}

/// `Storage.clear(scope)` — `scope`Storage.clear().
///
/// # Safety
///
/// Reads `scope` (String); writes the future handle.
pub unsafe extern "C" fn storage_clear(vm: *mut VM) {
    unsafe {
        let vm_ref = &mut *vm;
        let Some(scope) = read_string_slot(vm_ref, 1, "Storage.clear", "scope") else {
            return;
        };
        let handle = create_pending_future();
        #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
        {
            crate::js::wlift_storage_clear(handle, &scope);
        }
        #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
        {
            resolve_future(handle, format!("MOCK_STORAGE_CLEAR:{}", scope));
        }
        write_handle(vm_ref, handle);
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Publish Storage foreign methods under library name `"storage"`.
pub fn register_static_symbols() {
    use wren_lift::runtime::foreign::register_plugin_symbol_unsafe;
    register_plugin_symbol_unsafe("storage", "storage_get", storage_get);
    register_plugin_symbol_unsafe("storage", "storage_set", storage_set);
    register_plugin_symbol_unsafe("storage", "storage_remove", storage_remove);
    register_plugin_symbol_unsafe("storage", "storage_clear", storage_clear);
}
