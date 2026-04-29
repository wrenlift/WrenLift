//! `wlift_hello` — a deliberately trivial plugin that exists to
//! prove out the on-demand wasm plugin-loader pipeline before
//! tackling the heavier `wlift_window` / `wlift_gpu` ports.
//!
//! Architecture
//! ------------
//!
//! Plugins compiled to `wasm32-unknown-unknown` ship as their own
//! `.wasm` modules, packed into a `.hatch` bundle's `NativeLib`
//! section under the `wasm` key. The host runtime (`wlift_wasm`)
//! instantiates the module at hatch-install time, links the
//! plugin's imports against host-provided wasm bridges, and
//! registers the plugin's exported foreign-method shims into the
//! foreign-method registry.
//!
//! The plugin's pointers live in *its own* linear memory, distinct
//! from the host's. Crossing that boundary directly with C strings
//! does not work — `wrenSetSlotString(*const c_char)` would read
//! past the host's allocations. So instead of importing
//! `wrenSetSlotString` from the host, the plugin imports a small
//! set of higher-level JS-mediated bridges (`wlift_set_slot_str`
//! / `wlift_get_slot_str`) that perform the cross-memory copy on
//! the host side before invoking the real C API. JS already
//! brokers the host ↔ plugin instantiation, so it's the natural
//! place for the marshalling.
//!
//! Build
//! -----
//!
//!   cargo build --target wasm32-unknown-unknown -p wlift_hello --release
//!
//! The output `target/wasm32-unknown-unknown/release/wlift_hello.wasm`
//! is the artefact `[native_libs]` references with `wasm = "..."`.

use core::ffi::c_void;

// --- Host-provided imports -------------------------------------
// Each fn lives in the implicit `env` module, which the JS-side
// loader provides at instantiation time. The host signs these
// imports as `i32`-only types so the wasm linker doesn't need
// to know about Rust pointer ABI; the underlying values are still
// addresses + lengths in the plugin's linear memory.
#[link(wasm_import_module = "env")]
extern "C" {
    /// Read the string at `slot` on `vm`'s slot stack into the
    /// plugin's linear memory at `out`. Up to `max` bytes are
    /// written; returns the actual number written (or `-1` if
    /// the slot is not a string). The string is *not*
    /// null-terminated.
    fn wlift_get_slot_str(vm: *mut c_void, slot: i32, out: *mut u8, max: i32) -> i32;

    /// Copy `len` bytes from the plugin's memory at `ptr` into
    /// the host, allocate a Wren string, and store it at `slot`
    /// on `vm`'s slot stack.
    fn wlift_set_slot_str(vm: *mut c_void, slot: i32, ptr: *const u8, len: i32);
}

// --- Foreign methods exposed to Wren ---------------------------

/// `Hello.greet(_)` — reads a name from slot 1, writes
/// `"hello, <name>"` (or `"hello, stranger"` if the slot wasn't
/// a string) into slot 0.
///
/// The export name is what the loader looks up against the
/// foreign-method registry; declared in the Wren wrapper as
/// `#!symbol = "wlift_hello_greet"` on the foreign-method body.
#[no_mangle]
pub extern "C" fn wlift_hello_greet(vm: *mut c_void) {
    // Stack-allocated read buffer. 256 bytes is plenty for
    // demo names; longer strings get truncated, which is a
    // demo-grade limitation worth noting in the comments above
    // rather than something to engineer around.
    let mut name_buf = [0u8; 256];
    let n = unsafe { wlift_get_slot_str(vm, 1, name_buf.as_mut_ptr(), name_buf.len() as i32) };

    let mut out: Vec<u8> = Vec::with_capacity(64);
    out.extend_from_slice(b"hello, ");
    if n <= 0 {
        out.extend_from_slice(b"stranger");
    } else {
        out.extend_from_slice(&name_buf[..n as usize]);
    }
    unsafe {
        wlift_set_slot_str(vm, 0, out.as_ptr(), out.len() as i32);
    }
}
