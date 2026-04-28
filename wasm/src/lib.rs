//! wasm-bindgen entry shim for the WrenLift interpreter.
//!
//! Three things exported to JS / wasi hosts:
//!
//!   * `version()` — returns a `&str` describing the wlift build,
//!     useful as a smoke test that the wasm module loaded at all.
//!   * `run(source: &str) -> RunResult` — compiles the source as a
//!     `<wasm>` module, drives the BC interpreter, and returns the
//!     captured `System.print` output plus an `ok` flag.
//!   * `resolve_future` / `reject_future` — host-side handle store
//!     for the Promise ↔ `Fiber.yield` bridge. JS Promise callbacks
//!     flip a handle's state; the awaiting fiber wakes on its next
//!     yield-loop iteration. See
//!     `project_wasm_promise_fiber_bridge.md` for the full design.
//!
//! No JIT, no plugin loading, no fs / sockets. Future phases grow
//! the surface (incremental compile, multi-module install,
//! browser-API foreign classes); this is the minimal demo.

use std::collections::HashMap;
use std::sync::Mutex;

use wasm_bindgen::prelude::*;

use wren_lift::runtime::engine::{ExecutionMode, InterpretResult};
use wren_lift::runtime::vm::{VMConfig, VM};

pub mod browser;

// `js` carries the `extern "C"` block that imports browser-side
// shims (today: `_wlift_set_timeout`). Compiled only on the
// `wasm32-unknown-unknown` target, which is the only one where
// wasm-bindgen's JS interop applies. Wasi (the smoke target)
// short-circuits the same paths to a synchronous resolution.
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
pub mod js {
    use wasm_bindgen::prelude::*;

    // Shims live on `globalThis` so the embedding page (or worker)
    // provides them by simple assignment, e.g.
    //
    //   globalThis._wlift_fetch = (h, url) => { fetch(url)... };
    //
    // before `init()`. No `module = "..."` means we don't have to
    // ship a separate JS file alongside `pkg/`; the host wires
    // these up however it likes (inline `<script>`, an external
    // module, etc.).
    #[wasm_bindgen]
    extern "C" {
        /// JS-provided shim:
        ///
        ///   globalThis._wlift_set_timeout = (handle, ms) => {
        ///       setTimeout(() => resolve_future(handle, ""), ms);
        ///   };
        ///
        /// `resolve_future` is exported from this same wasm module
        /// (`pkg/wlift_wasm.js`); the host imports it and closes
        /// over it before installing the shim.
        #[wasm_bindgen(js_namespace = globalThis, js_name = _wlift_set_timeout)]
        pub fn wlift_set_timeout(handle: u32, ms: f64);

        /// JS-provided shim:
        ///
        ///   globalThis._wlift_fetch = (handle, url) => {
        ///       fetch(url)
        ///         .then(r => r.text())
        ///         .then(t => resolve_future(handle, t))
        ///         .catch(e => reject_future(handle, String(e)));
        ///   };
        ///
        /// Same wiring as `_wlift_set_timeout`.
        #[wasm_bindgen(js_namespace = globalThis, js_name = _wlift_fetch)]
        pub fn wlift_fetch(handle: u32, url: &str);
    }
}

// ---------------------------------------------------------------------------
// Promise ↔ Fiber.yield handle store (Phase 1.1 scaffolding)
// ---------------------------------------------------------------------------
//
// Foreign methods that call browser async APIs (`fetch`,
// `setTimeout`, `WebSocket.send`, ...) mint a handle here, kick
// off the JS Promise, and return the handle to Wren wrapped in a
// `Future` instance. JS resolves/rejects via the
// `resolve_future` / `reject_future` exports below; the Wren
// `Future.await` method polls `peek_state` in a `Fiber.yield`
// loop until the handle flips out of `Pending`.
//
// The store is a global Mutex<HashMap> for now. Multi-VM hosts
// (server-side wasi running multiple wlift instances) eventually
// want a per-VM store, but that's a refactor with no consumers
// in Phase 1.1 — defer until at least one host ships multiple
// concurrent VMs.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[wasm_bindgen]
pub enum FutureState {
    Pending = 0,
    Resolved = 1,
    Rejected = 2,
}

#[derive(Debug)]
struct FutureSlot {
    state: FutureState,
    /// Resolved value, currently encoded as a UTF-8 string (JSON
    /// for fetch payloads, plain text for echo APIs). Switching
    /// to a richer encoding (e.g. wasm-bindgen `JsValue` round-
    /// trip via the wlift heap) lands when there's a second
    /// consumer that actually needs it.
    value: Option<String>,
    /// Error message on `Rejected`. Surfaced to Wren via
    /// `Fiber.abort` inside `Future.await`.
    error: Option<String>,
}

fn future_store() -> &'static Mutex<HashMap<u32, FutureSlot>> {
    static STORE: std::sync::OnceLock<Mutex<HashMap<u32, FutureSlot>>> = std::sync::OnceLock::new();
    STORE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn next_future_handle() -> u32 {
    static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(1);
    NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// Internal — Phase 1.1+ foreign methods call this to mint a
/// handle and pre-stash a `Pending` slot. Public so the future
/// `core::browser` foreign-method registry (lives in this crate
/// for now) can call it directly without going through the JS
/// host.
pub fn create_pending_future() -> u32 {
    let handle = next_future_handle();
    let mut store = future_store().lock().expect("future store poisoned");
    store.insert(
        handle,
        FutureSlot {
            state: FutureState::Pending,
            value: None,
            error: None,
        },
    );
    handle
}

/// JS-callable: flip a handle to `Resolved` and stash the value.
/// Idempotent — resolving an already-resolved/rejected handle is
/// a no-op (matches Promise semantics; the first settlement wins).
#[wasm_bindgen]
pub fn resolve_future(handle: u32, value: String) {
    let mut store = future_store().lock().expect("future store poisoned");
    if let Some(slot) = store.get_mut(&handle) {
        if slot.state == FutureState::Pending {
            slot.state = FutureState::Resolved;
            slot.value = Some(value);
        }
    }
}

/// JS-callable: flip a handle to `Rejected` with an error message.
#[wasm_bindgen]
pub fn reject_future(handle: u32, error: String) {
    let mut store = future_store().lock().expect("future store poisoned");
    if let Some(slot) = store.get_mut(&handle) {
        if slot.state == FutureState::Pending {
            slot.state = FutureState::Rejected;
            slot.error = Some(error);
        }
    }
}

/// JS-callable: read the current state of a handle. Used by
/// JS-side debugging / introspection tooling AND by the
/// `browser::browser_peek_state` foreign method that Wren's
/// `Future.await` polls in its yield loop.
#[wasm_bindgen]
pub fn future_state(handle: u32) -> FutureState {
    let store = future_store().lock().expect("future store poisoned");
    store
        .get(&handle)
        .map(|s| s.state)
        .unwrap_or(FutureState::Rejected)
}

/// Take the resolved value (or rejection error) out of a slot
/// and drop the slot. Called from `browser::browser_take_value`
/// after `peekState` reports non-pending — the slot is one-shot
/// from the Wren side, matching Promise's "settled once" rule.
///
/// Returns `None` if the handle was never registered, was
/// already taken, or hasn't settled yet (the await loop should
/// only call this after `peekState != 0`).
pub fn future_store_take_value(handle: u32) -> Option<String> {
    let mut store = future_store().lock().expect("future store poisoned");
    let slot = store.remove(&handle)?;
    match slot.state {
        FutureState::Pending => None,
        FutureState::Resolved => slot.value,
        FutureState::Rejected => slot.error,
    }
}

// ---------------------------------------------------------------------------
// Init + version
// ---------------------------------------------------------------------------

/// One-time setup. The wasm-bindgen `start` attribute makes this
/// run on module instantiation, before any JS-side `run()` /
/// `version()` call.
///
/// Two jobs:
///
///   * Install `console_error_panic_hook` so panics surface as a
///     readable JS `console.error` instead of the bare
///     "unreachable executed" trap.
///
///   * Publish every statically-linked plugin's foreign-method
///     symbols to the runtime's foreign-method registry. This is
///     the wasm replacement for the host's `libloading::dlsym`
///     pass — Wren `foreign class` declarations annotated with
///     `#!native = "wlift_image"` look up here at install time.
#[wasm_bindgen(start)]
pub fn _wasm_init() {
    #[cfg(target_arch = "wasm32")]
    {
        console_error_panic_hook::set_once();
        register_static_plugins();
    }
}

/// Static-link plugin registry. Wasi smoke binaries call this from
/// `main` before they spin up a VM. Browser builds get it via the
/// `_wasm_init` shim above.
#[cfg(target_arch = "wasm32")]
pub fn register_static_plugins() {
    wlift_image::register_static_symbols();
    browser::register_static_symbols();
}

/// Host-side stub so non-wasm callers (the smoke `_start` binary
/// running on a native target via `cargo run -p wlift_wasm --bin
/// smoke` for local debugging — outside the wasi target) still
/// compile. The host wlift loads `libwlift_image.dylib` /
/// `.so` / `.dll` via dlsym; this no-ops there.
#[cfg(not(target_arch = "wasm32"))]
pub fn register_static_plugins() {}

/// Build identifier — hard-coded for the moment so JS can sanity-
/// check the loaded wasm matches what its bundler thought it was
/// importing.
#[wasm_bindgen]
pub fn version() -> String {
    format!(
        "wlift_wasm {} (interpreter-only; wasm32 build)",
        env!("CARGO_PKG_VERSION")
    )
}

/// Result of a single `run` call, exported to JS as a structural
/// object via `wasm_bindgen`'s getter convention.
#[wasm_bindgen]
pub struct RunResult {
    output: String,
    ok: bool,
    error_kind: ErrorKind,
}

#[wasm_bindgen]
#[derive(Clone, Copy)]
pub enum ErrorKind {
    None,
    CompileError,
    RuntimeError,
}

#[wasm_bindgen]
impl RunResult {
    #[wasm_bindgen(getter)]
    pub fn output(&self) -> String {
        self.output.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn ok(&self) -> bool {
        self.ok
    }

    #[wasm_bindgen(getter, js_name = errorKind)]
    pub fn error_kind(&self) -> ErrorKind {
        self.error_kind
    }
}

/// Compile + interpret a Wren source string. Output captured into
/// the returned `RunResult`. The VM lives only for the duration
/// of this call — every `run` is a fresh interpreter, no shared
/// state across calls. (A multi-call API that holds onto a `VM`
/// instance lands in Phase 1.1.)
#[wasm_bindgen]
pub fn run(source: &str) -> RunResult {
    let mut vm = VM::new(VMConfig {
        // No JIT on wasm — gated out at compile time, but the mode
        // gate also short-circuits all the tier-up bookkeeping the
        // BC interpreter would otherwise pay per call.
        execution_mode: ExecutionMode::Interpreter,
        ..VMConfig::default()
    });
    vm.output_buffer = Some(String::new());

    let result = vm.interpret("main", source);
    let output = vm.take_output();

    let (ok, error_kind) = match result {
        InterpretResult::Success => (true, ErrorKind::None),
        InterpretResult::CompileError => (false, ErrorKind::CompileError),
        InterpretResult::RuntimeError => (false, ErrorKind::RuntimeError),
    };

    RunResult {
        output,
        ok,
        error_kind,
    }
}
