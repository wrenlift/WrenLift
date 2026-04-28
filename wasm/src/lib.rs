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

use std::collections::{HashMap, VecDeque};
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

        /// JS-provided shim:
        ///
        ///   globalThis._wlift_ws_open = (handle, url) => {
        ///       const ws = new WebSocket(url);
        ///       ws.addEventListener("open",    () => ws_open(handle));
        ///       ws.addEventListener("message", (e) => ws_message(handle, String(e.data)));
        ///       ws.addEventListener("close",   () => ws_close(handle));
        ///       ws.addEventListener("error",   () => ws_close(handle));
        ///       globalThis.__wlift_sockets ??= new Map();
        ///       globalThis.__wlift_sockets.set(handle, ws);
        ///   };
        ///
        /// The page keeps a `Map<handle, WebSocket>` so subsequent
        /// `_wlift_ws_send` / `_wlift_ws_close` calls can find the
        /// underlying socket.
        #[wasm_bindgen(js_namespace = globalThis, js_name = _wlift_ws_open)]
        pub fn wlift_ws_open(handle: u32, url: &str);

        /// JS-provided shim:
        ///
        ///   globalThis._wlift_ws_send = (handle, text) => {
        ///       globalThis.__wlift_sockets?.get(handle)?.send(text);
        ///   };
        #[wasm_bindgen(js_namespace = globalThis, js_name = _wlift_ws_send)]
        pub fn wlift_ws_send(handle: u32, text: &str);

        /// JS-provided shim:
        ///
        ///   globalThis._wlift_ws_close = (handle) => {
        ///       const ws = globalThis.__wlift_sockets?.get(handle);
        ///       if (ws) { ws.close(); globalThis.__wlift_sockets.delete(handle); }
        ///   };
        #[wasm_bindgen(js_namespace = globalThis, js_name = _wlift_ws_close)]
        pub fn wlift_ws_close(handle: u32);
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
// WebSocket handle store (Phase 1.2)
// ---------------------------------------------------------------------------
//
// `Browser.connect(url)` mints a *connection* handle here (not a
// future), then returns it wrapped in a Wren `WebSocket` instance.
// Subsequent `send` / `recv` / `close` foreign methods drive the
// connection through this store.
//
// The contract differs from `FutureSlot` because a socket is
// multi-shot — many messages flow over one handle, each delivered
// to the next pending `recv()` future or buffered for a later
// caller. The store therefore holds:
//
//   * `state` — Connecting → Open → Closed lifecycle.
//   * `inbox` — messages that arrived with no waiter ready.
//   * `waiters` — future handles parked by `recv()`. The next
//                 inbound message wakes the head waiter.
//
// Browser-host JS wires its `WebSocket` listeners to call
// `ws_open` / `ws_message` / `ws_close` below. Wasi has no real
// socket; the wasi smoke runs a loopback (send → inbox) so the
// bridge can be exercised without a network stack.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[wasm_bindgen]
pub enum WebSocketState {
    Connecting = 0,
    Open = 1,
    Closed = 2,
}

#[derive(Debug)]
struct WebSocketSlot {
    state: WebSocketState,
    inbox: VecDeque<String>,
    waiters: VecDeque<u32>,
}

fn websocket_store() -> &'static Mutex<HashMap<u32, WebSocketSlot>> {
    static STORE: std::sync::OnceLock<Mutex<HashMap<u32, WebSocketSlot>>> =
        std::sync::OnceLock::new();
    STORE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn next_websocket_handle() -> u32 {
    static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(1);
    NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// Mint a fresh socket handle in `Connecting` state. Foreign
/// methods call this before kicking off the JS-side `WebSocket`
/// constructor (browser) or flipping the state synchronously
/// (wasi loopback).
pub fn create_websocket() -> u32 {
    let handle = next_websocket_handle();
    let mut store = websocket_store().lock().expect("websocket store poisoned");
    store.insert(
        handle,
        WebSocketSlot {
            state: WebSocketState::Connecting,
            inbox: VecDeque::new(),
            waiters: VecDeque::new(),
        },
    );
    handle
}

/// JS-callable: flip a socket to `Open`. Browser hosts call this
/// from the `WebSocket`'s `open` listener so subsequent `send`
/// calls aren't queued in JS-land for nothing.
#[wasm_bindgen]
pub fn ws_open(handle: u32) {
    let mut store = websocket_store().lock().expect("websocket store poisoned");
    if let Some(slot) = store.get_mut(&handle) {
        if slot.state == WebSocketState::Connecting {
            slot.state = WebSocketState::Open;
        }
    }
}

/// JS-callable: deliver an incoming message. Wakes the head
/// `recv()` future if one is parked; otherwise buffers the
/// message for a later caller. Drops the message silently if the
/// handle is unknown — late arrivals on a closed/torn-down socket
/// shouldn't crash the page.
#[wasm_bindgen]
pub fn ws_message(handle: u32, msg: String) {
    let waiter = {
        let mut store = websocket_store().lock().expect("websocket store poisoned");
        let Some(slot) = store.get_mut(&handle) else {
            return;
        };
        if let Some(w) = slot.waiters.pop_front() {
            Some(w)
        } else {
            slot.inbox.push_back(msg.clone());
            None
        }
    };
    if let Some(w) = waiter {
        resolve_future(w, msg);
    }
}

/// JS-callable: flip a socket to `Closed` and reject any parked
/// `recv()` futures. Idempotent — closing twice is a no-op.
#[wasm_bindgen]
pub fn ws_close(handle: u32) {
    // Drain the waiter list under the store lock, then settle the
    // futures outside it — `reject_future` reaches into the future
    // store, and re-entering the websocket lock isn't necessary.
    let waiters: Vec<u32> = {
        let mut store = websocket_store().lock().expect("websocket store poisoned");
        match store.get_mut(&handle) {
            Some(slot) if slot.state != WebSocketState::Closed => {
                slot.state = WebSocketState::Closed;
                slot.waiters.drain(..).collect()
            }
            _ => Vec::new(),
        }
    };
    for w in waiters {
        reject_future(w, "WebSocket closed".to_string());
    }
}

/// JS-callable: read the current state of a socket. Used by JS
/// debug tooling and by `browser::browser_ws_state` if a Wren-
/// side state probe ends up wanted.
#[wasm_bindgen]
pub fn websocket_state(handle: u32) -> WebSocketState {
    let store = websocket_store().lock().expect("websocket store poisoned");
    store
        .get(&handle)
        .map(|s| s.state)
        .unwrap_or(WebSocketState::Closed)
}

/// Internal: enqueue a message into a socket's inbox without going
/// through the JS event path. The wasi mock uses this to loopback
/// `send()` straight back to `recv()` — there's no real socket on
/// wasi, but the bridge wiring (handle → future → fiber resume)
/// still gets exercised end-to-end.
pub fn websocket_inbox_push(handle: u32, msg: String) {
    ws_message(handle, msg);
}

/// Internal: park a future handle on a socket's waiter queue, or
/// resolve it immediately if the inbox already has a message.
/// Returns the future handle; callers wrap it in a Wren `Future`.
pub fn websocket_recv(handle: u32) -> u32 {
    let future = create_pending_future();
    let immediate = {
        let mut store = websocket_store().lock().expect("websocket store poisoned");
        match store.get_mut(&handle) {
            Some(slot) if slot.state == WebSocketState::Closed => Some(Err(())),
            Some(slot) => match slot.inbox.pop_front() {
                Some(msg) => Some(Ok(msg)),
                None => {
                    slot.waiters.push_back(future);
                    None
                }
            },
            None => Some(Err(())),
        }
    };
    match immediate {
        Some(Ok(msg)) => resolve_future(future, msg),
        Some(Err(())) => reject_future(future, "WebSocket closed".to_string()),
        None => {}
    }
    future
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
